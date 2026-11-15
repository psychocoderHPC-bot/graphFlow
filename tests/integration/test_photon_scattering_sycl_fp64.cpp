/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "PhotonScattering.hpp"
#include "ScalarTransport.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if !ALPAKA_LANG_ONEAPI
#    error "test_photon_scattering_sycl_fp64.cpp requires a SYCL compilation (GRAPHFLOW_ENABLE_SYCL=ON)"
#endif

namespace
{
    using SyclSelector
        = decltype(alpaka::onHost::makeDeviceSelector(alpaka::api::OneApi{}, alpaka::deviceKind::intelGpu));
    using SyclDevice = decltype(std::declval<SyclSelector&>().makeDevice(0u));

    SyclDevice makeSyclDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::OneApi{}, alpaka::deviceKind::intelGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{
                "SYCL is compiled in but no Intel GPU is visible; this test requires real hardware"};
        return selector.makeDevice(0u);
    }

    /** Restore one environment variable when the scope ends, even after a failed check. */
    class ScopedEnv
    {
    public:
        explicit ScopedEnv(std::string name) : m_name(std::move(name))
        {
            char const* const value = std::getenv(m_name.c_str());
            if(value != nullptr)
            {
                m_existed = true;
                m_previous = value;
            }
        }

        ~ScopedEnv()
        {
            if(m_existed)
                setenv(m_name.c_str(), m_previous.c_str(), 1);
            else
                unsetenv(m_name.c_str());
        }

        ScopedEnv(ScopedEnv const&) = delete;
        ScopedEnv& operator=(ScopedEnv const&) = delete;

        void unset() const
        {
            unsetenv(m_name.c_str());
        }

    private:
        std::string m_name;
        bool m_existed = false;
        std::string m_previous;
    };

    using graphFlow::photon::Event;
    using graphFlow::photon::Parameters;
    using graphFlow::photon::ScatteringResult;
    using graphFlow::photon::ScatterMode;

    std::uint64_t countEvent(ScatteringResult const& result, Event event)
    {
        return result.eventCounts[static_cast<std::uint32_t>(event)];
    }

    void checkTerminalAccounting(ScatteringResult const& result, std::uint64_t photons)
    {
        REQUIRE(result.terminal.size() == static_cast<std::size_t>(photons));
        REQUIRE(result.terminalVisits.size() == static_cast<std::size_t>(photons));
        CHECK(result.inconsistent == 0u);
        CHECK(result.terminated == photons);
        std::uint64_t sum = 0u;
        for(std::uint32_t event = 0u; event < graphFlow::photon::eventBins; ++event)
            sum += result.eventCounts[event];
        CHECK(sum == photons);
        for(std::size_t id = 0u; id < result.terminal.size(); ++id)
        {
            CHECK(result.terminalVisits[id] == 1u);
            CHECK(result.terminal[id].event != graphFlow::photon::missingEvent);
            CHECK(result.terminal[id].event != Event::scatter);
        }
    }

    void checkAgainstScalar(
        ScatteringResult const& result,
        graphFlow::test::scalarTransport::ScalarOutcome const& reference)
    {
        REQUIRE(result.terminal.size() == reference.terminal.size());
        for(std::size_t id = 0u; id < result.terminal.size(); ++id)
        {
            auto const& actual = result.terminal[id];
            auto const& expected = reference.terminal[id];
            CHECK(static_cast<std::uint32_t>(actual.event) == expected.event);
            CHECK(actual.scatterCount == expected.scatterCount);
            // Tight double-precision margins: a float fallback cannot meet them.
            CHECK(actual.position.x == Catch::Approx(expected.x).margin(1e-9));
            CHECK(actual.position.y == Catch::Approx(expected.y).margin(1e-9));
            CHECK(actual.position.z == Catch::Approx(expected.z).margin(1e-9));
            CHECK(actual.direction.x == Catch::Approx(expected.ux).margin(1e-12));
            CHECK(actual.direction.y == Catch::Approx(expected.uy).margin(1e-12));
            CHECK(actual.direction.z == Catch::Approx(expected.uz).margin(1e-12));
        }
    }
} // namespace

/** The CTest wrapper sets GRAPHFLOW_SYCL_FP64_EMULATION=1, IGC_EnableDPEmulation=1 and
 * OverrideDefaultFP64Settings=1. A manual run must set the same three variables before the
 * process starts. The test keeps the oneAPI photon kernels off the DG2 sub-group reduction path
 * (block barriers and atomics only) and runs under a CTest timeout so a device loss cannot hang
 * the suite.
 */
TEST_CASE("SYCL photon FP64 emulation runs the exact double path", "[sycl-fp64][photon]")
{
    if(!graphFlow::photon::syclFp64EmulationRequested())
    {
        FAIL(
            "run this case with GRAPHFLOW_SYCL_FP64_EMULATION=1, IGC_EnableDPEmulation=1 and "
            "OverrideDefaultFP64Settings=1 in the process environment");
    }

    auto device = makeSyclDevice();

    // Fail closed: the opt-in is not valid without both IGC switches, and the run must not
    // silently fall back to single precision.
    {
        ScopedEnv enableIcg{"IGC_EnableDPEmulation"};
        ScopedEnv overrideSettings{"OverrideDefaultFP64Settings"};
        enableIcg.unset();
        CHECK_THROWS_AS(graphFlow::photon::requireSyclFp64Emulation(device), std::runtime_error);
        overrideSettings.unset();
        CHECK_THROWS_AS(graphFlow::photon::requireSyclFp64Emulation(device), std::runtime_error);
    }
    CHECK_NOTHROW(graphFlow::photon::requireSyclFp64Emulation(device));
    CHECK(graphFlow::photon::syclDeviceSupportsDouble(device));

    constexpr std::uint64_t photons = 257u;
    Parameters parameters;
    parameters.numPhotons = photons;
    parameters.chunkSize = 31u;
    parameters.batchChunks = 2u;
    parameters.seed = 0u;
    parameters.thickness = 1.0;
    parameters.sigmaS = 1.0;
    parameters.sigmaA = 0.2;
    parameters.mode = ScatterMode::forward;
    parameters.maxScatters = 512u;
    parameters.tileSize = 32u;
    parameters.frameCap = 2u;

    ScatteringResult const result
        = graphFlow::photon::runPhotonScatteringAuto(device, parameters, graphFlow::photon::tuningOf(parameters));
    static_assert(std::same_as<std::remove_cvref_t<decltype(result)>, ScatteringResult>);
    checkTerminalAccounting(result, photons);
    CHECK(
        countEvent(result, Event::transmitted) + countEvent(result, Event::reflected)
            + countEvent(result, Event::absorbed) + countEvent(result, Event::capped)
        == photons);

    auto const reference = graphFlow::test::scalarTransport::runScalarTransport(
        parameters.seed,
        photons,
        1.0,
        1.0,
        0.2,
        false,
        parameters.maxScatters);
    checkAgainstScalar(result, reference);
}
