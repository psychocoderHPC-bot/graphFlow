/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "PhotonScattering.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if !ALPAKA_LANG_CUDA
#    error "test_photon_scattering_cuda.cpp requires a CUDA compilation (GRAPHFLOW_ENABLE_CUDA=ON)"
#endif

namespace
{
    using CudaSelector
        = decltype(alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu));
    using CudaDevice = decltype(std::declval<CudaSelector&>().makeDevice(0u));

    CudaDevice makeCudaDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{
                "CUDA is compiled in but no NVIDIA GPU is visible; this test requires real hardware"};
        return selector.makeDevice(0u);
    }

    using graphFlow::photon::Event;
    using graphFlow::photon::Parameters;
    using graphFlow::photon::ScatterMode;
    using graphFlow::photon::ScatteringResult;

    std::uint64_t countEvent(ScatteringResult const& result, Event event)
    {
        return result.eventCounts[static_cast<std::uint32_t>(event)];
    }

    double poissonPmf(double mean, std::uint32_t k)
    {
        double logP = -mean + static_cast<double>(k) * std::log(mean);
        for(std::uint32_t i = 2u; i <= k; ++i)
            logP -= std::log(static_cast<double>(i));
        return std::exp(logP);
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

    Parameters forwardParameters(std::uint64_t photons, std::uint32_t capacity, std::uint32_t batchChunks)
    {
        Parameters parameters;
        parameters.numPhotons = photons;
        parameters.chunkSize = capacity;
        parameters.batchChunks = batchChunks;
        parameters.seed = 1234u;
        parameters.thickness = 1.0;
        parameters.sigmaS = 3.0;
        parameters.sigmaA = 0.0;
        parameters.mode = ScatterMode::forward;
        parameters.maxScatters = 512u;
        return parameters;
    }
} // namespace

TEST_CASE("CUDA photon forward transport reproduces the Poisson count", "[photon][cuda][physics]")
{
    auto device = makeCudaDevice();
    constexpr std::uint64_t photons = 50000u;
    Parameters parameters = forwardParameters(photons, 256u, 4u);
    parameters.tileSize = 32u;
    parameters.frameCap = 2u;

    ScatteringResult const result = graphFlow::photon::runPhotonScattering(device, parameters);
    checkTerminalAccounting(result, photons);
    CHECK(countEvent(result, Event::transmitted) == photons);
    CHECK(countEvent(result, Event::reflected) == 0u);
    CHECK(countEvent(result, Event::absorbed) == 0u);
    CHECK(countEvent(result, Event::capped) == 0u);

    double mean = 0.0;
    std::uint64_t atLeastTwo = 0u;
    for(auto const& record : result.terminal)
    {
        mean += static_cast<double>(record.scatterCount);
        if(record.scatterCount >= 2u)
            ++atLeastTwo;
    }
    mean /= static_cast<double>(photons);
    CHECK(mean == Catch::Approx(3.0).margin(6.0 * std::sqrt(3.0 / static_cast<double>(photons))));

    double const expected = 1.0 - poissonPmf(3.0, 0u) - poissonPmf(3.0, 1u);
    double const probability = static_cast<double>(atLeastTwo) / static_cast<double>(photons);
    double const sigma = std::sqrt(expected * (1.0 - expected) / static_cast<double>(photons));
    CHECK(probability == Catch::Approx(expected).margin(6.0 * sigma));
}

TEST_CASE("CUDA photon isotropic and absorbing runs match closed forms", "[photon][cuda][physics]")
{
    auto device = makeCudaDevice();
    constexpr std::uint64_t photons = 50000u;

    Parameters isotropic;
    isotropic.numPhotons = photons;
    isotropic.chunkSize = 256u;
    isotropic.tileSize = 32u;
    isotropic.frameCap = 2u;
    isotropic.batchChunks = 4u;
    isotropic.seed = 99u;
    isotropic.thickness = 1.0;
    isotropic.sigmaS = 1.0;
    isotropic.sigmaA = 0.0;
    isotropic.mode = ScatterMode::isotropic;
    isotropic.maxScatters = 512u;
    auto const scattered = graphFlow::photon::runPhotonScattering(device, isotropic);
    checkTerminalAccounting(scattered, photons);
    std::uint64_t uncollided = 0u;
    for(auto const& record : scattered.terminal)
        if(record.scatterCount == 0u)
            ++uncollided;
    double const uncollidedExpected = std::exp(-1.0);
    double const uncollidedSigma
        = std::sqrt(uncollidedExpected * (1.0 - uncollidedExpected) / static_cast<double>(photons));
    CHECK(static_cast<double>(uncollided) / static_cast<double>(photons)
          == Catch::Approx(uncollidedExpected).margin(6.0 * uncollidedSigma));
    for(auto const& record : scattered.terminal)
    {
        double const norm = record.direction.x * record.direction.x
                            + record.direction.y * record.direction.y
                            + record.direction.z * record.direction.z;
        CHECK(norm == Catch::Approx(1.0).margin(1e-12));
    }

    Parameters absorbing;
    absorbing.numPhotons = photons;
    absorbing.chunkSize = 256u;
    absorbing.tileSize = 32u;
    absorbing.frameCap = 2u;
    absorbing.batchChunks = 4u;
    absorbing.seed = 5u;
    absorbing.thickness = 1.0;
    absorbing.sigmaS = 0.0;
    absorbing.sigmaA = 1.0;
    absorbing.mode = ScatterMode::forward;
    absorbing.maxScatters = 8u;
    auto const absorbed = graphFlow::photon::runPhotonScattering(device, absorbing);
    checkTerminalAccounting(absorbed, photons);
    CHECK(countEvent(absorbed, Event::reflected) == 0u);
    CHECK(countEvent(absorbed, Event::capped) == 0u);
    double const transmission = static_cast<double>(countEvent(absorbed, Event::transmitted))
                                / static_cast<double>(photons);
    double const expected = std::exp(-1.0);
    double const sigma = std::sqrt(expected * (1.0 - expected) / static_cast<double>(photons));
    CHECK(transmission == Catch::Approx(expected).margin(6.0 * sigma));
}

TEST_CASE("CUDA photon vacuum and cap outcomes stay distinct", "[photon][cuda][cap]")
{
    auto device = makeCudaDevice();

    Parameters vacuum;
    vacuum.numPhotons = 1000u;
    vacuum.chunkSize = 64u;
    vacuum.tileSize = 32u;
    vacuum.frameCap = 2u;
    vacuum.seed = 7u;
    vacuum.thickness = 1.0;
    vacuum.sigmaS = 0.0;
    vacuum.sigmaA = 0.0;
    vacuum.mode = ScatterMode::forward;
    auto const ballistic = graphFlow::photon::runPhotonScattering(device, vacuum);
    checkTerminalAccounting(ballistic, 1000u);
    CHECK(countEvent(ballistic, Event::transmitted) == 1000u);
    for(auto const& record : ballistic.terminal)
    {
        CHECK(record.event == Event::transmitted);
        CHECK(record.scatterCount == 0u);
        CHECK(record.position.z == 1.0);
    }

    constexpr std::uint64_t photons = 512u;
    Parameters capped;
    capped.numPhotons = photons;
    capped.chunkSize = 31u;
    capped.tileSize = 32u;
    capped.frameCap = 2u;
    capped.seed = 42u;
    capped.thickness = 1.0;
    capped.sigmaS = 1.0;
    capped.sigmaA = 0.0;
    capped.mode = ScatterMode::forward;
    capped.maxScatters = 2u;
    auto const capResult = graphFlow::photon::runPhotonScattering(device, capped);
    checkTerminalAccounting(capResult, photons);
    CHECK(capResult.capped > 0u);
    CHECK(countEvent(capResult, Event::transmitted) + countEvent(capResult, Event::reflected)
              + countEvent(capResult, Event::absorbed)
          == photons - capResult.capped);
    for(auto const& record : capResult.terminal)
    {
        if(record.event == Event::capped)
            CHECK(record.scatterCount == capped.maxScatters);
    }
}

TEST_CASE("CUDA photon results are independent of chunk and batch tuning", "[photon][cuda][tuning]")
{
    auto device = makeCudaDevice();
    Parameters small = forwardParameters(513u, 32u, 1u);
    small.tileSize = 32u;
    small.frameCap = 2u;
    Parameters large = forwardParameters(513u, 128u, 8u);
    large.tileSize = 32u;
    large.frameCap = 2u;

    auto const a = graphFlow::photon::runPhotonScattering(device, small);
    auto const b = graphFlow::photon::runPhotonScattering(device, large);
    REQUIRE(a.terminal.size() == b.terminal.size());
    for(std::size_t id = 0u; id < a.terminal.size(); ++id)
    {
        CHECK(a.terminal[id].event == b.terminal[id].event);
        CHECK(a.terminal[id].scatterCount == b.terminal[id].scatterCount);
    }
}
