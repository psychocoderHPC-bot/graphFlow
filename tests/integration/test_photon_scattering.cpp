/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "PhotonCli.hpp"
#include "PhotonFunctors.hpp"
#include "PhotonScattering.hpp"
#include "ScalarTransport.hpp"
#include "TestDevice.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
    using graphFlow::photon::AdvanceToEvent;
    using graphFlow::photon::ClassifyEvent;
    using graphFlow::photon::Event;
    using graphFlow::photon::GeneratePhoton;
    using graphFlow::photon::Parameters;
    using graphFlow::photon::Photon;
    using graphFlow::photon::ProposeFlight;
    using graphFlow::photon::RecordTerminal;
    using graphFlow::photon::Scatter;
    using graphFlow::photon::ScatterMode;
    using graphFlow::photon::Slab;
    using graphFlow::photon::TerminalRecord;
    using graphFlow::photon::Vec3;

    Photon makePhoton(double z, Vec3 direction, std::uint32_t scatterCount = 0u)
    {
        Photon photon;
        photon.id = 0u;
        photon.position = Vec3{0.0, 0.0, z};
        photon.direction = direction;
        photon.scatterCount = scatterCount;
        return photon;
    }

    std::uint64_t countEvent(graphFlow::photon::ScatteringResult const& result, Event event)
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

    void checkTerminalAccounting(
        graphFlow::photon::ScatteringResult const& result,
        std::uint64_t numPhotons)
    {
        REQUIRE(result.terminal.size() == static_cast<std::size_t>(numPhotons));
        REQUIRE(result.terminalVisits.size() == static_cast<std::size_t>(numPhotons));
        CHECK(result.inconsistent == 0u);
        CHECK(result.terminated == numPhotons);
        std::uint64_t sum = 0u;
        for(std::uint32_t event = 0u; event < graphFlow::photon::eventBins; ++event)
            sum += result.eventCounts[event];
        CHECK(sum == numPhotons);
        for(std::size_t id = 0u; id < result.terminal.size(); ++id)
        {
            CHECK(result.terminalVisits[id] == 1u);
            CHECK(result.terminal[id].event != graphFlow::photon::missingEvent);
            CHECK(result.terminal[id].event != Event::scatter);
        }
    }

    void checkAgainstScalar(
        graphFlow::photon::ScatteringResult const& result,
        graphFlow::test::scalarTransport::ScalarOutcome const& reference)
    {
        REQUIRE(result.terminal.size() == reference.terminal.size());
        for(std::size_t id = 0u; id < result.terminal.size(); ++id)
        {
            auto const& actual = result.terminal[id];
            auto const& expected = reference.terminal[id];
            CHECK(static_cast<std::uint32_t>(actual.event) == expected.event);
            CHECK(actual.scatterCount == expected.scatterCount);
            CHECK(actual.position.x == Catch::Approx(expected.x).margin(1e-9));
            CHECK(actual.position.y == Catch::Approx(expected.y).margin(1e-9));
            CHECK(actual.position.z == Catch::Approx(expected.z).margin(1e-9));
            CHECK(actual.direction.x == Catch::Approx(expected.ux).margin(1e-12));
            CHECK(actual.direction.y == Catch::Approx(expected.uy).margin(1e-12));
            CHECK(actual.direction.z == Catch::Approx(expected.uz).margin(1e-12));
        }
    }

    /** Test-only proposal source: fixed path lengths that exercise the production resolver. */
    template<typename TCounts>
    struct ScriptedFlight
    {
        Slab slab{};
        std::uint32_t maxScatters = 0u;
        mutable TCounts flightCounts;

        ALPAKA_FN_ACC void operator()(auto const& acc, graphFlow::Status& status, Photon& photon) const
        {
            double sampledDistance = 0.0;
            switch(photon.scatterCount)
            {
            case 0u:
                sampledDistance = 0.2;
                break;
            case 1u:
                sampledDistance = 0.4;
                break;
            case 2u:
                sampledDistance = 0.2;
                break;
            case 3u:
                sampledDistance = 0.5;
                break;
            default:
                status.set(graphFlow::StatusCode::invalid);
                return;
            }
            photon.proposal = graphFlow::photon::resolveFlight(
                slab,
                photon,
                sampledDistance,
                0.0,
                maxScatters);
            if(!photon.proposal.valid)
                status.set(graphFlow::StatusCode::invalid);
            alpaka::onAcc::atomicAdd(
                acc,
                &flightCounts[0u],
                std::uint32_t{1u},
                alpaka::onAcc::scope::device);
        }
    };

    /** Test-only angular kernel for the scripted three-scatter history. */
    struct ScriptedScatter
    {
        ALPAKA_FN_ACC void operator()(auto const&, graphFlow::Status& status, Photon& photon) const
        {
            if(photon.proposal.event != Event::scatter)
            {
                status.set(graphFlow::StatusCode::invalid);
                return;
            }
            switch(photon.scatterCount)
            {
            case 0u:
                photon.direction = Vec3{0.0, 0.0, 0.5};
                break;
            case 1u:
                photon.direction = Vec3{0.0, 0.0, -0.5};
                break;
            case 2u:
                photon.direction = Vec3{0.0, 0.0, -1.0};
                break;
            default:
                status.set(graphFlow::StatusCode::invalid);
                return;
            }
            ++photon.scatterCount;
        }
    };

    struct ScriptedOutcome
    {
        TerminalRecord terminal{};
        std::uint32_t visits = 0u;
        std::uint32_t flights = 0u;
    };

    ScriptedOutcome runScripted()
    {
        auto device = graphFlow::test::makeCpuDevice();
        auto records = alpaka::onHost::alloc<TerminalRecord>(device, alpaka::Vec{std::uint64_t{1u}});
        auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        auto flights = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            std::vector<TerminalRecord> const initial{
                TerminalRecord{graphFlow::photon::missingEvent, 0u, Vec3{}, Vec3{}}};
            std::vector<std::uint32_t> const zero{0u};
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, records, initial);
            alpaka::onHost::memcpy(queue, visits, zero);
            alpaka::onHost::memcpy(queue, flights, zero);
        }

        Slab const slab{1.0, 1.0, 0.0};
        graphFlow::Flow<Photon, ALPAKA_TYPEOF(device)> flow{device};
        auto source = flow.source("generate", GeneratePhoton{});
        using CountView = decltype(flights.getView());
        auto flight = flow.transform(
            "scripted flight",
            ScriptedFlight<CountView>{slab, 3u, flights.getView()});
        auto route = flow.partition("advance and route", ClassifyEvent{}, AdvanceToEvent{slab});
        auto scatter = flow.transform("scripted scatter", ScriptedScatter{});
        using RecordView = decltype(records.getView());
        using VisitView = decltype(visits.getView());
        auto terminal = flow.sink(
            "terminal",
            RecordTerminal<RecordView, VisitView>{records.getView(), visits.getView()});
        flow.connect(source, flight);
        flow.connect(flight, route);
        flow.connect(route.template out<Event::scatter>(), scatter);
        flow.connect(scatter, flight);
        flow.connect(route.template out<Event::transmitted>(), terminal);
        flow.connect(route.template out<Event::reflected>(), terminal);
        flow.connect(route.template out<Event::absorbed>(), terminal);
        flow.connect(route.template out<Event::capped>(), terminal);

        graphFlow::Tuning tuning;
        tuning.chunkCapacity = 8u;
        tuning.liveParticles = 1u;
        tuning.tileSize = 4u;
        tuning.batchChunks = 1u;
        tuning.frameCap = 1u;
        flow.configure(tuning);
        (void) flow.run(source, 1u);

        ScriptedOutcome outcome;
        std::vector<TerminalRecord> hostRecords(1u);
        std::vector<std::uint32_t> hostVisits(1u, 0u);
        std::vector<std::uint32_t> hostFlights(1u, 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, hostRecords, records);
            alpaka::onHost::memcpy(queue, hostVisits, visits);
            alpaka::onHost::memcpy(queue, hostFlights, flights);
        }
        outcome.terminal = hostRecords[0u];
        outcome.visits = hostVisits[0u];
        outcome.flights = hostFlights[0u];
        return outcome;
    }
} // namespace

TEST_CASE("photon payload and functors satisfy the public Flow contracts", "[photon][flow]")
{
    static_assert(std::is_trivially_copyable_v<Photon>);
    static_assert(std::is_default_constructible_v<Photon>);
    static_assert(graphFlow::SourceFor<GeneratePhoton, Photon>);
    static_assert(graphFlow::TransformFor<ProposeFlight, Photon>);
    static_assert(graphFlow::PartitionClassifierFor<ClassifyEvent, Photon>);
    static_assert(graphFlow::UpdateFor<AdvanceToEvent, Event, Photon>);
    static_assert(graphFlow::TransformFor<Scatter, Photon>);
    static_assert(ClassifyEvent::maxBins == graphFlow::photon::eventBins);
}

TEST_CASE("photon flight geometry handles faces, ties, vacuum and invalid states", "[photon][geometry]")
{
    using graphFlow::photon::applyEvent;
    using graphFlow::photon::faceFirst;
    using graphFlow::photon::resolveFlight;

    Slab const vacuum{1.0, 0.0, 0.0};
    Slab const material{1.0, 1.0, 0.0};

    auto const positive = resolveFlight(vacuum, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.0, 0.0, 8u);
    CHECK(positive.valid);
    CHECK(positive.event == Event::transmitted);
    CHECK(positive.distance == Catch::Approx(0.8).margin(1e-15));

    auto const negative = resolveFlight(vacuum, makePhoton(0.2, Vec3{0.0, 0.0, -1.0}), 0.0, 0.0, 8u);
    CHECK(negative.valid);
    CHECK(negative.event == Event::reflected);
    CHECK(negative.distance == Catch::Approx(0.2).margin(1e-15));

    auto const parallelVacuum = resolveFlight(vacuum, makePhoton(0.5, Vec3{1.0, 0.0, 0.0}), 0.0, 0.0, 8u);
    CHECK_FALSE(parallelVacuum.valid);

    auto const parallelMaterial = resolveFlight(material, makePhoton(0.5, Vec3{1.0, 0.0, 0.0}), 0.5, 0.0, 8u);
    CHECK(parallelMaterial.valid);
    CHECK(parallelMaterial.event == Event::scatter);
    CHECK(parallelMaterial.distance == Catch::Approx(0.5).margin(1e-15));

    // A tie between the face distance and the sampled flight belongs to the face.
    auto const tie = resolveFlight(material, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.8, 0.0, 8u);
    CHECK(tie.valid);
    CHECK(tie.event == Event::transmitted);
    CHECK(tie.distance == Catch::Approx(0.8).margin(1e-15));
    CHECK(faceFirst(0.8, 0.8));
    CHECK_FALSE(faceFirst(0.8, 0.79));

    auto const inside = resolveFlight(material, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.79, 0.0, 8u);
    CHECK(inside.valid);
    CHECK(inside.event == Event::scatter);
    CHECK(inside.distance == Catch::Approx(0.79).margin(1e-15));

    auto const capped = resolveFlight(material, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.3, 0.0, 0u);
    CHECK(capped.valid);
    CHECK(capped.event == Event::capped);

    auto const absorbed = resolveFlight(
        Slab{1.0, 0.0, 1.0},
        makePhoton(0.2, Vec3{0.0, 0.0, 1.0}),
        0.3,
        0.0,
        8u);
    CHECK(absorbed.valid);
    CHECK(absorbed.event == Event::absorbed);

    // The face advance clamps z exactly and propagates the transverse position.
    Photon photon = makePhoton(0.3, Vec3{0.6, 0.0, 0.8});
    photon.proposal = graphFlow::photon::Proposal{Event::transmitted, 0.875, true};
    CHECK(applyEvent(material, Event::transmitted, photon) == graphFlow::UpdateResult::success);
    CHECK(photon.position.z == 1.0);
    CHECK(photon.position.x == Catch::Approx(0.525).margin(1e-15));

    Photon clamped = makePhoton(0.3, Vec3{0.0, 0.0, -1.0});
    clamped.proposal = graphFlow::photon::Proposal{Event::reflected, 0.3, true};
    CHECK(applyEvent(material, Event::reflected, clamped) == graphFlow::UpdateResult::success);
    CHECK(clamped.position.z == 0.0);

    // Non-finite geometry, invalid proposals and inconsistent bin/event pairs are rejected.
    Photon nanPhoton = makePhoton(std::numeric_limits<double>::quiet_NaN(), Vec3{0.0, 0.0, 1.0});
    CHECK_FALSE(resolveFlight(vacuum, nanPhoton, 0.0, 0.0, 8u).valid);
    Photon infDirection = makePhoton(0.2, Vec3{0.0, 0.0, std::numeric_limits<double>::infinity()});
    CHECK_FALSE(resolveFlight(vacuum, infDirection, 0.0, 0.0, 8u).valid);
    Photon outside = makePhoton(1.5, Vec3{0.0, 0.0, 1.0});
    CHECK_FALSE(resolveFlight(vacuum, outside, 0.0, 0.0, 8u).valid);
    Photon missing = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
    CHECK(applyEvent(vacuum, Event::transmitted, missing) == graphFlow::UpdateResult::invalid);
    Photon mismatched = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
    mismatched.proposal = graphFlow::photon::Proposal{Event::transmitted, 0.8, true};
    CHECK(applyEvent(vacuum, Event::reflected, mismatched) == graphFlow::UpdateResult::invalid);
    Photon wrongFaceDirection = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
    wrongFaceDirection.proposal = graphFlow::photon::Proposal{Event::reflected, 0.2, true};
    CHECK(applyEvent(vacuum, Event::reflected, wrongFaceDirection) == graphFlow::UpdateResult::invalid);
}

TEST_CASE("scripted photon history uses three scatters and four flights", "[photon][scripted]")
{
    ScriptedOutcome const outcome = runScripted();
    CHECK(outcome.flights == 4u);
    CHECK(outcome.visits == 1u);
    CHECK(outcome.terminal.event == Event::reflected);
    CHECK(outcome.terminal.scatterCount == 3u);
    CHECK(outcome.terminal.position.z == 0.0);
    CHECK(outcome.terminal.position.x == Catch::Approx(0.0).margin(1e-15));
    CHECK(outcome.terminal.position.y == Catch::Approx(0.0).margin(1e-15));
    CHECK(outcome.terminal.direction.z == Catch::Approx(-1.0).margin(1e-15));
}

TEST_CASE("photon graph matches the independent scalar transport", "[photon][scalar]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    struct Case
    {
        std::uint64_t photons;
        std::uint32_t chunkSize;
        std::uint32_t batchChunks;
        std::uint64_t seed;
        double sigmaS;
        double sigmaA;
        ScatterMode mode;
    };
    std::vector<Case> const cases{
        {257u, 1u, 1u, 0u, 1.0, 0.2, ScatterMode::forward},
        {257u, 128u, 8u, 42u, 1.0, 0.2, ScatterMode::forward},
        {4097u, 31u, 2u, 7u, 0.7, 0.3, ScatterMode::isotropic},
        {4097u, 128u, 8u, 42u, 0.7, 0.3, ScatterMode::isotropic}};

    for(auto const& testCase : cases)
    {
        CAPTURE(testCase.photons, testCase.chunkSize, testCase.batchChunks, testCase.seed);
        Parameters parameters;
        parameters.numPhotons = testCase.photons;
        parameters.chunkSize = testCase.chunkSize;
        parameters.batchChunks = testCase.batchChunks;
        parameters.seed = testCase.seed;
        parameters.thickness = 1.0;
        parameters.sigmaS = testCase.sigmaS;
        parameters.sigmaA = testCase.sigmaA;
        parameters.mode = testCase.mode;
        parameters.maxScatters = 512u;

        auto const result = graphFlow::photon::runPhotonScattering(device, parameters);
        checkTerminalAccounting(result, testCase.photons);

        auto const reference = graphFlow::test::scalarTransport::runScalarTransport(
            testCase.seed,
            testCase.photons,
            1.0,
            testCase.sigmaS,
            testCase.sigmaA,
            testCase.mode == ScatterMode::isotropic,
            parameters.maxScatters);
        checkAgainstScalar(result, reference);
    }
}

TEST_CASE("photon transport is independent of chunk capacity and batch size", "[photon][tuning]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    auto run = [&](std::uint32_t chunkSize, std::uint32_t batchChunks)
    {
        Parameters parameters;
        parameters.numPhotons = 513u;
        parameters.chunkSize = chunkSize;
        parameters.batchChunks = batchChunks;
        parameters.seed = 42u;
        parameters.thickness = 1.0;
        parameters.sigmaS = 0.8;
        parameters.sigmaA = 0.2;
        parameters.mode = ScatterMode::isotropic;
        parameters.maxScatters = 256u;
        return graphFlow::photon::runPhotonScattering(device, parameters);
    };

    auto const small = run(1u, 1u);
    auto const large = run(128u, 8u);
    REQUIRE(small.terminal.size() == large.terminal.size());
    for(std::size_t id = 0u; id < small.terminal.size(); ++id)
    {
        CHECK(small.terminal[id].event == large.terminal[id].event);
        CHECK(small.terminal[id].scatterCount == large.terminal[id].scatterCount);
        CHECK(small.terminal[id].position.x == large.terminal[id].position.x);
        CHECK(small.terminal[id].position.y == large.terminal[id].position.y);
        CHECK(small.terminal[id].position.z == large.terminal[id].position.z);
        CHECK(small.terminal[id].direction.x == large.terminal[id].direction.x);
        CHECK(small.terminal[id].direction.y == large.terminal[id].direction.y);
        CHECK(small.terminal[id].direction.z == large.terminal[id].direction.z);
    }
}

TEST_CASE("forward photon scattering reproduces the Poisson collision count", "[photon][physics]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t photons = 200000u;
    Parameters parameters;
    parameters.numPhotons = photons;
    parameters.chunkSize = 1024u;
    parameters.seed = 1234u;
    parameters.thickness = 1.0;
    parameters.sigmaS = 3.0;
    parameters.sigmaA = 0.0;
    parameters.mode = ScatterMode::forward;
    parameters.maxScatters = 512u;

    auto const result = graphFlow::photon::runPhotonScattering(device, parameters);
    checkTerminalAccounting(result, photons);
    CHECK(countEvent(result, Event::transmitted) == photons);
    CHECK(countEvent(result, Event::reflected) == 0u);
    CHECK(countEvent(result, Event::absorbed) == 0u);
    CHECK(countEvent(result, Event::capped) == 0u);

    double const mean = 3.0;
    double const expectedAtLeastTwo = 1.0 - poissonPmf(mean, 0u) - poissonPmf(mean, 1u);
    CHECK(expectedAtLeastTwo == Catch::Approx(0.8008517265285443).margin(1e-12));

    std::uint64_t atLeastTwo = 0u;
    double meanObserved = 0.0;
    for(auto const& record : result.terminal)
    {
        meanObserved += static_cast<double>(record.scatterCount);
        if(record.scatterCount >= 2u)
            ++atLeastTwo;
    }
    meanObserved /= static_cast<double>(photons);
    double const meanError = std::sqrt(mean / static_cast<double>(photons));
    CHECK(meanObserved == Catch::Approx(mean).margin(6.0 * meanError));
    double const probability = static_cast<double>(atLeastTwo) / static_cast<double>(photons);
    double const sigma = std::sqrt(expectedAtLeastTwo * (1.0 - expectedAtLeastTwo) / static_cast<double>(photons));
    CHECK(probability == Catch::Approx(expectedAtLeastTwo).margin(6.0 * sigma));

    // Chi-square over k = 0..9 plus a merged tail, df = 10.
    double expected[11]{};
    double observed[11]{};
    for(std::uint32_t k = 0u; k < 10u; ++k)
        expected[k] = static_cast<double>(photons) * poissonPmf(mean, k);
    expected[10] = static_cast<double>(photons);
    for(std::uint32_t k = 0u; k < 10u; ++k)
        expected[10] -= expected[k];
    for(auto const& record : result.terminal)
    {
        std::uint32_t const bin = record.scatterCount < 10u ? record.scatterCount : 10u;
        observed[bin] += 1.0;
    }
    double chiSquare = 0.0;
    for(std::uint32_t k = 0u; k < 11u; ++k)
        chiSquare += (observed[k] - expected[k]) * (observed[k] - expected[k]) / expected[k];
    CHECK(chiSquare < 35.0);
}

TEST_CASE("isotropic photon scattering reproduces the uncollided fraction", "[photon][physics]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t photons = 100000u;
    Parameters parameters;
    parameters.numPhotons = photons;
    parameters.chunkSize = 1024u;
    parameters.seed = 99u;
    parameters.thickness = 1.0;
    parameters.sigmaS = 1.0;
    parameters.sigmaA = 0.0;
    parameters.mode = ScatterMode::isotropic;
    parameters.maxScatters = 512u;

    auto const result = graphFlow::photon::runPhotonScattering(device, parameters);
    checkTerminalAccounting(result, photons);

    std::uint64_t uncollided = 0u;
    for(auto const& record : result.terminal)
        if(record.scatterCount == 0u)
            ++uncollided;

    double const expected = std::exp(-1.0);
    double const sigma = std::sqrt(expected * (1.0 - expected) / static_cast<double>(photons));
    CHECK(static_cast<double>(uncollided) / static_cast<double>(photons)
          == Catch::Approx(expected).margin(6.0 * sigma));

    for(auto const& record : result.terminal)
    {
        double const norm = record.direction.x * record.direction.x
                            + record.direction.y * record.direction.y
                            + record.direction.z * record.direction.z;
        CHECK(norm == Catch::Approx(1.0).margin(1e-12));
    }
}

TEST_CASE("pure absorption and vacuum have closed-form transmission", "[photon][physics]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t photons = 100000u;

    Parameters absorbing;
    absorbing.numPhotons = photons;
    absorbing.chunkSize = 1024u;
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
    CHECK(countEvent(absorbed, Event::transmitted) + countEvent(absorbed, Event::absorbed) == photons);
    double const transmission = static_cast<double>(countEvent(absorbed, Event::transmitted))
                                / static_cast<double>(photons);
    double const expected = std::exp(-1.0);
    double const sigma = std::sqrt(expected * (1.0 - expected) / static_cast<double>(photons));
    CHECK(transmission == Catch::Approx(expected).margin(6.0 * sigma));

    Parameters vacuum;
    vacuum.numPhotons = 1000u;
    vacuum.chunkSize = 128u;
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
        CHECK(record.direction.z == Catch::Approx(1.0).margin(1e-15));
    }
}

TEST_CASE("scatter cap emits a distinct non-physical outcome", "[photon][cap]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t photons = 512u;
    Parameters parameters;
    parameters.numPhotons = photons;
    parameters.chunkSize = 31u;
    parameters.seed = 42u;
    parameters.thickness = 1.0;
    parameters.sigmaS = 1.0;
    parameters.sigmaA = 0.0;
    parameters.mode = ScatterMode::forward;
    parameters.maxScatters = 2u;

    auto const result = graphFlow::photon::runPhotonScattering(device, parameters);
    checkTerminalAccounting(result, photons);
    CHECK(result.capped > 0u);
    CHECK(countEvent(result, Event::transmitted) + countEvent(result, Event::reflected)
              + countEvent(result, Event::absorbed)
          == photons - result.capped);
    for(auto const& record : result.terminal)
    {
        if(record.event == Event::capped)
            CHECK(record.scatterCount == parameters.maxScatters);
        else
            CHECK(record.event != Event::scatter);
    }
}

TEST_CASE("photon CLI rejects invalid physical configuration", "[photon][cli]")
{
    auto const properties = graphFlow::test::makeCpuDevice().getDeviceProperties();

    graphFlow::photon::cli::Options valid;
    valid.numPhotons = 17u;
    valid.thickness = 1.0;
    valid.sigmaS = 0.5;
    valid.sigmaA = 0.1;
    valid.mode = "isotropic";
    auto const resolved = graphFlow::photon::cli::resolveOptions(valid, properties);
    CHECK(resolved.parameters.numPhotons == 17u);
    CHECK(resolved.parameters.mode == ScatterMode::isotropic);
    CHECK(resolved.parameters.chunkSize > 0u);

    std::vector<std::string> const aliasArgs{
        "graphFlow_photon_scattering",
        "--N",
        "5",
        "--L",
        "2",
        "--sigmaS",
        "0.3",
        "--sigmaA",
        "0.1",
        "--cap",
        "4",
        "--C",
        "8",
        "--P",
        "4",
        "--T",
        "4",
        "--Q",
        "2"};
    std::vector<char const*> aliasArgv;
    aliasArgv.reserve(aliasArgs.size());
    for(auto const& argument : aliasArgs)
        aliasArgv.push_back(argument.c_str());
    auto const aliases = graphFlow::photon::cli::parseOptions(
        static_cast<int>(aliasArgv.size()),
        aliasArgv.data());
    CHECK(aliases.numPhotons == 5u);
    CHECK(aliases.thickness == Catch::Approx(2.0));
    CHECK(aliases.sigmaS == Catch::Approx(0.3));
    CHECK(aliases.sigmaA == Catch::Approx(0.1));
    CHECK(aliases.maxScatters == 4u);
    REQUIRE(aliases.chunkSize.has_value());
    CHECK(*aliases.chunkSize == 8u);
    REQUIRE(aliases.liveParticles.has_value());
    CHECK(*aliases.liveParticles == 4u);
    REQUIRE(aliases.tileSize.has_value());
    CHECK(*aliases.tileSize == 4u);
    CHECK(aliases.batchChunks == 2u);

    graphFlow::photon::cli::Options negativeSigma = valid;
    negativeSigma.sigmaS = -1.0;
    CHECK_THROWS_AS(
        graphFlow::photon::cli::resolveOptions(negativeSigma, properties),
        std::invalid_argument);

    graphFlow::photon::cli::Options negativeAbsorption = valid;
    negativeAbsorption.sigmaA = -1.0;
    CHECK_THROWS_AS(
        graphFlow::photon::cli::resolveOptions(negativeAbsorption, properties),
        std::invalid_argument);

    graphFlow::photon::cli::Options zeroThickness = valid;
    zeroThickness.thickness = 0.0;
    CHECK_THROWS_AS(
        graphFlow::photon::cli::resolveOptions(zeroThickness, properties),
        std::invalid_argument);

    graphFlow::photon::cli::Options unknownMode = valid;
    unknownMode.mode = "ballistic";
    CHECK_THROWS_AS(
        graphFlow::photon::cli::resolveOptions(unknownMode, properties),
        std::invalid_argument);

    graphFlow::photon::cli::Options zeroBudget = valid;
    zeroBudget.poolPayloadBudgetBytes = 0u;
    CHECK_THROWS_AS(
        graphFlow::photon::cli::resolveOptions(zeroBudget, properties),
        std::invalid_argument);
}
