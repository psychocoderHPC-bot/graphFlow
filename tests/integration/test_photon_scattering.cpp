/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "PhotonCli.hpp"
#include "PhotonDebug.hpp"
#include "PhotonFunctors.hpp"
#include "PhotonScattering.hpp"
#include "ScalarTransport.hpp"
#include "TestDevice.hpp"
#include "graphFlow/flow/Debug.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
    using graphFlow::photon::AdvanceToEvent;
    using graphFlow::photon::ClassifyEvent;
    using graphFlow::photon::directionTag;
    using graphFlow::photon::Event;
    using graphFlow::photon::GeneratePhoton;
    using graphFlow::photon::idTag;
    using graphFlow::photon::NoTrace;
    using graphFlow::photon::Parameters;
    using graphFlow::photon::Photon;
    using graphFlow::photon::positionTag;
    using graphFlow::photon::proposalTag;
    using graphFlow::photon::ProposeFlight;
    using graphFlow::photon::RecordTerminal;
    using graphFlow::photon::Scatter;
    using graphFlow::photon::scatterCountTag;
    using graphFlow::photon::ScatterMode;
    using graphFlow::photon::ScatterT;
    using graphFlow::photon::Slab;
    using graphFlow::photon::TerminalRecord;
    using graphFlow::photon::Vec3;

    Photon makePhoton(double z, Vec3 direction, std::uint32_t scatterCount = 0u)
    {
        Photon photon;
        photon[idTag] = 0u;
        photon[positionTag] = Vec3{0.0, 0.0, z};
        photon[directionTag] = direction;
        photon[scatterCountTag] = scatterCount;
        photon[proposalTag] = graphFlow::photon::Proposal{};
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

    void checkTerminalAccounting(graphFlow::photon::ScatteringResult const& result, std::uint64_t numPhotons)
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
            switch(photon[scatterCountTag])
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
            photon[proposalTag] = graphFlow::photon::resolveFlight(
                slab,
                graphFlow::photon::sigmaTotal(slab),
                photon,
                sampledDistance,
                0.0,
                maxScatters);
            if(!photon[proposalTag].valid)
                status.set(graphFlow::StatusCode::invalid);
            alpaka::onAcc::atomicAdd(acc, &flightCounts[0u], std::uint32_t{1u}, alpaka::onAcc::scope::device);
        }
    };

    /** Test-only angular kernel for the scripted three-scatter history. */
    template<typename TTrace>
    struct ScriptedScatter
    {
        mutable TTrace trace;

        ALPAKA_FN_ACC void operator()(auto const&, graphFlow::Status& status, Photon& photon) const
        {
            if(photon[proposalTag].event != Event::scatter)
            {
                status.set(graphFlow::StatusCode::invalid);
                return;
            }
            switch(photon[scatterCountTag])
            {
            case 0u:
                photon[directionTag] = Vec3{0.0, 0.0, 0.5};
                break;
            case 1u:
                photon[directionTag] = Vec3{0.0, 0.0, -0.5};
                break;
            case 2u:
                photon[directionTag] = Vec3{0.0, 0.0, -1.0};
                break;
            default:
                status.set(graphFlow::StatusCode::invalid);
                return;
            }
            trace.record(
                photon[idTag],
                graphFlow::photon::TraceKind::scatter,
                photon[scatterCountTag] + 1u,
                photon[positionTag],
                photon[directionTag]);
            ++photon[scatterCountTag];
        }
    };

    struct ScriptedOutcome
    {
        TerminalRecord terminal{};
        std::uint32_t visits = 0u;
        std::uint32_t flights = 0u;
        graphFlow::photon::PhotonTrace trace;
    };

    ScriptedOutcome runScripted(std::uint32_t traceVertices = 16u)
    {
        auto device = graphFlow::test::makeCpuDevice();
        auto records = alpaka::onHost::alloc<TerminalRecord>(device, alpaka::Vec{std::uint64_t{1u}});
        auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        auto flights = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        auto traceBuffer
            = alpaka::onHost::alloc<graphFlow::photon::TraceVertex>(device, alpaka::Vec{std::uint64_t{traceVertices}});
        auto traceCounts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        auto traceFlags = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            std::vector<TerminalRecord> const initial{
                TerminalRecord{graphFlow::photon::missingEvent, 0u, Vec3{}, Vec3{}}};
            std::vector<std::uint32_t> const zero{0u};
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, records, initial);
            alpaka::onHost::memcpy(queue, visits, zero);
            alpaka::onHost::memcpy(queue, flights, zero);
            alpaka::onHost::memcpy(queue, traceCounts, zero);
            alpaka::onHost::memcpy(queue, traceFlags, zero);
        }

        Slab const slab{1.0, 1.0, 0.0};
        using TraceVertexView = decltype(traceBuffer.getView());
        using TraceCountView = decltype(traceCounts.getView());
        using TraceFlagView = decltype(traceFlags.getView());
        using Trace = graphFlow::photon::TraceWriter<TraceVertexView, TraceCountView, TraceFlagView>;
        Trace const trace{traceBuffer.getView(), traceCounts.getView(), traceFlags.getView(), traceVertices, 1u};

        graphFlow::SoAFlow<Photon, ALPAKA_TYPEOF(device)> flow{device};
        auto source = flow.source("generate", graphFlow::photon::GeneratePhotonT<Trace>{trace});
        using CountView = decltype(flights.getView());
        auto flight = flow.transform("scripted flight", ScriptedFlight<CountView>{slab, 3u, flights.getView()});
        auto route = flow.partition("advance and route", ClassifyEvent{}, AdvanceToEvent{slab});
        auto scatter = flow.transform("scripted scatter", ScriptedScatter<Trace>{trace});
        using RecordView = decltype(records.getView());
        using VisitView = decltype(visits.getView());
        auto terminal = flow.sink(
            "terminal",
            RecordTerminal<RecordView, VisitView, Trace>{records.getView(), visits.getView(), trace});
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
        std::vector<std::uint32_t> hostTraceCounts(1u, 0u);
        std::vector<std::uint32_t> hostTraceFlags(1u, 0u);
        std::vector<graphFlow::photon::TraceVertex> hostTrace(traceVertices);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, hostRecords, records);
            alpaka::onHost::memcpy(queue, hostVisits, visits);
            alpaka::onHost::memcpy(queue, hostFlights, flights);
            alpaka::onHost::memcpy(queue, hostTraceCounts, traceCounts);
            alpaka::onHost::memcpy(queue, hostTraceFlags, traceFlags);
            alpaka::onHost::memcpy(queue, hostTrace, traceBuffer);
        }
        outcome.terminal = hostRecords[0u];
        outcome.visits = hostVisits[0u];
        outcome.flights = hostFlights[0u];
        outcome.trace.id = 0u;
        outcome.trace.count = hostTraceCounts[0u];
        outcome.trace.truncated = hostTraceFlags[0u] != 0u;
        outcome.trace.vertices.assign(
            hostTrace.begin(),
            hostTrace.begin() + static_cast<std::ptrdiff_t>(outcome.trace.count));
        return outcome;
    }

    /** Test-only source that injects a non-finite position into the photon flow. */
    struct CorruptPhotonSource
    {
        ALPAKA_FN_ACC Photon operator()([[maybe_unused]] auto const& acc, std::uint64_t id) const
        {
            Photon photon;
            photon[idTag] = id;
            photon[positionTag] = Vec3{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
            photon[directionTag] = Vec3{0.0, 0.0, 1.0};
            photon[scatterCountTag] = 0u;
            photon[proposalTag] = graphFlow::photon::Proposal{};
            return photon;
        }
    };
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
    static_assert(graphFlow::AttributeRecord<Photon>);
    static_assert(ClassifyEvent::maxBins == graphFlow::photon::eventBins);

    // A subset proxy narrows access only; loading it still copies every attribute.
    using ScatterCountSubset = graphFlow::Set<graphFlow::photon::ScatterCountTag>;
    Photon original = makePhoton(0.3, Vec3{0.1, 0.2, 0.8}, 4u);
    original[proposalTag] = graphFlow::photon::Proposal{Event::scatter, 0.25, true};
    auto proxy = original.proxy<ScatterCountSubset>();
    proxy[scatterCountTag] = 9u;
    Photon const copied = proxy.load();
    CHECK(copied[idTag] == original[idTag]);
    CHECK(copied[positionTag] == original[positionTag]);
    CHECK(copied[directionTag] == original[directionTag]);
    CHECK(copied[scatterCountTag] == 9u);
    CHECK(copied[proposalTag].event == original[proposalTag].event);
    CHECK(copied[proposalTag].distance == original[proposalTag].distance);
    CHECK(copied[proposalTag].valid == original[proposalTag].valid);

    // The Record specialization must key the RNG by the logical id and scatter count.
    CHECK(graphFlow::rng::streamId(original) == original[idTag]);
    CHECK(graphFlow::rng::streamEvent(original) == original[scatterCountTag]);
}

TEST_CASE("photon flight geometry handles faces, ties and vacuum", "[photon][geometry]")
{
    using graphFlow::photon::applyEvent;
    using graphFlow::photon::faceFirst;
    using graphFlow::photon::resolveFlight;

    Slab const vacuum{1.0, 0.0, 0.0};
    Slab const material{1.0, 1.0, 0.0};
    double const vacuumTotal = graphFlow::photon::sigmaTotal(vacuum);
    double const materialTotal = graphFlow::photon::sigmaTotal(material);

    auto const positive = resolveFlight(vacuum, vacuumTotal, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.0, 0.0, 8u);
    CHECK(positive.valid);
    CHECK(positive.event == Event::transmitted);
    CHECK(positive.distance == Catch::Approx(0.8).margin(1e-15));

    auto const negative = resolveFlight(vacuum, vacuumTotal, makePhoton(0.2, Vec3{0.0, 0.0, -1.0}), 0.0, 0.0, 8u);
    CHECK(negative.valid);
    CHECK(negative.event == Event::reflected);
    CHECK(negative.distance == Catch::Approx(0.2).margin(1e-15));

    auto const parallelVacuum = resolveFlight(vacuum, vacuumTotal, makePhoton(0.5, Vec3{1.0, 0.0, 0.0}), 0.0, 0.0, 8u);
    CHECK_FALSE(parallelVacuum.valid);

    auto const parallelMaterial
        = resolveFlight(material, materialTotal, makePhoton(0.5, Vec3{1.0, 0.0, 0.0}), 0.5, 0.0, 8u);
    CHECK(parallelMaterial.valid);
    CHECK(parallelMaterial.event == Event::scatter);
    CHECK(parallelMaterial.distance == Catch::Approx(0.5).margin(1e-15));

    // A tie between the face distance and the sampled flight belongs to the face.
    auto const tie = resolveFlight(material, materialTotal, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.8, 0.0, 8u);
    CHECK(tie.valid);
    CHECK(tie.event == Event::transmitted);
    CHECK(tie.distance == Catch::Approx(0.8).margin(1e-15));
    CHECK(faceFirst(0.8, 0.8));
    CHECK_FALSE(faceFirst(0.8, 0.79));

    auto const inside = resolveFlight(material, materialTotal, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.79, 0.0, 8u);
    CHECK(inside.valid);
    CHECK(inside.event == Event::scatter);
    CHECK(inside.distance == Catch::Approx(0.79).margin(1e-15));

    auto const capped = resolveFlight(material, materialTotal, makePhoton(0.2, Vec3{0.0, 0.0, 1.0}), 0.3, 0.0, 0u);
    CHECK(capped.valid);
    CHECK(capped.event == Event::capped);

    Slab const absorbing{1.0, 0.0, 1.0};
    auto const absorbed = resolveFlight(
        absorbing,
        graphFlow::photon::sigmaTotal(absorbing),
        makePhoton(0.2, Vec3{0.0, 0.0, 1.0}),
        0.3,
        0.0,
        8u);
    CHECK(absorbed.valid);
    CHECK(absorbed.event == Event::absorbed);

    // The face advance clamps z exactly and propagates the transverse position.
    Photon photon = makePhoton(0.3, Vec3{0.6, 0.0, 0.8});
    photon[proposalTag] = graphFlow::photon::Proposal{Event::transmitted, 0.875, true};
    CHECK(applyEvent(material, Event::transmitted, photon) == graphFlow::UpdateResult::success);
    CHECK(photon[positionTag].z == 1.0);
    CHECK(photon[positionTag].x == Catch::Approx(0.525).margin(1e-15));

    Photon clamped = makePhoton(0.3, Vec3{0.0, 0.0, -1.0});
    clamped[proposalTag] = graphFlow::photon::Proposal{Event::reflected, 0.3, true};
    CHECK(applyEvent(material, Event::reflected, clamped) == graphFlow::UpdateResult::success);
    CHECK(clamped[positionTag].z == 0.0);

    // Invalid proposals and inconsistent bin/event pairs are release-path errors.
    Photon missing = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
    CHECK(applyEvent(vacuum, Event::transmitted, missing) == graphFlow::UpdateResult::invalid);
    Photon mismatched = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
    mismatched[proposalTag] = graphFlow::photon::Proposal{Event::transmitted, 0.8, true};
    CHECK(applyEvent(vacuum, Event::reflected, mismatched) == graphFlow::UpdateResult::invalid);
    Photon wrongFaceDirection = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
    wrongFaceDirection[proposalTag] = graphFlow::photon::Proposal{Event::reflected, 0.2, true};
    CHECK(applyEvent(vacuum, Event::reflected, wrongFaceDirection) == graphFlow::UpdateResult::invalid);
}

TEST_CASE("photon debug validation rejects corrupted state", "[photon][debug]")
{
    using graphFlow::photon::applyEvent;
    using graphFlow::photon::resolveFlight;
    using graphFlow::photon::validatePhoton;
    using graphFlow::photon::validateProposal;

    Slab const vacuum{1.0, 0.0, 0.0};
    Slab const material{1.0, 1.0, 0.0};
    double const vacuumTotal = graphFlow::photon::sigmaTotal(vacuum);

    Photon const nanPosition = makePhoton(std::numeric_limits<double>::quiet_NaN(), Vec3{0.0, 0.0, 1.0});
    Photon const infDirection = makePhoton(0.2, Vec3{0.0, 0.0, std::numeric_limits<double>::infinity()});
    Photon const outside = makePhoton(1.5, Vec3{0.0, 0.0, 1.0});

    if constexpr(graphFlow::debug::checksEnabled)
    {
        CHECK(validatePhoton(material, makePhoton(0.5, Vec3{0.0, 0.0, 1.0})));
        CHECK_FALSE(validatePhoton(material, nanPosition));
        CHECK_FALSE(validatePhoton(material, infDirection));
        CHECK_FALSE(validatePhoton(material, outside));

        CHECK_FALSE(resolveFlight(vacuum, vacuumTotal, nanPosition, 0.0, 0.0, 8u).valid);
        CHECK_FALSE(resolveFlight(vacuum, vacuumTotal, infDirection, 0.0, 0.0, 8u).valid);
        CHECK_FALSE(resolveFlight(vacuum, vacuumTotal, outside, 0.0, 0.0, 8u).valid);

        Photon nanProposal = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
        nanProposal[proposalTag]
            = graphFlow::photon::Proposal{Event::transmitted, std::numeric_limits<double>::quiet_NaN(), true};
        CHECK_FALSE(validateProposal(material, nanProposal));
        CHECK(applyEvent(material, Event::transmitted, nanProposal) == graphFlow::UpdateResult::invalid);

        Photon negativeDistance = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
        negativeDistance[proposalTag] = graphFlow::photon::Proposal{Event::absorbed, -0.5, true};
        CHECK_FALSE(validateProposal(material, negativeDistance));

        Photon wrongPlane = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
        wrongPlane[proposalTag] = graphFlow::photon::Proposal{Event::transmitted, 0.4, true};
        CHECK_FALSE(validateProposal(material, wrongPlane));

        Photon nanAdvance = makePhoton(0.2, Vec3{0.0, 0.0, 1.0});
        nanAdvance[positionTag].x = std::numeric_limits<double>::quiet_NaN();
        nanAdvance[proposalTag] = graphFlow::photon::Proposal{Event::transmitted, 0.8, true};
        CHECK(applyEvent(material, Event::transmitted, nanAdvance) == graphFlow::UpdateResult::invalid);
    }
    else
    {
        // Release trusts host-validated state. The resolver carries no per-event state branch, so
        // corrupted values reach a proposal instead of a status error.
        CHECK(resolveFlight(vacuum, vacuumTotal, nanPosition, 0.0, 0.0, 8u).valid);
        CHECK(resolveFlight(vacuum, vacuumTotal, infDirection, 0.0, 0.0, 8u).valid);
        CHECK(resolveFlight(vacuum, vacuumTotal, outside, 0.0, 0.0, 8u).valid);
    }
}

TEST_CASE("debug validation fails a corrupted photon batch before publication", "[photon][debug]")
{
    if constexpr(!graphFlow::debug::checksEnabled)
    {
        SUCCEED("debug validation is compiled out");
        return;
    }

    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    static_assert(
        std::same_as<typename graphFlow::SoAFlow<Photon, Device>::Storage, graphFlow::photon::PhotonStorage<Device>>);
    constexpr std::uint64_t photons = 1u;
    auto records = alpaka::onHost::alloc<TerminalRecord>(device, alpaka::Vec{photons});
    auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{photons});
    {
        std::vector<TerminalRecord> const initial(
            static_cast<std::size_t>(photons),
            TerminalRecord{graphFlow::photon::missingEvent, 0u, Vec3{}, Vec3{}});
        std::vector<std::uint32_t> const zero(static_cast<std::size_t>(photons), 0u);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, records, initial);
        alpaka::onHost::memcpy(queue, visits, zero);
    }

    Slab const slab{1.0, 0.0, 0.0};
    using RecordView = decltype(records.getView());
    using VisitView = decltype(visits.getView());

    graphFlow::SoAFlow<Photon, Device> flow{device};
    auto source = flow.source("corrupt", CorruptPhotonSource{});
    auto flight = flow.transform("propose flight", ProposeFlight{slab, 8u});
    auto route = flow.partition("advance and route", ClassifyEvent{}, AdvanceToEvent{slab});
    auto scatter = flow.transform("scatter", ScatterT<>{slab, ScatterMode::forward, NoTrace{}});
    auto terminal
        = flow.sink("terminal", RecordTerminal<RecordView, VisitView>{records.getView(), visits.getView(), NoTrace{}});
    flow.connect(source, flight);
    flow.connect(flight, route);
    flow.connect(route.template out<Event::scatter>(), scatter);
    flow.connect(scatter, flight);
    flow.connect(route.template out<Event::transmitted>(), terminal);
    flow.connect(route.template out<Event::reflected>(), terminal);
    flow.connect(route.template out<Event::absorbed>(), terminal);
    flow.connect(route.template out<Event::capped>(), terminal);

    graphFlow::Tuning tuning;
    tuning.chunkCapacity = 1u;
    tuning.liveParticles = 1u;
    tuning.tileSize = 1u;
    tuning.batchChunks = 1u;
    tuning.frameCap = 1u;
    flow.configure(tuning);

    CHECK_THROWS_AS(flow.run(source, photons), graphFlow::BatchError);

    std::vector<TerminalRecord> hostRecords(static_cast<std::size_t>(photons));
    std::vector<std::uint32_t> hostVisits(static_cast<std::size_t>(photons), 0u);
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, hostRecords, records);
        alpaka::onHost::memcpy(queue, hostVisits, visits);
    }
    CHECK(hostRecords[0].event == graphFlow::photon::missingEvent);
    CHECK(hostVisits[0] == 0u);
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

    REQUIRE(outcome.trace.count == 5u);
    CHECK_FALSE(outcome.trace.truncated);
    auto const& vertices = outcome.trace.vertices;
    std::vector<graphFlow::photon::TraceKind> const kinds{
        graphFlow::photon::TraceKind::source,
        graphFlow::photon::TraceKind::scatter,
        graphFlow::photon::TraceKind::scatter,
        graphFlow::photon::TraceKind::scatter,
        graphFlow::photon::TraceKind::reflected};
    std::vector<double> const heights{0.0, 0.2, 0.4, 0.3, 0.0};
    std::vector<double> const directions{1.0, 0.5, -0.5, -1.0, -1.0};
    std::vector<std::uint32_t> const counts{0u, 1u, 2u, 3u, 3u};
    for(std::size_t step = 0u; step < vertices.size(); ++step)
    {
        CAPTURE(step);
        CHECK(vertices[step].kind == kinds[step]);
        CHECK(vertices[step].scatterCount == counts[step]);
        CHECK(vertices[step].position.x == Catch::Approx(0.0).margin(1e-15));
        CHECK(vertices[step].position.y == Catch::Approx(0.0).margin(1e-15));
        CHECK(vertices[step].position.z == Catch::Approx(heights[step]).margin(1e-15));
        CHECK(vertices[step].direction.z == Catch::Approx(directions[step]).margin(1e-15));
    }

    // A bounded trace truncates independently of the simulation cap: the history still terminates
    // reflected, not capped, and all four flights still run.
    ScriptedOutcome const truncated = runScripted(2u);
    CHECK(truncated.flights == 4u);
    CHECK(truncated.trace.truncated);
    CHECK(truncated.trace.count == 2u);
    CHECK(truncated.terminal.event == Event::reflected);
    CHECK(truncated.terminal.scatterCount == 3u);
    REQUIRE(truncated.trace.vertices.size() == 2u);
    CHECK(truncated.trace.vertices[0u].kind == graphFlow::photon::TraceKind::source);
    CHECK(truncated.trace.vertices[1u].kind == graphFlow::photon::TraceKind::scatter);
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
    auto run = [&](std::uint32_t chunkSize, std::uint32_t batchChunks, std::uint32_t occupancyTarget = 0u)
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
        graphFlow::Tuning tuning = graphFlow::photon::tuningOf(parameters);
        if(occupancyTarget != 0u)
            tuning.occupancyTargetChunks = occupancyTarget;
        return graphFlow::photon::runPhotonScattering(device, parameters, tuning);
    };

    auto const small = run(1u, 1u);
    auto const large = run(128u, 8u);
    // The photon graph registers five lanes, so a target of five is reachable and binds.
    auto const gated = run(128u, 8u, 5u);
    CHECK(gated.stats.occupancyTargetChunks == 5u);
    CHECK(gated.stats.liveWorkChunks == 0u);
    // The physics comparison below is the subject here; whether the gate actually engaged is
    // asserted by the admission tests on their live-work record. `stats.refills` depends on
    // scheduler interleaving, so it stays a reported statistic rather than an assertion.

    auto const compare = [&](graphFlow::photon::ScatteringResult const& expected)
    {
        REQUIRE(small.terminal.size() == expected.terminal.size());
        for(std::size_t id = 0u; id < small.terminal.size(); ++id)
        {
            CHECK(small.terminal[id].event == expected.terminal[id].event);
            CHECK(small.terminal[id].scatterCount == expected.terminal[id].scatterCount);
            CHECK(small.terminal[id].position.x == expected.terminal[id].position.x);
            CHECK(small.terminal[id].position.y == expected.terminal[id].position.y);
            CHECK(small.terminal[id].position.z == expected.terminal[id].position.z);
            CHECK(small.terminal[id].direction.x == expected.terminal[id].direction.x);
            CHECK(small.terminal[id].direction.y == expected.terminal[id].direction.y);
            CHECK(small.terminal[id].direction.z == expected.terminal[id].direction.z);
        }
    };
    compare(large);
    compare(gated);
}

TEST_CASE("forward photon scattering reproduces the Poisson collision count", "[photon][physics][diagnostics]")
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

TEST_CASE("isotropic photon scattering reproduces the uncollided fraction", "[photon][physics][diagnostics]")
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
    CHECK(
        static_cast<double>(uncollided) / static_cast<double>(photons) == Catch::Approx(expected).margin(6.0 * sigma));

    for(auto const& record : result.terminal)
    {
        double const norm = record.direction.x * record.direction.x + record.direction.y * record.direction.y
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
    double const transmission
        = static_cast<double>(countEvent(absorbed, Event::transmitted)) / static_cast<double>(photons);
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
    CHECK(
        countEvent(result, Event::transmitted) + countEvent(result, Event::reflected)
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

TEST_CASE("photon trace is bounded and independent of the simulation cap", "[photon][trace]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    Parameters parameters;
    parameters.numPhotons = 64u;
    parameters.chunkSize = 16u;
    parameters.seed = 5u;
    parameters.thickness = 1.0;
    parameters.sigmaS = 1.0;
    parameters.sigmaA = 0.0;
    parameters.mode = ScatterMode::isotropic;
    parameters.maxScatters = 2u;
    parameters.trace = graphFlow::photon::TraceConfig{true, 8u, 3u};

    auto const traced = graphFlow::photon::runPhotonScattering(device, parameters);
    checkTerminalAccounting(traced, parameters.numPhotons);
    REQUIRE(traced.traces.size() == 8u);
    CHECK(traced.capped > 0u);
    CHECK(traced.traceTruncated > 0u);
    std::uint64_t truncated = 0u;
    for(auto const& trace : traced.traces)
    {
        CAPTURE(trace.id);
        CHECK(trace.id < 8u);
        CHECK(trace.count <= 3u);
        REQUIRE(trace.vertices.size() == trace.count);
        if(!trace.vertices.empty())
        {
            CHECK(trace.vertices.front().kind == graphFlow::photon::TraceKind::source);
            if(!trace.truncated)
            {
                auto const last = trace.vertices.back().kind;
                CHECK(last != graphFlow::photon::TraceKind::source);
                CHECK(last != graphFlow::photon::TraceKind::scatter);
            }
            else
            {
                CHECK(trace.count == parameters.trace.maxVertices);
            }
        }
        if(trace.truncated)
            ++truncated;
    }
    CHECK(truncated == traced.traceTruncated);

    Parameters untraced = parameters;
    untraced.trace = graphFlow::photon::TraceConfig{};
    auto const baseline = graphFlow::photon::runPhotonScattering(device, untraced);
    CHECK(baseline.traces.empty());
    for(std::uint32_t event = 0u; event < graphFlow::photon::eventBins; ++event)
        CHECK(traced.eventCounts[event] == baseline.eventCounts[event]);
    REQUIRE(traced.terminal.size() == baseline.terminal.size());
    for(std::size_t id = 0u; id < traced.terminal.size(); ++id)
    {
        CHECK(traced.terminal[id].event == baseline.terminal[id].event);
        CHECK(traced.terminal[id].scatterCount == baseline.terminal[id].scatterCount);
        CHECK(traced.terminal[id].position.z == baseline.terminal[id].position.z);
        CHECK(traced.terminal[id].direction.z == baseline.terminal[id].direction.z);
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
        "2",
        "--output-dir",
        "/tmp/photon-cli-output",
        "--trace-photons",
        "3",
        "--trace-vertices",
        "7"};
    std::vector<char const*> aliasArgv;
    aliasArgv.reserve(aliasArgs.size());
    for(auto const& argument : aliasArgs)
        aliasArgv.push_back(argument.c_str());
    auto const aliases = graphFlow::photon::cli::parseOptions(static_cast<int>(aliasArgv.size()), aliasArgv.data());
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
    REQUIRE(aliases.outputDirectory.has_value());
    CHECK(*aliases.outputDirectory == "/tmp/photon-cli-output");
    CHECK(aliases.tracePhotons == 3u);
    CHECK(aliases.traceVertices == 7u);

    graphFlow::photon::cli::Options negativeSigma = valid;
    negativeSigma.sigmaS = -1.0;
    CHECK_THROWS_AS(graphFlow::photon::cli::resolveOptions(negativeSigma, properties), std::invalid_argument);

    graphFlow::photon::cli::Options negativeAbsorption = valid;
    negativeAbsorption.sigmaA = -1.0;
    CHECK_THROWS_AS(graphFlow::photon::cli::resolveOptions(negativeAbsorption, properties), std::invalid_argument);

    graphFlow::photon::cli::Options zeroThickness = valid;
    zeroThickness.thickness = 0.0;
    CHECK_THROWS_AS(graphFlow::photon::cli::resolveOptions(zeroThickness, properties), std::invalid_argument);

    graphFlow::photon::cli::Options unknownMode = valid;
    unknownMode.mode = "ballistic";
    CHECK_THROWS_AS(graphFlow::photon::cli::resolveOptions(unknownMode, properties), std::invalid_argument);

    graphFlow::photon::cli::Options zeroBudget = valid;
    zeroBudget.poolPayloadBudgetBytes = 0u;
    CHECK_THROWS_AS(graphFlow::photon::cli::resolveOptions(zeroBudget, properties), std::invalid_argument);
}

TEST_CASE("photon CLI exposes the chunk-slot layout", "[photon][cli][arena]")
{
    using Catch::Matchers::ContainsSubstring;

    auto const properties = graphFlow::test::makeCpuDevice().getDeviceProperties();
    auto const parse = [](std::vector<std::string> const& arguments)
    {
        std::vector<char const*> parseArgv;
        parseArgv.reserve(arguments.size());
        for(auto const& argument : arguments)
            parseArgv.push_back(argument.c_str());
        return graphFlow::photon::cli::parseOptions(static_cast<int>(parseArgv.size()), parseArgv.data());
    };

    // The CPU backend defaults to the slot arena with a 1 GiB payload budget.
    graphFlow::photon::cli::Options defaults;
    defaults.numPhotons = 17u;
    CHECK(defaults.poolMode == graphFlow::PoolMode::fixed);
    CHECK_FALSE(defaults.poolModeExplicit);
    CHECK_FALSE(defaults.slotArenaBudgetBytes.has_value());
    auto const resolvedDefaults = graphFlow::photon::cli::resolveOptions(defaults, properties);
    CHECK(resolvedDefaults.parameters.poolMode == graphFlow::PoolMode::slotArena);
    CHECK(resolvedDefaults.parameters.poolPayloadBudgetBytes == 1'073'741'824u);
    CHECK(resolvedDefaults.parameters.slotArenaBudgetBytes == 0u);
    CHECK(resolvedDefaults.policy.poolMode == graphFlow::PoolMode::slotArena);
    CHECK(resolvedDefaults.policy.memoryBudgetBytes == 1'073'741'824u);
    REQUIRE(resolvedDefaults.tuning.policyMemoryBudgetBytes.has_value());
    CHECK(*resolvedDefaults.tuning.policyMemoryBudgetBytes == 1'073'741'824u);
    CHECK_FALSE(resolvedDefaults.tuning.slotArenaBudgetBytes.has_value());

    // An explicit fixed layout keeps the fixed policy but still receives the 1 GiB CPU budget.
    graphFlow::photon::cli::Options fixedDefaults;
    fixedDefaults.numPhotons = 17u;
    fixedDefaults.poolMode = graphFlow::PoolMode::fixed;
    fixedDefaults.poolModeExplicit = true;
    auto const resolvedFixedDefaults = graphFlow::photon::cli::resolveOptions(fixedDefaults, properties);
    CHECK(resolvedFixedDefaults.parameters.poolMode == graphFlow::PoolMode::fixed);
    CHECK(resolvedFixedDefaults.policy.poolMode == graphFlow::PoolMode::fixed);
    CHECK(resolvedFixedDefaults.policy.memoryBudgetBytes == 1'073'741'824u);
    CHECK(resolvedFixedDefaults.parameters.poolPayloadBudgetBytes == 1'073'741'824u);

    // The flags exist on this example because the photon flow has transform nodes, so the arena
    // and the per-chunk schedule both have a real effect. Here only the layout is exercised.
    auto const options = parse(
        {"graphFlow_photon_scattering",
         "--photons",
         "17",
         "--pool-mode",
         "slotArena",
         "--slot-arena-budget",
         "67108864"});
    CHECK(options.poolMode == graphFlow::PoolMode::slotArena);
    REQUIRE(options.slotArenaBudgetBytes.has_value());
    CHECK(*options.slotArenaBudgetBytes == 67'108'864u);

    auto const resolved = graphFlow::photon::cli::resolveOptions(options, properties);
    CHECK(resolved.parameters.poolMode == graphFlow::PoolMode::slotArena);
    CHECK(resolved.parameters.slotArenaBudgetBytes == 67'108'864u);
    CHECK(resolved.policy.poolMode == graphFlow::PoolMode::slotArena);
    REQUIRE(resolved.tuning.slotArenaBudgetBytes.has_value());
    CHECK(*resolved.tuning.slotArenaBudgetBytes == 67'108'864u);
    // The 64 MiB arena is a sub-budget of the 1 GiB CPU default, so the pool budget stays at 1 GiB.
    REQUIRE(resolved.tuning.poolPayloadBudgetBytes.has_value());
    CHECK(*resolved.tuning.poolPayloadBudgetBytes == 1'073'741'824u);

    // Zero keeps the whole pool budget, exactly like an absent flag.
    graphFlow::photon::cli::Options zero;
    zero.numPhotons = 17u;
    zero.poolMode = graphFlow::PoolMode::slotArena;
    zero.poolModeExplicit = true;
    zero.slotArenaBudgetBytes = 0u;
    auto const resolvedZero = graphFlow::photon::cli::resolveOptions(zero, properties);
    CHECK(resolvedZero.parameters.poolMode == graphFlow::PoolMode::slotArena);
    CHECK_FALSE(resolvedZero.tuning.slotArenaBudgetBytes.has_value());

    CHECK_THROWS_WITH(
        parse({"graphFlow_photon_scattering", "--pool-mode", "arena"}),
        ContainsSubstring("expected fixed or slotArena"));

    // The budget only applies to the slot arena, so a fixed run rejects it instead of ignoring it.
    graphFlow::photon::cli::Options fixedWithBudget;
    fixedWithBudget.numPhotons = 17u;
    fixedWithBudget.poolMode = graphFlow::PoolMode::fixed;
    fixedWithBudget.poolModeExplicit = true;
    fixedWithBudget.slotArenaBudgetBytes = 1024u;
    CHECK_THROWS_WITH(
        graphFlow::photon::cli::resolveOptions(fixedWithBudget, properties),
        ContainsSubstring("--slot-arena-budget requires --pool-mode slotArena"));

    // An explicit --pool-budget-bytes is never rewritten: the arena only raises a derived budget,
    // so a larger explicit pool budget survives unchanged.
    auto const explicitLarge = parse(
        {"graphFlow_photon_scattering",
         "--photons",
         "17",
         "--pool-mode",
         "slotArena",
         "--pool-budget-bytes",
         "134217728",
         "--slot-arena-budget",
         "67108864"});
    auto const resolvedLarge = graphFlow::photon::cli::resolveOptions(explicitLarge, properties);
    CHECK(resolvedLarge.parameters.poolPayloadBudgetBytes == 134'217'728u);
    REQUIRE(resolvedLarge.tuning.poolPayloadBudgetBytes.has_value());
    CHECK(*resolvedLarge.tuning.poolPayloadBudgetBytes == 134'217'728u);

    // A pool budget smaller than the arena cannot hold the arena, so the run is rejected with a
    // message that names both byte counts instead of silently lowering the user's budget.
    std::string budgetError;
    try
    {
        auto const conflict = parse(
            {"graphFlow_photon_scattering",
             "--photons",
             "17",
             "--chunk-size",
             "32",
             "--pool-mode",
             "slotArena",
             "--pool-budget-bytes",
             "1048576",
             "--slot-arena-budget",
             "67108864"});
        (void) graphFlow::photon::cli::resolveOptions(conflict, properties);
    }
    catch(std::invalid_argument const& error)
    {
        budgetError = error.what();
    }
    CHECK(budgetError.find("--slot-arena-budget (67108864 bytes)") != std::string::npos);
    CHECK(budgetError.find("--pool-budget-bytes (1048576 bytes)") != std::string::npos);
}

TEST_CASE("photon slot-arena run reports the arena statistics end to end", "[photon][arena]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    auto const properties = device.getDeviceProperties();
    constexpr std::uint64_t photons = 512u;

    std::vector<std::string> const arguments{
        "graphFlow_photon_scattering",
        "--photons",
        "512",
        "--chunk-size",
        "64",
        "--seed",
        "42",
        "--max-scatters",
        "64",
        "--pool-mode",
        "slotArena",
        "--slot-arena-budget",
        "16777216"};
    std::vector<char const*> parseArgv;
    parseArgv.reserve(arguments.size());
    for(auto const& argument : arguments)
        parseArgv.push_back(argument.c_str());
    auto const options = graphFlow::photon::cli::parseOptions(static_cast<int>(parseArgv.size()), parseArgv.data());
    auto const resolved = graphFlow::photon::cli::resolveOptions(options, properties);

    auto const result = graphFlow::photon::runPhotonScattering(device, resolved.parameters, resolved.tuning);
    checkTerminalAccounting(result, photons);

    // A slot-arena run publishes the arena keys, so the high-water mark of a real run is visible
    // in the scheduler line instead of only being derivable from the code.
    CHECK(result.stats.slotArena);
    CHECK(result.stats.poolSlots == 16'777'216u / graphFlow::photon::Photon::attribute_list::slotBytes);
    CHECK(result.stats.requiredArenaSlots > 0u);
    CHECK(result.stats.requiredArenaSlots <= result.stats.poolSlots);
    std::string const schedulerLine = graphFlow::formatSchedulerStats(result.stats);
    CHECK(schedulerLine.find(" poolSlots=") != std::string::npos);
    CHECK(schedulerLine.find(" poolBytes=") != std::string::npos);
    CHECK(schedulerLine.find(" peakLiveFullChunks=") != std::string::npos);
    CHECK(schedulerLine.find(" requiredArenaSlots=") != std::string::npos);
}
