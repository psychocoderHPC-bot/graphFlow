/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
    struct AdmissionTrace
    {
        std::vector<std::uint32_t> fills;
        std::vector<std::uint64_t> firstIds;
        /** Registry live work (`reserved + ready + inflight`) sampled at every source admission. */
        std::vector<std::uint32_t> liveWorkAtAdmission;
        std::uint64_t admissions = 0u;
        std::uint64_t retirements = 0u;
        /** Source admissions that happened before the first sink retirement. */
        std::uint64_t admitsBeforeFirstRetirement = 0u;
        graphFlow::SchedulerStats stats;
        graphFlow::ChunkRegistrySummary registry;
    };

    /** Run a source -> sink graph and record every source chunk the scheduler admits.
     *
     * The graph has no transform, so the pool only has to hold the admitted source chunks and the
     * chunk the sink still owns. The hook records the fill before any device work, which is exactly
     * the value the admission policy selected.
     */
    template<typename TDevice>
    AdmissionTrace traceSourceAdmission(
        TDevice device,
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t liveParticles,
        std::uint32_t numChunks,
        std::uint32_t occupancyTarget = 0u)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;

        graphFlow::ChunkPool pool{device, numChunks, capacity};
        AdmissionTrace trace;
        Runner* runnerPtr = nullptr;

        typename Runner::Hooks sourceHooks;
        sourceHooks.launch = [&trace, &runnerPtr](typename Runner::Queue const&, typename Runner::Batch const& batch)
        {
            trace.fills.push_back(batch.chunk().fill);
            trace.firstIds.push_back(batch.firstId);
            trace.liveWorkAtAdmission.push_back(runnerPtr->snapshot().liveWorkChunks);
            ++trace.admissions;
            if(trace.retirements == 0u)
                ++trace.admitsBeforeFirstRetirement;
        };
        sourceHooks.advance
            = [](typename Runner::Queue const&, typename Runner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sourceHooks.finish = [](typename Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        { outputs.push_back(batch.chunk()); };
        sourceHooks.abort = [] {};

        typename Runner::Hooks sinkHooks;
        sinkHooks.launch = [](typename Runner::Queue const&, typename Runner::Batch const&) {};
        sinkHooks.advance = [](typename Runner::Queue const&, typename Runner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sinkHooks.finish = [&trace](typename Runner::Batch const&, std::vector<graphFlow::ChunkDescriptor>&)
        { ++trace.retirements; };
        sinkHooks.abort = [] {};

        Runner runner{device, pool};
        runnerPtr = &runner;
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, std::move(sinkHooks));
        runner.setSource(source, particles, liveParticles);
        runner.setStallTimeout(std::chrono::milliseconds{10000});
        if(occupancyTarget != 0u)
            runner.setOccupancyTarget(occupancyTarget);
        trace.stats = runner.run();
        trace.registry = runner.chunkRegistrySummary();
        runner.validateChunkRegistry();

        CHECK(pool.freeCount() == pool.numChunks());
        CHECK(pool.peakLive() <= pool.numChunks());
        CHECK(trace.stats.poolChunks == numChunks);

        REQUIRE(trace.firstIds.size() == trace.fills.size());
        std::uint64_t expectedFirstId = 0u;
        for(std::size_t i = 0u; i < trace.fills.size(); ++i)
        {
            CHECK(trace.firstIds[i] == expectedFirstId);
            expectedFirstId += trace.fills[i];
        }
        CHECK(expectedFirstId == particles);
        return trace;
    }

    /** Expected admission sequence under `min(C, P, remaining)`.
     *
     * A zero P is the documented "budget disabled" mode of the runner, so the cap is C there.
     */
    std::vector<std::uint32_t> expectedFills(
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t liveParticles)
    {
        std::uint64_t const admissionCap
            = liveParticles == 0u ? capacity : std::min<std::uint64_t>(capacity, liveParticles);
        std::vector<std::uint32_t> fills;
        std::uint64_t remaining = particles;
        while(remaining != 0u)
        {
            std::uint32_t const fill = static_cast<std::uint32_t>(std::min<std::uint64_t>(admissionCap, remaining));
            fills.push_back(fill);
            remaining -= fill;
        }
        return fills;
    }
} // namespace

TEST_CASE("source admission fills chunks with min(C,P,remaining)", "[integration][admission]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    SECTION("P below the chunk capacity limits each admission")
    {
        constexpr std::uint64_t particles = 500u;
        constexpr std::uint32_t capacity = 128u;
        constexpr std::uint32_t liveParticles = 48u;

        AdmissionTrace const trace = traceSourceAdmission(device, particles, capacity, liveParticles, 8u);
        CAPTURE(trace.fills.size());

        CHECK(trace.fills == expectedFills(particles, capacity, liveParticles));
        CHECK(trace.fills.front() == liveParticles);
        CHECK(trace.fills.back() == 20u);
    }

    SECTION("P equal to the chunk capacity keeps full chunks")
    {
        constexpr std::uint64_t particles = 500u;
        constexpr std::uint32_t capacity = 128u;
        constexpr std::uint32_t liveParticles = 128u;

        AdmissionTrace const trace = traceSourceAdmission(device, particles, capacity, liveParticles, 8u);

        CHECK(trace.fills == expectedFills(particles, capacity, liveParticles));
        CHECK(trace.fills == std::vector<std::uint32_t>{128u, 128u, 128u, 116u});
    }

    SECTION("P zero disables the particle budget and fills up to C")
    {
        constexpr std::uint64_t particles = 1000u;
        constexpr std::uint32_t capacity = 128u;
        constexpr std::uint32_t liveParticles = 0u;

        AdmissionTrace const trace = traceSourceAdmission(device, particles, capacity, liveParticles, 16u);

        CHECK(trace.fills == expectedFills(particles, capacity, liveParticles));
        CHECK(trace.fills.front() == capacity);
        CHECK(trace.fills.back() == 104u);
    }

    SECTION("the tail is a partial chunk, not an empty or padded admission")
    {
        constexpr std::uint64_t particles = 1000u;
        constexpr std::uint32_t capacity = 128u;
        constexpr std::uint32_t liveParticles = 256u;

        AdmissionTrace const trace = traceSourceAdmission(device, particles, capacity, liveParticles, 12u);

        CHECK(trace.fills == expectedFills(particles, capacity, liveParticles));
        REQUIRE(trace.fills.size() == 8u);
        CHECK(trace.fills.back() == 104u);
        for(std::size_t i = 0u; i + 1u < trace.fills.size(); ++i)
            CHECK(trace.fills[i] == capacity);
    }

    SECTION("N much larger than P completes with a bounded chunk pool")
    {
        constexpr std::uint64_t particles = 100000u;
        constexpr std::uint32_t capacity = 128u;
        constexpr std::uint32_t liveParticles = 1024u;
        constexpr std::uint32_t numChunks = 16u;

        AdmissionTrace const trace = traceSourceAdmission(device, particles, capacity, liveParticles, numChunks);

        CHECK(trace.fills == expectedFills(particles, capacity, liveParticles));
        REQUIRE(trace.fills.size() == 782u);
        CHECK(trace.fills.back() == 32u);
        CHECK(trace.stats.peakLiveChunks <= numChunks);
        CHECK(trace.stats.peakLiveChunks < particles / capacity);
    }

    SECTION("zero particles admit no chunk")
    {
        AdmissionTrace const trace = traceSourceAdmission(device, 0u, 128u, 256u, 8u);

        CHECK(trace.fills.empty());
        CHECK(trace.firstIds.empty());
    }
}

TEST_CASE("one-particle source chunks are rejected by the admission policy", "[integration][admission][mutation]")
{
    // N is an exact multiple of C and P covers several chunks, so every admitted chunk must be
    // full. A regression that admits one particle per source chunk would yield 4096 chunks with
    // fill 1 and fail the size, fill and first-id assertions below.
    constexpr std::uint64_t particles = 4096u;
    constexpr std::uint32_t capacity = 128u;
    constexpr std::uint32_t liveParticles = 512u;

    auto const device = graphFlow::test::makeCpuDevice();
    AdmissionTrace const trace = traceSourceAdmission(device, particles, capacity, liveParticles, 8u);

    REQUIRE(trace.fills.size() == 32u);
    CHECK(
        std::all_of(
            trace.fills.begin(),
            trace.fills.end(),
            [](std::uint32_t const fill) { return fill == capacity; }));
    CHECK(std::count(trace.fills.begin(), trace.fills.end(), 1u) == 0u);
    CHECK(trace.fills == expectedFills(particles, capacity, liveParticles));
}

TEST_CASE("occupancy gate is off by default and keeps the historical output", "[integration][admission]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    AdmissionTrace const trace = traceSourceAdmission(device, 1000u, 128u, 256u, 12u);

    CHECK(trace.stats.occupancyTargetChunks == 0u);
    CHECK(trace.stats.peakLiveWorkChunks == 0u);
    CHECK(trace.stats.refills == 0u);
    CHECK(trace.fills == expectedFills(1000u, 128u, 256u));
    // The default stats line must stay byte-identical: the occupancy keys are appended only when
    // a target is configured.
    CHECK(graphFlow::formatSchedulerStats(trace.stats).find("occupancyTargetChunks") == std::string::npos);
}

TEST_CASE("occupancy target bounds live chunk work and the run completes", "[integration][admission]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    // P would allow eight full chunks, so only the chunk-count gate can hold live work at three.
    constexpr std::uint32_t target = 3u;
    AdmissionTrace const trace = traceSourceAdmission(device, 1024u, 128u, 1024u, 16u, target);

    CHECK(trace.stats.occupancyTargetChunks == target);
    CHECK(trace.stats.peakLiveWorkChunks <= target);
    REQUIRE(trace.liveWorkAtAdmission.size() == trace.fills.size());
    for(std::uint32_t const liveWork : trace.liveWorkAtAdmission)
        CHECK(liveWork <= target);
    // Scheduling changed but the admitted particle stream did not.
    CHECK(trace.fills == expectedFills(1024u, 128u, 1024u));
    CHECK(trace.stats.sourceSunk == 1024u);
    CHECK(trace.stats.liveWorkChunks == 0u);
    CHECK(trace.stats.refills > 0u);
    CHECK(trace.registry.faults == 0u);
    CHECK(trace.registry.acquisitions == trace.registry.releases);
}

TEST_CASE("occupancy target stalls the source until a sink retirement refills it", "[integration][admission]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    // Two registered lanes (source and sink) and a target of two: the source may only prefill the
    // pipeline, then it has to wait for a retirement to feed the next chunk.
    constexpr std::uint32_t target = 2u;
    AdmissionTrace const trace = traceSourceAdmission(device, 512u, 64u, 512u, 16u, target);

    CHECK(trace.admitsBeforeFirstRetirement == target);
    CHECK(trace.retirements > 0u);
    CHECK(trace.fills.size() > target); // the source resumed after the backpressure
    CHECK(trace.stats.refills > 0u);
    REQUIRE(trace.liveWorkAtAdmission.size() == trace.fills.size());
    for(std::uint32_t const liveWork : trace.liveWorkAtAdmission)
        CHECK(liveWork <= target);

    // No chunk is lost or retired twice: every source particle reached the sink exactly once and
    // the registry balances acquisitions against releases with no faults.
    CHECK(trace.stats.sourceGenerated == 512u);
    CHECK(trace.stats.sourceSunk == 512u);
    CHECK(trace.retirements == trace.registry.retirements);
    CHECK(trace.registry.acquisitions == trace.registry.releases);
    CHECK(trace.registry.faults == 0u);
    CHECK(trace.stats.liveWorkChunks == 0u);
}

namespace
{
    /** Fragmentation case: one output chunk per particle grows the live work past the source target.
     *
     * The body is a function template so `GraphRunner<TDevice>` and its `Hooks` lambdas are
     * instantiated in a dependent context: cudafe++ (CUDA 12.5) crashes on the same lambdas
     * written directly in a non-template function.
     */
    template<typename TDevice>
    void runFragmentedOccupancyGate(TDevice device)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;

        constexpr std::uint64_t particles = 16u;
        constexpr std::uint32_t capacity = 8u;
        constexpr std::uint32_t liveParticles = 16u;
        constexpr std::uint32_t numChunks = 32u;
        constexpr std::uint32_t target = 4u;

        graphFlow::ChunkPool pool{device, numChunks, capacity};
        std::vector<graphFlow::ChunkDescriptor> pending;
        std::uint64_t sunk = 0u;
        std::uint64_t admissions = 0u;

        typename Runner::Hooks sourceHooks;
        sourceHooks.launch
            = [&admissions](typename Runner::Queue const&, typename Runner::Batch const&) { ++admissions; };
        sourceHooks.advance
            = [](typename Runner::Queue const&, typename Runner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sourceHooks.finish = [](typename Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        { outputs.push_back(batch.chunk()); };
        sourceHooks.abort = [] {};

        // One output chunk per particle: the downstream chunk count grows well past the source count.
        typename Runner::Hooks fragmentHooks;
        fragmentHooks.launch = [](typename Runner::Queue const&, typename Runner::Batch const&) {};
        fragmentHooks.advance
            = [&pool,
               &pending](typename Runner::Queue const&, typename Runner::Batch const& batch, std::uint32_t& allocated)
        {
            pending.clear();
            auto const outputs = static_cast<std::uint32_t>(batch.particles());
            for(std::uint32_t bin = 0u; bin < outputs; ++bin)
                pending.push_back(graphFlow::ChunkDescriptor{pool.acquire(), 1u, bin});
            allocated = outputs;
            return false;
        };
        fragmentHooks.finish
            = [&pending](typename Runner::Batch const&, std::vector<graphFlow::ChunkDescriptor>& outputs)
        {
            outputs.insert(outputs.end(), pending.begin(), pending.end());
            pending.clear();
        };
        fragmentHooks.abort = [&pending] { pending.clear(); };

        typename Runner::Hooks sinkHooks;
        sinkHooks.launch = [](typename Runner::Queue const&, typename Runner::Batch const&) {};
        sinkHooks.advance = [](typename Runner::Queue const&, typename Runner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sinkHooks.finish = [&sunk](typename Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>&)
        { sunk += batch.particles(); };
        sinkHooks.abort = [] {};

        Runner runner{device, pool};
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        std::vector<int> const fragmentRoutes(capacity, 2);
        runner.addNode("fragment", graphFlow::NodeRole::transform, capacity, fragmentRoutes, std::move(fragmentHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, std::move(sinkHooks));
        runner.setSource(source, particles, liveParticles);
        runner.setStallTimeout(std::chrono::milliseconds{10000});
        runner.setOccupancyTarget(target);

        graphFlow::SchedulerStats const stats = runner.run();
        graphFlow::ChunkRegistrySummary const registry = runner.chunkRegistrySummary();
        runner.validateChunkRegistry();

        CHECK(stats.occupancyTargetChunks == target);
        CHECK(admissions == 2u); // two eight-particle source chunks
        CHECK(stats.sourceGenerated == particles);
        CHECK(stats.sourceSunk == particles);
        CHECK(sunk == particles);
        CHECK(stats.liveWorkChunks == 0u);
        CHECK(registry.faults == 0u);
        CHECK(registry.acquisitions == registry.releases);
        CHECK(registry.retirements == 16u); // sixteen one-particle sink deliveries
        CHECK(pool.freeCount() == pool.numChunks());

        // The gate constrains source admission, not the total: every source chunk admitted while
        // the pipeline is below the target can expand into one chunk per particle, so the peak
        // legitimately exceeds `D`. The reserve bounds the fragment's output credit, not live work.
        CHECK(stats.peakLiveWorkChunks >= target);
        CHECK(stats.peakLiveWorkChunks > target);
        CHECK(registry.peakReservedChunks <= capacity);
    }
} // namespace

TEST_CASE("occupancy gate counts chunks when the stream fragments downstream", "[integration][admission]")
{
    runFragmentedOccupancyGate(graphFlow::test::makeCpuDevice());
}

namespace
{
    /** Pre-run validation: an unreachable target throws before the first launch.
     *
     * The body is a function template so `GraphRunner<TDevice>` and its `Hooks` lambdas are
     * instantiated in a dependent context: cudafe++ (CUDA 12.5) crashes on the same lambdas
     * written directly in a non-template function.
     */
    template<typename TDevice>
    void expectInvalidOccupancyTargetRejected(TDevice device)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;

        using Catch::Matchers::ContainsSubstring;

        graphFlow::ChunkPool pool{device, 8u, 8u};
        std::uint32_t launches = 0u;

        typename Runner::Hooks sourceHooks;
        sourceHooks.launch = [&launches](typename Runner::Queue const&, typename Runner::Batch const&) { ++launches; };
        sourceHooks.advance
            = [](typename Runner::Queue const&, typename Runner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sourceHooks.finish = [](typename Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        { outputs.push_back(batch.chunk()); };
        sourceHooks.abort = [] {};

        typename Runner::Hooks sinkHooks;
        sinkHooks.launch = [&launches](typename Runner::Queue const&, typename Runner::Batch const&) { ++launches; };
        sinkHooks.advance = [](typename Runner::Queue const&, typename Runner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sinkHooks.finish = [](typename Runner::Batch const&, std::vector<graphFlow::ChunkDescriptor>&) {};
        sinkHooks.abort = [] {};

        Runner runner{device, pool};
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, std::move(sinkHooks));
        runner.setSource(source, 8u, 8u);
        runner.setStallTimeout(std::chrono::milliseconds{10000});

        // Below the two registered lanes: rejected before the run.
        runner.setOccupancyTarget(1u);
        CHECK_THROWS_WITH(runner.run(), ContainsSubstring("below the registered lane count"));
        CHECK(launches == 0u);

        // Above `poolChunks - outputReserveChunks` (8 - 0): rejected before the run.
        runner.setOccupancyTarget(9u);
        CHECK_THROWS_WITH(runner.run(), ContainsSubstring("exceeds the reachable occupancy"));
        CHECK(launches == 0u);
    }
} // namespace

TEST_CASE("invalid occupancy target is rejected before any kernel runs", "[integration][admission]")
{
    expectInvalidOccupancyTargetRejected(graphFlow::test::makeCpuDevice());
}
