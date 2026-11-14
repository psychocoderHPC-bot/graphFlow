/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

namespace
{
    struct AdmissionTrace
    {
        std::vector<std::uint32_t> fills;
        std::vector<std::uint64_t> firstIds;
        graphFlow::SchedulerStats stats;
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
        std::uint32_t numChunks)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;

        graphFlow::ChunkPool pool{device, numChunks, capacity};
        AdmissionTrace trace;

        typename Runner::Hooks sourceHooks;
        sourceHooks.launch = [&trace](typename Runner::Queue const&, typename Runner::Batch const& batch)
        {
            trace.fills.push_back(batch.chunk().fill);
            trace.firstIds.push_back(batch.firstId);
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
        sinkHooks.finish = [](typename Runner::Batch const&, std::vector<graphFlow::ChunkDescriptor>&) {};
        sinkHooks.abort = [] {};

        Runner runner{device, pool};
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, std::move(sinkHooks));
        runner.setSource(source, particles, liveParticles);
        runner.setStallTimeout(std::chrono::milliseconds{10000});
        trace.stats = runner.run();

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
