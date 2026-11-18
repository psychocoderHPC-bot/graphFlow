/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ParticleTypes.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/flow/Tuning.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
    std::uint64_t chunkKey(graphFlow::ChunkHandle handle)
    {
        return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
    }

    /** Sink-side evidence of one depth run. Only the sink is instrumented, so the per-node
     * in-flight count is not polluted by the source lane. */
    struct SinkObservation
    {
        std::uint32_t inflight = 0u;
        std::uint32_t maxInflight = 0u;
        std::uint32_t launches = 0u;
        std::uint32_t retires = 0u;
        std::uint64_t particles = 0u;
        std::uint32_t duplicateDeliveries = 0u;
        std::unordered_set<std::uint64_t> delivered;
    };

    /** Drive a source -> sink graph with `depth` sink lanes.
     *
     * The sink hook enqueues a sleeping host task on its node queue. The completion event is
     * recorded after that task, so a batch stays in flight for the duration of the sleep. This is
     * test-only instrumentation: it forces the scheduler to keep several batches of the same node
     * live without depending on device timing. The source admits exactly `numChunks` full chunks,
     * and every input handle must reach the sink exactly once.
     */
    template<typename TDevice>
    void runDepthScenario(
        TDevice device,
        std::uint32_t depth,
        std::uint32_t numChunks,
        std::uint32_t capacity,
        bool idleWait = false)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;
        using Queue = typename Runner::Queue;
        using Batch = typename Runner::Batch;

        REQUIRE(depth != 0u);
        REQUIRE(numChunks > depth);
        graphFlow::ChunkPool<TDevice> pool{device, numChunks + 8u, capacity};

        auto observation = std::make_shared<SinkObservation>();

        typename Runner::Hooks sourceHooks;
        sourceHooks.launch = [](Queue const&, Batch const&) {};
        sourceHooks.advance = [](Queue const&, Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sourceHooks.finish = [](Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        { outputs.insert(outputs.end(), batch.chunks.begin(), batch.chunks.end()); };
        sourceHooks.abort = [] {};

        typename Runner::Hooks sinkHooks;
        sinkHooks.launch = [observation](Queue const& queue, Batch const&)
        {
            ++observation->launches;
            ++observation->inflight;
            observation->maxInflight = std::max(observation->maxInflight, observation->inflight);
            queue.enqueueHostFn([] { std::this_thread::sleep_for(std::chrono::milliseconds{50}); });
        };
        sinkHooks.advance = [](Queue const&, Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        sinkHooks.finish = [observation](Batch const& batch, std::vector<graphFlow::ChunkDescriptor>&)
        {
            REQUIRE(observation->inflight != 0u);
            --observation->inflight;
            ++observation->retires;
            for(graphFlow::ChunkDescriptor const& chunk : batch.chunks)
            {
                observation->particles += chunk.fill;
                if(!observation->delivered.insert(chunkKey(chunk.handle)).second)
                    ++observation->duplicateDeliveries;
            }
        };
        sinkHooks.abort = [observation]
        {
            if(observation->inflight != 0u)
                --observation->inflight;
        };

        Runner runner{device, pool};
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        auto const sink = runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, std::move(sinkHooks), depth);
        runner.setSource(source, static_cast<std::uint64_t>(numChunks) * capacity, 0u);
        runner.setStallTimeout(std::chrono::milliseconds{30000});
        runner.setIdleWait(idleWait);
        graphFlow::SchedulerStats const stats = runner.run();

        CHECK(observation->launches == numChunks);
        CHECK(observation->retires == numChunks);
        CHECK(observation->particles == static_cast<std::uint64_t>(numChunks) * capacity);
        CHECK(observation->duplicateDeliveries == 0u);
        CHECK(observation->inflight == 0u);
        // Depth is a hard bound: a node never submits more batches than it has lanes.
        CHECK(observation->maxInflight == depth);
        // More chunks than lanes reached the graph, so extra work waited in the ready deque.
        CHECK(stats.maxReadyDepth >= 1u);
        CHECK(stats.peakInflightBatches <= depth + 1u); // sink lanes plus the single source lane
        CHECK(runner.nodeCompleted(sink) == numChunks);
        // The wait mode must actually block instead of polling; the polling mode must not.
        CHECK(stats.idleWait == idleWait);
        if(idleWait)
            CHECK(stats.eventWaits > 0u);
        else
            CHECK(stats.eventWaits == 0u);
    }

    graphFlow::particle::FeedbackResult runFeedback(
        std::uint64_t numParticles,
        std::uint32_t capacity,
        std::uint32_t liveParticles,
        std::uint32_t batchChunks,
        std::uint32_t depth,
        std::optional<bool> idleWait = std::nullopt)
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = numParticles;
        parameters.chunkSize = capacity;
        parameters.liveParticles = liveParticles;
        parameters.seed = 42u;
        parameters.maxVisits = 64u;
        parameters.batchChunks = batchChunks;
        parameters.tileSize = 128u;

        graphFlow::Tuning tuning = graphFlow::particle::tuningOf(parameters);
        tuning.inflightBatches = depth;
        tuning.schedulerIdleWait = idleWait;
        return graphFlow::particle::runParticleFeedback(graphFlow::test::makeCpuDevice(), parameters, tuning);
    }
} // namespace

TEST_CASE("submitted depth rejects a zero value", "[integration][async][depth]")
{
    graphFlow::Tuning tuning;
    tuning.inflightBatches = 0u;
    CHECK_THROWS_AS(tuning.validate(), std::invalid_argument);
}

TEST_CASE("depth one keeps a single in-flight batch per node", "[integration][async][depth]")
{
    runDepthScenario(graphFlow::test::makeCpuDevice(), 1u, 12u, 8u);
}

TEST_CASE("depth above one overlaps batches and backpressures on lane exhaustion", "[integration][async][depth]")
{
    runDepthScenario(graphFlow::test::makeCpuDevice(), 4u, 16u, 8u);
}

TEST_CASE("idle wait blocks on completion and preserves the depth results", "[integration][async][depth][idle-wait]")
{
    runDepthScenario(graphFlow::test::makeCpuDevice(), 4u, 16u, 8u, true);
}

TEST_CASE("idle wait preserves the feedback histogram", "[integration][async][depth][idle-wait]")
{
    graphFlow::particle::FeedbackResult const polling = runFeedback(8192u, 1024u, 512u, 2u, 2u, false);
    graphFlow::particle::FeedbackResult const blocking = runFeedback(8192u, 1024u, 512u, 2u, 2u, true);
    CHECK(blocking.sunkParticles == polling.sunkParticles);
    CHECK(blocking.weights == polling.weights);
    CHECK(blocking.histogram == polling.histogram);
    CHECK(polling.stats.eventWaits == 0u);
    CHECK(blocking.stats.eventWaits > 0u);
    CHECK(blocking.stats.idleWait);
}

TEST_CASE("batch-level batching and submitted depth preserve the histogram", "[integration][async][depth][batching]")
{
    // The scalar oracle is the depth-1, Q=1 run. Every (Q, depth) pair must reproduce it, including
    // the batch-level classify/plan/scatter path when a node has several batches in flight.
    std::vector<std::uint32_t> referenceWeights;
    std::vector<graphFlow::particle::HistogramRow> referenceHistogram;
    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        for(std::uint32_t const depth : {1u, 2u, 4u})
        {
            CAPTURE(batchChunks, depth);
            graphFlow::particle::FeedbackResult const result = runFeedback(8192u, 1024u, 512u, batchChunks, depth);
            CHECK(result.sunkParticles == 8192u);
            if(referenceHistogram.empty())
            {
                referenceWeights = result.weights;
                referenceHistogram = result.histogram;
            }
            else
            {
                CHECK(result.weights == referenceWeights);
                CHECK(result.histogram == referenceHistogram);
            }
        }
    }
}

#if ALPAKA_LANG_CUDA
TEST_CASE("CUDA depth above one overlaps batches and backpressures", "[integration][async][depth][cuda]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        throw std::runtime_error{"CUDA is compiled in but no NVIDIA GPU is visible; this test requires hardware"};
    runDepthScenario(selector.makeDevice(0u), 4u, 16u, 8u);
}

TEST_CASE("CUDA rejects the CPU-only idle wait", "[integration][async][depth][cuda]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        throw std::runtime_error{"CUDA is compiled in but no NVIDIA GPU is visible; this test requires hardware"};
    auto device = selector.makeDevice(0u);
    graphFlow::ChunkPool<decltype(device)> pool{device, 4u, 8u};
    graphFlow::GraphRunner<decltype(device)> runner{device, pool};
    CHECK_THROWS_AS(runner.setIdleWait(true), std::invalid_argument);
    CHECK_NOTHROW(runner.setIdleWait(false));
}
#endif

#if ALPAKA_LANG_HIP
TEST_CASE("HIP depth above one overlaps batches and backpressures", "[integration][async][depth][hip]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        throw std::runtime_error{"HIP is compiled in but no AMD GPU is visible; this test requires hardware"};
    runDepthScenario(selector.makeDevice(0u), 4u, 16u, 8u);
}

TEST_CASE("HIP rejects the CPU-only idle wait", "[integration][async][depth][hip]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        throw std::runtime_error{"HIP is compiled in but no AMD GPU is visible; this test requires hardware"};
    auto device = selector.makeDevice(0u);
    graphFlow::ChunkPool<decltype(device)> pool{device, 4u, 8u};
    graphFlow::GraphRunner<decltype(device)> runner{device, pool};
    CHECK_THROWS_AS(runner.setIdleWait(true), std::invalid_argument);
    CHECK_NOTHROW(runner.setIdleWait(false));
}
#endif
