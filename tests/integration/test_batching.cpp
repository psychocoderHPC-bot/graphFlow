/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    /** Three quarters of the particles go to bin 0, one quarter to bin 1.
     *
     * A batch of two full chunks therefore has a bin-0 total of 1.5*C, so the scatter range of that
     * bin crosses an output chunk boundary as soon as the batch carries more than one chunk.
     */
    struct QuarterPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return id % 4u == 0u ? 1u : 0u;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            particle.weight = bin;
        }
    };

    /** `id % 8` routing node; the sink checks the id-indexed result. */
    struct Mod8Policy
    {
        static constexpr std::uint32_t maxBins = 8u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return static_cast<std::uint32_t>(id % maxBins);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            particle.weight = bin;
        }
    };

    struct FillLevel
    {
        std::uint32_t bin = 0u;
        std::uint32_t fill = 0u;

        friend bool operator==(FillLevel const&, FillLevel const&) = default;
    };

    /** One deterministically driven transform batch: its exact output plan and the acquired count. */
    struct DrivenBatch
    {
        std::vector<FillLevel> expectedFills;
        std::vector<FillLevel> actualFills;
        std::uint32_t allocated = 0u;
        std::uint64_t maxBinTotal = 0u;
    };

    struct DirectOutcome
    {
        std::vector<std::uint64_t> outputIds;
        std::vector<std::uint32_t> outputWeights;
        std::vector<DrivenBatch> batches;
        std::uint32_t classifyLaunches = 0u;
        std::uint32_t scatterLaunches = 0u;
    };

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }

    /** Plan the exact per-bin output fills of one batch whose input chunks are full. */
    std::vector<FillLevel> expectedFills(std::uint32_t chunks, std::uint32_t capacity)
    {
        // Every full chunk contributes 3*C/4 particles to bin 0 and C/4 to bin 1.
        std::uint64_t const totals[QuarterPolicy::maxBins]
            = {static_cast<std::uint64_t>(chunks) * capacity / 4u * 3u,
               static_cast<std::uint64_t>(chunks) * capacity / 4u};
        std::vector<FillLevel> fills;
        for(std::uint32_t bin = 0u; bin < QuarterPolicy::maxBins; ++bin)
        {
            for(std::uint64_t offset = 0u; offset < totals[bin]; offset += capacity)
            {
                std::uint64_t const remaining = totals[bin] - offset;
                fills.push_back(
                    FillLevel{bin, static_cast<std::uint32_t>(remaining < capacity ? remaining : capacity)});
            }
        }
        return fills;
    }

    /** Drive the transform stage directly with batches of `batchChunks` chunks.
     *
     * The driver keeps every input chunk live and does not publish outputs, so the pool must hold
     * all inputs plus all outputs. Every batch is launched exactly like the scheduler launches it:
     * one classify stage, one scatter stage, then `finish`.
     */
    DirectOutcome runDirect(std::uint32_t capacity, std::uint32_t batchChunks, std::uint64_t numParticles)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        using Runner = graphFlow::GraphRunner<Device>;
        std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);
        std::uint32_t const numChunks = static_cast<std::uint32_t>(numParticles / capacity);
        REQUIRE(static_cast<std::uint64_t>(numChunks) * capacity == numParticles);

        graphFlow::ChunkPool<Device> pool{device, numChunks * 3u + 8u, capacity};
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});

        std::vector<graphFlow::ChunkDescriptor> inputs;
        inputs.reserve(numChunks);
        graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        for(std::uint32_t chunkIndex = 0u; chunkIndex < numChunks; ++chunkIndex)
        {
            graphFlow::ChunkDescriptor const chunk{pool.acquire(), capacity, 0u};
            inputs.push_back(chunk);
            Runner::Batch const generateBatch{
                {chunk},
                static_cast<std::uint64_t>(chunkIndex) * capacity};
            generate.launch(initQueue, generateBatch);
        }
        alpaka::onHost::wait(initQueue);

        graphFlow::particle::TransformStage<Device, QuarterPolicy> stage{
            device,
            pool,
            QuarterPolicy{},
            capped,
            128u,
            batchChunks};
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

        DirectOutcome outcome;
        std::vector<graphFlow::ChunkDescriptor> outputs;
        for(std::size_t first = 0u; first < inputs.size(); first += batchChunks)
        {
            std::size_t const last = std::min(first + batchChunks, inputs.size());
            std::vector<graphFlow::ChunkDescriptor> const chunks(inputs.begin() + first, inputs.begin() + last);
            Runner::Batch const batch{chunks, 0u};

            stage.launch(queue, batch);
            drain(device, queue);

            std::uint32_t allocated = 0u;
            REQUIRE(stage.advance(queue, batch, allocated));
            drain(device, queue);
            REQUIRE_FALSE(stage.advance(queue, batch, allocated));

            std::vector<graphFlow::ChunkDescriptor> batchOutputs;
            stage.finish(batch, batchOutputs);

            DrivenBatch driven;
            driven.expectedFills = expectedFills(static_cast<std::uint32_t>(chunks.size()), capacity);
            driven.allocated = allocated;
            for(graphFlow::ChunkDescriptor const& descriptor : batchOutputs)
            {
                driven.actualFills.push_back(FillLevel{descriptor.bin, descriptor.fill});
                outputs.push_back(descriptor);
            }
            std::uint64_t const bin0 = static_cast<std::uint64_t>(chunks.size()) * capacity / 4u * 3u;
            driven.maxBinTotal = std::max(bin0, static_cast<std::uint64_t>(chunks.size()) * capacity / 4u);
            outcome.batches.push_back(std::move(driven));
        }
        outcome.classifyLaunches = stage.classifyLaunches();
        outcome.scatterLaunches = stage.scatterLaunches();

        std::vector<std::uint64_t> ids(static_cast<std::size_t>(pool.numChunks()) * capacity, 0u);
        std::vector<std::uint32_t> weights(static_cast<std::size_t>(pool.numChunks()) * capacity, 0u);
        {
            auto readQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(readQueue, ids, pool.ids());
            alpaka::onHost::memcpy(readQueue, weights, pool.weights());
        }
        for(graphFlow::ChunkDescriptor const& descriptor : outputs)
        {
            std::size_t const base = static_cast<std::size_t>(descriptor.handle.index) * capacity;
            for(std::uint32_t i = 0u; i < descriptor.fill; ++i)
            {
                // Every slot of an output chunk must carry that chunk's bin and the classifier
                // result of its id; this catches a wrong-table scatter as well as a lost particle.
                REQUIRE(weights[base + i] == descriptor.bin);
                REQUIRE(descriptor.bin == (ids[base + i] % 4u == 0u ? 1u : 0u));
                outcome.outputIds.push_back(ids[base + i]);
                outcome.outputWeights.push_back(weights[base + i]);
            }
        }
        return outcome;
    }

    /** Integration harness for the scheduler-driven batching path. */
    struct PipelineOutcome
    {
        std::vector<std::uint32_t> weights;
        std::vector<std::uint32_t> sinkFills;
        std::uint32_t joinBatches = 0u;
        std::uint32_t joinClassifyLaunches = 0u;
        std::uint32_t joinScatterLaunches = 0u;
        std::uint32_t joinMaxBatchChunks = 0u;
        std::uint32_t splitBatches = 0u;
        std::uint32_t splitClassifyLaunches = 0u;
        std::uint32_t splitScatterLaunches = 0u;
    };

    PipelineOutcome runPipeline(std::uint32_t capacity, std::uint32_t batchChunks, std::uint64_t numParticles)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        using Runner = graphFlow::GraphRunner<Device>;
        std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);

        // One splitter chunk becomes at most eight one-bin chunks, and every join output needs a
        // chunk too; the pool is sized generously because the test keeps the whole graph live.
        std::uint32_t const numChunks = static_cast<std::uint32_t>(numParticles / capacity) * 4u + 32u;
        graphFlow::ChunkPool<Device> pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{numParticles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
        graphFlow::particle::TransformStage<Device, Mod8Policy> split{
            device,
            pool,
            Mod8Policy{},
            capped,
            128u,
            batchChunks};
        graphFlow::particle::TransformStage<Device, Mod8Policy> join{
            device,
            pool,
            Mod8Policy{},
            capped,
            128u,
            batchChunks};
        graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

        PipelineOutcome outcome;
        auto sinkHooks = graphFlow::particle::hooksOf<Device>(sink);
        auto const sinkLaunch = sinkHooks.launch;
        sinkHooks.launch = [sinkLaunch, &outcome](Runner::Queue const& queue, Runner::Batch const& batch)
        {
            for(graphFlow::ChunkDescriptor const& chunk : batch.chunks)
                outcome.sinkFills.push_back(chunk.fill);
            sinkLaunch(queue, batch);
        };

        std::vector<int> const splitRoute(Mod8Policy::maxBins, 2);
        std::vector<int> const joinRoute(Mod8Policy::maxBins, 3);
        Runner runner{device, pool};
        auto const source = runner.addNode(
            "generate",
            graphFlow::NodeRole::source,
            1u,
            {1},
            graphFlow::particle::hooksOf<Device>(generate));
        auto const splitNode = runner.addNode(
            "split",
            graphFlow::NodeRole::transform,
            Mod8Policy::maxBins,
            splitRoute,
            graphFlow::particle::hooksOf<Device>(split));
        auto const joinNode = runner.addNode(
            "join",
            graphFlow::NodeRole::transform,
            Mod8Policy::maxBins,
            joinRoute,
            graphFlow::particle::hooksOf<Device>(join));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, std::move(sinkHooks));
        runner.setSource(source, numParticles);
        runner.setBatchChunks(batchChunks);
        runner.setStallTimeout(std::chrono::milliseconds{20'000});
        runner.run();

        outcome.joinBatches = runner.nodeCompleted(joinNode);
        outcome.joinClassifyLaunches = join.classifyLaunches();
        outcome.joinScatterLaunches = join.scatterLaunches();
        outcome.joinMaxBatchChunks = join.maxBatchChunks();
        outcome.splitBatches = runner.nodeCompleted(splitNode);
        outcome.splitClassifyLaunches = split.classifyLaunches();
        outcome.splitScatterLaunches = split.scatterLaunches();

        outcome.weights.assign(static_cast<std::size_t>(numParticles), 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, outcome.weights, result);
        }
        return outcome;
    }
} // namespace

TEST_CASE("batched transform plans exact batch-wide output fills", "[integration][batching]")
{
    // Ten full chunks per case. Q=2 gives a bin-0 total of 1.5*C per batch, so the scatter range
    // crosses a chunk boundary; Q=8 crosses six of them.
    std::vector<graphFlow::particle::HistogramRow> reference;
    for(std::uint32_t const capacity : {1024u, 8192u})
    {
        reference.clear();
        for(std::uint32_t const batchChunks : {1u, 2u, 8u})
        {
            CAPTURE(capacity, batchChunks);
            std::uint64_t const particles = 10ull * capacity;
            DirectOutcome const outcome = runDirect(capacity, batchChunks, particles);

            REQUIRE(outcome.batches.size() == (10u + batchChunks - 1u) / batchChunks);
            REQUIRE(outcome.classifyLaunches == outcome.batches.size());
            REQUIRE(outcome.scatterLaunches == outcome.batches.size());

            bool crossed = false;
            for(DrivenBatch const& batch : outcome.batches)
            {
                CHECK(batch.actualFills == batch.expectedFills);
                CHECK(batch.allocated == batch.expectedFills.size());
                crossed = crossed || batch.maxBinTotal > capacity;
            }
            // A single-chunk batch cannot cross an output chunk boundary; batched ones must.
            CHECK(crossed == (batchChunks > 1u));

            // The id-indexed output must contain every particle exactly once.
            std::vector<std::uint64_t> ids = outcome.outputIds;
            std::sort(ids.begin(), ids.end());
            std::vector<std::uint64_t> const expectedIds = [&]
            {
                std::vector<std::uint64_t> all(static_cast<std::size_t>(particles));
                std::iota(all.begin(), all.end(), std::uint64_t{0u});
                return all;
            }();
            CHECK(ids == expectedIds);

            for(std::size_t i = 0u; i < outcome.outputIds.size(); ++i)
                CHECK(outcome.outputWeights[i] == (outcome.outputIds[i] % 4u == 0u ? 1u : 0u));

            std::vector<graphFlow::particle::HistogramRow> const histogram
                = graphFlow::particle::histogramOf(outcome.outputWeights);
            std::vector<graphFlow::particle::HistogramRow> const expected{
                {0u, particles / 4u * 3u},
                {1u, particles / 4u}};
            CHECK(histogram == expected);
            if(reference.empty())
                reference = histogram;
            else
                CHECK(histogram == reference);
        }
    }
}

TEST_CASE("scheduler batches ready chunks with one classify and one scatter launch", "[integration][batching]")
{
    // The splitter turns one input chunk into eight one-bin chunks, so the first join dispatch has
    // eight ready chunks. The runner must hand the join node exactly `min(Q, 8)` of them, and every
    // join batch must issue exactly one classify and one scatter launch.
    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        CAPTURE(batchChunks);
        std::uint32_t const capacity = 1024u;
        std::uint64_t const particles = 3ull * capacity;
        PipelineOutcome const outcome = runPipeline(capacity, batchChunks, particles);

        CHECK(outcome.joinBatches != 0u);
        CHECK(outcome.joinMaxBatchChunks == std::min(batchChunks, 8u));
        CHECK(outcome.joinClassifyLaunches == outcome.joinBatches);
        CHECK(outcome.joinScatterLaunches == outcome.joinBatches);
        CHECK(outcome.splitClassifyLaunches == outcome.splitBatches);
        CHECK(outcome.splitScatterLaunches == outcome.splitBatches);

        // Conservation: every particle is delivered exactly once and carries its id % 8 weight.
        std::uint64_t fillSum = 0u;
        for(std::uint32_t const fill : outcome.sinkFills)
        {
            CHECK(fill >= 1u);
            CHECK(fill <= capacity);
            fillSum += fill;
        }
        CHECK(fillSum == particles);

        std::vector<std::uint32_t> host = outcome.weights;
        for(std::size_t id = 0u; id < host.size(); ++id)
            REQUIRE(host[id] == static_cast<std::uint32_t>(id % Mod8Policy::maxBins));
        std::sort(host.begin(), host.end());
        CHECK(host == [&]
        {
            std::vector<std::uint32_t> expected(static_cast<std::size_t>(particles));
            for(std::size_t id = 0u; id < expected.size(); ++id)
                expected[id] = static_cast<std::uint32_t>(id % Mod8Policy::maxBins);
            std::sort(expected.begin(), expected.end());
            return expected;
        }());
    }
}
