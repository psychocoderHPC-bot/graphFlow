/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using graphFlow::outputChunkBound;

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
        std::uint32_t maxBatchChunks = 0u;
    };

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }

    using IdRange = std::pair<std::uint64_t, std::uint32_t>; // (firstId, fill)

    /** Per-bin particle totals of one batch, from the id ranges its chunks cover. */
    std::array<std::uint64_t, QuarterPolicy::maxBins> binTotals(
        std::vector<IdRange> const& ranges)
    {
        std::array<std::uint64_t, QuarterPolicy::maxBins> totals{};
        for(IdRange const& range : ranges)
        {
            for(std::uint64_t id = range.first; id < range.first + range.second; ++id)
                ++totals[id % 4u == 0u ? 1u : 0u];
        }
        return totals;
    }

    /** Plan the exact per-bin output fills of one batch. */
    std::vector<FillLevel> expectedFills(
        std::vector<IdRange> const& ranges,
        std::uint32_t capacity)
    {
        std::array<std::uint64_t, QuarterPolicy::maxBins> const totals = binTotals(ranges);
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

    /** Read every output chunk and check the routing invariants against its bin. */
    template<typename TDevice>
    void collectOutputs(
        TDevice& device,
        graphFlow::ChunkPool<TDevice>& pool,
        std::uint32_t capacity,
        std::vector<graphFlow::ChunkDescriptor> const& outputs,
        DirectOutcome& outcome)
    {
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
    }

    /** Drive the transform stage directly with batches of `batchChunks` chunks.
     *
     * The driver keeps every input chunk live and does not publish outputs, so the pool must hold
     * all inputs plus all outputs. Every batch is launched exactly like the scheduler launches it:
     * one classify stage, one scatter stage, then `finish`. Explicit `fills` allow partial chunks
     * and chunk sizes that are not a multiple of the tile width.
     */
    DirectOutcome runDirect(
        std::uint32_t capacity,
        std::uint32_t batchChunks,
        std::vector<std::uint32_t> const& fills)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        using Runner = graphFlow::GraphRunner<Device>;
        std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);
        std::uint32_t const numChunks = static_cast<std::uint32_t>(fills.size());
        REQUIRE(numChunks != 0u);
        REQUIRE(batchChunks != 0u);

        // Inputs stay live and every batch's outputs stay live too, so this test-only pool holds
        // both. It is not the production pool bound; the direct driver never releases outputs.
        std::uint64_t const batchCount = (numChunks + batchChunks - 1u) / batchChunks;
        std::uint64_t const maxOutputsPerBatch = outputChunkBound(
            static_cast<std::uint64_t>(batchChunks) * capacity,
            capacity,
            QuarterPolicy::maxBins);
        std::uint64_t const poolChunks = numChunks + batchCount * maxOutputsPerBatch + 8u;
        REQUIRE(poolChunks <= std::numeric_limits<std::uint32_t>::max());
        graphFlow::ChunkPool<Device> pool{device, static_cast<std::uint32_t>(poolChunks), capacity};
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});

        std::vector<graphFlow::ChunkDescriptor> inputs;
        std::vector<IdRange> ranges;
        inputs.reserve(numChunks);
        ranges.reserve(numChunks);
        graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
        // GenerateStage::Queue is the node queue type, i.e. the runner's non-blocking queue.
        auto initQueue = device.makeQueue(alpaka::queueKind::nonBlocking);
        std::uint64_t firstId = 0u;
        for(std::uint32_t const fill : fills)
        {
            graphFlow::ChunkDescriptor const chunk{pool.acquire(), fill, 0u};
            inputs.push_back(chunk);
            ranges.emplace_back(firstId, fill);
            Runner::Batch const generateBatch{{chunk}, firstId};
            generate.launch(initQueue, generateBatch);
            firstId += fill;
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
            std::vector<IdRange> const batchRanges(ranges.begin() + first, ranges.begin() + last);
            Runner::Batch const batch{chunks, 0u};

            stage.launch(queue, batch);
            drain(device, queue);

            std::uint32_t allocated = 0u;
            REQUIRE(stage.advance(queue, batch, allocated));
            // `advance` zeroes its out parameter on entry, so keep the plan/scatter count before
            // the second call finalises the batch.
            std::uint32_t const plannedChunks = allocated;
            drain(device, queue);
            REQUIRE_FALSE(stage.advance(queue, batch, allocated));

            std::vector<graphFlow::ChunkDescriptor> batchOutputs;
            stage.finish(batch, batchOutputs);

            DrivenBatch driven;
            driven.expectedFills = expectedFills(batchRanges, capacity);
            driven.allocated = plannedChunks;
            for(graphFlow::ChunkDescriptor const& descriptor : batchOutputs)
            {
                driven.actualFills.push_back(FillLevel{descriptor.bin, descriptor.fill});
                outputs.push_back(descriptor);
            }
            std::array<std::uint64_t, QuarterPolicy::maxBins> const totals = binTotals(batchRanges);
            driven.maxBinTotal = std::max(totals[0], totals[1]);
            outcome.batches.push_back(std::move(driven));
        }
        outcome.classifyLaunches = stage.classifyLaunches();
        outcome.scatterLaunches = stage.scatterLaunches();
        outcome.maxBatchChunks = stage.maxBatchChunks();
        collectOutputs(device, pool, capacity, outputs, outcome);
        return outcome;
    }

    /** Expected sorted histogram of `particles` ids classified by `QuarterPolicy`. */
    std::vector<graphFlow::particle::HistogramRow> expectedHistogram(std::uint64_t particles)
    {
        std::uint64_t const bin1 = (particles + 3u) / 4u;
        return {{0u, particles - bin1}, {1u, bin1}};
    }

    /** Check id conservation, per-id weights and the sorted histogram of a direct run. */
    std::vector<graphFlow::particle::HistogramRow> checkDirectOutcome(
        DirectOutcome const& outcome,
        std::uint64_t particles)
    {
        std::vector<std::uint64_t> ids = outcome.outputIds;
        std::sort(ids.begin(), ids.end());
        std::vector<std::uint64_t> expectedIds(static_cast<std::size_t>(particles));
        std::iota(expectedIds.begin(), expectedIds.end(), std::uint64_t{0u});
        CHECK(ids == expectedIds);
        for(std::size_t i = 0u; i < outcome.outputIds.size(); ++i)
            CHECK(outcome.outputWeights[i] == (outcome.outputIds[i] % 4u == 0u ? 1u : 0u));
        std::vector<graphFlow::particle::HistogramRow> const histogram
            = graphFlow::particle::histogramOf(outcome.outputWeights);
        CHECK(histogram == expectedHistogram(particles));
        return histogram;
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
        std::uint32_t poolChunks = 0u;
    };

    PipelineOutcome runPipeline(
        std::uint32_t capacity,
        std::uint32_t batchChunks,
        std::uint64_t numParticles,
        std::uint32_t liveParticles = 0u,
        std::uint32_t poolChunks = 0u)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        using Runner = graphFlow::GraphRunner<Device>;
        std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);

        // Default: one splitter chunk becomes at most eight one-bin chunks, and every join output
        // needs a chunk too; the pool is sized generously because the test keeps the whole graph
        // live. A test may pass an exact pool instead.
        std::uint32_t const numChunks
            = poolChunks != 0u ? poolChunks : static_cast<std::uint32_t>(numParticles / capacity) * 4u + 32u;
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
        runner.setSource(source, numParticles, liveParticles);
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
        outcome.poolChunks = pool.numChunks();

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
            std::vector<std::uint32_t> const fills(10u, capacity);
            DirectOutcome const outcome = runDirect(capacity, batchChunks, fills);

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

            std::vector<graphFlow::particle::HistogramRow> const histogram
                = checkDirectOutcome(outcome, particles);
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

TEST_CASE("batched transform handles partial chunks and non-aligned tile tails", "[integration][batching]")
{
    // C=1000 with T=128 leaves a 104-particle tail tile in the full chunks. The other chunks are
    // partial, including a 5-particle tail, and all of them share batches.
    std::uint32_t const capacity = 1000u;
    std::vector<std::uint32_t> const fills{1000u, 37u, 613u, 1000u, 5u};
    std::uint64_t const particles = 2655u;

    std::vector<graphFlow::particle::HistogramRow> reference;
    for(std::uint32_t const batchChunks : {2u, 3u, 5u})
    {
        CAPTURE(batchChunks);
        DirectOutcome const outcome = runDirect(capacity, batchChunks, fills);

        REQUIRE(outcome.batches.size() == (fills.size() + batchChunks - 1u) / batchChunks);
        REQUIRE(outcome.classifyLaunches == outcome.batches.size());
        REQUIRE(outcome.scatterLaunches == outcome.batches.size());
        CHECK(
            outcome.maxBatchChunks
            == std::min<std::uint32_t>(batchChunks, static_cast<std::uint32_t>(fills.size())));

        bool crossed = false;
        for(DrivenBatch const& batch : outcome.batches)
        {
            CHECK(batch.actualFills == batch.expectedFills);
            CHECK(batch.allocated == batch.expectedFills.size());
            crossed = crossed || batch.maxBinTotal > capacity;
        }
        CHECK(crossed);

        std::vector<graphFlow::particle::HistogramRow> const histogram
            = checkDirectOutcome(outcome, particles);
        if(reference.empty())
            reference = histogram;
        else
            CHECK(histogram == reference);
    }
}

TEST_CASE("bounded P dispatches a Q=8 batch inside S=P+K*R(C)", "[integration][batching][pressure]")
{
    // One source chunk of C particles is split into eight one-bin chunks, so the join node has
    // eight ready chunks and dispatches them as one batch. The pool is exactly S = P + K*R(C) with
    // K=2 transform nodes and R(C) = outputChunkBound(C, C, 8); no Q-scaled reserve is needed.
    constexpr std::uint32_t capacity = 1024u;
    constexpr std::uint32_t batchChunks = 8u;
    constexpr std::uint64_t particles = 1024u;
    constexpr std::uint32_t liveParticles = 1024u;
    std::uint32_t const perTransform = static_cast<std::uint32_t>(
        outputChunkBound(capacity, capacity, Mod8Policy::maxBins));
    std::uint32_t const poolChunks = liveParticles + 2u * perTransform;

    PipelineOutcome const outcome
        = runPipeline(capacity, batchChunks, particles, liveParticles, poolChunks);

    CHECK(outcome.poolChunks == poolChunks);
    CHECK(outcome.joinBatches != 0u);
    CHECK(outcome.joinMaxBatchChunks == batchChunks);
    CHECK(outcome.joinClassifyLaunches == outcome.joinBatches);
    CHECK(outcome.joinScatterLaunches == outcome.joinBatches);

    std::uint64_t fillSum = 0u;
    for(std::uint32_t const fill : outcome.sinkFills)
        fillSum += fill;
    CHECK(fillSum == particles);
}

TEST_CASE("credit pressure shrinks a Q=8 batch to the largest fitting prefix", "[integration][batching][pressure]")
{
    // C=8 and eight particles: the splitter turns the single source chunk into eight 1-particle
    // chunks, so the join sees its full Q=8 ready list. A batch of k chunks reserves k output
    // chunks (B=8, one particle per bin), but the deliberately tight test pool leaves four
    // credits. The runner must dispatch the largest fitting prefix and then the rest.
    constexpr std::uint32_t capacity = 8u;
    constexpr std::uint32_t batchChunks = 8u;
    constexpr std::uint64_t particles = 8u;
    constexpr std::uint32_t liveParticles = 8u;
    constexpr std::uint32_t tightPool = 12u; // below S = P + K*R(C); test-only credit pressure

    PipelineOutcome const outcome
        = runPipeline(capacity, batchChunks, particles, liveParticles, tightPool);

    CHECK(outcome.poolChunks == tightPool);
    CHECK(outcome.joinMaxBatchChunks == 4u);
    CHECK(outcome.joinMaxBatchChunks < batchChunks);
    CHECK(outcome.joinBatches == 2u);
    CHECK(outcome.joinClassifyLaunches == outcome.joinBatches);
    CHECK(outcome.joinScatterLaunches == outcome.joinBatches);

    std::uint64_t fillSum = 0u;
    for(std::uint32_t const fill : outcome.sinkFills)
        fillSum += fill;
    CHECK(fillSum == particles);
}

TEST_CASE("single-chunk pool bound completes N much larger than P with Q=8", "[integration][batching][pressure]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.liveParticles = 256u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;
    parameters.batchChunks = 8u;
    parameters.stallTimeoutMs = 30'000u;

    graphFlow::particle::FeedbackResult const result
        = graphFlow::particle::runParticleFeedback(device, parameters);
    graphFlow::test::ScalarOutcome const reference
        = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);
    CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
    CHECK(result.weights == reference.weights);
    CHECK(result.sunkParticles == parameters.numParticles);
    CHECK(result.liveParticles == 256u);

    // Q does not enlarge the pool: S = P + K*R(C) = 256 + 3*2 = 262. Under fragmentation the
    // scheduler shrinks a batch instead of reserving a Q-scaled bound.
    std::uint32_t const expectedPool = graphFlow::particle::derivePoolSize(
        parameters.numParticles,
        parameters.chunkSize,
        256u,
        graphFlow::particle::FlowBins);
    CHECK(expectedPool == 256u + 3u * 2u);
    CHECK(result.stats.poolChunks == expectedPool);
    CHECK(result.stats.peakLiveChunks <= expectedPool);
}
