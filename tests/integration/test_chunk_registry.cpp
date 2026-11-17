/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/ChunkRegistry.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
    template<typename TRunner>
    typename TRunner::Hooks sourceHooks()
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        hooks.finish = [](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        { outputs.push_back(batch.chunk()); };
        hooks.abort = [] {};
        return hooks;
    }

    /** Sink that validates the registry from inside the scheduler loop. */
    template<typename TRunner>
    typename TRunner::Hooks validatingSinkHooks(TRunner& runner, std::uint64_t& sunk)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [&runner](auto const&, typename TRunner::Batch const& batch)
        {
            // The chunk is in the sink batch; dispatch has not moved it to inflight yet.
            graphFlow::ChunkPhase const phase = runner.chunkRegistry().at(batch.chunk().handle.index).phase;
            bool const inPipeline = phase == graphFlow::ChunkPhase::ready || phase == graphFlow::ChunkPhase::inflight;
            CHECK(inPipeline);
            runner.validateChunkRegistry();
        };
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        hooks.finish = [&sunk](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>&)
        { sunk += batch.particles(); };
        hooks.abort = [] {};
        return hooks;
    }

    /** Transform that conserves particles into one output chunk per input batch. */
    template<typename TRunner, typename TPool>
    typename TRunner::Hooks conservingTransformHooks(TPool& pool)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        hooks.finish = [&pool](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        {
            graphFlow::ChunkHandle const handle = pool.acquire();
            outputs.push_back(graphFlow::ChunkDescriptor{handle, static_cast<std::uint32_t>(batch.particles()), 0u});
        };
        hooks.abort = [] {};
        return hooks;
    }

    /** Transform that fragments every input batch into one output chunk per particle.
     *
     * With `maxBins == particles()` the output-credit reservation covers the fan-out exactly.
     */
    template<typename TRunner, typename TPool>
    typename TRunner::Hooks fragmentingTransformHooks(TPool& pool, std::vector<graphFlow::ChunkDescriptor>& pending)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [&pool, &pending](auto const&, typename TRunner::Batch const& batch, std::uint32_t& allocated)
        {
            pending.clear();
            auto const outputs = static_cast<std::uint32_t>(batch.particles());
            for(std::uint32_t bin = 0u; bin < outputs; ++bin)
                pending.push_back(graphFlow::ChunkDescriptor{pool.acquire(), 1u, bin});
            allocated = outputs;
            return false;
        };
        hooks.finish = [&pending](typename TRunner::Batch const&, std::vector<graphFlow::ChunkDescriptor>& outputs)
        {
            outputs.insert(outputs.end(), pending.begin(), pending.end());
            pending.clear();
        };
        hooks.abort = [] {};
        return hooks;
    }
} // namespace

TEST_CASE("chunk registry tracks a conserving pipeline", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Runner = graphFlow::GraphRunner<decltype(device)>;
    graphFlow::ChunkPool<decltype(device)> pool{device, 16u, 8u};
    Runner runner{device, pool};

    std::uint64_t sunk = 0u;
    auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
    runner.addNode("transform", graphFlow::NodeRole::transform, 1u, {2}, conservingTransformHooks<Runner>(pool));
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, validatingSinkHooks<Runner>(runner, sunk));
    runner.setSource(source, 16u, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    graphFlow::SchedulerStats const stats = runner.run();
    runner.validateChunkRegistry();

    CHECK(stats.sourceGenerated == 16u);
    CHECK(stats.sourceSunk == 16u);
    CHECK(sunk == 16u);
    CHECK(pool.freeCount() == pool.numChunks());

    graphFlow::ChunkRegistrySummary const summary = runner.chunkRegistrySummary();
    CHECK(summary.chunks == 16u);
    CHECK(summary.freeChunks == 16u);
    CHECK(summary.readyChunks == 0u);
    CHECK(summary.inflightChunks == 0u);
    CHECK(summary.reservedChunks == 0u);
    CHECK(summary.retiredChunks == 0u);
    CHECK(summary.acquisitions == 4u); // two source chunks and two transform outputs
    CHECK(summary.releases == 4u);
    CHECK(summary.routings == 4u);
    CHECK(summary.retirements == 2u);
    CHECK(summary.peakInflightChunks >= 1u);
    CHECK(stats.chunkRegistryAcquisitions == summary.acquisitions);
    CHECK(stats.chunkRegistryRoutings == summary.routings);
    CHECK(stats.chunkRegistryRetirements == summary.retirements);
    CHECK(stats.peakRegistryInflightChunks == summary.peakInflightChunks);

    for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
    {
        graphFlow::ChunkRecord const& record = runner.chunkRegistry()[index];
        CHECK(record.handle.index == index);
        CHECK(record.handle.generation == pool.generation(index));
        CHECK(record.phase == graphFlow::ChunkPhase::free);
        CHECK(record.ownerNode == graphFlow::invalidChunkNode);
        CHECK(record.destinationNode == -1);
    }
}

TEST_CASE("chunk registry tracks a fragmented multiplexing run", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Runner = graphFlow::GraphRunner<decltype(device)>;
    graphFlow::ChunkPool<decltype(device)> pool{device, 32u, 8u};
    Runner runner{device, pool};

    std::vector<graphFlow::ChunkDescriptor> pending;
    std::uint64_t sunk = 0u;
    auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
    runner.addNode(
        "fragment",
        graphFlow::NodeRole::transform,
        8u,
        {2, 2, 2, 2, 2, 2, 2, 2},
        fragmentingTransformHooks<Runner>(pool, pending));
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, validatingSinkHooks<Runner>(runner, sunk));
    runner.setSource(source, 16u, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    graphFlow::SchedulerStats const stats = runner.run();
    runner.validateChunkRegistry();

    CHECK(stats.sourceGenerated == 16u);
    CHECK(stats.sourceSunk == 16u);
    CHECK(sunk == 16u);

    graphFlow::ChunkRegistrySummary const summary = runner.chunkRegistrySummary();
    CHECK(summary.chunks == 32u);
    CHECK(summary.freeChunks == 32u);
    CHECK(summary.acquisitions == 18u); // two source chunks and sixteen one-particle outputs
    CHECK(summary.releases == 18u);
    CHECK(summary.routings == 18u);
    CHECK(summary.retirements == 16u);
    // The stream really fragmented: more chunks were live than the two source chunks.
    CHECK(summary.peakReservedChunks == 8u);
    CHECK(summary.peakReadyChunks >= 8u);
    CHECK(summary.peakInflightChunks >= 2u);
    // Every chunk incarnation was tracked and retired, not just the source chunks.
    CHECK(stats.chunkRegistryRoutings == 18u);
    CHECK(stats.chunkRegistryRetirements == 16u);

    for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
    {
        graphFlow::ChunkRecord const& record = runner.chunkRegistry()[index];
        CHECK(record.handle.generation == pool.generation(index));
        CHECK(record.phase == graphFlow::ChunkPhase::free);
    }
}

TEST_CASE("chunk registry resets for a second run on the same runner", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Runner = graphFlow::GraphRunner<decltype(device)>;
    graphFlow::ChunkPool<decltype(device)> pool{device, 16u, 8u};
    Runner runner{device, pool};

    std::uint64_t sunk = 0u;
    auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
    runner.addNode("transform", graphFlow::NodeRole::transform, 1u, {2}, conservingTransformHooks<Runner>(pool));
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, validatingSinkHooks<Runner>(runner, sunk));
    runner.setSource(source, 16u, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    graphFlow::SchedulerStats const first = runner.run();
    graphFlow::ChunkRegistrySummary const firstRegistry = runner.chunkRegistrySummary();
    graphFlow::SchedulerStats const second = runner.run();
    graphFlow::ChunkRegistrySummary const secondRegistry = runner.chunkRegistrySummary();

    CHECK(first.sourceSunk == 16u);
    CHECK(second.sourceSunk == 16u);
    CHECK(firstRegistry.acquisitions == secondRegistry.acquisitions);
    CHECK(firstRegistry.releases == secondRegistry.releases);
    CHECK(secondRegistry.freeChunks == secondRegistry.chunks);
    runner.validateChunkRegistry();
}
