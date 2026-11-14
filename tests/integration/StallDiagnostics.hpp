/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graphFlow::test::diagnostics
{
    /** Hooks that complete without touching device memory.
     *
     * The runner owns queue/event polling; these hooks only exercise the host scheduler. A source
     * publishes its input chunk unchanged, a sink retires it, and a transform never reports
     * progress. That is enough to drive the aggregate counters on every backend.
     */
    template<typename TRunner>
    typename TRunner::Hooks sourceHooks()
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t&) -> bool { return false; };
        hooks.finish = [](typename TRunner::Batch const& batch, std::vector<ChunkDescriptor>& outputs)
        { outputs.push_back(batch.chunk()); };
        hooks.abort = [] {};
        return hooks;
    }

    template<typename TRunner>
    typename TRunner::Hooks passiveHooks()
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t&) -> bool { return false; };
        hooks.finish = [](typename TRunner::Batch const&, std::vector<ChunkDescriptor>&) {};
        hooks.abort = [] {};
        return hooks;
    }

    struct StallOutcome
    {
        bool threw = false;
        graphFlow::SchedulerSnapshot snapshot;
        std::string message;
        std::string detail;
        std::string callbackDetail;
        std::uint32_t numChunks = 0u;
        std::uint32_t freeChunks = 0u;
    };

    /** Force a no-progress stall: one source chunk reaches a transform that can never reserve. */
    template<typename TDevice>
    StallOutcome runForcedStall(TDevice& device)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;
        constexpr std::uint32_t capacity = 8u;
        constexpr std::uint32_t numChunks = 4u;
        constexpr std::uint64_t particles = 8u;
        graphFlow::ChunkPool<TDevice> storage{device, numChunks, capacity};

        Runner runner{device, storage};
        auto const source
            = runner.addNode("secret-source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
        // A batch of 8 particles over eight bins can produce eight output chunks, but only three
        // chunks are free. The scheduler must keep the chunk queued and eventually report a stall.
        runner.addNode(
            "secret-transform",
            graphFlow::NodeRole::transform,
            8u,
            std::vector<int>(8u, 2),
            passiveHooks<Runner>());
        runner.addNode("secret-sink", graphFlow::NodeRole::sink, 0u, {}, passiveHooks<Runner>());
        runner.setSource(source, particles);
        runner.setBatchChunks(1u);
        runner.setStallTimeout(std::chrono::milliseconds{100});
        runner.setDiagnostics(true);

        StallOutcome outcome;
        runner.setStallDetailCallback([&outcome](std::string const& detail) { outcome.callbackDetail = detail; });
        try
        {
            runner.run();
        }
        catch(graphFlow::SchedulerStallError const& error)
        {
            outcome.threw = true;
            outcome.snapshot = error.snapshot();
            outcome.message = error.what();
            outcome.detail = error.detail();
        }
        outcome.numChunks = storage.numChunks();
        outcome.freeChunks = storage.freeCount();
        return outcome;
    }

    inline void checkStallOutcome(StallOutcome const& outcome)
    {
        REQUIRE(outcome.threw);
        CHECK(outcome.freeChunks == outcome.numChunks);

        auto const& snapshot = outcome.snapshot;
        CHECK(snapshot.sourceParticles == 8u);
        CHECK(snapshot.sourceGenerated == 8u);
        CHECK(snapshot.sourceSunk == 0u);
        CHECK(snapshot.sourceLiveParticles == 8u);
        CHECK(snapshot.chunkCapacity == 8u);
        CHECK(snapshot.poolChunks == 4u);
        CHECK(snapshot.freeChunks == 3u);
        CHECK(snapshot.readyChunks == 1u);
        CHECK(snapshot.readyNodes == 1u);
        CHECK(snapshot.readyDepth == 1u);
        CHECK(snapshot.inflightBatches == 0u);
        CHECK(snapshot.inflightChunks == 0u);
        CHECK(snapshot.reservedChunks == 0u);
        CHECK(snapshot.eventSlotsInUse == 0u);
        CHECK(snapshot.peakReadyChunks == 1u);
        CHECK(snapshot.peakReadyNodes == 1u);
        CHECK(snapshot.maxReadyDepth == 1u);
        CHECK(snapshot.peakInflightBatches >= 1u);
        CHECK(snapshot.peakInflightChunks >= 1u);
        CHECK(snapshot.peakEventSlotsInUse >= 1u);
        CHECK(snapshot.peakReservedChunks == 0u);
        CHECK(snapshot.peakLiveChunks == 1u);
        CHECK(snapshot.completedBatches >= 1u);
        CHECK(snapshot.stallTimeoutMs == 100u);

        // The message is aggregate key=value only; the node names stay in detail/callback.
        CHECK(outcome.message.find("sourceGenerated=8") != std::string::npos);
        CHECK(outcome.message.find("sourceLiveParticles=8") != std::string::npos);
        CHECK(outcome.message.find("chunkCapacity=8") != std::string::npos);
        CHECK(outcome.message.find("readyChunks=1") != std::string::npos);
        CHECK(outcome.message.find("peakReadyChunks=1") != std::string::npos);
        CHECK(outcome.message.find("peakInflightBatches=") != std::string::npos);
        CHECK(outcome.message.find("peakEventSlotsInUse=") != std::string::npos);
        CHECK(outcome.message.find("secret-source") == std::string::npos);
        CHECK(outcome.message.find("secret-transform") == std::string::npos);
        CHECK(outcome.message.find("secret-sink") == std::string::npos);
        CHECK(outcome.message.find("node '") == std::string::npos);
        CHECK(outcome.detail.find("secret-source") != std::string::npos);
        CHECK(outcome.callbackDetail == outcome.detail);
    }

    struct CompletedOutcome
    {
        graphFlow::SchedulerStats stats;
        graphFlow::SchedulerSnapshot snapshot;
        std::uint32_t numChunks = 0u;
        std::uint32_t freeChunks = 0u;
        std::uint32_t nodeCount = 0u;
    };

    /** Run a source -> sink graph and capture the completed-run aggregate stats. */
    template<typename TDevice>
    CompletedOutcome runCompleted(TDevice& device)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;
        constexpr std::uint32_t capacity = 8u;
        constexpr std::uint32_t numChunks = 4u;
        constexpr std::uint64_t particles = 8u;
        graphFlow::ChunkPool<TDevice> storage{device, numChunks, capacity};

        Runner runner{device, storage};
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
        // A transform acquires a fresh output chunk during host planning, so the input and the
        // output are live at the same time and the reservation peak is exercised.
        typename Runner::Hooks transformHooks;
        transformHooks.launch = [](auto const&, typename Runner::Batch const&) {};
        transformHooks.advance
            = [](auto const&, typename Runner::Batch const&, std::uint32_t&) -> bool { return false; };
        transformHooks.finish = [&storage](typename Runner::Batch const& batch, std::vector<ChunkDescriptor>& outputs)
        { outputs.push_back(ChunkDescriptor{storage.acquire(), batch.chunk().fill, 0u}); };
        transformHooks.abort = [] {};
        runner.addNode("transform", graphFlow::NodeRole::transform, 1u, {2}, std::move(transformHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, passiveHooks<Runner>());
        runner.setSource(source, particles);
        runner.setBatchChunks(1u);
        runner.setStallTimeout(std::chrono::milliseconds{5000});

        CompletedOutcome outcome;
        outcome.stats = runner.run();
        outcome.snapshot = runner.snapshot();
        outcome.numChunks = storage.numChunks();
        outcome.freeChunks = storage.freeCount();
        outcome.nodeCount = runner.nodeCount();
        return outcome;
    }

    inline void checkCompletedOutcome(CompletedOutcome const& outcome)
    {
        auto const& stats = outcome.stats;
        auto const& snapshot = outcome.snapshot;
        CHECK(outcome.freeChunks == outcome.numChunks);

        CHECK(stats.completedBatches == 3u);
        CHECK(stats.sourceParticles == 8u);
        CHECK(stats.sourceGenerated == 8u);
        CHECK(stats.sourceSunk == 8u);
        CHECK(stats.sourceLiveParticles == 0u);
        CHECK(stats.chunkCapacity == 8u);
        CHECK(stats.poolChunks == 4u);
        CHECK(stats.peakLiveChunks == 2u);

        // Peak invariants.
        CHECK(stats.sourceLiveParticles == stats.sourceGenerated - stats.sourceSunk);
        CHECK(stats.peakReadyChunks >= stats.peakReadyNodes);
        CHECK(stats.peakReadyChunks >= stats.maxReadyDepth);
        CHECK(stats.peakReadyChunks <= stats.poolChunks);
        CHECK(stats.peakInflightChunks >= stats.peakInflightBatches);
        CHECK(stats.peakInflightChunks <= stats.poolChunks);
        CHECK(stats.peakEventSlotsInUse >= stats.peakInflightBatches);
        CHECK(stats.peakReadyNodes <= outcome.nodeCount);
        CHECK(stats.peakInflightBatches <= outcome.nodeCount);
        CHECK(stats.peakEventSlotsInUse <= outcome.nodeCount);
        CHECK(stats.peakReservedChunks <= stats.poolChunks);
        CHECK(stats.peakReservedChunks == 1u);
        CHECK(stats.peakReadyChunks >= 1u);
        CHECK(stats.peakInflightBatches >= 1u);
        CHECK(stats.peakEventSlotsInUse >= 1u);

        // The on-demand snapshot mirrors the completed-run peak fields.
        CHECK(snapshot.completedBatches == stats.completedBatches);
        CHECK(snapshot.sourceParticles == stats.sourceParticles);
        CHECK(snapshot.sourceGenerated == stats.sourceGenerated);
        CHECK(snapshot.sourceSunk == stats.sourceSunk);
        CHECK(snapshot.sourceLiveParticles == stats.sourceLiveParticles);
        CHECK(snapshot.chunkCapacity == stats.chunkCapacity);
        CHECK(snapshot.poolChunks == stats.poolChunks);
        CHECK(snapshot.peakLiveChunks == stats.peakLiveChunks);
        CHECK(snapshot.peakReservedChunks == stats.peakReservedChunks);
        CHECK(snapshot.peakReadyChunks == stats.peakReadyChunks);
        CHECK(snapshot.peakReadyNodes == stats.peakReadyNodes);
        CHECK(snapshot.maxReadyDepth == stats.maxReadyDepth);
        CHECK(snapshot.peakInflightBatches == stats.peakInflightBatches);
        CHECK(snapshot.peakInflightChunks == stats.peakInflightChunks);
        CHECK(snapshot.peakEventSlotsInUse == stats.peakEventSlotsInUse);
        CHECK(snapshot.freeChunks == outcome.freeChunks);

        // A completed run leaves no aggregate work behind.
        CHECK(snapshot.readyChunks == 0u);
        CHECK(snapshot.readyNodes == 0u);
        CHECK(snapshot.inflightBatches == 0u);
        CHECK(snapshot.inflightChunks == 0u);
        CHECK(snapshot.reservedChunks == 0u);
        CHECK(snapshot.eventSlotsInUse == 0u);
    }
} // namespace graphFlow::test::diagnostics
