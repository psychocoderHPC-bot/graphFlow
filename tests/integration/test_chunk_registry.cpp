/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/ChunkRegistry.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace
{
    using Catch::Matchers::ContainsSubstring;

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

    /** Sink without registry assertions; used by the fault-injection run. */
    template<typename TRunner>
    typename TRunner::Hooks plainSinkHooks(std::uint64_t& sunk)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
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

    /** Sink that validates the registry from inside the scheduler loop.
     *
     * `launch` runs after routing and before the runner records the dispatch, so the chunk must be
     * ready at the sink. `finish` runs after the dispatch record and before the retirement record,
     * so the chunk must be inflight at the sink. Removing the dispatch record makes this fail.
     */
    template<typename TRunner>
    typename TRunner::Hooks validatingSinkHooks(TRunner& runner, std::uint64_t& sunk, std::uint32_t sinkNode)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [&runner, sinkNode](auto const&, typename TRunner::Batch const& batch)
        {
            graphFlow::ChunkRecord const& record = runner.chunkRegistry().at(batch.chunk().handle.index);
            bool const readyAtSink = record.phase == graphFlow::ChunkPhase::ready && record.ownerNode == sinkNode
                                     && record.destinationNode == static_cast<std::int32_t>(sinkNode);
            CHECK(readyAtSink);
            runner.validateChunkRegistry();
        };
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        hooks.finish =
            [&runner, &sunk, sinkNode](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>&)
        {
            for(graphFlow::ChunkDescriptor const& chunk : batch.chunks)
            {
                graphFlow::ChunkRecord const& record = runner.chunkRegistry().at(chunk.handle.index);
                bool const inflightAtSink = record.phase == graphFlow::ChunkPhase::inflight
                                            && record.ownerNode == sinkNode
                                            && record.destinationNode == static_cast<std::int32_t>(sinkNode);
                CHECK(inflightAtSink);
            }
            CHECK(runner.chunkRegistrySummary().liveChunks > 0u);
            sunk += batch.particles();
        };
        hooks.abort = [] {};
        return hooks;
    }

    /** Transform that conserves particles into one output chunk per input batch. */
    template<typename TRunner, typename TPool>
    typename TRunner::Hooks conservingTransformHooks(TRunner& runner, TPool& pool, std::uint32_t transformNode)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        };
        hooks.finish
            = [&runner,
               &pool,
               transformNode](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        {
            for(graphFlow::ChunkDescriptor const& input : batch.chunks)
            {
                graphFlow::ChunkRecord const& record = runner.chunkRegistry().at(input.handle.index);
                bool const inflightAtTransform
                    = record.phase == graphFlow::ChunkPhase::inflight && record.ownerNode == transformNode;
                CHECK(inflightAtTransform);
            }
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

    /** Chunk storage that suppresses every release observation.
     *
     * The pool still releases the slots; the registry misses the transitions. The next acquire of
     * a reused index hits a record that is not free, which records a sticky fault and proves that
     * the run does not abort. Dropping only the first release is not enough: its index may never
     * be acquired again in a short run.
     */
    template<typename TDevice>
    class DropReleaseStorage
    {
    public:
        using handle_type = graphFlow::ChunkHandle;

        DropReleaseStorage(TDevice device, std::uint32_t numChunks, std::uint32_t chunkCapacity)
            : m_pool{device, numChunks, chunkCapacity}
        {
        }

        [[nodiscard]] std::uint32_t numChunks() const
        {
            return m_pool.numChunks();
        }

        [[nodiscard]] std::uint32_t chunkCapacity() const
        {
            return m_pool.chunkCapacity();
        }

        [[nodiscard]] std::uint32_t freeCount() const
        {
            return m_pool.freeCount();
        }

        [[nodiscard]] std::uint32_t liveCount() const
        {
            return m_pool.liveCount();
        }

        [[nodiscard]] graphFlow::ChunkHandle acquire()
        {
            return m_pool.acquire();
        }

        void release(graphFlow::ChunkHandle handle)
        {
            m_pool.release(handle);
        }

        void acquireLease(graphFlow::ChunkHandle handle)
        {
            m_pool.acquireLease(handle);
        }

        void releaseLease(graphFlow::ChunkHandle handle)
        {
            m_pool.releaseLease(handle);
        }

        [[nodiscard]] std::uint32_t leaseCount(graphFlow::ChunkHandle handle) const
        {
            return m_pool.leaseCount(handle);
        }

        [[nodiscard]] bool isLive(graphFlow::ChunkHandle handle) const
        {
            return m_pool.isLive(handle);
        }

        [[nodiscard]] std::uint64_t base(graphFlow::ChunkHandle handle) const
        {
            return m_pool.base(handle);
        }

        [[nodiscard]] std::uint32_t generation(std::uint32_t index) const
        {
            return m_pool.generation(index);
        }

        void validateLive(graphFlow::ChunkHandle handle) const
        {
            m_pool.validateLive(handle);
        }

        [[nodiscard]] std::uint32_t peakLive() const
        {
            return m_pool.peakLive();
        }

        [[nodiscard]] std::uint32_t peakLeases() const
        {
            return m_pool.peakLeases();
        }

        void setChunkObservers(
            std::function<void(graphFlow::ChunkHandle)> acquired,
            [[maybe_unused]] std::function<void(graphFlow::ChunkHandle)> released)
        {
            m_pool.setChunkObservers(std::move(acquired), [](graphFlow::ChunkHandle) {});
        }

        void clearChunkObservers()
        {
            m_pool.clearChunkObservers();
        }

    private:
        graphFlow::ChunkPool<TDevice> m_pool;
    };
} // namespace

TEST_CASE("chunk registry tracks a conserving pipeline", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Runner = graphFlow::GraphRunner<decltype(device)>;
    graphFlow::ChunkPool<decltype(device)> pool{device, 16u, 8u};
    Runner runner{device, pool};

    std::uint64_t sunk = 0u;
    auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        1u,
        {2},
        conservingTransformHooks<Runner>(runner, pool, 1u));
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, validatingSinkHooks<Runner>(runner, sunk, 2u));
    runner.setSource(source, 16u, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    graphFlow::SchedulerStats const stats = runner.run();
    runner.validateChunkRegistry();

    CHECK(stats.sourceGenerated == 16u);
    CHECK(stats.sourceSunk == 16u);
    CHECK(sunk == 16u);
    CHECK(stats.chunkRegistryFaults == 0u);
    CHECK(pool.freeCount() == pool.numChunks());

    graphFlow::ChunkRegistrySummary const summary = runner.chunkRegistrySummary();
    CHECK(summary.chunks == 16u);
    CHECK(summary.liveChunks == 0u);
    CHECK(summary.freeChunks == 16u);
    CHECK(summary.readyChunks == 0u);
    CHECK(summary.inflightChunks == 0u);
    CHECK(summary.reservedChunks == 0u);
    CHECK(summary.retiredChunks == 0u);
    CHECK(summary.acquisitions == 4u); // two source chunks and two transform outputs
    CHECK(summary.releases == 4u);
    CHECK(summary.routings == 4u);
    CHECK(summary.retirements == 2u);
    CHECK(summary.reservations == 2u); // one non-zero reserve per transform batch
    CHECK(summary.reservedCredits == 2u);
    CHECK(summary.peakInflightChunks >= 1u);
    CHECK(stats.chunkRegistryAcquisitions == summary.acquisitions);
    CHECK(stats.chunkRegistryRoutings == summary.routings);
    CHECK(stats.chunkRegistryRetirements == summary.retirements);
    CHECK(stats.chunkRegistryFaults == summary.faults);
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
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, validatingSinkHooks<Runner>(runner, sunk, 2u));
    runner.setSource(source, 16u, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    graphFlow::SchedulerStats const stats = runner.run();
    runner.validateChunkRegistry();

    CHECK(stats.sourceGenerated == 16u);
    CHECK(stats.sourceSunk == 16u);
    CHECK(sunk == 16u);

    graphFlow::ChunkRegistrySummary const summary = runner.chunkRegistrySummary();
    CHECK(summary.chunks == 32u);
    CHECK(summary.liveChunks == 0u);
    CHECK(summary.freeChunks == 32u);
    CHECK(summary.acquisitions == 18u); // two source chunks and sixteen one-particle outputs
    CHECK(summary.releases == 18u);
    CHECK(summary.routings == 18u);
    CHECK(summary.retirements == 16u);
    CHECK(summary.reservations == 2u); // one eight-credit reserve per transform batch
    CHECK(summary.reservedCredits == 16u);
    // The stream really fragmented: more chunks were live than the two source chunks.
    CHECK(summary.peakReservedChunks == 8u);
    CHECK(summary.peakReadyChunks >= 8u);
    CHECK(summary.peakInflightChunks >= 2u);
    CHECK(stats.chunkRegistryRoutings == 18u);
    CHECK(stats.chunkRegistryRetirements == 16u);
    CHECK(stats.chunkRegistryFaults == 0u);

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
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        1u,
        {2},
        conservingTransformHooks<Runner>(runner, pool, 1u));
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, validatingSinkHooks<Runner>(runner, sunk, 2u));
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
    CHECK(secondRegistry.liveChunks == 0u);
    CHECK(secondRegistry.freeChunks == secondRegistry.chunks);
    runner.validateChunkRegistry();
}

TEST_CASE("chunk registry seeds from a pool that already advanced a generation", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Runner = graphFlow::GraphRunner<decltype(device)>;
    graphFlow::ChunkPool<decltype(device)> pool{device, 4u, 8u};
    graphFlow::ChunkHandle const recycled = pool.acquire();
    pool.release(recycled);
    REQUIRE(pool.generation(recycled.index) == 1u);

    Runner runner{device, pool};
    CHECK_NOTHROW(runner.validateChunkRegistry());
    CHECK(runner.chunkRegistry()[recycled.index].handle.generation == 1u);
    CHECK(runner.chunkRegistrySummary().freeChunks == 4u);
}

TEST_CASE("chunk registry rejects an impossible owner and destination pair", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ChunkPool<decltype(device)> pool{device, 2u, 8u};
    graphFlow::ChunkRegistry registry{2u};
    registry.seed(pool);

    graphFlow::ChunkHandle const handle = pool.acquire();
    registry.onAcquire(handle);
    pool.acquireLease(handle);
    // Dispatch without a route: the chunk is inflight at a non-source node but has no destination.
    registry.onDispatch(graphFlow::ChunkDescriptor{handle, 4u, 0u}, 4u);
    CHECK_THROWS_WITH(registry.validate(pool, 0u), ContainsSubstring("destination"));

    pool.releaseLease(handle);
    pool.release(handle);
}

TEST_CASE("chunk registry routes a generation mismatch through the fault channel", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ChunkPool<decltype(device)> pool{device, 1u, 8u};
    graphFlow::ChunkRegistry registry{1u};
    registry.seed(pool);

    graphFlow::ChunkHandle const handle = pool.acquire();
    registry.onAcquire(handle);
    pool.release(handle); // No observer installed: the registry misses the generation bump.
    REQUIRE(pool.generation(handle.index) == 1u);
    REQUIRE(registry.records()[handle.index].handle.generation == 0u);

    CHECK_THROWS_WITH(registry.validate(pool), ContainsSubstring("chunk registry fault"));
    CHECK(registry.summary().faults == 1u);
    CHECK_THAT(registry.faultDetail(), ContainsSubstring("generation"));
}

TEST_CASE("chunk registry phase mismatch does not abort a run", "[chunk-registry][integration]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Storage = DropReleaseStorage<decltype(device)>;
    using Runner = graphFlow::GraphRunner<decltype(device), Storage>;
    Storage storage{device, 16u, 8u};
    Runner runner{device, storage};

    std::uint64_t sunk = 0u;
    auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        1u,
        {2},
        conservingTransformHooks<Runner>(runner, storage, 1u));
    runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, plainSinkHooks<Runner>(sunk));
    runner.setSource(source, 16u, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    graphFlow::SchedulerStats const stats = runner.run();
    CHECK(stats.sourceSunk == 16u);
    CHECK(sunk == 16u);
    CHECK(stats.chunkRegistryFaults >= 1u);
    CHECK_FALSE(runner.chunkRegistryFaultDetail().empty());
    CHECK_THROWS_WITH(runner.validateChunkRegistry(), ContainsSubstring("chunk registry fault"));
}
