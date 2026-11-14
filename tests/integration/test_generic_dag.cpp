/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/ChunkStorage.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/core/Reservation.hpp"
#include "graphFlow/graph/GraphBuilder.hpp"
#include "graphFlow/graph/NodeBase.hpp"
#include "graphFlow/graph/RouteTable.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    /** Generic, non-particle payload used to exercise storage-independent execution. */
    struct GenericPayload
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;
    };

    constexpr std::uint32_t tileSize = 32u;
    constexpr std::uint32_t frameCap = 8u;
    constexpr std::uint32_t feedbackVisitLimit = 2u;

    std::uint64_t chunkKey(graphFlow::ChunkHandle handle)
    {
        return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
    }

    /** Host-side visit count carried with a chunk. Chunk handles include the generation, so a
     * recycled physical chunk always gets a fresh key. */
    using VisitMap = std::unordered_map<std::uint64_t, std::uint32_t>;

    struct FillKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto ids,
            alpaka::concepts::IMdSpan auto visits,
            std::uint64_t firstId,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint32_t const slot = base + i;
                ids[slot] = firstId + i;
                visits[slot] = 0u;
            }
        }
    };

    struct AddVisitKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto outIds,
            alpaka::concepts::IMdSpan auto outVisits,
            alpaka::concepts::IMdSpan auto inIds,
            alpaka::concepts::IMdSpan auto inVisits,
            std::uint32_t outBase,
            std::uint32_t inBase,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                outIds[outBase + i] = inIds[inBase + i];
                outVisits[outBase + i] = inVisits[inBase + i] + 1u;
            }
        }
    };

    struct CopyChunkKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto outIds,
            alpaka::concepts::IMdSpan auto outVisits,
            alpaka::concepts::IMdSpan auto inIds,
            alpaka::concepts::IMdSpan auto inVisits,
            std::uint32_t outBase,
            std::uint32_t inBase,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                outIds[outBase + i] = inIds[inBase + i];
                outVisits[outBase + i] = inVisits[inBase + i];
            }
        }
    };

    struct CountSinkKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto counts,
            alpaka::concepts::IMdSpan auto finalVisits,
            alpaka::concepts::IMdSpan auto ids,
            alpaka::concepts::IMdSpan auto visits,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint32_t const slot = base + i;
                std::size_t const id = static_cast<std::size_t>(ids[slot]);
                onAcc::atomicAdd(acc, &counts[id], std::uint32_t{1u}, onAcc::scope::device);
                finalVisits[id] = visits[slot];
            }
        }
    };

    template<typename TDevice>
    class SourceNode final : public graphFlow::NodeBase<TDevice>
    {
    public:
        using Base = graphFlow::NodeBase<TDevice>;
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        SourceNode(TDevice device, graphFlow::ChunkPool<TDevice>& pool, VisitMap& visits)
            : m_device{device}
            , m_pool{pool}
            , m_visits{visits}
        {
        }

        void submit(Queue const& queue, Batch const& batch) override
        {
            graphFlow::ChunkDescriptor const& chunk = batch.chunk();
            if(chunk.fill == 0u)
                return;
            graphFlow::LaunchPolicy const policy = graphFlow::launchPolicy(m_device, chunk.fill, tileSize, frameCap);
            queue.enqueue(
                policy.frameSpec(),
                alpaka::KernelBundle{
                    FillKernel{},
                    m_pool.ids(),
                    m_pool.visits(),
                    batch.firstId,
                    static_cast<std::uint32_t>(m_pool.base(chunk.handle)),
                    chunk.fill});
        }

        void complete(Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs) override
        {
            for(graphFlow::ChunkDescriptor const& chunk : batch.chunks)
            {
                graphFlow::ChunkDescriptor output = chunk;
                output.bin = 0u;
                m_visits[chunkKey(chunk.handle)] = 0u;
                outputs.push_back(output);
            }
        }

    private:
        TDevice m_device;
        graphFlow::ChunkPool<TDevice>& m_pool;
        VisitMap& m_visits;
    };

    /** A host-only transform that copies one input chunk into one output chunk.
     *
     * `T_feedback == false` increments the visit count and always routes to its single bin.
     * `T_feedback == true` reads the visit count and routes bin 1 back to the earlier transform
     * until `feedbackVisitLimit` is reached, then bin 0 to the sink.
     */
    template<typename TDevice, bool T_feedback>
    class TransformNode final : public graphFlow::NodeBase<TDevice>
    {
    public:
        using Base = graphFlow::NodeBase<TDevice>;
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        TransformNode(TDevice device, graphFlow::ChunkPool<TDevice>& pool, VisitMap& visits)
            : m_device{device}
            , m_pool{pool}
            , m_visits{visits}
        {
        }

        void submit(Queue const&, Batch const&) override
        {
            m_prepared = false;
            m_pendingOutputs.clear();
            m_pendingVisits.clear();
        }

        bool poll(Queue const& queue, Batch const& batch, std::uint32_t& allocated) override
        {
            if(m_prepared)
            {
                allocated = 0u;
                return false;
            }
            m_prepared = true;

            for(graphFlow::ChunkDescriptor const& input : batch.chunks)
            {
                std::uint32_t const inputVisit = m_visits.at(chunkKey(input.handle));
                graphFlow::ChunkDescriptor output;
                output.handle = m_pool.acquire();
                output.fill = input.fill;
                if constexpr(T_feedback)
                    output.bin = inputVisit < feedbackVisitLimit ? 1u : 0u;
                else
                    output.bin = 0u;

                m_pendingOutputs.push_back(output);
                m_pendingVisits.push_back(T_feedback ? inputVisit : inputVisit + 1u);

                if(input.fill == 0u)
                    continue;
                graphFlow::LaunchPolicy const policy
                    = graphFlow::launchPolicy(m_device, input.fill, tileSize, frameCap);
                if constexpr(T_feedback)
                {
                    queue.enqueue(
                        policy.frameSpec(),
                        alpaka::KernelBundle{
                            CopyChunkKernel{},
                            m_pool.ids(),
                            m_pool.visits(),
                            m_pool.ids(),
                            m_pool.visits(),
                            static_cast<std::uint32_t>(m_pool.base(output.handle)),
                            static_cast<std::uint32_t>(m_pool.base(input.handle)),
                            input.fill});
                }
                else
                {
                    queue.enqueue(
                        policy.frameSpec(),
                        alpaka::KernelBundle{
                            AddVisitKernel{},
                            m_pool.ids(),
                            m_pool.visits(),
                            m_pool.ids(),
                            m_pool.visits(),
                            static_cast<std::uint32_t>(m_pool.base(output.handle)),
                            static_cast<std::uint32_t>(m_pool.base(input.handle)),
                            input.fill});
                }
            }

            allocated = static_cast<std::uint32_t>(m_pendingOutputs.size());
            return !m_pendingOutputs.empty();
        }

        void complete(Batch const&, std::vector<graphFlow::ChunkDescriptor>& outputs) override
        {
            for(std::size_t i = 0u; i < m_pendingOutputs.size(); ++i)
            {
                m_visits[chunkKey(m_pendingOutputs[i].handle)] = m_pendingVisits[i];
                outputs.push_back(m_pendingOutputs[i]);
            }
            m_pendingOutputs.clear();
            m_pendingVisits.clear();
            m_prepared = false;
        }

        void retire() override
        {
            for(graphFlow::ChunkDescriptor const& output : m_pendingOutputs)
                m_pool.release(output.handle);
            m_pendingOutputs.clear();
            m_pendingVisits.clear();
            m_prepared = false;
        }

    private:
        TDevice m_device;
        graphFlow::ChunkPool<TDevice>& m_pool;
        VisitMap& m_visits;
        std::vector<graphFlow::ChunkDescriptor> m_pendingOutputs;
        std::vector<std::uint32_t> m_pendingVisits;
        bool m_prepared = false;
    };

    template<typename TDevice>
    class SinkNode final : public graphFlow::NodeBase<TDevice>
    {
    public:
        using Base = graphFlow::NodeBase<TDevice>;
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;
        using U32View = typename graphFlow::ChunkPool<TDevice>::U32View;

        SinkNode(TDevice device, graphFlow::ChunkPool<TDevice>& pool, U32View counts, U32View finalVisits)
            : m_device{device}
            , m_pool{pool}
            , m_counts{std::move(counts)}
            , m_finalVisits{std::move(finalVisits)}
        {
        }

        void submit(Queue const& queue, Batch const& batch) override
        {
            graphFlow::ChunkDescriptor const& chunk = batch.chunk();
            m_sunk += chunk.fill;
            if(chunk.fill == 0u)
                return;
            graphFlow::LaunchPolicy const policy = graphFlow::launchPolicy(m_device, chunk.fill, tileSize, frameCap);
            queue.enqueue(
                policy.frameSpec(),
                alpaka::KernelBundle{
                    CountSinkKernel{},
                    m_counts,
                    m_finalVisits,
                    m_pool.ids(),
                    m_pool.visits(),
                    static_cast<std::uint32_t>(m_pool.base(chunk.handle)),
                    chunk.fill});
        }

        void complete(Batch const&, std::vector<graphFlow::ChunkDescriptor>&) override
        {
        }

        [[nodiscard]] std::uint64_t sunkParticles() const noexcept
        {
            return m_sunk;
        }

    private:
        TDevice m_device;
        graphFlow::ChunkPool<TDevice>& m_pool;
        U32View m_counts;
        U32View m_finalVisits;
        std::uint64_t m_sunk = 0u;
    };

    struct DagResult
    {
        std::vector<std::uint32_t> counts;
        std::vector<std::uint32_t> finalVisits;
        std::uint64_t sunkParticles = 0u;
        graphFlow::SchedulerStats stats;
        std::uint32_t poolChunks = 0u;
        std::uint32_t peakLive = 0u;
    };

    std::uint32_t poolChunksFor(std::uint32_t liveParticles, std::uint32_t capacity)
    {
        std::uint64_t const live = liveParticles == 0u ? 1u : liveParticles;
        std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, 2u);
        return static_cast<std::uint32_t>(live + 2u * reserve);
    }

    template<typename TDevice>
    DagResult runDag(
        TDevice device,
        std::uint64_t numParticles,
        std::uint32_t capacity,
        std::uint32_t liveParticles,
        std::uint32_t batchChunks)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;

        std::uint32_t const numPoolChunks = poolChunksFor(liveParticles, capacity);
        graphFlow::ChunkPool<TDevice> pool{device, numPoolChunks, capacity};

        std::size_t const resultSize = numParticles == 0u ? 1u : static_cast<std::size_t>(numParticles);
        auto counts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});
        auto finalVisits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memset(queue, counts, std::uint8_t{0});
            alpaka::onHost::memset(queue, finalVisits, std::uint8_t{0});
        }

        VisitMap visits;
        SourceNode<TDevice> source{device, pool, visits};
        TransformNode<TDevice, false> first{device, pool, visits};
        TransformNode<TDevice, true> second{device, pool, visits};
        SinkNode<TDevice> sink{device, pool, counts, finalVisits};

        graphFlow::GraphBuilder<TDevice> builder;
        auto const sourceId
            = builder.addNode(graphFlow::NodeRole::source, "source", 1u, graphFlow::RouteTable{{1u}}, source);
        builder.addNode(graphFlow::NodeRole::transform, "first", 1u, graphFlow::RouteTable{{2u}}, first);
        builder.addNode(graphFlow::NodeRole::transform, "second", 2u, graphFlow::RouteTable{{3u, 1u}}, second);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        builder.setSource(sourceId, numParticles, liveParticles);
        auto const graph = builder.build();

        Runner runner{device, pool};
        graph.attachTo(runner);
        runner.setBatchChunks(batchChunks);
        graphFlow::SchedulerStats const stats = runner.run();

        std::vector<std::uint32_t> hostCounts(resultSize, 0u);
        std::vector<std::uint32_t> hostVisits(resultSize, 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, hostCounts, counts);
            alpaka::onHost::memcpy(queue, hostVisits, finalVisits);
        }

        hostCounts.resize(static_cast<std::size_t>(numParticles));
        hostVisits.resize(static_cast<std::size_t>(numParticles));

        return DagResult{
            std::move(hostCounts),
            std::move(hostVisits),
            sink.sunkParticles(),
            stats,
            pool.numChunks(),
            pool.peakLive()};
    }

    struct NullNode final : graphFlow::NodeBase<Device>
    {
        using Base = graphFlow::NodeBase<Device>;
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        void submit(Queue const&, Batch const&) override
        {
        }

        void complete(Batch const&, std::vector<graphFlow::ChunkDescriptor>&) override
        {
        }
    };
} // namespace

TEST_CASE("generic DAG runs a source, two transforms, a sink, and a feedback edge", "[integration][graph]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    std::uint64_t const particles = 257u;
    std::uint32_t const capacity = 8u;
    std::uint32_t const liveParticles = 32u;
    std::uint32_t const expectedPool = poolChunksFor(liveParticles, capacity);

    std::vector<DagResult> runs;
    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        CAPTURE(batchChunks);
        DagResult run = runDag(device, particles, capacity, liveParticles, batchChunks);
        REQUIRE(run.sunkParticles == particles);
        REQUIRE(run.counts.size() == particles);
        REQUIRE(run.finalVisits.size() == particles);
        CHECK(std::all_of(run.counts.begin(), run.counts.end(), [](std::uint32_t count) { return count == 1u; }));
        CHECK(
            std::all_of(
                run.finalVisits.begin(),
                run.finalVisits.end(),
                [](std::uint32_t visit) { return visit == feedbackVisitLimit; }));
        CHECK(run.stats.completedBatches > 0u);
        CHECK(run.poolChunks == expectedPool);
        CHECK(run.peakLive <= run.poolChunks);
        runs.push_back(std::move(run));
    }

    REQUIRE(runs.size() == 3u);
    CHECK(runs[0].counts == runs[1].counts);
    CHECK(runs[0].counts == runs[2].counts);
    CHECK(runs[0].finalVisits == runs[1].finalVisits);
    CHECK(runs[0].finalVisits == runs[2].finalVisits);
}

TEST_CASE("generic DAG N=0 completes without device work", "[integration][graph]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    DagResult const run = runDag(device, 0u, 8u, 0u, 1u);

    CHECK(run.sunkParticles == 0u);
    CHECK(run.counts.empty());
    CHECK(run.finalVisits.empty());
    CHECK(run.stats.completedBatches == 0u);
    CHECK(run.poolChunks == poolChunksFor(0u, 8u));
}

TEST_CASE("generic DAG pool stays bounded for N much larger than P", "[integration][graph]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    std::uint64_t const particles = 1000u;
    std::uint32_t const capacity = 8u;
    std::uint32_t const liveParticles = 32u;
    DagResult const run = runDag(device, particles, capacity, liveParticles, 8u);

    REQUIRE(run.sunkParticles == particles);
    CHECK(std::all_of(run.counts.begin(), run.counts.end(), [](std::uint32_t count) { return count == 1u; }));
    CHECK(
        std::all_of(
            run.finalVisits.begin(),
            run.finalVisits.end(),
            [](std::uint32_t visit) { return visit == feedbackVisitLimit; }));
    CHECK(run.poolChunks == poolChunksFor(liveParticles, capacity));
    CHECK(run.peakLive <= run.poolChunks);
}

TEST_CASE("generic DAG rejects invalid routes and sources", "[integration][graph]")
{
    NullNode source;
    NullNode transform;
    NullNode sink;

    {
        graphFlow::GraphBuilder<Device> builder;
        auto const sourceId
            = builder.addNode(graphFlow::NodeRole::source, "source", 1u, graphFlow::RouteTable{{5u}}, source);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        builder.setSource(sourceId, 0u);
        CHECK_THROWS_AS(builder.build(), std::invalid_argument); // route out of range
    }
    {
        graphFlow::GraphBuilder<Device> builder;
        auto const sourceId
            = builder.addNode(graphFlow::NodeRole::source, "source", 1u, graphFlow::RouteTable{{1u}}, source);
        builder.addNode(graphFlow::NodeRole::transform, "transform", 1u, graphFlow::RouteTable{{0u}}, transform);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        builder.setSource(sourceId, 0u);
        CHECK_THROWS_AS(builder.build(), std::invalid_argument); // route back to source
    }
    {
        graphFlow::GraphBuilder<Device> builder;
        builder.addNode(graphFlow::NodeRole::source, "source", 1u, graphFlow::RouteTable{{1u}}, source);
        auto const transformId
            = builder.addNode(graphFlow::NodeRole::transform, "transform", 1u, graphFlow::RouteTable{{2u}}, transform);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        CHECK_THROWS_AS(builder.setSource(transformId, 0u), std::invalid_argument); // not a source
    }
}

TEST_CASE("generic DAG runs on a payload-slot storage", "[integration][graph][storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Storage = graphFlow::PayloadSlotStorage<GenericPayload, Device>;
    using Runner = graphFlow::GraphRunner<Device, Storage>;
    static_assert(graphFlow::ChunkStorage<Storage>);

    constexpr std::uint32_t capacity = 4u;
    constexpr std::uint64_t particles = 8u;
    Storage storage{device, 4u, capacity};

    std::unordered_map<std::uint64_t, std::vector<GenericPayload>> readback;
    std::vector<std::uint64_t> sunk;

    Runner::Hooks sourceHooks{
        [&storage](Runner::Queue const& queue, Runner::Batch const& batch)
        {
            auto const& chunk = batch.chunk();
            std::vector<GenericPayload> values(chunk.fill);
            for(std::uint32_t i = 0u; i < chunk.fill; ++i)
                values[i] = GenericPayload{batch.firstId + i, i};
            auto chunkView = storage.payload().getSubView(
                alpaka::Vec{storage.base(chunk.handle)},
                alpaka::Vec{std::uint64_t{chunk.fill}});
            alpaka::onHost::memcpy(queue, chunkView, values);
            alpaka::onHost::wait(queue);
        },
        [](Runner::Queue const&, Runner::Batch const&, std::uint32_t&) { return false; },
        [](Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs) { outputs = batch.chunks; },
        []() {}};

    Runner::Hooks sinkHooks{
        [&storage, &readback](Runner::Queue const& queue, Runner::Batch const& batch)
        {
            auto const& chunk = batch.chunk();
            std::vector<GenericPayload> values(chunk.fill);
            auto chunkView = storage.payload().getSubView(
                alpaka::Vec{storage.base(chunk.handle)},
                alpaka::Vec{std::uint64_t{chunk.fill}});
            alpaka::onHost::memcpy(queue, values, chunkView);
            alpaka::onHost::wait(queue);
            readback.emplace(chunkKey(chunk.handle), std::move(values));
        },
        [](Runner::Queue const&, Runner::Batch const&, std::uint32_t&) { return false; },
        [&readback, &sunk](Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>&)
        {
            auto const found = readback.find(chunkKey(batch.chunk().handle));
            REQUIRE(found != readback.end());
            for(GenericPayload const& payload : found->second)
                sunk.push_back(payload.id);
        },
        []() {}};

    Runner runner{device, storage};
    auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks);
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, sinkHooks);
    runner.setSource(source, particles, 0u);

    graphFlow::SchedulerStats const stats = runner.run();
    CHECK(stats.completedBatches == 4u);
    CHECK(stats.poolChunks == 4u);
    CHECK(storage.freeCount() == storage.numChunks());

    std::sort(sunk.begin(), sunk.end());
    REQUIRE(sunk.size() == particles);
    for(std::uint64_t i = 0u; i < particles; ++i)
        CHECK(sunk[static_cast<std::size_t>(i)] == i);
}
