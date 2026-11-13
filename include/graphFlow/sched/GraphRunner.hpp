/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/Reservation.hpp"

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <concepts>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace graphFlow
{
    /** Role of a graph node. The role only decides input-ownership retirement; the work itself is
     * described by the node hooks. */
    enum class NodeRole
    {
        source,
        transform,
        sink
    };

    /** Batch error raised by a node hook, e.g. a device status word or a planning mismatch. */
    class BatchError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    struct SchedulerStats
    {
        std::uint64_t completedBatches = 0u;
        std::uint64_t loopIterations = 0u;
        std::uint64_t yields = 0u;
    };

    /** Host-side dataflow scheduler.
     *
     * One non-blocking queue and one event per node. Every node has at most one in-flight batch and
     * one host ready deque, so live device work is bounded by the node count and the chunk pool. The
     * scheduler thread owns queue submission, routing, event retirement, free-list updates and
     * output-chunk reservation; it never blocks on a queue in the steady state.
     *
     * Completion protocol per batch:
     *  - `launch` enqueues the first device stage,
     *  - the runner enqueues the node event and polls `event.isComplete()`,
     *  - `advance` runs on the host after the event fired. It may enqueue a second stage and return
     *    `true`; the runner re-arms the event. Returning `false` means the batch is finished,
     *  - `finish` publishes the output descriptors; the runner routes them by bin and only then
     *    retires the input chunk.
     */
    template<typename TDevice>
    class GraphRunner
    {
    public:
        using Queue = ALPAKA_TYPEOF(std::declval<TDevice&>().makeQueue(alpaka::queueKind::nonBlocking));
        using Event = ALPAKA_TYPEOF(std::declval<TDevice&>().makeEvent());

        struct Batch
        {
            ChunkDescriptor chunk{};
            std::uint64_t firstId = 0u;
        };

        struct Hooks
        {
            std::function<void(Queue const&, Batch const&)> launch;
            /** Returns true when a further device stage was enqueued. `allocated` reports the number
             * of output chunks physically acquired from the pool during host planning. */
            std::function<bool(Queue const&, ChunkDescriptor const&, std::uint32_t& allocated)> advance;
            std::function<void(ChunkDescriptor const&, std::vector<ChunkDescriptor>&)> finish;
        };

        GraphRunner(TDevice device, ChunkPool<TDevice>& pool)
            : m_device{device}
            , m_pool{pool}
        {
        }

        /** Register a node. `route[bin]` is the destination node index for that bin. */
        std::uint32_t addNode(
            std::string name,
            NodeRole role,
            std::uint32_t bins,
            std::vector<int> route,
            Hooks hooks)
        {
            if(role == NodeRole::sink)
            {
                if(!route.empty())
                    throw std::invalid_argument{"graphFlow: a sink node has no routes"};
            }
            else if(route.size() != bins)
            {
                throw std::invalid_argument{"graphFlow: route table size must equal the bin count"};
            }
            for(int const dst : route)
            {
                if(dst < 0)
                    throw std::invalid_argument{"graphFlow: negative route destination"};
            }
            auto node = std::make_unique<Node>(
                std::move(name),
                role,
                bins,
                std::move(route),
                std::move(hooks),
                m_device.makeQueue(alpaka::queueKind::nonBlocking),
                m_device.makeEvent());
            m_nodes.push_back(std::move(node));
            return static_cast<std::uint32_t>(m_nodes.size() - 1u);
        }

        /** Declare the single source node that is admitted up to `numParticles` particles. */
        void setSource(std::uint32_t node, std::uint64_t numParticles)
        {
            if(node >= m_nodes.size())
                throw std::invalid_argument{"graphFlow: source node out of range"};
            m_sourceNode = node;
            m_sourceParticles = numParticles;
        }

        /** Threads per block for a node launch.
         *
         * The host backend executes exactly one real thread per block (`api/host/exec/Serial.hpp`
         * rejects any other block extent), so the CPU path is single-threaded per block while the
         * kernel keeps the tile and barrier structure required by accelerators.
         */
        [[nodiscard]] static std::uint32_t threadsPerBlock(TDevice const& device, std::uint32_t requested)
        {
            if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Host>)
            {
                alpaka::unused(device);
                alpaka::unused(requested);
                return 1u;
            }
            else
            {
                alpaka::unused(device);
                return requested;
            }
        }

        SchedulerStats run()
        {
            if(!m_sourceNode)
                throw std::logic_error{"graphFlow: no source node configured"};
            for(auto const& nodePtr : m_nodes)
            {
                for(int const dst : nodePtr->route)
                {
                    if(dst >= static_cast<int>(m_nodes.size()))
                        throw std::logic_error{"graphFlow: route destination out of range"};
                }
            }

            SchedulerStats stats;
            std::uint64_t generated = 0u;
            std::uint32_t reservedTotal = 0u;
            std::uint64_t idleSpins = 0u;
            auto lastProgress = std::chrono::steady_clock::now();

            auto availableChunks = [&]() -> std::uint32_t
            {
                if(reservedTotal > m_pool.freeCount())
                    throw std::logic_error{"graphFlow: reservation accounting underflow"};
                return m_pool.freeCount() - reservedTotal;
            };

            while(true)
            {
                bool progress = false;

                // 1) Retire completed events.
                for(auto& nodePtr : m_nodes)
                {
                    Node& node = *nodePtr;
                    if(!node.armed || !node.event.isComplete())
                        continue;
                    node.armed = false;
                    Batch const batch = *node.inflight;

                    std::uint32_t allocated = 0u;
                    bool const moreWork = node.hooks.advance(node.queue, batch.chunk, allocated);
                    if(allocated != 0u)
                    {
                        if(allocated > node.reserved)
                            throw std::logic_error{"graphFlow: node acquired more chunks than reserved"};
                        reservedTotal -= node.reserved;
                        node.reserved = 0u;
                    }

                    if(moreWork)
                    {
                        node.queue.enqueue(node.event);
                        node.armed = true;
                    }
                    else
                    {
                        std::vector<ChunkDescriptor> outputs;
                        node.hooks.finish(batch.chunk, outputs);
                        node.inflight.reset();
                        ++node.completedChunks;
                        ++stats.completedBatches;

                        for(ChunkDescriptor const& output : outputs)
                        {
                            if(output.bin >= node.route.size())
                                throw std::runtime_error{
                                    "graphFlow: node '" + node.name + "' produced an unroutable bin"};
                            int const dst = node.route[output.bin];
                            if(dst < 0)
                                throw std::runtime_error{
                                    "graphFlow: node '" + node.name + "' bin " + std::to_string(output.bin)
                                    + " has no route"};
                            m_nodes[static_cast<std::size_t>(dst)]->ready.push_back(output);
                        }

                        // A source hands its chunk downstream; every other node retires its input
                        // only after the last device access of this batch has completed.
                        if(node.role != NodeRole::source)
                            m_pool.release(batch.chunk.handle);
                    }
                    progress = true;
                }

                // 2) Admit the next source chunk while particles are left and storage is free.
                Node& source = *m_nodes[*m_sourceNode];
                if(generated < m_sourceParticles && !source.inflight && m_pool.freeCount() > reservedTotal)
                {
                    std::uint64_t const remaining = m_sourceParticles - generated;
                    std::uint32_t const fill = remaining < m_pool.chunkCapacity()
                                                 ? static_cast<std::uint32_t>(remaining)
                                                 : m_pool.chunkCapacity();
                    ChunkDescriptor chunk{m_pool.acquire(), fill, 0u};
                    Batch const batch{chunk, generated};
                    generated += fill;
                    source.hooks.launch(source.queue, batch);
                    source.queue.enqueue(source.event);
                    source.inflight = batch;
                    source.armed = true;
                    progress = true;
                }

                // 3) Dispatch ready work in fair round-robin order, one batch per pass.
                std::size_t const nodeCount = m_nodes.size();
                for(std::size_t k = 0u; k < nodeCount; ++k)
                {
                    std::size_t const index = (m_roundRobin + k) % nodeCount;
                    Node& node = *m_nodes[index];
                    if(node.ready.empty() || node.inflight)
                        continue;

                    ChunkDescriptor const chunk = node.ready.front();
                    std::uint32_t reserve = 0u;
                    if(node.role == NodeRole::transform)
                    {
                        reserve = static_cast<std::uint32_t>(
                            outputChunkBound(chunk.fill, m_pool.chunkCapacity(), node.bins));
                        if(availableChunks() < reserve)
                            continue; // keep the chunk queued until storage credits exist
                    }
                    node.ready.pop_front();
                    Batch const batch{chunk, 0u};
                    node.hooks.launch(node.queue, batch);
                    node.queue.enqueue(node.event);
                    node.inflight = batch;
                    node.reserved = reserve;
                    reservedTotal += reserve;
                    node.armed = true;
                    progress = true;
                    break;
                }
                ++m_roundRobin;

                // 4) Termination: source exhausted, every ready list and in-flight batch empty,
                //    and no output-chunk reservation outstanding.
                bool const sourceExhausted = generated >= m_sourceParticles;
                bool drained = true;
                for(auto const& nodePtr : m_nodes)
                {
                    if(nodePtr->inflight || !nodePtr->ready.empty())
                    {
                        drained = false;
                        break;
                    }
                }
                if(sourceExhausted && drained && reservedTotal == 0u)
                    break;

                if(!progress)
                {
                    ++idleSpins;
                    if(idleSpins > 64u)
                    {
                        std::this_thread::yield();
                        ++stats.yields;
                    }
                    auto const idle = std::chrono::steady_clock::now() - lastProgress;
                    if(idleSpins > 200'000'000u
                       || std::chrono::duration_cast<std::chrono::seconds>(idle).count() > 120)
                        throw std::runtime_error{"graphFlow: scheduler stalled"};
                }
                else
                {
                    idleSpins = 0u;
                    lastProgress = std::chrono::steady_clock::now();
                }
                ++stats.loopIterations;
            }

            // Drain every queue so the caller may read result buffers immediately.
            for(auto& nodePtr : m_nodes)
            {
                if(nodePtr->armed)
                    nodePtr->queue.waitFor(nodePtr->event);
                alpaka::onHost::wait(nodePtr->queue);
                nodePtr->armed = false;
            }
            if(m_pool.freeCount() != m_pool.numChunks())
                throw std::logic_error{"graphFlow: chunk leak after the run"};
            return stats;
        }

        [[nodiscard]] std::uint32_t nodeCompleted(std::uint32_t node) const
        {
            return m_nodes.at(node)->completedChunks;
        }

    private:
        struct Node
        {
            Node(
                std::string nodeName,
                NodeRole nodeRole,
                std::uint32_t nodeBins,
                std::vector<int> nodeRoute,
                Hooks nodeHooks,
                Queue nodeQueue,
                Event nodeEvent)
                : name{std::move(nodeName)}
                , role{nodeRole}
                , bins{nodeBins}
                , route{std::move(nodeRoute)}
                , hooks{std::move(nodeHooks)}
                , queue{std::move(nodeQueue)}
                , event{std::move(nodeEvent)}
            {
            }

            std::string name;
            NodeRole role;
            std::uint32_t bins;
            std::vector<int> route;
            Hooks hooks;
            Queue queue;
            Event event;
            std::deque<ChunkDescriptor> ready;
            std::optional<Batch> inflight;
            bool armed = false;
            std::uint32_t reserved = 0u;
            std::uint32_t completedChunks = 0u;
        };

        TDevice m_device;
        ChunkPool<TDevice>& m_pool;
        std::vector<std::unique_ptr<Node>> m_nodes;
        std::optional<std::uint32_t> m_sourceNode;
        std::uint64_t m_sourceParticles = 0u;
        std::size_t m_roundRobin = 0u;
    };
} // namespace graphFlow
