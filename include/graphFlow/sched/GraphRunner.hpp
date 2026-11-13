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
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
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
        std::uint32_t poolChunks = 0u;
        std::uint32_t peakLiveChunks = 0u;
        std::uint32_t peakReservedChunks = 0u;
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
     *
     * Source admission is expressed in particles, not in free-list entries: a source chunk is only
     * admitted while fewer than `liveParticles` (P) particles are between source and sink, and its
     * fill is capped at `P - liveParticles`, so the data-carrying chunk count never exceeds P. The
     * pool therefore holds P data chunks plus the `K*R` output credits of the live transform
     * batches, independent of the total particle count N. (`ceil(P/C)` alone is not a safe bound
     * once split routing can fragment particles into one chunk each; see derivePoolSize.)
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

        /** Declare the single source node and its particle-admission capacity P.
         *
         * `liveParticles == 0` disables the particle budget (used by small unit tests); the example
         * always supplies the configured P.
         */
        void setSource(std::uint32_t node, std::uint64_t numParticles, std::uint32_t liveParticles = 0u)
        {
            if(node >= m_nodes.size())
                throw std::invalid_argument{"graphFlow: source node out of range"};
            m_sourceNode = node;
            m_sourceParticles = numParticles;
            m_sourceLiveParticles = liveParticles;
        }

        /** Maximum quiet time before the scheduler reports a stall instead of spinning forever. */
        void setStallTimeout(std::chrono::milliseconds timeout)
        {
            m_stallTimeout = timeout;
        }

        /** Threads per block for a node launch.
         *
         * The host backend executes exactly one real thread per block (`api/host/exec/Serial.hpp`
         * rejects any other block extent), so `ThreadSpec{numBlocks, T}` maps its thread extent to
         * `1` on `api::Host` while the kernel keeps the tile and barrier structure required by
         * accelerators. The transform stages launch one block per tile and pass the logical tile
         * width `T` separately from this resolved block extent.
         *
         * Accelerator launches use the requested extent but never exceed
         * `DeviceProperties::maxThreadsPerBlock`. An invalid extent (zero, or larger than the
         * device limit) is rejected here instead of being handed to `ThreadSpec`.
         */
        [[nodiscard]] static std::uint32_t threadsPerBlock(TDevice const& device, std::uint32_t requested)
        {
            if(requested == 0u)
                throw std::invalid_argument{"graphFlow: threads per block must be nonzero"};
            if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Host>)
            {
                alpaka::unused(device);
                return 1u;
            }
            else
            {
                std::uint32_t const limit = device.getDeviceProperties().maxThreadsPerBlock;
                if(limit == 0u)
                    throw std::runtime_error{"graphFlow: device reports maxThreadsPerBlock == 0"};
                if(requested > limit)
                    throw std::invalid_argument{
                        "graphFlow: requested " + std::to_string(requested)
                        + " threads per block exceeds maxThreadsPerBlock " + std::to_string(limit)};
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
            std::uint64_t sunkParticles = 0u;
            std::uint32_t reservedTotal = 0u;
            std::uint64_t idleSpins = 0u;
            auto lastProgress = std::chrono::steady_clock::now();

            auto availableChunks = [&]() -> std::uint32_t
            {
                if(reservedTotal > m_pool.freeCount())
                    throw std::logic_error{"graphFlow: reservation accounting underflow"};
                return m_pool.freeCount() - reservedTotal;
            };

            try
            {
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
                        else if(!moreWork && node.reserved != 0u)
                        {
                            // A finished batch that produced no output releases its credit.
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
                            ++node.completedChunks;
                            ++stats.completedBatches;

                            for(std::size_t o = 0u; o < outputs.size(); ++o)
                            {
                                ChunkDescriptor const& output = outputs[o];
                                try
                                {
                                    if(output.bin >= node.route.size())
                                        throw std::runtime_error{
                                            "graphFlow: node '" + node.name + "' produced an unroutable bin"};
                                    int const dst = node.route[output.bin];
                                    if(dst < 0)
                                        throw std::runtime_error{
                                            "graphFlow: node '" + node.name + "' bin "
                                            + std::to_string(output.bin) + " has no route"};
                                    m_nodes[static_cast<std::size_t>(dst)]->ready.push_back(output);
                                }
                                catch(...)
                                {
                                    // Do not leak outputs that were not published; the input is
                                    // released by teardown because `inflight` is still set.
                                    for(std::size_t rest = o; rest < outputs.size(); ++rest)
                                    {
                                        if(outputs[rest].handle == batch.chunk.handle)
                                            node.inputOwned = false;
                                        m_pool.release(outputs[rest].handle);
                                    }
                                    throw;
                                }
                                node.inputOwned = false;
                            }

                            if(node.role == NodeRole::sink)
                                sunkParticles += batch.chunk.fill;

                            // A source hands its chunk downstream; every other node retires its input
                            // only after the completion event of the last device stage has fired.
                            if(node.role != NodeRole::source)
                            {
                                if(node.leaseHeld)
                                {
                                    m_pool.releaseLease(batch.chunk.handle);
                                    node.leaseHeld = false;
                                }
                                m_pool.release(batch.chunk.handle);
                                node.inputOwned = false;
                            }
                            node.inflight.reset();
                        }
                        progress = true;
                    }

                    // 2) Admit the next source chunk. The admission decision is the configured
                    //    particle budget P (chunk form ceil(P/C)); the free-list comparison is only
                    //    the physical storage precondition (one chunk beyond the output reserve),
                    //    so ready input chunks cannot consume the output-credit reserve.
                    Node& source = *m_nodes[*m_sourceNode];
                    bool const withinAdmission
                        = m_sourceLiveParticles == 0u
                          || (generated - sunkParticles) < static_cast<std::uint64_t>(m_sourceLiveParticles);
                    if(generated < m_sourceParticles && !source.inflight && withinAdmission
                       && m_pool.freeCount() > reservedTotal)
                    {
                        std::uint64_t const liveParticles = generated - sunkParticles;
                        std::uint64_t budget = m_sourceParticles - generated;
                        std::uint32_t const fillCap
                            = m_sourceLiveParticles == 0u
                                  ? m_pool.chunkCapacity()
                                  : static_cast<std::uint32_t>(
                                        static_cast<std::uint64_t>(m_sourceLiveParticles) - liveParticles);
                        if(budget > fillCap)
                            budget = fillCap;
                        if(budget > m_pool.chunkCapacity())
                            budget = m_pool.chunkCapacity();
                        std::uint32_t const fill = static_cast<std::uint32_t>(budget);
                        ChunkDescriptor chunk{m_pool.acquire(), fill, 0u};
                        Batch const batch{chunk, generated};
                        generated += fill;
                        // Publish the in-flight batch before the hook so a throwing launch is
                        // still owned by the teardown path.
                        source.inflight = batch;
                        source.inputOwned = true;
                        source.hooks.launch(source.queue, batch);
                        source.queue.enqueue(source.event);
                        source.armed = true;
                        progress = true;
                    }

                    // 3) Dispatch ready work. Sinks run first under storage pressure because they
                    //    release chunks without reserving outputs. Otherwise fair round-robin.
                    std::size_t const nodeCount = m_nodes.size();
                    bool dispatched = false;
                    auto tryDispatch = [&](std::size_t index) -> bool
                    {
                        Node& node = *m_nodes[index];
                        // The single in-flight slot is claimed before the ready entry is consumed,
                        // and the entry itself is only popped after a successful launch. A throwing
                        // hook therefore cannot leave an un-armed batch in the slot or let the same
                        // chunk be owned by `ready` and `inflight` at the same time.
                        if(node.ready.empty() || node.inflight)
                            return false;

                        ChunkDescriptor const chunk = node.ready.front();
                        std::uint32_t reserve = 0u;
                        if(node.role == NodeRole::transform)
                        {
                            reserve = static_cast<std::uint32_t>(
                                outputChunkBound(chunk.fill, m_pool.chunkCapacity(), node.bins));
                            if(availableChunks() < reserve)
                                return false; // keep the chunk queued until storage credits exist
                        }

                        Batch const batch{chunk, 0u};
                        // Publish ownership before the hook. The chunk stays at the front of `ready`
                        // until the launch and the event enqueue succeeded, so a failed launch has
                        // exactly one owner (`ready`, released by teardown) instead of two.
                        node.inflight = batch;
                        node.inputOwned = true;
                        try
                        {
                            if(node.role != NodeRole::source)
                            {
                                m_pool.acquireLease(chunk.handle);
                                node.leaseHeld = true;
                            }
                            node.hooks.launch(node.queue, batch);
                            node.queue.enqueue(node.event);
                        }
                        catch(...)
                        {
                            // Roll the reserved slot and the lease back while preserving the queued
                            // chunk. The device may have observed a partially enqueued stage, so
                            // drain the queue before `ready` ownership is released by teardown.
                            try
                            {
                                alpaka::onHost::wait(node.queue);
                            }
                            catch(...)
                            {
                                // Keep the original launch error; teardown releases `ready` anyway.
                            }
                            if(node.leaseHeld)
                            {
                                node.leaseHeld = false;
                                try
                                {
                                    m_pool.releaseLease(chunk.handle);
                                }
                                catch(...)
                                {
                                }
                            }
                            node.inputOwned = false;
                            node.inflight.reset();
                            throw;
                        }
                        node.ready.pop_front();
                        node.reserved = reserve;
                        reservedTotal += reserve;
                        if(reservedTotal > stats.peakReservedChunks)
                            stats.peakReservedChunks = reservedTotal;
                        node.armed = true;
                        return true;
                    };

                    // Sinks free storage without reserving output chunks, so they always run
                    // before transforms under any remaining pressure.
                    for(std::size_t k = 0u; k < nodeCount && !dispatched; ++k)
                    {
                        if(m_nodes[k]->role == NodeRole::sink)
                            dispatched = tryDispatch(k);
                    }
                    if(!dispatched)
                    {
                        for(std::size_t k = 0u; k < nodeCount; ++k)
                        {
                            std::size_t const index = (m_roundRobin + k) % nodeCount;
                            if(tryDispatch(index))
                            {
                                dispatched = true;
                                break;
                            }
                        }
                    }
                    if(dispatched)
                        progress = true;
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
                        if(idleSpins > 200'000'000u || idle > m_stallTimeout)
                            throw std::runtime_error{stallMessage(generated, sunkParticles, reservedTotal)};
                    }
                    else
                    {
                        idleSpins = 0u;
                        lastProgress = std::chrono::steady_clock::now();
                    }
                    ++stats.loopIterations;
                }
            }
            catch(...)
            {
                std::exception_ptr const original = std::current_exception();
                try
                {
                    teardown();
                }
                catch(...)
                {
                    // A teardown failure must not hide the original batch error.
                }
                std::rethrow_exception(original);
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

            stats.poolChunks = m_pool.numChunks();
            stats.peakLiveChunks = m_pool.peakLive();
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
            bool leaseHeld = false;
            bool inputOwned = false;
            std::uint32_t reserved = 0u;
            std::uint32_t completedChunks = 0u;
        };

        [[nodiscard]] std::string stallMessage(
            std::uint64_t generated,
            std::uint64_t sunkParticles,
            std::uint32_t reservedTotal) const
        {
            std::ostringstream message;
            message << "graphFlow: scheduler made no progress for " << m_stallTimeout.count()
                    << " ms (generated=" << generated << ", sunk=" << sunkParticles
                    << ", liveParticles=" << (generated - sunkParticles)
                    << ", free=" << m_pool.freeCount() << ", reserved=" << reservedTotal << ')';
            for(auto const& nodePtr : m_nodes)
            {
                message << "; node '" << nodePtr->name << "' ready=" << nodePtr->ready.size()
                        << " inflight=" << (nodePtr->inflight ? 1 : 0);
            }
            return message.str();
        }

        /** Release every chunk the runner still owns after a hook or routing error.
         *
         * The node that raised has already been removed from `inflight` only on the normal path, so
         * an exception leaves its input owned here. Waiting for the completion event first keeps
         * the device side from touching a recycled chunk.
         */
        void teardown()
        {
            for(auto& nodePtr : m_nodes)
            {
                Node& node = *nodePtr;
                if(node.inflight)
                {
                    if(node.armed)
                        node.queue.waitFor(node.event);
                    else
                        alpaka::onHost::wait(node.queue);
                    node.armed = false;
                    if(node.leaseHeld)
                    {
                        m_pool.releaseLease(node.inflight->chunk.handle);
                        node.leaseHeld = false;
                    }
                    if(node.inputOwned)
                    {
                        m_pool.release(node.inflight->chunk.handle);
                        node.inputOwned = false;
                    }
                    node.inflight.reset();
                    node.reserved = 0u;
                }
                for(ChunkDescriptor const& chunk : node.ready)
                    m_pool.release(chunk.handle);
                node.ready.clear();
            }
        }

        TDevice m_device;
        ChunkPool<TDevice>& m_pool;
        std::vector<std::unique_ptr<Node>> m_nodes;
        std::optional<std::uint32_t> m_sourceNode;
        std::uint64_t m_sourceParticles = 0u;
        std::uint32_t m_sourceLiveParticles = 0u;
        std::size_t m_roundRobin = 0u;
        std::chrono::milliseconds m_stallTimeout{120'000};
    };
} // namespace graphFlow
