/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/ChunkStorage.hpp"
#include "graphFlow/core/Reservation.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
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

    /** Failure-injection points used by the failure-path tests. */
    enum class FailurePoint
    {
        eventEnqueue,
        leaseAcquire
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
     * one host ready deque; a transform batch may carry up to `batchChunks` ready input chunks, so
     * live device work is bounded by the node count, the batch size and the chunk pool. The
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
     * pool therefore holds P data chunks plus `K*R` output credits for the live transform batches,
     * where `R = outputChunkBound(C, C, B)` and K is the number of transform nodes. The bound is
     * independent of the total particle count N. (`ceil(P/C)` alone is not a safe bound once split
     * routing can fragment particles into one chunk each; see derivePoolSize.)
     *
     * A transform batch reserves the adaptive amount `outputChunkBound(sum(fill), C, B)` for the
     * chunks it actually takes and starts only when `availableChunks()` covers it. A ready list with
     * Q chunks is therefore a cap, not a requirement: under fragmentation the batch shrinks to the
     * largest prefix that fits the current credits, so progress never depends on a full Q batch.
     */
    template<typename TDevice, typename TStorage = ChunkPool<TDevice>>
    class GraphRunner
    {
        static_assert(
            ChunkStorage<TStorage>,
            "graphFlow::GraphRunner: the storage type does not satisfy ChunkStorage");

    public:
        using Queue = ALPAKA_TYPEOF(std::declval<TDevice&>().makeQueue(alpaka::queueKind::nonBlocking));
        using Event = ALPAKA_TYPEOF(std::declval<TDevice&>().makeEvent());
        using Storage = TStorage;

        /** One scheduler dispatch: the input chunks a node owns plus the source id cursor.
         *
         * A transform batch may carry up to `batchChunks` ready input chunks; source and sink
         * batches always carry exactly one. `firstId` is the id of the first particle of the first
         * chunk and is only meaningful for a source batch.
         */
        struct Batch
        {
            std::vector<ChunkDescriptor> chunks{};
            std::uint64_t firstId = 0u;

            [[nodiscard]] ChunkDescriptor const& chunk() const
            {
                if(chunks.empty())
                    throw std::logic_error{"graphFlow: empty batch"};
                return chunks.front();
            }

            /** Total number of particles carried by the batch. */
            [[nodiscard]] std::uint64_t particles() const
            {
                std::uint64_t sum = 0u;
                for(ChunkDescriptor const& chunk : chunks)
                    sum += chunk.fill;
                return sum;
            }
        };

        struct Hooks
        {
            std::function<void(Queue const&, Batch const&)> launch;
            /** Returns true when a further device stage was enqueued. `allocated` reports the number
             * of output chunks physically acquired from the pool during host planning. */
            std::function<bool(Queue const&, Batch const&, std::uint32_t& allocated)> advance;
            std::function<void(Batch const&, std::vector<ChunkDescriptor>&)> finish;
            /** Release resources a batch acquired but never published. The runner calls this
             * exactly once per in-flight node from teardown, after the node queue is drained. */
            std::function<void()> abort;
        };

        /** Test hook: `hook(point, node, item)` runs before the injected operation and may throw.
         *
         * `item` is the 1-based event-enqueue count of the node for `eventEnqueue`, and the 0-based
         * position within the batch for `leaseAcquire`. Production callers leave the hook empty.
         */
        void setFailureHook(std::function<void(FailurePoint, std::uint32_t, std::uint32_t)> hook)
        {
            m_failureHook = std::move(hook);
        }

        GraphRunner(TDevice device, TStorage& storage)
            : m_device{device}
            , m_storage{storage}
        {
        }

        /** Register a node. `route[bin]` is the destination node index for that bin.
         *
         * `maxBins` is the node's partition width: the classifier `maxBins` for a partition node
         * and the route table size. The adaptive output credit is derived from exactly this
         * width, never from the hash table entry count, the tile width, or the batch size Q.
         */
        std::uint32_t addNode(
            std::string name,
            NodeRole role,
            std::uint32_t maxBins,
            std::vector<int> route,
            Hooks hooks)
        {
            if(role == NodeRole::sink)
            {
                if(!route.empty())
                    throw std::invalid_argument{"graphFlow: a sink node has no routes"};
            }
            else if(route.size() != maxBins)
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
                maxBins,
                std::move(route),
                std::move(hooks),
                m_device.makeQueue(alpaka::queueKind::nonBlocking),
                m_device.makeEvent(),
                static_cast<std::uint32_t>(m_nodes.size()));
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

        /** Maximum number of ready input chunks a transform node dispatches as one batch.
         *
         * `1` reproduces the single-chunk dispatch of the earlier scheduler. The value is applied
         * to transform nodes only; source and sink nodes always dispatch one chunk per batch. Every
         * node still has at most one in-flight batch.
         */
        void setBatchChunks(std::uint32_t batchChunks)
        {
            if(batchChunks == 0u)
                throw std::invalid_argument{"graphFlow: batchChunks must be non-zero"};
            m_batchChunks = batchChunks;
        }

        /** Maximum quiet time before the scheduler reports a stall instead of spinning forever. */
        void setStallTimeout(std::chrono::milliseconds timeout)
        {
            m_stallTimeout = timeout;
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
                if(reservedTotal > m_storage.freeCount())
                    throw std::logic_error{"graphFlow: reservation accounting underflow"};
                return m_storage.freeCount() - reservedTotal;
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
                        bool const moreWork = node.hooks.advance(node.queue, batch, allocated);
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
                            enqueueEvent(node);
                        }
                        else
                        {
                            std::vector<ChunkDescriptor> outputs;
                            node.hooks.finish(batch, outputs);
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
                                        if(isInput(batch, outputs[rest].handle))
                                            node.inputOwned = false;
                                        m_storage.release(outputs[rest].handle);
                                    }
                                    throw;
                                }
                            }

                            // A source publishes its input chunks as outputs, so a fully routed
                            // batch has transferred them downstream. Every other node still owns
                            // its input chunks and releases them below. Setting this per routed
                            // output would drop the input release when a later output is
                            // unroutable.
                            if(node.role == NodeRole::source)
                                node.inputOwned = false;

                            if(node.role == NodeRole::sink)
                                sunkParticles += batch.particles();

                            // A source hands its chunk downstream; every other node retires its input
                            // chunks only after the completion event of the last device stage (the
                            // scatter) has fired.
                            if(node.role != NodeRole::source)
                            {
                                if(node.leaseHeld)
                                {
                                    for(ChunkDescriptor const& input : batch.chunks)
                                        m_storage.releaseLease(input.handle);
                                    node.leaseHeld = false;
                                }
                                for(ChunkDescriptor const& input : batch.chunks)
                                    m_storage.release(input.handle);
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
                    std::uint64_t const liveParticles = generated - sunkParticles;
                    std::uint64_t const remaining = m_sourceParticles - generated;
                    std::uint64_t const capacity = m_storage.chunkCapacity();
                    std::uint64_t const configuredLive = m_sourceLiveParticles;
                    // A source chunk must be large enough to amortise one classify/plan/scatter
                    // round trip. With a configured budget P, require min(C, P, remaining)
                    // particles; otherwise wait for the sinks to release the budget. A tail or a
                    // deliberately small P still gets a smaller chunk.
                    std::uint64_t const minFill
                        = configuredLive == 0u
                              ? std::min(capacity, remaining)
                              : std::min(std::min(capacity, configuredLive), remaining);
                    bool const withinAdmission
                        = configuredLive == 0u || liveParticles < configuredLive;
                    bool const enoughBudget
                        = configuredLive == 0u
                          || (liveParticles < configuredLive && (configuredLive - liveParticles) >= minFill);
                    if(generated < m_sourceParticles && !source.inflight && withinAdmission && enoughBudget
                       && m_storage.freeCount() > reservedTotal)
                    {
                        std::uint64_t budget = remaining;
                        std::uint32_t const fillCap
                            = configuredLive == 0u
                                  ? m_storage.chunkCapacity()
                                  : static_cast<std::uint32_t>(configuredLive - liveParticles);
                        if(budget > fillCap)
                            budget = fillCap;
                        if(budget > m_storage.chunkCapacity())
                            budget = m_storage.chunkCapacity();
                        std::uint32_t const fill = static_cast<std::uint32_t>(budget);
                        ChunkDescriptor chunk{m_storage.acquire(), fill, 0u};
                        Batch const batch{{chunk}, generated};
                        generated += fill;
                        // Publish the in-flight batch before the hook so a throwing launch is
                        // still owned by the teardown path.
                        source.inflight = batch;
                        source.inputOwned = true;
                        source.hooks.launch(source.queue, batch);
                        enqueueEvent(source);
                        progress = true;
                    }

                    // 3) Dispatch ready work. Sinks run first under storage pressure because they
                    //    release chunks without reserving outputs. Otherwise fair round-robin.
                    std::size_t const nodeCount = m_nodes.size();
                    bool dispatched = false;
                    auto tryDispatch = [&](std::size_t index) -> bool
                    {
                        Node& node = *m_nodes[index];
                        if(node.ready.empty() || node.inflight)
                            return false;

                        // A transform node batches up to `batchChunks` ready input chunks; source
                        // and sink nodes always take one. The prefix is only removed from `ready`
                        // after the launch and the event enqueue succeeded, so a throwing hook
                        // cannot leave an un-armed batch in the slot or let the same chunk be owned
                        // by `ready` and `inflight` at the same time.
                        bool const batchable = node.role == NodeRole::transform;
                        std::size_t wanted
                            = batchable ? std::min<std::size_t>(m_batchChunks, node.ready.size()) : 1u;
                        std::uint32_t reserve = 0u;
                        if(batchable)
                        {
                            // Start with the largest requested prefix and shrink it until its
                            // adaptive reserve fits the storage credits. Because
                            // outputChunkBound(sum(fill)) is monotone in sum(fill), the fitting
                            // prefixes are contiguous and this finds the largest one; a fragmented
                            // ready list dispatches a smaller batch instead of stalling.
                            std::uint64_t particles = 0u;
                            for(std::size_t i = 0u; i < wanted; ++i)
                                particles += node.ready[i].fill;
                            while(wanted != 0u)
                            {
                                std::uint32_t const bound = checkedReserveBound(particles, node.maxBins);
                                if(availableChunks() >= bound)
                                {
                                    reserve = bound;
                                    break;
                                }
                                --wanted;
                                particles -= node.ready[wanted].fill;
                            }
                            if(wanted == 0u)
                                return false; // keep the batch queued until one chunk fits
                        }
                        std::vector<ChunkDescriptor> selected;
                        selected.reserve(wanted);
                        for(std::size_t i = 0u; i < wanted; ++i)
                            selected.push_back(node.ready[i]);

                        Batch const batch{selected, 0u};
                        node.inflight = batch;
                        node.inputOwned = true;
                        std::size_t leased = 0u;
                        try
                        {
                            if(node.role != NodeRole::source)
                            {
                                for(std::size_t i = 0u; i < selected.size(); ++i)
                                {
                                    if(m_failureHook)
                                        m_failureHook(
                                            FailurePoint::leaseAcquire,
                                            node.id,
                                            static_cast<std::uint32_t>(i));
                                    m_storage.acquireLease(selected[i].handle);
                                    leased = i + 1u;
                                }
                                node.leaseHeld = true;
                            }
                            node.hooks.launch(node.queue, batch);
                            enqueueEvent(node);
                        }
                        catch(...)
                        {
                            // Roll the reserved slot and the leases back while preserving the queued
                            // chunks. `leased` counts the acquisitions that completed, so a throw
                            // in the middle of the loop cannot strand a lease.
                            try
                            {
                                alpaka::onHost::wait(node.queue);
                            }
                            catch(...)
                            {
                                // Keep the original launch error; teardown releases `ready` anyway.
                            }
                            node.leaseHeld = false;
                            for(std::size_t i = 0u; i < leased; ++i)
                            {
                                try
                                {
                                    m_storage.releaseLease(selected[i].handle);
                                }
                                catch(...)
                                {
                                }
                            }
                            node.inputOwned = false;
                            node.inflight.reset();
                            throw;
                        }
                        for(std::size_t i = 0u; i < selected.size(); ++i)
                            node.ready.pop_front();
                        node.reserved = reserve;
                        reservedTotal += reserve;
                        if(reservedTotal > stats.peakReservedChunks)
                            stats.peakReservedChunks = reservedTotal;
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
            if(m_storage.freeCount() != m_storage.numChunks())
                throw std::logic_error{"graphFlow: chunk leak after the run"};

            stats.poolChunks = m_storage.numChunks();
            stats.peakLiveChunks = m_storage.peakLive();
            return stats;
        }

        [[nodiscard]] std::uint32_t nodeCount() const noexcept
        {
            return static_cast<std::uint32_t>(m_nodes.size());
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
                Event nodeEvent,
                std::uint32_t nodeId)
                : name{std::move(nodeName)}
                , role{nodeRole}
                , maxBins{nodeBins}
                , route{std::move(nodeRoute)}
                , hooks{std::move(nodeHooks)}
                , queue{std::move(nodeQueue)}
                , event{std::move(nodeEvent)}
                , id{nodeId}
            {
            }

            std::string name;
            NodeRole role;
            std::uint32_t maxBins;
            std::vector<int> route;
            Hooks hooks;
            Queue queue;
            Event event;
            std::uint32_t id;
            std::deque<ChunkDescriptor> ready;
            std::optional<Batch> inflight;
            bool armed = false;
            bool leaseHeld = false;
            bool inputOwned = false;
            std::uint32_t reserved = 0u;
            std::uint32_t completedChunks = 0u;
            std::uint32_t eventEnqueues = 0u;
        };

        /** Adaptive output-chunk reserve of a transform batch with `particles` input particles.
         *
         * Host planning never acquires more than `outputChunkBound(sum(fill), C, B)`, so reserving
         * exactly this amount is sufficient for the batch. `B` is the node's partition `maxBins`,
         * also for a hash partition. The single-chunk pool bound `K*R` stays in place; a batch
         * that does not fit shrinks its chunk prefix rather than the pool.
         */
        [[nodiscard]] std::uint32_t checkedReserveBound(
            std::uint64_t particles,
            std::uint32_t maxBins) const
        {
            return checkedOutputChunkBound(
                particles,
                m_storage.chunkCapacity(),
                maxBins,
                "graphFlow: transform batch output reserve");
        }

        /** Enqueue the node completion event, with an optional injected failure for tests. */
        void enqueueEvent(Node& node)
        {
            ++node.eventEnqueues;
            if(m_failureHook)
                m_failureHook(FailurePoint::eventEnqueue, node.id, node.eventEnqueues);
            node.queue.enqueue(node.event);
            node.armed = true;
        }

        /** True when `handle` is one of the batch's input chunks. */
        [[nodiscard]] static bool isInput(Batch const& batch, ChunkHandle handle)
        {
            for(ChunkDescriptor const& input : batch.chunks)
            {
                if(input.handle == handle)
                    return true;
            }
            return false;
        }

        [[nodiscard]] std::string stallMessage(
            std::uint64_t generated,
            std::uint64_t sunkParticles,
            std::uint32_t reservedTotal) const
        {
            std::ostringstream message;
            message << "graphFlow: scheduler made no progress for " << m_stallTimeout.count()
                    << " ms (generated=" << generated << ", sunk=" << sunkParticles
                    << ", liveParticles=" << (generated - sunkParticles)
                    << ", free=" << m_storage.freeCount() << ", reserved=" << reservedTotal << ')';
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
                    // A batch that acquired output chunks but never published them (for example
                    // when the completion-event enqueue failed) must hand them back exactly once.
                    if(node.hooks.abort)
                    {
                        try
                        {
                            node.hooks.abort();
                        }
                        catch(...)
                        {
                            // A release failure must not hide the original batch error.
                        }
                    }
                    if(node.leaseHeld)
                    {
                        for(ChunkDescriptor const& input : node.inflight->chunks)
                            m_storage.releaseLease(input.handle);
                        node.leaseHeld = false;
                    }
                    if(node.inputOwned)
                    {
                        for(ChunkDescriptor const& input : node.inflight->chunks)
                            m_storage.release(input.handle);
                        node.inputOwned = false;
                    }
                    node.inflight.reset();
                    node.reserved = 0u;
                }
                for(ChunkDescriptor const& chunk : node.ready)
                    m_storage.release(chunk.handle);
                node.ready.clear();
            }
        }

        TDevice m_device;
        TStorage& m_storage;
        std::vector<std::unique_ptr<Node>> m_nodes;
        std::optional<std::uint32_t> m_sourceNode;
        std::uint64_t m_sourceParticles = 0u;
        std::uint32_t m_sourceLiveParticles = 0u;
        std::size_t m_roundRobin = 0u;
        std::uint32_t m_batchChunks = 1u;
        std::function<void(FailurePoint, std::uint32_t, std::uint32_t)> m_failureHook;
        std::chrono::milliseconds m_stallTimeout{120'000};
    };
} // namespace graphFlow
