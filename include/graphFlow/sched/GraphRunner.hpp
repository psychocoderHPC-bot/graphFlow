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

    /** Cheap aggregate scheduler counters and peaks of one `GraphRunner::run()`.
     *
     * The peaks are updated from the existing per-node loops, never from a per-particle or
     * per-chunk scan. They are diagnostic only and do not influence scheduling.
     */
    struct SchedulerStats
    {
        std::uint64_t completedBatches = 0u;
        std::uint64_t loopIterations = 0u;
        std::uint64_t yields = 0u;
        std::uint32_t poolChunks = 0u;
        std::uint32_t peakLiveChunks = 0u;
        std::uint32_t peakReservedChunks = 0u;
        std::uint32_t peakReadyChunks = 0u;
        std::uint32_t peakReadyNodes = 0u;
        std::uint32_t maxReadyDepth = 0u;
        std::uint32_t peakInflightBatches = 0u;
        std::uint32_t peakInflightChunks = 0u;
        std::uint32_t peakEventSlotsInUse = 0u;
        std::uint64_t sourceParticles = 0u;
        std::uint64_t sourceGenerated = 0u;
        std::uint64_t sourceSunk = 0u;
        std::uint64_t sourceLiveParticles = 0u;
        std::uint32_t chunkCapacity = 0u;
    };

    /** One-line aggregate key=value rendering of `SchedulerStats`.
     *
     * Examples print this after a run. The string contains counters only, never topology.
     */
    [[nodiscard]] inline std::string formatSchedulerStats(SchedulerStats const& stats)
    {
        std::ostringstream message;
        message << "completedBatches=" << stats.completedBatches << " loopIterations=" << stats.loopIterations
                << " yields=" << stats.yields << " poolChunks=" << stats.poolChunks
                << " peakLiveChunks=" << stats.peakLiveChunks << " peakReservedChunks=" << stats.peakReservedChunks
                << " peakReadyChunks=" << stats.peakReadyChunks << " peakReadyNodes=" << stats.peakReadyNodes
                << " maxReadyDepth=" << stats.maxReadyDepth << " peakInflightBatches=" << stats.peakInflightBatches
                << " peakInflightChunks=" << stats.peakInflightChunks
                << " peakEventSlotsInUse=" << stats.peakEventSlotsInUse << " sourceParticles=" << stats.sourceParticles
                << " sourceGenerated=" << stats.sourceGenerated << " sourceSunk=" << stats.sourceSunk
                << " sourceLiveParticles=" << stats.sourceLiveParticles << " chunkCapacity=" << stats.chunkCapacity;
        return message.str();
    }

    /** Point-in-time aggregate state plus the peak counters of one scheduler run.
     *
     * The snapshot deliberately contains no node names, node ids, or route tables. Per-node
     * detail stays behind `GraphRunner::setDiagnostics()` and its optional detail callback so a
     * stall message can be logged or compared without leaking the graph topology.
     */
    struct SchedulerSnapshot
    {
        // Aggregate state at capture time.
        std::uint32_t readyChunks = 0u;
        std::uint32_t readyNodes = 0u;
        std::uint32_t readyDepth = 0u;
        std::uint32_t inflightBatches = 0u;
        std::uint32_t inflightChunks = 0u;
        std::uint32_t reservedChunks = 0u;
        std::uint32_t eventSlotsInUse = 0u;
        std::uint32_t freeChunks = 0u;

        // Peaks since the run started.
        std::uint32_t poolChunks = 0u;
        std::uint32_t peakLiveChunks = 0u;
        std::uint32_t peakReservedChunks = 0u;
        std::uint32_t peakReadyChunks = 0u;
        std::uint32_t peakReadyNodes = 0u;
        std::uint32_t maxReadyDepth = 0u;
        std::uint32_t peakInflightBatches = 0u;
        std::uint32_t peakInflightChunks = 0u;
        std::uint32_t peakEventSlotsInUse = 0u;

        // Source progress and storage shape.
        std::uint64_t sourceParticles = 0u;
        std::uint64_t sourceGenerated = 0u;
        std::uint64_t sourceSunk = 0u;
        std::uint64_t sourceLiveParticles = 0u;
        std::uint32_t chunkCapacity = 0u;

        // Run counters.
        std::uint64_t completedBatches = 0u;
        std::uint64_t loopIterations = 0u;
        std::uint64_t yields = 0u;
        std::uint64_t stallTimeoutMs = 0u;
    };

    namespace detail
    {
        /** Aggregate key=value rendering of one snapshot.
         *
         * The output contains only aggregate counters. Node names, node ids, and routes never
         * enter this string; they are available through the diagnostics detail callback.
         */
        inline std::string formatSchedulerSnapshot(SchedulerSnapshot const& snapshot)
        {
            std::ostringstream message;
            message << "graphFlow: scheduler stalled"
                    << " (timeoutMs=" << snapshot.stallTimeoutMs << ", sourceParticles=" << snapshot.sourceParticles
                    << ", sourceGenerated=" << snapshot.sourceGenerated << ", sourceSunk=" << snapshot.sourceSunk
                    << ", sourceLiveParticles=" << snapshot.sourceLiveParticles
                    << ", chunkCapacity=" << snapshot.chunkCapacity << ", poolChunks=" << snapshot.poolChunks
                    << ", freeChunks=" << snapshot.freeChunks << ", reservedChunks=" << snapshot.reservedChunks
                    << ", readyChunks=" << snapshot.readyChunks << ", readyNodes=" << snapshot.readyNodes
                    << ", readyDepth=" << snapshot.readyDepth << ", inflightBatches=" << snapshot.inflightBatches
                    << ", inflightChunks=" << snapshot.inflightChunks
                    << ", eventSlotsInUse=" << snapshot.eventSlotsInUse
                    << ", completedBatches=" << snapshot.completedBatches
                    << ", loopIterations=" << snapshot.loopIterations << ", yields=" << snapshot.yields
                    << ", peakLiveChunks=" << snapshot.peakLiveChunks
                    << ", peakReservedChunks=" << snapshot.peakReservedChunks
                    << ", peakReadyChunks=" << snapshot.peakReadyChunks
                    << ", peakReadyNodes=" << snapshot.peakReadyNodes << ", maxReadyDepth=" << snapshot.maxReadyDepth
                    << ", peakInflightBatches=" << snapshot.peakInflightBatches
                    << ", peakInflightChunks=" << snapshot.peakInflightChunks
                    << ", peakEventSlotsInUse=" << snapshot.peakEventSlotsInUse << ')';
            return message.str();
        }
    } // namespace detail

    /** Stall error carrying the aggregate snapshot and an optional topology-free detail string.
     *
     * `what()` is the aggregate snapshot rendered as key=value pairs and never contains node
     * names, node ids, or routes. Per-node detail is exposed through `detail()` only and is
     * populated when diagnostics are enabled.
     */
    class SchedulerStallError : public std::runtime_error
    {
    public:
        SchedulerStallError(SchedulerSnapshot snapshot, std::string detail = {})
            : std::runtime_error{detail::formatSchedulerSnapshot(snapshot)}
            , m_snapshot{snapshot}
            , m_detail{std::move(detail)}
        {
        }

        [[nodiscard]] SchedulerSnapshot const& snapshot() const noexcept
        {
            return m_snapshot;
        }

        [[nodiscard]] std::string const& detail() const noexcept
        {
            return m_detail;
        }

    private:
        SchedulerSnapshot m_snapshot;
        std::string m_detail;
    };

    /** Host-side dataflow scheduler.
     *
     * One non-blocking queue per node and one event per submitted-depth lane. A node has one host
     * ready deque and up to `depth` in-flight batches; a transform batch may carry up to
     * `batchChunks` ready input chunks, so live device work is bounded by the total lane count, the
     * batch size and the chunk pool. The depth defaults to one and preserves the historical
     * one-batch-per-node scheduling. The
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

        GraphRunner(TDevice device, TStorage& storage) : m_device{device}, m_storage{storage}
        {
        }

        /** Builds the hooks of one submitted-depth lane of a node. */
        using HooksFactory = std::function<Hooks(std::uint32_t)>;

        /** Register a node. `route[bin]` is the destination node index for that bin.
         *
         * `maxBins` is the node's partition width: the classifier `maxBins` for a partition node
         * and the route table size. The adaptive output credit is derived from exactly this
         * width, never from the hash table entry count, the tile width, or the batch size Q.
         *
         * `depth` is the number of concurrently in-flight batches of this node. The default of one
         * preserves the historical one-batch-per-node scheduling. A depth above one replicates the
         * same hooks over every lane, so a caller that passes stateful hooks must use the factory
         * overload and supply one independent hook set per lane.
         */
        std::uint32_t addNode(
            std::string name,
            NodeRole role,
            std::uint32_t maxBins,
            std::vector<int> route,
            Hooks hooks,
            std::uint32_t depth = 1u)
        {
            return addNode(
                std::move(name),
                role,
                maxBins,
                std::move(route),
                HooksFactory{[hooks = std::move(hooks)](std::uint32_t) { return hooks; }},
                depth);
        }

        /** Register a node and create one independent hook set per submitted-depth lane. */
        std::uint32_t addNode(
            std::string name,
            NodeRole role,
            std::uint32_t maxBins,
            std::vector<int> route,
            HooksFactory makeHooks,
            std::uint32_t depth)
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
            if(depth == 0u)
                throw std::invalid_argument{"graphFlow: node depth must be non-zero"};

            std::vector<Hooks> laneHooks;
            std::vector<Event> laneEvents;
            laneHooks.reserve(depth);
            laneEvents.reserve(depth);
            for(std::uint32_t lane = 0u; lane < depth; ++lane)
            {
                laneHooks.push_back(makeHooks(lane));
                laneEvents.push_back(m_device.makeEvent());
            }
            auto node = std::make_unique<Node>(
                std::move(name),
                role,
                maxBins,
                std::move(route),
                std::move(laneHooks),
                std::move(laneEvents),
                m_device.makeQueue(alpaka::queueKind::nonBlocking),
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
         * to transform nodes only; source and sink nodes always dispatch one chunk per batch. The
         * per-node in-flight depth is configured separately at `addNode()`.
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

        /** Opt in to per-node stall detail.
         *
         * Aggregate counters are always available in `SchedulerStallError::what()` and in
         * `snapshot()`. Node names and per-node ready/in-flight depth are only collected when
         * diagnostics are enabled and stay out of `what()`.
         */
        void setDiagnostics(bool enabled)
        {
            m_diagnostics = enabled;
        }

        /** Receives the optional per-node stall detail before the stall error is thrown. */
        void setStallDetailCallback(std::function<void(std::string const&)> callback)
        {
            m_stallDetailCallback = std::move(callback);
        }

        /** Aggregate state of the scheduler.
         *
         * The snapshot is built on demand only; the steady-state loop updates cheap counters and
         * never formats a string. Calling this after a run returns the completed-run peaks with
         * empty ready/in-flight aggregates.
         */
        [[nodiscard]] SchedulerSnapshot snapshot() const
        {
            return buildSnapshot();
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

            m_stats = SchedulerStats{};
            m_generated = 0u;
            m_sunkParticles = 0u;
            m_reservedTotal = 0u;
            m_roundRobin = 0u;
            SchedulerStats& stats = m_stats;
            std::uint64_t& generated = m_generated;
            std::uint64_t& sunkParticles = m_sunkParticles;
            std::uint32_t& reservedTotal = m_reservedTotal;
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

                    // 1) Retire completed events. Each node has up to `depth` lanes, so every
                    //    lane owns its batch, event, leases and output credit independently.
                    for(auto& nodePtr : m_nodes)
                    {
                        Node& node = *nodePtr;
                        for(Lane& lane : node.lanes)
                        {
                            if(!lane.armed || !lane.event.isComplete())
                                continue;
                            lane.armed = false;
                            Batch const batch = *lane.inflight;

                            std::uint32_t allocated = 0u;
                            bool const moreWork = lane.hooks.advance(node.queue, batch, allocated);
                            if(allocated != 0u)
                            {
                                if(allocated > lane.reserved)
                                    throw std::logic_error{"graphFlow: node acquired more chunks than reserved"};
                                reservedTotal -= lane.reserved;
                                lane.reserved = 0u;
                            }
                            else if(!moreWork && lane.reserved != 0u)
                            {
                                // A finished batch that produced no output releases its credit.
                                reservedTotal -= lane.reserved;
                                lane.reserved = 0u;
                            }

                            if(moreWork)
                            {
                                enqueueEvent(node, lane);
                            }
                            else
                            {
                                std::vector<ChunkDescriptor> outputs;
                                lane.hooks.finish(batch, outputs);
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
                                        // Do not leak outputs that were not published; the input
                                        // is released by teardown because `inflight` is still set.
                                        for(std::size_t rest = o; rest < outputs.size(); ++rest)
                                        {
                                            if(isInput(batch, outputs[rest].handle))
                                                lane.inputOwned = false;
                                            m_storage.release(outputs[rest].handle);
                                        }
                                        throw;
                                    }
                                }

                                // A source publishes its input chunks as outputs, so a fully
                                // routed batch has transferred them downstream. Every other node
                                // still owns its input chunks and releases them below. Setting
                                // this per routed output would drop the input release when a
                                // later output is unroutable.
                                if(node.role == NodeRole::source)
                                    lane.inputOwned = false;

                                if(node.role == NodeRole::sink)
                                    sunkParticles += batch.particles();

                                // A source hands its chunk downstream; every other node retires
                                // its input chunks only after the completion event of the last
                                // device stage (the scatter) has fired.
                                if(node.role != NodeRole::source)
                                {
                                    if(lane.leaseHeld)
                                    {
                                        for(ChunkDescriptor const& input : batch.chunks)
                                            m_storage.releaseLease(input.handle);
                                        lane.leaseHeld = false;
                                    }
                                    for(ChunkDescriptor const& input : batch.chunks)
                                        m_storage.release(input.handle);
                                    lane.inputOwned = false;
                                }
                                lane.inflight.reset();
                            }
                            progress = true;
                        }
                    }
                    observePeaks(stats);

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
                    std::uint64_t const minFill = configuredLive == 0u
                                                      ? std::min(capacity, remaining)
                                                      : std::min(std::min(capacity, configuredLive), remaining);
                    bool const withinAdmission = configuredLive == 0u || liveParticles < configuredLive;
                    bool const enoughBudget
                        = configuredLive == 0u
                          || (liveParticles < configuredLive && (configuredLive - liveParticles) >= minFill);
                    Lane* sourceLane = Node::freeLane(source);
                    if(generated < m_sourceParticles && sourceLane != nullptr && withinAdmission && enoughBudget
                       && m_storage.freeCount() > reservedTotal)
                    {
                        std::uint64_t budget = remaining;
                        std::uint32_t const fillCap = configuredLive == 0u
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
                        sourceLane->inflight = batch;
                        sourceLane->inputOwned = true;
                        sourceLane->hooks.launch(source.queue, batch);
                        enqueueEvent(source, *sourceLane);
                        progress = true;
                    }
                    observePeaks(stats);

                    // 3) Dispatch ready work. Sinks run first under storage pressure because they
                    //    release chunks without reserving outputs. Otherwise fair round-robin.
                    std::size_t const nodeCount = m_nodes.size();
                    bool dispatched = false;
                    auto tryDispatch = [&](std::size_t index) -> bool
                    {
                        Node& node = *m_nodes[index];
                        if(node.ready.empty())
                            return false;
                        Lane* lane = Node::freeLane(node);
                        if(lane == nullptr)
                            return false;

                        // A transform node batches up to `batchChunks` ready input chunks; source
                        // and sink nodes always take one. The prefix is only removed from `ready`
                        // after the launch and the event enqueue succeeded, so a throwing hook
                        // cannot leave an un-armed batch in the slot or let the same chunk be owned
                        // by `ready` and `inflight` at the same time.
                        bool const batchable = node.role == NodeRole::transform;
                        std::size_t wanted = batchable ? std::min<std::size_t>(m_batchChunks, node.ready.size()) : 1u;
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
                        lane->inflight = batch;
                        lane->inputOwned = true;
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
                                lane->leaseHeld = true;
                            }
                            lane->hooks.launch(node.queue, batch);
                            enqueueEvent(node, *lane);
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
                            lane->leaseHeld = false;
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
                            lane->inputOwned = false;
                            lane->inflight.reset();
                            throw;
                        }
                        for(std::size_t i = 0u; i < selected.size(); ++i)
                            node.ready.pop_front();
                        lane->reserved = reserve;
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
                    observePeaks(stats);

                    // 4) Termination: source exhausted, every ready list and in-flight batch empty,
                    //    and no output-chunk reservation outstanding.
                    bool const sourceExhausted = generated >= m_sourceParticles;
                    bool drained = true;
                    for(auto const& nodePtr : m_nodes)
                    {
                        bool laneInflight = false;
                        for(Lane const& lane : nodePtr->lanes)
                            laneInflight = laneInflight || lane.inflight.has_value();
                        if(laneInflight || !nodePtr->ready.empty())
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
                            throw stallError();
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
                for(Lane& lane : nodePtr->lanes)
                {
                    if(lane.armed)
                        nodePtr->queue.waitFor(lane.event);
                    lane.armed = false;
                }
                alpaka::onHost::wait(nodePtr->queue);
            }
            if(m_storage.freeCount() != m_storage.numChunks())
                throw std::logic_error{"graphFlow: chunk leak after the run"};

            stats.poolChunks = m_storage.numChunks();
            stats.peakLiveChunks = m_storage.peakLive();
            stats.sourceParticles = m_sourceParticles;
            stats.sourceGenerated = generated;
            stats.sourceSunk = sunkParticles;
            stats.sourceLiveParticles = generated - sunkParticles;
            stats.chunkCapacity = m_storage.chunkCapacity();
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
        /** One submitted-depth slot of a node: its own hooks, event and batch ownership. */
        struct Lane
        {
            Lane(Hooks laneHooks, Event laneEvent) : hooks{std::move(laneHooks)}, event{std::move(laneEvent)}
            {
            }

            Hooks hooks;
            Event event;
            std::optional<Batch> inflight;
            bool armed = false;
            bool leaseHeld = false;
            bool inputOwned = false;
            std::uint32_t reserved = 0u;
        };

        struct Node
        {
            Node(
                std::string nodeName,
                NodeRole nodeRole,
                std::uint32_t nodeBins,
                std::vector<int> nodeRoute,
                std::vector<Hooks> nodeHooks,
                std::vector<Event> nodeEvents,
                Queue nodeQueue,
                std::uint32_t nodeId)
                : name{std::move(nodeName)}
                , role{nodeRole}
                , maxBins{nodeBins}
                , route{std::move(nodeRoute)}
                , queue{std::move(nodeQueue)}
                , id{nodeId}
            {
                if(nodeHooks.empty() || nodeHooks.size() != nodeEvents.size())
                    throw std::logic_error{"graphFlow: invalid node depth"};
                lanes.reserve(nodeHooks.size());
                for(std::size_t i = 0u; i < nodeHooks.size(); ++i)
                    lanes.push_back(Lane{std::move(nodeHooks[i]), std::move(nodeEvents[i])});
            }

            /** First lane that does not own a batch, or null when the depth is exhausted. */
            [[nodiscard]] static Lane* freeLane(Node& node)
            {
                for(Lane& lane : node.lanes)
                {
                    if(!lane.inflight)
                        return &lane;
                }
                return nullptr;
            }

            std::string name;
            NodeRole role;
            std::uint32_t maxBins;
            std::vector<int> route;
            Queue queue;
            std::uint32_t id;
            std::deque<ChunkDescriptor> ready;
            std::vector<Lane> lanes;
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
        [[nodiscard]] std::uint32_t checkedReserveBound(std::uint64_t particles, std::uint32_t maxBins) const
        {
            return checkedOutputChunkBound(
                particles,
                m_storage.chunkCapacity(),
                maxBins,
                "graphFlow: transform batch output reserve");
        }

        /** Enqueue one lane's completion event, with an optional injected failure for tests. */
        void enqueueEvent(Node& node, Lane& lane)
        {
            ++node.eventEnqueues;
            if(m_failureHook)
                m_failureHook(FailurePoint::eventEnqueue, node.id, node.eventEnqueues);
            node.queue.enqueue(lane.event);
            lane.armed = true;
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

        /** Fold the current per-node aggregate into the peak counters.
         *
         * This runs at the three points where state changes (retirement, admission, dispatch) and
         * only reads the small node vector. It never inspects chunks or particles.
         */
        void observePeaks(SchedulerStats& stats) const
        {
            std::uint32_t readyChunks = 0u;
            std::uint32_t readyNodes = 0u;
            std::uint32_t maxReadyDepth = 0u;
            std::uint32_t inflightBatches = 0u;
            std::uint32_t inflightChunks = 0u;
            std::uint32_t eventSlots = 0u;
            for(auto const& nodePtr : m_nodes)
            {
                Node const& node = *nodePtr;
                auto const depth = static_cast<std::uint32_t>(node.ready.size());
                readyChunks += depth;
                if(depth != 0u)
                    ++readyNodes;
                maxReadyDepth = std::max(maxReadyDepth, depth);
                for(Lane const& lane : node.lanes)
                {
                    if(lane.inflight)
                    {
                        ++inflightBatches;
                        inflightChunks += static_cast<std::uint32_t>(lane.inflight->chunks.size());
                    }
                    if(lane.armed)
                        ++eventSlots;
                }
            }

            stats.peakReadyChunks = std::max(stats.peakReadyChunks, readyChunks);
            stats.peakReadyNodes = std::max(stats.peakReadyNodes, readyNodes);
            stats.maxReadyDepth = std::max(stats.maxReadyDepth, maxReadyDepth);
            stats.peakInflightBatches = std::max(stats.peakInflightBatches, inflightBatches);
            stats.peakInflightChunks = std::max(stats.peakInflightChunks, inflightChunks);
            stats.peakEventSlotsInUse = std::max(stats.peakEventSlotsInUse, eventSlots);
        }

        /** Build the aggregate snapshot of the current scheduler state and run peaks. */
        [[nodiscard]] SchedulerSnapshot buildSnapshot() const
        {
            SchedulerSnapshot snapshot;
            for(auto const& nodePtr : m_nodes)
            {
                Node const& node = *nodePtr;
                auto const depth = static_cast<std::uint32_t>(node.ready.size());
                snapshot.readyChunks += depth;
                if(depth != 0u)
                    ++snapshot.readyNodes;
                snapshot.readyDepth = std::max(snapshot.readyDepth, depth);
                for(Lane const& lane : node.lanes)
                {
                    if(lane.inflight)
                    {
                        ++snapshot.inflightBatches;
                        snapshot.inflightChunks += static_cast<std::uint32_t>(lane.inflight->chunks.size());
                    }
                    if(lane.armed)
                        ++snapshot.eventSlotsInUse;
                }
            }

            snapshot.reservedChunks = m_reservedTotal;
            snapshot.freeChunks = m_storage.freeCount();
            snapshot.poolChunks = m_storage.numChunks();
            snapshot.peakLiveChunks = m_storage.peakLive();
            snapshot.peakReservedChunks = m_stats.peakReservedChunks;
            snapshot.peakReadyChunks = m_stats.peakReadyChunks;
            snapshot.peakReadyNodes = m_stats.peakReadyNodes;
            snapshot.maxReadyDepth = m_stats.maxReadyDepth;
            snapshot.peakInflightBatches = m_stats.peakInflightBatches;
            snapshot.peakInflightChunks = m_stats.peakInflightChunks;
            snapshot.peakEventSlotsInUse = m_stats.peakEventSlotsInUse;
            snapshot.completedBatches = m_stats.completedBatches;
            snapshot.loopIterations = m_stats.loopIterations;
            snapshot.yields = m_stats.yields;
            snapshot.sourceParticles = m_sourceParticles;
            snapshot.sourceGenerated = m_generated;
            snapshot.sourceSunk = m_sunkParticles;
            snapshot.sourceLiveParticles = m_generated - m_sunkParticles;
            snapshot.chunkCapacity = m_storage.chunkCapacity();
            snapshot.stallTimeoutMs = static_cast<std::uint64_t>(m_stallTimeout.count());
            return snapshot;
        }

        /** Per-node detail, collected only when diagnostics are enabled. */
        [[nodiscard]] std::string diagnosticsDetail() const
        {
            std::ostringstream message;
            bool first = true;
            for(auto const& nodePtr : m_nodes)
            {
                if(!first)
                    message << "; ";
                first = false;
                std::uint32_t laneInflight = 0u;
                for(Lane const& lane : nodePtr->lanes)
                    laneInflight += lane.inflight ? 1u : 0u;
                message << "node '" << nodePtr->name << "' ready=" << nodePtr->ready.size()
                        << " inflight=" << laneInflight;
            }
            return message.str();
        }

        /** Create the stall error and optionally publish the per-node detail first. */
        [[nodiscard]] SchedulerStallError stallError() const
        {
            SchedulerSnapshot const snapshot = buildSnapshot();
            std::string detail;
            if(m_diagnostics)
            {
                detail = diagnosticsDetail();
                if(m_stallDetailCallback)
                    m_stallDetailCallback(detail);
            }
            return SchedulerStallError{snapshot, std::move(detail)};
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
                bool anyInflight = false;
                for(Lane& lane : node.lanes)
                {
                    anyInflight = anyInflight || lane.inflight.has_value();
                    if(lane.inflight && lane.armed)
                        node.queue.waitFor(lane.event);
                }
                if(anyInflight)
                    alpaka::onHost::wait(node.queue);
                for(Lane& lane : node.lanes)
                {
                    if(!lane.inflight)
                        continue;
                    lane.armed = false;
                    // A batch that acquired output chunks but never published them (for example
                    // when the completion-event enqueue failed) must hand them back exactly once.
                    if(lane.hooks.abort)
                    {
                        try
                        {
                            lane.hooks.abort();
                        }
                        catch(...)
                        {
                            // A release failure must not hide the original batch error.
                        }
                    }
                    if(lane.leaseHeld)
                    {
                        for(ChunkDescriptor const& input : lane.inflight->chunks)
                            m_storage.releaseLease(input.handle);
                        lane.leaseHeld = false;
                    }
                    if(lane.inputOwned)
                    {
                        for(ChunkDescriptor const& input : lane.inflight->chunks)
                            m_storage.release(input.handle);
                        lane.inputOwned = false;
                    }
                    lane.inflight.reset();
                    lane.reserved = 0u;
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
        std::function<void(std::string const&)> m_stallDetailCallback;
        std::chrono::milliseconds m_stallTimeout{120000};
        SchedulerStats m_stats;
        std::uint64_t m_generated = 0u;
        std::uint64_t m_sunkParticles = 0u;
        std::uint32_t m_reservedTotal = 0u;
        bool m_diagnostics = false;
    };
} // namespace graphFlow
