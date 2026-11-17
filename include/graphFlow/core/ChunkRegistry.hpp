/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/ChunkStorage.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace graphFlow
{
    /** Node id value used by a chunk record that has no producing or owner node yet. */
    inline constexpr std::uint32_t invalidChunkNode = std::numeric_limits<std::uint32_t>::max();

    /** Host-visible pipeline state of one chunk incarnation.
     *
     * The registry is observability only. `GraphRunner` updates it at the chunk transitions that
     * already exist and never reads it to make a scheduling decision. Phase 1 of the chunk
     * pipeline redesign is the first consumer that reads the live-chunk count from the registry.
     */
    enum class ChunkPhase : std::uint8_t
    {
        free,
        ready,
        inflight,
        reserved,
        retired
    };

    [[nodiscard]] inline char const* chunkPhaseName(ChunkPhase phase) noexcept
    {
        switch(phase)
        {
        case ChunkPhase::free:
            return "free";
        case ChunkPhase::ready:
            return "ready";
        case ChunkPhase::inflight:
            return "inflight";
        case ChunkPhase::reserved:
            return "reserved";
        case ChunkPhase::retired:
            return "retired";
        }
        return "unknown";
    }

    /** State of one physical pool slot for its current incarnation. */
    struct ChunkRecord
    {
        ChunkHandle handle{};
        std::uint32_t producingNode = invalidChunkNode;
        std::uint32_t ownerNode = invalidChunkNode;
        std::int32_t destinationNode = -1;
        ChunkPhase phase = ChunkPhase::free;
        std::uint32_t fill = 0u;
        std::uint32_t bin = 0u;
    };

    /** Aggregate view of the registry: phase occupancy, peaks and transition counters.
     *
     * `liveChunks` counts every incarnation that is still held by the pool, so it includes the
     * transient `retired` state. `faults` is the number of invariant violations the observers
     * recorded; a non-zero value is sticky for the run.
     */
    struct ChunkRegistrySummary
    {
        std::uint32_t chunks = 0u;
        std::uint32_t liveChunks = 0u;
        std::uint32_t freeChunks = 0u;
        std::uint32_t readyChunks = 0u;
        std::uint32_t inflightChunks = 0u;
        std::uint32_t reservedChunks = 0u;
        std::uint32_t retiredChunks = 0u;

        std::uint32_t peakReadyChunks = 0u;
        std::uint32_t peakInflightChunks = 0u;
        std::uint32_t peakReservedChunks = 0u;

        std::uint64_t acquisitions = 0u;
        std::uint64_t releases = 0u;
        std::uint64_t reservations = 0u;
        std::uint64_t reservedCredits = 0u;
        std::uint64_t routings = 0u;
        std::uint64_t dispatches = 0u;
        std::uint64_t retirements = 0u;
        std::uint64_t faults = 0u;
    };

    /** Per-chunk pipeline ledger owned by the scheduler thread.
     *
     * Acquire and release are reported from the storage boundary, so every incarnation is seen,
     * including the output chunks a transform or partition stage acquires while planning. A caller
     * that has no observer support can still drive the record through `onDispatch()` and
     * `onRoute()`; those calls resynchronise a stale record.
     *
     * Every mutator is `noexcept`. A diagnostic must never abort a run: on an invariant violation
     * the registry records the first fault in a fixed-size record and returns without changing the
     * pool. `validate()` and the diagnostics surface report the sticky fault. Building the detail
     * string is the only operation that allocates, and it is not on the observer path.
     */
    class ChunkRegistry
    {
    public:
        ChunkRegistry() = default;

        explicit ChunkRegistry(std::uint32_t numChunks)
        {
            resize(numChunks);
        }

        /** Allocate the record vector. The only allocating operation; called at construction. */
        void resize(std::uint32_t numChunks)
        {
            m_records.assign(numChunks, ChunkRecord{});
            clearFault();
            for(auto& count : m_counts)
                count = 0u;
            for(auto& peak : m_peaks)
                peak = 0u;
            m_counts[phaseIndex(ChunkPhase::free)] = numChunks;
            m_counters = {};
        }

        /** Seed every record from the storage's current generation.
         *
         * This runs in the runner constructor and at the start of every run. It never throws and
         * never inspects liveness: a chunk that is already live when a runner is constructed is
         * left `free` with the pool generation, which is the same state the pre-registry runner
         * observed. No allocation, so it is safe on the run-start path.
         */
        template<typename TStorage>
        void seed(TStorage const& storage) noexcept
        {
            if(m_records.size() != storage.numChunks())
            {
                noteFault(
                    "record count does not match the pool",
                    invalidChunkNode,
                    ChunkPhase::free,
                    ChunkPhase::free);
                return;
            }
            try
            {
                for(std::uint32_t index = 0u; index < m_records.size(); ++index)
                {
                    ChunkRecord& record = m_records[index];
                    record = ChunkRecord{};
                    record.handle = ChunkHandle{index, storage.generation(index)};
                }
            }
            catch(...)
            {
                noteFault("storage generation query failed", invalidChunkNode, ChunkPhase::free, ChunkPhase::free);
                return;
            }
            for(auto& count : m_counts)
                count = 0u;
            m_counts[phaseIndex(ChunkPhase::free)] = static_cast<std::uint32_t>(m_records.size());
            for(auto& peak : m_peaks)
                peak = 0u;
            m_counters = {};
            clearFault();
        }

        /** Reset for a new run without allocating or throwing. */
        template<typename TStorage>
        void reset(TStorage const& storage) noexcept
        {
            seed(storage);
        }

        /** Declare whether the storage reports acquire and release transitions.
         *
         * The runner sets this after installing the storage observers. With observers installed
         * a chunk that reaches dispatch or routing while its record is still `free` means the
         * acquire notification was skipped, which is recorded as a sticky fault. Without observer
         * support the same state is the documented fallback and is resynchronised silently.
         */
        void setObserverInstalled(bool installed) noexcept
        {
            m_observerInstalled = installed;
        }

        /** Report a chunk acquired from the pool. The record enters `reserved`. */
        void onAcquire(ChunkHandle handle) noexcept
        {
            ChunkRecord* record = recordFor(handle.index);
            if(record == nullptr)
                return;
            if(record->phase != ChunkPhase::free)
            {
                noteFault("acquire of a non-free chunk", handle.index, ChunkPhase::free, record->phase);
                return;
            }
            record->handle = handle;
            record->producingNode = invalidChunkNode;
            record->ownerNode = invalidChunkNode;
            record->destinationNode = -1;
            record->fill = 0u;
            record->bin = 0u;
            transition(*record, ChunkPhase::reserved);
            ++m_counters.acquisitions;
        }

        /** Report a released chunk. `handle` carries the post-release generation. */
        void onRelease(ChunkHandle handle) noexcept
        {
            ChunkRecord* record = recordFor(handle.index);
            if(record == nullptr)
                return;
            if(record->phase == ChunkPhase::free)
            {
                noteFault("release of an already free chunk", handle.index, ChunkPhase::retired, record->phase);
                return;
            }
            record->handle = handle;
            record->producingNode = invalidChunkNode;
            record->ownerNode = invalidChunkNode;
            record->destinationNode = -1;
            record->fill = 0u;
            record->bin = 0u;
            transition(*record, ChunkPhase::free);
            ++m_counters.releases;
        }

        /** Report a node publishing a descriptor to `destinationNode = route[bin]`. */
        void onRoute(ChunkDescriptor const& chunk, std::uint32_t producingNode, std::int32_t destinationNode) noexcept
        {
            if(destinationNode < 0)
            {
                noteFault("route to a negative destination", chunk.handle.index, ChunkPhase::ready, ChunkPhase::ready);
                return;
            }
            ChunkRecord* record = synchronise(chunk.handle);
            if(record == nullptr)
                return;
            if(record->producingNode == invalidChunkNode)
                record->producingNode = producingNode;
            record->ownerNode = static_cast<std::uint32_t>(destinationNode);
            record->destinationNode = destinationNode;
            record->fill = chunk.fill;
            record->bin = chunk.bin;
            transition(*record, ChunkPhase::ready);
            ++m_counters.routings;
        }

        /** Report a lane taking ownership of a chunk. */
        void onDispatch(ChunkDescriptor const& chunk, std::uint32_t ownerNode) noexcept
        {
            ChunkRecord* record = synchronise(chunk.handle);
            if(record == nullptr)
                return;
            if(record->phase == ChunkPhase::inflight)
            {
                noteFault("dispatch of an inflight chunk", chunk.handle.index, ChunkPhase::ready, record->phase);
                return;
            }
            if(record->phase == ChunkPhase::ready && record->ownerNode != ownerNode)
            {
                noteFault(
                    "dispatch by a node that does not own the ready chunk",
                    chunk.handle.index,
                    ChunkPhase::ready,
                    record->phase);
                return;
            }
            if(record->producingNode == invalidChunkNode)
                record->producingNode = ownerNode;
            record->ownerNode = ownerNode;
            record->fill = chunk.fill;
            record->bin = chunk.bin;
            transition(*record, ChunkPhase::inflight);
            ++m_counters.dispatches;
        }

        /** Report a sink consuming a chunk, before the release returns it to the pool. */
        void onSinkArrival(ChunkDescriptor const& chunk, std::uint32_t sinkNode) noexcept
        {
            ChunkRecord* record = synchronise(chunk.handle);
            if(record == nullptr)
                return;
            if(record->phase != ChunkPhase::inflight)
            {
                noteFault(
                    "sink arrival of a non-inflight chunk",
                    chunk.handle.index,
                    ChunkPhase::inflight,
                    record->phase);
                return;
            }
            record->ownerNode = sinkNode;
            record->destinationNode = static_cast<std::int32_t>(sinkNode);
            record->fill = chunk.fill;
            record->bin = chunk.bin;
            transition(*record, ChunkPhase::retired);
            ++m_counters.retirements;
        }

        /** Report a non-zero output-credit reservation of one lane.
         *
         * The reservation is a credit count, not a chunk. The node id is kept in the signature for
         * future per-node accounting, but only aggregate credits are stored; a zero-credit
         * dispatch (a sink) is not a reservation and is not counted.
         */
        void onReserve([[maybe_unused]] std::uint32_t node, std::uint32_t credits) noexcept
        {
            if(credits == 0u)
                return;
            ++m_counters.reservations;
            m_counters.reservedCredits += credits;
        }

        [[nodiscard]] std::vector<ChunkRecord> const& records() const noexcept
        {
            return m_records;
        }

        /** Chunks that currently occupy the pipeline: `reserved + ready + inflight`.
         *
         * This is the Phase 1 admission quantity, a count of chunks and never of particles. It
         * deliberately excludes the transient `retired` phase even though `summary().liveChunks`
         * includes it: a retired chunk has already been consumed by a sink and only waits for the
         * release that returns its slot to `free`. Counting it as live work would let the
         * occupancy gate over-admit by the number of chunks a sink retires in one loop iteration.
         * It also excludes `free`, so the value is bounded by the pool size minus the free list.
         */
        [[nodiscard]] std::uint32_t liveWorkChunks() const noexcept
        {
            return m_counts[phaseIndex(ChunkPhase::ready)] + m_counts[phaseIndex(ChunkPhase::inflight)]
                   + m_counts[phaseIndex(ChunkPhase::reserved)];
        }

        [[nodiscard]] ChunkRegistrySummary summary() const noexcept
        {
            ChunkRegistrySummary result = m_counters;
            result.chunks = static_cast<std::uint32_t>(m_records.size());
            result.freeChunks = m_counts[phaseIndex(ChunkPhase::free)];
            result.liveChunks = result.chunks - result.freeChunks;
            result.readyChunks = m_counts[phaseIndex(ChunkPhase::ready)];
            result.inflightChunks = m_counts[phaseIndex(ChunkPhase::inflight)];
            result.reservedChunks = m_counts[phaseIndex(ChunkPhase::reserved)];
            result.retiredChunks = m_counts[phaseIndex(ChunkPhase::retired)];
            result.peakReadyChunks = m_peaks[phaseIndex(ChunkPhase::ready)];
            result.peakInflightChunks = m_peaks[phaseIndex(ChunkPhase::inflight)];
            result.peakReservedChunks = m_peaks[phaseIndex(ChunkPhase::reserved)];
            return result;
        }

        [[nodiscard]] bool hasFault() const noexcept
        {
            return m_fault.set;
        }

        /** Human-readable detail of the first sticky fault. Allocates; diagnostics only. */
        [[nodiscard]] std::string faultDetail() const
        {
            if(!m_fault.set)
                return {};
            std::string detail = "graphFlow: chunk registry fault: ";
            detail += m_fault.reason;
            if(m_fault.index != invalidChunkNode)
            {
                detail += " (chunk " + std::to_string(m_fault.index);
                if(m_fault.expected != m_fault.actual)
                    detail += ", phase " + std::string{chunkPhaseName(m_fault.actual)} + ", expected "
                              + chunkPhaseName(m_fault.expected);
                detail += ')';
            }
            return detail;
        }

        /** Strict consistency check. Throws `std::logic_error` on a sticky fault or the first
         * violated invariant.
         *
         * `sourceNode` is exempt from the inflight-lease requirement: source chunks are handed
         * downstream without a reader lease. Pass `invalidChunkNode` when no source is known.
         */
        template<typename TStorage>
        requires ChunkStorage<TStorage>
        void validate(TStorage const& storage, std::uint32_t sourceNode = invalidChunkNode) const
        {
            if(m_fault.set)
                throw std::logic_error{faultDetail()};
            if(m_records.size() != storage.numChunks())
                throw std::logic_error{
                    "graphFlow: chunk registry has " + std::to_string(m_records.size()) + " records for "
                    + std::to_string(storage.numChunks()) + " pool chunks"};

            std::uint32_t phaseCounts[5] = {};
            for(std::size_t index = 0u; index < m_records.size(); ++index)
            {
                ChunkRecord const& record = m_records[index];
                ++phaseCounts[phaseIndex(record.phase)];
                if(record.handle.index != index)
                    throw std::logic_error{
                        "graphFlow: chunk registry record " + std::to_string(index) + " tracks handle index "
                        + std::to_string(record.handle.index)};
                if(record.handle.generation != storage.generation(static_cast<std::uint32_t>(index)))
                {
                    noteFault(
                        "record generation does not match the pool",
                        static_cast<std::uint32_t>(index),
                        record.phase,
                        record.phase);
                    throw std::logic_error{faultDetail()};
                }

                std::uint32_t const leases = storage.leaseCount(record.handle);
                if(record.phase == ChunkPhase::free && leases != 0u)
                    throw std::logic_error{
                        "graphFlow: chunk registry chunk " + std::to_string(index)
                        + " is free but holds a reader lease"};
                if(record.phase == ChunkPhase::inflight)
                {
                    if(record.ownerNode == sourceNode)
                    {
                        if(leases != 0u)
                            throw std::logic_error{
                                "graphFlow: chunk registry source chunk " + std::to_string(index)
                                + " is inflight with a reader lease"};
                    }
                    else if(leases == 0u)
                    {
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index)
                            + " is inflight without a reader lease"};
                    }
                }

                // Phase and ownership must describe one consistent state. A dropped route leaves
                // an impossible owner/destination pair once the chunk is dispatched, which the
                // counts alone cannot detect.
                if(record.phase == ChunkPhase::free || record.phase == ChunkPhase::reserved)
                {
                    if(record.ownerNode != invalidChunkNode || record.destinationNode >= 0)
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index) + " is "
                            + chunkPhaseName(record.phase) + " but still names an owner or destination"};
                    if(record.fill != 0u)
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index) + " is "
                            + chunkPhaseName(record.phase) + " with a non-zero fill"};
                }
                else
                {
                    if(record.ownerNode == invalidChunkNode)
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index) + " is "
                            + chunkPhaseName(record.phase) + " without an owner"};
                    if(record.fill == 0u || record.fill > storage.chunkCapacity())
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index) + " has fill "
                            + std::to_string(record.fill) + " outside [1, " + std::to_string(storage.chunkCapacity())
                            + ']'};
                }
                if(record.phase == ChunkPhase::ready || record.phase == ChunkPhase::retired)
                {
                    if(record.destinationNode != static_cast<std::int32_t>(record.ownerNode))
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index) + " is "
                            + chunkPhaseName(record.phase) + " at a node other than its destination"};
                }
                if(record.phase == ChunkPhase::inflight)
                {
                    if(record.ownerNode == sourceNode)
                    {
                        if(record.destinationNode >= 0)
                            throw std::logic_error{
                                "graphFlow: chunk registry source chunk " + std::to_string(index)
                                + " is inflight with a destination"};
                    }
                    else if(record.destinationNode != static_cast<std::int32_t>(record.ownerNode))
                    {
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index)
                            + " is inflight at a node other than its destination"};
                    }
                }
            }

            std::uint32_t total = 0u;
            for(std::uint32_t const count : phaseCounts)
                total += count;
            if(total != m_records.size())
                throw std::logic_error{"graphFlow: chunk registry phase counts do not match the record count"};
            if(phaseCounts[phaseIndex(ChunkPhase::free)] != storage.freeCount())
                throw std::logic_error{
                    "graphFlow: chunk registry reports " + std::to_string(phaseCounts[phaseIndex(ChunkPhase::free)])
                    + " free chunks but the pool reports " + std::to_string(storage.freeCount())};
            if(m_counters.acquisitions < m_counters.releases)
                throw std::logic_error{"graphFlow: chunk registry release count exceeds the acquisition count"};
            if(m_counters.acquisitions - m_counters.releases != storage.liveCount())
                throw std::logic_error{
                    "graphFlow: chunk registry tracks " + std::to_string(m_counters.acquisitions - m_counters.releases)
                    + " live incarnations but the pool reports " + std::to_string(storage.liveCount())};
        }

    private:
        struct Fault
        {
            bool set = false;
            char const* reason = "";
            std::uint32_t index = invalidChunkNode;
            ChunkPhase expected = ChunkPhase::free;
            ChunkPhase actual = ChunkPhase::free;
        };

        static constexpr std::size_t phaseIndex(ChunkPhase phase) noexcept
        {
            return static_cast<std::size_t>(phase);
        }

        void noteFault(char const* reason, std::uint32_t index, ChunkPhase expected, ChunkPhase actual) const noexcept
        {
            ++m_counters.faults;
            if(m_fault.set)
                return;
            m_fault.set = true;
            m_fault.reason = reason;
            m_fault.index = index;
            m_fault.expected = expected;
            m_fault.actual = actual;
        }

        void clearFault() noexcept
        {
            m_fault = Fault{};
        }

        ChunkRecord* recordFor(std::uint32_t index) noexcept
        {
            if(index >= m_records.size())
            {
                noteFault("handle index out of range", index, ChunkPhase::free, ChunkPhase::free);
                return nullptr;
            }
            return &m_records[index];
        }

        /** Bring a record in line with a handle whose acquire was not observed. */
        ChunkRecord* synchronise(ChunkHandle handle) noexcept
        {
            ChunkRecord* record = recordFor(handle.index);
            if(record == nullptr)
                return nullptr;
            if(record->handle.generation != handle.generation || record->phase == ChunkPhase::free)
            {
                if(record->phase == ChunkPhase::free && m_observerInstalled)
                    noteFault("acquire notification was not observed", handle.index, ChunkPhase::free, record->phase);
                record->handle = handle;
                record->producingNode = invalidChunkNode;
                record->ownerNode = invalidChunkNode;
                record->destinationNode = -1;
                record->fill = 0u;
                record->bin = 0u;
                transition(*record, ChunkPhase::reserved);
                ++m_counters.acquisitions;
            }
            return record;
        }

        void transition(ChunkRecord& record, ChunkPhase next) noexcept
        {
            if(record.phase == next)
                return;
            --m_counts[phaseIndex(record.phase)];
            ++m_counts[phaseIndex(next)];
            record.phase = next;
            if(next == ChunkPhase::ready)
                m_peaks[phaseIndex(ChunkPhase::ready)]
                    = std::max(m_peaks[phaseIndex(ChunkPhase::ready)], m_counts[phaseIndex(ChunkPhase::ready)]);
            if(next == ChunkPhase::inflight)
                m_peaks[phaseIndex(ChunkPhase::inflight)]
                    = std::max(m_peaks[phaseIndex(ChunkPhase::inflight)], m_counts[phaseIndex(ChunkPhase::inflight)]);
            if(next == ChunkPhase::reserved)
                m_peaks[phaseIndex(ChunkPhase::reserved)]
                    = std::max(m_peaks[phaseIndex(ChunkPhase::reserved)], m_counts[phaseIndex(ChunkPhase::reserved)]);
        }

        std::vector<ChunkRecord> m_records;
        std::uint32_t m_counts[5] = {};
        std::uint32_t m_peaks[5] = {};
        mutable ChunkRegistrySummary m_counters;
        mutable Fault m_fault;
        bool m_observerInstalled = false;
    };
} // namespace graphFlow
