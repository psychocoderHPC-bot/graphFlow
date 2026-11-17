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
     * pipeline redesign is the first consumer that reads the live-work count from the registry.
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
     * The counts are maintained incrementally by the registry transitions, so a summary is O(1)
     * and never scans the records.
     */
    struct ChunkRegistrySummary
    {
        std::uint32_t chunks = 0u;
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
    };

    /** Per-chunk pipeline ledger owned by the scheduler thread.
     *
     * Acquire and release are reported from the storage boundary, so every incarnation is seen,
     * including the output chunks a transform or partition stage acquires while planning. A caller
     * that has no observer support can still drive the record through `onDispatch()` and
     * `onRoute()`; those calls resynchronise a stale record.
     */
    class ChunkRegistry
    {
    public:
        ChunkRegistry() = default;

        explicit ChunkRegistry(std::uint32_t numChunks)
        {
            resize(numChunks);
        }

        void resize(std::uint32_t numChunks)
        {
            m_records.assign(numChunks, ChunkRecord{});
            for(auto& count : m_counts)
                count = 0u;
            for(auto& peak : m_peaks)
                peak = 0u;
            m_counts[phaseIndex(ChunkPhase::free)] = numChunks;
            m_counters = {};
        }

        /** Reset for a new run. The storage must be fully free, which every runner run guarantees.
         *
         * Records take the pool's current generation so the generation invariant holds from the
         * first transition even after an earlier run bumped the counters.
         */
        template<typename TStorage>
        void reset(TStorage const& storage)
        {
            if(storage.liveCount() != 0u)
                throw std::logic_error{"graphFlow: chunk registry cannot reset while chunks are live"};
            resize(storage.numChunks());
            for(std::uint32_t index = 0u; index < m_records.size(); ++index)
                m_records[index].handle = ChunkHandle{index, storage.generation(index)};
        }

        /** Report a chunk acquired from the pool. The record enters `reserved`. */
        void onAcquire(ChunkHandle handle)
        {
            ChunkRecord& record = at(handle);
            if(record.phase != ChunkPhase::free)
                throw std::logic_error{
                    "graphFlow: chunk registry acquired chunk " + std::to_string(handle.index) + " while it was "
                    + chunkPhaseName(record.phase)};
            record.handle = handle;
            record.producingNode = invalidChunkNode;
            record.ownerNode = invalidChunkNode;
            record.destinationNode = -1;
            record.fill = 0u;
            record.bin = 0u;
            transition(record, ChunkPhase::reserved);
            ++m_counters.acquisitions;
        }

        /** Report a released chunk. `handle` carries the post-release generation. */
        void onRelease(ChunkHandle handle)
        {
            ChunkRecord& record = at(handle);
            record.handle = handle;
            record.producingNode = invalidChunkNode;
            record.ownerNode = invalidChunkNode;
            record.destinationNode = -1;
            record.fill = 0u;
            record.bin = 0u;
            transition(record, ChunkPhase::free);
            ++m_counters.releases;
        }

        /** Report a node publishing a descriptor to `destinationNode = route[bin]`. */
        void onRoute(ChunkDescriptor const& chunk, std::uint32_t producingNode, std::int32_t destinationNode)
        {
            if(destinationNode < 0)
                throw std::logic_error{"graphFlow: chunk registry routed a chunk to a negative destination"};
            ChunkRecord& record = synchronise(chunk.handle);
            if(record.producingNode == invalidChunkNode)
                record.producingNode = producingNode;
            record.ownerNode = static_cast<std::uint32_t>(destinationNode);
            record.destinationNode = destinationNode;
            record.fill = chunk.fill;
            record.bin = chunk.bin;
            transition(record, ChunkPhase::ready);
            ++m_counters.routings;
        }

        /** Report a lane taking ownership of a chunk. */
        void onDispatch(ChunkDescriptor const& chunk, std::uint32_t ownerNode)
        {
            ChunkRecord& record = synchronise(chunk.handle);
            if(record.producingNode == invalidChunkNode)
                record.producingNode = ownerNode;
            record.ownerNode = ownerNode;
            record.fill = chunk.fill;
            record.bin = chunk.bin;
            transition(record, ChunkPhase::inflight);
            ++m_counters.dispatches;
        }

        /** Report a sink consuming a chunk, before the release returns it to the pool. */
        void onSinkArrival(ChunkDescriptor const& chunk, std::uint32_t sinkNode)
        {
            ChunkRecord& record = synchronise(chunk.handle);
            record.ownerNode = sinkNode;
            record.destinationNode = static_cast<std::int32_t>(sinkNode);
            record.fill = chunk.fill;
            record.bin = chunk.bin;
            transition(record, ChunkPhase::retired);
            ++m_counters.retirements;
        }

        /** Report the output-credit reservation of one lane. This is a credit count, not a chunk. */
        void onReserve([[maybe_unused]] std::uint32_t node, std::uint32_t credits) noexcept
        {
            ++m_counters.reservations;
            m_counters.reservedCredits += credits;
        }

        [[nodiscard]] std::vector<ChunkRecord> const& records() const noexcept
        {
            return m_records;
        }

        [[nodiscard]] ChunkRegistrySummary summary() const noexcept
        {
            ChunkRegistrySummary result = m_counters;
            result.chunks = static_cast<std::uint32_t>(m_records.size());
            result.freeChunks = m_counts[phaseIndex(ChunkPhase::free)];
            result.readyChunks = m_counts[phaseIndex(ChunkPhase::ready)];
            result.inflightChunks = m_counts[phaseIndex(ChunkPhase::inflight)];
            result.reservedChunks = m_counts[phaseIndex(ChunkPhase::reserved)];
            result.retiredChunks = m_counts[phaseIndex(ChunkPhase::retired)];
            result.peakReadyChunks = m_peaks[phaseIndex(ChunkPhase::ready)];
            result.peakInflightChunks = m_peaks[phaseIndex(ChunkPhase::inflight)];
            result.peakReservedChunks = m_peaks[phaseIndex(ChunkPhase::reserved)];
            return result;
        }

        /** Strict consistency check. Throws `std::logic_error` on the first violation.
         *
         * `sourceNode` is exempt from the inflight-lease requirement: source chunks are handed
         * downstream without a reader lease. Pass `invalidChunkNode` when no source is known.
         */
        template<typename TStorage>
        requires ChunkStorage<TStorage>
        void validate(TStorage const& storage, std::uint32_t sourceNode = invalidChunkNode) const
        {
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
                    throw std::logic_error{
                        "graphFlow: chunk registry record " + std::to_string(index) + " generation "
                        + std::to_string(record.handle.generation) + " does not match the pool generation "
                        + std::to_string(storage.generation(static_cast<std::uint32_t>(index)))};

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
                if(record.phase == ChunkPhase::ready || record.phase == ChunkPhase::inflight
                   || record.phase == ChunkPhase::retired)
                {
                    if(record.fill == 0u || record.fill > storage.chunkCapacity())
                        throw std::logic_error{
                            "graphFlow: chunk registry chunk " + std::to_string(index) + " has fill "
                            + std::to_string(record.fill) + " outside [1, " + std::to_string(storage.chunkCapacity())
                            + ']'};
                }
                if((record.phase == ChunkPhase::ready || record.phase == ChunkPhase::inflight)
                   && record.ownerNode != sourceNode && record.destinationNode < 0)
                {
                    throw std::logic_error{
                        "graphFlow: chunk registry chunk " + std::to_string(index)
                        + " is ready or inflight without a destination"};
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
        static constexpr std::size_t phaseIndex(ChunkPhase phase) noexcept
        {
            return static_cast<std::size_t>(phase);
        }

        ChunkRecord& at(ChunkHandle handle)
        {
            if(handle.index >= m_records.size())
                throw std::logic_error{
                    "graphFlow: chunk registry handle index " + std::to_string(handle.index) + " out of range"};
            return m_records[handle.index];
        }

        /** Bring a record in line with a handle whose acquire was not observed. */
        ChunkRecord& synchronise(ChunkHandle handle)
        {
            ChunkRecord& record = at(handle);
            if(record.handle.generation != handle.generation || record.phase == ChunkPhase::free)
            {
                record.handle = handle;
                record.producingNode = invalidChunkNode;
                record.ownerNode = invalidChunkNode;
                record.destinationNode = -1;
                record.fill = 0u;
                record.bin = 0u;
                transition(record, ChunkPhase::reserved);
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
        ChunkRegistrySummary m_counters;
    };
} // namespace graphFlow
