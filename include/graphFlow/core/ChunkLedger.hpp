/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace graphFlow::detail
{
    /** Host-side chunk bookkeeping shared by the chunk-storage adapters.
     *
     * The ledger owns the free list, the per-chunk generation and reader-lease counters, and the
     * peak statistics. It deliberately knows nothing about the payload type or its arrays: a
     * storage adapter combines this lifecycle with its own payload memory. `ChunkPool` keeps its
     * historical in-class implementation so existing behavior stays bit-identical; new adapters
     * reuse this ledger instead of copying the free-list logic.
     */
    class ChunkLedger
    {
    public:
        explicit ChunkLedger(std::uint32_t numChunks)
            : m_numChunks{numChunks}
            , m_generations(numChunks, 0u)
            , m_leases(numChunks, 0u)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::ChunkLedger: numChunks must be non-zero"};
            m_free.reserve(numChunks);
            // Deterministic free list order; the last entry is handed out first.
            for(std::uint32_t i = 0u; i < numChunks; ++i)
                m_free.push_back(numChunks - 1u - i);
        }

        [[nodiscard]] std::uint32_t numChunks() const
        {
            return m_numChunks;
        }

        [[nodiscard]] std::uint32_t freeCount() const
        {
            return static_cast<std::uint32_t>(m_free.size());
        }

        [[nodiscard]] std::uint32_t liveCount() const
        {
            return m_numChunks - freeCount();
        }

        /** Pop one chunk from the free list. Throws when the ledger is exhausted. */
        ChunkHandle acquire()
        {
            if(m_free.empty())
                throw std::runtime_error{"graphFlow::ChunkLedger: pool exhausted"};
            std::uint32_t const index = m_free.back();
            m_free.pop_back();
            m_peakLive = liveCount() > m_peakLive ? liveCount() : m_peakLive;
            return ChunkHandle{index, m_generations[index]};
        }

        /** Return a chunk and invalidate all outstanding handles. */
        void release(ChunkHandle handle)
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkLedger: chunk index out of range"};
            if(handle.generation != m_generations[handle.index])
                throw std::invalid_argument{"graphFlow::ChunkLedger: stale chunk handle"};
            if(m_leases[handle.index] != 0u)
                throw std::logic_error{"graphFlow::ChunkLedger: release while a reader lease is outstanding"};
            ++m_generations[handle.index];
            m_free.push_back(handle.index);
        }

        /** Register an asynchronous reader of a chunk. */
        void acquireLease(ChunkHandle handle)
        {
            validateLive(handle);
            ++m_leases[handle.index];
            m_peakLeases = m_leases[handle.index] > m_peakLeases ? m_leases[handle.index] : m_peakLeases;
        }

        void releaseLease(ChunkHandle handle)
        {
            validateLive(handle);
            if(m_leases[handle.index] == 0u)
                throw std::logic_error{"graphFlow::ChunkLedger: lease release without a matching acquire"};
            --m_leases[handle.index];
        }

        [[nodiscard]] std::uint32_t leaseCount(ChunkHandle handle) const
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkLedger: chunk index out of range"};
            return m_leases[handle.index];
        }

        /** Highest number of live (acquired, not released) chunks observed so far. */
        [[nodiscard]] std::uint32_t peakLive() const
        {
            return m_peakLive;
        }

        /** Highest per-chunk reader-lease count observed so far. */
        [[nodiscard]] std::uint32_t peakLeases() const
        {
            return m_peakLeases;
        }

        /** Check that a handle still refers to a live chunk. */
        [[nodiscard]] bool isLive(ChunkHandle handle) const
        {
            return handle.index < m_numChunks && m_generations[handle.index] == handle.generation;
        }

        [[nodiscard]] std::uint32_t generation(std::uint32_t index) const
        {
            return m_generations.at(index);
        }

        void validateLive(ChunkHandle handle) const
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkLedger: chunk index out of range"};
            if(handle.generation != m_generations[handle.index])
                throw std::invalid_argument{"graphFlow::ChunkLedger: stale chunk handle"};
        }

    private:
        std::uint32_t m_numChunks;
        std::vector<std::uint32_t> m_generations;
        std::vector<std::uint32_t> m_leases;
        std::vector<std::uint32_t> m_free;
        std::uint32_t m_peakLive = 0u;
        std::uint32_t m_peakLeases = 0u;
    };
} // namespace graphFlow::detail
