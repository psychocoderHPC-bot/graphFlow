/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/PoolBudget.hpp"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace graphFlow::detail
{
    /** Test-only accessor for the private lifecycle counters.
     *
     * The unit tests need to drive a generation or lease counter to its 32-bit limit; production
     * code never refers to this type. It is declared here so the counters can stay private.
     */
    struct ChunkLedgerTestAccess;

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

        /** Optional observation hooks for the chunk registry.
         *
         * The hooks run after a successful acquire and after a successful release, on the thread
         * that owns the ledger. They are empty by default; the scheduler installs them once per
         * run. A hook must not change the lifecycle, so the ledger never reads its value.
         */
        void setChunkObservers(std::function<void(ChunkHandle)> acquired, std::function<void(ChunkHandle)> released)
        {
            m_acquiredObserver = std::move(acquired);
            m_releasedObserver = std::move(released);
        }

        void clearChunkObservers()
        {
            m_acquiredObserver = {};
            m_releasedObserver = {};
        }

        /** Pop one chunk from the free list. Throws when the ledger is exhausted. */
        ChunkHandle acquire()
        {
            if(m_free.empty())
                throw std::runtime_error{"graphFlow::ChunkLedger: pool exhausted"};
            std::uint32_t const index = m_free.back();
            ChunkHandle const handle{index, m_generations[index]};
            // Notify before removing the slot: an observer fault can never leak the slot. The
            // observer is a diagnostic, so its failure must never abort the acquire.
            if(m_acquiredObserver)
            {
                try
                {
                    m_acquiredObserver(handle);
                }
                catch(...)
                {
                    // Observer hooks are diagnostics and must not throw. This catch is
                    // deliberate: a throwing hook must never abort the operation or leak a
                    // slot. The failure is silent and is not counted.
                }
            }
            m_free.pop_back();
            m_peakLive = liveCount() > m_peakLive ? liveCount() : m_peakLive;
            return handle;
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
            m_generations[handle.index]
                = checkedCounterIncrement(m_generations[handle.index], "graphFlow::ChunkLedger generation");
            m_free.push_back(handle.index);
            if(m_releasedObserver)
            {
                try
                {
                    m_releasedObserver(ChunkHandle{handle.index, m_generations[handle.index]});
                }
                catch(...)
                {
                    // Observer hooks are diagnostics and must not throw. This catch is
                    // deliberate: a throwing hook must never abort the operation or leak a
                    // slot. The failure is silent and is not counted.
                }
            }
        }

        /** Register an asynchronous reader of a chunk. */
        void acquireLease(ChunkHandle handle)
        {
            validateLive(handle);
            m_leases[handle.index] = checkedCounterIncrement(m_leases[handle.index], "graphFlow::ChunkLedger lease");
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
        friend struct ChunkLedgerTestAccess;

        std::uint32_t m_numChunks;
        std::vector<std::uint32_t> m_generations;
        std::vector<std::uint32_t> m_leases;
        std::vector<std::uint32_t> m_free;
        std::uint32_t m_peakLive = 0u;
        std::uint32_t m_peakLeases = 0u;
        std::function<void(ChunkHandle)> m_acquiredObserver;
        std::function<void(ChunkHandle)> m_releasedObserver;
    };
} // namespace graphFlow::detail
