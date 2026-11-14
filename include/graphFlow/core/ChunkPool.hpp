/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/PoolBudget.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace graphFlow
{
    /** Fixed-size SoA chunk pool with a host-side free list.
     *
     * Particle `i` of chunk `k` lives at slot `k * chunkCapacity + i` of every payload array.
     * Payload memory is allocated once with `onHost::alloc<T>(device, Vec{numChunks * C})`; the hot
     * path never allocates. Device views are public so kernels can be launched against the pool.
     *
     * Kernels index payload arrays with 32-bit slot arithmetic (`chunkIndex * C + tileOffset + i`),
     * so the constructor rejects a pool whose total slot count exceeds `UINT32_MAX` instead of
     * letting `numChunks * C` wrap. This caps `numChunks` for large `C`; the limit is the supported
     * pool size, not a silent truncation.
     */
    template<typename TDevice>
    class ChunkPool
    {
    public:
        using IdView = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<PoolIdValue>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using U32View = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<PoolU32Value>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

        ChunkPool(TDevice device, std::uint32_t numChunks, std::uint32_t chunkCapacity)
            : m_numChunks{numChunks}
            , m_chunkCapacity{chunkCapacity}
            , m_slots{checkedSlotCount(numChunks, chunkCapacity)}
            , m_ids{alpaka::onHost::alloc<PoolIdValue>(device, alpaka::Vec{m_slots})}
            , m_properties{alpaka::onHost::alloc<PoolU32Value>(device, alpaka::Vec{m_slots})}
            , m_weights{alpaka::onHost::alloc<PoolU32Value>(device, alpaka::Vec{m_slots})}
            , m_visits{alpaka::onHost::alloc<PoolU32Value>(device, alpaka::Vec{m_slots})}
            , m_binIds{alpaka::onHost::alloc<PoolU32Value>(device, alpaka::Vec{m_slots})}
            , m_generations(numChunks, 0u)
            , m_leases(numChunks, 0u)
        {
            m_free.reserve(numChunks);
            // Deterministic free list order; the last entry is handed out first.
            for(std::uint32_t i = 0u; i < numChunks; ++i)
                m_free.push_back(numChunks - 1u - i);
        }

        [[nodiscard]] std::uint32_t numChunks() const
        {
            return m_numChunks;
        }

        [[nodiscard]] std::uint32_t chunkCapacity() const
        {
            return m_chunkCapacity;
        }

        [[nodiscard]] std::uint32_t freeCount() const
        {
            return static_cast<std::uint32_t>(m_free.size());
        }

        [[nodiscard]] std::uint32_t liveCount() const
        {
            return m_numChunks - freeCount();
        }

        /** Pop one chunk from the free list. Throws when the pool is exhausted. */
        ChunkHandle acquire()
        {
            if(m_free.empty())
                throw std::runtime_error{"graphFlow::ChunkPool: pool exhausted"};
            std::uint32_t const index = m_free.back();
            m_free.pop_back();
            m_peakLive = liveCount() > m_peakLive ? liveCount() : m_peakLive;
            return ChunkHandle{index, m_generations[index]};
        }

        /** Return a chunk to the free list and invalidate all outstanding handles.
         *
         * Releasing a chunk while a reader lease is outstanding is a use-after-free bug, so the
         * debug lease check rejects it instead of silently handing the memory to a new producer.
         */
        void release(ChunkHandle handle)
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkPool: chunk index out of range"};
            if(handle.generation != m_generations[handle.index])
                throw std::invalid_argument{"graphFlow::ChunkPool: stale chunk handle"};
            if(m_leases[handle.index] != 0u)
                throw std::logic_error{"graphFlow::ChunkPool: release while a reader lease is outstanding"};
            ++m_generations[handle.index];
            m_free.push_back(handle.index);
        }

        /** Register an asynchronous reader of a chunk.
         *
         * The runner holds a lease from the launch of a non-source batch until the completion event
         * of the batch's last device stage has been observed. Together with the check in `release`
         * this enforces release-after-last-use without a device-side reference count.
         */
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
                throw std::logic_error{"graphFlow::ChunkPool: lease release without a matching acquire"};
            --m_leases[handle.index];
        }

        [[nodiscard]] std::uint32_t leaseCount(ChunkHandle handle) const
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkPool: chunk index out of range"};
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

        /** Slot offset of the first particle of a chunk. */
        [[nodiscard]] std::uint64_t base(ChunkHandle handle) const
        {
            return static_cast<std::uint64_t>(handle.index) * m_chunkCapacity;
        }

        [[nodiscard]] std::uint32_t generation(std::uint32_t index) const
        {
            return m_generations.at(index);
        }

        void validateLive(ChunkHandle handle) const
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkPool: chunk index out of range"};
            if(handle.generation != m_generations[handle.index])
                throw std::invalid_argument{"graphFlow::ChunkPool: stale chunk handle"};
        }

        [[nodiscard]] IdView const& ids() const
        {
            return m_ids;
        }

        [[nodiscard]] U32View const& properties() const
        {
            return m_properties;
        }

        [[nodiscard]] U32View const& weights() const
        {
            return m_weights;
        }

        [[nodiscard]] U32View const& visits() const
        {
            return m_visits;
        }

        [[nodiscard]] U32View const& binIds() const
        {
            return m_binIds;
        }

        [[nodiscard]] IdView& ids()
        {
            return m_ids;
        }

        [[nodiscard]] U32View& properties()
        {
            return m_properties;
        }

        [[nodiscard]] U32View& weights()
        {
            return m_weights;
        }

        [[nodiscard]] U32View& visits()
        {
            return m_visits;
        }

        [[nodiscard]] U32View& binIds()
        {
            return m_binIds;
        }

    private:
        /** Validated total slot count; kernels address slots with a 32-bit index. */
        [[nodiscard]] static std::uint64_t checkedSlotCount(std::uint32_t numChunks, std::uint32_t chunkCapacity)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::ChunkPool: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::ChunkPool: chunkCapacity must be non-zero"};
            std::uint64_t const slots = static_cast<std::uint64_t>(numChunks) * chunkCapacity;
            if(slots > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{
                    "graphFlow::ChunkPool: numChunks * chunkCapacity exceeds the 32-bit slot index"};
            return slots;
        }

        std::uint32_t m_numChunks;
        std::uint32_t m_chunkCapacity;
        std::uint64_t m_slots;
        IdView m_ids;
        U32View m_properties;
        U32View m_weights;
        U32View m_visits;
        U32View m_binIds;
        std::vector<std::uint32_t> m_generations;
        std::vector<std::uint32_t> m_leases;
        std::vector<std::uint32_t> m_free;
        std::uint32_t m_peakLive = 0u;
        std::uint32_t m_peakLeases = 0u;
    };
} // namespace graphFlow
