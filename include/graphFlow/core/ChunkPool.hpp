/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
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
     */
    template<typename TDevice>
    class ChunkPool
    {
    public:
        using IdView = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint64_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using U32View = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

        ChunkPool(TDevice device, std::uint32_t numChunks, std::uint32_t chunkCapacity)
            : m_numChunks{numChunks}
            , m_chunkCapacity{chunkCapacity}
            , m_ids{alpaka::onHost::alloc<std::uint64_t>(device, alpaka::Vec{static_cast<std::uint64_t>(numChunks) * chunkCapacity})}
            , m_properties{
                  alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(numChunks) * chunkCapacity})}
            , m_weights{
                  alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(numChunks) * chunkCapacity})}
            , m_visits{
                  alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(numChunks) * chunkCapacity})}
            , m_binIds{
                  alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(numChunks) * chunkCapacity})}
            , m_generations(numChunks, 0u)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::ChunkPool: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::ChunkPool: chunkCapacity must be non-zero"};
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
            return ChunkHandle{index, m_generations[index]};
        }

        /** Return a chunk to the free list and invalidate all outstanding handles. */
        void release(ChunkHandle handle)
        {
            if(handle.index >= m_numChunks)
                throw std::invalid_argument{"graphFlow::ChunkPool: chunk index out of range"};
            if(handle.generation != m_generations[handle.index])
                throw std::invalid_argument{"graphFlow::ChunkPool: stale chunk handle"};
            ++m_generations[handle.index];
            m_free.push_back(handle.index);
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
        std::uint32_t m_numChunks;
        std::uint32_t m_chunkCapacity;
        IdView m_ids;
        U32View m_properties;
        U32View m_weights;
        U32View m_visits;
        U32View m_binIds;
        std::vector<std::uint32_t> m_generations;
        std::vector<std::uint32_t> m_free;
    };
} // namespace graphFlow
