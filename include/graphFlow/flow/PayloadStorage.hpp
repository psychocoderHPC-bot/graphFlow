/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/flow/Payload.hpp"

#include <concepts>
#include <cstdint>
#include <utility>

namespace graphFlow
{
    /** Chunk-storage contract consumed by the scheduler and the public graph layer.
     *
     * The contract covers chunk geometry, ownership, generation checks, and reader leases. It
     * deliberately does not name payload fields: an adapter owns its field layout, while the
     * scheduler only needs a `ChunkHandle`-based lifecycle. Concrete adapters may expose their
     * field views in addition to this contract.
     */
    template<typename TStorage>
    concept PayloadStorage
        = requires(TStorage storage, typename TStorage::handle_type handle, std::uint32_t chunkIndex) {
              typename TStorage::device_type;
              typename TStorage::handle_type;
              requires std::same_as<typename TStorage::handle_type, ChunkHandle>;

              { storage.numChunks() } -> std::convertible_to<std::uint32_t>;
              { storage.chunkCapacity() } -> std::convertible_to<std::uint32_t>;
              { storage.freeCount() } -> std::convertible_to<std::uint32_t>;
              { storage.liveCount() } -> std::convertible_to<std::uint32_t>;

              { storage.acquire() } -> std::convertible_to<typename TStorage::handle_type>;
              storage.release(handle);
              storage.acquireLease(handle);
              storage.releaseLease(handle);
              { storage.leaseCount(handle) } -> std::convertible_to<std::uint32_t>;
              { storage.isLive(handle) } -> std::same_as<bool>;
              { storage.base(handle) } -> std::convertible_to<std::uint64_t>;
              { storage.generation(chunkIndex) } -> std::convertible_to<std::uint32_t>;
              storage.validateLive(handle);
          };

    /** Default payload storage of the particle path.
     *
     * Adapts the existing fixed-capacity `ChunkPool` without changing its arrays, free list, or
     * generation/lease behavior. `ids`, `properties`, `weights`, `visits`, and `binIds` keep the
     * SoA layout expected by the current kernels; `pool()` exposes the underlying pool for callers
     * that need the complete existing surface.
     *
     * The adapter is additive: the particle example and the current kernels keep using
     * `ChunkPool` directly until the public graph API consumes this contract.
     */
    template<typename TDevice>
    class ParticleStorage
    {
    public:
        using device_type = TDevice;
        using handle_type = ChunkHandle;
        using Pool = ChunkPool<TDevice>;
        using IdView = typename Pool::IdView;
        using U32View = typename Pool::U32View;

        ParticleStorage(TDevice device, std::uint32_t numChunks, std::uint32_t chunkCapacity)
            : m_pool{std::move(device), numChunks, chunkCapacity}
        {
        }

        [[nodiscard]] std::uint32_t numChunks() const
        {
            return m_pool.numChunks();
        }

        [[nodiscard]] std::uint32_t chunkCapacity() const
        {
            return m_pool.chunkCapacity();
        }

        [[nodiscard]] std::uint32_t freeCount() const
        {
            return m_pool.freeCount();
        }

        [[nodiscard]] std::uint32_t liveCount() const
        {
            return m_pool.liveCount();
        }

        [[nodiscard]] ChunkHandle acquire()
        {
            return m_pool.acquire();
        }

        void release(ChunkHandle handle)
        {
            m_pool.release(handle);
        }

        void acquireLease(ChunkHandle handle)
        {
            m_pool.acquireLease(handle);
        }

        void releaseLease(ChunkHandle handle)
        {
            m_pool.releaseLease(handle);
        }

        [[nodiscard]] std::uint32_t leaseCount(ChunkHandle handle) const
        {
            return m_pool.leaseCount(handle);
        }

        [[nodiscard]] std::uint32_t peakLive() const
        {
            return m_pool.peakLive();
        }

        [[nodiscard]] std::uint32_t peakLeases() const
        {
            return m_pool.peakLeases();
        }

        [[nodiscard]] bool isLive(ChunkHandle handle) const
        {
            return m_pool.isLive(handle);
        }

        [[nodiscard]] std::uint64_t base(ChunkHandle handle) const
        {
            return m_pool.base(handle);
        }

        [[nodiscard]] std::uint32_t generation(std::uint32_t index) const
        {
            return m_pool.generation(index);
        }

        void validateLive(ChunkHandle handle) const
        {
            m_pool.validateLive(handle);
        }

        [[nodiscard]] IdView const& ids() const
        {
            return m_pool.ids();
        }

        [[nodiscard]] U32View const& properties() const
        {
            return m_pool.properties();
        }

        [[nodiscard]] U32View const& weights() const
        {
            return m_pool.weights();
        }

        [[nodiscard]] U32View const& visits() const
        {
            return m_pool.visits();
        }

        [[nodiscard]] U32View const& binIds() const
        {
            return m_pool.binIds();
        }

        [[nodiscard]] IdView& ids()
        {
            return m_pool.ids();
        }

        [[nodiscard]] U32View& properties()
        {
            return m_pool.properties();
        }

        [[nodiscard]] U32View& weights()
        {
            return m_pool.weights();
        }

        [[nodiscard]] U32View& visits()
        {
            return m_pool.visits();
        }

        [[nodiscard]] U32View& binIds()
        {
            return m_pool.binIds();
        }

        [[nodiscard]] Pool& pool()
        {
            return m_pool;
        }

        [[nodiscard]] Pool const& pool() const
        {
            return m_pool;
        }

    private:
        Pool m_pool;
    };

} // namespace graphFlow
