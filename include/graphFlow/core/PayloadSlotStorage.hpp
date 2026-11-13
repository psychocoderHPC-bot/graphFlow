/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/ChunkLedger.hpp"
#include "graphFlow/core/ChunkStorage.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace graphFlow
{
    /** Generic array-of-struct payload storage with one slot per payload record.
     *
     * The payload array is a single `onHost::alloc<TPayload>` of `numChunks * chunkCapacity`
     * records, so slot `k * chunkCapacity + i` is the `i`-th payload of chunk `k` and
     * `slotBytes == sizeof(TPayload)`. Chunk lifecycle, generations, reader leases and the free
     * list live in the shared `detail::ChunkLedger`, not in the payload array.
     *
     * This is the default storage for a non-particle POD payload. The particle path keeps its SoA
     * `ParticleStorage`; binding an arbitrary payload (for example a photon record) to the
     * particle pool would silently reinterpret its fields and is therefore never the default.
     */
    template<typename TPayload, typename TDevice>
    class PayloadSlotStorage
    {
        static_assert(
            std::is_trivially_copyable_v<TPayload>,
            "graphFlow::PayloadSlotStorage: the payload must be trivially copyable");
        static_assert(
            std::is_default_constructible_v<TPayload>,
            "graphFlow::PayloadSlotStorage: the payload must be default constructible");

    public:
        using device_type = TDevice;
        using handle_type = ChunkHandle;
        using PayloadType = TPayload;
        using PayloadView
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<TPayload>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

        /** Payload bytes of one slot; the single source of truth for this storage. */
        static constexpr std::uint64_t slotBytes = sizeof(TPayload);

        PayloadSlotStorage(TDevice device, std::uint32_t numChunks, std::uint32_t chunkCapacity)
            : m_chunkCapacity{chunkCapacity}
            , m_slots{checkedSlotCount(numChunks, chunkCapacity)}
            , m_ledger{numChunks}
            , m_payload{alpaka::onHost::alloc<TPayload>(device, alpaka::Vec{m_slots})}
        {
        }

        [[nodiscard]] std::uint32_t numChunks() const
        {
            return m_ledger.numChunks();
        }

        [[nodiscard]] std::uint32_t chunkCapacity() const
        {
            return m_chunkCapacity;
        }

        [[nodiscard]] std::uint32_t freeCount() const
        {
            return m_ledger.freeCount();
        }

        [[nodiscard]] std::uint32_t liveCount() const
        {
            return m_ledger.liveCount();
        }

        [[nodiscard]] ChunkHandle acquire()
        {
            return m_ledger.acquire();
        }

        void release(ChunkHandle handle)
        {
            m_ledger.release(handle);
        }

        void acquireLease(ChunkHandle handle)
        {
            m_ledger.acquireLease(handle);
        }

        void releaseLease(ChunkHandle handle)
        {
            m_ledger.releaseLease(handle);
        }

        [[nodiscard]] std::uint32_t leaseCount(ChunkHandle handle) const
        {
            return m_ledger.leaseCount(handle);
        }

        [[nodiscard]] std::uint32_t peakLive() const
        {
            return m_ledger.peakLive();
        }

        [[nodiscard]] std::uint32_t peakLeases() const
        {
            return m_ledger.peakLeases();
        }

        [[nodiscard]] bool isLive(ChunkHandle handle) const
        {
            return m_ledger.isLive(handle);
        }

        /** Slot offset of the first payload of a chunk. */
        [[nodiscard]] std::uint64_t base(ChunkHandle handle) const
        {
            return static_cast<std::uint64_t>(handle.index) * m_chunkCapacity;
        }

        [[nodiscard]] std::uint32_t generation(std::uint32_t index) const
        {
            return m_ledger.generation(index);
        }

        void validateLive(ChunkHandle handle) const
        {
            m_ledger.validateLive(handle);
        }

        /** Total number of payload slots (`numChunks * chunkCapacity`). */
        [[nodiscard]] std::uint64_t slotCount() const
        {
            return m_slots;
        }

        /** Device view of the contiguous payload array. */
        [[nodiscard]] PayloadView const& payload() const
        {
            return m_payload;
        }

        [[nodiscard]] PayloadView& payload()
        {
            return m_payload;
        }

    private:
        /** Validated total slot count; kernels address slots with a 32-bit index. */
        [[nodiscard]] static std::uint64_t checkedSlotCount(std::uint32_t numChunks, std::uint32_t chunkCapacity)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: chunkCapacity must be non-zero"};
            std::uint64_t const slots = static_cast<std::uint64_t>(numChunks) * chunkCapacity;
            if(slots > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{
                    "graphFlow::PayloadSlotStorage: numChunks * chunkCapacity exceeds the 32-bit slot index"};
            return slots;
        }

        std::uint32_t m_chunkCapacity;
        std::uint64_t m_slots;
        detail::ChunkLedger m_ledger;
        PayloadView m_payload;
    };
} // namespace graphFlow
