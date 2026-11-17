/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/ChunkLedger.hpp"
#include "graphFlow/core/ChunkStorage.hpp"
#include "graphFlow/core/PoolBudget.hpp"
#include "graphFlow/core/PoolMode.hpp"
#include "graphFlow/core/SlotAllocator.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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
     *
     * Like `ChunkPool`, the payload array is clamped to `payloadBudgetBytes`: a larger requested
     * chunk count is reduced to the largest fitting count and `numChunks()` reports the physical
     * count. The clamp is explicit and keeps the ledger and the payload array in sync, so a
     * generic flow cannot allocate more payload than the documented 4 GiB default.
     *
     * The `SlotArenaOptions` constructor is the opt-in slot-arena layout (`PoolMode::slotArena`).
     * The payload array then has `arenaSlots` slots and every chunk owns the contiguous range its
     * committed fill asks for at `acquire(fill)`, so `base(handle)` returns a stored start instead
     * of `handle.index * chunkCapacity`. The fixed constructor keeps the historical row layout and
     * computes the base exactly as before; nothing about its allocation, free-list order or
     * generation behavior changes.
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

        PayloadSlotStorage(
            TDevice device,
            std::uint32_t numChunks,
            std::uint32_t chunkCapacity,
            std::uint64_t payloadBudgetBytes = defaultPoolPayloadBudgetBytes)
            : m_chunkCapacity{chunkCapacity}
            , m_chunks{checkedChunkCount(numChunks, chunkCapacity, payloadBudgetBytes)}
            , m_slots{checkedPoolSlotCount(m_chunks, chunkCapacity, "graphFlow::PayloadSlotStorage")}
            , m_ledger{m_chunks}
            , m_payload{alpaka::onHost::alloc<TPayload>(device, alpaka::Vec{m_slots})}
        {
        }

        /** Slot-arena construction: one payload array of `options.arenaSlots` slots.
         *
         * `numChunks` is not clamped here: a chunk index costs one slot at minimum, so a request
         * for more chunk indices than the arena has slots is rejected instead of silently reduced.
         */
        PayloadSlotStorage(
            TDevice device,
            std::uint32_t numChunks,
            std::uint32_t chunkCapacity,
            SlotArenaOptions const& options)
            : m_chunkCapacity{chunkCapacity}
            , m_chunks{checkedArenaChunkCount(numChunks, chunkCapacity)}
            , m_slots{checkedArenaSlots(m_chunks, options)}
            , m_ledger{m_chunks}
            , m_payload{alpaka::onHost::alloc<TPayload>(device, alpaka::Vec{m_slots})}
            , m_arena{std::in_place, m_chunks, m_slots}
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

        /** Payload bytes actually allocated by this storage after the budget clamp. */
        [[nodiscard]] std::uint64_t payloadBytes() const
        {
            return m_slots * slotBytes;
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
            return acquire(m_chunkCapacity);
        }

        /** Acquire a chunk index and, in slot-arena mode, a range of exactly `fill` slots.
         *
         * The fixed layout ignores `fill` beyond validating it: its chunk rows are
         * `chunkCapacity` wide by construction, so a smaller fill only means the scheduler will
         * not use the remaining slots. The slot-arena layout commits the fill, which is what makes
         * a one-particle fragment own one slot.
         */
        [[nodiscard]] ChunkHandle acquire(std::uint32_t fill)
        {
            if(fill == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: fill must be non-zero"};
            if(fill > m_chunkCapacity)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: fill exceeds chunkCapacity"};
            ChunkHandle const handle = m_ledger.acquire();
            if(!m_arena)
                return handle;
            try
            {
                m_arena->acquire(handle.index, fill);
            }
            catch(...)
            {
                // Giving the index back keeps the ledger and the arena consistent: a failed
                // allocation must not leak a chunk index.
                m_ledger.release(handle);
                throw;
            }
            return handle;
        }

        void release(ChunkHandle handle)
        {
            if(!m_arena)
            {
                m_ledger.release(handle);
                return;
            }
            // Validate before touching the arena so a stale handle or an outstanding lease leaves
            // both the range and the ledger index in place.
            m_ledger.validateLive(handle);
            if(m_ledger.leaseCount(handle) != 0u)
                throw std::logic_error{"graphFlow::PayloadSlotStorage: release while a reader lease is outstanding"};
            m_arena->release(handle.index);
            m_ledger.release(handle);
        }

        void setChunkObservers(std::function<void(ChunkHandle)> acquired, std::function<void(ChunkHandle)> released)
        {
            m_ledger.setChunkObservers(std::move(acquired), std::move(released));
        }

        void clearChunkObservers()
        {
            m_ledger.clearChunkObservers();
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

        /** Slot offset of the first payload of a chunk.
         *
         * The index is range-checked so a malformed handle cannot produce an out-of-bounds slot
         * offset. The product is formed in 64 bits; the constructor already proved it fits the
         * 32-bit slot index used by the kernels.
         */
        [[nodiscard]] std::uint64_t base(ChunkHandle handle) const
        {
            if(handle.index >= m_chunks)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: chunk index out of range"};
            if(!m_arena)
                return static_cast<std::uint64_t>(handle.index) * m_chunkCapacity;
            // A stored base is only meaningful while the chunk owns its range; the generation
            // check is what keeps a stale handle from reading a reused index's range.
            m_ledger.validateLive(handle);
            return m_arena->base(handle.index);
        }

        /** Slot count a chunk currently owns; `chunkCapacity` in the fixed layout. */
        [[nodiscard]] std::uint32_t fill(ChunkHandle handle) const
        {
            if(handle.index >= m_chunks)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: chunk index out of range"};
            if(!m_arena)
                return m_chunkCapacity;
            m_ledger.validateLive(handle);
            return m_arena->fill(handle.index);
        }

        /** Layout this storage was built with. */
        [[nodiscard]] PoolMode poolMode() const
        {
            return m_arena ? PoolMode::slotArena : PoolMode::fixed;
        }

        /** Debug/test check: the arena's free and live ranges tile the whole arena exactly once. */
        void verifyArena() const
        {
            if(m_arena)
                m_arena->verify();
        }

        /** Leak check: no chunk and no slot range may survive a run. */
        void verifyArenaReleased() const
        {
            if(m_ledger.freeCount() != m_chunks)
                throw std::logic_error{
                    "graphFlow::PayloadSlotStorage: " + std::to_string(m_ledger.liveCount())
                    + " chunk(s) are still live at teardown"};
            if(m_arena)
                m_arena->verifyReleased();
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
        /** Chunk count after clamping the payload to `payloadBudgetBytes`.
         *
         * The clamp keeps the requested count when it fits, so existing small storages and their
         * lease/free-list order stay unchanged. A request above the ceiling is reduced instead of
         * being rejected, and `numChunks()` reports the reduced physical count.
         */
        [[nodiscard]] static std::uint32_t checkedChunkCount(
            std::uint32_t numChunks,
            std::uint32_t chunkCapacity,
            std::uint64_t payloadBudgetBytes)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: chunkCapacity must be non-zero"};
            std::uint64_t const maxChunks = maxPayloadChunks(payloadBudgetBytes, chunkCapacity, slotBytes);
            if(maxChunks == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: payload budget cannot hold one chunk"};
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(numChunks, maxChunks));
        }

        [[nodiscard]] static std::uint32_t checkedArenaChunkCount(std::uint32_t numChunks, std::uint32_t chunkCapacity)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::PayloadSlotStorage: chunkCapacity must be non-zero"};
            return numChunks;
        }

        /** Arena size after the 32-bit cap and the one-slot-per-chunk lower bound. */
        [[nodiscard]] static std::uint64_t checkedArenaSlots(std::uint32_t numChunks, SlotArenaOptions const& options)
        {
            std::uint64_t const slots = options.arenaSlots != 0u
                                            ? options.arenaSlots
                                            : maxPayloadSlots(options.payloadBudgetBytes, slotBytes);
            checkedArenaSlotCount(slots, "graphFlow::PayloadSlotStorage slot arena");
            if(static_cast<std::uint64_t>(numChunks) > slots)
                throw std::invalid_argument{
                    "graphFlow::PayloadSlotStorage: " + std::to_string(numChunks)
                    + " chunks need at least one slot each but the arena holds " + std::to_string(slots)};
            return slots;
        }

        std::uint32_t m_chunkCapacity;
        std::uint32_t m_chunks;
        std::uint64_t m_slots;
        detail::ChunkLedger m_ledger;
        PayloadView m_payload;
        /** Absent in the fixed layout; the fixed base stays `index * chunkCapacity`. */
        std::optional<ArenaSlotMap> m_arena;
    };
} // namespace graphFlow
