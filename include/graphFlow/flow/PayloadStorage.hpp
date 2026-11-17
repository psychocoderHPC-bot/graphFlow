/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/ChunkLedger.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/core/PoolMode.hpp"
#include "graphFlow/core/SlotAllocator.hpp"
#include "graphFlow/flow/Attributes.hpp"
#include "graphFlow/flow/Payload.hpp"

#include <concepts>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace graphFlow
{
    /** Canonical particle payload declared in `graphFlow/graph/Classifier.hpp`; only this exact
     * type selects the particle SoA storage by default. */
    struct Particle;

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

        /** Payload bytes of one particle slot; matches the SoA arrays of `ChunkPool`. */
        static constexpr std::uint64_t slotBytes = payloadBytesPerSlot;

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

        void setChunkObservers(std::function<void(ChunkHandle)> acquired, std::function<void(ChunkHandle)> released)
        {
            m_pool.setChunkObservers(std::move(acquired), std::move(released));
        }

        void clearChunkObservers()
        {
            m_pool.clearChunkObservers();
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

    namespace detail
    {
        /** Non-owning, device-copyable pack of one column view per attribute. */
        template<typename... TViews>
        struct SoAColumnViewPack;

        template<>
        struct SoAColumnViewPack<>
        {
        };

        template<typename TView, typename... TRest>
        struct SoAColumnViewPack<TView, TRest...>
        {
            TView view;
            SoAColumnViewPack<TRest...> rest;

            constexpr SoAColumnViewPack(TView first, TRest... remaining) : view{first}, rest{remaining...}
            {
            }

            template<std::size_t I>
            ALPAKA_FN_ACC constexpr decltype(auto) get() &
            {
                if constexpr(I == 0u)
                    return (view);
                else
                    return rest.template get<I - 1u>();
            }

            template<std::size_t I>
            ALPAKA_FN_ACC constexpr decltype(auto) get() const&
            {
                if constexpr(I == 0u)
                    return (view);
                else
                    return rest.template get<I - 1u>();
            }
        };
    } // namespace detail

    /** AoS-over-SoA payload storage with one device column per attribute.
     *
     * Every attribute gets one `onHost::alloc<value_type>` of `numChunks * chunkCapacity`
     * elements; a `binIds` scratch column of 32-bit words is added for the scatter path. There is
     * no per-particle allocation and no single contiguous record array: a logical record is
     * gathered from the columns at one slot index.
     *
     * Each chunk is one row of the column layout. `base(handle)` is the first slot of the row and
     * `slotIndex(handle, row)` is the row-major slot index `base(handle) + row`. `slotBytes` is the
     * accounting size of the logical AoS record: the sum of the naturally aligned attribute widths
     * plus the 32-bit scratch word. It drives the payload budget exactly like `ChunkPool` and
     * `PayloadSlotStorage`, so the default storages keep their existing sizes.
     *
     * `proxy<Set<Tags...>>(slot)` returns a `RecordProxy` limited to the tags a node declares.
     * Copying that proxy to a `Record` copies every attribute, including attributes outside the
     * node's subset.
     *
     * The `SlotArenaOptions` constructor is the opt-in slot-arena layout (`PoolMode::slotArena`).
     * The columns then hold `arenaSlots` elements and every chunk owns the contiguous range its
     * committed fill asked for, so `base(handle)` is a stored start and `slotIndex(handle, row)`
     * is `base + row` with `row < fill(handle)`. The fixed constructor keeps the historical row
     * layout `handle.index * chunkCapacity`.
     */
    template<typename TDevice, typename... TAttrs>
    class SoAStorage
    {
        using AttributeTraits = AttributeList<TAttrs...>;

        static_assert(AttributeTraits::size == sizeof...(TAttrs), "graphFlow::SoAStorage: invalid attribute list");

        template<typename TAttr>
        using ColumnType = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<typename TAttr::value_type>(
                std::declval<TDevice>(),
                alpaka::Vec{std::uint64_t{1u}}));

        template<typename TAttr>
        using ColumnView = ALPAKA_TYPEOF(alpaka::makeView(std::declval<ColumnType<TAttr>&>()));

        using ScratchType = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

    public:
        using device_type = TDevice;
        using handle_type = ChunkHandle;
        using attribute_list = AttributeTraits;
        using attribute_set = typename AttributeTraits::attribute_set;
        using record_type = Record<TAttrs...>;
        using bin_ids_view_type = ScratchType;

        /** Accounting bytes of one logical AoS record, including the bin-id scratch word. */
        static constexpr std::uint64_t slotBytes = AttributeTraits::slotBytes;
        /** Sum of the naturally aligned attribute widths, without the scratch word. */
        static constexpr std::uint64_t attributeBytes = AttributeTraits::attributeBytes;
        static constexpr std::uint64_t scratchBytes = AttributeTraits::scratchBytes;

        SoAStorage(
            TDevice device,
            std::uint32_t numChunks,
            std::uint32_t chunkCapacity,
            std::uint64_t payloadBudgetBytes = defaultPoolPayloadBudgetBytes)
            : m_chunkCapacity{chunkCapacity}
            , m_chunks{checkedChunkCount(numChunks, chunkCapacity, payloadBudgetBytes)}
            , m_slots{checkedPoolSlotCount(m_chunks, chunkCapacity, "graphFlow::SoAStorage")}
            , m_ledger{m_chunks}
            , m_columns{allocateColumn<TAttrs>(device, m_slots)...}
            , m_binIds{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{m_slots})}
        {
        }

        /** Slot-arena construction: every column holds `options.arenaSlots` elements. */
        SoAStorage(
            TDevice device,
            std::uint32_t numChunks,
            std::uint32_t chunkCapacity,
            SlotArenaOptions const& options)
            : m_chunkCapacity{chunkCapacity}
            , m_chunks{checkedArenaChunkCount(numChunks, chunkCapacity)}
            , m_slots{checkedArenaSlots(m_chunks, options)}
            , m_ledger{m_chunks}
            , m_columns{allocateColumn<TAttrs>(device, m_slots)...}
            , m_binIds{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{m_slots})}
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

        /** Payload bytes accounted by the budget after the clamp. */
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

        /** Acquire a chunk index and, in slot-arena mode, a range of exactly `fill` slots. */
        [[nodiscard]] ChunkHandle acquire(std::uint32_t fill)
        {
            if(fill == 0u)
                throw std::invalid_argument{"graphFlow::SoAStorage: fill must be non-zero"};
            if(fill > m_chunkCapacity)
                throw std::invalid_argument{"graphFlow::SoAStorage: fill exceeds chunkCapacity"};
            ChunkHandle const handle = m_ledger.acquire();
            if(!m_arena)
                return handle;
            try
            {
                m_arena->acquire(handle.index, fill);
            }
            catch(...)
            {
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
            m_ledger.validateLive(handle);
            if(m_ledger.leaseCount(handle) != 0u)
                throw std::logic_error{"graphFlow::SoAStorage: release while a reader lease is outstanding"};
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

        /** First slot index of a chunk row. */
        [[nodiscard]] std::uint64_t base(ChunkHandle handle) const
        {
            if(handle.index >= m_chunks)
                throw std::invalid_argument{"graphFlow::SoAStorage: chunk index out of range"};
            if(!m_arena)
                return static_cast<std::uint64_t>(handle.index) * m_chunkCapacity;
            m_ledger.validateLive(handle);
            return m_arena->base(handle.index);
        }

        /** Row-major slot index inside a chunk row.
         *
         * The fixed layout bounds the row by `chunkCapacity`; the slot-arena layout bounds it by
         * the fill the chunk committed, because the slots beyond it belong to another chunk.
         */
        [[nodiscard]] std::uint64_t slotIndex(std::uint64_t chunkIndex, std::uint32_t row) const
        {
            if(chunkIndex >= m_chunks)
                throw std::invalid_argument{"graphFlow::SoAStorage: chunk index out of range"};
            if(!m_arena)
            {
                if(row >= m_chunkCapacity)
                    throw std::invalid_argument{"graphFlow::SoAStorage: row index out of range"};
                return chunkIndex * m_chunkCapacity + row;
            }
            auto const index = static_cast<std::uint32_t>(chunkIndex);
            if(row >= m_arena->fill(index))
                throw std::invalid_argument{"graphFlow::SoAStorage: row index out of range"};
            return m_arena->base(index) + row;
        }

        /** Slot count a chunk currently owns; `chunkCapacity` in the fixed layout. */
        [[nodiscard]] std::uint32_t fill(ChunkHandle handle) const
        {
            if(handle.index >= m_chunks)
                throw std::invalid_argument{"graphFlow::SoAStorage: chunk index out of range"};
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
                    "graphFlow::SoAStorage: " + std::to_string(m_ledger.liveCount())
                    + " chunk(s) are still live at teardown"};
            if(m_arena)
                m_arena->verifyReleased();
        }

        /** Row-major slot index of a live chunk row. */
        [[nodiscard]] std::uint64_t slotIndex(ChunkHandle handle, std::uint32_t row) const
        {
            validateLive(handle);
            return slotIndex(static_cast<std::uint64_t>(handle.index), row);
        }

        [[nodiscard]] std::uint32_t generation(std::uint32_t index) const
        {
            return m_ledger.generation(index);
        }

        void validateLive(ChunkHandle handle) const
        {
            m_ledger.validateLive(handle);
        }

        /** Number of slots in each attribute column. */
        [[nodiscard]] std::uint64_t slotCount() const
        {
            return m_slots;
        }

        /** Device column of an attribute. */
        template<typename Tag>
        [[nodiscard]] auto& column()
        {
            static_assert(
                AttributeTraits::template contains<Tag>,
                "graphFlow::SoAStorage: column<Tag>() names an unknown attribute");
            return std::get<AttributeTraits::template index<Tag>>(m_columns);
        }

        template<typename Tag>
        [[nodiscard]] auto const& column() const
        {
            static_assert(
                AttributeTraits::template contains<Tag>,
                "graphFlow::SoAStorage: column<Tag>() names an unknown attribute");
            return std::get<AttributeTraits::template index<Tag>>(m_columns);
        }

        /** Scratch column for the bin id assigned to each slot. */
        [[nodiscard]] bin_ids_view_type const& binIds() const
        {
            return m_binIds;
        }

        [[nodiscard]] bin_ids_view_type& binIds()
        {
            return m_binIds;
        }

        /** Gather one logical record from the attribute columns. */
        [[nodiscard]] record_type loadRecord(std::uint32_t slot) const
        {
            checkSlot(slot);
            return record_type{column<typename TAttrs::tag_type>()[slot]...};
        }

        /** Scatter one logical record into the attribute columns. */
        void storeRecord(std::uint32_t slot, record_type const& record)
        {
            checkSlot(slot);
            (assignColumn<TAttrs>(slot, record), ...);
        }

        /** Per-slot accessor for a node that declares only `Set<Tags...>`. */
        class RowAccess
        {
        public:
            using record_type = Record<TAttrs...>;
            using views_type = detail::SoAColumnViewPack<ColumnView<TAttrs>...>;

            views_type views;
            std::uint32_t slot = 0u;

            template<typename Tag>
            ALPAKA_FN_ACC constexpr decltype(auto) get() const
            {
                constexpr std::size_t attributeIndex = AttributeTraits::template index<Tag>;
                return views.template get<attributeIndex>()[slot];
            }

            template<typename Tag, typename U>
            ALPAKA_FN_ACC constexpr void set(U&& value)
            {
                constexpr std::size_t attributeIndex = AttributeTraits::template index<Tag>;
                views.template get<attributeIndex>()[slot] = std::forward<U>(value);
            }

            [[nodiscard]] ALPAKA_FN_ACC constexpr record_type load() const
            {
                return record_type{get<typename TAttrs::tag_type>()...};
            }

            ALPAKA_FN_ACC constexpr void store(record_type const& record)
            {
                (set<typename TAttrs::tag_type>(record.template get<typename TAttrs::tag_type>()), ...);
            }
        };

        /** Narrow proxy over one slot; the set is checked against the storage attributes. */
        template<typename TSet>
        [[nodiscard]] auto proxy(std::uint32_t slot)
        {
            static_assert(
                detail::SetCompatible<TSet, AttributeTraits>::value,
                "graphFlow::SoAStorage: proxy Set contains a tag that is not an attribute");
            checkSlot(slot);
            return RecordProxy<RowAccess, TSet>{RowAccess{columnViews(), slot}};
        }

    private:
        [[nodiscard]] static std::uint32_t checkedChunkCount(
            std::uint32_t numChunks,
            std::uint32_t chunkCapacity,
            std::uint64_t payloadBudgetBytes)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::SoAStorage: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::SoAStorage: chunkCapacity must be non-zero"};
            std::uint64_t const maxChunks = maxPayloadChunks(payloadBudgetBytes, chunkCapacity, slotBytes);
            if(maxChunks == 0u)
                throw std::invalid_argument{"graphFlow::SoAStorage: payload budget cannot hold one chunk"};
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(numChunks, maxChunks));
        }

        [[nodiscard]] static std::uint32_t checkedArenaChunkCount(std::uint32_t numChunks, std::uint32_t chunkCapacity)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::SoAStorage: numChunks must be non-zero"};
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::SoAStorage: chunkCapacity must be non-zero"};
            return numChunks;
        }

        /** Arena size after the 32-bit cap and the one-slot-per-chunk lower bound. */
        [[nodiscard]] static std::uint64_t checkedArenaSlots(std::uint32_t numChunks, SlotArenaOptions const& options)
        {
            std::uint64_t const slots = options.arenaSlots != 0u
                                            ? options.arenaSlots
                                            : maxPayloadSlots(options.payloadBudgetBytes, slotBytes);
            checkedArenaSlotCount(slots, "graphFlow::SoAStorage slot arena");
            if(static_cast<std::uint64_t>(numChunks) > slots)
                throw std::invalid_argument{
                    "graphFlow::SoAStorage: " + std::to_string(numChunks)
                    + " chunks need at least one slot each but the arena holds " + std::to_string(slots)};
            return slots;
        }

        template<typename TAttr>
        [[nodiscard]] static ColumnType<TAttr> allocateColumn(TDevice const& device, std::uint64_t slots)
        {
            return alpaka::onHost::alloc<typename TAttr::value_type>(device, alpaka::Vec{slots});
        }

        [[nodiscard]] auto columnViews()
        {
            return detail::SoAColumnViewPack<ColumnView<TAttrs>...>{
                alpaka::makeView(column<typename TAttrs::tag_type>())...};
        }

        template<typename TAttr>
        void assignColumn(std::uint32_t slot, record_type const& record)
        {
            column<typename TAttr::tag_type>()[slot] = record.template get<typename TAttr::tag_type>();
        }

        void checkSlot(std::uint32_t slot) const
        {
            if(slot >= m_slots)
                throw std::invalid_argument{"graphFlow::SoAStorage: slot index out of range"};
        }

        std::uint32_t m_chunkCapacity;
        std::uint32_t m_chunks;
        std::uint64_t m_slots;
        detail::ChunkLedger m_ledger;
        std::tuple<ColumnType<TAttrs>...> m_columns;
        ScratchType m_binIds;
        /** Absent in the fixed layout; the fixed base stays `index * chunkCapacity`. */
        std::optional<ArenaSlotMap> m_arena;
    };

    namespace detail
    {
        template<typename>
        struct AlwaysFalse : std::false_type
        {
        };

        /** True if `TStorage` is one `SoAStorage` specialization. */
        template<typename TStorage>
        struct IsSoAStorage : std::false_type
        {
        };

        template<typename TDevice, typename... TAttrs>
        struct IsSoAStorage<SoAStorage<TDevice, TAttrs...>> : std::true_type
        {
        };

        /** Maps a `Record<Fields...>` to its matching `SoAStorage<Device, Fields...>`. */
        template<typename TRecord, typename TDevice>
        struct SoAStorageForRecord
        {
            static_assert(
                AlwaysFalse<TRecord>::value,
                "graphFlow::SoAFlow: the payload type must be a graphFlow::Record<Fields...>");

            /** Placeholder so naming `::type` instantiates this body and reports the message. */
            using type = void;
        };

        template<typename... Fields, typename TDevice>
        struct SoAStorageForRecord<Record<Fields...>, TDevice>
        {
            using type = SoAStorage<TDevice, Fields...>;
        };

        /** Checks whether an explicit storage can hold `TRecord` without reinterpreting fields.
         *
         * Non-SoA storages keep their own layout and are always accepted. A `SoAStorage` must have
         * exactly the record's field pack, including order, so `Flow` cannot silently pair a record
         * with columns of a different type.
         */
        template<typename TRecord, typename TStorage>
        struct SoAStorageMatchesRecord : std::bool_constant<!IsSoAStorage<TStorage>::value>
        {
        };

        template<typename... Fields, typename TDevice, typename... TAttrs>
        struct SoAStorageMatchesRecord<Record<Fields...>, SoAStorage<TDevice, TAttrs...>>
            : std::bool_constant<std::same_as<std::tuple<Fields...>, std::tuple<TAttrs...>>>
        {
        };
    } // namespace detail

    /** Default storage of a `Flow` payload.
     *
     * The canonical particle payload keeps the SoA `ParticleStorage` the kernels expect. Every
     * other POD payload gets the generic `PayloadSlotStorage`, which treats the payload as one
     * record per slot. Selecting the particle pool for an arbitrary payload would reinterpret its
     * fields as ids, properties, weights and visits, so that binding is never the default.
     */
    template<typename TPayload, typename TDevice>
    using DefaultStorage = std::conditional_t<
        std::same_as<std::remove_cv_t<TPayload>, Particle>,
        ParticleStorage<TDevice>,
        PayloadSlotStorage<TPayload, TDevice>>;

} // namespace graphFlow
