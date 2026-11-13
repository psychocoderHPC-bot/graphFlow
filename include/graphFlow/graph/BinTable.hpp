/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow
{
    /** Selects the dense per-block bin table.
     *
     * The dense path is unchanged: it keeps one shared `counts[B]` word per bin and is the
     * correctness reference. It is only usable while `B` fits in the block's shared memory.
     */
    struct DenseBinTable
    {
        static constexpr bool isHash = false;
    };

    /** Key value marking an unused slot of the large-bin hash table.
     *
     * An unscoped enum constant is an integral constant expression and is usable from device code.
     * A namespace-scope `inline constexpr` variable is not visible to nvcc device compilation
     * without an extra device symbol, which broke the CUDA build.
     */
    enum : std::uint32_t
    {
        hashEmptyKey = 0xFFFF'FFFFu
    };

    /** One open-addressing slot of the per-block hash table.
     *
     * `key` holds the bin id and is only ever changed by `atomicCas` from `hashEmptyKey` to a bin.
     * `count` is accumulated with block-scope `atomicAdd`. `base` and `localRank` are written by
     * the scatter rebuild before the particles read them.
     */
    struct HashBinEntry
    {
        std::uint32_t key = hashEmptyKey;
        std::uint32_t count = 0u;
        std::uint32_t base = 0u;
        std::uint32_t localRank = 0u;
    };

    /** Smallest power of two greater than or equal to `value`. */
    constexpr std::uint32_t nextPowerOfTwo(std::uint32_t value)
    {
        std::uint32_t result = 1u;
        while(result < value)
            result <<= 1u;
        return result;
    }

    /** Selects the large-bin shared-memory hash table.
     *
     * The table has `nextPowerOfTwo(T_SlotsPerTile * T_TileSize)` entries, i.e. it is sized by a
     * compile-time multiple of the tile size `T_TileSize`. The default load factor of two is the
     * starting point from the plan; raising `T_SlotsPerTile` to four is a measurement decision and
     * does not change the algorithm. The mask is valid because `entries` is a power of two.
     *
     * `T_TileSize` is a compile-time property of the policy. The kernel is still launched with the
     * configured runtime block extent; the table capacity is independent of the number of real
     * threads, which keeps the host backend (one thread per block) and an accelerator build on the
     * same code path.
     */
    template<std::uint32_t T_TileSize, std::uint32_t T_SlotsPerTile = 2u>
    struct HashBinTable
    {
        static constexpr bool isHash = true;
        static constexpr std::uint32_t tileSize = T_TileSize;
        static constexpr std::uint32_t slotsPerTile = T_SlotsPerTile;
        static constexpr std::uint32_t requestedEntries = T_SlotsPerTile * T_TileSize;
        static constexpr std::uint32_t entries = nextPowerOfTwo(requestedEntries);
        static constexpr std::uint32_t mask = entries - 1u;

        static_assert(T_TileSize > 0u, "a hash bin table needs a non-zero tile size");
        static_assert(T_SlotsPerTile > 0u, "a hash bin table needs at least one slot per tile particle");
        static_assert(entries >= requestedEntries, "hash bin table size overflow");
    };

    /** Mixing hash used for the open-addressing probe sequence.
     *
     * Multiplication by an odd constant is a bijection modulo a power of two, so distinct bin ids
     * below `entries` never collide. Larger ids are folded into the table and resolved by linear
     * probing.
     */
    constexpr std::uint32_t hashBinSlot(std::uint32_t bin, std::uint32_t mask)
    {
        return (bin * 0x9E37'79B1u) & mask;
    }

    /** Insert one bin or bump its count. Returns false when the table is full.
     *
     * The key is changed only by `atomicCas` from `hashEmptyKey` to the bin id. A slot that already
     * holds the bin is reused, so concurrent insertions of the same bin converge on one entry and
     * only the count is atomic.
     */
    template<typename TAcc, typename TBinTable>
    ALPAKA_FN_ACC bool hashInsert(TAcc const& acc, HashBinEntry* table, std::uint32_t bin)
    {
        std::uint32_t slot = hashBinSlot(bin, TBinTable::mask);
        for(std::uint32_t probe = 0u; probe < TBinTable::entries; ++probe)
        {
            std::uint32_t const old = alpaka::onAcc::atomicCas(
                acc,
                &table[slot].key,
                static_cast<std::uint32_t>(hashEmptyKey),
                bin,
                alpaka::onAcc::scope::block);
            if(old == hashEmptyKey || old == bin)
            {
                alpaka::onAcc::atomicAdd(
                    acc,
                    &table[slot].count,
                    std::uint32_t{1u},
                    alpaka::onAcc::scope::block);
                return true;
            }
            slot = (slot + 1u) & TBinTable::mask;
        }
        return false;
    }

    /** Look up a bin in a fully built table. Returns nullptr when it is absent. */
    template<typename TBinTable>
    ALPAKA_FN_ACC HashBinEntry* hashFind(HashBinEntry* table, std::uint32_t bin)
    {
        std::uint32_t slot = hashBinSlot(bin, TBinTable::mask);
        for(std::uint32_t probe = 0u; probe < TBinTable::entries; ++probe)
        {
            if(table[slot].key == bin)
                return &table[slot];
            if(table[slot].key == hashEmptyKey)
                return nullptr;
            slot = (slot + 1u) & TBinTable::mask;
        }
        return nullptr;
    }
} // namespace graphFlow
