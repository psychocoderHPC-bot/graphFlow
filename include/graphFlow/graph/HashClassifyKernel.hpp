/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/Classifier.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow
{
    /** Large-bin histogram kernel for one tile of one input chunk.
     *
     * The launch uses one thread block per tile, so the block index selects the tile
     * `[tileOffset, tileOffset + count)` this block owns. Per particle the classifier is applied
     * once and its bin is inserted into a per-block open-addressing hash table (`atomicCas` on the
     * key, `atomicAdd` on the count). After the insertion barrier every occupied entry is flushed
     * into the device-scope `counts[bin]` total, so the shared table never has to be as wide as the
     * bin count.
     *
     * A full table is a hard batch error: the block records it in a shared flag, writes
     * `counts[maxBins]` and returns without flushing. The host then aborts the batch before any
     * output chunk is planned, so no partial output is published. There is no dense fallback.
     *
     * An out-of-range classifier result (`bin >= maxBins`) writes the same status word and is not
     * inserted, so it can never index the shared table or the global totals. Every thread reaches
     * every block barrier, including inactive tail lanes.
     */
    template<typename TPolicy, typename TBinTable>
    struct HashClassifyKernel
    {
        static_assert(TBinTable::isHash, "HashClassifyKernel requires a hash bin table");

        TPolicy policy{};

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto ids,
            alpaka::concepts::IMdSpan auto properties,
            alpaka::concepts::IMdSpan auto weights,
            alpaka::concepts::IMdSpan auto visits,
            alpaka::concepts::IMdSpan auto binIds,
            alpaka::concepts::IMdSpan auto counts,
            std::uint32_t chunkIndex,
            std::uint32_t chunkCapacity,
            std::uint32_t fill,
            std::uint32_t tileSize,
            std::uint32_t maxBins) const
        {
            using namespace alpaka;

            auto table = onAcc::declareSharedMdArray<HashBinEntry, uniqueId()>(
                acc,
                CVec<std::uint32_t, TBinTable::entries>{});
            auto& tableFull = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

            auto const [tile] = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
            std::uint32_t const tileOffset = static_cast<std::uint32_t>(tile) * tileSize;
            std::uint32_t const remaining = fill > tileOffset ? fill - tileOffset : 0u;
            std::uint32_t const count = remaining < tileSize ? remaining : tileSize;
            std::uint32_t const base = chunkIndex * chunkCapacity;

            for(auto [entry] : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{std::uint32_t{TBinTable::entries}}))
                table[entry] = HashBinEntry{hashEmptyKey, 0u, 0u, 0u};
            // Exactly one worker writes the scalar flag; the barrier publishes it.
            for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{1u}}))
            {
                alpaka::unused(one);
                tableFull = 0u;
            }
            onAcc::syncBlockThreads(acc);

            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
            {
                std::uint32_t const slot = base + tileOffset + i;
                Particle particle{ids[slot], properties[slot], weights[slot], visits[slot]};
                std::uint32_t const bin
                    = policy.classify(particle.id, particle.property, particle.weight, particle.visit);
                if(bin >= maxBins)
                {
                    // Hard batch error: report and never index the table.
                    onAcc::atomicExch(acc, &counts[maxBins], std::uint32_t{1u}, onAcc::scope::device);
                    continue;
                }
                binIds[slot] = bin;
                if(!hashInsert<ALPAKA_TYPEOF(acc), TBinTable>(acc, table.data(), bin))
                {
                    onAcc::atomicExch(acc, &tableFull, std::uint32_t{1u}, onAcc::scope::block);
                    break;
                }
            }
            onAcc::syncBlockThreads(acc);

            if(tableFull != 0u)
            {
                onAcc::atomicExch(acc, &counts[maxBins], std::uint32_t{1u}, onAcc::scope::device);
                return;
            }

            for(auto [entry] : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{std::uint32_t{TBinTable::entries}}))
            {
                std::uint32_t const bin = table[entry].key;
                if(bin != hashEmptyKey)
                    onAcc::atomicAdd(acc, &counts[bin], table[entry].count, onAcc::scope::device);
            }
        }
    };
} // namespace graphFlow
