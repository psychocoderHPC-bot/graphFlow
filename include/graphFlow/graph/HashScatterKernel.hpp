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
    /** Large-bin scatter kernel for one input chunk.
     *
     * One thread block processes the whole chunk. It rebuilds the per-bin counts from the stored
     * `binIds` into a per-block hash table, reserves one device-scope range
     * `base[bin] = atomicAdd(&next[bin], count, device)` per occupied entry, resets the per-entry
     * `localRank`, and then lets every particle claim `rank = atomicAdd(&localRank, 1, block)`.
     * The resulting `j = base[bin] + rank` may span chunk boundaries; the output chunk is resolved
     * through `binBase[bin] + j / C` exactly like the dense path.
     *
     * A full table during the rebuild is a hard batch error: the block writes `counts[maxBins]`
     * and returns before reserving or writing any output. The host sees the status word and does
     * not publish the partial batch.
     */
    template<typename TPolicy, typename TBinTable>
    struct HashScatterKernel
    {
        static_assert(TBinTable::isHash, "HashScatterKernel requires a hash bin table");

        TPolicy policy{};

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto idsIn,
            alpaka::concepts::IMdSpan auto propertiesIn,
            alpaka::concepts::IMdSpan auto weightsIn,
            alpaka::concepts::IMdSpan auto visitsIn,
            alpaka::concepts::IMdSpan auto binIds,
            alpaka::concepts::IMdSpan auto idsOut,
            alpaka::concepts::IMdSpan auto propertiesOut,
            alpaka::concepts::IMdSpan auto weightsOut,
            alpaka::concepts::IMdSpan auto visitsOut,
            alpaka::concepts::IMdSpan auto table,
            alpaka::concepts::IMdSpan auto next,
            alpaka::concepts::IMdSpan auto counts,
            alpaka::concepts::IMdSpan auto capped,
            std::uint32_t inBase,
            std::uint32_t outCapacity,
            std::uint32_t fill,
            std::uint32_t maxBins) const
        {
            using namespace alpaka;

            auto hashTable = onAcc::declareSharedMdArray<HashBinEntry, uniqueId()>(
                acc,
                CVec<std::uint32_t, TBinTable::entries>{});
            auto& tableFull = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

            for(auto [entry] : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{std::uint32_t{TBinTable::entries}}))
                hashTable[entry] = HashBinEntry{hashEmptyKey, 0u, 0u, 0u};
            for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{1u}}))
            {
                alpaka::unused(one);
                tableFull = 0u;
            }
            onAcc::syncBlockThreads(acc);

            // Rebuild the per-bin counts from the stored bin ids; never recompute the classifier.
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{fill}))
            {
                std::uint32_t const bin = binIds[inBase + i];
                if(bin >= maxBins)
                {
                    onAcc::atomicExch(acc, &counts[maxBins], std::uint32_t{1u}, onAcc::scope::device);
                    continue;
                }
                if(!hashInsert<ALPAKA_TYPEOF(acc), TBinTable>(acc, hashTable.data(), bin))
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
                std::uint32_t const bin = hashTable[entry].key;
                if(bin == hashEmptyKey)
                    continue;
                if(bin >= maxBins)
                {
                    onAcc::atomicExch(acc, &counts[maxBins], std::uint32_t{1u}, onAcc::scope::device);
                    continue;
                }
                hashTable[entry].base
                    = onAcc::atomicAdd(acc, &next[bin], hashTable[entry].count, onAcc::scope::device);
                hashTable[entry].localRank = 0u;
            }
            onAcc::syncBlockThreads(acc);

            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{fill}))
            {
                std::uint32_t const slotIn = inBase + i;
                std::uint32_t const bin = binIds[slotIn];
                if(bin >= maxBins)
                    continue;
                HashBinEntry* entry = hashFind<TBinTable>(hashTable.data(), bin);
                if(entry == nullptr)
                {
                    // The rebuild inserted every stored bin; a miss is an internal error.
                    onAcc::atomicExch(acc, &counts[maxBins], std::uint32_t{1u}, onAcc::scope::device);
                    continue;
                }
                std::uint32_t const rank
                    = onAcc::atomicAdd(acc, &entry->localRank, std::uint32_t{1u}, onAcc::scope::block);
                std::uint32_t const j = entry->base + rank;
                std::uint32_t const chunk = table[maxBins + table[bin] + j / outCapacity];
                std::uint32_t const offset = j % outCapacity;

                Particle particle{idsIn[slotIn], propertiesIn[slotIn], weightsIn[slotIn], visitsIn[slotIn]};
                policy.apply(acc, bin, particle, &counts[maxBins], &capped[0u]);

                std::uint32_t const slotOut = chunk * outCapacity + offset;
                idsOut[slotOut] = particle.id;
                propertiesOut[slotOut] = particle.property;
                weightsOut[slotOut] = particle.weight;
                visitsOut[slotOut] = particle.visit;
            }
            onAcc::syncBlockThreads(acc);
        }
    };
} // namespace graphFlow
