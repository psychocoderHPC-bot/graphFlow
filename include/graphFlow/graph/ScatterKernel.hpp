/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/graph/Classifier.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow
{
    /** Dense scatter kernel for one input chunk.
     *
     * One thread block processes one input chunk tile by tile. Per tile the block:
     *  1. rebuilds the per-bin counts from the stored `binIds` (the classifier is not recomputed),
     *  2. reserves one device-scope range `base[bin] = atomicAdd(&next[bin], count, device)` per
     *     nonempty bin,
     *  3. assigns `j = base[bin] + localRank` with a block-scope shared cursor,
     *  4. resolves `outChunk = table[T_B + table[bin] + j / C]` and `offset = j % C`,
     *  5. applies the policy payload mutation and writes the payload fields to the output slot.
     *
     * Ranges of different tiles in the same bin are disjoint; a range may span chunk boundaries.
     * `counts[T_B]` is the status word and is set to 1 on a checked weight overflow.
     */
    template<typename TPolicy, std::uint32_t T_B = TPolicy::maxBins>
    struct ScatterKernel
    {
        static_assert(T_B > 0u, "a dense policy needs at least one bin");

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
            std::uint32_t fill) const
        {
            using namespace alpaka;

            auto tileCounts
                = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
            auto tileBase = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
            auto tileCursor
                = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});

            auto const threadExtents = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            std::uint32_t const tileWidth = static_cast<std::uint32_t>(threadExtents.product());

            for(std::uint32_t tileBaseIdx = 0u; tileBaseIdx < fill; tileBaseIdx += tileWidth)
            {
                std::uint32_t const particles = fill - tileBaseIdx;
                std::uint32_t const tileLength = particles < tileWidth ? particles : tileWidth;

                for(auto [bin] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                    tileCounts[bin] = 0u;
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{tileLength}))
                {
                    std::uint32_t const bin = binIds[inBase + tileBaseIdx + i];
                    onAcc::atomicAdd(acc, &tileCounts[bin], std::uint32_t{1u}, onAcc::scope::block);
                }
                onAcc::syncBlockThreads(acc);

                for(auto [bin] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                {
                    std::uint32_t const inTile = tileCounts[bin];
                    if(inTile != 0u)
                    {
                        tileBase[bin] = onAcc::atomicAdd(acc, &next[bin], inTile, onAcc::scope::device);
                        tileCursor[bin] = 0u;
                    }
                }
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{tileLength}))
                {
                    std::uint32_t const slotIn = inBase + tileBaseIdx + i;
                    std::uint32_t const bin = binIds[slotIn];
                    std::uint32_t const rank = onAcc::atomicAdd(acc, &tileCursor[bin], std::uint32_t{1u}, onAcc::scope::block);
                    std::uint32_t const j = tileBase[bin] + rank;
                    std::uint32_t const chunk = table[std::uint32_t{T_B} + table[bin] + j / outCapacity];
                    std::uint32_t const offset = j % outCapacity;

                    Particle particle{idsIn[slotIn], propertiesIn[slotIn], weightsIn[slotIn], visitsIn[slotIn]};
                    policy.apply(acc, bin, particle, &counts[T_B], &capped[0u]);

                    std::uint32_t const slotOut = chunk * outCapacity + offset;
                    idsOut[slotOut] = particle.id;
                    propertiesOut[slotOut] = particle.property;
                    weightsOut[slotOut] = particle.weight;
                    visitsOut[slotOut] = particle.visit;
                }
                // All threads must leave the tile before the next tile zeroes the shared tables.
                onAcc::syncBlockThreads(acc);
            }
        }
    };
} // namespace graphFlow
