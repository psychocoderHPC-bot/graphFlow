/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/flow/Debug.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/graph/TileDescriptor.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

namespace graphFlow
{
    /** Scatter strategy of the dense path.
     *
     * `SharedStaging` is the measured default. The block first rebuilds the per-bin counts in
     * shared memory, reserves one device-scope range per nonempty `(tile, bin)`, and then uses a
     * block-scope cursor to spread the particles of that range. It performs one device atomic per
     * nonempty bin instead of one per particle.
     *
     * `Direct` removes the shared histogram and every block barrier: each particle reserves its
     * own single slot with `atomicAdd(&next[bin], 1, device)` and writes immediately. It is the
     * compact reference for sparse tiles, but it issues many more contended device atomics. The
     * P03 benchmark (fwk394, Xeon W-2175, C=1024, T=128, N=1<<20, 14 OpenMP threads) measured
     * shared staging ahead on every distribution: balanced 5.48 vs 15.85 ms, one-hot 2.13 vs
     * 48.31 ms, skewed 2.21 vs 37.25 ms, sparse (B=1024) 4.19 vs 6.62 ms. The default is therefore
     * `SharedStaging`; `Direct` stays available as the compact correctness reference.
     *
     * A policy opts into the other variant by declaring `static constexpr ScatterMode scatterMode`.
     * Policies without the member keep the default, so no existing call site changes.
     */
    enum class ScatterMode : std::uint8_t
    {
        SharedStaging,
        Direct
    };

    namespace detail
    {
        /** Read the optional `TPolicy::scatterMode` knob; `SharedStaging` when it is absent. */
        template<typename TPolicy, typename = void>
        struct ScatterModeOf
        {
            static constexpr ScatterMode value = ScatterMode::SharedStaging;
        };

        template<typename TPolicy>
        struct ScatterModeOf<TPolicy, std::void_t<decltype(TPolicy::scatterMode)>>
        {
            static constexpr ScatterMode value = TPolicy::scatterMode;
        };
    } // namespace detail

    /** Dense scatter kernel for one batch of input chunks.
     *
     * The launch uses a frame specification. Each block strides over the batch-wide tile descriptor
     * list, so one block may process several tiles. The descriptor supplies `chunkIndex`,
     * `tileOffset` and `count`; the block then:
     *  1. rebuilds the per-bin counts from the stored `binIds` of that tile (the classifier is not
     *     recomputed),
     *  2. reserves one device-scope range `base[bin] = atomicAdd(&next[bin], count, device)` per
     *     nonempty bin,
     *  3. assigns `j = base[bin] + localRank` with a block-scope shared cursor,
     *  4. resolves `outChunk = table[T_B + table[bin] + j / C]` and `offset = j % C`,
     *  5. applies the policy payload mutation and writes the payload fields to the output slot.
     *
     * `next[bin]` is shared by all descriptors of the batch, so ranges of different tiles and of
     * different input chunks are disjoint; a range may span several output chunk boundaries. The
     * output table `[binBase[B], chunkIndex[R]]` is planned across the whole batch, exactly as the
     * single-chunk path did per chunk. `counts[T_B]` is the status word and is set to 1 on a checked
     * weight overflow. Every thread reaches every block barrier, including inactive tail lanes.
     *
     * `TMode == ScatterMode::Direct` skips the shared histogram and does one device atomic per
     * particle. It is kept as the measured alternative and the correctness reference; the default
     * is the shared-staging path.
     */
    template<
        typename TPolicy,
        std::uint32_t T_B = TPolicy::maxBins,
        ScatterMode TMode = detail::ScatterModeOf<TPolicy>::value>
    struct ScatterKernel
    {
        static_assert(T_B > 0u, "a dense policy needs at least one bin");

        TPolicy policy{};

        // Explicit parameters keep the abbreviated count below the nvcc 12.6 name-hiding ICE threshold.
        template<
            alpaka::concepts::IMdSpan TIdsIn,
            alpaka::concepts::IMdSpan TPropertiesIn,
            alpaka::concepts::IMdSpan TWeightsIn,
            alpaka::concepts::IMdSpan TVisitsIn,
            alpaka::concepts::IMdSpan TBinIds,
            alpaka::concepts::IMdSpan TIdsOut>
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            TIdsIn idsIn,
            TPropertiesIn propertiesIn,
            TWeightsIn weightsIn,
            TVisitsIn visitsIn,
            TBinIds binIds,
            TIdsOut idsOut,
            alpaka::concepts::IMdSpan auto propertiesOut,
            alpaka::concepts::IMdSpan auto weightsOut,
            alpaka::concepts::IMdSpan auto visitsOut,
            alpaka::concepts::IMdSpan auto table,
            alpaka::concepts::IMdSpan auto next,
            alpaka::concepts::IMdSpan auto counts,
            alpaka::concepts::IMdSpan auto capped,
            alpaka::concepts::IMdSpan auto tiles,
            std::uint32_t chunkCapacity) const
        {
            using namespace alpaka;

            std::uint32_t const numTiles = static_cast<std::uint32_t>(tiles.getExtents().x());

            if constexpr(TMode == ScatterMode::Direct)
            {
                // Debug-only counter of particles that reached the output write. Every read,
                // atomic and barrier is inside `if constexpr`, so release carries no counter
                // instruction and no extra block synchronization; only the 4-byte shared slot may
                // remain if the compiler keeps the unused declaration.
                auto& written = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

                for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
                {
                    std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                    TileDescriptor const tile = tiles[tileIndex];
                    std::uint32_t const inBase = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                    std::uint32_t const count = tile.count;

                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
                        {
                            alpaka::unused(one);
                            written = 0u;
                        }
                        onAcc::syncBlockThreads(acc);
                    }

                    for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                    {
                        std::uint32_t const slotIn = inBase + i;
                        std::uint32_t const bin = binIds[slotIn];
                        if constexpr(graphFlow::debug::checksEnabled)
                        {
                            if(bin >= T_B)
                            {
                                onAcc::atomicExch(
                                    acc,
                                    &counts[std::uint32_t{T_B}],
                                    std::uint32_t{1u},
                                    onAcc::scope::device);
                                continue;
                            }
                        }
                        std::uint32_t const j
                            = onAcc::atomicAdd(acc, &next[bin], std::uint32_t{1u}, onAcc::scope::device);
                        std::uint32_t const chunk = table[std::uint32_t{T_B} + table[bin] + j / chunkCapacity];
                        std::uint32_t const offset = j % chunkCapacity;

                        Particle particle{idsIn[slotIn], propertiesIn[slotIn], weightsIn[slotIn], visitsIn[slotIn]};
                        policy.apply(acc, bin, particle, &counts[T_B], &capped[0u]);

                        std::uint32_t const slotOut = chunk * chunkCapacity + offset;
                        idsOut[slotOut] = particle.id;
                        propertiesOut[slotOut] = particle.property;
                        weightsOut[slotOut] = particle.weight;
                        visitsOut[slotOut] = particle.visit;

                        if constexpr(graphFlow::debug::checksEnabled)
                            onAcc::atomicAdd(acc, &written, std::uint32_t{1u}, onAcc::scope::block);
                    }

                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        onAcc::syncBlockThreads(acc);
                        for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
                        {
                            alpaka::unused(one);
                            // A mismatch means a particle was dropped without a status word.
                            if(written != count)
                                onAcc::atomicExch(
                                    acc,
                                    &counts[std::uint32_t{T_B}],
                                    std::uint32_t{1u},
                                    onAcc::scope::device);
                        }
                    }
                }
            }
            else
            {
                auto tileCounts
                    = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
                auto tileBase
                    = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
                auto tileCursor
                    = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
                // Debug-only write counter; `if constexpr` guards every use.
                auto& written = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

                for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
                {
                    std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                    TileDescriptor const tile = tiles[tileIndex];
                    std::uint32_t const inBase = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                    std::uint32_t const count = tile.count;

                    for(auto [bin] :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                        tileCounts[bin] = 0u;
                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
                        {
                            alpaka::unused(one);
                            written = 0u;
                        }
                    }
                    onAcc::syncBlockThreads(acc);

                    for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                    {
                        std::uint32_t const slotIn = inBase + i;
                        std::uint32_t const bin = binIds[slotIn];
                        if constexpr(graphFlow::debug::checksEnabled)
                        {
                            if(bin >= T_B)
                            {
                                onAcc::atomicExch(
                                    acc,
                                    &counts[std::uint32_t{T_B}],
                                    std::uint32_t{1u},
                                    onAcc::scope::device);
                                continue;
                            }
                        }
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

                    for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                    {
                        std::uint32_t const slotIn = inBase + i;
                        std::uint32_t const bin = binIds[slotIn];
                        if constexpr(graphFlow::debug::checksEnabled)
                        {
                            // The count phase skipped an out-of-range bin; skip it here too so the
                            // validation build cannot index the shared tables out of bounds.
                            if(bin >= T_B)
                            {
                                onAcc::atomicExch(
                                    acc,
                                    &counts[std::uint32_t{T_B}],
                                    std::uint32_t{1u},
                                    onAcc::scope::device);
                                continue;
                            }
                        }
                        std::uint32_t const rank
                            = onAcc::atomicAdd(acc, &tileCursor[bin], std::uint32_t{1u}, onAcc::scope::block);
                        std::uint32_t const j = tileBase[bin] + rank;
                        std::uint32_t const chunk = table[std::uint32_t{T_B} + table[bin] + j / chunkCapacity];
                        std::uint32_t const offset = j % chunkCapacity;

                        Particle particle{idsIn[slotIn], propertiesIn[slotIn], weightsIn[slotIn], visitsIn[slotIn]};
                        policy.apply(acc, bin, particle, &counts[T_B], &capped[0u]);

                        std::uint32_t const slotOut = chunk * chunkCapacity + offset;
                        idsOut[slotOut] = particle.id;
                        propertiesOut[slotOut] = particle.property;
                        weightsOut[slotOut] = particle.weight;
                        visitsOut[slotOut] = particle.visit;

                        if constexpr(graphFlow::debug::checksEnabled)
                            onAcc::atomicAdd(acc, &written, std::uint32_t{1u}, onAcc::scope::block);
                    }

                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        onAcc::syncBlockThreads(acc);
                        for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
                        {
                            alpaka::unused(one);
                            if(written != count)
                                onAcc::atomicExch(
                                    acc,
                                    &counts[std::uint32_t{T_B}],
                                    std::uint32_t{1u},
                                    onAcc::scope::device);
                        }
                    }
                }
            }
        }
    };
} // namespace graphFlow
