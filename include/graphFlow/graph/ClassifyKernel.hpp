/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/graph/Classifier.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow
{
    /** Dense histogram kernel for one input chunk.
     *
     * One thread block processes one chunk. The chunk is walked tile by tile; tile width is the
     * number of real threads in the block, so the same code is correct for a single-threaded host
     * block and for a multi-threaded accelerator block. Every thread reaches every block barrier,
     * including threads that own no particle in a tail tile.
     *
     * Per tile the kernel builds a shared `counts[T_B]` histogram with block-scope atomics, then
     * flushes every nonempty bin into the device-scope `counts[bin]` total. `counts[T_B]` is the
     * status word and is set to 1 on an out-of-range classifier result; the particle is dropped in
     * that case and never indexes outside `[0, T_B)`.
     */
    template<typename TPolicy, std::uint32_t T_B = TPolicy::maxBins>
    struct ClassifyKernel
    {
        static_assert(T_B > 0u, "a dense policy needs at least one bin");

        TPolicy policy{};

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto ids,
            alpaka::concepts::IMdSpan auto properties,
            alpaka::concepts::IMdSpan auto weights,
            alpaka::concepts::IMdSpan auto visits,
            alpaka::concepts::IMdSpan auto binIds,
            alpaka::concepts::IMdSpan auto counts,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;

            auto tileCounts
                = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});

            auto const threadExtents = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            std::uint32_t const tileWidth = static_cast<std::uint32_t>(threadExtents.product());

            for(std::uint32_t tileBase = 0u; tileBase < fill; tileBase += tileWidth)
            {
                std::uint32_t const particles = fill - tileBase;
                std::uint32_t const tileLength = particles < tileWidth ? particles : tileWidth;

                for(auto [bin] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                    tileCounts[bin] = 0u;
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{tileLength}))
                {
                    std::uint32_t const slot = base + tileBase + i;
                    Particle particle{ids[slot], properties[slot], weights[slot], visits[slot]};
                    std::uint32_t const bin
                        = policy.classify(particle.id, particle.property, particle.weight, particle.visit);
                    if(bin >= T_B)
                    {
                        // Hard batch error: report and never index outside the shared table.
                        counts[T_B] = 1u;
                        continue;
                    }
                    binIds[slot] = bin;
                    onAcc::atomicAdd(acc, &tileCounts[bin], std::uint32_t{1u}, onAcc::scope::block);
                }
                onAcc::syncBlockThreads(acc);

                for(auto [bin] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                {
                    std::uint32_t const inTile = tileCounts[bin];
                    if(inTile != 0u)
                        onAcc::atomicAdd(acc, &counts[bin], inTile, onAcc::scope::device);
                }
                // All threads must leave the tile before the next tile zeroes the shared table.
                onAcc::syncBlockThreads(acc);
            }
        }
    };
} // namespace graphFlow
