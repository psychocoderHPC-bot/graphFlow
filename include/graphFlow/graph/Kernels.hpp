/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow
{
    /** Source kernel: materialise `fill` fresh particles into one chunk.
     *
     * The particle id is `firstId + i`, independent of the chunk index, the tile, the thread and
     * the scheduling order. `property` is the Philox draw at offset 0; `weight` and `visit` start at
     * zero. The kernel writes every field it owns, so the chunk needs no host-side initialisation.
     */
    struct GenerateKernel
    {
        std::uint64_t seed = 0u;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto ids,
            alpaka::concepts::IMdSpan auto properties,
            alpaka::concepts::IMdSpan auto weights,
            alpaka::concepts::IMdSpan auto visits,
            std::uint64_t firstId,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint32_t const slot = base + i;
                std::uint64_t const id = firstId + i;
                ids[slot] = id;
                properties[slot] = rng::propertyDraw(seed, id);
                weights[slot] = 0u;
                visits[slot] = 0u;
            }
        }
    };

    /** Sink kernel: write the final weight of each particle into the id-indexed result buffer. */
    struct SinkKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto result,
            alpaka::concepts::IMdSpan auto ids,
            alpaka::concepts::IMdSpan auto weights,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint32_t const slot = base + i;
                result[static_cast<std::size_t>(ids[slot])] = weights[slot];
            }
        }
    };
} // namespace graphFlow
