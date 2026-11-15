/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

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
     * The public `Flow` API selects the mode through `Tuning::scatterMode`. A low-level policy may
     * opt into the other variant by declaring `static constexpr ScatterMode scatterMode`.
     * Policies without the member keep the default, so no existing call site changes.
     */
    enum class ScatterMode : std::uint8_t
    {
        SharedStaging,
        Direct
    };
} // namespace graphFlow
