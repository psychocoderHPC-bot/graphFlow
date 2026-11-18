/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace graphFlow
{
    /** Kernel mapping of the batch-wide transform stage.
     *
     * `batch` is the historical launch and stays the default: one block owns one input chunk and
     * the threads of the block stride over the chunk's payloads. `simdBlocks` keeps one block per
     * chunk but walks the batch chunk list with an index map over `blocksInGrid` and processes the
     * payloads through `onAcc::SimdAlgo::concurrent()`, so every thread handles SIMD packs of the
     * storage columns. `simdWarps` assigns one chunk per warp: the flat warp id walks the chunk
     * list and the lanes of that warp process the chunk's payloads with a warp-scoped
     * `SimdAlgo`.
     *
     * The mapping changes launch geometry only. The output position of a payload, the tail of a
     * partially filled chunk and the per-payload RNG stream stay exactly the batch semantics.
     * `simdBlocks` and `simdWarps` need a storage whose payload columns can be assembled lane by
     * lane, which is the SoA payload layout.
     */
    enum class TransformKernelMode : std::uint8_t
    {
        batch,
        simdBlocks,
        simdWarps
    };

    /** Diagnostic name of a transform kernel mapping. */
    inline constexpr char const* transformKernelModeName(TransformKernelMode mode) noexcept
    {
        switch(mode)
        {
        case TransformKernelMode::simdBlocks:
            return "simdBlocks";
        case TransformKernelMode::simdWarps:
            return "simdWarps";
        default:
            return "batch";
        }
    }
} // namespace graphFlow
