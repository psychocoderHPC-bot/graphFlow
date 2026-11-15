/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace graphFlow
{
    /** Bin-table implementation of a partition node.
     *
     * `dense` keeps one block-shared counter per bin and is the measured default up to
     * `Config::denseBinLimit` bins. `hash` uses the open-addressing table sized from the tile
     * width, so it supports many bins without a `bins`-word shared array. It always stages through
     * shared memory and therefore has no `ScatterMode::Direct` variant.
     *
     * `automatic` is the default of `Config` and `Tuning`: it resolves to `dense` when the
     * partition width fits `Config::denseBinLimit` and to `hash` above it. The resolved value is
     * written back into the run configuration before any stage is constructed.
     */
    enum class BinPath : std::uint8_t
    {
        automatic,
        dense,
        hash
    };
} // namespace graphFlow
