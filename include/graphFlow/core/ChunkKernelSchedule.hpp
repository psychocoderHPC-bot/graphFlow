/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace graphFlow
{
    /** Kernel schedule of the transform stage.
     *
     * `off` keeps the historical batch-wide transform launch: one kernel walks every chunk of the
     * batch and assigns one block per chunk. `perChunk` issues one kernel launch per input chunk
     * and spreads that chunk over its own grid, so a partial tail is launched with its own fill
     * and never shares a launch with a neighbour.
     *
     * The switch changes launch geometry only. The storage chunk capacity, the pool size, the
     * payload layout and the host-side output-position planning stay what the device policy
     * derived.
     */
    enum class ChunkKernelSchedule : std::uint8_t
    {
        off,
        perChunk
    };
} // namespace graphFlow
