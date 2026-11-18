/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/PoolBudget.hpp"

#include <cstdint>

namespace graphFlow
{
    /** Chunk-slot memory layout of a storage.
     *
     * `fixed` is the historical layout and stays the default: chunk `k` owns the fixed row
     * `[k * chunkCapacity, (k + 1) * chunkCapacity)`, so a fragment of one particle still costs a
     * full chunk of slots. `slotArena` gives every chunk a contiguous range of exactly the fill it
     * commits at acquisition, so a one-particle fragment owns one slot; the range start is stored
     * per chunk instead of being derived from the chunk index.
     *
     * `slotArena` changes how a chunk's slots are found, so a storage built in that mode resolves
     * the base through a per-chunk table. The partition and transform stages copy that table to a
     * device mirror on the node queue immediately before the kernel that indexes it, so a kernel
     * never recomputes `index * chunkCapacity`. `Config::validate()` checks the arena reservation
     * `P + K*R*C` against the payload budget; the fixed layout keeps its closed-form base.
     */
    enum class PoolMode
    {
        fixed,
        slotArena
    };

    /** Diagnostic name of a pool mode. */
    inline constexpr char const* poolModeName(PoolMode mode) noexcept
    {
        return mode == PoolMode::slotArena ? "slotArena" : "fixed";
    }

    /** Geometry of a slot-arena storage.
     *
     * `arenaSlots` is the total slot count of the single payload array. It must fit the 32-bit
     * kernel slot index and at least one slot per chunk index. A zero `arenaSlots` derives the size
     * from `payloadBudgetBytes` as `budget / slotBytes`; the fixed layout derives its chunk count as
     * `budget / (chunkCapacity * slotBytes)` instead.
     */
    struct SlotArenaOptions
    {
        std::uint64_t payloadBudgetBytes = defaultPoolPayloadBudgetBytes;
        std::uint64_t arenaSlots = 0u; // zero means "derive from the payload budget"
    };
} // namespace graphFlow
