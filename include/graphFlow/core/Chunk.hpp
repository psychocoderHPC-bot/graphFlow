/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace graphFlow
{
    /** Stable reference to a pooled chunk.
     *
     * The generation is incremented every time the chunk is returned to the pool. A handle whose
     * generation does not match the pool's current generation refers to a retired chunk and is
     * rejected instead of silently aliasing new data.
     */
    struct ChunkHandle
    {
        std::uint32_t index = 0u;
        std::uint32_t generation = 0u;

        friend constexpr bool operator==(ChunkHandle const& lhs, ChunkHandle const& rhs) = default;
    };

    /** Host-visible description of a filled chunk.
     *
     * Only nonempty chunks are published. Every device read and write is bounded by `fill`.
     * `bin` carries the bin identity assigned by the producing node.
     */
    struct ChunkDescriptor
    {
        ChunkHandle handle{};
        std::uint32_t fill = 0u;
        std::uint32_t bin = 0u;
    };
} // namespace graphFlow
