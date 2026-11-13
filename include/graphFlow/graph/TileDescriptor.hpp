/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace graphFlow
{
    /** One tile of one input chunk of a transform batch.
     *
     * The host flattens every input chunk of a batch into a compact descriptor list and copies it
     * to the device once. The classify and scatter launches use one thread block per descriptor, so
     * a batch is two launches regardless of how many chunks or tiles it carries.
     *
     * `chunkIndex` selects the chunk in the pool, `tileOffset` is the first particle slot of the
     * tile within the chunk and `count` is the number of particles this tile owns
     * (`count <= tileSize`, the tail tile is shorter). The kernels index payload and metadata with
     * the 32-bit expression `chunkIndex * chunkCapacity + tileOffset + i`. `ChunkPool` rejects
     * `numChunks * chunkCapacity > UINT32_MAX`, so every slot index is representable and the
     * descriptor arithmetic cannot wrap.
     */
    struct TileDescriptor
    {
        std::uint32_t chunkIndex = 0u;
        std::uint32_t tileOffset = 0u;
        std::uint32_t count = 0u;
    };
} // namespace graphFlow
