/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace graphFlow
{
    /** Static configuration of one graph run.
     *
     * `chunkCapacity` (C) is the number of particle slots per chunk, `numChunks` the pool size,
     * `tileSize` (T) the requested threads per block and the per-tile histogram width, and
     * `maxBins` the runtime upper bound for the compile-time bin count of a node.
     */
    struct Config
    {
        std::uint32_t chunkCapacity = 128u; // C
        std::uint32_t numChunks = 0u;       // S, zero means "derive from N"
        std::uint32_t tileSize = 128u;      // T
        std::uint32_t maxBins = 64u;        // B upper bound
        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint64_t numParticles = 0u; // N

        /** Largest bin count the dense shared table supports in this slice. */
        static constexpr std::uint32_t denseBinLimit = 1024u;

        void validate() const
        {
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::Config: chunkCapacity must be non-zero"};
            if(tileSize == 0u)
                throw std::invalid_argument{"graphFlow::Config: tileSize must be non-zero"};
            if(maxBins == 0u)
                throw std::invalid_argument{"graphFlow::Config: maxBins must be non-zero"};
            if(maxBins > denseBinLimit)
                throw std::invalid_argument{
                    "graphFlow::Config: maxBins " + std::to_string(maxBins) + " exceeds the dense bin limit of "
                    + std::to_string(denseBinLimit)};
        }
    };
} // namespace graphFlow
