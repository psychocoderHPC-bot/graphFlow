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
     * `chunkCapacity` (C) is the number of particle slots per chunk, `numChunks` (S) the pool size,
     * `liveParticles` (P) the configured source-admission capacity in particles, `tileSize` (T) the
     * requested threads per block and the per-tile histogram width, `batchChunks` (Q) the maximum
     * number of ready chunks one transform batch dispatches, and `maxBins` (B) the runtime upper
     * bound for the compile-time bin count of a node.
     *
     * The pool is `S = P + K*R` with `R = outputChunkBound(C, C, B)` for K transform nodes, while a
     * transform batch reserves the adaptive `outputChunkBound(sum(fill), C, B)` for the chunks it
     * actually takes. `batchChunks` (Q) caps the batch size; under fragmentation a smaller prefix
     * is dispatched, so Q does not enlarge the pool.
     *
     * The `policy*` fields are the knobs of the device-aware chunk policy in
     * `graphFlow/core/DevicePolicy.hpp`: `makeDevicePolicy` reads them through `policyConfigOf()`
     * and `applyDevicePolicy()` writes the chosen `chunkCapacity`, `liveParticles` and `tileSize`
     * back. A zero `policyMemoryBudgetBytes` derives the budget from the device properties.
     */
    struct Config
    {
        std::uint32_t chunkCapacity = 128u; // C
        std::uint32_t numChunks = 0u;       // S = P + K*R, zero means "derive"
        std::uint32_t liveParticles = 0u;   // P, zero means "derive from N and C"
        std::uint32_t tileSize = 128u;      // T
        std::uint32_t maxBins = 64u;        // B upper bound
        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint64_t numParticles = 0u; // N
        std::uint32_t batchChunks = 1u; // Q, ready chunks per transform batch

        // Device-aware chunk policy knobs, see include/graphFlow/core/DevicePolicy.hpp.
        std::uint32_t policyWaves = 1u;
        std::uint32_t policyTargetBlocksPerSm = 4u;
        std::uint32_t policyLogicalTileWidth = 128u;
        std::uint32_t policyMaxChunkCapacity = 8192u;
        std::uint32_t policyLiveParticlesPerChunk = 2u;
        std::uint64_t policyMemoryBudgetBytes = 0u; // zero means "derive from the device"

        /** Largest bin count the dense shared table supports in this slice. */
        static constexpr std::uint32_t denseBinLimit = 1024u;

        void validate() const
        {
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::Config: chunkCapacity must be non-zero"};
            if(tileSize == 0u)
                throw std::invalid_argument{"graphFlow::Config: tileSize must be non-zero"};
            if(batchChunks == 0u)
                throw std::invalid_argument{"graphFlow::Config: batchChunks must be non-zero"};
            if(maxBins == 0u)
                throw std::invalid_argument{"graphFlow::Config: maxBins must be non-zero"};
            if(maxBins > denseBinLimit)
                throw std::invalid_argument{
                    "graphFlow::Config: maxBins " + std::to_string(maxBins) + " exceeds the dense bin limit of "
                    + std::to_string(denseBinLimit)};
            if(policyWaves == 0u)
                throw std::invalid_argument{"graphFlow::Config: policyWaves must be non-zero"};
            if(policyTargetBlocksPerSm == 0u)
                throw std::invalid_argument{"graphFlow::Config: policyTargetBlocksPerSm must be non-zero"};
            if(policyLogicalTileWidth == 0u)
                throw std::invalid_argument{"graphFlow::Config: policyLogicalTileWidth must be non-zero"};
            if(policyMaxChunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::Config: policyMaxChunkCapacity must be non-zero"};
            if(policyLiveParticlesPerChunk == 0u)
                throw std::invalid_argument{"graphFlow::Config: policyLiveParticlesPerChunk must be non-zero"};
        }
    };
} // namespace graphFlow
