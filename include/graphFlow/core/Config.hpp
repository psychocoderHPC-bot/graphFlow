/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/PoolBudget.hpp"

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
     * back, plus the augmented pool payload budget. A zero `policyMemoryBudgetBytes` derives the
     * policy budget from the device properties.
     *
     * `poolPayloadBudgetBytes` bounds the fixed-capacity pool payload arrays, not just the live
     * working set: `numChunks * chunkCapacity * slotBytes` must fit. `slotBytes` defaults to the
     * 24-byte particle SoA slot; a generic storage sets it to `sizeof(TPayload)`. The budget
     * itself defaults to the documented accelerator default. `applyDevicePolicy()` replaces it with
     * the policy memory budget plus `poolOutputReserveChunks` output-credit chunks, so the programmatic
     * `applyDevicePolicy` + `derivePoolSize` + `Config::validate` path is consistent. The CLI
     * passes the exact graph reserve instead. Zero is rejected because it would disable the cap.
     * The slot-based pool remains the next memory optimization; this budget only makes the
     * current fixed-capacity design explicit and enforceable.
     */
    struct Config
    {
        std::uint32_t chunkCapacity = 128u; // C
        std::uint32_t numChunks = 0u; // S = P + K*R, zero means "derive"
        std::uint32_t liveParticles = 0u; // P, zero means "derive from N and C"
        std::uint32_t tileSize = 128u; // T
        std::uint32_t maxBins = 64u; // B upper bound
        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint64_t numParticles = 0u; // N
        std::uint32_t batchChunks = 1u; // Q, ready chunks per transform batch
        std::uint32_t frameCap = 8u; // blocks per multiprocessor cap of the frame launch
        std::uint64_t poolPayloadBudgetBytes = defaultPoolPayloadBudgetBytes; // total pool cap
        std::uint64_t poolOutputReserveChunks = defaultPoolOutputReserveChunks; // K*R reserve

        // Device-aware chunk policy knobs, see include/graphFlow/core/DevicePolicy.hpp.
        std::uint32_t policyWaves = 1u;
        std::uint32_t policyTargetBlocksPerSm = 4u;
        std::uint32_t policyLogicalTileWidth = 128u;
        std::uint32_t policyMaxChunkCapacity = 8192u;
        std::uint32_t policyLiveParticlesPerChunk = 2u;
        std::uint64_t policyMemoryBudgetBytes = 0u; // zero means "derive from the device"
        /** Payload bytes of one storage slot; the particle SoA default is 24. */
        std::uint64_t slotBytes = payloadBytesPerSlot;

        /** Largest bin count the dense shared table supports in this slice.
         *
         * This is a dense-path limit. The hash path sizes its shared table from the tile width,
         * not from `maxBins`, so a hash classifier may declare more bins. `DenseNodePolicy`
         * enforces this limit at compile time for the dense kernels; `Config` only rejects a
         * zero bin count here and leaves the hash upper bound to the partition implementation.
         *
         * Measured on fwk394 (Xeon W-2175, C=1024, T=128, N=1<<20, 14 OpenMP threads): the dense
         * table beat the 2*T and 4*T hash tables at B=128, 256 (2*T), 512 (4*T), and 1024, e.g.
         * B=1024: dense 4.06 ms vs 2*T 5.11 ms and 4*T 4.90 ms. The dense/hash crossover is
         * therefore at or above 1024 on the CPU reference, so the default stays 1024. The 4*T table
         * only narrowed the hash gap at B>=512 and was never faster than dense, so the default
         * hash capacity stays 2*T. On an accelerator the 3*B*4-byte dense shared footprint limits
         * occupancy before the CPU crossover; use `denseBinLimitForSharedMemory()` when a caller
         * has the device properties.
         */
        static constexpr std::uint32_t denseBinLimit = 1024u;

        /** Shared-memory words the dense scatter stages per bin.
         *
         * `ScatterKernel` keeps three `uint32_t` arrays indexed by bin in block-shared memory:
         * the per-tile counts, the device-range base, and the block cursor. The classify kernel
         * keeps one more. A device policy that wants the dense path must leave room for the
         * scatter's three tables, so this is the conservative per-bin cost.
         */
        static constexpr std::uint64_t denseScatterWordsPerBin = 3u;

        /** Device-specific dense bin cap for the block-shared scatter tables.
         *
         * Keeps `denseScatterWordsPerBin` words per bin inside `1/sharedFractionDenominator` of
         * the block-shared budget so two resident blocks still fit. The result never exceeds the
         * measured `denseBinLimit`; devices with less shared memory get a smaller dense cap and
         * use the hash path above it. With the CI host and the A30 (64 KiB) the cap is the
         * measured 1024, so 2*T and 4*T dense tables fit for T=128. This helper does not change
         * any default; callers that have `DeviceProperties` can use it to pick a table.
         */
        [[nodiscard]] static constexpr std::uint32_t denseBinLimitForSharedMemory(
            std::uint64_t sharedMemPerBlockBytes,
            std::uint64_t sharedFractionDenominator = 4u) noexcept
        {
            if(sharedFractionDenominator == 0u)
                return 0u;
            std::uint64_t const bytesPerBin = denseScatterWordsPerBin * sizeof(std::uint32_t);
            std::uint64_t const limit = (sharedMemPerBlockBytes / sharedFractionDenominator) / bytesPerBin;
            return static_cast<std::uint32_t>(limit < denseBinLimit ? limit : denseBinLimit);
        }

        void validate() const
        {
            if(chunkCapacity == 0u)
                throw std::invalid_argument{"graphFlow::Config: chunkCapacity must be non-zero"};
            if(poolPayloadBudgetBytes == 0u)
                throw std::invalid_argument{"graphFlow::Config: poolPayloadBudgetBytes must be non-zero"};
            if(numChunks != 0u)
                validatePayloadBudget(
                    numChunks,
                    chunkCapacity,
                    poolPayloadBudgetBytes,
                    "graphFlow::Config",
                    slotBytes);
            if(tileSize == 0u)
                throw std::invalid_argument{"graphFlow::Config: tileSize must be non-zero"};
            if(batchChunks == 0u)
                throw std::invalid_argument{"graphFlow::Config: batchChunks must be non-zero"};
            if(frameCap == 0u)
                throw std::invalid_argument{"graphFlow::Config: frameCap must be non-zero"};
            if(maxBins == 0u)
                throw std::invalid_argument{"graphFlow::Config: maxBins must be non-zero"};
            // No denseBinLimit check here: the hash path supports larger partition widths. The
            // dense shared-table limit is still enforced by DenseNodePolicy and the dense
            // kernels, which instantiate their shared counts[maxBins] table at compile time.
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
