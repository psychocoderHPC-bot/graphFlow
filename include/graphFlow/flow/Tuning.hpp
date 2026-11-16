/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/BinPath.hpp"
#include "graphFlow/core/ChunkKernelSchedule.hpp"
#include "graphFlow/core/ScatterMode.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace graphFlow
{
    /** Optional execution knobs of `Flow`.
     *
     * Absent values are derived from the device, the run size and the graph. Tuning never changes
     * payload semantics or routes. Every present value must be non-zero; zero has no distinct
     * "automatic" meaning here because absence already selects the automatic path.
     */
    struct Tuning
    {
        /** Root RNG seed of this run. graphFlow derives every per-item stream from it. */
        std::uint64_t seed = 0u;

        // Batching.
        std::optional<std::uint32_t> batchChunks;
        /** Maximum number of concurrently in-flight batches per node.
         *
         * Absence keeps the historical one-batch-per-node scheduling. A value above one lets the
         * runner submit several independent batches of the same node before the first completes,
         * which overlaps host planning with device work and bounds the accepted work by the event
         * set of each node. The default never changes.
         */
        std::optional<std::uint32_t> inflightBatches;

        // Per-chunk transform schedule. Absence keeps `ChunkKernelSchedule::off`, the batch-wide
        // transform launch; `perChunk` is opt-in and changes launch geometry only.
        std::optional<ChunkKernelSchedule> chunkKernelSchedule;
        /** Chunk queues the per-chunk transform path owns; absence keeps one serial queue. */
        std::optional<std::uint32_t> chunkQueueDepth;
        /** Per-chunk latency cap in microseconds; zero means "no cap" and is a valid value. */
        std::optional<std::uint32_t> chunkLatencyCapUs;

        // Storage and pool limits.
        std::optional<std::uint32_t> chunkCapacity;
        std::optional<std::uint32_t> liveParticles;
        std::optional<std::uint32_t> tileSize;
        std::optional<std::uint32_t> numChunks;
        std::optional<std::uint32_t> frameCap;
        std::optional<std::uint64_t> poolPayloadBudgetBytes;

        // Measured partition-path axes.
        /** Dense scatter strategy. Absence keeps `ScatterMode::SharedStaging`.
         *
         * `Direct` skips the per-tile shared histogram and reserves one device slot per particle.
         * It is the compact reference and is only valid for the dense bin path; the run is
         * rejected when combined with `BinPath::hash`.
         */
        std::optional<ScatterMode> scatterMode;
        /** Bin-table path of every partition node. Absence keeps `BinPath::automatic`.
         *
         * `automatic` selects dense up to `Config::denseBinLimit` bins and hash above it. An
         * explicit `dense` above the limit, or a hash run with a tile size above
         * `Config::hashBinTableTileSize`, is rejected before any device work.
         */
        std::optional<BinPath> binPath;

        // Device-policy knobs; see core/DevicePolicy.hpp.
        std::optional<std::uint32_t> policyWaves;
        std::optional<std::uint32_t> policyTargetBlocksPerSm;
        std::optional<std::uint32_t> policyLogicalTileWidth;
        std::optional<std::uint32_t> policyMaxChunkCapacity;
        std::optional<std::uint32_t> policyLiveParticlesPerChunk;
        std::optional<std::uint64_t> policyMemoryBudgetBytes;

        /** Opt-in scheduler diagnostics of the concrete runner. */
        bool diagnostics = false;

        /** Maximum scheduler quiet time before `Flow::run` reports a stall. */
        std::optional<std::chrono::milliseconds> stallTimeout;

        /** Reject present zero values before they reach the scheduler or the pool. */
        void validate() const
        {
            validateNonZero(batchChunks, "batchChunks");
            validateNonZero(inflightBatches, "inflightBatches");
            // `chunkLatencyCapUs` is deliberately excluded: zero means "no cap" and is valid.
            validateNonZero(chunkQueueDepth, "chunkQueueDepth");
            validateNonZero(chunkCapacity, "chunkCapacity");
            validateNonZero(liveParticles, "liveParticles");
            validateNonZero(tileSize, "tileSize");
            validateNonZero(numChunks, "numChunks");
            validateNonZero(frameCap, "frameCap");
            validateNonZero(poolPayloadBudgetBytes, "poolPayloadBudgetBytes");
            validateNonZero(policyWaves, "policyWaves");
            validateNonZero(policyTargetBlocksPerSm, "policyTargetBlocksPerSm");
            validateNonZero(policyLogicalTileWidth, "policyLogicalTileWidth");
            validateNonZero(policyMaxChunkCapacity, "policyMaxChunkCapacity");
            validateNonZero(policyLiveParticlesPerChunk, "policyLiveParticlesPerChunk");
            validateNonZero(policyMemoryBudgetBytes, "policyMemoryBudgetBytes");
            if(stallTimeout.has_value() && stallTimeout->count() <= 0)
                throw std::invalid_argument{"graphFlow::Tuning: stallTimeout must be positive"};
        }

    private:
        template<typename TValue>
        static void validateNonZero(std::optional<TValue> const& value, char const* name)
        {
            if(value.has_value() && *value == 0u)
                throw std::invalid_argument{std::string{"graphFlow::Tuning: "} + name + " must be non-zero"};
        }
    };
} // namespace graphFlow
