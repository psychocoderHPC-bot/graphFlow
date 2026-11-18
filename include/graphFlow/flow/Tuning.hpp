/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/BinPath.hpp"
#include "graphFlow/core/ChunkKernelSchedule.hpp"
#include "graphFlow/core/PoolMode.hpp"
#include "graphFlow/core/ScatterMode.hpp"
#include "graphFlow/core/TransformKernelMode.hpp"

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
     * payload semantics or routes. Every present value must be non-zero, except the two knobs
     * whose zero is a defined setting: `chunkLatencyCapUs` (no cap) and `occupancyTargetChunks`
     * (gate disabled, matching `Config::occupancyTargetChunks`).
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
        /** Kernel mapping of the batch-wide transform stage.
         *
         * Absence keeps `Config::transformKernelMode` (`batch`). `simdBlocks` assigns one chunk
         * per block and `simdWarps` one chunk per warp, both walking the batch chunk list with an
         * index map and processing the payloads through `onAcc::SimdAlgo::concurrent()`. The
         * mapping changes launch geometry only; output positions, tails and RNG streams stay the
         * batch semantics. The SIMD mappings require a SoA payload storage.
         */
        std::optional<TransformKernelMode> transformKernelMode;

        // Storage and pool limits.
        std::optional<std::uint32_t> chunkCapacity;
        std::optional<std::uint32_t> liveParticles;
        /** Steady-state chunk occupancy target D. Absence and an explicit zero both keep the
         * historical particle-budget admission; a non-zero value adds the chunk-count gate
         * described in `Config`.
         */
        std::optional<std::uint32_t> occupancyTargetChunks;
        std::optional<std::uint32_t> tileSize;
        std::optional<std::uint32_t> numChunks;
        std::optional<std::uint32_t> frameCap;
        std::optional<std::uint64_t> poolPayloadBudgetBytes;
        /** Payload budget of the slot arena in bytes; absence and zero keep the whole pool budget.
         *
         * This sizes the arena independently of `poolPayloadBudgetBytes`, which is what lets a
         * user realize the memory reduction of the slot layout: the payload array becomes
         * `floor(slotArenaBudgetBytes / slotBytes)` slots instead of the whole pool budget. The
         * run is rejected before any kernel launches when the `P + K*R*C` reservation does not
         * fit the bytes, with the required and available slot counts.
         */
        std::optional<std::uint64_t> slotArenaBudgetBytes;
        /** Chunk-slot layout; absence keeps `Config::poolMode` (the fixed layout).
         *
         * `slotArena` selects the phase-2 layout and its `P + K*R*C` reservation. The run is
         * rejected when the arena does not fit `slotArenaBudgetBytes` (or `poolPayloadBudgetBytes`
         * when the arena knob is absent).
         */
        std::optional<PoolMode> poolMode;
        /** Validation-only stale base-table injection; see `Config::debugStaleBaseTable`. */
        bool debugStaleBaseTable = false;

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
            // `occupancyTargetChunks` is excluded as well: zero disables the gate, which is the
            // same semantics `Config` documents, so an explicit `--occupancy-target 0` is an
            // off switch rather than an error.
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
