/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

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
        // Batching.
        std::optional<std::uint32_t> batchChunks;

        // Storage and pool limits.
        std::optional<std::uint32_t> chunkCapacity;
        std::optional<std::uint32_t> liveParticles;
        std::optional<std::uint32_t> tileSize;
        std::optional<std::uint32_t> numChunks;
        std::optional<std::uint32_t> frameCap;
        std::optional<std::uint64_t> poolPayloadBudgetBytes;

        // Device-policy knobs; see core/DevicePolicy.hpp.
        std::optional<std::uint32_t> policyWaves;
        std::optional<std::uint32_t> policyTargetBlocksPerSm;
        std::optional<std::uint32_t> policyLogicalTileWidth;
        std::optional<std::uint32_t> policyMaxChunkCapacity;
        std::optional<std::uint32_t> policyLiveParticlesPerChunk;
        std::optional<std::uint64_t> policyMemoryBudgetBytes;

        /** Opt-in scheduler diagnostics of the concrete runner. */
        bool diagnostics = false;

        /** Reject present zero values before they reach the scheduler or the pool. */
        void validate() const
        {
            validateNonZero(batchChunks, "batchChunks");
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
