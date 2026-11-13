/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/PoolBudget.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace graphFlow
{
    /** Backwards-compatible name of the per-slot payload size.
     *
     * The value is derived once from the pool array types in `PoolBudget.hpp`; all budget
     * arithmetic uses `payloadBytesPerSlot`. Per-chunk host metadata and the `O(N)` result
     * buffer of the example are counted separately.
     */
    inline constexpr std::uint64_t payloadBytesPerParticle = payloadBytesPerSlot;

    /** Default payload budget of the host backend in bytes.
     *
     * The host executor runs one real thread per block, so a larger chunk pool buys no device
     * parallelism. 16 MiB keeps the host default near the previous 128-particle working set while
     * still allowing a reasonable chunk capacity; `Config::policyMemoryBudgetBytes` overrides it.
     */
    inline constexpr std::uint64_t hostMemoryBudgetBytes = 16ull * 1024ull * 1024ull;

    /** Upper payload budget for an accelerator in bytes.
     *
     * An accelerator uses one eighth of the reported device memory, capped here at 3 GiB. On a
     * 24 GiB device this selects `C=8192`, `P=16384`, about 3 GiB of payload instead of the
     * previous 12.9 GB `C=8192`, `P=65536` pool. `Config::policyMemoryBudgetBytes` overrides it.
     */
    inline constexpr std::uint64_t acceleratorMemoryBudgetCapBytes = 3ull * 1024ull * 1024ull * 1024ull;

    /** Bytes per entry of the per-tile shared-memory hash table.
     *
     * `{uint32 key, uint32 count, uint32 base, uint32 localRank}`; the default hash path sizes the
     * table to `2*T` entries. The default tile width therefore leaves `2*T*16` bytes of shared
     * memory free. An explicit T is not constrained by this estimate.
     */
    inline constexpr std::uint64_t hashTableEntryBytes = 16u;

    /** Knobs of the device-aware chunk policy. */
    struct DevicePolicyConfig
    {
        /** Number of device waves the chunk capacity should cover. */
        std::uint32_t waves = 1u;
        /** Blocks per multiprocessor used by the wave estimate; no occupancy API is consulted. */
        std::uint32_t targetBlocksPerSm = 4u;
        /** Logical tile width to aim for before rounding to the warp size. */
        std::uint32_t logicalTileWidth = 128u;
        /** Configured upper bound for the chunk capacity C. */
        std::uint32_t maxChunkCapacity = 8192u;
        /** Default P as a multiple of C (`P = min(numParticles, factor * C)` before clamping). */
        std::uint32_t liveParticlesPerChunk = 2u;
        /** Explicit payload budget in bytes; zero derives it from the device properties. */
        std::uint64_t memoryBudgetBytes = 0u;
        /** Fraction of `globalMemCapacityBytes` used by the derived accelerator budget. */
        std::uint32_t memoryBudgetDivisor = 8u;
        /** Derived budget of the host backend (warp size one). */
        std::uint64_t hostBudgetBytes = hostMemoryBudgetBytes;
        /** Upper bound of the derived accelerator budget. */
        std::uint64_t acceleratorBudgetCapBytes = acceleratorMemoryBudgetCapBytes;
    };

    /** Optional explicit overrides; an absent value is derived from the device.
     *
     * `chunkCapacity` and `tileSize` are preserved exactly when they fit the device limits.
     * `liveParticles` is preserved exactly when it fits the payload budget and rejected with
     * `std::invalid_argument` when it does not; only the auto-derived P is clamped. Zero or a
     * value beyond the device limits or `numParticles` is rejected with `std::invalid_argument`.
     */
    struct DevicePolicyOverrides
    {
        std::optional<std::uint32_t> chunkCapacity; // C
        std::optional<std::uint32_t> liveParticles; // P
        std::optional<std::uint32_t> tileSize;      // T
    };

    /** Result of the device-aware chunk policy. */
    struct DevicePolicy
    {
        std::uint32_t chunkCapacity = 0u;     // C
        std::uint32_t liveParticles = 0u;     // P
        bool liveParticlesExplicit = false;   // P came from an override, not the default
        std::uint32_t tileSize = 0u;          // T
        std::uint32_t waves = 0u;             // wave target used for the estimate
        std::uint32_t targetBlocksPerSm = 0u; // blocks per multiprocessor used for the estimate
        std::uint64_t memoryBudgetBytes = 0u; // resolved payload budget
        std::string description;              // short human-readable summary
    };

    namespace detail
    {
        /** Multiplication that saturates at `uint64` maximum instead of overflowing. */
        inline std::uint64_t saturatedProduct(std::uint64_t lhs, std::uint64_t rhs)
        {
            if(lhs != 0u && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
                return std::numeric_limits<std::uint64_t>::max();
            return lhs * rhs;
        }

        /** Whether `liveParticles * chunkCapacity * payloadBytesPerSlot` fits the budget.
         *
         * Both products are checked with division so no intermediate value can overflow.
         */
        inline bool payloadFitsBudget(
            std::uint64_t liveParticles,
            std::uint64_t chunkCapacity,
            std::uint64_t budgetBytes)
        {
            if(liveParticles == 0u || chunkCapacity == 0u)
                return true;
            if(liveParticles > budgetBytes / chunkCapacity)
                return false;
            return liveParticles * chunkCapacity <= budgetBytes / payloadBytesPerSlot;
        }

        /** Largest C in `[0, upperBound]` whose default P fits the budget.
         *
         * `memory(C) = min(numParticles, particlesPerChunk * C) * C * payloadBytes` is
         * nondecreasing, so a binary search finds the largest capacity whose default P fits.
         * This keeps the host default near its previous working set when the fixed host budget
         * is small; on accelerators C is usually near the documented value and P is clamped.
         */
        inline std::uint64_t largestChunkForBudget(
            std::uint64_t numParticles,
            std::uint64_t particlesPerChunk,
            std::uint64_t budgetBytes,
            std::uint64_t upperBound)
        {
            std::uint64_t lower = 0u;
            std::uint64_t upper = upperBound;
            while(lower < upper)
            {
                std::uint64_t const middle = lower + (upper - lower + 1u) / 2u;
                std::uint64_t const live = std::min(numParticles, saturatedProduct(particlesPerChunk, middle));
                if(payloadFitsBudget(live, middle, budgetBytes))
                    lower = middle;
                else
                    upper = middle - 1u;
            }
            return lower;
        }

        inline std::uint64_t roundDownToMultiple(std::uint64_t value, std::uint64_t multiple)
        {
            return value / multiple * multiple;
        }

        /** Largest live-particle capacity that fits the budget for a fixed chunk capacity.
         *
         * The budget caps the live working set `P * C * payloadBytesPerSlot`, not the chunk
         * capacity. The result is rounded down to a whole tile when that leaves at least one
         * tile, so the default clamp does not sit exactly on the usable-memory boundary. A
         * result of zero means not even one live particle fits.
         */
        inline std::uint64_t largestLiveParticlesForBudget(
            std::uint64_t chunkCapacity,
            std::uint64_t budgetBytes,
            std::uint64_t tileSize)
        {
            if(chunkCapacity == 0u)
                return 0u;
            std::uint64_t maxLive = budgetBytes / (chunkCapacity * payloadBytesPerSlot);
            if(tileSize > 1u && maxLive >= tileSize)
                maxLive = roundDownToMultiple(maxLive, tileSize);
            return maxLive;
        }

        /** Payload budget: explicit, host profile, or a fraction of the device memory. */
        inline std::uint64_t resolveMemoryBudget(
            alpaka::onHost::DeviceProperties const& properties,
            DevicePolicyConfig const& config)
        {
            if(config.memoryBudgetBytes != 0u)
                return config.memoryBudgetBytes;
            if(properties.warpSize <= 1u)
                return config.hostBudgetBytes;
            std::uint64_t const byCapacity = properties.globalMemCapacityBytes / config.memoryBudgetDivisor;
            if(byCapacity == 0u)
                return std::min(config.hostBudgetBytes, config.acceleratorBudgetCapBytes);
            return std::min(byCapacity, config.acceleratorBudgetCapBytes);
        }
    } // namespace detail

    /** Derive the default chunk capacity C, live-particle capacity P and tile width T.
     *
     * The rules follow the plan and `missing.md`:
     *
     * - `T` is at least the warp size, a multiple of it and at most `maxThreadsPerBlock`. The
     *   logical width starts at 128 (or the warp size when larger) and is rounded down to a warp
     *   multiple. When the device still allows a warp multiple, it is capped so the default
     *   `2*T`-entry hash table fits into `sharedMemPerBlockBytes`. The host reports warp size one
     *   and 64 KiB of shared memory, so it keeps the 128-wide logical tile that the existing
     *   block-extent-one launch mapping expects.
     * - `C` starts at the wave estimate
     *   `waves * multiProcessorCount * targetBlocksPerSm * T`, is capped by `maxBlocksPerGrid`
     *   (a chunk needs `ceil(C/T)` tiles), by `maxChunkCapacity`, by the run size and, for the
     *   default P, by the payload budget. A derived C is rounded down to a multiple of T. The
     *   budget cap on C keeps the fixed host budget small; an explicit C is kept and P is
     *   clamped instead.
     * - Default `P` is `min(numParticles, liveParticlesPerChunk * C)` and is clamped to the
     *   largest value with `P * C * payloadBytesPerSlot <= budget`, rounded down to a tile
     *   multiple when that leaves at least one tile. The pool bound is `S = P + K*R` chunks;
     *   the policy budget covers the `P` term and leaves the `K*R` output credits to the
     *   configured pool size.
     *
     * `C` and `T` explicit overrides are kept when they fit the device limits. An explicit `P`
     * is kept exactly: it is rejected with `std::invalid_argument` when `P * C` exceeds the
     * budget, when it is above `numParticles`, or when it is zero. The auto-derived default `P`
     * is clamped instead. A budget that cannot hold one live particle, or one tile for a derived
     * C, raises as well.
     */
    inline DevicePolicy makeDevicePolicy(
        alpaka::onHost::DeviceProperties const& properties,
        std::uint64_t numParticles,
        DevicePolicyOverrides const& overrides = {},
        DevicePolicyConfig const& config = {})
    {
        if(config.waves == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: waves must be non-zero"};
        if(config.targetBlocksPerSm == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: targetBlocksPerSm must be non-zero"};
        if(config.logicalTileWidth == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: logicalTileWidth must be non-zero"};
        if(config.maxChunkCapacity == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: maxChunkCapacity must be non-zero"};
        if(config.liveParticlesPerChunk == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: liveParticlesPerChunk must be non-zero"};
        if(config.memoryBudgetDivisor == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: memoryBudgetDivisor must be non-zero"};
        if(config.hostBudgetBytes == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: hostBudgetBytes must be non-zero"};
        if(config.acceleratorBudgetCapBytes == 0u)
            throw std::invalid_argument{
                "graphFlow::DevicePolicy: acceleratorBudgetCapBytes must be non-zero"};

        std::uint64_t const warpSize = properties.warpSize == 0u ? 1u : properties.warpSize;
        std::uint64_t const threadLimit = properties.maxThreadsPerBlock;
        if(threadLimit == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: device reports maxThreadsPerBlock == 0"};
        if(threadLimit < warpSize)
            throw std::invalid_argument{
                "graphFlow::DevicePolicy: maxThreadsPerBlock is smaller than the warp size"};
        if(properties.maxBlocksPerGrid == 0u)
            throw std::invalid_argument{"graphFlow::DevicePolicy: device reports maxBlocksPerGrid == 0"};
        if(overrides.liveParticles
           && (*overrides.liveParticles == 0u
               || std::uint64_t{*overrides.liveParticles} > numParticles))
            throw std::invalid_argument{
                "graphFlow::DevicePolicy: liveParticles must be non-zero and not exceed numParticles"};

        // --- Logical tile width T.
        std::uint32_t tile = 0u;
        if(overrides.tileSize)
        {
            if(*overrides.tileSize == 0u)
                throw std::invalid_argument{"graphFlow::DevicePolicy: tileSize must be non-zero"};
            if(std::uint64_t{*overrides.tileSize} > threadLimit)
                throw std::invalid_argument{
                    "graphFlow::DevicePolicy: tileSize exceeds maxThreadsPerBlock"};
            tile = *overrides.tileSize;
        }
        else
        {
            std::uint64_t const target = std::max<std::uint64_t>(config.logicalTileWidth, warpSize);
            std::uint64_t const usable = threadLimit / warpSize * warpSize;
            std::uint64_t candidate = target / warpSize * warpSize;
            candidate = std::max(candidate, warpSize);
            // Leave room for the default 2*T-entry shared-memory hash table. The cap is skipped
            // when it would drop below one warp, because the dense path does not need that table.
            std::uint64_t const sharedTileLimit
                = std::uint64_t{properties.sharedMemPerBlockBytes} / (2u * hashTableEntryBytes);
            std::uint64_t const sharedTileLimitAligned = sharedTileLimit / warpSize * warpSize;
            if(sharedTileLimitAligned >= warpSize)
                candidate = std::min(candidate, sharedTileLimitAligned);
            candidate = std::min(candidate, usable);
            tile = static_cast<std::uint32_t>(candidate);
        }

        std::uint64_t const budgetBytes = detail::resolveMemoryBudget(properties, config);

        // --- Launch and wave bounds for the chunk capacity C.
        std::uint64_t const gridCapacity = std::uint64_t{properties.maxBlocksPerGrid} * tile;
        std::uint64_t const launchCapacity
            = std::min<std::uint64_t>(config.maxChunkCapacity, gridCapacity);
        if(!overrides.chunkCapacity && launchCapacity < tile)
            throw std::invalid_argument{
                "graphFlow::DevicePolicy: maxChunkCapacity and maxBlocksPerGrid cannot hold one tile"};

        std::uint64_t const sms
            = properties.multiProcessorCount == 0u ? 1u : properties.multiProcessorCount;
        std::uint64_t const waveCapacity = detail::saturatedProduct(
            detail::saturatedProduct(
                detail::saturatedProduct(std::uint64_t{config.waves}, sms),
                config.targetBlocksPerSm),
            tile);

        // --- Chunk capacity C.
        std::uint32_t chunkCapacity = 0u;
        char const* capacityCap = "explicit";
        if(overrides.chunkCapacity)
        {
            std::uint64_t const requested = *overrides.chunkCapacity;
            if(requested == 0u)
                throw std::invalid_argument{"graphFlow::DevicePolicy: chunkCapacity must be non-zero"};
            if(requested > config.maxChunkCapacity)
                throw std::invalid_argument{
                    "graphFlow::DevicePolicy: chunkCapacity exceeds the configured maximum of "
                    + std::to_string(config.maxChunkCapacity)};
            if(requested > gridCapacity)
                throw std::invalid_argument{
                    "graphFlow::DevicePolicy: chunkCapacity needs more tiles than maxBlocksPerGrid allows"};
            chunkCapacity = *overrides.chunkCapacity;
        }
        else
        {
            std::uint64_t candidate = std::min(waveCapacity, launchCapacity);
            if(waveCapacity <= launchCapacity)
                capacityCap = "wave";
            else if(config.maxChunkCapacity <= gridCapacity)
                capacityCap = "maximum";
            else
                capacityCap = "grid";

            std::uint64_t const runCapacity = std::max<std::uint64_t>(numParticles, tile);
            if(candidate > runCapacity)
            {
                candidate = runCapacity;
                capacityCap = "particles";
            }

            if(overrides.liveParticles)
            {
                // An explicit P is kept; shrink the derived C so that P*C fits the budget. The
                // final P clamp below covers the explicit-C path.
                std::uint64_t const liveBytes
                    = detail::saturatedProduct(*overrides.liveParticles, payloadBytesPerSlot);
                std::uint64_t const byBudget = budgetBytes / liveBytes;
                if(byBudget < candidate)
                {
                    candidate = byBudget;
                    capacityCap = "memory";
                }
            }
            else
            {
                std::uint64_t const byBudget = detail::largestChunkForBudget(
                    numParticles,
                    config.liveParticlesPerChunk,
                    budgetBytes,
                    candidate);
                if(byBudget < candidate)
                {
                    candidate = byBudget;
                    capacityCap = "memory";
                }
            }

            candidate = detail::roundDownToMultiple(candidate, tile);
            if(candidate < tile)
                throw std::invalid_argument{
                    "graphFlow::DevicePolicy: the payload memory budget cannot hold one tile of the chunk"};
            chunkCapacity = static_cast<std::uint32_t>(candidate);
        }

        // --- Live-particle capacity P.
        // Explicit and auto-derived P are handled differently: an explicit P is a user request
        // and must fail when it does not fit, while the default is clamped to the largest
        // feasible tile-rounded value. `liveParticlesExplicit` records which one was used.
        bool const liveParticlesExplicit = overrides.liveParticles.has_value();
        std::uint32_t liveParticles = 0u;
        if(liveParticlesExplicit)
        {
            liveParticles = *overrides.liveParticles;
        }
        else if(numParticles != 0u)
        {
            std::uint64_t const live = std::min<std::uint64_t>(
                numParticles,
                detail::saturatedProduct(config.liveParticlesPerChunk, chunkCapacity));
            liveParticles = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(live, std::numeric_limits<std::uint32_t>::max()));
        }

        if(liveParticles != 0u)
        {
            std::uint64_t maxLive
                = detail::largestLiveParticlesForBudget(chunkCapacity, budgetBytes, tile);
            if(maxLive == 0u && !overrides.chunkCapacity)
            {
                // Last resort: the launch C cannot hold one particle. Reduce the derived C to
                // the largest tile multiple that still does, then clamp again.
                std::uint64_t const maxSlots = budgetBytes / payloadBytesPerSlot;
                std::uint64_t const shrunk = detail::roundDownToMultiple(
                    std::min<std::uint64_t>(chunkCapacity, maxSlots),
                    tile);
                if(shrunk < tile)
                    throw std::invalid_argument{
                        "graphFlow::DevicePolicy: the payload memory budget cannot hold one tile of the chunk"};
                chunkCapacity = static_cast<std::uint32_t>(shrunk);
                capacityCap = "memory";
                maxLive = detail::largestLiveParticlesForBudget(chunkCapacity, budgetBytes, tile);
            }
            if(maxLive == 0u)
                throw std::invalid_argument{
                    "graphFlow::DevicePolicy: the payload memory budget is too small for one live particle"};
            if(liveParticlesExplicit)
            {
                if(std::uint64_t{liveParticles} > maxLive)
                    throw std::invalid_argument{
                        "graphFlow::DevicePolicy: explicit liveParticles "
                        + std::to_string(liveParticles) + " with chunkCapacity "
                        + std::to_string(chunkCapacity) + " requires "
                        + std::to_string(requiredPayloadBytes(liveParticles, chunkCapacity))
                        + " bytes, exceeding the payload budget of " + std::to_string(budgetBytes)
                        + " bytes; the largest feasible P is " + std::to_string(maxLive)};
            }
            else if(std::uint64_t{liveParticles} > maxLive)
            {
                liveParticles = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(maxLive, std::numeric_limits<std::uint32_t>::max()));
                capacityCap = "memory";
            }
        }

        if(!detail::payloadFitsBudget(liveParticles, chunkCapacity, budgetBytes))
            throw std::invalid_argument{
                "graphFlow::DevicePolicy: liveParticles and chunkCapacity exceed the payload memory budget"};

        DevicePolicy policy;
        policy.chunkCapacity = chunkCapacity;
        policy.liveParticles = liveParticles;
        policy.liveParticlesExplicit = liveParticlesExplicit;
        policy.tileSize = tile;
        policy.waves = config.waves;
        policy.targetBlocksPerSm = config.targetBlocksPerSm;
        policy.memoryBudgetBytes = budgetBytes;
        policy.description = std::string{"C="} + std::to_string(chunkCapacity) + " P="
                             + std::to_string(liveParticles) + " T=" + std::to_string(tile)
                             + " waves=" + std::to_string(config.waves) + " blocksPerSm="
                             + std::to_string(config.targetBlocksPerSm) + " budget="
                             + std::to_string(budgetBytes) + "B warp="
                             + std::to_string(properties.warpSize) + " sms="
                             + std::to_string(properties.multiProcessorCount) + " cap=" + capacityCap;
        return policy;
    }

    /** Map the policy knobs stored in `Config` to a `DevicePolicyConfig`. */
    inline DevicePolicyConfig policyConfigOf(Config const& config)
    {
        DevicePolicyConfig policyConfig;
        policyConfig.waves = config.policyWaves;
        policyConfig.targetBlocksPerSm = config.policyTargetBlocksPerSm;
        policyConfig.logicalTileWidth = config.policyLogicalTileWidth;
        policyConfig.maxChunkCapacity = config.policyMaxChunkCapacity;
        policyConfig.liveParticlesPerChunk = config.policyLiveParticlesPerChunk;
        policyConfig.memoryBudgetBytes = config.policyMemoryBudgetBytes;
        return policyConfig;
    }

    /** Copy the chosen C, P, T and the augmented pool budget into a `Config`.
     *
     * The policy memory budget covers the live working set `P*C`. The fixed-capacity pool also
     * holds the `K*R` output-credit chunks, so `poolPayloadBudgetBytes` becomes the policy budget
     * plus the payload of `Config::poolOutputReserveChunks` chunks. This keeps the documented
     * default (`applyDevicePolicy`, then `derivePoolSize`, then `Config::validate`) consistent.
     */
    inline void applyDevicePolicy(Config& config, DevicePolicy const& policy)
    {
        config.chunkCapacity = policy.chunkCapacity;
        config.liveParticles = policy.liveParticles;
        config.tileSize = policy.tileSize;
        config.poolPayloadBudgetBytes = payloadBudgetWithReserve(
            policy.memoryBudgetBytes,
            policy.chunkCapacity,
            config.poolOutputReserveChunks);
    }
} // namespace graphFlow
