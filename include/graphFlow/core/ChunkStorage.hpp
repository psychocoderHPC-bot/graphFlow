/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"

#include <concepts>
#include <cstdint>

namespace graphFlow
{
    /** Narrow storage contract consumed by `GraphRunner`.
     *
     * The scheduler only needs chunk geometry, the acquire/release lifecycle, generation checks
     * and reader leases. It never touches payload fields, so the contract stays independent of the
     * payload layout: `ChunkPool`, the particle SoA adapter and the generic payload-slot adapter all
     * satisfy it. Keeping the contract narrow is what lets the runner stay a template without a
     * virtual dispatch in the hot path.
     */
    template<typename TStorage>
    concept ChunkStorage = requires(TStorage storage, ChunkHandle handle, std::uint32_t chunkIndex) {
        { storage.numChunks() } -> std::convertible_to<std::uint32_t>;
        { storage.chunkCapacity() } -> std::convertible_to<std::uint32_t>;
        { storage.freeCount() } -> std::convertible_to<std::uint32_t>;
        { storage.liveCount() } -> std::convertible_to<std::uint32_t>;

        { storage.acquire() } -> std::same_as<ChunkHandle>;
        storage.release(handle);
        storage.acquireLease(handle);
        storage.releaseLease(handle);
        { storage.leaseCount(handle) } -> std::convertible_to<std::uint32_t>;
        { storage.isLive(handle) } -> std::same_as<bool>;
        { storage.base(handle) } -> std::convertible_to<std::uint64_t>;
        { storage.generation(chunkIndex) } -> std::convertible_to<std::uint32_t>;
        storage.validateLive(handle);

        { storage.peakLive() } -> std::convertible_to<std::uint32_t>;
        { storage.peakLeases() } -> std::convertible_to<std::uint32_t>;
    };
} // namespace graphFlow
