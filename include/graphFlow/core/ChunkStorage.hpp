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
     *
     * `numChunks()` is the physical chunk count the storage actually allocated. The concrete
     * storage clamps a larger request to its payload budget, so the scheduler always admits
     * against the memory that exists; it never sees a chunk that has no payload slot.
     *
     * `base(handle)` is the first slot of the chunk's own contiguous range, not necessarily
     * `handle.index * chunkCapacity`. The fixed-layout storages derive it as `index * chunkCapacity`
     * because every chunk owns a full row; a slot-arena storage stores a per-chunk start, because a
     * chunk owns exactly the fill it committed. Both fulfil the same contract: while `handle`
     * refers to a live chunk, its slots start at `base(handle)`, lie inside the storage's slot
     * count and belong to no other chunk, and reader leases keep that mapping stable for the
     * duration of a batch. A consumer must therefore never recompute the base from the chunk
     * index; it reads it from the storage.
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
