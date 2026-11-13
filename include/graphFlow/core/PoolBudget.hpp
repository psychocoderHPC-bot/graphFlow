/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace graphFlow
{
    /** Value types of the chunk-pool payload arrays.
     *
     * `ChunkPool` uses these aliases for `ids` and the `property`, `weight`, `visit` and
     * `binIds` arrays. Keeping the budget arithmetic on the same aliases means a payload type
     * change cannot silently desynchronise the memory budget from the allocation.
     */
    using PoolIdValue = std::uint64_t;
    using PoolU32Value = std::uint32_t;

    /** Number of `PoolU32Value` payload arrays in a chunk slot. */
    inline constexpr std::uint32_t poolPayloadU32Arrays = 4u;

    /** Payload bytes of one particle slot, derived from the actual pool arrays.
     *
     * One `std::uint64_t` id plus `property`, `weight`, `visit` and `binIds` (`std::uint32_t`).
     * This is the single source of truth for the per-slot payload size; kernels, the policy and
     * the pool budget must not repeat the numeric value.
     */
    inline constexpr std::uint64_t payloadBytesPerSlot
        = sizeof(PoolIdValue) + static_cast<std::uint64_t>(poolPayloadU32Arrays) * sizeof(PoolU32Value);

    /** Default total payload budget of the fixed-capacity pool in bytes.
     *
     * This is the documented accelerator default and the value used when a caller does not
     * resolve a tighter device-policy budget. It replaces the previous accidental ~12.9 GB pool
     * (`C=8192`, `P=65536`) with a hard cap; the example resolves the device policy budget plus
     * the graph's `K*R` output-credit reserve instead.
     */
    inline constexpr std::uint64_t defaultPoolPayloadBudgetBytes = 3ull * 1024ull * 1024ull * 1024ull;

    /** Payload bytes of `numChunks * chunkCapacity` slots, checked for 64-bit overflow.
     *
     * Both multiplications are guarded so a malformed configuration raises `std::overflow_error`
     * instead of wrapping into a small budget that the caller would accept.
     */
    inline constexpr std::uint64_t requiredPayloadBytes(
        std::uint64_t numChunks,
        std::uint64_t chunkCapacity)
    {
        if(chunkCapacity != 0u && numChunks > std::numeric_limits<std::uint64_t>::max() / chunkCapacity)
            throw std::overflow_error{"graphFlow::requiredPayloadBytes: slot count overflows 64 bits"};
        std::uint64_t const slots = numChunks * chunkCapacity;
        if(slots != 0u && payloadBytesPerSlot > std::numeric_limits<std::uint64_t>::max() / slots)
            throw std::overflow_error{"graphFlow::requiredPayloadBytes: byte count overflows 64 bits"};
        return slots * payloadBytesPerSlot;
    }

    /** Reject a fixed-capacity pool whose payload arrays exceed `budgetBytes`.
     *
     * The check is `numChunks * chunkCapacity * payloadBytesPerSlot <= budgetBytes`. A zero
     * budget is invalid: it would otherwise silently disable the cap.
     */
    inline void validatePayloadBudget(
        std::uint64_t numChunks,
        std::uint64_t chunkCapacity,
        std::uint64_t budgetBytes,
        std::string const& context = "graphFlow::validatePayloadBudget")
    {
        if(budgetBytes == 0u)
            throw std::invalid_argument{context + ": payload budget must be non-zero"};
        std::uint64_t const required = requiredPayloadBytes(numChunks, chunkCapacity);
        if(required > budgetBytes)
            throw std::invalid_argument{
                context + ": " + std::to_string(numChunks) + " chunks x " + std::to_string(chunkCapacity)
                + " slots x " + std::to_string(payloadBytesPerSlot) + " bytes requires "
                + std::to_string(required) + " bytes, exceeding the payload budget of "
                + std::to_string(budgetBytes) + " bytes"};
    }
} // namespace graphFlow
