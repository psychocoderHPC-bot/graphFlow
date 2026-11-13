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

    /** Conservative output-credit reserve in chunks for the reference graph.
     *
     * The example graph has `K=3` transform nodes and `R=2` output chunks per transform, so the
     * documented defaults need six reserve chunks. Eight leaves a small margin and fits the
     * fixed-capacity pool budget of the programmatic default. Callers with a larger `K*R` set
     * `Config::poolOutputReserveChunks` explicitly.
     */
    inline constexpr std::uint64_t defaultPoolOutputReserveChunks = 8u;

    /** Default total payload budget of the fixed-capacity pool in bytes.
     *
     * This is the documented accelerator default and the value used when a caller does not
     * resolve a tighter device-policy budget. It is larger than the T1 accelerator policy cap so
     * the policy-derived `C=8192`, `P=16384` pair still fits after the `K*R` output-credit
     * reserve is added. It replaces the previous accidental ~12.9 GB pool (`C=8192`, `P=65536`)
     * with a hard cap; the example resolves the device policy budget plus the exact `K*R`
     * reserve instead.
     */
    inline constexpr std::uint64_t defaultPoolPayloadBudgetBytes = 4ull * 1024ull * 1024ull * 1024ull;

    /** Payload bytes of `numChunks * chunkCapacity` slots, checked for 64-bit overflow.
     *
     * Both multiplications are guarded so a malformed configuration raises `std::overflow_error`
     * instead of wrapping into a small budget that the caller would accept. `slotBytes` defaults
     * to the particle SoA slot size, so existing callers keep their 24-byte particle budget. A
     * generic storage passes its own `slotBytes` (for example `sizeof(TPayload)` in
     * `PayloadSlotStorage`).
     */
    inline constexpr std::uint64_t requiredPayloadBytes(
        std::uint64_t numChunks,
        std::uint64_t chunkCapacity,
        std::uint64_t slotBytes = payloadBytesPerSlot)
    {
        if(slotBytes == 0u)
            throw std::invalid_argument{"graphFlow::requiredPayloadBytes: slotBytes must be non-zero"};
        if(chunkCapacity != 0u && numChunks > std::numeric_limits<std::uint64_t>::max() / chunkCapacity)
            throw std::overflow_error{"graphFlow::requiredPayloadBytes: slot count overflows 64 bits"};
        std::uint64_t const slots = numChunks * chunkCapacity;
        if(slots != 0u && slotBytes > std::numeric_limits<std::uint64_t>::max() / slots)
            throw std::overflow_error{"graphFlow::requiredPayloadBytes: byte count overflows 64 bits"};
        return slots * slotBytes;
    }

    /** Add the payload bytes of `reserveChunks` output-credit chunks to a policy budget.
     *
     * The device policy budget covers the live working set `P*C`; the fixed-capacity pool also
     * holds the `K*R` output-credit chunks. The addition is checked and a zero policy budget is
     * rejected so the cap cannot be disabled accidentally.
     */
    inline std::uint64_t payloadBudgetWithReserve(
        std::uint64_t policyBudgetBytes,
        std::uint64_t chunkCapacity,
        std::uint64_t reserveChunks = defaultPoolOutputReserveChunks,
        std::uint64_t slotBytes = payloadBytesPerSlot)
    {
        if(policyBudgetBytes == 0u)
            throw std::invalid_argument{"graphFlow::payloadBudgetWithReserve: policy budget must be non-zero"};
        std::uint64_t const reserveBytes = requiredPayloadBytes(reserveChunks, chunkCapacity, slotBytes);
        if(policyBudgetBytes > std::numeric_limits<std::uint64_t>::max() - reserveBytes)
            throw std::overflow_error{"graphFlow::payloadBudgetWithReserve: budget overflows 64 bits"};
        return policyBudgetBytes + reserveBytes;
    }

    /** Reject a fixed-capacity pool whose payload arrays exceed `budgetBytes`.
     *
     * The check is `numChunks * chunkCapacity * slotBytes <= budgetBytes`, with `slotBytes`
     * defaulting to the 24-byte particle SoA slot. A zero budget is invalid: it would otherwise
     * silently disable the cap.
     */
    inline void validatePayloadBudget(
        std::uint64_t numChunks,
        std::uint64_t chunkCapacity,
        std::uint64_t budgetBytes,
        std::string const& context,
        std::uint64_t slotBytes)
    {
        if(budgetBytes == 0u)
            throw std::invalid_argument{context + ": payload budget must be non-zero"};
        std::uint64_t const required = requiredPayloadBytes(numChunks, chunkCapacity, slotBytes);
        if(required > budgetBytes)
            throw std::invalid_argument{
                context + ": " + std::to_string(numChunks) + " chunks x " + std::to_string(chunkCapacity)
                + " slots x " + std::to_string(slotBytes) + " bytes requires "
                + std::to_string(required) + " bytes, exceeding the payload budget of "
                + std::to_string(budgetBytes) + " bytes"};
    }

    /** Particle-budget overload; `slotBytes` defaults to the particle SoA slot size. */
    inline void validatePayloadBudget(
        std::uint64_t numChunks,
        std::uint64_t chunkCapacity,
        std::uint64_t budgetBytes,
        std::string const& context = "graphFlow::validatePayloadBudget")
    {
        validatePayloadBudget(numChunks, chunkCapacity, budgetBytes, context, payloadBytesPerSlot);
    }

    /** Slot-size overload without an explicit diagnostic context. */
    inline void validatePayloadBudget(
        std::uint64_t numChunks,
        std::uint64_t chunkCapacity,
        std::uint64_t budgetBytes,
        std::uint64_t slotBytes)
    {
        validatePayloadBudget(
            numChunks,
            chunkCapacity,
            budgetBytes,
            "graphFlow::validatePayloadBudget",
            slotBytes);
    }
} // namespace graphFlow
