/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace graphFlow::rng
{
    /** Philox engine handed to an opting-in flow functor.
     *
     * The engine is constructed by graphFlow for one logical `(id, event, purpose)` position.
     * Users draw floating-point values through `alpaka::rand::distribution::UniformReal` or
     * `NormalReal`; integer decisions use the engine's unsigned `operator()`. Functors never see a
     * seed, an offset, or a purpose count.
     */
    using Engine = alpaka::rand::engine::Philox4x32x10;

    /** Bit budget of the packed stream position.
     *
     * A stream key is a bijective packing of three fields plus the run seed:
     *  - `streamIdBits` low bits hold the global logical id,
     *  - `streamEventBits` bits hold the logical event (particle visit or photon scatter count),
     *  - `streamPurposeBits` bits hold an internal graphFlow stage purpose.
     *
     * The limits are checked by `checkedStreamKey()` and asserted in `makeEngine()`. Ids must be
     * unique within one run. Particle and photon ids live in separate examples; a graph that mixes
     * domains must allocate disjoint id ranges so the domains cannot share an engine.
     */
    inline constexpr std::uint32_t streamIdBits = 32u;
    inline constexpr std::uint32_t streamEventBits = 16u;
    inline constexpr std::uint32_t streamPurposeBits = 16u;

    static_assert(streamIdBits + streamEventBits + streamPurposeBits == 64u);

    inline constexpr std::uint64_t maxStreamId = (std::uint64_t{1u} << streamIdBits) - 1u;
    inline constexpr std::uint64_t maxStreamEvent = (std::uint64_t{1u} << streamEventBits) - 1u;
    inline constexpr std::uint32_t maxStreamPurpose = (std::uint32_t{1u} << streamPurposeBits) - 1u;

    /** Internal purpose of graph node `nodeKey`.
     *
     * Flow assigns `2 * nodeKey + 0` to the node functor and `2 * nodeKey + 1` to an optional
     * partition updater. Registration order is part of the reproducibility contract; users never
     * choose a purpose.
     */
    constexpr std::uint32_t streamPurposeForNode(std::uint64_t nodeKey, std::uint32_t subPurpose = 0u)
    {
        return static_cast<std::uint32_t>((nodeKey << 1u) | (subPurpose & 1u));
    }

    /** Packed `(id, event, purpose)` position without the run seed. */
    constexpr std::uint64_t packedStreamPosition(std::uint64_t globalId, std::uint64_t event, std::uint32_t purpose)
    {
        return (globalId & maxStreamId) | ((event & maxStreamEvent) << streamIdBits)
               | (static_cast<std::uint64_t>(purpose & maxStreamPurpose) << (streamIdBits + streamEventBits));
    }

    /** Bijective 64 bit multiply/xor mixer (SplitMix64 finalizer).
     *
     * Each step is invertible: two odd multiplications and three xor-shift permutations. The
     * composition is therefore a bijection on `uint64_t`, which keeps a packed position and its
     * engine key one-to-one for a fixed run seed.
     */
    constexpr std::uint64_t mixStreamKey(std::uint64_t value)
    {
        value ^= value >> 30u;
        value *= 0xBF58'476D'1CE4'E5B9ull;
        value ^= value >> 27u;
        value *= 0x94D0'49BB'1331'11EBull;
        value ^= value >> 31u;
        return value;
    }

    /** Engine key of one logical stream position.
     *
     * `packed -> mix(packed ^ mix(baseSeed))` is a bijection of the packed position for a fixed
     * base seed, so distinct positions cannot alias within a run. The engine is then constructed
     * with subsequence and offset zero: the pinned alpaka constructor-offset defect is never
     * exercised, and no event-indexed advance loop is used.
     */
    constexpr std::uint64_t streamKey(
        std::uint64_t baseSeed,
        std::uint64_t globalId,
        std::uint64_t event,
        std::uint32_t purpose)
    {
        std::uint64_t const packed = packedStreamPosition(globalId, event, purpose);
        return mixStreamKey(packed ^ mixStreamKey(baseSeed));
    }

    /** Host-side stream-key check.
     *
     * Rejects a position outside the documented bit budget instead of masking it into an
     * overlapping stream. Device code uses `makeEngine()` and relies on the asserted precondition.
     */
    constexpr std::uint64_t checkedStreamKey(
        std::uint64_t baseSeed,
        std::uint64_t globalId,
        std::uint64_t event,
        std::uint32_t purpose)
    {
        if(globalId > maxStreamId)
            throw std::overflow_error{"graphFlow::rng: global id exceeds the 32 bit stream-id range"};
        if(event > maxStreamEvent)
            throw std::overflow_error{"graphFlow::rng: event exceeds the 16 bit stream-event range"};
        if(purpose > maxStreamPurpose)
            throw std::overflow_error{"graphFlow::rng: purpose exceeds the 16 bit stream-purpose range"};
        return streamKey(baseSeed, globalId, event, purpose);
    }

    /** Initialise the engine of one logical `(id, event, purpose)` position. */
    ALPAKA_FN_ACC inline Engine makeEngine(
        std::uint64_t baseSeed,
        std::uint64_t globalId,
        std::uint64_t event,
        std::uint32_t purpose)
    {
        ALPAKA_ASSERT_ACC(globalId <= maxStreamId);
        ALPAKA_ASSERT_ACC(event <= maxStreamEvent);
        ALPAKA_ASSERT_ACC(purpose <= maxStreamPurpose);
        // Subsequence and offset stay zero. The constructor offset of the pinned Philox engine is
        // not equivalent to advancing a fresh engine, so the key carries the whole position.
        return Engine{streamKey(baseSeed, globalId, event, purpose), 0u, 0u};
    }

    /** Payload stream fields, customisable through specialisation.
     *
     * The defaults read `id` and either `visit` (particle) or `scatterCount` (photon). A payload
     * without those fields owns one shared `(0, 0)` position; a custom payload type can specialise
     * this trait to provide stable logical fields.
     */
    template<typename TPayload>
    struct StreamTraits
    {
        static ALPAKA_FN_ACC std::uint64_t id(TPayload const& payload) noexcept
        {
            if constexpr(requires { payload.id; })
                return static_cast<std::uint64_t>(payload.id);
            else
                return 0u;
        }

        static ALPAKA_FN_ACC std::uint64_t event(TPayload const& payload) noexcept
        {
            if constexpr(requires { payload.visit; })
                return static_cast<std::uint64_t>(payload.visit);
            else if constexpr(requires { payload.scatterCount; })
                return static_cast<std::uint64_t>(payload.scatterCount);
            else
                return 0u;
        }
    };

    template<typename TPayload>
    ALPAKA_FN_ACC std::uint64_t streamId(TPayload const& payload) noexcept
    {
        return StreamTraits<TPayload>::id(payload);
    }

    template<typename TPayload>
    ALPAKA_FN_ACC std::uint64_t streamEvent(TPayload const& payload) noexcept
    {
        return StreamTraits<TPayload>::event(payload);
    }
} // namespace graphFlow::rng
