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
    /** Number of purpose draws reserved per visit by the particle mapping.
     *
     * The value is the default for `decisionOffset()`, `checkedDecisionOffset()` and `draw()`. It
     * keeps the existing particle-feedback mapping `4 * (1 + 3 * visit + purpose)` unchanged.
     * Applications with a different number of draws per visit pass their own value, for example
     * `photonPurposesPerVisit`.
     */
    inline constexpr std::uint32_t purposesPerVisit = 3u;

    /** Purpose indices used to derive the draw position of a particle decision.
     *
     * The mapping is part of the reproducibility contract: it must not depend on the chunk, tile,
     * thread, backend or scheduling order.
     */
    inline constexpr std::uint32_t purposeProperty = 0u;
    inline constexpr std::uint32_t purposeFilter = 0u;
    inline constexpr std::uint32_t purposeProcessA = 1u;
    inline constexpr std::uint32_t purposeProcessB = 2u;

    /** Photon purposes: four independent draws per completed scatter count. */
    inline constexpr std::uint32_t purposeFlight = 0u;
    inline constexpr std::uint32_t purposeCollisionKind = 1u;
    inline constexpr std::uint32_t purposeCosine = 2u;
    inline constexpr std::uint32_t purposeAzimuth = 3u;

    /** Number of photon draws per visit. */
    inline constexpr std::uint32_t photonPurposesPerVisit = 4u;

    /** Philox offset of one decision.
     *
     * Offset 0 is the property draw. With the default `purposesPerVisit == 3`, Filter, A and B use
     * `4 * (1 + 3 * visit + purpose)`, so every decision owns a disjoint block of four 32 bit
     * outputs. A visit with `purposesPerVisit` draws uses
     * `4 * (1 + purposesPerVisit * visit + purpose)`, which likewise reserves one four-word block
     * per `(visit, purpose)` position. All arithmetic is performed in 64 bit.
     */
    constexpr std::uint64_t decisionOffset(
        std::uint64_t visit,
        std::uint32_t purpose,
        std::uint32_t purposesPerVisitValue = purposesPerVisit)
    {
        return 4ull
            * (1ull + static_cast<std::uint64_t>(purposesPerVisitValue) * visit
               + static_cast<std::uint64_t>(purpose));
    }

    /** Checked 64 bit draw offset.
     *
     * Host-side counterpart of `decisionOffset()`. A purpose outside the configured visit or a
     * result that exceeds the 64 bit range raises instead of wrapping. Device code uses
     * `decisionOffset()` because it cannot propagate host exceptions; both share the same formula.
     */
    constexpr std::uint64_t checkedDecisionOffset(
        std::uint64_t visit,
        std::uint32_t purpose,
        std::uint32_t purposesPerVisitValue = purposesPerVisit)
    {
        if(purposesPerVisitValue == 0u)
            throw std::invalid_argument{"graphFlow::rng: purposesPerVisit must be non-zero"};
        if(purpose >= purposesPerVisitValue)
            throw std::invalid_argument{"graphFlow::rng: purpose must be below purposesPerVisit"};

        constexpr std::uint64_t maxOffset = std::numeric_limits<std::uint64_t>::max();
        // Check the scaled visit term before forming it, then check the final block count.
        if(visit > (maxOffset - 1u - purpose) / purposesPerVisitValue)
            throw std::overflow_error{"graphFlow::rng: draw offset overflows 64 bits"};
        std::uint64_t const block = 1u + static_cast<std::uint64_t>(purposesPerVisitValue) * visit + purpose;
        if(block > maxOffset / 4u)
            throw std::overflow_error{"graphFlow::rng: draw offset overflows 64 bits"};
        return 4u * block;
    }

    /** Compile-time purpose checked offset.
     *
     * `T_purpose` must be below `T_purposesPerVisit`; the static assertion makes an invalid purpose
     * a compilation error instead of a runtime range failure.
     */
    template<std::uint32_t T_purpose, std::uint32_t T_purposesPerVisit = purposesPerVisit>
    constexpr std::uint64_t checkedDecisionOffset(std::uint64_t visit)
    {
        static_assert(T_purpose < T_purposesPerVisit, "purpose must be below purposesPerVisit");
        return checkedDecisionOffset(visit, T_purpose, T_purposesPerVisit);
    }

    /** Draw the 32 bit output at an absolute offset of a particle's subsequence.
     *
     * Verified pinned behaviour: the constructor offset of the single-value `Philox4x32x10` is not
     * equivalent to advancing a fresh engine. `PhiloxSingle` computes the within-block position in
     * `skip()`, then `advanceState()` resets it to zero, and `skip4()` derives its carry words from
     * `counter[0] < temp[1]` / `counter[0] < temp[2]` instead of the corresponding counter words.
     * For a nonzero subsequence this makes even `skip4(0)` change counter word 3
     * (`include/alpaka/rand/engine/philox/PhiloxSingle.hpp:84-99`,
     * `PhiloxBaseCommon.hpp:63-90`). `tests/unit/test_philox.cpp` records the observation.
     *
     * To keep the documented contract "offset counts individual 32 bit outputs" this helper
     * constructs at offset zero and advances the public engine `offset` times. Draw positions stay
     * key based and independent of chunks, tiles, threads and scheduling order.
     */
    ALPAKA_FN_ACC inline std::uint32_t drawAt(
        std::uint64_t seed,
        std::uint64_t particleId,
        std::uint64_t offset)
    {
        alpaka::rand::engine::Philox4x32x10 engine(seed, particleId, 0u);
        std::uint32_t value = engine();
        for(std::uint64_t i = 0u; i < offset; ++i)
            value = engine();
        return value;
    }

    /** Property draw of a fresh particle: offset 0 of the particle subsequence. */
    ALPAKA_FN_ACC inline std::uint32_t propertyDraw(std::uint64_t seed, std::uint64_t particleId)
    {
        return drawAt(seed, particleId, 0u);
    }

    /** One key-based Philox draw for a `(seed, id, visit, purpose)` position.
     *
     * `subsequence` is the particle id, `offset` counts individual 32 bit outputs. The engine is
     * constructed per draw, which is exactly the equivalence `Philox4x32x10(seed, id, offset)()`.
     * `purposesPerVisitValue` defaults to the three-purpose particle mapping; a four-purpose caller
     * passes `photonPurposesPerVisit`.
     */
    ALPAKA_FN_ACC inline std::uint32_t draw(
        std::uint64_t seed,
        std::uint64_t particleId,
        std::uint32_t visit,
        std::uint32_t purpose,
        std::uint32_t purposesPerVisitValue = purposesPerVisit)
    {
        return drawAt(seed, particleId, decisionOffset(visit, purpose, purposesPerVisitValue));
    }

    /** Compile-time purpose draw.
     *
     * The purpose is a template argument so `purpose < purposesPerVisit` is enforced by a static
     * assertion. The value is identical to the runtime `draw()` overload.
     */
    template<std::uint32_t T_purpose, std::uint32_t T_purposesPerVisit = purposesPerVisit>
    ALPAKA_FN_ACC inline std::uint32_t draw(
        std::uint64_t seed,
        std::uint64_t particleId,
        std::uint32_t visit)
    {
        static_assert(T_purpose < T_purposesPerVisit, "purpose must be below purposesPerVisit");
        return draw(seed, particleId, visit, T_purpose, T_purposesPerVisit);
    }
} // namespace graphFlow::rng
