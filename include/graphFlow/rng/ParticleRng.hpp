/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow::rng
{
    /** Purpose indices used to derive the draw position of a particle decision.
     *
     * The mapping is part of the reproducibility contract: it must not depend on the chunk, tile,
     * thread, backend or scheduling order.
     */
    inline constexpr std::uint32_t purposeProperty = 0u;
    inline constexpr std::uint32_t purposeFilter = 0u;
    inline constexpr std::uint32_t purposeProcessA = 1u;
    inline constexpr std::uint32_t purposeProcessB = 2u;

    /** Philox offset of one decision.
     *
     * Offset 0 is the property draw. Filter, A and B use `4 * (1 + 3 * visit + purpose)` so every
     * decision owns a disjoint block of four 32 bit outputs.
     */
    constexpr std::uint64_t decisionOffset(std::uint32_t visit, std::uint32_t purpose)
    {
        return 4ull * (1ull + 3ull * static_cast<std::uint64_t>(visit) + purpose);
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

    /** One key-based Philox draw.
     *
     * `subsequence` is the particle id, `offset` counts individual 32 bit outputs. The engine is
     * constructed per draw, which is exactly the equivalence `Philox4x32x10(seed, id, offset)()`.
     */
    ALPAKA_FN_ACC inline std::uint32_t draw(
        std::uint64_t seed,
        std::uint64_t particleId,
        std::uint32_t visit,
        std::uint32_t purpose)
    {
        return drawAt(seed, particleId, decisionOffset(visit, purpose));
    }
} // namespace graphFlow::rng
