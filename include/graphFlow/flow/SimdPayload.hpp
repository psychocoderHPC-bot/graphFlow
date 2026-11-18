/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace graphFlow
{
    /** Opt-in structured SIMD form of a payload.
     *
     * The SIMD transform mappings hand a functor one pack of `T_width` payloads and take the
     * transformed pack back. `Pack` is a structured type whose members are `alpaka::Simd` packs,
     * so the functor can express the payload arithmetic on `Simd` values instead of iterating
     * over lanes. A payload that does not specialise this template cannot be used with
     * `TransformKernelMode::simdBlocks` or `TransformKernelMode::simdWarps`.
     *
     * A specialisation is declared next to the payload type and provides
     *  - `using Pack = ...;` with a `static constexpr std::uint32_t width`,
     *  - `static Pack load(std::tuple<TColumnPacks...> const& columns)`, which transposes the
     *    loaded per-column packs (payload attribute declaration order) into the structured pack,
     *  - `static Payload lane(Pack const& pack, std::uint32_t lane)`, which extracts one payload.
     *
     * `load` and `lane` are the only transposition points: the storage keeps one column per
     * attribute, and a payload attribute such as a three-vector is interleaved inside its column.
     */
    template<typename TPayload, std::uint32_t T_width>
    struct SimdPayload;
} // namespace graphFlow
