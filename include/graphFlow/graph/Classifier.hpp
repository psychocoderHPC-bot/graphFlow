/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstdint>

namespace graphFlow
{
    /** One particle as seen by a node policy.
     *
     * The kernel assembles the struct from the SoA input chunk, lets the policy mutate it, and
     * writes the result to the output chunk. A policy therefore cannot double-apply a mutation when
     * a batch is retried or scattered from a different tile order.
     */
    struct Particle
    {
        std::uint64_t id = 0u;
        std::uint32_t property = 0u;
        std::uint32_t weight = 0u;
        std::uint32_t visit = 0u;
    };

    /** A dense node policy.
     *
     * `maxBins` is the compile-time bin count used for the shared histogram. `classify` must be
     * pure and return a value in `[0, maxBins)`; values outside that range are rejected by the
     * kernel and surface as a batch error. `apply` is called once per particle in the scatter phase
     * with the bin previously returned by `classify`.
     */
    template<typename T>
    concept DenseNodePolicy = requires(T policy) {
        { T::maxBins } -> std::convertible_to<std::uint32_t>;
        requires(T::maxBins > 0u);
        requires(T::maxBins <= 1024u);
    };

    /** Default payload mutation for a pure routing node. */
    struct NoPayloadChange
    {
        static constexpr std::uint32_t maxBins = 2u;

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            [[maybe_unused]] std::uint32_t bin,
            [[maybe_unused]] Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
        }
    };
} // namespace graphFlow
