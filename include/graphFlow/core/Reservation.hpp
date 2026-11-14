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
    /** Overflow-checked ceiling division on 64 bit unsigned integers. */
    inline constexpr std::uint64_t ceilDivChecked(std::uint64_t numerator, std::uint64_t denominator)
    {
        if(denominator == 0u)
            throw std::invalid_argument{"ceilDivChecked: denominator must be non-zero"};
        return numerator / denominator + (numerator % denominator != 0u ? 1u : 0u);
    }

    /** Maximum number of output chunks a transform batch of `n` particles may produce.
     *
     * A batch with `n` particles, capacity `C` and at most `B` bins can produce at most
     * `ceil(n/C)` chunks if every particle lands in one bin, and at most `min(B, n)` chunks if
     * every particle lands in its own bin. The bound
     *
     *     R = min(n, ceil(n/C) + min(B, n) - 1)
     *
     * covers both extremes. It is deliberately conservative; host planning allocates the exact
     * `sum_b ceil(total_b/C)` chunks and releases the surplus.
     *
     * All arithmetic is performed in 64 bit and every overflow is reported as an exception.
     */
    inline constexpr std::uint64_t outputChunkBound(
        std::uint64_t n,
        std::uint64_t capacity,
        std::uint64_t maxBins)
    {
        if(n == 0u)
            return 0u;
        if(capacity == 0u)
            throw std::invalid_argument{"outputChunkBound: capacity must be non-zero"};
        if(maxBins == 0u)
            throw std::invalid_argument{"outputChunkBound: maxBins must be non-zero"};

        std::uint64_t const perChunk = ceilDivChecked(n, capacity);
        std::uint64_t const bins = maxBins < n ? maxBins : n;
        if(perChunk > std::numeric_limits<std::uint64_t>::max() - bins + 1u)
            throw std::overflow_error{"outputChunkBound: chunk count overflow"};
        std::uint64_t const bound = perChunk + bins - 1u;
        return bound < n ? bound : n;
    }

    /** Output-chunk credit of one transform batch, narrowed to the 32-bit chunk index.
     *
     * `maxBins` is the partition width, i.e. the classifier's `maxBins` for a partition node.
     * This is the same input for the dense and the hash path: the hash table entry count, the
     * tile width and the batch size `Q` never enter the output credit. `Q` only affects the
     * particle count through `n = sum(fill over the accepted prefix)`. Callers that pass a
     * smaller width (the route count, a tile width, or the table entries) can under-reserve a
     * hash batch whose particles spread over more bins.
     */
    inline std::uint32_t checkedOutputChunkBound(
        std::uint64_t n,
        std::uint64_t capacity,
        std::uint64_t maxBins,
        std::string const& context)
    {
        std::uint64_t const bound = outputChunkBound(n, capacity, maxBins);
        if(bound > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error{context + ": output-chunk credit exceeds 32 bits"};
        return static_cast<std::uint32_t>(bound);
    }
} // namespace graphFlow
