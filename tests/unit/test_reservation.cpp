/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/Reservation.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

using graphFlow::ceilDivChecked;
using graphFlow::outputChunkBound;

namespace
{
    /** Enumerate every distribution of `remaining` particles over `bins` bins. */
    void forEachDistribution(
        std::uint32_t remaining,
        std::uint32_t bins,
        std::vector<std::uint32_t>& totals,
        std::function<void(std::vector<std::uint32_t> const&)> const& visit)
    {
        if(bins == 1u)
        {
            totals.push_back(remaining);
            visit(totals);
            totals.pop_back();
            return;
        }
        for(std::uint32_t i = 0u; i <= remaining; ++i)
        {
            totals.push_back(i);
            forEachDistribution(remaining - i, bins - 1u, totals, visit);
            totals.pop_back();
        }
    }
} // namespace

TEST_CASE("output chunk bound matches the closed form", "[reservation]")
{
    using graphFlow::outputChunkBound;

    CHECK(outputChunkBound(0u, 128u, 2u) == 0u);
    CHECK(outputChunkBound(0u, 1u, 1u) == 0u);
    CHECK(outputChunkBound(1u, 1u, 2u) == 1u);
    CHECK(outputChunkBound(17u, 127u, 2u) == 2u);
    CHECK(outputChunkBound(128u, 128u, 2u) == 2u);
    CHECK(outputChunkBound(4097u, 128u, 2u) == 34u); // ceil(4097/128) + 2 - 1 = 33 + 1
    CHECK(outputChunkBound(4097u, 1u, 2u) == 4097u); // capped at n
    CHECK(outputChunkBound(64u, 128u, 64u) == 64u);
    CHECK(outputChunkBound(1u, 128u, 64u) == 1u); // capped at n
}

TEST_CASE("reservation arithmetic is checked", "[reservation]")
{
    CHECK_THROWS_AS(ceilDivChecked(4u, 0u), std::invalid_argument);
    CHECK_THROWS_AS(outputChunkBound(4u, 0u, 2u), std::invalid_argument);
    CHECK_THROWS_AS(outputChunkBound(4u, 2u, 0u), std::invalid_argument);

    std::uint64_t const max = std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(outputChunkBound(max, 1u, 2u), std::overflow_error);
}

TEST_CASE("the bound covers every bin distribution", "[reservation]")
{
    // Exhaustive over small shapes: exact chunk count must never exceed the reservation bound.
    for(std::uint32_t n = 0u; n <= 14u; ++n)
    {
        for(std::uint32_t capacity = 1u; capacity <= 6u; ++capacity)
        {
            for(std::uint32_t bins = 1u; bins <= 5u; ++bins)
            {
                std::uint64_t const bound = outputChunkBound(n, capacity, bins);
                std::vector<std::uint32_t> totals;
                std::uint64_t exactMax = 0u;
                forEachDistribution(
                    n,
                    bins,
                    totals,
                    [&](std::vector<std::uint32_t> const& distribution)
                    {
                        std::uint64_t exact = 0u;
                        for(std::uint32_t const value : distribution)
                            exact += ceilDivChecked(value, capacity);
                        if(exact > bound)
                            FAIL(
                                "n=" << n << " C=" << capacity << " B=" << bins << ": exact " << exact
                                     << " exceeds bound " << bound);
                        exactMax = exact > exactMax ? exact : exactMax;
                    });
                CHECK(exactMax <= bound);
            }
        }
    }
}

TEST_CASE("config validation rejects degenerate values", "[config]")
{
    graphFlow::Config config;
    config.validate();

    graphFlow::Config badCapacity;
    badCapacity.chunkCapacity = 0u;
    CHECK_THROWS_AS(badCapacity.validate(), std::invalid_argument);

    graphFlow::Config badTile;
    badTile.tileSize = 0u;
    CHECK_THROWS_AS(badTile.validate(), std::invalid_argument);

    graphFlow::Config badBins;
    badBins.maxBins = 0u;
    CHECK_THROWS_AS(badBins.validate(), std::invalid_argument);

    graphFlow::Config tooManyBins;
    tooManyBins.maxBins = graphFlow::Config::denseBinLimit + 1u;
    CHECK_THROWS_AS(tooManyBins.validate(), std::invalid_argument);

    graphFlow::Config zeroParticles;
    zeroParticles.numParticles = 0u;
    zeroParticles.numChunks = 8u;
    CHECK_NOTHROW(zeroParticles.validate());
}
