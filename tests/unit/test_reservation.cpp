/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/Reservation.hpp"
#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/graph/DenseBinTable.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

using graphFlow::ceilDivChecked;
using graphFlow::checkedOutputChunkBound;
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

namespace
{
    /** A partition width above the dense shared-table limit is a hash partition, not a dense one. */
    struct WideBinsPolicy
    {
        [[maybe_unused]] static constexpr std::uint32_t maxBins = graphFlow::Config::denseBinLimit + 1u;
    };

    std::uint64_t exactChunkCount(std::vector<std::uint32_t> const& totals, std::uint32_t capacity)
    {
        std::uint64_t exact = 0u;
        for(std::uint32_t const total : totals)
            exact += ceilDivChecked(total, capacity);
        return exact;
    }
} // namespace

// The dense shared-table check stays on the dense policy concept; Config itself must accept a
// larger hash partition width.
static_assert(!graphFlow::DenseNodePolicy<WideBinsPolicy>);
static_assert(graphFlow::DenseNodePolicy<graphFlow::NoPayloadChange>);

TEST_CASE("config validation rejects degenerate values", "[config]")
{
    graphFlow::Config config;
    config.validate();

    // Reference the policy member so nvcc does not diagnose the concept-only use as unused.
    CHECK(WideBinsPolicy::maxBins == graphFlow::Config::denseBinLimit + 1u);

    graphFlow::Config badCapacity;
    badCapacity.chunkCapacity = 0u;
    CHECK_THROWS_AS(badCapacity.validate(), std::invalid_argument);

    graphFlow::Config badTile;
    badTile.tileSize = 0u;
    CHECK_THROWS_AS(badTile.validate(), std::invalid_argument);

    graphFlow::Config badBins;
    badBins.maxBins = 0u;
    CHECK_THROWS_AS(badBins.validate(), std::invalid_argument);

    // The hash path sizes its shared table from the tile width, so maxBins may exceed the dense
    // limit. The dense policy concept above still rejects such a width at compile time.
    graphFlow::Config hashBins;
    hashBins.maxBins = graphFlow::Config::denseBinLimit + 1u;
    CHECK_NOTHROW(hashBins.validate());
    hashBins.maxBins = 65536u;
    CHECK_NOTHROW(hashBins.validate());

    graphFlow::Config zeroParticles;
    zeroParticles.numParticles = 0u;
    zeroParticles.numChunks = 8u;
    CHECK_NOTHROW(zeroParticles.validate());
}

TEST_CASE("hash output credit uses maxBins, not table entries or Q", "[reservation][hash]")
{
    constexpr std::uint32_t capacity = 64u;
    for(std::uint32_t const maxBins : {graphFlow::Config::denseBinLimit + 1u, 4096u, 65536u})
    {
        CAPTURE(maxBins);
        // One full chunk spread over at least C bins can always fill C output chunks.
        CHECK(outputChunkBound(capacity, capacity, maxBins) == capacity);
        CHECK(checkedOutputChunkBound(capacity, capacity, maxBins, "test") == capacity);
        // A batch that is one particle per bin needs one chunk per particle.
        CHECK(outputChunkBound(1000u, capacity, maxBins) == 1000u);
    }

    // The hash table capacity and the tile width are not output credits.
    CHECK(graphFlow::HashBinTable<128u>::entries == 256u);
    CHECK(graphFlow::HashBinTable<128u>::entries != capacity);
}

TEST_CASE("hash reserve covers Q=1/2/8 and partial batches", "[reservation][hash]")
{
    constexpr std::uint32_t capacity = 8u;
    for(std::uint32_t const maxBins : {graphFlow::Config::denseBinLimit + 1u, 4096u, 65536u})
    {
        for(std::uint32_t const Q : {1u, 2u, 8u})
        {
            std::uint64_t const full = std::uint64_t{capacity} * Q;
            for(std::uint64_t const n : {std::uint64_t{0u}, std::uint64_t{1u}, full / 2u, full - 1u, full})
            {
                CAPTURE(maxBins, Q, n);
                std::uint64_t const bound = checkedOutputChunkBound(n, capacity, maxBins, "test");
                CHECK(bound == outputChunkBound(n, capacity, maxBins));

                std::vector<std::uint32_t> totals(maxBins, 0u);
                // One bin: ceil(n/C) chunks, the low end of the bound.
                totals[0] = static_cast<std::uint32_t>(n);
                CHECK(exactChunkCount(totals, capacity) <= bound);

                // One particle per bin, the high end for a wide hash partition.
                std::fill(totals.begin(), totals.end(), 0u);
                std::uint64_t remaining = n;
                for(std::uint32_t bin = 0u; bin < maxBins && remaining != 0u; ++bin, --remaining)
                    totals[bin] = 1u;
                CHECK(exactChunkCount(totals, capacity) <= bound);
                CHECK(graphFlow::planDenseBins(totals, capacity).numOutputChunks == exactChunkCount(totals, capacity));

                // A bin just over C forces a second chunk while the rest stays fragmented.
                std::fill(totals.begin(), totals.end(), 0u);
                std::uint32_t const head = static_cast<std::uint32_t>(std::min<std::uint64_t>(n, capacity + 1u));
                totals[0] = head;
                if(n > head)
                    totals[1] = static_cast<std::uint32_t>(n - head);
                CHECK(exactChunkCount(totals, capacity) <= bound);
                CHECK(graphFlow::planDenseBins(totals, capacity).numOutputChunks == exactChunkCount(totals, capacity));
            }
        }
    }
}

TEST_CASE("checked output credit reports 32-bit overflow", "[reservation]")
{
    std::uint64_t const over32 = std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 10u;
    CHECK_THROWS_AS(checkedOutputChunkBound(over32, 1u, 1u, "test"), std::overflow_error);
    CHECK_THROWS_AS(
        checkedOutputChunkBound(std::numeric_limits<std::uint64_t>::max(), 1u, 2u, "test"),
        std::overflow_error);
}
