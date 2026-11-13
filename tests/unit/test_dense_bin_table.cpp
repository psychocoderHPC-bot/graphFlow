/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/graph/DenseBinTable.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using graphFlow::buildDenseTable;
using graphFlow::DenseBinPlan;
using graphFlow::planDenseBins;

TEST_CASE("dense bin plan assigns consecutive chunks per bin", "[bintable]")
{
    DenseBinPlan const plan = planDenseBins({3u, 0u, 2u}, 2u);

    CHECK(plan.binBase == std::vector<std::uint32_t>{0u, 2u, 2u});
    CHECK(plan.numOutputChunks == 3u);
    CHECK(plan.fills == std::vector<std::uint32_t>{2u, 1u, 2u});
    CHECK(plan.tableWords() == 3u + 3u);

    std::vector<std::uint32_t> const table = buildDenseTable(plan, {10u, 11u, 12u});
    CHECK(table == std::vector<std::uint32_t>{0u, 2u, 2u, 10u, 11u, 12u});
}

TEST_CASE("dense bin plan handles empty and single-bin batches", "[bintable]")
{
    DenseBinPlan const empty = planDenseBins({0u, 0u}, 8u);
    CHECK(empty.binBase == std::vector<std::uint32_t>{0u, 0u});
    CHECK(empty.numOutputChunks == 0u);
    CHECK(empty.fills.empty());

    DenseBinPlan const one = planDenseBins({5u}, 5u);
    CHECK(one.binBase == std::vector<std::uint32_t>{0u});
    CHECK(one.numOutputChunks == 1u);
    CHECK(one.fills == std::vector<std::uint32_t>{5u});

    DenseBinPlan const exact = planDenseBins({4u, 4u}, 4u);
    CHECK(exact.numOutputChunks == 2u);
    CHECK(exact.fills == std::vector<std::uint32_t>{4u, 4u});
}

TEST_CASE("dense bin plan honours the reservation bound", "[bintable]")
{
    // 7 particles, capacity 2, 3 bins -> 4 + 3 - 1 = 6 reserved, 3 exact.
    DenseBinPlan const plan = planDenseBins({3u, 2u, 2u}, 2u);
    CHECK(plan.numOutputChunks == 4u); // ceil(3/2) + ceil(2/2) + ceil(2/2)
    CHECK(graphFlow::outputChunkBound(7u, 2u, 3u) == 6u);
    CHECK(plan.numOutputChunks <= graphFlow::outputChunkBound(7u, 2u, 3u));

    CHECK_THROWS_AS(planDenseBins({}, 1u), std::invalid_argument);
    CHECK_THROWS_AS(planDenseBins({1u}, 0u), std::invalid_argument);
    CHECK_THROWS_AS(buildDenseTable(plan, {1u}), std::invalid_argument);
}
