/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/graph/RouteTable.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using graphFlow::RouteTable;

TEST_CASE("route table carries one destination per bin", "[unit][graph][route]")
{
    RouteTable const routes{{1u, 2u, 3u}};

    REQUIRE(routes.size() == 3u);
    REQUIRE(routes.bins() == 3u);
    REQUIRE_FALSE(routes.empty());
    CHECK(routes.destination(0u) == 1u);
    CHECK(routes.destination(1u) == 2u);
    CHECK(routes[2u] == 3u);
    CHECK(routes.destinations().size() == 3u);
    CHECK(routes.destinations().front() == 1u);
    CHECK_THROWS_AS(routes.destination(3u), std::out_of_range);
}

TEST_CASE("empty route table represents a sink", "[unit][graph][route]")
{
    RouteTable const sink{};

    REQUIRE(sink.empty());
    CHECK(sink.size() == 0u);
    CHECK(sink.bins() == 0u);
    CHECK(sink.destinations().empty());
    CHECK_THROWS_AS(sink.destination(0u), std::out_of_range);
}

TEST_CASE("route table validates its explicit bin count", "[unit][graph][route]")
{
    RouteTable const routes{2u, std::vector<graphFlow::NodeId>{4u, 5u}};

    REQUIRE(routes.bins() == 2u);
    CHECK(routes.destination(1u) == 5u);
    CHECK_NOTHROW(routes.validateBins(2u));
    CHECK_THROWS_AS(routes.validateBins(1u), std::invalid_argument);
    CHECK_THROWS_AS(routes.validateBins(3u), std::invalid_argument);
    CHECK_THROWS_AS((RouteTable{3u, std::vector<graphFlow::NodeId>{1u, 2u}}), std::invalid_argument);
}

TEST_CASE("feedback edges are ordinary earlier destinations", "[unit][graph][route]")
{
    // Nothing in the table distinguishes a forward edge from a feedback edge. A feedback edge is
    // simply a destination that names an earlier node id.
    RouteTable const routes{{0u, 2u}};

    REQUIRE(routes.bins() == 2u);
    CHECK(routes.destination(0u) == 0u);
    CHECK(routes.destination(1u) == 2u);
}
