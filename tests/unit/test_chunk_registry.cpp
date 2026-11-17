/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/core/ChunkRegistry.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

TEST_CASE("chunk registry follows one incarnation through every phase", "[chunk-registry][unit]")
{
    graphFlow::ChunkRegistry registry{4u};
    REQUIRE(registry.summary().chunks == 4u);
    REQUIRE(registry.summary().freeChunks == 4u);

    graphFlow::ChunkHandle const handle{2u, 0u};
    registry.onAcquire(handle);
    {
        graphFlow::ChunkRegistrySummary const summary = registry.summary();
        CHECK(summary.reservedChunks == 1u);
        CHECK(summary.freeChunks == 3u);
        CHECK(summary.acquisitions == 1u);
        CHECK(summary.peakReservedChunks == 1u);
        CHECK(registry.records()[2u].phase == graphFlow::ChunkPhase::reserved);
        CHECK(registry.records()[2u].handle == handle);
    }

    graphFlow::ChunkDescriptor const chunk{handle, 4u, 1u};
    registry.onDispatch(chunk, 5u);
    {
        graphFlow::ChunkRecord const& record = registry.records()[2u];
        CHECK(record.phase == graphFlow::ChunkPhase::inflight);
        CHECK(record.ownerNode == 5u);
        CHECK(record.producingNode == 5u);
        CHECK(record.fill == 4u);
        CHECK(record.bin == 1u);
    }

    registry.onRoute(chunk, 5u, 7);
    {
        graphFlow::ChunkRecord const& record = registry.records()[2u];
        CHECK(record.phase == graphFlow::ChunkPhase::ready);
        CHECK(record.ownerNode == 7u);
        CHECK(record.destinationNode == 7);
        CHECK(record.producingNode == 5u);
    }

    registry.onDispatch(chunk, 7u);
    registry.onSinkArrival(chunk, 7u);
    {
        graphFlow::ChunkRegistrySummary const summary = registry.summary();
        CHECK(summary.retiredChunks == 1u);
        CHECK(summary.retirements == 1u);
        CHECK(summary.routings == 1u);
        CHECK(summary.dispatches == 2u);
    }

    graphFlow::ChunkHandle const released{2u, 1u};
    registry.onRelease(released);
    {
        graphFlow::ChunkRegistrySummary const summary = registry.summary();
        CHECK(summary.freeChunks == 4u);
        CHECK(summary.releases == 1u);
        CHECK(registry.records()[2u].phase == graphFlow::ChunkPhase::free);
        CHECK(registry.records()[2u].handle == released);
        CHECK(registry.records()[2u].ownerNode == graphFlow::invalidChunkNode);
    }
}

TEST_CASE("chunk registry rejects an acquire of a non-free chunk", "[chunk-registry][unit]")
{
    graphFlow::ChunkRegistry registry{2u};
    registry.onAcquire(graphFlow::ChunkHandle{0u, 0u});
    CHECK_THROWS_AS(registry.onAcquire(graphFlow::ChunkHandle{0u, 0u}), std::logic_error);
    CHECK_THROWS_AS(registry.onAcquire(graphFlow::ChunkHandle{9u, 0u}), std::logic_error);
}

TEST_CASE("chunk registry rejects an unroutable destination", "[chunk-registry][unit]")
{
    graphFlow::ChunkRegistry registry{2u};
    registry.onAcquire(graphFlow::ChunkHandle{1u, 0u});
    CHECK_THROWS_AS(registry.onRoute(graphFlow::ChunkDescriptor{{1u, 0u}, 1u, 0u}, 0u, -1), std::logic_error);
}

TEST_CASE("chunk registry resynchronises a chunk whose acquire was not observed", "[chunk-registry][unit]")
{
    graphFlow::ChunkRegistry registry{2u};
    // No onAcquire: the record still reports a free chunk of the current generation.
    graphFlow::ChunkHandle const handle{1u, 3u};
    registry.onRoute(graphFlow::ChunkDescriptor{handle, 2u, 0u}, 4u, 5);
    graphFlow::ChunkRegistrySummary const summary = registry.summary();
    CHECK(summary.acquisitions == 1u);
    CHECK(summary.routings == 1u);
    CHECK(summary.readyChunks == 1u);
    CHECK(registry.records()[1u].producingNode == 4u);
    CHECK(registry.records()[1u].ownerNode == 5u);
}

TEST_CASE("chunk registry reports output-credit reservations separately", "[chunk-registry][unit]")
{
    graphFlow::ChunkRegistry registry{2u};
    registry.onReserve(0u, 3u);
    registry.onReserve(0u, 2u);
    CHECK(registry.summary().reservations == 2u);
    CHECK(registry.summary().reservedCredits == 5u);
    CHECK(registry.summary().reservedChunks == 0u);
}
