/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

TEST_CASE("chunk pool hands out unique chunks and detects exhaustion", "[pool]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    graphFlow::ChunkPool pool{device, 3u, 4u};

    CHECK(pool.numChunks() == 3u);
    CHECK(pool.chunkCapacity() == 4u);
    CHECK(pool.freeCount() == 3u);
    CHECK(pool.liveCount() == 0u);

    auto first = pool.acquire();
    auto second = pool.acquire();
    auto third = pool.acquire();

    CHECK(pool.freeCount() == 0u);
    CHECK(pool.liveCount() == 3u);
    CHECK(first.generation == 0u);
    CHECK(pool.isLive(first));
    CHECK(pool.base(first) == static_cast<std::uint64_t>(first.index) * 4u);

    std::vector<std::uint32_t> indices{first.index, second.index, third.index};
    std::sort(indices.begin(), indices.end());
    CHECK(indices == std::vector<std::uint32_t>{0u, 1u, 2u});

    CHECK_THROWS_AS(pool.acquire(), std::runtime_error);
}

TEST_CASE("chunk generation invalidates stale handles", "[pool]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    graphFlow::ChunkPool pool{device, 2u, 1u};

    graphFlow::ChunkHandle const handle = pool.acquire();
    CHECK(handle.generation == 0u);
    CHECK(pool.isLive(handle));

    pool.release(handle);
    CHECK_FALSE(pool.isLive(handle));
    CHECK(pool.generation(handle.index) == 1u);

    graphFlow::ChunkHandle const reused = pool.acquire();
    CHECK(reused.index == handle.index);
    CHECK(reused.generation == 1u);
    CHECK(pool.isLive(reused));

    // Releasing the retired generation again is rejected instead of double-freeing the chunk.
    CHECK_THROWS_AS(pool.release(handle), std::invalid_argument);

    // Out-of-range handles are rejected as well.
    CHECK_THROWS_AS(pool.release(graphFlow::ChunkHandle{7u, 0u}), std::invalid_argument);
}

TEST_CASE("chunk pool rejects degenerate configurations", "[pool]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    CHECK_THROWS_AS((graphFlow::ChunkPool{device, 0u, 4u}), std::invalid_argument);
    CHECK_THROWS_AS((graphFlow::ChunkPool{device, 4u, 0u}), std::invalid_argument);
}
