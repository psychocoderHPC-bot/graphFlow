/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace
{
    alpaka::onHost::DeviceProperties acceleratorProperties()
    {
        alpaka::onHost::DeviceProperties properties{};
        properties.warpSize = 32u;
        properties.maxThreadsPerBlock = 1024u;
        properties.maxBlocksPerGrid = 0x7FFF'FFFFu;
        properties.multiProcessorCount = 108u;
        return properties;
    }
} // namespace

TEST_CASE("frame count follows ceil, grid and frameCap bounds", "[framespec]")
{
    using namespace graphFlow;

    auto const properties = acceleratorProperties();
    CHECK(numFramesFor(0u, 128u, 8u, properties) == 0u);
    CHECK(numFramesFor(1u, 128u, 8u, properties) == 1u);
    CHECK(numFramesFor(17u, 128u, 8u, properties) == 1u);
    CHECK(numFramesFor(127u, 128u, 8u, properties) == 1u);
    CHECK(numFramesFor(128u, 128u, 8u, properties) == 1u);
    CHECK(numFramesFor(129u, 128u, 8u, properties) == 2u);
    CHECK(numFramesFor(4097u, 128u, 8u, properties) == 33u);
    CHECK(numFramesFor(1u << 20, 32u, 8u, properties) == 8u * 108u); // frameCap * SMs binds

    // A generous cap is still bounded by maxBlocksPerGrid.
    auto fewBlocks = properties;
    fewBlocks.maxBlocksPerGrid = 64u;
    CHECK(numFramesFor(1u << 20, 128u, 8u, fewBlocks) == 64u);
    // frameCap=1 with one SM yields one block for an arbitrarily large amount of work.
    auto oneSm = properties;
    oneSm.multiProcessorCount = 1u;
    CHECK(numFramesFor(1u << 20, 128u, 1u, oneSm) == 1u);
    // Uncapped is still bounded by maxBlocksPerGrid.
    CHECK(
        numFramesFor(
            (std::uint64_t{properties.maxBlocksPerGrid} + 1u) * 128u,
            128u,
            std::numeric_limits<std::uint32_t>::max(),
            properties)
        == properties.maxBlocksPerGrid);
}

TEST_CASE("frame count rejects invalid tile and frame caps", "[framespec]")
{
    using namespace graphFlow;

    auto const properties = acceleratorProperties();
    CHECK_THROWS_AS(numFramesFor(1024u, 0u, 8u, properties), std::invalid_argument);
    CHECK_THROWS_AS(numFramesFor(1024u, 128u, 0u, properties), std::invalid_argument);
    CHECK_THROWS_AS(numFramesFor(1024u, properties.maxThreadsPerBlock + 1u, 8u, properties), std::invalid_argument);

    auto noBlocks = properties;
    noBlocks.maxBlocksPerGrid = 0u;
    CHECK_THROWS_AS(numFramesFor(1024u, 128u, 8u, noBlocks), std::runtime_error);

    auto noThreads = properties;
    noThreads.maxThreadsPerBlock = 0u;
    CHECK_THROWS_AS(numFramesFor(1024u, 128u, 8u, noThreads), std::runtime_error);
}

TEST_CASE("host launch policy uses a unit frame extent", "[framespec]")
{
    using namespace graphFlow;

    auto device = test::makeCpuDevice();
    auto const properties = device.getDeviceProperties();

    LaunchPolicy const policy = launchPolicy(device, 4097u, 128u, 8u);
    CHECK(policy.frameExtent == 1u);
    CHECK(policy.tileSize == 128u);
    CHECK(policy.numFrames == numFramesFor(4097u, 128u, 8u, properties));
    CHECK(policy.numFrames > 0u);
    CHECK(policy.frameSpec().getNumFrames().x() == policy.numFrames);
    CHECK(policy.frameSpec().getFrameExtents().x() == policy.frameExtent);

    CHECK(launchPolicy(device, 0u, 128u, 8u).numFrames == 0u);
    CHECK_THROWS_AS(launchPolicy(device, 1024u, properties.maxThreadsPerBlock + 1u, 8u), std::invalid_argument);
}
