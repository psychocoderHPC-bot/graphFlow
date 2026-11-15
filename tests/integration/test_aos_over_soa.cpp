/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/AosOverSoa.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("CPU AoS flow matches the scalar oracle", "[integration][flow][aos]")
{
    auto device = graphFlow::test::makeCpuDevice();
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 129ull})
        graphFlow::test::aos::checkFlowAgainstOracle(device, particles);
}

TEST_CASE("CPU AoS storage keeps the bin-id scratch and one launch per stage", "[integration][flow][aos]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::test::aos::checkScratchColumn(device);
    graphFlow::test::aos::checkLaunchCountsAndCopyAll(device);
}

TEST_CASE("CPU default payload storages stay the regression baseline", "[integration][flow][aos][default-storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::test::aos::checkDefaultStorages(device);
}
