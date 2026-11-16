/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/AosOverSoa.hpp"

#include <catch2/catch_test_macros.hpp>

#if !ALPAKA_LANG_HIP
#    error "test_aos_over_soa_hip.cpp requires a HIP compilation (GRAPHFLOW_ENABLE_HIP=ON)"
#endif

TEST_CASE("HIP AoS flow matches the scalar oracle", "[hip][flow][aos]")
{
    auto device = graphFlow::test::makeHipDevice();
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 129ull})
    {
        graphFlow::test::aos::checkFlowAgainstOracle(device, particles);
        graphFlow::test::aos::checkSubsetFlowAgainstOracle(device, particles);
    }
}

TEST_CASE("HIP AoS storage keeps the bin-id scratch and one launch per stage", "[hip][flow][aos]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::aos::checkScratchColumn(device);
    graphFlow::test::aos::checkLaunchCountsAndCopyAll(device);
}

TEST_CASE("HIP default payload storages stay the regression baseline", "[hip][flow][aos][default-storage]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::aos::checkDefaultStorages(device);
}
