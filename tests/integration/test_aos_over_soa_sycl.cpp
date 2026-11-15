/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/AosOverSoa.hpp"

#include <catch2/catch_test_macros.hpp>

#if !ALPAKA_LANG_ONEAPI
#    error "test_aos_over_soa_sycl.cpp requires a oneAPI compilation (GRAPHFLOW_ENABLE_SYCL=ON)"
#endif

TEST_CASE("SYCL AoS flow matches the scalar oracle", "[sycl][flow][aos]")
{
    auto device = graphFlow::test::makeSyclDevice();
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 129ull})
        graphFlow::test::aos::checkFlowAgainstOracle(device, particles);
}

TEST_CASE("SYCL AoS storage keeps the bin-id scratch and one launch per stage", "[sycl][flow][aos]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::aos::checkScratchColumn(device);
    graphFlow::test::aos::checkLaunchCountsAndCopyAll(device);
}

TEST_CASE("SYCL default payload storages stay the regression baseline", "[sycl][flow][aos][default-storage]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::aos::checkDefaultStorages(device);
}
