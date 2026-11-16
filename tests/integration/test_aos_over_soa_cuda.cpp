/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/AosOverSoa.hpp"

#include <catch2/catch_test_macros.hpp>

#if !ALPAKA_LANG_CUDA
#    error "test_aos_over_soa_cuda.cpp requires a CUDA compilation (GRAPHFLOW_ENABLE_CUDA=ON)"
#endif

TEST_CASE("CUDA AoS flow matches the scalar oracle", "[cuda][flow][aos]")
{
    auto device = graphFlow::test::makeCudaDevice();
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 129ull})
    {
        graphFlow::test::aos::checkFlowAgainstOracle(device, particles);
        graphFlow::test::aos::checkSubsetFlowAgainstOracle(device, particles);
    }
}

TEST_CASE("CUDA AoS storage keeps the bin-id scratch and one launch per stage", "[cuda][flow][aos]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::aos::checkScratchColumn(device);
    graphFlow::test::aos::checkLaunchCountsAndCopyAll(device);
}

TEST_CASE("CUDA default payload storages stay the regression baseline", "[cuda][flow][aos][default-storage]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::aos::checkDefaultStorages(device);
}
