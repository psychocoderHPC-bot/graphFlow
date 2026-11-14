/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/FailurePaths.hpp"

#include <catch2/catch_test_macros.hpp>

#if !ALPAKA_LANG_CUDA
#    error "test_failure_paths_cuda.cpp requires a CUDA compilation (GRAPHFLOW_ENABLE_CUDA=ON)"
#endif

TEST_CASE("CUDA failure: invalid classifier result rejects the batch", "[cuda][failure][invalid]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::failure::checkInvalidClassifierResult(device);
}

TEST_CASE("CUDA failure: invalid updater result rejects the batch", "[cuda][failure][invalid]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::failure::checkInvalidUpdaterResult(device);
}

TEST_CASE("CUDA failure: a full hash table aborts without a partial sink write", "[cuda][failure][hash]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::failure::checkHashTableFull(device);
}

TEST_CASE("CUDA failure: pool exhaustion during planning releases every chunk", "[cuda][failure][pool]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::failure::checkPoolExhaustion(device);
}

TEST_CASE("CUDA failure: an enqueue failure aborts in-flight work exactly once", "[cuda][failure][abort]")
{
    auto device = graphFlow::test::makeCudaDevice();
    graphFlow::test::failure::checkEnqueueAbort(device);
}
