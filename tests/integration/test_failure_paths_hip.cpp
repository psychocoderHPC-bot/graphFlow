/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/FailurePaths.hpp"

#include <catch2/catch_test_macros.hpp>

#if !ALPAKA_LANG_HIP
#    error "test_failure_paths_hip.cpp requires a HIP compilation (GRAPHFLOW_ENABLE_HIP=ON)"
#endif

TEST_CASE("HIP failure: invalid classifier result rejects the batch", "[hip][failure][invalid]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::failure::checkInvalidClassifierResult(device);
}

TEST_CASE("HIP failure: invalid updater result rejects the batch", "[hip][failure][invalid]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::failure::checkInvalidUpdaterResult(device);
}

TEST_CASE("HIP failure: a full hash table aborts without a partial sink write", "[hip][failure][hash]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::failure::checkHashTableFull(device);
}

TEST_CASE("HIP failure: pool exhaustion during planning releases every chunk", "[hip][failure][pool]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::failure::checkPoolExhaustion(device);
}

TEST_CASE("HIP failure: an enqueue failure aborts in-flight work exactly once", "[hip][failure][abort]")
{
    auto device = graphFlow::test::makeHipDevice();
    graphFlow::test::failure::checkEnqueueAbort(device);
}
