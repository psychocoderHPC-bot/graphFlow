/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "integration/FailurePaths.hpp"

#include <catch2/catch_test_macros.hpp>

#if !ALPAKA_LANG_ONEAPI
#    error "test_failure_paths_sycl.cpp requires a SYCL compilation (GRAPHFLOW_ENABLE_SYCL=ON)"
#endif

TEST_CASE("SYCL failure: invalid classifier result rejects the batch", "[sycl][failure][invalid]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::failure::checkInvalidClassifierResult(device);
}

TEST_CASE("SYCL failure: invalid updater result rejects the batch", "[sycl][failure][invalid]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::failure::checkInvalidUpdaterResult(device);
}

TEST_CASE("SYCL failure: a full hash table aborts without a partial sink write", "[sycl][failure][hash]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::failure::checkHashTableFull(device);
}

TEST_CASE("SYCL failure: pool exhaustion during planning releases every chunk", "[sycl][failure][pool]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::failure::checkPoolExhaustion(device);
}

TEST_CASE("SYCL failure: an enqueue failure aborts in-flight work exactly once", "[sycl][failure][abort]")
{
    auto device = graphFlow::test::makeSyclDevice();
    graphFlow::test::failure::checkEnqueueAbort(device);
}
