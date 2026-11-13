/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"

#include "graphFlow/core/Backend.hpp"
#include "graphFlow/detail/QueueProbe.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

TEST_CASE("CPU backend selection creates a host device", "[backend]")
{
    using namespace graphFlow;

    CHECK(parseBackend("cpu") == Backend::cpu);
    CHECK(parseBackend("cuda") == Backend::cuda);
    CHECK(parseBackend("hip") == Backend::hip);
    CHECK_FALSE(parseBackend("tpu").has_value());
    CHECK(isBackendCompiled(Backend::cpu));

    int visited = 0;
    int const outcome = withBackendDevice(
        Backend::cpu,
        0u,
        [&](auto device, Backend selected, std::string const& deviceName)
        {
            CHECK(selected == Backend::cpu);
            CHECK(backendName(selected) == std::string_view{"cpu"});
            CHECK_FALSE(device.getName().empty());
            CHECK_FALSE(deviceName.empty());
            ++visited;
            return 7;
        });
    CHECK(outcome == 7);
    CHECK(visited == 1);
}

TEST_CASE("unknown backend names are rejected", "[backend]")
{
    using namespace graphFlow;

    CHECK_THROWS_AS(requireBackend("tpu"), BackendError);
    CHECK_THROWS_WITH(
        requireBackend("tpu"),
        Catch::Matchers::ContainsSubstring("unknown backend 'tpu'"));
}

TEST_CASE("uncompiled backends report the matching CMake option", "[backend]")
{
    using namespace graphFlow;

    for(Backend const backend : {Backend::cuda, Backend::hip})
    {
        if(isBackendCompiled(backend))
            continue;

        std::string expectedFlag = backend == Backend::cuda ? "GRAPHFLOW_ENABLE_CUDA" : "GRAPHFLOW_ENABLE_HIP";
        try
        {
            withBackendDevice(backend, 0u, [](auto, Backend, std::string const&) { return 0; });
            FAIL("an uncompiled backend must not select a device");
        }
        catch(BackendUnavailable const& error)
        {
            CHECK(std::string{error.what()}.find(expectedFlag) != std::string::npos);
        }
    }
}

TEST_CASE("device index outside the selector range is rejected", "[backend]")
{
    using namespace graphFlow;

    CHECK_THROWS_AS(
        withBackendDevice(Backend::cpu, 1'000'000u, [](auto, Backend, std::string const&) { return 0; }),
        std::out_of_range);
}

TEST_CASE("queue independence probe runs on the CPU backend", "[backend][queue]")
{
    using namespace graphFlow;

    auto device = test::makeCpuDevice();
    QueueProbeResult const result
        = probeQueueIndependence(device, 20'000'000u, std::chrono::milliseconds{10'000});

    // Reporting is part of the contract: the caller records the overlap verdict.
    std::cout << "# queue-probe overlap=" << toString(result.overlap)
              << " queues-alias=" << (result.queuesAlias ? "yes" : "no")
              << " second-before-first=" << (result.secondCompletedWhileFirstInFlight ? "yes" : "no")
              << " delay-iterations=" << result.delayIterations
              << " poll-us=" << result.pollDurationUs << '\n';

    CHECK(result.delayIterations == 20'000'000u);
    CHECK(result.overlap != QueueOverlap::unknown);
    if(result.overlap == QueueOverlap::yes)
        CHECK(result.secondCompletedWhileFirstInFlight);
    CHECK(result.pollDurationUs > 0u);
}
