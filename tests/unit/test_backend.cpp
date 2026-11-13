/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"

#include "graphFlow/core/Backend.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/detail/QueueProbe.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
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

namespace
{
    /** Policy memory invariant: the payload of the live-particle working set fits the budget. */
    bool policyFitsPayloadBudget(graphFlow::DevicePolicy const& policy)
    {
        std::uint64_t const bytes = std::uint64_t{policy.liveParticles} * policy.chunkCapacity
                                    * graphFlow::payloadBytesPerParticle;
        return bytes <= policy.memoryBudgetBytes;
    }
} // namespace

TEST_CASE("device policy derives CPU defaults", "[policy]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();
    DevicePolicy const policy = makeDevicePolicy(properties, 1'000'000u);

    std::uint32_t const warp = properties.warpSize == 0u ? 1u : properties.warpSize;

    CHECK(policy.tileSize >= warp);
    CHECK(policy.tileSize % warp == 0u);
    CHECK(policy.tileSize <= properties.maxThreadsPerBlock);
    if(warp == 1u)
        CHECK(policy.tileSize == std::min<std::uint32_t>(128u, properties.maxThreadsPerBlock));

    CHECK(policy.chunkCapacity > 0u);
    CHECK(policy.chunkCapacity % policy.tileSize == 0u);
    CHECK(policy.liveParticles <= 1'000'000u);
    CHECK(policy.liveParticles > 0u);
    CHECK(policy.liveParticles == std::min<std::uint32_t>(1'000'000u, 2u * policy.chunkCapacity));
    CHECK(policy.waves == 1u);
    CHECK(policy.targetBlocksPerSm == 4u);
    CHECK(policy.memoryBudgetBytes > 0u);
    CHECK(policyFitsPayloadBudget(policy));
    CHECK_FALSE(policy.description.empty());
    CHECK(policy.description.find("C=") != std::string::npos);
}

TEST_CASE("device policy keeps C a tile multiple and P within N", "[policy]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();
    for(std::uint64_t const n :
        {std::uint64_t{0u},
         std::uint64_t{1u},
         std::uint64_t{17u},
         std::uint64_t{127u},
         std::uint64_t{128u},
         std::uint64_t{4097u},
         std::uint64_t{1'000'000u}})
    {
        DevicePolicy const policy = makeDevicePolicy(properties, n);
        INFO("N=" << n);
        CHECK(policy.chunkCapacity % policy.tileSize == 0u);
        CHECK(policy.liveParticles <= n);
        CHECK(policyFitsPayloadBudget(policy));
        CHECK(policy.memoryBudgetBytes > 0u);
        if(n == 0u)
            CHECK(policy.liveParticles == 0u);
        else
            CHECK(policy.liveParticles > 0u);
    }
}

TEST_CASE("device policy caps the payload pool by the memory budget", "[policy]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();

    DevicePolicyConfig lowBudget = policyConfigOf(Config{});
    lowBudget.waves = 64u; // make the memory cap bind instead of the wave estimate
    lowBudget.memoryBudgetBytes = 64ull * 1024ull * 1024ull;

    DevicePolicyConfig highBudget = policyConfigOf(Config{});
    highBudget.waves = 64u;
    highBudget.memoryBudgetBytes = 1024ull * 1024ull * 1024ull;

    DevicePolicy const low = makeDevicePolicy(properties, 4'000'000u, {}, lowBudget);
    DevicePolicy const high = makeDevicePolicy(properties, 4'000'000u, {}, highBudget);

    CHECK(low.memoryBudgetBytes == lowBudget.memoryBudgetBytes);
    CHECK(high.memoryBudgetBytes == highBudget.memoryBudgetBytes);
    CHECK(low.chunkCapacity < high.chunkCapacity);
    CHECK(low.chunkCapacity % low.tileSize == 0u);
    CHECK(high.chunkCapacity % high.tileSize == 0u);
    CHECK(policyFitsPayloadBudget(low));
    CHECK(policyFitsPayloadBudget(high));
}

TEST_CASE("device policy derives the budget from the reported device memory", "[policy]")
{
    using namespace graphFlow;

    // Accelerator-shaped public fields on a real properties object. No private field is
    // fabricated and no occupancy data is invented.
    auto properties = test::makeCpuDevice().getDeviceProperties();
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 0x7FFFFFFFu;
    properties.multiProcessorCount = 108u;

    properties.globalMemCapacityBytes = 24ull * 1024ull * 1024ull * 1024ull;
    DevicePolicy const high = makeDevicePolicy(properties, 100'000'000u);

    properties.globalMemCapacityBytes = 2ull * 1024ull * 1024ull * 1024ull;
    DevicePolicy const low = makeDevicePolicy(properties, 100'000'000u);

    CHECK(high.tileSize == 128u);
    CHECK(high.memoryBudgetBytes == 3ull * 1024ull * 1024ull * 1024ull);
    CHECK(high.chunkCapacity == 8192u);
    CHECK(high.liveParticles == 16384u);
    CHECK(policyFitsPayloadBudget(high));

    CHECK(low.tileSize == 128u);
    CHECK(low.memoryBudgetBytes == 256ull * 1024ull * 1024ull);
    CHECK(low.chunkCapacity < high.chunkCapacity);
    CHECK(policyFitsPayloadBudget(low));
}

TEST_CASE("device policy preserves explicit overrides", "[policy]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();
    DevicePolicyConfig config = policyConfigOf(Config{});
    config.memoryBudgetBytes = 1024ull * 1024ull * 1024ull;

    DevicePolicyOverrides all;
    all.chunkCapacity = 256u;
    all.liveParticles = 512u;
    all.tileSize = 64u;
    DevicePolicy const explicitPolicy = makeDevicePolicy(properties, 4096u, all, config);
    CHECK(explicitPolicy.chunkCapacity == 256u);
    CHECK(explicitPolicy.liveParticles == 512u);
    CHECK(explicitPolicy.tileSize == 64u);
    CHECK(policyFitsPayloadBudget(explicitPolicy));

    // C and T are independent: a capacity that is not a tile multiple is kept exactly.
    DevicePolicyOverrides chunkAndTile;
    chunkAndTile.chunkCapacity = 100u;
    chunkAndTile.tileSize = 64u;
    DevicePolicy const odd = makeDevicePolicy(properties, 4096u, chunkAndTile, config);
    CHECK(odd.chunkCapacity == 100u);
    CHECK(odd.tileSize == 64u);
    CHECK(odd.liveParticles == 200u); // min(N, 2*C)
    CHECK(policyFitsPayloadBudget(odd));

    DevicePolicyOverrides tileOnly;
    tileOnly.tileSize = 32u;
    DevicePolicy const tilePolicy = makeDevicePolicy(properties, 4096u, tileOnly, config);
    CHECK(tilePolicy.tileSize == 32u);
    CHECK(tilePolicy.chunkCapacity % tilePolicy.tileSize == 0u);

    DevicePolicyOverrides liveOnly;
    liveOnly.liveParticles = 100u;
    DevicePolicy const livePolicy = makeDevicePolicy(properties, 4096u, liveOnly, config);
    CHECK(livePolicy.liveParticles == 100u);
    CHECK(policyFitsPayloadBudget(livePolicy));

    // An explicit C squeezes the derived P instead of rejecting C.
    DevicePolicyConfig moderate;
    moderate.memoryBudgetBytes = 8ull * 1024ull * 1024ull;
    DevicePolicyOverrides bigChunk;
    bigChunk.chunkCapacity = 8192u;
    DevicePolicy const squeezed = makeDevicePolicy(properties, 100'000u, bigChunk, moderate);
    CHECK(squeezed.chunkCapacity == 8192u);
    CHECK(squeezed.liveParticles <= 8ull * 1024ull * 1024ull / (8192ull * 24ull));
    CHECK(policyFitsPayloadBudget(squeezed));
}

TEST_CASE("device policy rejects impossible overrides", "[policy]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();
    DevicePolicyConfig config = policyConfigOf(Config{});
    config.memoryBudgetBytes = 1024ull * 1024ull * 1024ull;

    DevicePolicyOverrides overrides;
    overrides.chunkCapacity = 0u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, overrides, config), std::invalid_argument);

    overrides = {};
    overrides.tileSize = 0u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, overrides, config), std::invalid_argument);

    overrides = {};
    overrides.tileSize = properties.maxThreadsPerBlock + 1u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, overrides, config), std::invalid_argument);

    overrides = {};
    overrides.chunkCapacity = config.maxChunkCapacity + 1u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, overrides, config), std::invalid_argument);

    overrides = {};
    overrides.liveParticles = 0u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, overrides, config), std::invalid_argument);

    overrides = {};
    overrides.liveParticles = 1001u; // above N
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, overrides, config), std::invalid_argument);

    // C needs more tiles than maxBlocksPerGrid allows.
    auto oneBlockDevice = properties;
    oneBlockDevice.maxBlocksPerGrid = 1u;
    DevicePolicyOverrides gridChunk;
    gridChunk.chunkCapacity = 256u;
    CHECK_THROWS_AS(makeDevicePolicy(oneBlockDevice, 1000u, gridChunk, config), std::invalid_argument);

    // Explicit P with a budget too small for one tile.
    DevicePolicyConfig tinyBudget;
    tinyBudget.memoryBudgetBytes = 1024ull;
    DevicePolicyOverrides live;
    live.liveParticles = 512u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 4096u, live, tinyBudget), std::invalid_argument);

    // Explicit P and C that exceed the budget together.
    DevicePolicyConfig smallBudget;
    smallBudget.memoryBudgetBytes = 64ull * 1024ull;
    DevicePolicyOverrides both;
    both.chunkCapacity = 4096u;
    both.liveParticles = 1024u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 100'000u, both, smallBudget), std::invalid_argument);

    // Explicit C that cannot hold even one live particle.
    DevicePolicyConfig noRoom;
    noRoom.memoryBudgetBytes = 1024ull;
    DevicePolicyOverrides hugeChunk;
    hugeChunk.chunkCapacity = 8192u;
    CHECK_THROWS_AS(makeDevicePolicy(properties, 1000u, hugeChunk, noRoom), std::invalid_argument);
}

TEST_CASE("device policy honors shared-memory and launch limits", "[policy]")
{
    using namespace graphFlow;

    auto properties = test::makeCpuDevice().getDeviceProperties();
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 4u;
    properties.multiProcessorCount = 1u;
    properties.sharedMemPerBlockBytes = 32u * 64u; // room for a 2*T*16-byte table at T=64

    DevicePolicyConfig config = policyConfigOf(Config{});
    config.memoryBudgetBytes = 64ull * 1024ull * 1024ull;
    DevicePolicy const policy = makeDevicePolicy(properties, 1'000'000u, {}, config);

    CHECK(policy.tileSize == 64u);
    CHECK(policy.tileSize % 32u == 0u);
    CHECK(policy.chunkCapacity <= std::uint64_t{properties.maxBlocksPerGrid} * policy.tileSize);
    CHECK(policyFitsPayloadBudget(policy));
}

TEST_CASE("device policy is deterministic", "[policy]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();
    DevicePolicy const first = makeDevicePolicy(properties, 4097u);
    DevicePolicy const second = makeDevicePolicy(properties, 4097u);

    CHECK(first.chunkCapacity == second.chunkCapacity);
    CHECK(first.liveParticles == second.liveParticles);
    CHECK(first.tileSize == second.tileSize);
    CHECK(first.waves == second.waves);
    CHECK(first.targetBlocksPerSm == second.targetBlocksPerSm);
    CHECK(first.memoryBudgetBytes == second.memoryBudgetBytes);
    CHECK(first.description == second.description);
}

TEST_CASE("Config carries the device policy knobs", "[policy][config]")
{
    using namespace graphFlow;

    auto const properties = test::makeCpuDevice().getDeviceProperties();

    Config config;
    config.numParticles = 1'000'000u;
    config.policyWaves = 2u;
    config.policyTargetBlocksPerSm = 2u;
    config.policyMaxChunkCapacity = 2048u;
    config.policyLiveParticlesPerChunk = 3u;
    config.policyMemoryBudgetBytes = 128ull * 1024ull * 1024ull;
    config.validate();

    DevicePolicyConfig const policyConfig = policyConfigOf(config);
    CHECK(policyConfig.waves == 2u);
    CHECK(policyConfig.targetBlocksPerSm == 2u);
    CHECK(policyConfig.maxChunkCapacity == 2048u);
    CHECK(policyConfig.liveParticlesPerChunk == 3u);
    CHECK(policyConfig.memoryBudgetBytes == config.policyMemoryBudgetBytes);

    DevicePolicy const policy = makeDevicePolicy(properties, config.numParticles, {}, policyConfig);
    CHECK(policy.waves == 2u);
    CHECK(policy.targetBlocksPerSm == 2u);
    applyDevicePolicy(config, policy);
    CHECK(config.chunkCapacity == policy.chunkCapacity);
    CHECK(config.liveParticles == policy.liveParticles);
    CHECK(config.tileSize == policy.tileSize);
    CHECK_NOTHROW(config.validate());

    Config badKnobs;
    badKnobs.policyWaves = 0u;
    CHECK_THROWS_AS(badKnobs.validate(), std::invalid_argument);
}
