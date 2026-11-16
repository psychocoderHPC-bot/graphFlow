/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/flow/Tuning.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>

namespace
{
    /** A30-shaped properties: 56 SMs, warp 32, 1024 threads and a wide grid. */
    alpaka::onHost::DeviceProperties a30Properties()
    {
        alpaka::onHost::DeviceProperties properties{};
        properties.warpSize = 32u;
        properties.maxThreadsPerBlock = 1024u;
        properties.maxBlocksPerGrid = 0x7FFF'FFFFu;
        properties.multiProcessorCount = 56u;
        // 24 GiB / 8 is the documented 3 GiB policy cap, which holds the C=8192, P=16384 pair.
        properties.globalMemCapacityBytes = 24ull * 1024ull * 1024ull * 1024ull;
        return properties;
    }

    alpaka::onHost::DeviceProperties withSms(alpaka::onHost::DeviceProperties properties, std::uint32_t sms)
    {
        properties.multiProcessorCount = sms;
        return properties;
    }

    /** RDNA3-shaped properties as reported by the HIP backend on hal. */
    alpaka::onHost::DeviceProperties hip256Properties()
    {
        alpaka::onHost::DeviceProperties properties = a30Properties();
        properties.warpSize = 32u;
        properties.maxThreadsPerBlock = 256u;
        properties.multiProcessorCount = 48u;
        return properties;
    }
} // namespace

TEST_CASE("chunk schedule policy selects 256-wide blocks and reports the maximum clamp", "[policy][chunk-schedule]")
{
    using namespace graphFlow;

    // The wave term is 10-40x maxChunkCapacity on every documented target GPU, so the maximum
    // clamp sets C and the wave and item knobs cannot lift it.
    struct Target
    {
        char const* name;
        std::uint32_t sms;
    };

    for(Target const target : {Target{"A30", 56u}, Target{"RTX 2080 Ti", 30u}, Target{"GV100", 80u}})
    {
        CAPTURE(target.name);
        auto const properties = withSms(a30Properties(), target.sms);
        for(std::uint32_t const threadsPerBlock : {256u, 512u})
        {
            for(std::uint32_t const wavesPerChunk : {2u, 4u})
            {
                CAPTURE(threadsPerBlock, wavesPerChunk);
                ChunkScheduleConfig config;
                config.threadsPerBlock = threadsPerBlock;
                config.wavesPerChunk = wavesPerChunk;
                ChunkSchedulePolicy const policy = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
                CHECK(policy.chunkCapacity == 8192u);
                CHECK(policy.limitName == "maximum");
                CHECK(policy.threadsPerBlock == threadsPerBlock);
                CHECK(policy.wavesPerChunk == wavesPerChunk);
                CHECK(policy.itemsPerThread == 1u);
                CHECK(policy.multiprocessorCount == target.sms);
                CHECK(policy.chunkCapacity % policy.threadsPerBlock == 0u);
            }
        }
    }
}

TEST_CASE("chunk schedule policy reports the wave clamp and honours items per thread", "[policy][chunk-schedule]")
{
    using namespace graphFlow;

    auto const properties = a30Properties();
    ChunkScheduleConfig config;
    config.maxChunkCapacity = 200000u;

    ChunkSchedulePolicy const waves2 = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
    // 56 * 4 * 256 * 1 * 2
    CHECK(waves2.chunkCapacity == 114688u);
    CHECK(waves2.limitName == "wave");

    config.wavesPerChunk = 3u;
    ChunkSchedulePolicy const waves3 = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
    CHECK(waves3.chunkCapacity == 172032u);
    CHECK(waves3.limitName == "wave");

    config.wavesPerChunk = 2u;
    config.itemsPerThread = 2u;
    ChunkSchedulePolicy const items2 = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
    // 56 * 4 * 256 * 2 * 2 exceeds the configured maximum, which then binds.
    CHECK(items2.chunkCapacity == 200000u - 200000u % 256u);
    CHECK(items2.limitName == "maximum");
    CHECK(items2.itemsPerThread == 2u);
}

TEST_CASE("chunk schedule policy clamps by grid, memory and latency", "[policy][chunk-schedule]")
{
    using namespace graphFlow;

    auto const properties = a30Properties();

    SECTION("grid limit")
    {
        auto gridLimited = properties;
        gridLimited.maxBlocksPerGrid = 8u;
        ChunkSchedulePolicy const policy = makeChunkSchedulePolicy(gridLimited, 4ull << 30u);
        CHECK(policy.chunkCapacity == 8u * 256u);
        CHECK(policy.limitName == "grid");
    }

    SECTION("payload budget")
    {
        ChunkScheduleConfig config;
        config.slotBytes = 24u;
        // 1024 slots * 24 B, below the 8192 maximum.
        ChunkSchedulePolicy const policy = makeChunkSchedulePolicy(properties, 1024ull * 24ull, config);
        CHECK(policy.chunkCapacity == 1024u);
        CHECK(policy.limitName == "memory");
    }

    SECTION("latency cap is a wave budget")
    {
        ChunkScheduleConfig config;
        config.maxChunkCapacity = 200000u;
        config.wavesPerChunk = 4u;
        config.waveLatencyUs = 10u;
        config.latencyCapUs = 25u; // two waves
        ChunkSchedulePolicy const policy = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
        CHECK(policy.wavesPerChunk == 2u);
        CHECK(policy.chunkCapacity == 114688u);
        CHECK(policy.limitName == "latency");

        // A cap below one wave still allows the one wave a launch needs.
        config.latencyCapUs = 3u;
        ChunkSchedulePolicy const oneWave = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
        CHECK(oneWave.wavesPerChunk == 1u);
        CHECK(oneWave.chunkCapacity == 57344u);
        CHECK(oneWave.limitName == "latency");

        // Zero disables the clamp; four waves exceed the configured maximum, so that binds.
        config.latencyCapUs = 0u;
        ChunkSchedulePolicy const uncapped = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
        CHECK(uncapped.wavesPerChunk == 4u);
        CHECK(uncapped.chunkCapacity == 200000u - 200000u % 256u);
        CHECK(uncapped.limitName == "maximum");

        // A generous cap keeps the requested wave count.
        config.latencyCapUs = 100u;
        ChunkSchedulePolicy const generous = makeChunkSchedulePolicy(properties, 4ull << 30u, config);
        CHECK(generous.wavesPerChunk == 4u);
        CHECK(generous.limitName == "maximum");
    }
}

TEST_CASE("chunk schedule policy rejects unsupported knobs and thread counts", "[policy][chunk-schedule]")
{
    using namespace graphFlow;

    auto const properties = a30Properties();
    auto const budget = 4ull << 30u;

    ChunkScheduleConfig config;
    config.wavesPerChunk = 1u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config.wavesPerChunk = 5u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);

    config = {};
    config.itemsPerThread = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config = {};
    config.targetBlocksPerSm = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config = {};
    config.maxChunkCapacity = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config = {};
    config.waveLatencyUs = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config = {};
    config.slotBytes = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);

    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, 0u), std::invalid_argument);

    config = {};
    config.threadsPerBlock = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config.threadsPerBlock = 100u; // not a warp multiple
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);
    config.threadsPerBlock = 2048u; // above maxThreadsPerBlock
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, budget, config), std::invalid_argument);

    auto noThreads = properties;
    noThreads.maxThreadsPerBlock = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(noThreads, budget), std::runtime_error);
    auto noBlocks = properties;
    noBlocks.maxBlocksPerGrid = 0u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(noBlocks, budget), std::runtime_error);

    // A budget that cannot hold one block rounds down to zero and is rejected.
    ChunkScheduleConfig narrow;
    narrow.slotBytes = 24u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, 23u, narrow), std::invalid_argument);

    // HIP-class device: 512-wide blocks are unsupported and rejected like an explicit tile.
    auto const hip = hip256Properties();
    ChunkScheduleConfig wide;
    wide.threadsPerBlock = 512u;
    CHECK_THROWS_AS(makeChunkSchedulePolicy(hip, budget, wide), std::invalid_argument);
    ChunkScheduleConfig accepted;
    accepted.threadsPerBlock = 256u;
    CHECK(makeChunkSchedulePolicy(hip, budget, accepted).threadsPerBlock == 256u);
}

TEST_CASE("chunk schedule frames cover the C/T matrix on degenerate inputs", "[policy][chunk-schedule][framespec]")
{
    using namespace graphFlow;

    auto const properties = a30Properties();
    for(std::uint32_t const capacity : {1u, 31u, 127u, 128u, 8192u})
    {
        for(std::uint32_t const tileSize : {32u, 128u, 512u})
        {
            CAPTURE(capacity, tileSize);
            std::uint32_t const frames = numFramesFor(capacity, tileSize, 8u, properties);
            CHECK(frames >= 1u);
            // Every fill is covered: the stride loop walks exactly `fill` items.
            CHECK(std::uint64_t{frames} * tileSize >= capacity);
        }
    }

    // The existing launch policy rejects an unsupported thread count, which is the rejection the
    // per-chunk path relies on for HIP-class devices.
    CHECK_THROWS_AS(numFramesFor(1024u, 512u, 8u, hip256Properties()), std::invalid_argument);
    CHECK_NOTHROW(numFramesFor(1024u, 256u, 8u, hip256Properties()));
}

TEST_CASE("hash bin path rejects wide tiles and the dense path accepts them", "[policy][chunk-schedule][config]")
{
    using namespace graphFlow;

    Config dense;
    dense.maxBins = 64u;
    dense.tileSize = 512u;
    dense.binPath = BinPath::dense;
    CHECK_NOTHROW(dense.validate());

    Config hash;
    hash.maxBins = 4096u; // above denseBinLimit, so automatic resolves to hash
    hash.tileSize = 129u;
    CHECK_THROWS_AS(hash.validate(), std::invalid_argument);

    hash.tileSize = Config::hashBinTableTileSize;
    CHECK_NOTHROW(hash.validate());

    Config explicitHash;
    explicitHash.maxBins = 64u;
    explicitHash.tileSize = 512u;
    explicitHash.binPath = BinPath::hash;
    CHECK_THROWS_AS(explicitHash.validate(), std::invalid_argument);
}

TEST_CASE("Config and Tuning default to the off schedule and validate queue depth", "[policy][chunk-schedule][config]")
{
    using namespace graphFlow;

    Config const defaults;
    CHECK(defaults.chunkKernelSchedule == ChunkKernelSchedule::off);
    CHECK(defaults.chunkQueueDepth == 1u);
    CHECK(defaults.chunkLatencyCapUs == 0u);
    CHECK_NOTHROW(defaults.validate());

    Config perChunk;
    perChunk.chunkKernelSchedule = ChunkKernelSchedule::perChunk;
    perChunk.chunkQueueDepth = 8u;
    perChunk.chunkLatencyCapUs = 25u;
    CHECK_NOTHROW(perChunk.validate());

    perChunk.chunkQueueDepth = 0u;
    CHECK_THROWS_AS(perChunk.validate(), std::invalid_argument);

    Tuning tuning;
    CHECK_FALSE(tuning.chunkKernelSchedule.has_value());
    CHECK_FALSE(tuning.chunkQueueDepth.has_value());
    CHECK_FALSE(tuning.chunkLatencyCapUs.has_value());
    CHECK_NOTHROW(tuning.validate());

    // Zero latency means "no cap" and is valid; zero queue depth is not.
    tuning.chunkLatencyCapUs = 0u;
    tuning.chunkQueueDepth = 4u;
    CHECK_NOTHROW(tuning.validate());
    tuning.chunkQueueDepth = 0u;
    CHECK_THROWS_AS(tuning.validate(), std::invalid_argument);
}

TEST_CASE("DevicePolicy reports the guarded multiprocessor count", "[policy][chunk-schedule]")
{
    using namespace graphFlow;

    auto const properties = a30Properties();
    DevicePolicy const policy = makeDevicePolicy(properties, 100'000'000u);
    CHECK(policy.multiprocessorCount == 56u);
    // The field is additive: the documented off-path sizing is unchanged.
    CHECK(policy.chunkCapacity == 8192u);
    CHECK(policy.liveParticles == 16384u);

    auto noSms = properties;
    noSms.multiProcessorCount = 0u;
    DevicePolicy const guarded = makeDevicePolicy(noSms, 100'000'000u);
    CHECK(guarded.multiprocessorCount == 1u);
}
