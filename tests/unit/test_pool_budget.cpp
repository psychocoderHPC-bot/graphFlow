/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ParticleStages.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/core/PoolBudget.hpp"
#include "graphFlow/core/Reservation.hpp"
#include "graphFlow/graph/BinTable.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <limits>

TEST_CASE("pool payload bytes are derived from the pool arrays", "[pool][budget]")
{
    using namespace graphFlow;

    CHECK(
        payloadBytesPerSlot
        == sizeof(PoolIdValue) + static_cast<std::uint64_t>(poolPayloadU32Arrays) * sizeof(PoolU32Value));

    CHECK(requiredPayloadBytes(0u, 0u) == 0u);
    CHECK(requiredPayloadBytes(0u, 7u) == 0u);
    CHECK(requiredPayloadBytes(3u, 7u) == std::uint64_t{3u} * 7u * payloadBytesPerSlot);

    // Both multiplications are checked; wrapping would silently produce a small "required" size.
    CHECK_THROWS_AS(
        requiredPayloadBytes(std::numeric_limits<std::uint64_t>::max(), 2u),
        std::overflow_error);
    CHECK_THROWS_AS(
        requiredPayloadBytes(std::numeric_limits<std::uint64_t>::max() / payloadBytesPerSlot + 1u, 1u),
        std::overflow_error);
}

TEST_CASE("pool budget accepts an exact fit and rejects one byte over", "[pool][budget]")
{
    using namespace graphFlow;

    std::uint64_t const chunks = 7u;
    std::uint64_t const capacity = 13u;
    std::uint64_t const bytes = requiredPayloadBytes(chunks, capacity);

    CHECK_NOTHROW(validatePayloadBudget(chunks, capacity, bytes));
    CHECK_THROWS_AS(validatePayloadBudget(chunks, capacity, bytes - 1u), std::invalid_argument);
    CHECK_THROWS_WITH(
        validatePayloadBudget(chunks, capacity, bytes - 1u),
        Catch::Matchers::ContainsSubstring("exceeding the payload budget"));

    // A zero budget must not act as "unlimited".
    CHECK_THROWS_AS(validatePayloadBudget(chunks, capacity, 0u), std::invalid_argument);
}

TEST_CASE("policy-derived pool defaults fit the payload budget", "[pool][budget][policy]")
{
    using namespace graphFlow;

    auto const cpuProperties = test::makeCpuDevice().getDeviceProperties();
    for(std::uint64_t const numParticles : {std::uint64_t{0u}, std::uint64_t{4097u}, std::uint64_t{1'000'000u}})
    {
        DevicePolicy const policy = makeDevicePolicy(cpuProperties, numParticles);
        std::uint64_t const chunks
            = particle::derivePoolSize(numParticles, policy.chunkCapacity, policy.liveParticles, particle::FlowBins);
        std::uint64_t const budget
            = particle::poolPayloadBudgetBytes(policy.memoryBudgetBytes, policy.chunkCapacity);

        INFO("N=" << numParticles << " C=" << policy.chunkCapacity << " P=" << policy.liveParticles);
        CHECK(requiredPayloadBytes(chunks, policy.chunkCapacity) <= budget);
        CHECK_NOTHROW(validatePayloadBudget(chunks, policy.chunkCapacity, budget));
    }

    // Accelerator-shaped public properties select the documented C=8192, P=16384 defaults. The
    // K*R reserve must be part of the budget, otherwise the pool would exceed the policy budget.
    auto properties = cpuProperties;
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 0x7FFFFFFFu;
    properties.multiProcessorCount = 108u;
    properties.globalMemCapacityBytes = 24ull * 1024ull * 1024ull * 1024ull;

    DevicePolicy const policy = makeDevicePolicy(properties, 100'000'000u);
    CHECK(policy.chunkCapacity == 8192u);
    CHECK(policy.liveParticles == 16384u);

    std::uint64_t const chunks = particle::derivePoolSize(
        100'000'000u,
        policy.chunkCapacity,
        policy.liveParticles,
        particle::FlowBins);
    std::uint64_t const budget
        = particle::poolPayloadBudgetBytes(policy.memoryBudgetBytes, policy.chunkCapacity);
    CHECK_NOTHROW(validatePayloadBudget(chunks, policy.chunkCapacity, budget));
    // The budget replaces the former ~12.9 GB default with the documented pool default.
    CHECK(requiredPayloadBytes(chunks, policy.chunkCapacity) < defaultPoolPayloadBudgetBytes);
}

TEST_CASE("A30 usable memory limits the default P to 16128", "[pool][budget][policy]")
{
    using namespace graphFlow;

    // The A30 reports a 1/8 budget of 3,171,090,432 B after ECC, just below the 3 GiB cap.
    // The documented C=8192, P=2*C=16384 pair needs 3,221,225,472 B and does not fit.
    std::uint64_t const usableBytes = 8ull * 3'171'090'432ull;
    CHECK(usableBytes / 8u == 3'171'090'432ull);

    auto properties = test::makeCpuDevice().getDeviceProperties();
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 0x7FFFFFFFu;
    properties.multiProcessorCount = 56u; // A30
    properties.globalMemCapacityBytes = usableBytes;

    DevicePolicy const policy = makeDevicePolicy(properties, 100'000'000u);
    CHECK(policy.memoryBudgetBytes == 3'171'090'432ull);
    CHECK(policy.chunkCapacity > 0u);
    CHECK(policy.chunkCapacity % policy.tileSize == 0u);
    CHECK(policy.liveParticles == 16128u);
    CHECK_FALSE(policy.liveParticlesExplicit);
    CHECK(requiredPayloadBytes(policy.liveParticles, policy.chunkCapacity) <= policy.memoryBudgetBytes);

    // An explicit documented C keeps the chunk size and clamps the auto-derived P instead.
    DevicePolicyOverrides chunkOnly;
    chunkOnly.chunkCapacity = 8192u;
    DevicePolicy const documented = makeDevicePolicy(properties, 100'000'000u, chunkOnly);
    CHECK(documented.chunkCapacity == 8192u);
    CHECK(documented.liveParticles == 16128u);
    CHECK_FALSE(documented.liveParticlesExplicit);
    CHECK(requiredPayloadBytes(documented.liveParticles, documented.chunkCapacity) <= documented.memoryBudgetBytes);

    // The former default pair is the exact rejection this clamp fixes.
    CHECK(requiredPayloadBytes(16384u, 8192u) == 3'221'225'472ull);
    CHECK_THROWS_AS(
        validatePayloadBudget(16384u, 8192u, policy.memoryBudgetBytes),
        std::invalid_argument);

    // applyDevicePolicy + derivePoolSize + Config::validate must accept the clamped default.
    Config config;
    applyDevicePolicy(config, policy);
    config.numParticles = 100'000'000u;
    config.numChunks = particle::derivePoolSize(
        config.numParticles,
        config.chunkCapacity,
        config.liveParticles,
        particle::FlowBins);
    CHECK_NOTHROW(config.validate());
    CHECK(requiredPayloadBytes(config.numChunks, config.chunkCapacity) <= config.poolPayloadBudgetBytes);

    // The CLI path adds the exact K*R(C) reserve instead of Config::poolOutputReserveChunks.
    std::uint64_t const cliBudget
        = particle::poolPayloadBudgetBytes(policy.memoryBudgetBytes, policy.chunkCapacity);
    CHECK_NOTHROW(validatePayloadBudget(config.numChunks, config.chunkCapacity, cliBudget));
}

TEST_CASE("explicit P over the A30 budget is rejected", "[pool][budget][policy]")
{
    using namespace graphFlow;

    auto properties = test::makeCpuDevice().getDeviceProperties();
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 0x7FFFFFFFu;
    properties.multiProcessorCount = 56u;
    properties.globalMemCapacityBytes = 8ull * 3'171'090'432ull;

    // The T13 acceptance preset used C=8192, P=16384 explicitly. On the real A30 that pair
    // exceeds the usable budget: explicit P is a user request and must fail, not be clamped.
    DevicePolicyOverrides overPair;
    overPair.chunkCapacity = 8192u;
    overPair.liveParticles = 16384u;
    CHECK_THROWS_WITH(
        makeDevicePolicy(properties, 100'000'000u, overPair),
        Catch::Matchers::ContainsSubstring("explicit liveParticles 16384")
            && Catch::Matchers::ContainsSubstring("largest feasible P is 16128"));

    DevicePolicyOverrides overOld;
    overOld.chunkCapacity = 8192u;
    overOld.liveParticles = 65536u;
    CHECK_THROWS_AS(
        makeDevicePolicy(properties, 100'000'000u, overOld),
        std::invalid_argument);

    // The largest feasible explicit P still passes and builds a valid pool.
    DevicePolicyOverrides fitting;
    fitting.chunkCapacity = 8192u;
    fitting.liveParticles = 16128u;
    DevicePolicy const policy = makeDevicePolicy(properties, 100'000'000u, fitting);
    CHECK(policy.chunkCapacity == 8192u);
    CHECK(policy.liveParticles == 16128u);
    CHECK(policy.liveParticlesExplicit);
    CHECK(requiredPayloadBytes(policy.liveParticles, policy.chunkCapacity) <= policy.memoryBudgetBytes);

    Config config;
    applyDevicePolicy(config, policy);
    config.numParticles = 100'000'000u;
    config.numChunks = particle::derivePoolSize(
        config.numParticles,
        config.chunkCapacity,
        config.liveParticles,
        particle::FlowBins);
    CHECK_NOTHROW(config.validate());
    CHECK(requiredPayloadBytes(config.numChunks, config.chunkCapacity) <= config.poolPayloadBudgetBytes);
}

TEST_CASE("applyDevicePolicy budgets the K*R reserve for the default pool", "[pool][budget][policy]")
{
    using namespace graphFlow;

    auto properties = test::makeCpuDevice().getDeviceProperties();
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 0x7FFFFFFFu;
    properties.multiProcessorCount = 108u;
    properties.globalMemCapacityBytes = 24ull * 1024ull * 1024ull * 1024ull;

    DevicePolicy const policy = makeDevicePolicy(properties, 100'000'000u);
    CHECK(policy.chunkCapacity == 8192u);
    CHECK(policy.liveParticles == 16384u);

    Config config;
    applyDevicePolicy(config, policy);
    config.numParticles = 100'000'000u;
    config.numChunks
        = particle::derivePoolSize(config.numParticles, config.chunkCapacity, config.liveParticles, particle::FlowBins);
    CHECK(config.poolPayloadBudgetBytes >= policy.memoryBudgetBytes);
    CHECK(
        config.poolPayloadBudgetBytes
        == payloadBudgetWithReserve(
            policy.memoryBudgetBytes,
            policy.chunkCapacity,
            config.poolOutputReserveChunks));
    CHECK_NOTHROW(config.validate());

    // A default Config that only copies the policy values also fits the documented default.
    Config direct;
    direct.chunkCapacity = policy.chunkCapacity;
    direct.liveParticles = policy.liveParticles;
    direct.numParticles = 100'000'000u;
    direct.numChunks
        = particle::derivePoolSize(direct.numParticles, direct.chunkCapacity, direct.liveParticles, particle::FlowBins);
    CHECK_NOTHROW(direct.validate());

    // The former C=8192, P=65536 pair still exceeds the default budget.
    Config old;
    old.chunkCapacity = 8192u;
    old.liveParticles = 65536u;
    old.numChunks = particle::derivePoolSize(
        100'000'000u,
        old.chunkCapacity,
        old.liveParticles,
        particle::FlowBins);
    CHECK_THROWS_AS(old.validate(), std::invalid_argument);
}

TEST_CASE("the former 12.9 GB default pool is rejected", "[pool][budget]")
{
    using namespace graphFlow;

    std::uint32_t const oldCapacity = 8192u;
    std::uint32_t const oldLiveParticles = 65536u;
    std::uint64_t const oldChunks
        = particle::derivePoolSize(100'000'000u, oldCapacity, oldLiveParticles, particle::FlowBins);
    CHECK(oldChunks == std::uint64_t{65542u});
    CHECK(requiredPayloadBytes(oldChunks, oldCapacity) > defaultPoolPayloadBudgetBytes);
    CHECK_THROWS_AS(
        validatePayloadBudget(oldChunks, oldCapacity, defaultPoolPayloadBudgetBytes),
        std::invalid_argument);

    Config config;
    config.chunkCapacity = oldCapacity;
    config.numChunks = static_cast<std::uint32_t>(oldChunks);
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);

    // Config::validate accepts the exact budget boundary and rejects one byte less.
    config.poolPayloadBudgetBytes = requiredPayloadBytes(oldChunks, oldCapacity);
    CHECK_NOTHROW(config.validate());
    config.poolPayloadBudgetBytes -= 1u;
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);
}

TEST_CASE("pool size keeps S=P+K*R for small P and N >> P", "[pool][budget]")
{
    using namespace graphFlow;

    std::uint64_t const perTransform = outputChunkBound(128u, 128u, particle::FlowBins);
    CHECK(perTransform == std::uint64_t{particle::FlowBins});

    // P < C and a small P: S is independent of N.
    CHECK(
        particle::derivePoolSize(4097u, 128u, 10u, particle::FlowBins)
        == std::uint64_t{10u} + particle::TransformNodes * perTransform);
    CHECK(
        particle::derivePoolSize(4097u, 128u, 256u, particle::FlowBins)
        == std::uint64_t{256u} + particle::TransformNodes * perTransform);
    CHECK(
        particle::derivePoolSize(1'000'000'000u, 128u, 256u, particle::FlowBins)
        == std::uint64_t{256u} + particle::TransformNodes * perTransform);

    // N=0 keeps the header-only one-chunk pool and stays inside the default budget.
    CHECK(particle::derivePoolSize(0u, 128u, 0u, particle::FlowBins) == std::uint64_t{1u});
    CHECK(
        requiredPayloadBytes(1u, 128u) == std::uint64_t{128u} * payloadBytesPerSlot);
    CHECK_NOTHROW(validatePayloadBudget(1u, 128u, defaultPoolPayloadBudgetBytes));
}

TEST_CASE("bounded P with Q>1 completes without a full batch", "[pool][budget][batching]")
{
    using namespace graphFlow;

    particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.liveParticles = 256u; // fragmented pool cannot always supply Q chunks
    parameters.batchChunks = 8u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    particle::FeedbackResult const result = particle::runParticleFeedback(test::makeCpuDevice(), parameters);
    test::ScalarOutcome const reference
        = test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

    CHECK(result.histogram == particle::histogramOf(reference.weights));
    CHECK(result.sunkParticles == parameters.numParticles);
    // Q must not enlarge the pool. `Flow` derives the reserve from the registered bins: the
    // filter has two outputs and each A/B partition has sink, feedback and capped, so the reserve
    // is 2 + 3 + 3 = 8 chunks.
    CHECK(result.stats.poolChunks == parameters.liveParticles + 8u);
}

TEST_CASE("pool budget arithmetic accepts a generic slot size", "[pool][budget][slot]")
{
    using namespace graphFlow;

    CHECK(requiredPayloadBytes(2u, 3u) == std::uint64_t{2u} * 3u * payloadBytesPerSlot);
    CHECK(requiredPayloadBytes(2u, 3u, 24u) == 144u);
    CHECK(requiredPayloadBytes(2u, 3u, 96u) == 576u);
    CHECK_THROWS_AS(requiredPayloadBytes(2u, 3u, 0u), std::invalid_argument);
    CHECK_THROWS_AS(
        requiredPayloadBytes(std::numeric_limits<std::uint64_t>::max() / 96u + 1u, 1u, 96u),
        std::overflow_error);

    CHECK_NOTHROW(validatePayloadBudget(2u, 3u, 144u, 24u));
    CHECK_NOTHROW(validatePayloadBudget(2u, 3u, 576u, 96u));
    CHECK_THROWS_AS(validatePayloadBudget(2u, 3u, 144u, 96u), std::invalid_argument);
    CHECK_NOTHROW(validatePayloadBudget(2u, 3u, 576u, "graphFlow::test", 96u));
    CHECK_THROWS_WITH(
        validatePayloadBudget(2u, 3u, 144u, "graphFlow::test", 96u),
        Catch::Matchers::ContainsSubstring("96 bytes"));

    CHECK(payloadBudgetWithReserve(1024u, 4u, 2u, 96u) == 1024u + std::uint64_t{2u} * 4u * 96u);
    CHECK(
        payloadBudgetWithReserve(1024u, 4u)
        == 1024u + defaultPoolOutputReserveChunks * 4u * payloadBytesPerSlot);
}

TEST_CASE("device policy honors the configured slot size", "[pool][budget][policy][slot]")
{
    using namespace graphFlow;

    auto properties = test::makeCpuDevice().getDeviceProperties();

    DevicePolicyConfig config;
    config.slotBytes = 96u;
    config.memoryBudgetBytes = 96ull * 1024ull * 1024ull;
    config.maxChunkCapacity = 4096u;

    DevicePolicy const policy = makeDevicePolicy(properties, 1'000'000u, {}, config);
    CHECK(policy.slotBytes == 96u);
    CHECK(requiredPayloadBytes(policy.liveParticles, policy.chunkCapacity, 96u) <= policy.memoryBudgetBytes);

    Config derived;
    derived.slotBytes = 96u;
    applyDevicePolicy(derived, policy);
    CHECK(derived.slotBytes == 96u);
    CHECK(
        derived.poolPayloadBudgetBytes
        == payloadBudgetWithReserve(
            policy.memoryBudgetBytes,
            policy.chunkCapacity,
            derived.poolOutputReserveChunks,
            96u));

    // The default stays the 24-byte particle SoA slot and produces the historical budget.
    DevicePolicy const particlePolicy = makeDevicePolicy(properties, 1'000'000u);
    CHECK(particlePolicy.slotBytes == payloadBytesPerSlot);
    Config particleConfig;
    applyDevicePolicy(particleConfig, particlePolicy);
    CHECK(particleConfig.slotBytes == payloadBytesPerSlot);
    CHECK(
        particleConfig.poolPayloadBudgetBytes
        == payloadBudgetWithReserve(
            particlePolicy.memoryBudgetBytes,
            particlePolicy.chunkCapacity,
            particleConfig.poolOutputReserveChunks));

    // Config::validate applies the configured slot size to its pool check.
    Config exact;
    exact.chunkCapacity = 4u;
    exact.numChunks = 2u;
    exact.slotBytes = 96u;
    exact.poolPayloadBudgetBytes = requiredPayloadBytes(2u, 4u, 96u);
    CHECK_NOTHROW(exact.validate());
    exact.poolPayloadBudgetBytes -= 1u;
    CHECK_THROWS_AS(exact.validate(), std::invalid_argument);
}

TEST_CASE("hash output credits use maxBins and the pool budget rejects an under-sized cap", "[pool][budget][hash]")
{
    using namespace graphFlow;

    // A hash partition may declare more bins than the dense shared table supports. Its output
    // credit is still outputChunkBound(C, C, maxBins): the hash table entry count, the tile
    // width, and Q do not enter it.
    constexpr std::uint32_t capacity = 64u;
    constexpr std::uint32_t maxBins = Config::denseBinLimit + 1u;
    constexpr std::uint64_t transformNodes = 3u;
    std::uint64_t const perTransform = checkedOutputChunkBound(capacity, capacity, maxBins, "test");
    CHECK(perTransform == capacity);
    CHECK(perTransform != HashBinTable<128u>::entries);
    CHECK(outputChunkBound(capacity, capacity, maxBins) == perTransform);

    std::uint64_t const reserve = transformNodes * perTransform;
    std::uint64_t const liveParticles = 256u;
    std::uint64_t const chunks = liveParticles + reserve;
    std::uint64_t const budget = payloadBudgetWithReserve(
        requiredPayloadBytes(liveParticles, capacity),
        capacity,
        reserve);

    CHECK_NOTHROW(validatePayloadBudget(chunks, capacity, budget));
    CHECK_THROWS_AS(validatePayloadBudget(chunks, capacity, budget - 1u), std::invalid_argument);
    CHECK_THROWS_WITH(
        validatePayloadBudget(chunks, capacity, budget - 1u, "graphFlow::test"),
        Catch::Matchers::ContainsSubstring("exceeding the payload budget"));

    // The relaxed Config maxBins check accepts the hash partition width and keeps the pool check.
    Config config;
    config.chunkCapacity = capacity;
    config.numChunks = static_cast<std::uint32_t>(chunks);
    config.maxBins = maxBins;
    config.poolPayloadBudgetBytes = budget;
    CHECK_NOTHROW(config.validate());
    config.poolPayloadBudgetBytes -= 1u;
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);

    // A credit that does not fit the 32-bit chunk index is rejected instead of wrapping.
    std::uint64_t const over32 = std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 10u;
    CHECK_THROWS_AS(checkedOutputChunkBound(over32, 1u, 1u, "test"), std::overflow_error);
}

TEST_CASE("hash transform reserve does not grow with Q", "[pool][budget][hash]")
{
    using namespace graphFlow;

    constexpr std::uint32_t capacity = 128u;
    constexpr std::uint32_t maxBins = 65536u;
    std::uint64_t const perTransform = checkedOutputChunkBound(capacity, capacity, maxBins, "test");
    CHECK(perTransform == capacity);

    // Q only changes the runtime prefix bound of one batch. The pool reserve per node stays
    // outputChunkBound(C, C, maxBins), so the derived pool size is Q-independent.
    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        CAPTURE(batchChunks);
        std::uint64_t const fullBatch = std::uint64_t{capacity} * batchChunks;
        CHECK(checkedOutputChunkBound(fullBatch, capacity, maxBins, "test") == fullBatch);
        CHECK(perTransform == capacity);
    }

    std::uint64_t const liveParticles = 1024u;
    std::uint64_t const reserve = 3u * perTransform;
    std::uint64_t const chunks = liveParticles + reserve;
    std::uint64_t const budget = payloadBudgetWithReserve(
        requiredPayloadBytes(liveParticles, capacity),
        capacity,
        reserve);
    CHECK_NOTHROW(validatePayloadBudget(chunks, capacity, budget));
}
