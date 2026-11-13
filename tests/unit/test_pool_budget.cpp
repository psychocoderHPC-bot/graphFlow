/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/core/PoolBudget.hpp"
#include "graphFlow/core/Reservation.hpp"

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
    parameters.stallTimeoutMs = 10'000u;

    particle::FeedbackResult const result = particle::runParticleFeedback(test::makeCpuDevice(), parameters);
    test::ScalarOutcome const reference
        = test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

    CHECK(result.histogram == particle::histogramOf(reference.weights));
    CHECK(result.sunkParticles == parameters.numParticles);
    // T5 batching is integrated, so Q>1 is live. Q must not enlarge the pool: S stays P + K*R(C)
    // and a fragmented run shrinks the batch prefix instead of reserving Q times the credits.
    CHECK(
        result.stats.poolChunks
        == particle::derivePoolSize(
            parameters.numParticles,
            parameters.chunkSize,
            parameters.liveParticles,
            particle::FlowBins));
}
