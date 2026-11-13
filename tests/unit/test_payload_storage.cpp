/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/flow/Payload.hpp"
#include "graphFlow/flow/PayloadStorage.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    struct SamplePayload
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;
    };

    std::vector<std::uint32_t> patternU32(std::size_t size, std::uint32_t seed)
    {
        std::vector<std::uint32_t> values(size);
        for(std::size_t i = 0u; i < size; ++i)
            values[i] = seed + static_cast<std::uint32_t>(i);
        return values;
    }
} // namespace

TEST_CASE("particle storage exposes the chunk-pool geometry", "[payload][storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ParticleStorage storage{device, 3u, 4u};

    CHECK(storage.numChunks() == 3u);
    CHECK(storage.chunkCapacity() == 4u);
    CHECK(storage.freeCount() == 3u);
    CHECK(storage.liveCount() == 0u);
    CHECK(storage.peakLive() == 0u);
    CHECK(storage.peakLeases() == 0u);

    CHECK_THROWS_AS((graphFlow::ParticleStorage{device, 0u, 4u}), std::invalid_argument);
    CHECK_THROWS_AS((graphFlow::ParticleStorage{device, 4u, 0u}), std::invalid_argument);
}

TEST_CASE("particle storage preserves acquire and release behavior", "[payload][storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ParticleStorage storage{device, 3u, 4u};

    auto const first = storage.acquire();
    auto const second = storage.acquire();
    auto const third = storage.acquire();

    CHECK(storage.freeCount() == 0u);
    CHECK(storage.liveCount() == 3u);
    CHECK(storage.peakLive() == 3u);
    CHECK(first.generation == 0u);
    CHECK(storage.isLive(first));
    CHECK(storage.base(first) == static_cast<std::uint64_t>(first.index) * 4u);

    std::vector<std::uint32_t> indices{first.index, second.index, third.index};
    std::sort(indices.begin(), indices.end());
    CHECK(indices == std::vector<std::uint32_t>{0u, 1u, 2u});

    CHECK_THROWS_AS(storage.acquire(), std::runtime_error);

    storage.release(second);
    CHECK(storage.liveCount() == 2u);
    CHECK_FALSE(storage.isLive(second));
    CHECK(storage.generation(second.index) == 1u);

    auto const reused = storage.acquire();
    CHECK(reused.index == second.index);
    CHECK(reused.generation == 1u);
    CHECK(storage.isLive(reused));
}

TEST_CASE("particle storage rejects stale and out-of-range handles", "[payload][storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ParticleStorage storage{device, 2u, 1u};

    auto const handle = storage.acquire();
    storage.release(handle);

    CHECK_FALSE(storage.isLive(handle));
    CHECK_THROWS_AS(storage.release(handle), std::invalid_argument);
    CHECK_THROWS_AS(storage.validateLive(handle), std::invalid_argument);
    CHECK_THROWS_AS(storage.release(graphFlow::ChunkHandle{7u, 0u}), std::invalid_argument);
    CHECK_THROWS_AS(storage.leaseCount(graphFlow::ChunkHandle{7u, 0u}), std::invalid_argument);

    auto const reused = storage.acquire();
    CHECK(reused.index == handle.index);
    CHECK(reused.generation == handle.generation + 1u);
    storage.release(reused);
}

TEST_CASE("particle storage enforces reader leases", "[payload][storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ParticleStorage storage{device, 1u, 1u};

    auto const handle = storage.acquire();
    storage.acquireLease(handle);
    storage.acquireLease(handle);

    CHECK(storage.leaseCount(handle) == 2u);
    CHECK(storage.peakLeases() == 2u);
    CHECK_THROWS_AS(storage.release(handle), std::logic_error);

    storage.releaseLease(handle);
    CHECK(storage.leaseCount(handle) == 1u);
    CHECK_THROWS_AS(storage.release(handle), std::logic_error);

    storage.releaseLease(handle);
    CHECK(storage.leaseCount(handle) == 0u);
    CHECK_NOTHROW(storage.release(handle));
    CHECK_FALSE(storage.isLive(handle));
    CHECK_THROWS_AS(storage.releaseLease(handle), std::invalid_argument);
}

TEST_CASE("particle storage forwards the particle SoA arrays", "[payload][storage]")
{
    static_assert(graphFlow::Payload<SamplePayload>);
    static_assert(!graphFlow::Payload<std::string>);
    static_assert(graphFlow::PayloadStorage<graphFlow::ParticleStorage<Device>>);
    static_assert(!graphFlow::PayloadStorage<graphFlow::ChunkPool<Device>>);

    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ParticleStorage storage{device, 2u, 3u};
    auto queue = device.makeQueue(alpaka::queueKind::blocking);

    std::size_t const slots = static_cast<std::size_t>(storage.numChunks()) * storage.chunkCapacity();
    std::vector<std::uint64_t> ids(slots);
    for(std::size_t i = 0u; i < slots; ++i)
        ids[i] = 10000u + static_cast<std::uint64_t>(i);

    std::vector<std::uint32_t> properties = patternU32(slots, 100u);
    std::vector<std::uint32_t> weights = patternU32(slots, 200u);
    std::vector<std::uint32_t> visits = patternU32(slots, 300u);
    std::vector<std::uint32_t> binIds = patternU32(slots, 400u);

    alpaka::onHost::memcpy(queue, storage.ids(), ids);
    alpaka::onHost::memcpy(queue, storage.properties(), properties);
    alpaka::onHost::memcpy(queue, storage.weights(), weights);
    alpaka::onHost::memcpy(queue, storage.visits(), visits);
    alpaka::onHost::memcpy(queue, storage.binIds(), binIds);

    std::vector<std::uint64_t> readIds(slots);
    std::vector<std::uint32_t> readProperties(slots);
    std::vector<std::uint32_t> readWeights(slots);
    std::vector<std::uint32_t> readVisits(slots);
    std::vector<std::uint32_t> readBinIds(slots);
    alpaka::onHost::memcpy(queue, readIds, storage.pool().ids());
    alpaka::onHost::memcpy(queue, readProperties, storage.pool().properties());
    alpaka::onHost::memcpy(queue, readWeights, storage.pool().weights());
    alpaka::onHost::memcpy(queue, readVisits, storage.pool().visits());
    alpaka::onHost::memcpy(queue, readBinIds, storage.pool().binIds());

    CHECK(readIds == ids);
    CHECK(readProperties == properties);
    CHECK(readWeights == weights);
    CHECK(readVisits == visits);
    CHECK(readBinIds == binIds);

    // The mutable accessors and the wrapped pool address the same storage.
    std::vector<std::uint32_t> weightUpdate = patternU32(slots, 500u);
    alpaka::onHost::memcpy(queue, storage.pool().weights(), weightUpdate);
    alpaka::onHost::memcpy(queue, readWeights, storage.weights());
    CHECK(readWeights == weightUpdate);
}
