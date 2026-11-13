/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkStorage.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/flow/Payload.hpp"
#include "graphFlow/flow/PayloadStorage.hpp"
#include "graphFlow/graph/Classifier.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    struct SlotPayload
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;
    };

    /** 96-byte payload used to pin the generic budget and slot-size arithmetic. */
    struct WidePayload
    {
        std::uint64_t words[12] = {};

        friend bool operator==(WidePayload const&, WidePayload const&) = default;
    };

    /** A non-particle payload must not select the particle SoA pool. */
    struct Photon
    {
        std::uint64_t id = 0u;
        std::uint32_t wavelength = 0u;
    };

    using SlotStorage = graphFlow::PayloadSlotStorage<SlotPayload, Device>;
} // namespace

TEST_CASE("payload-slot storage exposes the chunk geometry", "[payload][storage][slot]")
{
    static_assert(graphFlow::Payload<SlotPayload>);
    static_assert(graphFlow::ChunkStorage<SlotStorage>);
    static_assert(graphFlow::PayloadStorage<SlotStorage>);
    static_assert(SlotStorage::slotBytes == sizeof(SlotPayload));
    static_assert(graphFlow::PayloadSlotStorage<WidePayload, Device>::slotBytes == 96u);

    auto device = graphFlow::test::makeCpuDevice();
    SlotStorage storage{device, 3u, 4u};

    CHECK(storage.numChunks() == 3u);
    CHECK(storage.chunkCapacity() == 4u);
    CHECK(storage.freeCount() == 3u);
    CHECK(storage.liveCount() == 0u);
    CHECK(storage.peakLive() == 0u);
    CHECK(storage.peakLeases() == 0u);
    CHECK(storage.slotCount() == 12u);

    CHECK_THROWS_AS((SlotStorage{device, 0u, 4u}), std::invalid_argument);
    CHECK_THROWS_AS((SlotStorage{device, 4u, 0u}), std::invalid_argument);
}

TEST_CASE("payload-slot storage preserves acquire and release behavior", "[payload][storage][slot]")
{
    auto device = graphFlow::test::makeCpuDevice();
    SlotStorage storage{device, 3u, 4u};

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

    CHECK_THROWS_AS(storage.release(second), std::invalid_argument);
    CHECK_THROWS_AS(storage.validateLive(second), std::invalid_argument);
    CHECK_THROWS_AS(storage.release(graphFlow::ChunkHandle{7u, 0u}), std::invalid_argument);
    CHECK_THROWS_AS(storage.leaseCount(graphFlow::ChunkHandle{7u, 0u}), std::invalid_argument);
}

TEST_CASE("payload-slot storage enforces reader leases", "[payload][storage][slot][lease]")
{
    auto device = graphFlow::test::makeCpuDevice();
    SlotStorage storage{device, 1u, 1u};

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

TEST_CASE("payload-slot storage keeps records aligned in one AoS allocation", "[payload][storage][slot]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::PayloadSlotStorage<WidePayload, Device> storage{device, 2u, 3u};
    auto queue = device.makeQueue(alpaka::queueKind::blocking);

    auto const handle = storage.acquire();
    std::size_t const slots = static_cast<std::size_t>(storage.slotCount());
    std::vector<WidePayload> host(slots);
    for(std::size_t i = 0u; i < slots; ++i)
        host[i].words[0] = 1000u + i;

    alpaka::onHost::memcpy(queue, storage.payload(), host);

    std::vector<WidePayload> read(slots);
    alpaka::onHost::memcpy(queue, read, storage.payload());
    CHECK(read == host);

    std::size_t const base = static_cast<std::size_t>(storage.base(handle));
    for(std::size_t i = 0u; i < storage.chunkCapacity(); ++i)
        CHECK(read[base + i].words[0] == 1000u + base + i);

    auto const address = reinterpret_cast<std::uintptr_t>(storage.payload().data());
    CHECK(address % alignof(WidePayload) == 0u);
    CHECK((address + base * sizeof(WidePayload)) % alignof(WidePayload) == 0u);
}

TEST_CASE("default storage binds only the particle payload to the SoA pool", "[payload][storage][default]")
{
    static_assert(std::same_as<
                  graphFlow::DefaultStorage<SlotPayload, Device>,
                  graphFlow::PayloadSlotStorage<SlotPayload, Device>>);
    static_assert(
        std::same_as<graphFlow::DefaultStorage<Photon, Device>, graphFlow::PayloadSlotStorage<Photon, Device>>);
    static_assert(
        std::same_as<graphFlow::DefaultStorage<graphFlow::Particle, Device>, graphFlow::ParticleStorage<Device>>);
    static_assert(graphFlow::ParticleStorage<Device>::slotBytes == graphFlow::payloadBytesPerSlot);
    static_assert(sizeof(graphFlow::Particle) == graphFlow::payloadBytesPerSlot);
}
