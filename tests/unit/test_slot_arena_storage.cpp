/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkStorage.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/core/PoolBudget.hpp"
#include "graphFlow/core/PoolMode.hpp"
#include "graphFlow/core/SlotAllocator.hpp"
#include "graphFlow/flow/Payload.hpp"
#include "graphFlow/flow/PayloadStorage.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    struct ArenaPayload
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;

        friend bool operator==(ArenaPayload const&, ArenaPayload const&) = default;
    };

    using ArenaStorage = graphFlow::PayloadSlotStorage<ArenaPayload, Device>;

    GRAPHFLOW_TAG(ArenaId, arenaIdTag);
    GRAPHFLOW_TAG(ArenaValue, arenaValueTag);

    using ArenaIdField = graphFlow::Field<ArenaIdTag, std::uint64_t>;
    using ArenaValueField = graphFlow::Field<ArenaValueTag, std::uint32_t>;
    using ArenaRecord = graphFlow::Record<ArenaIdField, ArenaValueField>;
    using ArenaSoAStorage = graphFlow::SoAStorage<Device, ArenaIdField, ArenaValueField>;
} // namespace

TEST_CASE("slot arena payload storage round-trips payloads at the stored base", "[slot][arena][storage]")
{
    static_assert(graphFlow::ChunkStorage<ArenaStorage>);
    static_assert(graphFlow::PayloadStorage<ArenaStorage>);

    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::SlotArenaOptions options;
    options.arenaSlots = 8u;
    ArenaStorage storage{device, 3u, 4u, options};

    CHECK(storage.poolMode() == graphFlow::PoolMode::slotArena);
    CHECK(storage.numChunks() == 3u);
    CHECK(storage.chunkCapacity() == 4u);
    CHECK(storage.slotCount() == 8u);
    CHECK(storage.payloadBytes() == 8u * ArenaStorage::slotBytes);
    CHECK(storage.freeCount() == 3u);

    // A one-particle fragment owns one slot, so the second base is `base + 1`, not `base + C`.
    auto const first = storage.acquire(1u);
    auto const second = storage.acquire(1u);
    auto const third = storage.acquire(2u);
    CHECK(storage.fill(first) == 1u);
    CHECK(storage.fill(second) == 1u);
    CHECK(storage.fill(third) == 2u);
    CHECK(storage.base(first) == 0u);
    CHECK(storage.base(second) == 1u);
    CHECK(storage.base(third) == 2u);
    CHECK_NOTHROW(storage.verifyArena());

    // The mapped host base table publishes the same rows the map owns.
    CHECK(storage.baseTableHost().data()[first.index] == static_cast<std::uint32_t>(storage.base(first)));
    CHECK(storage.baseTableHost().data()[second.index] == static_cast<std::uint32_t>(storage.base(second)));
    CHECK(storage.baseTableHost().data()[third.index] == static_cast<std::uint32_t>(storage.base(third)));
    CHECK(storage.fillTableHost().data()[first.index] == 1u);
    CHECK(storage.fillTableHost().data()[third.index] == 2u);

    std::size_t const slots = static_cast<std::size_t>(storage.slotCount());
    std::vector<ArenaPayload> host(slots);
    for(std::size_t i = 0u; i < slots; ++i)
    {
        host[i].id = 9000u + i;
        host[i].weight = 7u;
    }
    // Write one record per owned slot; the arena layout decides where each chunk lives.
    for(auto const& handle : {first, second, third})
    {
        std::size_t const base = static_cast<std::size_t>(storage.base(handle));
        for(std::uint32_t row = 0u; row < storage.fill(handle); ++row)
            host[base + row] = ArenaPayload{100u + base + row, 1u};
    }
    auto queue = device.makeQueue(alpaka::queueKind::blocking);
    alpaka::onHost::memcpy(queue, storage.payload(), host);

    std::vector<ArenaPayload> read(slots);
    alpaka::onHost::memcpy(queue, read, storage.payload());
    CHECK(read[0] == ArenaPayload{100u, 1u});
    CHECK(read[1] == ArenaPayload{101u, 1u});
    CHECK(read[2] == ArenaPayload{102u, 1u});
    CHECK(read[3] == ArenaPayload{103u, 1u});
    // Slots no chunk owns keep their previous content.
    CHECK(read[7].id == 9007u);

    // Release returns the ranges; a later acquire reuses them in first-fit order.
    storage.release(second);
    storage.release(first);
    storage.release(third);
    CHECK_NOTHROW(storage.verifyArenaReleased());

    auto const refilled = storage.acquire(4u);
    CHECK(storage.fill(refilled) == 4u);
    CHECK(storage.base(refilled) == 0u);
    CHECK_THROWS_AS(storage.acquire(5u), std::invalid_argument);
    storage.release(refilled);
    CHECK_NOTHROW(storage.verifyArenaReleased());
}

TEST_CASE("slot arena payload storage rejects stale handles and detects leaks", "[slot][arena][storage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::SlotArenaOptions options;
    options.arenaSlots = 4u;
    ArenaStorage storage{device, 2u, 4u, options};

    auto const first = storage.acquire(1u);
    auto const second = storage.acquire(3u);
    CHECK(storage.base(first) == 0u);
    CHECK(storage.base(second) == 1u);

    // A stored base is only valid for the live generation of its index.
    storage.release(first);
    CHECK_THROWS_AS(storage.base(first), std::invalid_argument);
    CHECK_THROWS_AS(storage.fill(first), std::invalid_argument);
    CHECK_THROWS_AS(storage.release(first), std::invalid_argument);

    // The range of the released index is free again, so a 4-slot request fits at slot 0.
    storage.release(second);
    auto const whole = storage.acquire(4u);
    CHECK(storage.base(whole) == 0u);
    CHECK_THROWS_AS(storage.verifyArenaReleased(), std::logic_error);
    storage.release(whole);
    CHECK_NOTHROW(storage.verifyArenaReleased());

    // Reader leases still block a release, and the range survives the rejected release.
    auto const leased = storage.acquire(1u);
    storage.acquireLease(leased);
    CHECK_THROWS_AS(storage.release(leased), std::logic_error);
    CHECK(storage.base(leased) == 0u);
    storage.releaseLease(leased);
    storage.release(leased);
    CHECK_NOTHROW(storage.verifyArenaReleased());

    // Exhaustion is a loud error, and a failed acquire does not lose the chunk index.
    auto const only = storage.acquire(1u);
    auto const last = storage.acquire(1u);
    CHECK_THROWS_AS(storage.acquire(1u), std::runtime_error);
    CHECK(storage.freeCount() == 0u);
    storage.release(only);
    storage.release(last);
    CHECK_NOTHROW(storage.verifyArenaReleased());
}

TEST_CASE("slot arena SoA storage round-trips records at the stored base", "[slot][arena][storage]")
{
    static_assert(graphFlow::ChunkStorage<ArenaSoAStorage>);

    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::SlotArenaOptions options;
    options.arenaSlots = 6u;
    ArenaSoAStorage storage{device, 2u, 4u, options};

    CHECK(storage.poolMode() == graphFlow::PoolMode::slotArena);
    CHECK(storage.slotCount() == 6u);

    auto const handle = storage.acquire(2u);
    CHECK(storage.fill(handle) == 2u);
    CHECK(storage.base(handle) == 0u);
    CHECK(storage.slotIndex(handle, 0u) == 0u);
    CHECK(storage.slotIndex(handle, 1u) == 1u);
    // The row bound is the committed fill, not the chunk capacity.
    CHECK_THROWS_AS(storage.slotIndex(handle, 2u), std::invalid_argument);

    ArenaRecord record{};
    record[arenaIdTag] = 42u;
    record[arenaValueTag] = 5u;
    storage.storeRecord(0u, record);
    record[arenaIdTag] = 43u;
    record[arenaValueTag] = 6u;
    storage.storeRecord(1u, record);
    CHECK(storage.loadRecord(0u)[arenaIdTag] == 42u);
    CHECK(storage.loadRecord(1u)[arenaValueTag] == 6u);
    CHECK_NOTHROW(storage.verifyArena());

    CHECK(storage.baseTableHost().data()[handle.index] == 0u);
    CHECK(storage.fillTableHost().data()[handle.index] == 2u);
    storage.release(handle);
    CHECK(storage.baseTableHost().data()[handle.index] == 0xFFFF'FFFFu);
    CHECK(storage.fillTableHost().data()[handle.index] == 0u);
    auto const reused = storage.acquire(2u);
    CHECK(storage.base(reused) == 0u);
    CHECK(storage.loadRecord(0u)[arenaIdTag] == 42u);
    storage.release(reused);
    CHECK_NOTHROW(storage.verifyArenaReleased());

    // The fixed constructors of both storages keep `index * chunkCapacity` unchanged.
    ArenaSoAStorage fixedSoA{device, 2u, 4u};
    CHECK(fixedSoA.poolMode() == graphFlow::PoolMode::fixed);
    auto const fixedHandle = fixedSoA.acquire();
    CHECK(fixedSoA.base(fixedHandle) == static_cast<std::uint64_t>(fixedHandle.index) * 4u);
    // The fixed layout publishes `index * chunkCapacity` rows so the kernels have one table path.
    for(std::uint32_t index = 0u; index < fixedSoA.numChunks(); ++index)
    {
        CHECK(fixedSoA.baseTableHost().data()[index] == index * 4u);
        CHECK(fixedSoA.fillTableHost().data()[index] == 4u);
    }
    CHECK(fixedSoA.fill(fixedHandle) == 4u);
    CHECK(fixedSoA.slotIndex(fixedHandle, 3u) == fixedSoA.base(fixedHandle) + 3u);
    CHECK_THROWS_AS(fixedSoA.slotIndex(fixedHandle, 4u), std::invalid_argument);
    fixedSoA.release(fixedHandle);
}

TEST_CASE("slot arena budget helpers invert the fixed-pool reservation", "[slot][arena][budget]")
{
    using namespace graphFlow;

    // 3 chunks of 8 slots and two output-credit chunks: fixed 3*8*24, arena 3 + 2*8 slots.
    constexpr std::uint64_t chunks = 3u;
    constexpr std::uint64_t capacity = 8u;
    constexpr std::uint64_t reserve = 2u;
    CHECK(requiredArenaSlots(chunks, capacity, reserve) == 19u);
    CHECK(requiredArenaBytes(chunks, capacity, reserve) == 19u * payloadBytesPerSlot);
    CHECK(requiredArenaBytes(chunks, capacity, reserve) < requiredPayloadBytes(chunks + reserve, capacity));
    CHECK(requiredArenaBytes(0u, capacity, 4u) == 4u * capacity * payloadBytesPerSlot);

    CHECK(maxPayloadSlots(10u * payloadBytesPerSlot) == 10u);
    CHECK(maxPayloadSlots(10u * payloadBytesPerSlot + payloadBytesPerSlot - 1u) == 10u);
    CHECK_THROWS_AS(maxPayloadSlots(0u), std::invalid_argument);
    CHECK_THROWS_AS(maxPayloadSlots(8u, 0u), std::invalid_argument);

    std::uint64_t const budget = 24u * payloadBytesPerSlot;
    CHECK(maxLiveParticlesForArena(budget, capacity, reserve) == 8u);
    CHECK(maxLiveParticlesForArena(budget, capacity, 100u) == 0u);

    CHECK_NOTHROW(validateSlotArenaBudget(8u, capacity, reserve, 24u * payloadBytesPerSlot, "arena"));
    CHECK_THROWS_AS(
        validateSlotArenaBudget(9u, capacity, reserve, 24u * payloadBytesPerSlot, "arena"),
        std::invalid_argument);
    CHECK_THROWS_AS(validateSlotArenaBudget(1u, capacity, reserve, 0u, "arena"), std::invalid_argument);

    CHECK(checkedArenaSlotCount(1u, "arena") == 1u);
    CHECK_THROWS_AS(checkedArenaSlotCount(0u, "arena"), std::invalid_argument);
    constexpr std::uint64_t tooWide = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1u;
    CHECK_THROWS_AS(checkedArenaSlotCount(tooWide, "arena"), std::overflow_error);

    // The arena storage derives its size from the budget when no explicit slot count is given.
    auto device = test::makeCpuDevice();
    SlotArenaOptions derived;
    derived.payloadBudgetBytes = 6u * ArenaStorage::slotBytes;
    ArenaStorage storage{device, 2u, 4u, derived};
    CHECK(storage.slotCount() == 6u);
    CHECK_THROWS_AS((ArenaStorage{device, 7u, 4u, derived}), std::invalid_argument);
}

TEST_CASE("config validates the slot-arena reservation and keeps the fixed default", "[slot][arena][config]")
{
    using namespace graphFlow;

    Config config;
    config.chunkCapacity = 4u;
    config.numChunks = 4u;
    CHECK(config.poolMode == PoolMode::fixed);
    CHECK_NOTHROW(config.validate());
    CHECK(std::string{poolModeName(PoolMode::slotArena)} == "slotArena");
    CHECK(std::string{poolModeName(PoolMode::fixed)} == "fixed");

    // P=2, C=4, K*R=2: the reservation is 2 + 2*4 = 10 slots, not (2+2)*4 = 16.
    config.poolMode = PoolMode::slotArena;
    config.liveParticles = 2u;
    config.poolOutputReserveChunks = 2u;
    config.numChunks = 4u;
    config.slotArenaSlots = requiredArenaSlots(2u, 4u, 2u);
    config.poolPayloadBudgetBytes = requiredArenaBytes(2u, 4u, 2u);
    CHECK(config.slotArenaSlots == 10u);
    CHECK_NOTHROW(config.validate());

    // One slot short of the P + K*R*C reservation is rejected.
    config.slotArenaSlots = 9u;
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);

    // One byte short of the budget is rejected as well.
    config.slotArenaSlots = 10u;
    config.poolPayloadBudgetBytes = requiredArenaBytes(2u, 4u, 2u) - 1u;
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);

    // More chunk indices than the arena has slots cannot be acquired one slot each.
    config.poolPayloadBudgetBytes = requiredArenaBytes(2u, 4u, 2u);
    config.numChunks = 11u;
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);
}
