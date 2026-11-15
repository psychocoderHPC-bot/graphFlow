/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/graphFlow.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    struct IdTag : graphFlow::TagBase
    {
    };

    struct PropertyTag : graphFlow::TagBase
    {
    };

    struct WeightTag : graphFlow::TagBase
    {
    };

    struct VisitTag : graphFlow::TagBase
    {
    };

    using Id = graphFlow::Field<IdTag, std::uint64_t>;
    using Property = graphFlow::Field<PropertyTag, std::uint32_t>;
    using Weight = graphFlow::Field<WeightTag, std::uint32_t>;
    using Visit = graphFlow::Field<VisitTag, std::uint32_t>;

    using ParticleRecord = graphFlow::Record<Id, Property, Weight, Visit>;
    using ParticleAttributes = graphFlow::AttributeList<Id, Property, Weight, Visit>;
    using ParticleColumns = graphFlow::SoAStorage<Device, Id, Property, Weight, Visit>;

    struct Pair
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;
    };
} // namespace

TEST_CASE("attribute records expose named fields and copy all attributes", "[attributes][static]")
{
    static_assert(std::is_empty_v<IdTag>);
    static_assert(std::derived_from<IdTag, llama_lite::TagBase>);
    static_assert(std::same_as<Id, llama_lite::Field<IdTag, std::uint64_t>>);
    static_assert(std::is_trivially_copyable_v<Id>);
    static_assert(std::is_trivially_copyable_v<ParticleRecord>);
    static_assert(std::same_as<typename ParticleRecord::record_type, llama_lite::Record<Id, Property, Weight, Visit>>);
    static_assert(
        std::same_as<typename ParticleAttributes::record_type, llama_lite::Record<Id, Property, Weight, Visit>>);
    static_assert(ParticleAttributes::size == 4u);
    static_assert(ParticleAttributes::contains<WeightTag>);
    static_assert(!ParticleAttributes::contains<struct UnknownTag>);
    static_assert(ParticleAttributes::template index<VisitTag> == 3u);
    static_assert(graphFlow::Set<IdTag, WeightTag>::contains<WeightTag>);
    static_assert(!graphFlow::Set<IdTag>::contains<WeightTag>);

    static_assert(ParticleAttributes::attributeBytes == 20u);
    static_assert(ParticleAttributes::slotBytes == 24u);
    static_assert(ParticleColumns::slotBytes == ParticleAttributes::slotBytes);

    ParticleRecord record{};
    record.set<IdTag>(7u);
    record.set<WeightTag>(3u);
    CHECK(record.get<IdTag>() == 7u);
    CHECK(record.get<WeightTag>() == 3u);

    auto narrow = record.proxy<graphFlow::Set<IdTag>>();
    static_assert(std::same_as<decltype(narrow)::set_type, graphFlow::Set<IdTag>>);
    CHECK(narrow.get<IdTag>() == 7u);
    narrow.set<IdTag>(11u);
    CHECK(record.get<IdTag>() == 11u);

    auto full = record.proxy();
    full.set<VisitTag>(1u);

    ParticleRecord copy = narrow;
    CHECK(copy.get<IdTag>() == 11u);
    CHECK(copy.get<WeightTag>() == 3u);
    CHECK(copy.get<VisitTag>() == 1u);

    static_assert(std::same_as<
                  typename graphFlow::DefaultStorage<graphFlow::Particle, Device>,
                  graphFlow::ParticleStorage<Device>>);
    static_assert(
        std::same_as<typename graphFlow::DefaultStorage<Pair, Device>, graphFlow::PayloadSlotStorage<Pair, Device>>);
    static_assert(graphFlow::ParticleStorage<Device>::slotBytes == graphFlow::payloadBytesPerSlot);
    static_assert(graphFlow::PayloadSlotStorage<Pair, Device>::slotBytes == sizeof(Pair));
}

TEST_CASE("SoA attribute storage keeps one column per attribute and row slots", "[attributes][storage]")
{
    static_assert(graphFlow::PayloadStorage<ParticleColumns>);
    static_assert(graphFlow::ChunkStorage<ParticleColumns>);
    static_assert(ParticleColumns::attributeBytes == 20u);
    static_assert(ParticleColumns::scratchBytes == sizeof(std::uint32_t));
    static_assert(ParticleColumns::slotBytes == 24u);

    auto device = graphFlow::test::makeCpuDevice();
    ParticleColumns storage{device, 2u, 3u};

    CHECK(storage.numChunks() == 2u);
    CHECK(storage.chunkCapacity() == 3u);
    CHECK(storage.slotCount() == 6u);
    CHECK(storage.payloadBytes() == 6u * ParticleColumns::slotBytes);
    CHECK_THROWS_AS((ParticleColumns{device, 0u, 3u}), std::invalid_argument);
    CHECK_THROWS_AS((ParticleColumns{device, 2u, 0u}), std::invalid_argument);

    auto const handle = storage.acquire();
    auto const base = storage.base(handle);
    CHECK(base == static_cast<std::uint64_t>(handle.index) * 3u);
    CHECK(storage.slotIndex(handle, 2u) == base + 2u);
    CHECK_THROWS_AS(storage.slotIndex(handle, 3u), std::invalid_argument);

    auto initial = storage.proxy<typename ParticleColumns::attribute_set>(static_cast<std::uint32_t>(base));
    initial.set<IdTag>(41u);
    initial.set<PropertyTag>(2u);
    initial.set<WeightTag>(5u);
    initial.set<VisitTag>(6u);

    auto row = storage.proxy<graphFlow::Set<IdTag, WeightTag>>(static_cast<std::uint32_t>(base));
    row.set<IdTag>(42u);
    row.set<WeightTag>(7u);
    CHECK(storage.column<IdTag>()[base] == 42u);
    CHECK(storage.column<WeightTag>()[base] == 7u);
    CHECK(row.get<IdTag>() == 42u);

    ParticleRecord loaded = row;
    CHECK(loaded.get<IdTag>() == 42u);
    CHECK(loaded.get<PropertyTag>() == 2u);
    CHECK(loaded.get<WeightTag>() == 7u);
    CHECK(loaded.get<VisitTag>() == 6u);

    ParticleRecord const scattered{9u, 2u, 4u, 6u};
    storage.storeRecord(static_cast<std::uint32_t>(base + 1u), scattered);
    row.store(scattered);
    CHECK(storage.column<IdTag>()[base] == 9u);
    CHECK(storage.column<VisitTag>()[base] == 6u);
    auto const gathered = storage.loadRecord(static_cast<std::uint32_t>(base + 1u));
    CHECK(gathered.get<IdTag>() == 9u);
    CHECK(gathered.get<PropertyTag>() == 2u);
    CHECK(gathered.get<WeightTag>() == 4u);
    CHECK(gathered.get<VisitTag>() == 6u);

    storage.binIds()[base] = 3u;
    CHECK(storage.binIds()[base] == 3u);

    CHECK_THROWS_AS(
        storage.proxy<graphFlow::Set<IdTag>>(static_cast<std::uint32_t>(storage.slotCount())),
        std::invalid_argument);
}
