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

    inline constexpr IdTag idTag{};
    inline constexpr PropertyTag propertyTag{};
    inline constexpr WeightTag weightTag{};
    inline constexpr VisitTag visitTag{};

    using Id = graphFlow::Field<IdTag, std::uint64_t>;
    using Property = graphFlow::Field<PropertyTag, std::uint32_t>;
    using Weight = graphFlow::Field<WeightTag, std::uint32_t>;
    using Visit = graphFlow::Field<VisitTag, std::uint32_t>;

    using ParticleRecord = graphFlow::Record<Id, Property, Weight, Visit>;
    using ParticleAttributes = graphFlow::AttributeList<Id, Property, Weight, Visit>;
    using ParticleColumns = graphFlow::SoAStorage<Device, Id, Property, Weight, Visit>;

    // Declared only for the negative lookup below. nvcc 12.5 crashes on the equivalent
    // elaborated-type-specifier form (`contains<struct UnknownTag>`).
    struct UnknownTag;

    struct Pair
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;
    };

    // The declaration helper expands to the tag and field pair used by the checks below.
    GRAPHFLOW_ATTRIBUTE(HelperId, std::uint64_t);
    GRAPHFLOW_ATTRIBUTE(HelperValue, std::uint32_t);
    GRAPHFLOW_TAG(SubscriptId, subscriptId);
    GRAPHFLOW_TAG(SubscriptValue);

    using HelperAttributes = graphFlow::Attributes<HelperIdField, HelperValueField>;
    using HelperRecord = graphFlow::Record<HelperIdField, HelperValueField>;

    using ParticleSubset = graphFlow::Set<IdTag, WeightTag>;
    using ParticleSubsetProxy = graphFlow::MutableRecordProxy<ParticleRecord, ParticleSubset>;

    struct BumpWeightSubset
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, ParticleSubsetProxy& proxy) const
        {
            proxy.template set<WeightTag>(proxy.template get<WeightTag>() + 1u);
        }
    };

    struct BumpWeightRecord
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, ParticleRecord& record) const
        {
            record.template set<WeightTag>(record.template get<WeightTag>() + 1u);
        }
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
    static_assert(!ParticleAttributes::contains<UnknownTag>);
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

TEST_CASE("record and proxy subscripts use constexpr tags for reads and writes", "[attributes][static]")
{
    static_assert(std::is_empty_v<decltype(idTag)>);
    static_assert(std::same_as<std::remove_cvref_t<decltype(idTag)>, IdTag>);
    static_assert(std::is_empty_v<SubscriptIdTag>);
    static_assert(std::same_as<std::remove_cvref_t<decltype(subscriptId)>, SubscriptIdTag>);
    static_assert(std::is_empty_v<SubscriptValueTag>);
    static_assert(std::same_as<std::remove_cvref_t<decltype(SubscriptValueTag_v)>, SubscriptValueTag>);

    ParticleRecord record{5u, 2u, 3u, 1u};

    static_assert(std::same_as<decltype(record[idTag]), std::uint64_t&>);
    static_assert(std::same_as<decltype(record[weightTag]), std::uint32_t&>);
    CHECK(&record[idTag] == &record.get<IdTag>());
    CHECK(record[idTag] == 5u);
    CHECK(record[propertyTag] == 2u);

    record[idTag] = 17u;
    record[weightTag] += 4u;
    CHECK(record.get<IdTag>() == 17u);
    CHECK(record.get<WeightTag>() == 7u);

    ParticleRecord const& constRecord = record;
    static_assert(std::same_as<decltype(constRecord[idTag]), std::uint64_t const&>);
    static_assert(std::same_as<decltype(constRecord[visitTag]), std::uint32_t const&>);
    CHECK(constRecord[idTag] == 17u);
    CHECK(constRecord[visitTag] == 1u);

    auto narrow = record.proxy<graphFlow::Set<IdTag, WeightTag>>();
    static_assert(std::same_as<decltype(narrow[idTag]), std::uint64_t&>);
    CHECK(narrow[idTag] == 17u);
    narrow[idTag] = 23u;
    narrow[weightTag] -= 2u;
    CHECK(record[propertyTag] == 2u);
    CHECK(record[visitTag] == 1u);

    auto constProxy = constRecord.proxy<graphFlow::Set<IdTag, WeightTag>>();
    static_assert(std::same_as<decltype(constProxy[idTag]), std::uint64_t const&>);
    CHECK(constProxy[idTag] == 23u);
    CHECK(constProxy[weightTag] == 5u);

    ParticleRecord copy = narrow;
    CHECK(copy[idTag] == 23u);
    CHECK(copy[propertyTag] == 2u);
    CHECK(copy[weightTag] == 5u);
    CHECK(copy[visitTag] == 1u);

    using SubscriptRecord = graphFlow::
        Record<graphFlow::Field<SubscriptIdTag, std::uint32_t>, graphFlow::Field<SubscriptValueTag, std::uint32_t>>;
    SubscriptRecord declared{4u, 6u};
    static_assert(std::same_as<decltype(declared[subscriptId]), std::uint32_t&>);
    CHECK(declared[subscriptId] == 4u);
    CHECK(declared[SubscriptValueTag_v] == 6u);
    declared[subscriptId] = 8u;
    declared[SubscriptValueTag_v] += 1u;
    CHECK(declared[subscriptId] == 8u);
    CHECK(declared[SubscriptValueTag_v] == 7u);

    constexpr ParticleRecord compileTimeRecord{1u, 2u, 3u, 4u};
    static_assert(compileTimeRecord[idTag] == 1u);
    static_assert(compileTimeRecord[visitTag] == 4u);
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

TEST_CASE("record proxies expose only their set and copy every attribute", "[attributes][static]")
{
    using Narrow = graphFlow::Set<IdTag>;
    using Wide = graphFlow::Set<PropertyTag, VisitTag>;

    ParticleRecord record{13u, 2u, 5u, 1u};

    auto narrow = record.proxy<Narrow>();
    CHECK(narrow.get<IdTag>() == 13u);
    narrow.set<IdTag>(21u);
    CHECK(record.get<IdTag>() == 21u);
    // The node subset does not hide the other attributes from the record.
    CHECK(record.get<PropertyTag>() == 2u);
    CHECK(record.get<WeightTag>() == 5u);
    CHECK(record.get<VisitTag>() == 1u);

    auto wide = record.proxy<Wide>();
    wide.set<PropertyTag>(7u);
    wide.set<VisitTag>(9u);
    CHECK(record.get<PropertyTag>() == 7u);
    CHECK(record.get<VisitTag>() == 9u);
    CHECK(record.get<WeightTag>() == 5u);

    // Copying a narrow proxy to a record copies every attribute, including the ones outside the
    // declared subset, so a stage that only names one attribute cannot drop the rest.
    ParticleRecord copy = narrow;
    CHECK(copy.get<IdTag>() == 21u);
    CHECK(copy.get<PropertyTag>() == 7u);
    CHECK(copy.get<WeightTag>() == 5u);
    CHECK(copy.get<VisitTag>() == 9u);

    // store() of a full record through a narrow proxy forwards the full record as well.
    ParticleRecord const replacement{3u, 4u, 6u, 8u};
    narrow.store(replacement);
    CHECK(record.get<IdTag>() == 3u);
    CHECK(record.get<PropertyTag>() == 4u);
    CHECK(record.get<WeightTag>() == 6u);
    CHECK(record.get<VisitTag>() == 8u);

    // A const record yields a read-only proxy; set() is unavailable on it.
    ParticleRecord const& constRecord = record;
    auto constProxy = constRecord.proxy<Narrow>();
    CHECK(constProxy.get<IdTag>() == 3u);
    static_assert(std::same_as<
                  typename decltype(constProxy)::access_type,
                  graphFlow::detail::ConstRefRecordAccess<ParticleRecord>>);
    static_assert(std::same_as<decltype(constProxy)::set_type, Narrow>);
}

TEST_CASE("attribute slot accounting matches the default payload storage", "[attributes][static]")
{
    using PairId = graphFlow::Field<IdTag, std::uint64_t>;
    using PairWeight = graphFlow::Field<WeightTag, std::uint32_t>;
    using PairAttributes = graphFlow::AttributeList<PairId, PairWeight>;
    using PairColumns = graphFlow::SoAStorage<Device, PairId, PairWeight>;

    // The slot is the natural AoS size of the attributes plus the 32-bit bin-id scratch word.
    static_assert(PairAttributes::attributeBytes == sizeof(std::uint64_t) + sizeof(std::uint32_t));
    static_assert(PairAttributes::scratchBytes == sizeof(std::uint32_t));
    static_assert(PairAttributes::slotBytes == PairAttributes::attributeBytes + PairAttributes::scratchBytes);
    static_assert(PairColumns::attributeBytes == PairAttributes::attributeBytes);
    static_assert(PairColumns::scratchBytes == PairAttributes::scratchBytes);
    static_assert(PairColumns::slotBytes == PairAttributes::slotBytes);

    // The default particle storage keeps the historical per-slot payload size, so the attribute
    // adapter is byte-identical to the pool it replaces.
    static_assert(graphFlow::ParticleStorage<Device>::slotBytes == ParticleAttributes::slotBytes);
    static_assert(ParticleAttributes::slotBytes == graphFlow::payloadBytesPerSlot);
    static_assert(graphFlow::PayloadSlotStorage<Pair, Device>::slotBytes == sizeof(Pair));

    auto device = graphFlow::test::makeCpuDevice();
    PairColumns storage{device, 2u, 3u};
    CHECK(storage.payloadBytes() == storage.slotCount() * PairColumns::slotBytes);
    CHECK(storage.slotCount() == 6u);

    // The scratch column covers every slot and stays independent of the attribute columns.
    storage.storeRecord(5u, graphFlow::Record<PairId, PairWeight>{11u, 12u});
    storage.binIds()[5u] = 2u;
    CHECK(storage.binIds()[5u] == 2u);
    CHECK(storage.column<IdTag>()[5u] == 11u);
    CHECK(storage.column<WeightTag>()[5u] == 12u);
}

TEST_CASE("attribute declaration helpers generate a tag and field pair", "[attributes][static]")
{
    static_assert(std::is_empty_v<HelperIdTag>);
    static_assert(std::derived_from<HelperIdTag, graphFlow::TagBase>);
    static_assert(std::same_as<HelperIdField, graphFlow::Field<HelperIdTag, std::uint64_t>>);
    static_assert(std::same_as<HelperValueField, graphFlow::Field<HelperValueTag, std::uint32_t>>);

    static_assert(std::same_as<HelperAttributes, graphFlow::AttributeList<HelperIdField, HelperValueField>>);
    static_assert(std::same_as<HelperAttributes, graphFlow::Attributes<HelperIdField, HelperValueField>>);
    static_assert(HelperAttributes::size == 2u);
    static_assert(HelperAttributes::template contains<HelperIdTag>);
    static_assert(!HelperAttributes::template contains<UnknownTag>);
    static_assert(HelperAttributes::template index<HelperValueTag> == 1u);
    static_assert(std::same_as<HelperAttributes::value_type<HelperIdTag>, std::uint64_t>);

    static_assert(HelperRecord::size == 2u);
    static_assert(std::same_as<typename HelperRecord::attribute_list, HelperAttributes>);

    HelperRecord record{9u, 4u};
    CHECK(record.get<HelperIdTag>() == 9u);
    CHECK(record.get<HelperValueTag>() == 4u);
    record.set<HelperValueTag>(7u);
    CHECK(record.get<HelperValueTag>() == 7u);
}

TEST_CASE("subset transform registration hands the functor a RecordProxy", "[attributes][static]")
{
    static_assert(graphFlow::AttributeRecord<ParticleRecord>);
    static_assert(graphFlow::AttributeSubset<ParticleRecord, ParticleSubset>);
    static_assert(!graphFlow::AttributeSubset<ParticleRecord, graphFlow::Set<HelperIdTag>>);
    static_assert(graphFlow::SubsetTransformFor<BumpWeightSubset, ParticleRecord, ParticleSubset>);
    static_assert(!graphFlow::SubsetTransformFor<BumpWeightRecord, ParticleRecord, ParticleSubset>);

    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::Flow<ParticleRecord, Device, ParticleColumns> flow{device};

    // The explicit Set selects the additive overload; the plain call keeps the Record overload.
    auto subset = flow.transform<ParticleSubset>("subset", BumpWeightSubset{});
    auto full = flow.transform("full", BumpWeightRecord{});

    static_assert(std::same_as<decltype(subset), graphFlow::TransformHandle>);
    static_assert(std::same_as<decltype(full), graphFlow::TransformHandle>);
    CHECK(subset.valid());
    CHECK(full.valid());
    CHECK(subset.name() == "subset");
    CHECK(full.name() == "full");
}
