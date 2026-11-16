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

    struct WeightTag : graphFlow::TagBase
    {
    };

    struct VisitTag : graphFlow::TagBase
    {
    };

    using Id = graphFlow::Field<IdTag, std::uint64_t>;
    using Weight = graphFlow::Field<WeightTag, std::uint32_t>;
    using Visit = graphFlow::Field<VisitTag, std::uint32_t>;

    using SampleRecord = graphFlow::Record<Id, Weight>;
    using SampleStorage = graphFlow::SoAStorage<Device, Id, Weight>;
    using SampleFlow = graphFlow::Flow<SampleRecord, Device, SampleStorage>;
} // namespace

TEST_CASE("SoAFlow deduces the SoA storage from the record", "[soa][flow][static]")
{
    static_assert(std::same_as<graphFlow::SoAFlow<SampleRecord, Device>, SampleFlow>);
    static_assert(std::same_as<typename graphFlow::SoAFlow<SampleRecord, Device>::PayloadType, SampleRecord>);
    static_assert(std::same_as<typename graphFlow::SoAFlow<SampleRecord, Device>::Storage, SampleStorage>);
    static_assert(std::same_as<typename SampleStorage::record_type, SampleRecord>);
    static_assert(std::same_as<typename graphFlow::SoAFlow<SampleRecord, Device>::Storage::record_type, SampleRecord>);
}

TEST_CASE("SoAFlow constructs over a host device", "[soa][flow]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::SoAFlow<SampleRecord, Device> flow{device};
    CHECK_FALSE(flow.tuning().chunkCapacity.has_value());
}

TEST_CASE("a SoA storage must match the payload record field pack", "[soa][flow][static]")
{
    static_assert(graphFlow::detail::SoAStorageMatchesRecord<SampleRecord, SampleStorage>::value);
    static_assert(
        graphFlow::detail::SoAStorageMatchesRecord<SampleRecord, graphFlow::PayloadSlotStorage<SampleRecord, Device>>::
            value);
    static_assert(!graphFlow::detail::
                      SoAStorageMatchesRecord<SampleRecord, graphFlow::SoAStorage<Device, Id, Weight, Visit>>::value);
    static_assert(!graphFlow::detail::SoAStorageMatchesRecord<SampleRecord, graphFlow::SoAStorage<Device, Id>>::value);
    static_assert(
        !graphFlow::detail::SoAStorageMatchesRecord<SampleRecord, graphFlow::SoAStorage<Device, Weight, Id>>::value);
}
