/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/flow/Status.hpp"

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

    struct Pod
    {
        std::uint64_t id = 0u;
        std::uint32_t value = 0u;
    };

    using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;

    struct GeneratePod
    {
        ALPAKA_FN_ACC Pod operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Pod{id, 0u};
        }
    };

    struct FailSource
    {
        ALPAKA_FN_ACC Pod operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            std::uint64_t id) const
        {
            status.set(graphFlow::StatusCode::invalid);
            return Pod{id, 0u};
        }
    };

    struct FailSink
    {
        ALPAKA_FN_ACC void operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            [[maybe_unused]] Pod const& payload) const
        {
            status.set(graphFlow::StatusCode::overflow);
        }
    };

    struct FailTransform
    {
        ALPAKA_FN_ACC void operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            [[maybe_unused]] Pod& payload) const
        {
            status.set(graphFlow::StatusCode::invalid);
        }
    };

    struct StatusClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            [[maybe_unused]] Pod const& payload) const
        {
            status.set(graphFlow::StatusCode::invalid);
            return 0u;
        }
    };

    struct ValidClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            [[maybe_unused]] Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % 2u);
        }
    };

    struct InvalidUpdate
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            [[maybe_unused]] std::uint32_t bin,
            [[maybe_unused]] Pod& payload) const
        {
            status.set(graphFlow::StatusCode::invalid);
            return graphFlow::UpdateResult::invalid;
        }
    };

    struct BusyUpdate
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            [[maybe_unused]] std::uint32_t bin,
            Pod& payload) const
        {
            ++payload.value;
            return graphFlow::UpdateResult::success;
        }
    };

    struct OverflowUpdate
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            [[maybe_unused]] std::uint32_t bin,
            [[maybe_unused]] Pod& payload) const
        {
            status.set(graphFlow::StatusCode::overflow);
            return graphFlow::UpdateResult::overflow;
        }
    };

    struct GeometryPayload
    {
        double z = 0.0;
        double dz = 0.0;
    };

    struct ValidateGeometry
    {
        ALPAKA_FN_ACC void operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            GeometryPayload const& payload) const
        {
            // `z != z` or `dz != dz` detects a non-finite coordinate without device math
            // intrinsics. A zero direction component is the vacuum/no-face case.
            if(payload.z != payload.z || payload.dz != payload.dz || payload.dz == 0.0)
                status.set(graphFlow::StatusCode::invalid);
        }
    };

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }

    template<typename TStorage, typename TPayload>
    void writeChunk(
        Device& device,
        TStorage& storage,
        graphFlow::ChunkHandle handle,
        std::vector<TPayload> const& host)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(host.size())});
        alpaka::onHost::memcpy(queue, view, host);
    }

    graphFlow::Config configFor(std::uint32_t capacity, std::uint32_t batchChunks)
    {
        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = batchChunks;
        config.tileSize = capacity;
        config.frameCap = 1u;
        return config;
    }

    template<typename TStage, typename TQueue>
    void requireSingleUse(TStage& stage, TQueue const& queue)
    {
        using Batch = typename graphFlow::GraphRunner<Device, Storage>::Batch;
        CHECK_THROWS_AS(stage.launch(queue, Batch{}), std::logic_error);
    }
} // namespace

TEST_CASE("status codes decode onto the documented failure classes", "[unit][flow][status]")
{
    CHECK(graphFlow::decodeStatus(0u) == graphFlow::StatusCode::ok);
    CHECK(graphFlow::decodeStatus(1u) == graphFlow::StatusCode::invalid);
    CHECK(graphFlow::decodeStatus(2u) == graphFlow::StatusCode::overflow);
    CHECK(graphFlow::decodeStatus(99u) == graphFlow::StatusCode::invalid);
    CHECK(std::string{graphFlow::statusCodeName(graphFlow::StatusCode::invalid)} == "invalid");
    CHECK(std::string{graphFlow::statusCodeName(graphFlow::StatusCode::overflow)} == "overflow");
}

TEST_CASE("a source status failure is detected before publication", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 2u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::SourceStage<Device, Storage, Pod, FailSource> stage{device, storage, config, FailSource{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};
    stage.launch(queue, batch);
    drain(device, queue);

    std::uint32_t allocated = 0u;
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
    CHECK(allocated == 0u);
    stage.abort();
    stage.abort();
    CHECK(storage.freeCount() == 1u);
    requireSingleUse(stage, queue);

    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("a sink status failure is detected before publication", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 2u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::SinkStage<Device, Storage, Pod, FailSink> stage{device, storage, config, FailSink{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};
    stage.launch(queue, batch);
    drain(device, queue);

    std::uint32_t allocated = 0u;
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
    CHECK(allocated == 0u);
    stage.abort();
    stage.abort();
    CHECK(storage.freeCount() == 1u);
    requireSingleUse(stage, queue);

    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("a transform status failure releases planned outputs exactly once", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 8u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::TransformStage<Device, Storage, Pod, FailTransform> stage{
        device,
        storage,
        config,
        FailTransform{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    writeChunk(device, storage, input.handle, std::vector<Pod>{Pod{0u, 0u}, Pod{1u, 0u}, Pod{2u, 0u}, Pod{3u, 0u}});
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};

    stage.launch(queue, batch);
    // The launch acquired the single output chunk before the device status existed.
    CHECK(storage.freeCount() == 6u);
    drain(device, queue);

    std::uint32_t allocated = 0u;
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
    CHECK(allocated == 0u);

    stage.abort();
    CHECK(storage.freeCount() == 7u);
    // A second abort is a no-op because the pending descriptor list was cleared.
    stage.abort();
    CHECK(storage.freeCount() == 7u);
    // No descriptor is published after the error path.
    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    CHECK(outputs.empty());
    requireSingleUse(stage, queue);

    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("a classifier status failure rejects the batch before planning", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 12u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::PartitionStage<Device, Storage, Pod, StatusClassifier, graphFlow::NoUpdate>
        stage{device, storage, config, StatusClassifier{}, graphFlow::NoUpdate{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
    CHECK(allocated == 0u);
    // No output chunk exists because the status is checked before host planning.
    CHECK(storage.freeCount() == 11u);

    stage.abort();
    stage.abort();
    CHECK(storage.freeCount() == 11u);
    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    CHECK(outputs.empty());
    requireSingleUse(stage, queue);

    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("an invalid updater result rejects the planned batch", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 12u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::PartitionStage<Device, Storage, Pod, ValidClassifier, InvalidUpdate>
        stage{device, storage, config, ValidClassifier{}, InvalidUpdate{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    REQUIRE(stage.advance(queue, batch, allocated));
    CHECK(allocated == 2u);
    CHECK(storage.freeCount() == 9u);
    drain(device, queue);
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);

    // The failed advance returns the planned chunks; `abort` must not release them a second time.
    CHECK(storage.freeCount() == 11u);
    stage.abort();
    CHECK(storage.freeCount() == 11u);
    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    CHECK(outputs.empty());
    requireSingleUse(stage, queue);

    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("a successful updater keeps the status word clear", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 12u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::PartitionStage<Device, Storage, Pod, ValidClassifier, BusyUpdate>
        stage{device, storage, config, ValidClassifier{}, BusyUpdate{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    REQUIRE(stage.advance(queue, batch, allocated));
    CHECK(allocated == 2u);
    drain(device, queue);
    CHECK_FALSE(stage.advance(queue, batch, allocated));

    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    REQUIRE(outputs.size() == 2u);
    for(graphFlow::ChunkDescriptor const& output : outputs)
        storage.release(output.handle);
    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("non-finite geometry and a vacuum ray fail the transform batch", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using GeometryStorage = graphFlow::PayloadSlotStorage<GeometryPayload, Device>;
    GeometryStorage storage{device, 4u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::TransformStage<Device, GeometryStorage, GeometryPayload, ValidateGeometry> stage{
        device,
        storage,
        config,
        ValidateGeometry{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    double const nan = std::numeric_limits<double>::quiet_NaN();
    writeChunk(
        device,
        storage,
        input.handle,
        std::vector<GeometryPayload>{{0.0, 0.0}, {nan, 1.0}, {1.0, 1.0}, {1.0, -1.0}});
    using Batch = graphFlow::GraphRunner<Device, GeometryStorage>::Batch;
    Batch const batch{{input}, 0u};

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
    stage.abort();
    CHECK(storage.freeCount() == 3u);
    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}

TEST_CASE("a checked updater overflow fails the planned batch", "[unit][flow][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Storage storage{device, 12u, 4u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::Config const config = configFor(4u, 1u);
    graphFlow::detail::PartitionStage<Device, Storage, Pod, ValidClassifier, OverflowUpdate>
        stage{device, storage, config, ValidClassifier{}, OverflowUpdate{}};

    graphFlow::ChunkDescriptor const input{storage.acquire(), 4u, 0u};
    using Batch = graphFlow::GraphRunner<Device, Storage>::Batch;
    Batch const batch{{input}, 0u};

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    REQUIRE(stage.advance(queue, batch, allocated));
    CHECK(allocated == 2u);
    drain(device, queue);
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
    CHECK(storage.freeCount() == 11u);
    stage.abort();
    CHECK(storage.freeCount() == 11u);
    requireSingleUse(stage, queue);
    storage.release(input.handle);
    CHECK(storage.freeCount() == storage.numChunks());
}
