/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    using U32View
        = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<Device>(), alpaka::Vec{std::uint64_t{1u}}));

    /** Simple POD payload of the typed execution path. */
    struct Pod
    {
        std::uint64_t id = 0u;
        std::uint32_t value = 0u;
    };

    struct GeneratePod
    {
        ALPAKA_FN_ACC Pod operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Pod{id, 0u};
        }
    };

    struct BumpPod
    {
        std::uint32_t amount = 0u;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod& payload) const
        {
            payload.value += amount;
        }
    };

    struct ParityClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % 2u);
        }
    };

    struct AddOnOddBin
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Pod& payload) const
        {
            if(bin == 1u)
                payload.value += 1u;
            return graphFlow::UpdateResult::success;
        }
    };

    template<typename TView>
    struct RecordPod
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            values[payload.id] = payload.value;
        }
    };

    struct BumpParticle
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle& particle) const
        {
            particle.weight += 2u;
            particle.visit += 1u;
        }
    };

    template<typename TView>
    struct RecordParticleWeight
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle const& particle)
            const
        {
            values[particle.id] = particle.weight;
        }
    };

    struct GenerateParticle
    {
        ALPAKA_FN_ACC graphFlow::Particle operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return graphFlow::Particle{id, 0u, 0u, 0u};
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

    template<typename TPayload, typename TStorage>
    std::vector<TPayload> readChunk(
        Device& device,
        TStorage& storage,
        graphFlow::ChunkHandle handle,
        std::uint32_t fill)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        std::vector<TPayload> host(fill);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(fill)});
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    template<typename TPayload, typename TStorage>
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

    graphFlow::Tuning tuningFor(
        std::uint32_t capacity,
        std::uint32_t tileSize,
        std::uint32_t batchChunks,
        std::uint32_t numChunks = 0u)
    {
        graphFlow::Tuning tuning;
        tuning.chunkCapacity = capacity;
        tuning.tileSize = tileSize;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        if(numChunks != 0u)
        {
            tuning.numChunks = numChunks;
            tuning.poolPayloadBudgetBytes = std::uint64_t{1u} << 24u;
        }
        return tuning;
    }

    struct PodRun
    {
        std::vector<std::uint32_t> values;
        graphFlow::SchedulerStats stats;
    };

    PodRun runPod(std::uint64_t particles, graphFlow::Tuning const& tuning, std::uint64_t* observed = nullptr)
    {
        auto device = graphFlow::test::makeCpuDevice();
        auto results
            = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::max<std::uint64_t>(particles, 1u)});

        graphFlow::Flow<Pod, Device> flow{device};
        auto source = flow.source("generate", GeneratePod{});
        auto bump = flow.transform("bump", BumpPod{3u});
        auto parity = flow.partition("parity", ParityClassifier{}, AddOnOddBin{});
        using ResultView = decltype(results.getView());
        auto sink = flow.sink("record", RecordPod<ResultView>{results.getView()});
        flow.connect(source, bump);
        flow.connect(bump, parity);
        flow.connect(parity.out<0u>(), sink);
        flow.connect(parity.out<1u>(), sink);
        flow.configure(tuning);

        PodRun run;
        run.stats = flow.run(source, particles);
        if(observed != nullptr)
            *observed = run.stats.completedBatches;
        std::vector<std::uint32_t> host(std::max<std::uint64_t>(particles, 1u));
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, results);
        host.resize(static_cast<std::size_t>(particles));
        run.values = std::move(host);
        return run;
    }

    void checkPodValues(std::vector<std::uint32_t> const& values)
    {
        for(std::uint64_t id = 0u; id < values.size(); ++id)
            CHECK(values[static_cast<std::size_t>(id)] == 3u + static_cast<std::uint32_t>(id % 2u));
    }
} // namespace

TEST_CASE("storage access trait exposes load and store views", "[unit][flow][execution]")
{
    using SlotAccess = graphFlow::detail::StorageAccess<graphFlow::PayloadSlotStorage<Pod, Device>, Pod>;
    using ParticleAccess = graphFlow::detail::StorageAccess<graphFlow::ParticleStorage<Device>, graphFlow::Particle>;
    static_assert(std::same_as<typename SlotAccess::View1, typename SlotAccess::View0>);
    static_assert(!std::same_as<typename ParticleAccess::View0, typename ParticleAccess::View1>);

    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::PayloadSlotStorage<Pod, Device> storage{device, 2u, 2u};
    auto const handle = storage.acquire();
    writeChunk(device, storage, handle, std::vector<Pod>{Pod{0u, 1u}, Pod{1u, 2u}});

    using Access = graphFlow::detail::StorageAccess<graphFlow::PayloadSlotStorage<Pod, Device>, Pod>;
    auto view0 = Access::view0(storage);
    auto view1 = Access::view1(storage);
    auto view2 = Access::view2(storage);
    auto view3 = Access::view3(storage);
    Access access;
    std::uint32_t const base = static_cast<std::uint32_t>(storage.base(handle));
    CHECK(access.load(view0, view1, view2, view3, base).value == 1u);
    access.store(view0, view1, view2, view3, base, Pod{7u, 9u});
    std::vector<Pod> const read = readChunk<Pod>(device, storage, handle, 2u);
    CHECK(read[0].id == 7u);
    CHECK(read[0].value == 9u);
}

TEST_CASE("typed flow executes source, transform, partition and sink", "[unit][flow][execution]")
{
    PodRun const run = runPod(17u, tuningFor(8u, 4u, 1u));
    REQUIRE(run.values.size() == 17u);
    checkPodValues(run.values);
    CHECK(run.stats.completedBatches != 0u);
}

TEST_CASE("typed flow covers empty, single and partial-chunk runs", "[unit][flow][execution]")
{
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 128ull})
    {
        PodRun const run = runPod(particles, tuningFor(8u, 4u, 2u));
        CAPTURE(particles);
        REQUIRE(run.values.size() == static_cast<std::size_t>(particles));
        checkPodValues(run.values);
    }
}

TEST_CASE("typed flow honours Q and multi-chunk transform batches", "[unit][flow][execution]")
{
    for(std::uint64_t particles : {17ull, 128ull, 4097ull})
    {
        for(std::uint32_t batchChunks : {1u, 2u, 8u})
        {
            CAPTURE(particles, batchChunks);
            PodRun const run = runPod(particles, tuningFor(32u, 8u, batchChunks));
            REQUIRE(run.values.size() == static_cast<std::size_t>(particles));
            checkPodValues(run.values);
        }
    }
}

TEST_CASE("typed flow is repeatable and keeps a bounded pool", "[unit][flow][execution]")
{
    auto device = graphFlow::test::makeCpuDevice();
    auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{32u}});

    graphFlow::Flow<Pod, Device> flow{device};
    auto source = flow.source("generate", GeneratePod{});
    auto bump = flow.transform("bump", BumpPod{3u});
    auto parity = flow.partition("parity", ParityClassifier{}, AddOnOddBin{});
    using ResultView = decltype(results.getView());
    auto sink = flow.sink("record", RecordPod<ResultView>{results.getView()});
    flow.connect(source, bump);
    flow.connect(bump, parity);
    flow.connect(parity.out<0u>(), sink);
    flow.connect(parity.out<1u>(), sink);

    // P is derived as min(N, 2*C) = 16; the graph has two transform nodes with R = 1 and R = 2.
    graphFlow::Tuning const tuning = tuningFor(8u, 4u, 1u, 19u);
    flow.configure(tuning);

    graphFlow::SchedulerStats first = flow.run(source, 32u);
    CHECK(first.poolChunks == 19u);
    CHECK(first.peakLiveChunks <= first.poolChunks);

    std::vector<std::uint32_t> host(32u);
    auto queue = device.makeQueue(alpaka::queueKind::blocking);
    alpaka::onHost::memcpy(queue, host, results);
    checkPodValues(host);

    // A second run rebuilds the stages and produces the same values.
    graphFlow::SchedulerStats second = flow.run(source, 32u);
    CHECK(second.poolChunks == 19u);
    alpaka::onHost::memcpy(queue, host, results);
    checkPodValues(host);
}

TEST_CASE("particle storage runs through the typed source and sink", "[unit][flow][execution]")
{
    auto device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t particles = 17u;
    auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});

    graphFlow::Flow<graphFlow::Particle, Device> flow{device};
    auto source = flow.source("generate", GenerateParticle{});
    auto bump = flow.transform("bump", BumpParticle{});
    using ResultView = decltype(results.getView());
    auto sink = flow.sink("record", RecordParticleWeight<ResultView>{results.getView()});
    flow.connect(source, bump);
    flow.connect(bump, sink);
    flow.configure(tuningFor(8u, 4u, 2u));

    auto const stats = flow.run(source, particles);
    CHECK(stats.poolChunks != 0u);

    std::vector<std::uint32_t> host(static_cast<std::size_t>(particles));
    auto queue = device.makeQueue(alpaka::queueKind::blocking);
    alpaka::onHost::memcpy(queue, host, results);
    for(std::uint32_t weight : host)
        CHECK(weight == 2u);
}

TEST_CASE("one transform batch issues exactly one launch", "[unit][flow][execution]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
    constexpr std::uint32_t capacity = 4u;
    Storage storage{device, 8u, capacity};

    graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
    writeChunk(device, storage, input.handle, std::vector<Pod>{Pod{0u, 1u}, Pod{1u, 2u}, Pod{2u, 3u}, Pod{3u, 4u}});

    graphFlow::Config config;
    config.chunkCapacity = capacity;
    config.batchChunks = 1u;
    config.tileSize = 4u;
    config.frameCap = 1u;

    graphFlow::detail::TransformStage<Device, Storage, Pod, BumpPod> stage{device, storage, config, BumpPod{3u}};
    using Runner = graphFlow::GraphRunner<Device, Storage>;
    Runner::Batch const batch{{input}, 0u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    CHECK_FALSE(stage.advance(queue, batch, allocated));
    CHECK(allocated == 1u);

    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    REQUIRE(outputs.size() == 1u);
    CHECK(stage.launches() == 1u);

    std::vector<Pod> const read = readChunk<Pod>(device, storage, outputs[0].handle, outputs[0].fill);
    REQUIRE(read.size() == capacity);
    for(std::uint32_t i = 0u; i < capacity; ++i)
        CHECK(read[i].value == i + 1u + 3u);

    for(graphFlow::ChunkDescriptor const& output : outputs)
        storage.release(output.handle);
    storage.release(input.handle);
}

TEST_CASE("one partition batch issues one classify and one scatter launch", "[unit][flow][execution]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
    constexpr std::uint32_t capacity = 4u;
    Storage storage{device, 12u, capacity};

    graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
    writeChunk(device, storage, input.handle, std::vector<Pod>{Pod{0u, 0u}, Pod{1u, 0u}, Pod{2u, 0u}, Pod{3u, 0u}});

    graphFlow::Config config;
    config.chunkCapacity = capacity;
    config.batchChunks = 1u;
    config.tileSize = 4u;
    config.frameCap = 1u;

    graphFlow::detail::PartitionStage<Device, Storage, Pod, ParityClassifier, AddOnOddBin>
        stage{device, storage, config, ParityClassifier{}, AddOnOddBin{}};
    using Runner = graphFlow::GraphRunner<Device, Storage>;
    Runner::Batch const batch{{input}, 0u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

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
    CHECK(stage.classifyLaunches() == 1u);
    CHECK(stage.scatterLaunches() == 1u);

    std::uint32_t total = 0u;
    for(graphFlow::ChunkDescriptor const& output : outputs)
    {
        std::vector<Pod> const read = readChunk<Pod>(device, storage, output.handle, output.fill);
        total += output.fill;
        for(Pod const& payload : read)
            CHECK(payload.value == (output.bin == 1u ? 1u : 0u));
    }
    CHECK(total == capacity);

    for(graphFlow::ChunkDescriptor const& output : outputs)
        storage.release(output.handle);
    storage.release(input.handle);
}
