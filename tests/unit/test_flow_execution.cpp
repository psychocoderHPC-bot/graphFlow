/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/graphFlow.hpp"
#include "integration/StallDiagnostics.hpp"

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

TEST_CASE(
    "a stale base table is rejected instead of writing into another chunk",
    "[unit][flow][execution][arena][injection]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
    constexpr std::uint32_t capacity = 32u;

    graphFlow::SlotArenaOptions options;
    options.arenaSlots = 128u;
    Storage storage{device, 8u, capacity, options};

    graphFlow::Config config;
    config.chunkCapacity = capacity;
    config.batchChunks = 1u;
    config.tileSize = 4u;
    config.frameCap = 1u;
    config.liveParticles = 1u;
    config.poolMode = graphFlow::PoolMode::slotArena;
    config.debugStaleBaseTable = true;

    using Stage = graphFlow::detail::PartitionStage<Device, Storage, Pod, ParityClassifier, AddOnOddBin>;
    if constexpr(!graphFlow::debug::checksEnabled)
    {
        // The injection is validation-only; a release build refuses it instead of corrupting.
        CHECK_THROWS_AS(Stage(device, storage, config, ParityClassifier{}, AddOnOddBin{}), std::invalid_argument);
        return;
    }
    else
    {
        Stage stage{device, storage, config, ParityClassifier{}, AddOnOddBin{}};
        using Runner = graphFlow::GraphRunner<Device, Storage>;
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        std::uint32_t allocated = 0u;

        // Batch 1: chunk A owns slots [0, 10), chunk B owns [10, 30). The stage publishes the
        // table once, so the device row of B is its base 10.
        auto const a = storage.acquire(10u);
        auto const b = storage.acquire(20u);
        REQUIRE(storage.base(b) == 10u);
        writeChunk(device, storage, b, std::vector<Pod>(20u, Pod{1u, 0u}));
        Runner::Batch const first{{graphFlow::ChunkDescriptor{b, 20u, 0u}}, 0u};
        stage.launch(queue, first);
        drain(device, queue);
        REQUIRE(stage.advance(queue, first, allocated));
        drain(device, queue);
        CHECK_FALSE(stage.advance(queue, first, allocated));

        std::vector<graphFlow::ChunkDescriptor> outputs;
        stage.finish(first, outputs);
        for(graphFlow::ChunkDescriptor const& output : outputs)
            storage.release(output.handle);
        storage.release(a);
        storage.release(b);

        // Batch 2 re-acquires the same chunk index for a coalesced range at base 0, so batch 1's
        // device row 10 is stale. The kernel-side comparison must reject the batch.
        auto const reused = storage.acquire(20u);
        REQUIRE(reused.index == b.index);
        REQUIRE(storage.base(reused) == 0u);
        writeChunk(device, storage, reused, std::vector<Pod>(20u, Pod{1u, 0u}));
        Runner::Batch const second{{graphFlow::ChunkDescriptor{reused, 20u, 0u}}, 20u};
        stage.launch(queue, second);
        drain(device, queue);
        CHECK_THROWS_AS(stage.advance(queue, second, allocated), graphFlow::BatchError);
        storage.release(reused);
    }
}

TEST_CASE("CPU scheduler stall snapshot is aggregate and topology free", "[stall][cpu][diagnostics-check]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::test::diagnostics::checkStallOutcome(graphFlow::test::diagnostics::runForcedStall(device));
}

TEST_CASE("CPU completed-run scheduler stats satisfy the peak invariants", "[stats][cpu][diagnostics-check]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::test::diagnostics::checkCompletedOutcome(graphFlow::test::diagnostics::runCompleted(device));
}

TEST_CASE("fixed layout publishes the chunk base table once per run", "[unit][flow][execution][arena]")
{
    using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
    using Runner = graphFlow::GraphRunner<Device, Storage>;
    constexpr std::uint32_t capacity = 8u;
    constexpr std::uint32_t batches = 3u;

    auto device = graphFlow::test::makeCpuDevice();
    auto makeStorage = [&](graphFlow::PoolMode mode)
    {
        if(mode == graphFlow::PoolMode::fixed)
            return Storage{device, 16u, capacity};
        graphFlow::SlotArenaOptions options;
        options.arenaSlots = 16u * capacity;
        return Storage{device, 16u, capacity, options};
    };

    // The fixed layout's rows are the immutable `index * chunkCapacity`, so the stage must publish
    // the device mirror once. The slot arena rewrites rows on acquire and release, so it keeps
    // publishing before every kernel that indexes the table.
    auto transformPublishes = [&](graphFlow::PoolMode mode)
    {
        Storage storage = makeStorage(mode);
        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = 1u;
        config.tileSize = 4u;
        config.frameCap = 1u;
        config.poolMode = mode;

        graphFlow::detail::TransformStage<Device, Storage, Pod, BumpPod> stage{device, storage, config, BumpPod{3u}};
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        std::uint64_t position = 0u;
        for(std::uint32_t batch = 0u; batch < batches; ++batch)
        {
            graphFlow::ChunkHandle const handle = storage.acquire();
            writeChunk(device, storage, handle, std::vector<Pod>(capacity, Pod{0u, 1u}));
            Runner::Batch const descriptors{{graphFlow::ChunkDescriptor{handle, capacity, 0u}}, position};
            stage.launch(queue, descriptors);
            drain(device, queue);
            std::uint32_t allocated = 0u;
            REQUIRE_FALSE(stage.advance(queue, descriptors, allocated));
            std::vector<graphFlow::ChunkDescriptor> outputs;
            stage.finish(descriptors, outputs);
            for(graphFlow::ChunkDescriptor const& output : outputs)
                storage.release(output.handle);
            storage.release(handle);
            position += capacity;
        }
        return stage.baseTablePublishes();
    };

    CHECK(transformPublishes(graphFlow::PoolMode::fixed) == 1u);
    CHECK(transformPublishes(graphFlow::PoolMode::slotArena) == batches);

    auto partitionPublishes = [&](graphFlow::PoolMode mode)
    {
        Storage storage = makeStorage(mode);
        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = 1u;
        config.tileSize = 4u;
        config.frameCap = 1u;
        config.poolMode = mode;

        graphFlow::detail::PartitionStage<Device, Storage, Pod, ParityClassifier, AddOnOddBin>
            stage{device, storage, config, ParityClassifier{}, AddOnOddBin{}};
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        for(std::uint32_t batch = 0u; batch < batches; ++batch)
        {
            graphFlow::ChunkHandle const handle = storage.acquire();
            std::vector<Pod> input(capacity);
            for(std::uint32_t i = 0u; i < capacity; ++i)
                input[i] = Pod{i, 1u};
            writeChunk(device, storage, handle, input);
            Runner::Batch const descriptors{{graphFlow::ChunkDescriptor{handle, capacity, 0u}}, 0u};
            stage.launch(queue, descriptors);
            drain(device, queue);
            std::uint32_t allocated = 0u;
            REQUIRE(stage.advance(queue, descriptors, allocated));
            drain(device, queue);
            REQUIRE_FALSE(stage.advance(queue, descriptors, allocated));
            std::vector<graphFlow::ChunkDescriptor> outputs;
            stage.finish(descriptors, outputs);
            for(graphFlow::ChunkDescriptor const& output : outputs)
                storage.release(output.handle);
            storage.release(handle);
        }
        return stage.baseTablePublishes();
    };

    CHECK(partitionPublishes(graphFlow::PoolMode::fixed) == 1u);
    // One publish before classify and one after the output acquires per batch.
    CHECK(partitionPublishes(graphFlow::PoolMode::slotArena) == 2u * batches);
}
