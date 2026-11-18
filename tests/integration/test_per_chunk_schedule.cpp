/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    /** Payload of the transform equality tests. `id` is the batch-global particle index. */
    struct Pod
    {
        std::uint64_t id = 0u;
        std::uint32_t value = 0u;
    };

    /** Position-independent, payload-dependent transform: it rewrites `value` from `id`. */
    struct AddIdValue
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod& payload) const
        {
            payload.value = static_cast<std::uint32_t>(payload.id % 97u) + 3u;
        }
    };

    std::uint32_t expectedValue(std::uint64_t id)
    {
        return static_cast<std::uint32_t>(id % 97u) + 3u;
    }

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
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

    struct StageRun
    {
        /** Output payloads indexed by batch-global output position. */
        std::vector<Pod> byPosition;
        std::uint32_t launches = 0u;
        std::uint32_t outputChunks = 0u;
    };

    /** Run one transform stage over explicit chunk fills and collect the batch-global output. */
    StageRun runStage(
        graphFlow::ChunkKernelSchedule schedule,
        std::uint32_t capacity,
        std::uint32_t tileSize,
        std::vector<std::uint32_t> const& fills)
    {
        using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
        using Runner = graphFlow::GraphRunner<Device, Storage>;

        auto device = graphFlow::test::makeCpuDevice();
        std::uint32_t const particles = std::accumulate(fills.begin(), fills.end(), 0u);
        std::uint32_t const outputChunks = (particles + capacity - 1u) / capacity;
        Storage storage{device, static_cast<std::uint32_t>(fills.size()) + outputChunks + 2u, capacity};

        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = static_cast<std::uint32_t>(fills.size());
        config.tileSize = tileSize;
        config.frameCap = 2u;
        config.chunkKernelSchedule = schedule;
        config.chunkQueueDepth = 1u;

        graphFlow::detail::TransformStage<Device, Storage, Pod, AddIdValue> stage{
            device,
            storage,
            config,
            AddIdValue{}};

        Runner::Batch batch;
        batch.firstId = 0u;
        std::uint64_t position = 0u;
        for(std::uint32_t const fill : fills)
        {
            graphFlow::ChunkHandle const handle = storage.acquire();
            std::vector<Pod> input(fill);
            for(std::uint32_t i = 0u; i < fill; ++i)
                input[i] = Pod{position + i, 0xFFFF'FFFFu};
            writeChunk<Pod>(device, storage, handle, input);
            batch.chunks.push_back(graphFlow::ChunkDescriptor{handle, fill, 0u});
            position += fill;
        }

        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        stage.launch(queue, batch);
        drain(device, queue);
        std::uint32_t allocated = 0u;
        CHECK_FALSE(stage.advance(queue, batch, allocated));
        CHECK(allocated == outputChunks);

        std::vector<graphFlow::ChunkDescriptor> outputs;
        stage.finish(batch, outputs);
        REQUIRE(outputs.size() == outputChunks);

        StageRun run;
        run.byPosition.resize(particles);
        for(std::uint32_t k = 0u; k < outputChunks; ++k)
        {
            std::uint32_t const fill = outputs[k].fill;
            std::vector<Pod> const read = readChunk<Pod>(device, storage, outputs[k].handle, fill);
            REQUIRE(read.size() == fill);
            std::uint64_t const start = static_cast<std::uint64_t>(k) * capacity;
            for(std::uint32_t i = 0u; i < fill; ++i)
                run.byPosition[start + i] = read[i];
        }
        run.launches = stage.launches();
        run.outputChunks = outputChunks;

        for(auto const& output : outputs)
            storage.release(output.handle);
        for(auto const& input : batch.chunks)
            storage.release(input.handle);
        return run;
    }

    /** Count positions whose payload differs from the expected transform result. */
    std::size_t wrongPositions(StageRun const& run)
    {
        std::size_t wrong = 0u;
        for(std::size_t i = 0u; i < run.byPosition.size(); ++i)
        {
            if(run.byPosition[i].id != i || run.byPosition[i].value != expectedValue(i))
                ++wrong;
        }
        return wrong;
    }

    /** Compare two stage runs payload-exactly and check both against the transform result. */
    void checkEqualAndExact(StageRun const& off, StageRun const& on)
    {
        REQUIRE(off.byPosition.size() == on.byPosition.size());
        REQUIRE(off.outputChunks == on.outputChunks);
        INFO("off wrong=" << wrongPositions(off) << " on wrong=" << wrongPositions(on));
        CHECK(wrongPositions(off) == 0u);
        CHECK(wrongPositions(on) == 0u);
        for(std::size_t i = 0u; i < off.byPosition.size(); ++i)
        {
            CAPTURE(i);
            CHECK(off.byPosition[i].id == on.byPosition[i].id);
            CHECK(off.byPosition[i].value == on.byPosition[i].value);
        }
    }

    /** Report `code` from every payload whose batch-global id is at or after `firstId`.
     *
     * The status word is per launch, so a functor that fails only inside one chunk pins the
     * failure to that chunk's slot.
     */
    struct FailFromId
    {
        std::uint64_t firstId = 0u;
        graphFlow::StatusCode code = graphFlow::StatusCode::overflow;

        ALPAKA_FN_ACC void operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            Pod& payload) const
        {
            payload.value = expectedValue(payload.id);
            if(payload.id >= firstId)
                status.set(code);
        }
    };

    /** What one per-chunk batch reported through `advance`. */
    struct StatusRun
    {
        bool threw = false;
        std::string message;
        std::uint32_t allocated = 0u;
        std::uint32_t launches = 0u;
        /** Output payloads indexed by batch-global output position; empty when the batch failed. */
        std::vector<Pod> byPosition;
    };

    /** Launch one per-chunk transform batch over explicit fills and record the status outcome. */
    template<typename TFunctor>
    StatusRun runStatusBatch(std::uint32_t capacity, std::vector<std::uint32_t> const& fills, TFunctor functor)
    {
        using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
        using Runner = graphFlow::GraphRunner<Device, Storage>;
        auto device = graphFlow::test::makeCpuDevice();
        std::uint32_t const particles = std::accumulate(fills.begin(), fills.end(), 0u);
        std::uint32_t const outputChunks = (particles + capacity - 1u) / capacity;
        Storage storage{device, static_cast<std::uint32_t>(fills.size()) + outputChunks + 2u, capacity};

        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = static_cast<std::uint32_t>(fills.size());
        config.tileSize = 32u;
        config.frameCap = 2u;
        config.chunkKernelSchedule = graphFlow::ChunkKernelSchedule::perChunk;
        config.chunkQueueDepth = 1u;

        graphFlow::detail::TransformStage<Device, Storage, Pod, TFunctor> stage{device, storage, config, functor};

        Runner::Batch batch;
        batch.firstId = 0u;
        std::uint64_t position = 0u;
        for(std::uint32_t const fill : fills)
        {
            graphFlow::ChunkHandle const handle = storage.acquire();
            std::vector<Pod> input(fill);
            for(std::uint32_t i = 0u; i < fill; ++i)
                input[i] = Pod{position + i, 0xFFFF'FFFFu};
            writeChunk<Pod>(device, storage, handle, input);
            batch.chunks.push_back(graphFlow::ChunkDescriptor{handle, fill, 0u});
            position += fill;
        }

        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        stage.launch(queue, batch);
        drain(device, queue);

        StatusRun run;
        run.launches = stage.launches();
        bool advanced = false;
        try
        {
            advanced = stage.advance(queue, batch, run.allocated);
        }
        catch(graphFlow::BatchError const& error)
        {
            run.threw = true;
            run.message = error.what();
        }
        CHECK_FALSE(advanced);

        std::vector<graphFlow::ChunkDescriptor> outputs;
        if(run.threw)
        {
            // The failed batch owns its planned chunks: `abort` returns them once and `finish`
            // then reports nothing, exactly like the batch-wide failure path.
            stage.abort();
        }
        else
        {
            stage.finish(batch, outputs);
            run.byPosition.resize(particles);
            for(std::uint32_t k = 0u; k < outputChunks; ++k)
            {
                std::uint32_t const fill = outputs[k].fill;
                std::vector<Pod> const read = readChunk<Pod>(device, storage, outputs[k].handle, fill);
                std::uint64_t const start = static_cast<std::uint64_t>(k) * capacity;
                for(std::uint32_t i = 0u; i < fill; ++i)
                    run.byPosition[start + i] = read[i];
            }
        }
        for(auto const& output : outputs)
            storage.release(output.handle);
        for(auto const& input : batch.chunks)
            storage.release(input.handle);
        return run;
    }

    /** Count positions whose payload differs from the expected transform result. */
    std::size_t wrongPositionsOf(std::vector<Pod> const& byPosition)
    {
        std::size_t wrong = 0u;
        for(std::size_t i = 0u; i < byPosition.size(); ++i)
        {
            if(byPosition[i].id != i || byPosition[i].value != expectedValue(i))
                ++wrong;
        }
        return wrong;
    }
} // namespace

TEST_CASE(
    "per-chunk transform output equals the batch-wide output across the C/T matrix",
    "[integration][chunk-schedule]")
{
    using graphFlow::ChunkKernelSchedule;

    struct Case
    {
        std::uint32_t capacity;
        std::vector<std::uint32_t> fills;
    };

    // Crossings are explicit: a chunk whose position is not a multiple of the capacity writes a
    // contiguous output range that spans two output chunks.
    std::vector<Case> const cases{
        {1u, {1u, 1u, 1u}},
        {31u, {20u, 20u, 20u, 7u}},
        {127u, {100u, 100u, 13u}},
        {128u, {100u, 100u, 37u}},
        {128u, {128u, 128u, 128u, 1u}},
        {8192u, {5000u, 8192u, 3000u}},
        {8192u, {8192u, 8192u, 5760u}},
    };

    for(Case const& testCase : cases)
    {
        for(std::uint32_t const tileSize : {32u, 128u, 512u})
        {
            CAPTURE(testCase.capacity, tileSize, testCase.fills.size());
            StageRun const off = runStage(ChunkKernelSchedule::off, testCase.capacity, tileSize, testCase.fills);
            StageRun const on = runStage(ChunkKernelSchedule::perChunk, testCase.capacity, tileSize, testCase.fills);
            checkEqualAndExact(off, on);
            CHECK(off.launches == 1u);
            CHECK(on.launches == static_cast<std::uint32_t>(testCase.fills.size()));
        }
    }
}

TEST_CASE("per-chunk transform issues one kernel launch per chunk at Q>1", "[integration][chunk-schedule]")
{
    using graphFlow::ChunkKernelSchedule;

    // Eight full chunks: the batch-wide path issues one kernel, the per-chunk path eight.
    std::vector<std::uint32_t> const fills(8u, 128u);
    StageRun const off = runStage(ChunkKernelSchedule::off, 128u, 128u, fills);
    StageRun const on = runStage(ChunkKernelSchedule::perChunk, 128u, 128u, fills);
    checkEqualAndExact(off, on);
    CHECK(off.launches == 1u);
    CHECK(on.launches == 8u);

    // A partial tail is its own launch and still lands on the same global output positions.
    std::vector<std::uint32_t> const tailed{128u, 128u, 128u, 5u};
    StageRun const tailOff = runStage(ChunkKernelSchedule::off, 128u, 128u, tailed);
    StageRun const tailOn = runStage(ChunkKernelSchedule::perChunk, 128u, 128u, tailed);
    checkEqualAndExact(tailOff, tailOn);
    CHECK(tailOff.launches == 1u);
    CHECK(tailOn.launches == 4u);

    // Eight one-particle chunks: the smallest capacity still gets one launch per chunk.
    std::vector<std::uint32_t> const unitChunks(8u, 1u);
    StageRun const unitOff = runStage(ChunkKernelSchedule::off, 1u, 128u, unitChunks);
    StageRun const unitOn = runStage(ChunkKernelSchedule::perChunk, 1u, 128u, unitChunks);
    checkEqualAndExact(unitOff, unitOn);
    CHECK(unitOff.launches == 1u);
    CHECK(unitOn.launches == 8u);
}

TEST_CASE(
    "per-chunk transform equals the batch-wide output for a large-C tail at Q=8",
    "[integration][chunk-schedule]")
{
    using graphFlow::ChunkKernelSchedule;

    // A 5000-particle first chunk shifts every later output position, so the second chunk's
    // [5000, 13192) range crosses the 8192 output boundary.
    std::vector<std::uint32_t> crossing{5000u};
    crossing.insert(crossing.end(), 7u, 8192u);
    StageRun const crossingOff = runStage(ChunkKernelSchedule::off, 8192u, 128u, crossing);
    StageRun const crossingOn = runStage(ChunkKernelSchedule::perChunk, 8192u, 128u, crossing);
    checkEqualAndExact(crossingOff, crossingOn);
    CHECK(crossingOff.launches == 1u);
    CHECK(crossingOn.launches == 8u);

    // Seven full chunks and a 5760-particle tail: the tail owns its own launch and fill.
    std::vector<std::uint32_t> tailed(7u, 8192u);
    tailed.push_back(5760u);
    StageRun const tailOff = runStage(ChunkKernelSchedule::off, 8192u, 128u, tailed);
    StageRun const tailOn = runStage(ChunkKernelSchedule::perChunk, 8192u, 128u, tailed);
    checkEqualAndExact(tailOff, tailOn);
    CHECK(tailOff.launches == 1u);
    CHECK(tailOn.launches == 8u);
}

namespace
{
    struct GeneratePod
    {
        ALPAKA_FN_ACC Pod operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Pod{id, 0u};
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

    /** One end-to-end flow result: the per-id values and the scheduler counters of the run. */
    struct FlowRun
    {
        std::vector<std::uint32_t> values;
        graphFlow::SchedulerStats stats;
    };

    /** Run source -> transform -> partition -> sink and return one value per particle id. */
    FlowRun runFlow(
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t tileSize,
        std::uint32_t batchChunks,
        graphFlow::ChunkKernelSchedule schedule,
        std::optional<std::uint32_t> chunkQueueDepth = 1u,
        std::uint32_t liveParticles = 0u,
        std::uint32_t inflightBatches = 1u,
        std::uint32_t occupancyTargetChunks = 0u)
    {
        auto device = graphFlow::test::makeCpuDevice();
        auto results
            = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::max<std::uint64_t>(particles, 1u)});

        graphFlow::Flow<Pod, Device> flow{device};
        auto source = flow.source("generate", GeneratePod{});
        auto transform = flow.transform("add", AddIdValue{});
        auto parity = flow.partition("parity", ParityClassifier{}, AddOnOddBin{});
        using ResultView = decltype(results.getView());
        auto sink = flow.sink("record", RecordPod<ResultView>{results.getView()});
        flow.connect(source, transform);
        flow.connect(transform, parity);
        flow.connect(parity.out<0u>(), sink);
        flow.connect(parity.out<1u>(), sink);

        graphFlow::Tuning tuning;
        tuning.chunkCapacity = capacity;
        tuning.tileSize = tileSize;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        tuning.chunkKernelSchedule = schedule;
        tuning.chunkQueueDepth = chunkQueueDepth;
        tuning.inflightBatches = inflightBatches;
        if(liveParticles != 0u)
            tuning.liveParticles = liveParticles;
        if(occupancyTargetChunks != 0u)
            tuning.occupancyTargetChunks = occupancyTargetChunks;
        flow.configure(tuning);
        graphFlow::SchedulerStats const stats = flow.run(source, particles);

        std::vector<std::uint32_t> host(static_cast<std::size_t>(particles));
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, results);
        return FlowRun{std::move(host), stats};
    }
} // namespace

TEST_CASE("per-chunk and batch-wide schedules produce identical flow results", "[integration][chunk-schedule]")
{
    using graphFlow::ChunkKernelSchedule;

    for(std::uint64_t const particles : {17ull, 1000ull, 4097ull})
    {
        for(std::uint32_t const capacity : {31u, 127u, 128u})
        {
            for(std::uint32_t const tileSize : {32u, 128u, 512u})
            {
                for(std::uint32_t const batchChunks : {1u, 8u})
                {
                    CAPTURE(particles, capacity, tileSize, batchChunks);
                    FlowRun const off = runFlow(particles, capacity, tileSize, batchChunks, ChunkKernelSchedule::off);
                    FlowRun const on
                        = runFlow(particles, capacity, tileSize, batchChunks, ChunkKernelSchedule::perChunk);
                    REQUIRE(off.values.size() == on.values.size());
                    CHECK(
                        std::memcmp(off.values.data(), on.values.data(), off.values.size() * sizeof(std::uint32_t))
                        == 0);
                    for(std::size_t id = 0u; id < on.values.size(); ++id)
                    {
                        CAPTURE(id);
                        REQUIRE(on.values[id] == expectedValue(id) + static_cast<std::uint32_t>(id % 2u));
                    }
                    // The off path counts batches; the per-chunk path counts kernels and reports the
                    // clamp of its launch-size policy.
                    CHECK(off.stats.chunkLaunches == 0u);
                    CHECK(off.stats.chunkLimitName == "none");
                    CHECK(on.stats.chunkLaunches > 0u);
                    CHECK(on.stats.chunkLimitName != "none");
                }
            }
        }
    }
}

TEST_CASE("per-chunk schedule runs with a multi-queue pool and credit pressure", "[integration][chunk-schedule]")
{
    using graphFlow::ChunkKernelSchedule;

    // Four chunk queues: eight ready chunks are spread over four concurrent lanes.
    std::uint64_t const particles = 4097u;
    std::uint32_t const capacity = 127u;
    std::uint32_t const chunks = static_cast<std::uint32_t>((particles + capacity - 1u) / capacity);

    FlowRun const off = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::off);
    FlowRun const on = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 4u);
    REQUIRE(off.values.size() == on.values.size());
    CHECK(std::memcmp(off.values.data(), on.values.data(), on.values.size() * sizeof(std::uint32_t)) == 0);
    // One kernel per dispatched input chunk, whatever the queue width. Admission may split the
    // source into chunks smaller than C, so the kernel count is at least ceil(N/C).
    CHECK(on.stats.chunkLaunches >= chunks);

    // A two-chunk live budget keeps the credit free-list short: dispatch must shrink to the
    // chunks that fit instead of stalling, and the per-chunk path must retire each chunk's lease
    // on its own (the run-end `freeCount == numChunks` check would throw otherwise).
    FlowRun const pressured = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 4u, 2u * capacity);
    REQUIRE(pressured.values.size() == on.values.size());
    CHECK(std::memcmp(pressured.values.data(), on.values.data(), on.values.size() * sizeof(std::uint32_t)) == 0);
    // The smaller live budget changes the source chunking, not the per-chunk contract: every
    // dispatched chunk is still exactly one kernel.
    CHECK(pressured.stats.chunkLaunches >= chunks);
}

TEST_CASE("per-chunk queue-depth default is two and keeps the physics", "[integration][chunk-schedule][queue-depth]")
{
    using graphFlow::ChunkKernelSchedule;

    // `runFlow` leaves `Tuning::chunkQueueDepth` absent in the default arm, so `Config` resolves
    // it to the measured default of two. The occupancy gate exposes the resolved lane count: this
    // graph registers one lane each for source, partition and sink plus
    // `inflightBatches * chunkQueueDepth` transform lanes, so depth one is four lanes and depth two
    // is five. A target of four therefore has to be rejected only by the default arm.
    std::uint64_t const particles = 4097u;
    std::uint32_t const capacity = 127u;
    std::uint32_t const chunks = static_cast<std::uint32_t>((particles + capacity - 1u) / capacity);

    FlowRun const off = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::off);
    FlowRun const serial = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 1u);
    FlowRun const depthTwo = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 2u);
    FlowRun const defaulted = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, std::nullopt);

    // The queue width is scheduling-only: depth two reproduces the serial depth and the
    // batch-wide path bit for bit.
    REQUIRE(depthTwo.values.size() == serial.values.size());
    CHECK(
        std::memcmp(serial.values.data(), depthTwo.values.data(), serial.values.size() * sizeof(std::uint32_t)) == 0);
    REQUIRE(depthTwo.values.size() == off.values.size());
    CHECK(std::memcmp(off.values.data(), depthTwo.values.data(), off.values.size() * sizeof(std::uint32_t)) == 0);
    for(std::size_t id = 0u; id < depthTwo.values.size(); ++id)
    {
        CAPTURE(id);
        REQUIRE(depthTwo.values[id] == expectedValue(id) + static_cast<std::uint32_t>(id % 2u));
    }
    CHECK(depthTwo.stats.chunkLaunches >= chunks);
    CHECK(depthTwo.stats.sourceSunk == particles);

    // An absent tuning value resolves to the same two-queue run.
    REQUIRE(defaulted.values.size() == depthTwo.values.size());
    CHECK(
        std::memcmp(defaulted.values.data(), depthTwo.values.data(), depthTwo.values.size() * sizeof(std::uint32_t))
        == 0);
    CHECK(defaulted.stats.sourceSunk == particles);

    // Four lanes is an explicit depth-one run and is rejected as the default: the absent value
    // resolves to depth two, whose five lanes need a target of at least five.
    CHECK_NOTHROW(runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 1u, 0u, 1u, 4u));
    CHECK_THROWS_AS(
        runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, std::nullopt, 0u, 1u, 4u),
        std::invalid_argument);
    FlowRun const gated
        = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, std::nullopt, 0u, 1u, 5u);
    REQUIRE(gated.values.size() == depthTwo.values.size());
    CHECK(
        std::memcmp(gated.values.data(), depthTwo.values.data(), depthTwo.values.size() * sizeof(std::uint32_t)) == 0);
}

TEST_CASE("per-chunk status detects a failure that follows a zero-fill chunk", "[integration][chunk-schedule][status]")
{
    // The zero-fill middle chunk owns slot 1 without a launch, so the failing chunk owns slot 2
    // while only two kernels ran. A scan bounded by the kernel count would stop at slot 1 and
    // miss the failure; the scan must use the batch chunk count.
    StatusRun const run = runStatusBatch(4u, {4u, 0u, 4u}, FailFromId{4u, graphFlow::StatusCode::overflow});
    CHECK(run.launches == 2u);
    CHECK(run.threw);
    CHECK(run.message.find("overflow") != std::string::npos);
    CHECK(run.allocated == 0u);
}

TEST_CASE("per-chunk status accepts a zero-fill tail chunk", "[integration][chunk-schedule][status]")
{
    // The last chunk is dispatched as a slot with fill zero: no kernel runs, the slot stays zero
    // from the batch memset, and the scan must neither report it nor read past the batch slots.
    StatusRun const run = runStatusBatch(4u, {4u, 4u, 0u}, FailFromId{1000u, graphFlow::StatusCode::overflow});
    CHECK(run.launches == 2u);
    CHECK_FALSE(run.threw);
    CHECK(run.message.empty());
    CHECK(run.allocated == 2u);
    REQUIRE(run.byPosition.size() == 8u);
    CHECK(wrongPositionsOf(run.byPosition) == 0u);
}

TEST_CASE(
    "per-chunk status detects a failure in a partially filled final chunk",
    "[integration][chunk-schedule][status]")
{
    // Only ids 8 and 9 fall into the tail chunk, so the reported code can only come from the last
    // dispatched slot; a truncated scan would leave the batch clean.
    StatusRun const run = runStatusBatch(4u, {4u, 4u, 2u}, FailFromId{8u, graphFlow::StatusCode::invalid});
    CHECK(run.launches == 3u);
    CHECK(run.threw);
    CHECK(run.message.find("invalid") != std::string::npos);
    CHECK(run.allocated == 0u);
}

TEST_CASE("per-chunk schedule runs two submitted batches over a four-queue pool", "[integration][chunk-schedule]")
{
    using graphFlow::ChunkKernelSchedule;

    // `inflightBatches = 2` with `chunkQueueDepth = 4` is the only configuration that builds
    // `depth * chunkQueues = 8` transform lanes over a four-queue pool, so two lanes share one
    // queue. Every other case runs at the default depth of one, where the lane count equals the
    // queue count and the pairing never happens.
    std::uint64_t const particles = 4097u;
    std::uint32_t const capacity = 127u;
    std::uint32_t const chunks = static_cast<std::uint32_t>((particles + capacity - 1u) / capacity);

    FlowRun const off = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::off);
    // One submitted batch over the same four-queue pool: the per-chunk reference the new
    // two-batch configuration has to reproduce.
    FlowRun const single = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 4u);
    FlowRun const paired = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 4u, 0u, 2u);

    // Physics: identical to the fixed schedule and to the single-batch per-chunk run.
    REQUIRE(off.values.size() == paired.values.size());
    CHECK(std::memcmp(off.values.data(), paired.values.data(), paired.values.size() * sizeof(std::uint32_t)) == 0);
    REQUIRE(single.values.size() == paired.values.size());
    CHECK(std::memcmp(single.values.data(), paired.values.data(), paired.values.size() * sizeof(std::uint32_t)) == 0);
    // No per-lane status or credit corruption: every id keeps its payload and its exact result,
    // which is the check the off/on matrix performs for one lane per queue.
    for(std::size_t id = 0u; id < paired.values.size(); ++id)
    {
        CAPTURE(id);
        REQUIRE(paired.values[id] == expectedValue(id) + static_cast<std::uint32_t>(id % 2u));
    }

    // Every chunk is accounted for at the run end: the runner's chunk-leak check would have thrown
    // if the free list did not tile the whole pool again, and every admitted particle reached a
    // sink. Admission may split the source into chunks smaller than C, so the kernel count is at
    // least ceil(N/C).
    CHECK(paired.stats.chunkLaunches >= chunks);
    CHECK(paired.stats.sourceSunk == particles);
    CHECK(paired.stats.sourceLiveParticles == 0u);

    // The lane count itself is observable through the occupancy gate, which rejects a target below
    // the registered lane count. This graph registers 2 source + 2 * 4 transform + 2 partition +
    // 2 sink = 14 lanes, so a target of 13 has to be rejected and a target of 14 has to run and
    // keep the physics. A lane count that drops one factor of `depth * chunkQueues` accepts 13.
    CHECK_THROWS_AS(
        runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 4u, 0u, 2u, 13u),
        std::invalid_argument);
    FlowRun const gated = runFlow(particles, capacity, 128u, 8u, ChunkKernelSchedule::perChunk, 4u, 0u, 2u, 14u);
    REQUIRE(gated.values.size() == paired.values.size());
    CHECK(std::memcmp(gated.values.data(), paired.values.data(), paired.values.size() * sizeof(std::uint32_t)) == 0);
}
