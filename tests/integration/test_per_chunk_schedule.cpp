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

    /** Run source -> transform -> partition -> sink and return one value per particle id. */
    std::vector<std::uint32_t> runFlow(
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t tileSize,
        std::uint32_t batchChunks,
        graphFlow::ChunkKernelSchedule schedule)
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
        tuning.chunkQueueDepth = 1u;
        flow.configure(tuning);
        flow.run(source, particles);

        std::vector<std::uint32_t> host(static_cast<std::size_t>(particles));
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, results);
        return host;
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
                    std::vector<std::uint32_t> const off
                        = runFlow(particles, capacity, tileSize, batchChunks, ChunkKernelSchedule::off);
                    std::vector<std::uint32_t> const on
                        = runFlow(particles, capacity, tileSize, batchChunks, ChunkKernelSchedule::perChunk);
                    REQUIRE(off.size() == on.size());
                    CHECK(std::memcmp(off.data(), on.data(), off.size() * sizeof(std::uint32_t)) == 0);
                    for(std::size_t id = 0u; id < on.size(); ++id)
                    {
                        CAPTURE(id);
                        REQUIRE(on[id] == expectedValue(id) + static_cast<std::uint32_t>(id % 2u));
                    }
                }
            }
        }
    }
}
