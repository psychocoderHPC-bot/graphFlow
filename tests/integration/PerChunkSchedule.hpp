/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

/** Shared device cases of the per-chunk transform schedule.
 *
 * The per-backend translation units (`test_per_chunk_execution_{cuda,hip,sycl}.cpp`) own the
 * device factory and the `TEST_CASE` declarations; the payload, the functors and the matrices
 * live here so every backend runs the identical comparison.
 */
namespace graphFlow::test::perChunk
{
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

    inline std::uint32_t expectedValue(std::uint64_t id)
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

    template<typename TDevice, typename TStorage>
    void writeChunk(TDevice& device, TStorage& storage, ChunkHandle handle, std::vector<Pod> const& host)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(host.size())});
        alpaka::onHost::memcpy(queue, view, host);
    }

    template<typename TDevice, typename TStorage>
    std::vector<Pod> readChunk(TDevice& device, TStorage& storage, ChunkHandle handle, std::uint32_t fill)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        std::vector<Pod> host(fill);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(fill)});
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    template<typename TDevice>
    struct StageRun
    {
        /** Output payloads indexed by batch-global output position. */
        std::vector<Pod> byPosition;
        std::uint32_t launches = 0u;
        std::uint32_t outputChunks = 0u;
    };

    /** Run one transform stage over explicit chunk fills and collect the batch-global output. */
    template<typename TDevice>
    StageRun<TDevice> runStage(
        TDevice& device,
        ChunkKernelSchedule schedule,
        std::uint32_t capacity,
        std::uint32_t tileSize,
        std::vector<std::uint32_t> const& fills)
    {
        using Storage = PayloadSlotStorage<Pod, TDevice>;
        using Runner = GraphRunner<TDevice, Storage>;

        std::uint32_t const particles = std::accumulate(fills.begin(), fills.end(), 0u);
        std::uint32_t const outputChunks = (particles + capacity - 1u) / capacity;
        Storage storage{device, static_cast<std::uint32_t>(fills.size()) + outputChunks + 2u, capacity};

        Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = static_cast<std::uint32_t>(fills.size());
        config.tileSize = tileSize;
        config.frameCap = 2u;
        config.chunkKernelSchedule = schedule;
        config.chunkQueueDepth = 1u;

        detail::TransformStage<TDevice, Storage, Pod, AddIdValue> stage{device, storage, config, AddIdValue{}};

        typename Runner::Batch batch;
        batch.firstId = 0u;
        std::uint64_t position = 0u;
        for(std::uint32_t const fill : fills)
        {
            ChunkHandle const handle = storage.acquire();
            std::vector<Pod> input(fill);
            for(std::uint32_t i = 0u; i < fill; ++i)
                input[i] = Pod{position + i, 0xFFFF'FFFFu};
            writeChunk<TDevice>(device, storage, handle, input);
            batch.chunks.push_back(ChunkDescriptor{handle, fill, 0u});
            position += fill;
        }

        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        stage.launch(queue, batch);
        drain(device, queue);
        std::uint32_t allocated = 0u;
        CHECK_FALSE(stage.advance(queue, batch, allocated));
        CHECK(allocated == outputChunks);

        std::vector<ChunkDescriptor> outputs;
        stage.finish(batch, outputs);
        REQUIRE(outputs.size() == outputChunks);

        StageRun<TDevice> run;
        run.byPosition.resize(particles);
        for(std::uint32_t k = 0u; k < outputChunks; ++k)
        {
            std::uint32_t const fill = outputs[k].fill;
            std::vector<Pod> const read = readChunk<TDevice>(device, storage, outputs[k].handle, fill);
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
    template<typename TDevice>
    std::size_t wrongPositions(StageRun<TDevice> const& run)
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
    template<typename TDevice>
    void checkEqualAndExact(StageRun<TDevice> const& off, StageRun<TDevice> const& on)
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

    /** Tile widths a device can launch, sorted; 512 is skipped below its block limit. */
    template<typename TDevice>
    std::vector<std::uint32_t> tileSizesFor(TDevice const& device)
    {
        auto const properties = device.getDeviceProperties();
        std::vector<std::uint32_t> tiles{32u, 128u};
        if(properties.maxThreadsPerBlock >= 512u)
            tiles.push_back(512u);
        return tiles;
    }

    /** A device whose block limit is below 512 must reject a 512-wide launch instead of narrowing it. */
    template<typename TDevice>
    void checkWideTileRejection(TDevice& device)
    {
        auto const properties = device.getDeviceProperties();
        if(properties.maxThreadsPerBlock >= 512u)
            return;
        CHECK_THROWS_AS(launchPolicy(device, 1024u, 512u, 2u), std::invalid_argument);

        ChunkScheduleConfig config;
        config.threadsPerBlock = 512u;
        CHECK_THROWS_AS(makeChunkSchedulePolicy(properties, 1ull << 30, config), std::invalid_argument);
    }

    /** Off/on equality, Q=1 and Q>=8, partial tails, over the C/T matrix of one device. */
    template<typename TDevice>
    void runStageMatrix(TDevice& device)
    {
        struct Case
        {
            std::uint32_t capacity;
            std::vector<std::uint32_t> fills;
        };

        std::vector<Case> const cases{
            // Q=1: one chunk per batch, including the degenerate capacities.
            {1u, {1u}},
            {31u, {31u}},
            {127u, {127u}},
            {128u, {128u}},
            {8192u, {8192u}},
            // Q>=8: full chunks.
            {31u, std::vector<std::uint32_t>(8u, 31u)},
            {127u, std::vector<std::uint32_t>(8u, 127u)},
            {128u, std::vector<std::uint32_t>(8u, 128u)},
            {8192u, std::vector<std::uint32_t>(8u, 8192u)},
            // Q>=8 with a partial tail: the tail owns its launch and fill.
            {128u, {128u, 128u, 128u, 5u}},
            {8192u, {8192u, 8192u, 8192u, 8192u, 8192u, 8192u, 8192u, 5760u}},
            // A 5000-particle first chunk shifts every later position: the second chunk's
            // [5000, 13192) range crosses the 8192 output boundary.
            {8192u, {5000u, 8192u, 8192u, 8192u, 8192u, 8192u, 8192u, 3000u}},
            // Eight one-particle chunks: the smallest capacity still gets one launch per chunk.
            {1u, std::vector<std::uint32_t>(8u, 1u)},
        };

        for(Case const& testCase : cases)
        {
            for(std::uint32_t const tileSize : tileSizesFor(device))
            {
                CAPTURE(testCase.capacity, tileSize, testCase.fills.size());
                StageRun<TDevice> const off
                    = runStage<TDevice>(device, ChunkKernelSchedule::off, testCase.capacity, tileSize, testCase.fills);
                StageRun<TDevice> const on = runStage<TDevice>(
                    device,
                    ChunkKernelSchedule::perChunk,
                    testCase.capacity,
                    tileSize,
                    testCase.fills);
                checkEqualAndExact(off, on);
                CHECK(off.launches == 1u);
                CHECK(on.launches == static_cast<std::uint32_t>(testCase.fills.size()));
            }
        }
    }

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
        ALPAKA_FN_ACC UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Pod& payload) const
        {
            if(bin == 1u)
                payload.value += 1u;
            return UpdateResult::success;
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

    /** Run source -> transform -> partition -> sink on the device and return one value per id. */
    template<typename TDevice>
    std::vector<std::uint32_t> runFlow(
        TDevice& device,
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t tileSize,
        std::uint32_t batchChunks,
        ChunkKernelSchedule schedule)
    {
        auto results
            = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::max<std::uint64_t>(particles, 1u)});

        Flow<Pod, TDevice> flow{device};
        auto source = flow.source("generate", GeneratePod{});
        auto transform = flow.transform("add", AddIdValue{});
        auto parity = flow.partition("parity", ParityClassifier{}, AddOnOddBin{});
        using ResultView = decltype(results.getView());
        auto sink = flow.sink("record", RecordPod<ResultView>{results.getView()});
        flow.connect(source, transform);
        flow.connect(transform, parity);
        flow.connect(parity.template out<0u>(), sink);
        flow.connect(parity.template out<1u>(), sink);

        Tuning tuning;
        tuning.chunkCapacity = capacity;
        tuning.tileSize = tileSize;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        tuning.chunkKernelSchedule = schedule;
        tuning.chunkQueueDepth = 4u;
        flow.configure(tuning);
        flow.run(source, particles);

        std::vector<std::uint32_t> host(static_cast<std::size_t>(particles));
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, results);
        return host;
    }

    /** End-to-end off/on equality through `Flow` at Q=1 and Q>=8 on one device. */
    template<typename TDevice>
    void runFlowMatrix(TDevice& device)
    {
        std::vector<std::uint32_t> const tiles = tileSizesFor(device);
        for(std::uint64_t const particles : {17ull, 4097ull})
        {
            for(std::uint32_t const capacity : {31u, 127u, 128u})
            {
                for(std::uint32_t const tileSize : tiles)
                {
                    for(std::uint32_t const batchChunks : {1u, 8u})
                    {
                        CAPTURE(particles, capacity, tileSize, batchChunks);
                        std::vector<std::uint32_t> const off = runFlow<TDevice>(
                            device,
                            particles,
                            capacity,
                            tileSize,
                            batchChunks,
                            ChunkKernelSchedule::off);
                        std::vector<std::uint32_t> const on = runFlow<TDevice>(
                            device,
                            particles,
                            capacity,
                            tileSize,
                            batchChunks,
                            ChunkKernelSchedule::perChunk);
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
} // namespace graphFlow::test::perChunk
