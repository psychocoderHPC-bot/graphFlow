/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "HookAdapters.hpp"
#include "ParticleStages.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/flow/Status.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace graphFlow::test::failure
{
    /** Small payload used by the typed failure checks. */
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

    /** Return a bin outside the declared two-bin range without touching the status word. */
    struct OutOfRangeClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            [[maybe_unused]] Pod const& payload) const
        {
            return maxBins;
        }
    };

    struct ValidClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % maxBins);
        }
    };

    /** Report an invalid updater result through both the status word and the return value. */
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

    /** Four bins so one full chunk needs four output chunks. */
    struct FourBinClassifier
    {
        static constexpr std::uint32_t maxBins = 4u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % maxBins);
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

    /** Sixteen distinct bins exceed the eight entries of `HashBinTable<4u>`. */
    struct ManyBinsPolicy
    {
        static constexpr std::uint32_t maxBins = 16u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return static_cast<std::uint32_t>(id % maxBins);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            [[maybe_unused]] std::uint32_t bin,
            [[maybe_unused]] graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
        }
    };

    inline constexpr std::uint32_t sentinel = 0xDEAD'BEEFu;

    inline graphFlow::Config configFor(std::uint32_t capacity, std::uint32_t batchChunks)
    {
        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = batchChunks;
        config.tileSize = capacity;
        config.frameCap = 1u;
        return config;
    }

    template<typename TDevice, typename TView>
    void initializeSentinel(TDevice& device, TView view)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(queue, view, sentinel);
    }

    template<typename TDevice, typename TView>
    std::vector<std::uint32_t> readValues(TDevice& device, TView view, std::size_t size)
    {
        std::vector<std::uint32_t> host(size);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    template<typename TDevice, typename TStorage>
    void writeChunk(TDevice& device, TStorage& storage, graphFlow::ChunkHandle handle, std::vector<Pod> const& host)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(host.size())});
        alpaka::onHost::memcpy(queue, view, host);
    }

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }

    template<typename TStorage>
    void checkNoOutstandingLeases(TStorage const& storage)
    {
        for(std::uint32_t index = 0u; index < storage.numChunks(); ++index)
            CHECK(storage.leaseCount(graphFlow::ChunkHandle{index, storage.generation(index)}) == 0u);
    }

    /** An out-of-range classifier result rejects the batch before any output chunk is planned. */
    template<typename TDevice>
    void checkInvalidClassifierResult(TDevice& device)
    {
        using Device = TDevice;
        using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
        constexpr std::uint32_t capacity = 4u;

        Storage storage{device, 8u, capacity};
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        graphFlow::Config const config = configFor(capacity, 1u);
        graphFlow::detail::PartitionStage<Device, Storage, Pod, OutOfRangeClassifier, graphFlow::NoUpdate>
            stage{device, storage, config, OutOfRangeClassifier{}, graphFlow::NoUpdate{}};

        graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
        writeChunk(
            device,
            storage,
            input.handle,
            std::vector<Pod>{Pod{0u, 0u}, Pod{1u, 0u}, Pod{2u, 0u}, Pod{3u, 0u}});
        using Batch = typename graphFlow::GraphRunner<Device, Storage>::Batch;
        Batch const batch{{input}, 0u};

        stage.launch(queue, batch);
        drain(device, queue);

        std::uint32_t allocated = 0u;
        CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);
        CHECK(allocated == 0u);
        CHECK(storage.freeCount() == 7u);

        stage.abort();
        CHECK(storage.freeCount() == 7u);
        std::vector<graphFlow::ChunkDescriptor> outputs;
        stage.finish(batch, outputs);
        CHECK(outputs.empty());
        CHECK_THROWS_AS(stage.launch(queue, Batch{}), std::logic_error);

        storage.release(input.handle);
        CHECK(storage.freeCount() == storage.numChunks());
    }

    /** An invalid updater result returns the planned chunks exactly once. */
    template<typename TDevice>
    void checkInvalidUpdaterResult(TDevice& device)
    {
        using Device = TDevice;
        using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
        constexpr std::uint32_t capacity = 4u;

        Storage storage{device, 12u, capacity};
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        graphFlow::Config const config = configFor(capacity, 1u);
        graphFlow::detail::PartitionStage<Device, Storage, Pod, ValidClassifier, InvalidUpdate>
            stage{device, storage, config, ValidClassifier{}, InvalidUpdate{}};

        graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
        writeChunk(
            device,
            storage,
            input.handle,
            std::vector<Pod>{Pod{0u, 0u}, Pod{1u, 0u}, Pod{2u, 0u}, Pod{3u, 0u}});
        using Batch = typename graphFlow::GraphRunner<Device, Storage>::Batch;
        Batch const batch{{input}, 0u};

        stage.launch(queue, batch);
        drain(device, queue);
        std::uint32_t allocated = 0u;
        REQUIRE(stage.advance(queue, batch, allocated));
        CHECK(allocated == 2u);
        CHECK(storage.freeCount() == 9u);
        drain(device, queue);
        CHECK_THROWS_AS(stage.advance(queue, batch, allocated), graphFlow::BatchError);

        // The failed advance returns the planned chunks; abort must not release them again.
        CHECK(storage.freeCount() == 11u);
        stage.abort();
        CHECK(storage.freeCount() == 11u);
        std::vector<graphFlow::ChunkDescriptor> outputs;
        stage.finish(batch, outputs);
        CHECK(outputs.empty());
        CHECK_THROWS_AS(stage.launch(queue, Batch{}), std::logic_error);

        storage.release(input.handle);
        CHECK(storage.freeCount() == storage.numChunks());
    }

    /** A full hash table aborts the batch without a partial sink write. */
    template<typename TDevice>
    void checkHashTableFull(TDevice& device)
    {
        using Device = TDevice;
        constexpr std::uint32_t capacity = 16u;
        constexpr std::uint64_t particles = 16u;
        constexpr std::uint32_t numChunks = 64u;

        graphFlow::ChunkPool<Device> pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        initializeSentinel(device, result.getView());
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memset(queue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
        graphFlow::particle::TransformStage<Device, ManyBinsPolicy, graphFlow::HashBinTable<4u>>
            transform{device, pool, ManyBinsPolicy{}, capped, 128u};
        graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

        std::vector<int> route(ManyBinsPolicy::maxBins, 2);
        graphFlow::GraphRunner<Device> runner{device, pool};
        auto const source = runner.addNode(
            "generate",
            graphFlow::NodeRole::source,
            1u,
            {1},
            graphFlow::particle::hooksOf<Device>(generate));
        runner.addNode(
            "hash transform",
            graphFlow::NodeRole::transform,
            ManyBinsPolicy::maxBins,
            route,
            graphFlow::particle::hooksOf<Device>(transform));
        auto const sinkNode
            = runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
        runner.setSource(source, particles);
        runner.setStallTimeout(std::chrono::milliseconds{10000});

        CHECK_THROWS_AS(runner.run(), graphFlow::BatchError);
        CHECK(pool.freeCount() == pool.numChunks());
        CHECK(pool.peakLive() <= pool.numChunks());
        CHECK(runner.nodeCompleted(sinkNode) == 0u);
        checkNoOutstandingLeases(pool);
        for(std::uint32_t const value : readValues(device, result.getView(), particles))
            CHECK(value == sentinel);
    }

    /** Pool exhaustion during host planning releases every staged chunk. */
    template<typename TDevice>
    void checkPoolExhaustion(TDevice& device)
    {
        using Device = TDevice;
        using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
        constexpr std::uint32_t capacity = 4u;

        // One input chunk plus exactly three output chunks: the four-bin plan fails on the last one.
        Storage storage{device, 4u, capacity};
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        graphFlow::Config const config = configFor(capacity, 1u);
        graphFlow::detail::PartitionStage<Device, Storage, Pod, FourBinClassifier, graphFlow::NoUpdate>
            stage{device, storage, config, FourBinClassifier{}, graphFlow::NoUpdate{}};

        graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
        writeChunk(
            device,
            storage,
            input.handle,
            std::vector<Pod>{Pod{0u, 0u}, Pod{1u, 0u}, Pod{2u, 0u}, Pod{3u, 0u}});
        using Batch = typename graphFlow::GraphRunner<Device, Storage>::Batch;
        Batch const batch{{input}, 0u};

        stage.launch(queue, batch);
        drain(device, queue);

        std::uint32_t allocated = 0u;
        CHECK_THROWS_AS(stage.advance(queue, batch, allocated), std::runtime_error);
        CHECK(allocated == 0u);
        // The three chunks acquired before the pool ran dry are back; only the input is live.
        CHECK(storage.freeCount() == 3u);

        stage.abort();
        CHECK(storage.freeCount() == 3u);
        std::vector<graphFlow::ChunkDescriptor> outputs;
        stage.finish(batch, outputs);
        CHECK(outputs.empty());
        CHECK_THROWS_AS(stage.launch(queue, Batch{}), std::logic_error);

        storage.release(input.handle);
        CHECK(storage.freeCount() == storage.numChunks());
    }

    /** A completion-event enqueue failure aborts one in-flight batch exactly once. */
    template<typename TDevice>
    void checkEnqueueAbort(TDevice& device)
    {
        using Device = TDevice;
        using Storage = graphFlow::PayloadSlotStorage<Pod, Device>;
        constexpr std::uint32_t capacity = 8u;
        constexpr std::uint64_t particles = 8u;
        constexpr std::uint32_t numChunks = 16u;

        Storage storage{device, numChunks, capacity};
        auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        initializeSentinel(device, results.getView());
        using ResultView = decltype(results.getView());

        graphFlow::Config const config = configFor(capacity, 1u);
        using SourceStage = graphFlow::detail::SourceStage<Device, Storage, Pod, GeneratePod>;
        using SinkStage = graphFlow::detail::SinkStage<Device, Storage, Pod, RecordPod<ResultView>>;
        auto source = std::make_unique<SourceStage>(device, storage, config, GeneratePod{});
        auto sink = std::make_unique<SinkStage>(device, storage, config, RecordPod<ResultView>{results.getView()});

        using Base = graphFlow::detail::StageBase<Device, Storage>;
        std::uint32_t abortCalls = 0u;
        auto sourceHooks = graphFlow::detail::makeHooks<Device, Storage>(static_cast<Base*>(source.get()));
        sourceHooks = graphFlow::test::wrapAbort<Device, Storage>(
            std::move(sourceHooks),
            [&abortCalls](auto const& sourceAbort)
            {
                ++abortCalls;
                sourceAbort();
            });

        using Runner = graphFlow::GraphRunner<Device, Storage>;
        Runner runner{device, storage};
        auto const sourceNode = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        auto const sinkNode = runner.addNode(
            "sink",
            graphFlow::NodeRole::sink,
            0u,
            {},
            graphFlow::detail::makeHooks<Device, Storage>(static_cast<Base*>(sink.get())));
        runner.setSource(sourceNode, particles);
        runner.setStallTimeout(std::chrono::milliseconds{5000});
        // The generate launch is already enqueued when the completion-event enqueue fails, so
        // teardown has to drain the queue and abort the in-flight batch.
        runner.setFailureHook(
            [](graphFlow::FailurePoint point, std::uint32_t node, std::uint32_t item)
            {
                if(point == graphFlow::FailurePoint::eventEnqueue && node == 0u && item == 1u)
                    throw std::runtime_error{"injected source event enqueue failure"};
            });

        bool failed = false;
        try
        {
            runner.run();
        }
        catch(std::runtime_error const&)
        {
            failed = true;
        }
        CHECK(failed);
        CHECK(abortCalls == 1u);
        CHECK(storage.freeCount() == storage.numChunks());
        CHECK(runner.nodeCompleted(sinkNode) == 0u);
        checkNoOutstandingLeases(storage);
        for(std::uint32_t const value : readValues(device, results.getView(), particles))
            CHECK(value == sentinel);
    }
} // namespace graphFlow::test::failure
