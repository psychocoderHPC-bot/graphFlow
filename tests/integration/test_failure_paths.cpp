/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "HookAdapters.hpp"
#include "ParticleFeedback.hpp"
#include "ParticleStages.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/flow/Flow.hpp"
#include "graphFlow/flow/Status.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    /** Route every particle into bin 0; the test graph wires bin 0 back to the source node. */
    struct AllToBinZeroPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            [[maybe_unused]] std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return 0u;
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

    /** Split a batch evenly over two bins so a two-chunk pool fails on the second output. */
    struct SplitEvenPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

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
} // namespace

TEST_CASE("a throwing source launch fails the run without stalling", "[integration][failure]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    using Runner = graphFlow::GraphRunner<Device>;

    std::uint64_t const particles = 16u;
    std::uint32_t const capacity = 8u;
    // P equals one chunk, so admission is blocked while the first generated batch is in flight and
    // the second source launch is the chunk routed back from the transform.
    std::uint32_t const live = 8u;
    std::uint32_t const numChunks = graphFlow::particle::derivePoolSize(particles, capacity, live, 2u);

    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});

    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
    graphFlow::particle::TransformStage<Device, AllToBinZeroPolicy> transform{
        device,
        pool,
        AllToBinZeroPolicy{},
        alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}}),
        128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

    std::uint32_t launches = 0u;
    auto sourceHooks = graphFlow::test::wrapLaunch<Device, graphFlow::ChunkPool<Device>>(
        graphFlow::particle::hooksOf<Device>(generate),
        [&launches](auto const& generateLaunch, auto const& queue, auto const& batch)
        {
            ++launches;
            if(launches == 2u)
                throw graphFlow::BatchError{"source launch failed on purpose"};
            generateLaunch(queue, batch);
        });

    Runner runner{device, pool};
    auto const source = runner.addNode("generate", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        2u,
        {0, 2}, // bin 0 feeds back into the source node, bin 1 goes to the sink
        graphFlow::particle::hooksOf<Device>(transform));
    auto const sinkNode
        = runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, particles, live);
    runner.setStallTimeout(std::chrono::milliseconds{500});

    std::string message;
    try
    {
        runner.run();
    }
    catch(graphFlow::BatchError const& error)
    {
        message = error.what();
    }
    // The injected batch error must surface; a stall timeout would report scheduler state instead.
    CHECK(message == "source launch failed on purpose");
    // The failing launch is the queued output dispatched back into the source, not an admission.
    CHECK(launches == 2u);
    CHECK(sink.sunkParticles() == 0u);
    CHECK(runner.nodeCompleted(sinkNode) == 0u);

    // Every chunk is released exactly once on the error path.
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLive() <= pool.numChunks());
}

TEST_CASE("a failed transform batch poisons the stage and restores the pool", "[integration][failure][stage]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    using Runner = graphFlow::GraphRunner<Device>;

    // Two chunks of capacity four: the input holds one, leaving room for exactly one of the two
    // output chunks the even split needs.
    graphFlow::ChunkPool pool{device, 2u, 4u};
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    alpaka::onHost::memset(queue, capped, std::uint8_t{0});

    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
    graphFlow::particle::TransformStage<Device, SplitEvenPolicy> stage{device, pool, SplitEvenPolicy{}, capped, 128u};

    graphFlow::ChunkDescriptor const chunk{pool.acquire(), 4u, 0u};
    Runner::Batch const batch{{chunk}, 0u};
    generate.launch(queue, batch);
    alpaka::onHost::wait(queue);

    stage.launch(queue, batch);
    // The mapped metadata copy runs as a deferred host task, so drain the queue before the host
    // planning step reads the counters.
    auto event = device.makeEvent();
    queue.enqueue(event);
    queue.waitFor(event);
    alpaka::onHost::wait(queue);

    std::uint32_t allocated = 0u;
    CHECK_THROWS_AS(stage.advance(queue, batch, allocated), std::runtime_error);

    // The partially acquired output chunk is back; only the input is still live.
    CHECK(pool.freeCount() == 1u);
    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    CHECK(outputs.empty());

    // The failed batch marks the stage single-use, so stale planning state cannot be replayed.
    CHECK_THROWS_AS(stage.launch(queue, batch), std::logic_error);

    pool.release(chunk.handle);
    CHECK(pool.freeCount() == pool.numChunks());
}

namespace
{
    struct BatchingFailureOutcome
    {
        bool threw = false;
        std::uint32_t numChunks = 0u;
        std::uint32_t freeChunks = 0u;
        std::uint32_t abortCalls = 0u;
        std::vector<std::uint32_t> leases;
    };

    /** Run source -> split -> merge -> sink with Q=2 and an injected failure hook.
     *
     * Node ids are fixed by construction: 0 generate, 1 split, 2 merge, 3 sink.
     */
    BatchingFailureOutcome runBatchingFailure(
        std::function<void(graphFlow::FailurePoint, std::uint32_t, std::uint32_t)> hook)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        using Runner = graphFlow::GraphRunner<Device>;

        constexpr std::uint32_t capacity = 8u;
        constexpr std::uint64_t particles = 8u;
        constexpr std::uint32_t numChunks = 32u;
        graphFlow::ChunkPool<Device> pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
        graphFlow::particle::TransformStage<Device, SplitEvenPolicy>
            split{device, pool, SplitEvenPolicy{}, capped, 128u, 2u};
        graphFlow::particle::TransformStage<Device, SplitEvenPolicy>
            merge{device, pool, SplitEvenPolicy{}, capped, 128u, 2u};
        graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

        BatchingFailureOutcome outcome;
        // Count abort invocations test-side. A second abort would attempt to release the already
        // returned pending chunks; its stale-handle error is swallowed by teardown, so the counter
        // is the observable that proves the exactly-once contract.
        auto mergeHooks = graphFlow::test::wrapAbort<Device, graphFlow::ChunkPool<Device>>(
            graphFlow::particle::hooksOf<Device>(merge),
            [&outcome](auto const& mergeAbort)
            {
                ++outcome.abortCalls;
                mergeAbort();
            });

        Runner runner{device, pool};
        auto const source = runner.addNode(
            "generate",
            graphFlow::NodeRole::source,
            1u,
            {1},
            graphFlow::particle::hooksOf<Device>(generate));
        runner
            .addNode("split", graphFlow::NodeRole::transform, 2u, {2, 2}, graphFlow::particle::hooksOf<Device>(split));
        runner.addNode("merge", graphFlow::NodeRole::transform, 2u, {3, 3}, std::move(mergeHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
        runner.setSource(source, particles);
        runner.setBatchChunks(2u);
        runner.setStallTimeout(std::chrono::milliseconds{5000});
        runner.setFailureHook(std::move(hook));

        try
        {
            runner.run();
        }
        catch(std::runtime_error const&)
        {
            outcome.threw = true;
        }
        outcome.numChunks = pool.numChunks();
        outcome.freeChunks = pool.freeCount();
        outcome.leases.reserve(pool.numChunks());
        for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
            outcome.leases.push_back(pool.leaseCount(graphFlow::ChunkHandle{index, pool.generation(index)}));
        return outcome;
    }
} // namespace

TEST_CASE("event enqueue failure releases a planned batch", "[integration][failure][batching]")
{
    // The second merge event follows the scatter, so host planning has already acquired the output
    // chunks. Teardown must hand them back instead of leaking them.
    BatchingFailureOutcome const outcome = runBatchingFailure(
        [](graphFlow::FailurePoint point, std::uint32_t node, std::uint32_t item)
        {
            if(point == graphFlow::FailurePoint::eventEnqueue && node == 2u && item == 2u)
                throw std::runtime_error{"injected event enqueue failure"};
        });
    CHECK(outcome.threw);
    CHECK(outcome.abortCalls == 1u); // exactly one abort, so the pending chunks are released once
    CHECK(outcome.freeChunks == outcome.numChunks);
    for(std::uint32_t const leases : outcome.leases)
        CHECK(leases == 0u);
}

TEST_CASE("lease acquisition failure rolls back acquired leases", "[integration][failure][batching]")
{
    // The merge batch carries two chunks; the second lease acquisition fails after the first
    // succeeded, so the runner must release that lease before teardown.
    BatchingFailureOutcome const outcome = runBatchingFailure(
        [](graphFlow::FailurePoint point, std::uint32_t node, std::uint32_t item)
        {
            if(point == graphFlow::FailurePoint::leaseAcquire && node == 2u && item == 1u)
                throw std::runtime_error{"injected lease failure"};
        });
    CHECK(outcome.threw);
    CHECK(outcome.abortCalls == 0u); // the failing dispatch never acquired output chunks
    CHECK(outcome.freeChunks == outcome.numChunks);
    for(std::uint32_t const leases : outcome.leases)
        CHECK(leases == 0u);
}

TEST_CASE("a later unroutable output still releases the batch input", "[integration][failure][routing]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    using Runner = graphFlow::GraphRunner<Device>;

    constexpr std::uint32_t capacity = 4u;
    constexpr std::uint64_t particles = 4u;
    constexpr std::uint32_t numChunks = 16u;
    graphFlow::ChunkPool<Device> pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
    graphFlow::particle::TransformStage<Device, SplitEvenPolicy>
        transform{device, pool, SplitEvenPolicy{}, capped, 128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

    auto transformHooks = graphFlow::test::wrapFinish<Device, graphFlow::ChunkPool<Device>>(
        graphFlow::particle::hooksOf<Device>(transform),
        [](auto const& baseFinish, auto const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        {
            baseFinish(batch, outputs);
            if(outputs.size() < 2u)
                throw std::runtime_error{"test expects two output chunks"};
            // The first descriptor routes fine; the second one is unroutable for a two-bin route table.
            outputs[1].bin = 200u;
        });

    Runner runner{device, pool};
    auto const source = runner.addNode(
        "generate",
        graphFlow::NodeRole::source,
        1u,
        {1},
        graphFlow::particle::hooksOf<Device>(generate));
    runner.addNode("transform", graphFlow::NodeRole::transform, 2u, {2, 2}, std::move(transformHooks));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, particles);
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    CHECK_THROWS_AS(runner.run(), std::runtime_error);

    // The input chunk must still be released even though the first output was routed.
    CHECK(pool.freeCount() == pool.numChunks());
}

namespace
{
    struct StatusPod
    {
        std::uint64_t id = 0u;
        std::uint32_t value = 0u;
    };

    struct GenerateStatusPod
    {
        ALPAKA_FN_ACC StatusPod operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return StatusPod{id, 0u};
        }
    };

    struct RejectStatusPod
    {
        ALPAKA_FN_ACC void operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Status& status,
            [[maybe_unused]] StatusPod& payload) const
        {
            status.set(graphFlow::StatusCode::invalid);
        }
    };

    struct BumpStatusPod
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, StatusPod& payload) const
        {
            ++payload.value;
        }
    };

    struct CappedClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            [[maybe_unused]] StatusPod const& payload) const
        {
            // A capped history is a legal named outcome, not a device failure.
            return 1u;
        }
    };

    template<typename TView>
    struct RecordStatusPod
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, StatusPod const& payload) const
        {
            values[payload.id] = payload.value;
        }
    };

    constexpr std::uint32_t statusSentinel = 0xDEAD'BEEFu;

    template<typename TView>
    void initializeSentinel(decltype(graphFlow::test::makeCpuDevice())& device, TView view)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(queue, view, statusSentinel);
    }

    template<typename TDevice, typename TView>
    std::vector<std::uint32_t> readValues(TDevice& device, TView view, std::size_t size)
    {
        std::vector<std::uint32_t> host(size);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    graphFlow::Tuning statusTuning(std::uint32_t batchChunks)
    {
        graphFlow::Tuning tuning;
        tuning.chunkCapacity = 8u;
        tuning.tileSize = 8u;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        return tuning;
    }

    struct TypedFailureOutcome
    {
        bool threw = false;
        std::uint32_t numChunks = 0u;
        std::uint32_t freeChunks = 0u;
        std::uint32_t abortCalls = 0u;
        std::uint32_t sinkCompleted = 0u;
        std::vector<std::uint32_t> leases;
        std::vector<std::uint32_t> values;
    };

    template<typename TTransform>
    TypedFailureOutcome runTypedStatusFailure(
        std::uint64_t particles,
        std::uint32_t batchChunks,
        std::uint32_t capacity = 8u,
        std::uint32_t numChunks = 64u)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        using Storage = graphFlow::PayloadSlotStorage<StatusPod, Device>;

        Storage storage{device, numChunks, capacity};
        auto results
            = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::max<std::uint64_t>(particles, 1u)});
        initializeSentinel(device, results.getView());
        using ResultView = decltype(results.getView());

        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = batchChunks;
        config.tileSize = capacity;
        config.frameCap = 1u;

        using SourceStage = graphFlow::detail::SourceStage<Device, Storage, StatusPod, GenerateStatusPod>;
        using TransformStage = graphFlow::detail::TransformStage<Device, Storage, StatusPod, TTransform>;
        using SinkStage = graphFlow::detail::SinkStage<Device, Storage, StatusPod, RecordStatusPod<ResultView>>;
        auto source = std::make_unique<SourceStage>(device, storage, config, GenerateStatusPod{});
        auto transform = std::make_unique<TransformStage>(device, storage, config, TTransform{});
        auto sink
            = std::make_unique<SinkStage>(device, storage, config, RecordStatusPod<ResultView>{results.getView()});
        auto* const transformPtr = transform.get();

        using Base = graphFlow::detail::StageBase<Device, Storage>;
        auto sourceHooks = graphFlow::detail::makeHooks<Device, Storage>(static_cast<Base*>(source.get()));
        auto transformHooks = graphFlow::detail::makeHooks<Device, Storage>(static_cast<Base*>(transform.get()));
        auto sinkHooks = graphFlow::detail::makeHooks<Device, Storage>(static_cast<Base*>(sink.get()));

        TypedFailureOutcome outcome;
        auto wrappedTransformHooks = graphFlow::test::wrapAbort<Device, Storage>(
            std::move(transformHooks),
            [&outcome](auto const& transformAbort)
            {
                ++outcome.abortCalls;
                transformAbort();
            });

        using Runner = graphFlow::GraphRunner<Device, Storage>;
        Runner runner{device, storage};
        auto const sourceNode = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
        runner.addNode("transform", graphFlow::NodeRole::transform, 1u, {2}, std::move(wrappedTransformHooks));
        auto const sinkNode = runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, std::move(sinkHooks));
        runner.setSource(sourceNode, particles);
        runner.setBatchChunks(batchChunks);
        runner.setStallTimeout(std::chrono::milliseconds{5000});

        try
        {
            runner.run();
        }
        catch(std::runtime_error const&)
        {
            outcome.threw = true;
        }
        outcome.numChunks = storage.numChunks();
        outcome.freeChunks = storage.freeCount();
        outcome.sinkCompleted = runner.nodeCompleted(sinkNode);
        outcome.leases.reserve(storage.numChunks());
        for(std::uint32_t index = 0u; index < storage.numChunks(); ++index)
            outcome.leases.push_back(storage.leaseCount(graphFlow::ChunkHandle{index, storage.generation(index)}));
        outcome.values
            = readValues(device, results.getView(), static_cast<std::size_t>(std::max<std::uint64_t>(particles, 1u)));

        if(outcome.threw)
        {
            // The failed stage is single-use after teardown.
            auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
            using RunnerBatch = typename Runner::Batch;
            CHECK_THROWS_AS(transformPtr->launch(queue, RunnerBatch{}), std::logic_error);
        }
        return outcome;
    }
} // namespace

TEST_CASE("typed flow fails before publishing a status-rejected batch", "[integration][failure][status]")
{
    for(std::uint64_t particles : {0ull, 1ull, 17ull})
    {
        for(std::uint32_t batchChunks : {1u, 2u})
        {
            CAPTURE(particles, batchChunks);
            TypedFailureOutcome const outcome = runTypedStatusFailure<RejectStatusPod>(particles, batchChunks);
            if(particles == 0u)
            {
                CHECK_FALSE(outcome.threw);
                CHECK(outcome.sinkCompleted == 0u);
                CHECK(outcome.abortCalls == 0u);
            }
            else
            {
                CHECK(outcome.threw);
                CHECK(outcome.abortCalls == 1u);
                CHECK(outcome.sinkCompleted == 0u);
            }
            CHECK(outcome.freeChunks == outcome.numChunks);
            for(std::uint32_t const leases : outcome.leases)
                CHECK(leases == 0u);
            for(std::uint32_t const value : outcome.values)
                CHECK(value == statusSentinel);
        }
    }
}

TEST_CASE("typed flow retries a failed status batch without stale output", "[integration][failure][status]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    constexpr std::uint64_t particles = 17u;
    auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    initializeSentinel(device, results.getView());

    graphFlow::Flow<StatusPod, Device> flow{device};
    auto source = flow.source("generate", GenerateStatusPod{});
    auto transform = flow.transform("reject", RejectStatusPod{});
    using View = decltype(results.getView());
    auto sink = flow.sink("record", RecordStatusPod<View>{results.getView()});
    flow.connect(source, transform);
    flow.connect(transform, sink);
    flow.configure(statusTuning(1u));

    CHECK_THROWS_AS(flow.run(source, particles), graphFlow::BatchError);
    for(std::uint32_t const value : readValues(device, results.getView(), particles))
        CHECK(value == statusSentinel);

    // A retry rebuilds storage and stages. The failed run must not have cached an output to republish.
    CHECK_THROWS_AS(flow.run(source, particles), graphFlow::BatchError);
    for(std::uint32_t const value : readValues(device, results.getView(), particles))
        CHECK(value == statusSentinel);
}

TEST_CASE("a capped outcome remains routable and does not fail the batch", "[integration][failure][status]")
{
    for(std::uint32_t batchChunks : {1u, 2u})
    {
        CAPTURE(batchChunks);
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        constexpr std::uint64_t particles = 17u;
        auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        initializeSentinel(device, results.getView());

        graphFlow::Flow<StatusPod, Device> flow{device};
        auto source = flow.source("generate", GenerateStatusPod{});
        auto route = flow.partition("route", CappedClassifier{});
        auto bump = flow.transform("bump", BumpStatusPod{});
        using View = decltype(results.getView());
        auto sink = flow.sink("record", RecordStatusPod<View>{results.getView()});
        flow.connect(source, route);
        flow.connect(route.out<0u>(), bump);
        flow.connect(route.out<1u>(), bump);
        flow.connect(bump, sink);
        flow.configure(statusTuning(batchChunks));

        CHECK_NOTHROW(flow.run(source, particles));
        for(std::uint32_t const value : readValues(device, results.getView(), particles))
            CHECK(value == 1u);
    }
}
