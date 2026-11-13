/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
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
    std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);

    std::uint64_t const particles = 16u;
    std::uint32_t const capacity = 8u;
    // P equals one chunk, so admission is blocked while the first generated batch is in flight and
    // the second source launch is the chunk routed back from the transform.
    std::uint32_t const live = 8u;
    std::uint32_t const numChunks = graphFlow::particle::derivePoolSize(particles, capacity, live, 2u);

    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});

    graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<Device, AllToBinZeroPolicy> transform{
        device,
        pool,
        AllToBinZeroPolicy{},
        alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}}),
        128u};
    graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

    std::uint32_t launches = 0u;
    auto sourceHooks = graphFlow::particle::hooksOf<Device>(generate);
    auto const generateLaunch = sourceHooks.launch;
    sourceHooks.launch = [generateLaunch, &launches](Runner::Queue const& queue, Runner::Batch const& batch)
    {
        ++launches;
        if(launches == 2u)
            throw graphFlow::BatchError{"source launch failed on purpose"};
        generateLaunch(queue, batch);
    };

    Runner runner{device, pool};
    auto const source
        = runner.addNode("generate", graphFlow::NodeRole::source, 1u, {1}, std::move(sourceHooks));
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
    std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);

    // Two chunks of capacity four: the input holds one, leaving room for exactly one of the two
    // output chunks the even split needs.
    graphFlow::ChunkPool pool{device, 2u, 4u};
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    alpaka::onHost::memset(queue, capped, std::uint8_t{0});

    graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<Device, SplitEvenPolicy> stage{
        device,
        pool,
        SplitEvenPolicy{},
        capped,
        128u};

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
        std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);

        constexpr std::uint32_t capacity = 8u;
        constexpr std::uint64_t particles = 8u;
        constexpr std::uint32_t numChunks = 32u;
        graphFlow::ChunkPool<Device> pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
        graphFlow::particle::TransformStage<Device, SplitEvenPolicy> split{
            device,
            pool,
            SplitEvenPolicy{},
            capped,
            128u,
            2u};
        graphFlow::particle::TransformStage<Device, SplitEvenPolicy> merge{
            device,
            pool,
            SplitEvenPolicy{},
            capped,
            128u,
            2u};
        graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

        BatchingFailureOutcome outcome;
        // Count abort invocations test-side. A second abort would attempt to release the already
        // returned pending chunks; its stale-handle error is swallowed by teardown, so the counter
        // is the observable that proves the exactly-once contract.
        auto mergeHooks = graphFlow::particle::hooksOf<Device>(merge);
        auto const mergeAbort = mergeHooks.abort;
        mergeHooks.abort = [&outcome, mergeAbort]()
        {
            ++outcome.abortCalls;
            mergeAbort();
        };

        Runner runner{device, pool};
        auto const source = runner.addNode(
            "generate",
            graphFlow::NodeRole::source,
            1u,
            {1},
            graphFlow::particle::hooksOf<Device>(generate));
        runner.addNode(
            "split",
            graphFlow::NodeRole::transform,
            2u,
            {2, 2},
            graphFlow::particle::hooksOf<Device>(split));
        runner.addNode(
            "merge",
            graphFlow::NodeRole::transform,
            2u,
            {3, 3},
            std::move(mergeHooks));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
        runner.setSource(source, particles);
        runner.setBatchChunks(2u);
        runner.setStallTimeout(std::chrono::milliseconds{5'000});
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
            outcome.leases.push_back(
                pool.leaseCount(graphFlow::ChunkHandle{index, pool.generation(index)}));
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
    std::uint32_t const threads = Runner::threadsPerBlock(device, 128u);

    constexpr std::uint32_t capacity = 4u;
    constexpr std::uint64_t particles = 4u;
    constexpr std::uint32_t numChunks = 16u;
    graphFlow::ChunkPool<Device> pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<Device, SplitEvenPolicy> transform{
        device,
        pool,
        SplitEvenPolicy{},
        capped,
        128u};
    graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

    auto transformHooks = graphFlow::particle::hooksOf<Device>(transform);
    auto const baseFinish = transformHooks.finish;
    transformHooks.finish = [baseFinish](
                                Runner::Batch const& batch,
                                std::vector<graphFlow::ChunkDescriptor>& outputs)
    {
        baseFinish(batch, outputs);
        if(outputs.size() < 2u)
            throw std::runtime_error{"test expects two output chunks"};
        // The first descriptor routes fine; the second one is unroutable for a two-bin route table.
        outputs[1].bin = 200u;
    };

    Runner runner{device, pool};
    auto const source = runner.addNode(
        "generate",
        graphFlow::NodeRole::source,
        1u,
        {1},
        graphFlow::particle::hooksOf<Device>(generate));
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        2u,
        {2, 2},
        std::move(transformHooks));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, particles);
    runner.setStallTimeout(std::chrono::milliseconds{10'000});

    CHECK_THROWS_AS(runner.run(), std::runtime_error);

    // The input chunk must still be released even though the first output was routed.
    CHECK(pool.freeCount() == pool.numChunks());
}
