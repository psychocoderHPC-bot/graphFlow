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
    Runner::Batch const batch{chunk, 0u};
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
    CHECK_THROWS_AS(stage.advance(queue, chunk, allocated), std::runtime_error);

    // The partially acquired output chunk is back; only the input is still live.
    CHECK(pool.freeCount() == 1u);
    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(chunk, outputs);
    CHECK(outputs.empty());

    // The failed batch marks the stage single-use, so stale planning state cannot be replayed.
    CHECK_THROWS_AS(stage.launch(queue, batch), std::logic_error);

    pool.release(chunk.handle);
    CHECK(pool.freeCount() == pool.numChunks());
}
