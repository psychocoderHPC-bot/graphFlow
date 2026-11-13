/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

TEST_CASE("bounded pool completes with N much larger than P", "[integration][pressure]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.liveParticles = 256u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;
    parameters.stallTimeoutMs = 10'000u;

    graphFlow::particle::FeedbackResult const result
        = graphFlow::particle::runParticleFeedback(device, parameters);

    // The acceptance contract still holds under the bounded working set.
    graphFlow::test::ScalarOutcome const reference
        = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);
    CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
    CHECK(result.weights == reference.weights);
    CHECK(result.sunkParticles == parameters.numParticles);

    // The pool is derived from P, not from N: P data chunks + K*R output credits. The plan's
    // ceil(P/C)+K*R = 8 is a lower bound that deadlocks once split routing fragments chunks; the
    // regression records the sound bound (see derivePoolSize and README.md).
    std::uint32_t const expectedPool
        = graphFlow::particle::derivePoolSize(parameters.numParticles, parameters.chunkSize, 256u, 2u);
    CHECK(expectedPool == 256u + 3u * 2u);
    CHECK(result.stats.poolChunks == expectedPool);
    CHECK(result.stats.peakLiveChunks <= expectedPool);
    CHECK(result.stats.peakLiveChunks < parameters.numParticles);
    CHECK(result.liveParticles == 256u);
}

namespace
{
    /** Wrap the production transform hooks and assert the input chunk was still live after the
     * completion event. A premature release would let a source acquire the chunk and bump its
     * generation, so `isLive` would fail here. */
    template<typename TDevice>
    typename graphFlow::GraphRunner<TDevice>::Hooks checkedHooks(
        graphFlow::ChunkPool<TDevice>& pool,
        graphFlow::particle::TransformStage<TDevice, graphFlow::particle::ProcessBPolicy>& stage,
        graphFlow::ChunkHandle& recorded)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;
        typename Runner::Hooks base = graphFlow::particle::hooksOf<TDevice>(stage);
        return typename Runner::Hooks{
            [&, base](typename Runner::Queue const& queue, typename Runner::Batch const& batch)
            {
                recorded = batch.chunk.handle;
                base.launch(queue, batch);
            },
            [&, base](typename Runner::Queue const& queue, graphFlow::ChunkDescriptor const& chunk, std::uint32_t& allocated)
            {
                CHECK(pool.isLive(recorded));
                return base.advance(queue, chunk, allocated);
            },
            [base](graphFlow::ChunkDescriptor const& chunk, std::vector<graphFlow::ChunkDescriptor>& outputs)
            { base.finish(chunk, outputs); }};
    }
} // namespace

TEST_CASE("input chunks stay live until their scatter completes", "[integration][pressure][lease]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    std::uint32_t const threads = graphFlow::GraphRunner<Device>::threadsPerBlock(device, 128u);

    std::uint64_t const particles = 64u;
    std::uint32_t const capacity = 8u;
    std::uint32_t const live = 16u;
    std::uint32_t const numChunks = graphFlow::particle::derivePoolSize(particles, capacity, live, 2u);

    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::Parameters parameters{};
    parameters.seed = 7u;
    parameters.maxVisits = 8u;

    graphFlow::particle::GenerateStage<Device> generate{pool, parameters.seed, threads};
    graphFlow::particle::ProcessBPolicy policy{parameters.seed, parameters.maxVisits};
    graphFlow::particle::TransformStage<Device, graphFlow::particle::ProcessBPolicy> transform{
        device,
        pool,
        policy,
        capped,
        threads};
    graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

    graphFlow::ChunkHandle recorded{};
    graphFlow::GraphRunner<Device> runner{device, pool};
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
        {2, 1}, // bin 1 feeds back into the same transform
        checkedHooks<Device>(pool, transform, recorded));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, particles, live);
    runner.setStallTimeout(std::chrono::milliseconds{10'000});
    runner.run();

    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLeases() >= 1u);
    CHECK(pool.peakLive() <= pool.numChunks());
    for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
        CHECK(pool.leaseCount(graphFlow::ChunkHandle{index, pool.generation(index)}) == 0u);
}
