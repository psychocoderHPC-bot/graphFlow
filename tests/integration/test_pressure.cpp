/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ParticleStages.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
    /** Select the accelerator compiled into this build, or the CPU backend otherwise.
     *
     * A CUDA/HIP build must have a visible device: the test throws instead of silently falling
     * back to the host backend, so a device-less accelerator run fails rather than skips.
     */
    auto makeNativeTestDevice()
    {
#if ALPAKA_LANG_CUDA
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"CUDA build requires a visible NVIDIA GPU"};
        return selector.makeDevice(0u);
#elif ALPAKA_LANG_HIP
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::hip, alpaka::deviceKind::amdGpu});
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"HIP build requires a visible AMD GPU"};
        return selector.makeDevice(0u);
#else
        return graphFlow::test::makeCpuDevice();
#endif
    }
} // namespace

TEST_CASE("bounded pool completes with N much larger than P", "[integration][pressure]")
{
    auto const device = makeNativeTestDevice();
    constexpr std::uint32_t expectedPool = 256u + 8u;

    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = 4097u;
        parameters.chunkSize = 128u;
        parameters.liveParticles = 256u;
        parameters.seed = 42u;
        parameters.maxVisits = 64u;
        parameters.batchChunks = batchChunks;

        CAPTURE(batchChunks);
        graphFlow::particle::FeedbackResult const result
            = graphFlow::particle::runParticleFeedback(device, parameters);

        // The acceptance contract still holds under the bounded working set.
        graphFlow::test::ScalarOutcome const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);
        CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
        CHECK(result.weights == reference.weights);
        CHECK(result.sunkParticles == parameters.numParticles);

        // The pool is derived from P, not from N or Q. `Flow` reserves one output chunk per
        // registered bin: filter 2 + A 3 + B 3 = 8, so S = P + 8.
        CHECK(result.stats.poolChunks == expectedPool);
        CHECK(result.stats.peakLiveChunks <= expectedPool);
        CHECK(result.stats.peakReservedChunks <= expectedPool);
        CHECK(result.stats.peakLiveChunks < parameters.numParticles);
        CHECK(result.stats.sourceParticles == parameters.numParticles);
        CHECK(result.stats.sourceGenerated == parameters.numParticles);
        CHECK(result.stats.sourceSunk == parameters.numParticles);
        CHECK(result.stats.sourceLiveParticles == 0u);
        CHECK(result.liveParticles == 256u);
    }
}

TEST_CASE("small P below C drains a fragmented pool across Q", "[integration][pressure][small-P]")
{
    auto const device = makeNativeTestDevice();

    for(std::uint32_t const liveParticles : {1u, 3u, 8u})
    {
        for(std::uint32_t const batchChunks : {1u, 8u})
        {
            graphFlow::particle::Parameters parameters{};
            parameters.numParticles = 131u;
            parameters.chunkSize = 128u;
            parameters.liveParticles = liveParticles;
            parameters.seed = 7u;
            parameters.maxVisits = 64u;
            parameters.batchChunks = batchChunks;

            CAPTURE(liveParticles, batchChunks);
            graphFlow::particle::FeedbackResult const result
                = graphFlow::particle::runParticleFeedback(device, parameters);
            graphFlow::test::ScalarOutcome const reference
                = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

            CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
            CHECK(result.weights == reference.weights);
            CHECK(result.sunkParticles == parameters.numParticles);
            CHECK(result.cappedParticles == reference.cappedParticles);

            // P < C makes every source chunk min(C, P, remaining) particles; the pool still
            // follows S = P + 8 and completes through many admission rounds.
            CHECK(result.stats.poolChunks == liveParticles + 8u);
            CHECK(result.stats.peakLiveChunks <= result.stats.poolChunks);
            CHECK(result.stats.sourceGenerated == parameters.numParticles);
            CHECK(result.stats.sourceSunk == parameters.numParticles);
            CHECK(result.stats.sourceLiveParticles == 0u);
            CHECK(result.liveParticles == liveParticles);
        }
    }
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
                recorded = batch.chunk().handle;
                base.launch(queue, batch);
            },
            [&,
             base](typename Runner::Queue const& queue, typename Runner::Batch const& batch, std::uint32_t& allocated)
            {
                CHECK(pool.isLive(recorded));
                return base.advance(queue, batch, allocated);
            },
            [base](typename Runner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
            { base.finish(batch, outputs); },
            [base]()
            {
                if(base.abort)
                    base.abort();
            }};
    }
} // namespace

TEST_CASE("input chunks stay live until their scatter completes", "[integration][pressure][lease]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);

    std::uint64_t const particles = 64u;
    std::uint32_t const capacity = 8u;
    std::uint32_t const live = 16u;
    std::uint32_t const numChunks = graphFlow::particle::derivePoolSize(particles, capacity, live, 2u);

    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::Parameters parameters{};
    parameters.seed = 7u;
    parameters.maxVisits = 8u;

    graphFlow::particle::GenerateStage<Device> generate{device, pool, parameters.seed, 128u};
    graphFlow::particle::ProcessBPolicy policy{parameters.seed, parameters.maxVisits};
    graphFlow::particle::TransformStage<Device, graphFlow::particle::ProcessBPolicy>
        transform{device, pool, policy, capped, 128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

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
    runner.setStallTimeout(std::chrono::milliseconds{10000});
    runner.run();

    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLeases() >= 1u);
    CHECK(pool.peakLive() <= pool.numChunks());
    for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
        CHECK(pool.leaseCount(graphFlow::ChunkHandle{index, pool.generation(index)}) == 0u);
}

TEST_CASE("a tight pool runs at capacity and reuses every chunk", "[integration][pressure][pool]")
{
    auto device = makeNativeTestDevice();
    using Device = ALPAKA_TYPEOF(device);

    std::uint64_t const particles = 64u;
    std::uint32_t const capacity = 8u;
    std::uint32_t const live = 16u;
    // Deliberately below the derived S = P + K*R. Two source chunks (P = 16, C = 8) plus the
    // in-flight transform make this pool tight: five chunks carry data and the sixth is the
    // minimum output-credit headroom a transform needs to make progress. A finished batch can
    // release before the next plan acquires its outputs, so the reserve chunk may stay unallocated
    // and the live peak legitimately stops at five. The full-capacity boundary is probed
    // explicitly below instead of being inferred from the scheduler's live peak.
    std::uint32_t const poolChunks = 6u;

    graphFlow::ChunkPool pool{device, poolChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, graphFlow::particle::missingWeight);
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::Parameters parameters{};
    parameters.seed = 7u;
    parameters.maxVisits = 8u;

    graphFlow::particle::GenerateStage<Device> generate{device, pool, parameters.seed, 128u};
    graphFlow::particle::ProcessBPolicy policy{parameters.seed, parameters.maxVisits};
    graphFlow::particle::TransformStage<Device, graphFlow::particle::ProcessBPolicy>
        transform{device, pool, policy, capped, 128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

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
        graphFlow::particle::hooksOf<Device>(transform));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, particles, live);
    runner.setStallTimeout(std::chrono::milliseconds{10000});
    graphFlow::SchedulerStats const stats = runner.run();

    // The tight pool completed without a leak. With two transform lanes the second scatter may
    // allocate its outputs while the first lane still holds its input, or after that input was
    // released, so the live peak legitimately varies by one and only the capacity bound is an
    // invariant of the run.
    CHECK(stats.poolChunks == poolChunks);
    CHECK(stats.peakLiveChunks <= poolChunks);
    CHECK(stats.peakReservedChunks >= 2u); // the transform reserved its output credit
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.liveCount() == 0u);
    CHECK(pool.peakLeases() >= 1u);

    // Snapshot the run's generations before the capacity probe. The probe releases every chunk
    // once, so checking reuse afterwards would only observe the probe itself. A chunk that never
    // becomes the head of the free list stays untouched, so the run may legitimately leave some
    // chunks unused; the eight source batches still force a reuse among the chunks it did use.
    std::vector<std::uint32_t> runGenerations;
    runGenerations.reserve(pool.numChunks());
    std::uint64_t generationSum = 0u;
    std::uint32_t distinctUsed = 0u;
    for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
    {
        runGenerations.push_back(pool.generation(index));
        generationSum += runGenerations.back();
        distinctUsed += runGenerations.back() != 0u ? 1u : 0u;
        CHECK(pool.leaseCount(graphFlow::ChunkHandle{index, runGenerations.back()}) == 0u);
    }
    CHECK(distinctUsed != 0u);
    CHECK(generationSum > distinctUsed); // at least one chunk was handed out more than once

    // Probe the physical capacity directly. Acquiring every chunk deterministically observes the
    // full-capacity boundary and touches the reserve chunk the scheduler may leave unused instead
    // of asserting a timing-dependent scheduler peak.
    {
        std::vector<graphFlow::ChunkHandle> held;
        held.reserve(pool.numChunks());
        for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
            held.push_back(pool.acquire());
        CHECK(pool.freeCount() == 0u);
        CHECK(pool.liveCount() == pool.numChunks());
        CHECK(pool.peakLive() == pool.numChunks());
        for(graphFlow::ChunkHandle const handle : held)
            pool.release(handle);
    }
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.liveCount() == 0u);

    // Every probe acquisition and release advanced its chunk by exactly one generation, which
    // confirms that the probe covered all chunks and that a release invalidates the old handle.
    for(std::uint32_t index = 0u; index < pool.numChunks(); ++index)
        CHECK(pool.generation(index) == runGenerations[index] + 1u);

    // The sink writes by id, so every particle must have left the feedback loop.
    std::vector<std::uint32_t> weights(static_cast<std::size_t>(particles), 0u);
    {
        auto readQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(readQueue, weights, result);
    }
    for(std::uint32_t const weight : weights)
    {
        CHECK(weight != graphFlow::particle::missingWeight);
        CHECK(weight <= parameters.maxVisits);
    }
}
