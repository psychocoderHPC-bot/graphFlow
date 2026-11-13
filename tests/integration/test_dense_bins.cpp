/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ParticleStages.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
    /** 64-bin classifier that tags the payload with its bin so the sink can verify it. */
    struct Bin64Policy
    {
        static constexpr std::uint32_t maxBins = 64u;

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
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            // The sink writes the weight, so encode the bin there for verification.
            particle.weight = bin;
        }
    };

    /** Classifier that violates its contract for every particle. */
    struct InvalidPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            [[maybe_unused]] std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return maxBins; // out of range on purpose
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

    struct Dense64Outcome
    {
        std::vector<std::uint32_t> weights;
        std::uint64_t sunkParticles = 0u;
        std::uint32_t sourceBatches = 0u;
        graphFlow::SchedulerStats stats;
    };

    /** Run the 64-bin pipeline. `liveParticles == 0` keeps the unbounded-admission sizing. */
    Dense64Outcome runDense64(
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t liveParticles,
        std::uint32_t tileSize = 128u)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);

        std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, Bin64Policy::maxBins);
        // With a bounded P the pool only needs P data chunks plus this graph's single transform
        // reserve; without P the historical generous sizing applies.
        std::uint64_t const poolChunks = liveParticles == 0u
                                             ? particles / capacity + 2u * reserve + 8u + 2u
                                             : static_cast<std::uint64_t>(liveParticles) + reserve;
        auto const numChunks = static_cast<std::uint32_t>(poolChunks);

        graphFlow::ChunkPool pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});

        graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
        graphFlow::particle::TransformStage<Device, Bin64Policy> transform{
            device,
            pool,
            Bin64Policy{},
            // The 64-bin test does not exercise the capped counter; a one-word buffer is enough.
            alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}}),
            tileSize};
        graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

        std::vector<int> route(Bin64Policy::maxBins, 2);
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
            Bin64Policy::maxBins,
            route,
            graphFlow::particle::hooksOf<Device>(transform));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
        runner.setSource(source, particles, liveParticles);
        runner.setStallTimeout(std::chrono::milliseconds{10'000});
        auto const stats = runner.run();

        Dense64Outcome outcome;
        outcome.weights.assign(static_cast<std::size_t>(particles), 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, outcome.weights, result);
        }
        outcome.sunkParticles = sink.sunkParticles();
        outcome.sourceBatches = runner.nodeCompleted(source);
        outcome.stats = stats;
        return outcome;
    }

    /** Sorted multiset the sink must produce: particle id modulo the bin count. */
    std::vector<std::uint32_t> expectedDense64(std::uint64_t particles)
    {
        std::vector<std::uint32_t> expected(static_cast<std::size_t>(particles));
        for(std::uint64_t id = 0u; id < particles; ++id)
            expected[static_cast<std::size_t>(id)] = static_cast<std::uint32_t>(id % Bin64Policy::maxBins);
        std::sort(expected.begin(), expected.end());
        return expected;
    }
} // namespace

TEST_CASE("dense pipeline handles 64 bins with partial chunks", "[integration][dense]")
{
    std::uint64_t const particles = 1000u;
    Dense64Outcome const outcome = runDense64(particles, 17u, 0u);

    REQUIRE(outcome.sunkParticles == particles);
    CHECK(outcome.stats.completedBatches != 0u);

    // The contract is multiset equality: a different valid scatter order must not fail the test.
    // The result buffer is id-indexed, so the per-id set is still checked through the sorted
    // comparison below without assuming a particular tile or chunk order.
    std::vector<std::uint32_t> host = outcome.weights;
    std::sort(host.begin(), host.end());
    CHECK(host == expectedDense64(particles));
}

TEST_CASE("dense pipeline handles 64 bins with a bounded particle budget", "[integration][dense]")
{
    std::uint64_t const particles = 1000u;
    std::uint32_t const capacity = 17u;
    // P below one chunk: every admitted fill is capped at P, so the gate is directly observable.
    std::uint32_t const liveParticles = 10u;
    Dense64Outcome const outcome = runDense64(particles, capacity, liveParticles);

    REQUIRE(outcome.sunkParticles == particles);

    std::vector<std::uint32_t> host = outcome.weights;
    std::sort(host.begin(), host.end());
    CHECK(host == expectedDense64(particles));

    // With P=10 the source must admit at least ceil(N/P)=100 chunks carrying at most 10 particles
    // each. Unbounded admission would need only ceil(N/capacity)=59, so this checks the P gate.
    CHECK(outcome.sourceBatches >= graphFlow::ceilDivChecked(particles, liveParticles));
    CHECK(outcome.sourceBatches > graphFlow::ceilDivChecked(particles, capacity));

    // The derived pool is P data chunks plus one transform's output credits.
    std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, Bin64Policy::maxBins);
    CHECK(outcome.stats.poolChunks == liveParticles + reserve);
    CHECK(outcome.stats.peakLiveChunks <= outcome.stats.poolChunks);
}

TEST_CASE("dense pipeline tiles a chunk larger than T", "[integration][dense]")
{
    // Both cases launch eight blocks for their single chunk. With C=1024 and 1024 particles
    // the chunk is exactly eight full tiles; with C=1000 the last tile holds only 104
    // particles.
    struct Case
    {
        std::uint64_t particles;
        std::uint32_t capacity;
    };
    for(Case const& testCase : {Case{1024u, 1024u}, Case{1000u, 1000u}})
    {
        CAPTURE(testCase.particles, testCase.capacity);
        Dense64Outcome const outcome = runDense64(testCase.particles, testCase.capacity, 0u, 128u);
        REQUIRE(outcome.sunkParticles == testCase.particles);

        std::vector<std::uint32_t> const expected = expectedDense64(testCase.particles);
        std::vector<std::uint32_t> host = outcome.weights;
        std::sort(host.begin(), host.end());
        CHECK(host == expected); // multiset equality, independent of tile order
        CHECK(
            graphFlow::particle::histogramOf(outcome.weights)
            == graphFlow::particle::histogramOf(expected));
    }
}

TEST_CASE("invalid classifier results abort the batch", "[integration][dense]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);

    graphFlow::ChunkPool pool{device, 32u, 8u};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{16u}});

    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
    graphFlow::particle::TransformStage<Device, InvalidPolicy> transform{
        device,
        pool,
        InvalidPolicy{},
        alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}}),
        128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

    graphFlow::GraphRunner<Device> runner{device, pool};
    auto const source = runner.addNode(
        "generate",
        graphFlow::NodeRole::source,
        1u,
        {1},
        graphFlow::particle::hooksOf<Device>(generate));
    runner.addNode(
        "invalid",
        graphFlow::NodeRole::transform,
        InvalidPolicy::maxBins,
        {2, 2},
        graphFlow::particle::hooksOf<Device>(transform));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, 16u);

    CHECK_THROWS_AS(runner.run(), graphFlow::BatchError);

    // The exception path must release the in-flight input and every queued chunk exactly once.
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLive() <= pool.numChunks());
}
