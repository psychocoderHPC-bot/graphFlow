/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

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
} // namespace

TEST_CASE("dense pipeline handles 64 bins with partial chunks", "[integration][dense]")
{
    auto device = graphFlow::test::makeCpuDevice();
    std::uint32_t const threads = graphFlow::GraphRunner<ALPAKA_TYPEOF(device)>::threadsPerBlock(device, 128u);

    std::uint64_t const particles = 1000u;
    std::uint32_t const capacity = 17u; // deliberately not a warp multiple
    std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, Bin64Policy::maxBins);
    std::uint32_t const numChunks = static_cast<std::uint32_t>(particles / capacity + 2u * reserve + 8u + 2u);

    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});

    graphFlow::particle::GenerateStage<ALPAKA_TYPEOF(device)> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<ALPAKA_TYPEOF(device), Bin64Policy> transform{
        device,
        pool,
        Bin64Policy{},
        // The 64-bin test does not exercise the capped counter; a one-word buffer is enough.
        alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}}),
        threads};
    graphFlow::particle::SinkStage<ALPAKA_TYPEOF(device)> sink{pool, result, threads};

    std::vector<int> route(Bin64Policy::maxBins, 2);
    graphFlow::GraphRunner<ALPAKA_TYPEOF(device)> runner{device, pool};
    auto const source = runner.addNode(
        "generate",
        graphFlow::NodeRole::source,
        1u,
        {1},
        graphFlow::particle::hooksOf<ALPAKA_TYPEOF(device)>(generate));
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        Bin64Policy::maxBins,
        route,
        graphFlow::particle::hooksOf<ALPAKA_TYPEOF(device)>(transform));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<ALPAKA_TYPEOF(device)>(sink));
    runner.setSource(source, particles);
    runner.run();

    REQUIRE(sink.sunkParticles() == particles);

    std::vector<std::uint32_t> host(particles, 0u);
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, result);
    }
    for(std::uint64_t id = 0u; id < particles; ++id)
    {
        CAPTURE(id);
        CHECK(host[id] == id % Bin64Policy::maxBins);
    }
}

TEST_CASE("invalid classifier results abort the batch", "[integration][dense]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    std::uint32_t const threads = graphFlow::GraphRunner<Device>::threadsPerBlock(device, 128u);

    graphFlow::ChunkPool pool{device, 32u, 8u};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{16u}});

    graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<Device, InvalidPolicy> transform{
        device,
        pool,
        InvalidPolicy{},
        alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}}),
        threads};
    graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

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
}
