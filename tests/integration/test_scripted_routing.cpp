/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "TestDevice.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace
{
    /** Philox-free scripted Filter: choose A (bin 0) or B (bin 1) by id and visit. */
    struct ScriptedFilterPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            std::uint32_t visit) const
        {
            if(id == 0u)
                return 0u; // A stops
            if(id == 1u)
                return visit == 0u ? 0u : 1u; // A survives, B stops
            if(id == 2u)
                return 1u; // B survives twice
            if(id == 3u)
                return visit == 0u ? 0u : (visit == 1u ? 1u : 0u); // A, B, A stops
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

    /** Philox-free scripted A: ids 1 and 3 survive the first A visit and feed back. */
    struct ScriptedAPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            std::uint32_t visit) const
        {
            if((id == 1u || id == 3u) && visit == 0u)
                return 1u; // feedback
            return 0u;     // sink
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            if(bin == 1u)
            {
                particle.weight += 1000u;
                particle.visit += 1u;
            }
        }
    };

    /** Philox-free scripted B: id 2 survives at visit 0, id 3 survives at visit 1. */
    struct ScriptedBPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            std::uint32_t visit) const
        {
            if(id == 2u && visit == 0u)
                return 1u;
            if(id == 3u && visit == 1u)
                return 1u;
            return 0u;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            if(bin == 1u)
            {
                particle.weight += 1u;
                particle.visit += 1u;
            }
        }
    };

    std::vector<std::uint32_t> runScripted(std::uint32_t capacity)
    {
        auto device = graphFlow::test::makeCpuDevice();
        using Device = ALPAKA_TYPEOF(device);
        std::uint32_t const threads = graphFlow::GraphRunner<Device>::threadsPerBlock(device, 128u);

        std::uint64_t const particles = 4u;
        std::uint32_t const live = graphFlow::particle::defaultLiveParticles(particles, capacity);
        std::uint32_t const numChunks
            = graphFlow::particle::derivePoolSize(particles, capacity, live, 2u);

        graphFlow::ChunkPool pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<Device> generate{pool, 42u, threads};
        graphFlow::particle::TransformStage<Device, ScriptedFilterPolicy> filter{
            device,
            pool,
            ScriptedFilterPolicy{},
            capped,
            threads};
        graphFlow::particle::TransformStage<Device, ScriptedAPolicy> processA{
            device,
            pool,
            ScriptedAPolicy{},
            capped,
            threads};
        graphFlow::particle::TransformStage<Device, ScriptedBPolicy> processB{
            device,
            pool,
            ScriptedBPolicy{},
            capped,
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
            "filter",
            graphFlow::NodeRole::transform,
            2u,
            {2, 3},
            graphFlow::particle::hooksOf<Device>(filter));
        runner.addNode(
            "A",
            graphFlow::NodeRole::transform,
            2u,
            {4, 1},
            graphFlow::particle::hooksOf<Device>(processA));
        runner.addNode(
            "B",
            graphFlow::NodeRole::transform,
            2u,
            {4, 1},
            graphFlow::particle::hooksOf<Device>(processB));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
        runner.setSource(source, particles, live);
        runner.run();

        std::vector<std::uint32_t> weights(particles, 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, weights, result);
        }
        return weights;
    }
} // namespace

TEST_CASE("scripted routing fixture matches the plan section 9.3 rows", "[integration][scripted]")
{
    std::vector<graphFlow::particle::HistogramRow> const expected{
        {0u, 1u},
        {1u, 1u},
        {1000u, 1u},
        {1001u, 1u}};

    for(std::uint32_t const capacity : {1u, 3u})
    {
        CAPTURE(capacity);
        std::vector<std::uint32_t> const weights = runScripted(capacity);
        CHECK(weights == std::vector<std::uint32_t>{0u, 1000u, 1u, 1001u});
        CHECK(graphFlow::particle::histogramOf(weights) == expected);
    }
}
