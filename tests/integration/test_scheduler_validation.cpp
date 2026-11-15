/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/flow/Debug.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
    using Catch::Matchers::ContainsSubstring;

    /** Host-only source: publishes its admitted input chunk unchanged. */
    template<typename TRunner>
    typename TRunner::Hooks sourceHooks()
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t&) -> bool { return false; };
        hooks.finish = [](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        { outputs.push_back(batch.chunk()); };
        hooks.abort = [] {};
        return hooks;
    }

    /** Host-only sink: records the delivered particle count and publishes no output. */
    template<typename TRunner>
    typename TRunner::Hooks sinkHooks(std::uint64_t& sunk)
    {
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, typename TRunner::Batch const&) {};
        hooks.advance = [](auto const&, typename TRunner::Batch const&, std::uint32_t&) -> bool { return false; };
        hooks.finish = [&sunk](typename TRunner::Batch const& batch, std::vector<graphFlow::ChunkDescriptor>&)
        { sunk += batch.particles(); };
        hooks.abort = [] {};
        return hooks;
    }

    enum class TransformMode
    {
        conserve,
        dropOneParticle,
        oversizeFill,
        duplicateHandle
    };

    /** Host-only transform that acquires one output chunk and publishes descriptors per mode. */
    template<typename TRunner, typename TPool>
    typename TRunner::Hooks transformHooks(TPool& pool, TransformMode mode, std::uint32_t capacity)
    {
        using Batch = typename TRunner::Batch;
        typename TRunner::Hooks hooks;
        hooks.launch = [](auto const&, Batch const&) {};
        hooks.advance = [](auto const&, Batch const&, std::uint32_t& allocated) -> bool
        {
            allocated = 0u;
            return false;
        };
        hooks.finish = [&pool, mode, capacity](Batch const& batch, std::vector<graphFlow::ChunkDescriptor>& outputs)
        {
            std::uint32_t const particles = static_cast<std::uint32_t>(batch.particles());
            graphFlow::ChunkHandle const handle = pool.acquire();
            switch(mode)
            {
            case TransformMode::conserve:
                outputs.push_back(graphFlow::ChunkDescriptor{handle, particles, 0u});
                break;
            case TransformMode::dropOneParticle:
                outputs.push_back(graphFlow::ChunkDescriptor{handle, particles - 1u, 0u});
                break;
            case TransformMode::oversizeFill:
                outputs.push_back(graphFlow::ChunkDescriptor{handle, capacity + 1u, 0u});
                break;
            case TransformMode::duplicateHandle:
                {
                    // One acquired chunk referenced by two descriptors: the batch conserves its
                    // particles but schedules the same chunk incarnation for the sink twice.
                    REQUIRE(particles >= 2u);
                    std::uint32_t const first = particles / 2u;
                    outputs.push_back(graphFlow::ChunkDescriptor{handle, first, 0u});
                    outputs.push_back(graphFlow::ChunkDescriptor{handle, particles - first, 0u});
                    break;
                }
            }
        };
        hooks.abort = [] {};
        return hooks;
    }

    template<typename TDevice>
    graphFlow::SchedulerStats runValidationGraph(
        TDevice device,
        TransformMode mode,
        std::uint64_t particles,
        std::uint32_t capacity,
        std::uint32_t poolChunks,
        std::uint64_t& sunk)
    {
        using Runner = graphFlow::GraphRunner<TDevice>;
        graphFlow::ChunkPool<TDevice> pool{device, poolChunks, capacity};
        Runner runner{device, pool};
        auto const source = runner.addNode("source", graphFlow::NodeRole::source, 1u, {1}, sourceHooks<Runner>());
        runner.addNode(
            "transform",
            graphFlow::NodeRole::transform,
            1u,
            {2},
            transformHooks<Runner>(pool, mode, capacity));
        runner.addNode("sink", graphFlow::NodeRole::sink, 0u, {}, sinkHooks<Runner>(sunk));
        runner.setSource(source, particles);
        runner.setStallTimeout(std::chrono::milliseconds{10000});
        return runner.run();
    }
} // namespace

TEST_CASE("scheduler validation accepts a conserving graph", "[scheduler][debug]")
{
    auto device = graphFlow::test::makeCpuDevice();
    std::uint64_t sunk = 0u;
    graphFlow::SchedulerStats const stats = runValidationGraph(device, TransformMode::conserve, 16u, 8u, 16u, sunk);
    CHECK(stats.sourceGenerated == 16u);
    CHECK(stats.sourceSunk == 16u);
    CHECK(sunk == 16u);
}

TEST_CASE("scheduler validation rejects an out-of-range output fill", "[scheduler][debug]")
{
    if constexpr(!graphFlow::debug::checksEnabled)
    {
        SUCCEED("scheduler validation is compiled out");
        return;
    }
    auto device = graphFlow::test::makeCpuDevice();
    std::uint64_t sunk = 0u;
    CHECK_THROWS_WITH(
        runValidationGraph(device, TransformMode::oversizeFill, 8u, 8u, 16u, sunk),
        ContainsSubstring("scheduler validation") && ContainsSubstring("chunk fill"));
}

TEST_CASE("scheduler validation rejects a batch that loses particles", "[scheduler][debug]")
{
    if constexpr(!graphFlow::debug::checksEnabled)
    {
        SUCCEED("scheduler validation is compiled out");
        return;
    }
    auto device = graphFlow::test::makeCpuDevice();
    std::uint64_t sunk = 0u;
    CHECK_THROWS_WITH(
        runValidationGraph(device, TransformMode::dropOneParticle, 8u, 8u, 16u, sunk),
        ContainsSubstring("conserved"));
}

TEST_CASE("scheduler validation rejects a duplicated sink delivery", "[scheduler][debug]")
{
    if constexpr(!graphFlow::debug::checksEnabled)
    {
        SUCCEED("scheduler validation is compiled out");
        return;
    }
    auto device = graphFlow::test::makeCpuDevice();
    std::uint64_t sunk = 0u;
    CHECK_THROWS_WITH(
        runValidationGraph(device, TransformMode::duplicateHandle, 8u, 8u, 16u, sunk),
        ContainsSubstring("delivered more than once"));
}

TEST_CASE("scheduler validation is absent from release builds", "[scheduler][release]")
{
    if constexpr(graphFlow::debug::checksEnabled)
    {
        SUCCEED("debug validation is enabled");
        return;
    }
    // A lossy transform completes without a check; the release hot path trusts the stages.
    auto device = graphFlow::test::makeCpuDevice();
    std::uint64_t sunk = 0u;
    graphFlow::SchedulerStats const stats
        = runValidationGraph(device, TransformMode::dropOneParticle, 8u, 8u, 16u, sunk);
    CHECK(stats.sourceGenerated == 8u);
    CHECK(stats.sourceSunk == 7u);
    CHECK(sunk == 7u);
}
