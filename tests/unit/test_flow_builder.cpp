/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    struct Payload
    {
        std::uint64_t id = 0u;
        std::uint32_t weight = 0u;
    };

    enum class Outcome : std::uint32_t
    {
        sink,
        feedback,
        capped
    };

    using Flow = graphFlow::Flow<Payload, Device>;

    struct Generate
    {
        ALPAKA_FN_ACC Payload operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Payload{id, 0u};
        }
    };

    struct OneBinClassifier
    {
        static constexpr std::uint32_t maxBins = 1u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            [[maybe_unused]] Payload const& payload) const
        {
            return 0u;
        }
    };

    struct TwoBinClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Payload const& payload) const
        {
            return payload.weight == 0u ? 0u : 1u;
        }
    };

    struct OutcomeClassifier
    {
        static constexpr std::uint32_t maxBins = 3u;

        ALPAKA_FN_ACC Outcome operator()([[maybe_unused]] auto const& accelerator, Payload const& payload) const
        {
            if(payload.weight == 0u)
                return Outcome::sink;
            return payload.weight < 4u ? Outcome::feedback : Outcome::capped;
        }
    };

    struct BumpUpdate
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            Outcome bin,
            Payload& payload) const
        {
            if(bin == Outcome::feedback)
                ++payload.weight;
            return graphFlow::UpdateResult::success;
        }
    };

    struct BumpTransform
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Payload& payload) const
        {
            ++payload.weight;
        }
    };

    struct ReplacingTransform
    {
        ALPAKA_FN_ACC Payload operator()([[maybe_unused]] auto const& accelerator, Payload const& payload) const
        {
            return Payload{payload.id, payload.weight + 2u};
        }
    };

    struct RecordSink
    {
        ALPAKA_FN_ACC void operator()(
            [[maybe_unused]] auto const& accelerator,
            [[maybe_unused]] Payload const& payload) const
        {
        }
    };

    struct DirectSource
    {
        Payload operator()(std::uint64_t id) const
        {
            return Payload{id, 1u};
        }
    };
} // namespace

TEST_CASE("flow registers source, partition, transform and sink nodes", "[unit][flow][builder]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};

    auto source = flow.source("generate", Generate{});
    auto filter = flow.partition("filter", OneBinClassifier{});
    auto sink = flow.sink("record", RecordSink{});
    auto transform = flow.transform("touch", BumpTransform{});
    auto replace = flow.transform("replace", ReplacingTransform{});

    CHECK(source.valid());
    CHECK(source.name() == "generate");
    CHECK(filter.valid());
    CHECK(filter.name() == "filter");
    CHECK(sink.valid());
    CHECK(transform.valid());
    CHECK(replace.valid());
    static_assert(decltype(source)::maxBins == 1u);
    static_assert(decltype(filter)::maxBins == 1u);
    static_assert(decltype(transform)::maxBins == 1u);

    flow.connect(source, filter);
    flow.connect(filter, transform);
    flow.connect(transform, sink);
}

TEST_CASE("handles stay valid when more nodes are registered", "[unit][flow][handle]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};

    auto source = flow.source("generate", Generate{});
    auto sink = flow.sink("record", RecordSink{});
    flow.connect(source, sink);

    auto extra = flow.sink("extra", RecordSink{});
    auto partition = flow.partition("late", OutcomeClassifier{});

    CHECK(source.valid());
    CHECK(sink.valid());
    CHECK(extra.valid());
    CHECK(partition.valid());
    CHECK(source.name() == "generate");
}

TEST_CASE("named and numeric partition ports address the same bin", "[unit][flow][ports]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};

    auto source = flow.source("generate", Generate{});
    auto ingress = flow.transform("ingress", BumpTransform{});
    auto filter = flow.partition("filter", OutcomeClassifier{});
    auto sink = flow.sink("record", RecordSink{});

    static_assert(decltype(filter)::maxBins == 3u);
    static_assert(std::same_as<decltype(filter)::bin_type, Outcome>);

    flow.connect(source, ingress);
    flow.connect(ingress, filter);
    flow.connect(filter.out<Outcome::sink>(), sink);
    flow.connect(filter.out<1u>(), sink);
    flow.connect(filter.out<Outcome::capped>(), sink);

    CHECK_THROWS_AS(flow.connect(filter.out<0u>(), sink), std::invalid_argument);
    CHECK_THROWS_AS(flow.connect(filter.out<Outcome::sink>(), sink), std::invalid_argument);
}

TEST_CASE("flow rejects invalid connections", "[unit][flow][connect]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};
    Flow other{device};

    auto source = flow.source("generate", Generate{});
    auto sink = flow.sink("record", RecordSink{});
    auto otherSink = other.sink("other record", RecordSink{});

    // A sink has no output.
    CHECK_THROWS_AS(flow.connect(sink, sink), std::invalid_argument);
    // A route may not target the source.
    CHECK_THROWS_AS(flow.connect(source, source), std::invalid_argument);
    // Cross-graph handles are rejected while both graphs are alive.
    CHECK_THROWS_AS(flow.connect(source, otherSink), std::invalid_argument);
}

TEST_CASE("a handle expires with its owning flow", "[unit][flow][handle]")
{
    graphFlow::SourceHandle stale;
    graphFlow::NodeHandle staleSink;
    {
        auto device = graphFlow::test::makeCpuDevice();
        Flow flow{device};
        stale = flow.source("generate", Generate{});
        staleSink = flow.sink("record", RecordSink{});
        CHECK(stale.valid());
        CHECK(staleSink.valid());
    }

    CHECK_FALSE(stale.valid());
    CHECK_FALSE(staleSink.valid());
    CHECK_THROWS_AS(stale.out<0u>(), std::logic_error);

    auto device = graphFlow::test::makeCpuDevice();
    Flow live{device};
    auto sink = live.sink("record", RecordSink{});
    CHECK_THROWS_AS(live.connect(staleSink, sink), std::logic_error);
}

TEST_CASE("duplicate and missing routes are rejected before run", "[unit][flow][routes]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};

    auto source = flow.source("generate", Generate{});
    auto ingress = flow.transform("ingress", BumpTransform{});
    auto filter = flow.partition("filter", TwoBinClassifier{});
    auto sink = flow.sink("record", RecordSink{});

    flow.connect(source, ingress);
    flow.connect(ingress, filter);
    flow.connect(filter.out<0u>(), sink);
    CHECK_THROWS_AS(flow.connect(filter.out<0u>(), sink), std::invalid_argument);
    // Bin 1 is missing.
    CHECK_THROWS_AS(flow.run(source, 16u), std::invalid_argument);

    flow.connect(filter.out<1u>(), sink);
    CHECK_THROWS_AS(flow.run(source, 16u), graphFlow::ExecutionUnavailable);
}

TEST_CASE("run rejects foreign and non-source handles", "[unit][flow][run]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};
    Flow other{device};

    auto source = flow.source("generate", Generate{});
    auto sink = flow.sink("record", RecordSink{});
    flow.connect(source, sink);

    auto otherSource = other.source("other generate", Generate{});
    auto otherSink = other.sink("other record", RecordSink{});
    other.connect(otherSource, otherSink);

    CHECK_THROWS_AS(flow.run(otherSource, 1u), std::invalid_argument);
    CHECK_THROWS_AS(flow.run(sink, 1u), std::invalid_argument);

    graphFlow::Tuning tuning;
    tuning.chunkCapacity = 8u;
    tuning.liveParticles = 8u;
    tuning.tileSize = 1u;
    tuning.numChunks = 8u;
    tuning.poolPayloadBudgetBytes = 1u << 20;
    flow.configure(tuning);
    CHECK_THROWS_AS(flow.run(source, 8u), graphFlow::ExecutionUnavailable);
}

TEST_CASE("a flow has exactly one source", "[unit][flow][source]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};

    auto source = flow.source("generate", Generate{});
    CHECK_THROWS_AS(flow.source("second generate", Generate{}), std::logic_error);
    CHECK_THROWS_AS(flow.source("direct generate", DirectSource{}), std::logic_error);

    auto sink = flow.sink("record", RecordSink{});
    flow.connect(source, sink);
    CHECK_THROWS_AS(flow.run(source, 0u), graphFlow::ExecutionUnavailable);
}

TEST_CASE("tuning rejects zero-valued knobs", "[unit][flow][tuning]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow flow{device};

    auto reject = [&flow](auto setter)
    {
        graphFlow::Tuning tuning;
        setter(tuning);
        CHECK_THROWS_AS(flow.configure(tuning), std::invalid_argument);
    };

    reject([](graphFlow::Tuning& tuning) { tuning.batchChunks = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.chunkCapacity = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.liveParticles = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.tileSize = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.numChunks = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.frameCap = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.poolPayloadBudgetBytes = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.policyWaves = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.policyTargetBlocksPerSm = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.policyLogicalTileWidth = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.policyMaxChunkCapacity = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.policyLiveParticlesPerChunk = 0u; });
    reject([](graphFlow::Tuning& tuning) { tuning.policyMemoryBudgetBytes = 0u; });

    graphFlow::Tuning valid;
    valid.chunkCapacity = 16u;
    valid.liveParticles = 16u;
    valid.tileSize = 1u;
    valid.numChunks = 16u;
    valid.batchChunks = 1u;
    valid.frameCap = 1u;
    valid.poolPayloadBudgetBytes = 1u << 20;
    valid.diagnostics = true;
    CHECK_NOTHROW(flow.configure(valid));
    CHECK(flow.tuning().chunkCapacity == 16u);
    CHECK(flow.tuning().diagnostics);
}

TEST_CASE("a flow is movable and its handles follow the graph", "[unit][flow][move]")
{
    auto device = graphFlow::test::makeCpuDevice();
    Flow first{device};

    auto source = first.source("generate", Generate{});
    auto sink = first.sink("record", RecordSink{});
    first.connect(source, sink);

    Flow second{std::move(first)};
    CHECK(source.valid());
    CHECK(sink.valid());

    Flow third{device};
    auto oldSource = third.source("old generate", Generate{});
    third = std::move(second);
    CHECK_FALSE(oldSource.valid());
    CHECK(source.valid());
    CHECK_THROWS_AS(third.run(oldSource, 0u), std::logic_error);

    graphFlow::Tuning tuning;
    tuning.chunkCapacity = 8u;
    tuning.liveParticles = 8u;
    tuning.tileSize = 1u;
    tuning.numChunks = 8u;
    tuning.poolPayloadBudgetBytes = 1u << 20;
    third.configure(tuning);
    CHECK_THROWS_AS(third.run(source, 8u), graphFlow::ExecutionUnavailable);
}
