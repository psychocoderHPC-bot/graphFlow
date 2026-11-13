/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/GraphBuilder.hpp"
#include "graphFlow/graph/NodeBase.hpp"
#include "graphFlow/graph/RouteTable.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    /** Host-only node with no device work; enough to exercise topology and attachment. */
    struct DummyNode : graphFlow::NodeBase<Device>
    {
        using Base = graphFlow::NodeBase<Device>;
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        void submit(Queue const&, Batch const&) override
        {
        }

        void complete(Batch const&, std::vector<graphFlow::ChunkDescriptor>&) override
        {
        }
    };
} // namespace

TEST_CASE("graph builder assigns stable node ids and exposes topology", "[unit][graph][builder]")
{
    graphFlow::GraphBuilder<Device> builder;
    DummyNode source;
    DummyNode first;
    DummyNode second;
    DummyNode sink;

    auto const sourceId = builder.addNode(
        graphFlow::NodeRole::source,
        "source",
        1u,
        graphFlow::RouteTable{{1u}},
        source);
    auto const firstId = builder.addNode(
        graphFlow::NodeRole::transform,
        "first",
        2u,
        graphFlow::RouteTable{{2u, 2u}},
        first);
    auto const secondId = builder.addNode(
        graphFlow::NodeRole::transform,
        "second",
        2u,
        graphFlow::RouteTable{{3u, 1u}},
        second);
    auto const sinkId = builder.addNode(
        graphFlow::NodeRole::sink,
        "sink",
        0u,
        graphFlow::RouteTable{},
        sink);

    CHECK(sourceId == 0u);
    CHECK(firstId == 1u);
    CHECK(secondId == 2u);
    CHECK(sinkId == 3u);
    CHECK(builder.nodeCount() == 4u);

    builder.setSource(sourceId, 100u, 16u);
    auto const graph = builder.build();

    REQUIRE(graph.nodeCount() == 4u);
    CHECK(graph.source() == sourceId);
    CHECK(graph.numParticles() == 100u);
    CHECK(graph.liveParticles() == 16u);
    CHECK(graph.role(sourceId) == graphFlow::NodeRole::source);
    CHECK(graph.role(firstId) == graphFlow::NodeRole::transform);
    CHECK(graph.role(sinkId) == graphFlow::NodeRole::sink);
    CHECK(graph.name(firstId) == "first");
    CHECK(graph.bins(firstId) == 2u);
    CHECK(graph.routes(firstId).destination(0u) == 2u);
    CHECK(graph.routes(firstId).destination(1u) == 2u);
    CHECK(graph.routes(secondId).destination(1u) == 1u); // feedback edge to an earlier node
    CHECK(graph.routes(sinkId).empty());
    CHECK(&graph.node(firstId) == &first);
    CHECK(&graph.node(sinkId) == &sink);
}

TEST_CASE("a built graph is an immutable snapshot", "[unit][graph][builder]")
{
    graphFlow::GraphBuilder<Device> builder;
    DummyNode source;
    DummyNode sink;

    auto const sourceId = builder.addNode(
        graphFlow::NodeRole::source,
        "source",
        1u,
        graphFlow::RouteTable{{1u}},
        source);
    builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
    builder.setSource(sourceId, 7u);

    auto const graph = builder.build();
    REQUIRE(graph.nodeCount() == 2u);

    DummyNode extra;
    builder.addNode(graphFlow::NodeRole::sink, "extra", 0u, graphFlow::RouteTable{}, extra);

    CHECK(builder.nodeCount() == 3u);
    CHECK(graph.nodeCount() == 2u);
    CHECK(graph.source() == sourceId);
}

TEST_CASE("graph builder validates node roles and route tables", "[unit][graph][builder]")
{
    graphFlow::GraphBuilder<Device> builder;
    DummyNode node;

    CHECK_THROWS_AS(
        builder.addNode(graphFlow::NodeRole::source, "source", 0u, graphFlow::RouteTable{}, node),
        std::invalid_argument);
    CHECK_THROWS_AS(
        builder.addNode(graphFlow::NodeRole::transform, "transform", 2u, graphFlow::RouteTable{{1u}}, node),
        std::invalid_argument);
    CHECK_THROWS_AS(
        builder.addNode(graphFlow::NodeRole::sink, "sink", 1u, graphFlow::RouteTable{}, node),
        std::invalid_argument);
    CHECK_THROWS_AS(
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{{1u}}, node),
        std::invalid_argument);
    CHECK_THROWS_AS(
        builder.addNode(static_cast<graphFlow::NodeRole>(99), "invalid", 0u, graphFlow::RouteTable{}, node),
        std::invalid_argument);
    CHECK(builder.nodeCount() == 0u);
}

TEST_CASE("graph builder validates the configured source", "[unit][graph][builder]")
{
    graphFlow::GraphBuilder<Device> builder;
    DummyNode source;
    DummyNode transform;
    DummyNode sink;

    auto const sourceId = builder.addNode(
        graphFlow::NodeRole::source,
        "source",
        1u,
        graphFlow::RouteTable{{1u}},
        source);
    auto const transformId = builder.addNode(
        graphFlow::NodeRole::transform,
        "transform",
        1u,
        graphFlow::RouteTable{{2u}},
        transform);
    builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);

    CHECK_THROWS_AS(builder.setSource(transformId, 1u), std::invalid_argument);
    CHECK_THROWS_AS(builder.setSource(99u, 1u), std::out_of_range);

    builder.setSource(sourceId, 1u);
    CHECK_NOTHROW(builder.build());
}

TEST_CASE("graph builder rejects missing or ambiguous source", "[unit][graph][builder]")
{
    {
        graphFlow::GraphBuilder<Device> builder;
        DummyNode transform;
        builder.addNode(
            graphFlow::NodeRole::transform,
            "transform",
            1u,
            graphFlow::RouteTable{{0u}},
            transform);
        CHECK_THROWS_AS(builder.build(), std::logic_error); // no source
    }
    {
        graphFlow::GraphBuilder<Device> builder;
        DummyNode source;
        DummyNode sink;
        auto const sourceId = builder.addNode(
            graphFlow::NodeRole::source,
            "source",
            1u,
            graphFlow::RouteTable{{1u}},
            source);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        CHECK_THROWS_AS(builder.build(), std::logic_error); // setSource() missing
        builder.setSource(sourceId, 0u);
        CHECK_NOTHROW(builder.build());
    }
    {
        graphFlow::GraphBuilder<Device> builder;
        DummyNode source;
        DummyNode secondSource;
        DummyNode sink;
        auto const sourceId = builder.addNode(
            graphFlow::NodeRole::source,
            "source",
            1u,
            graphFlow::RouteTable{{3u}},
            source);
        builder.addNode(
            graphFlow::NodeRole::source,
            "second-source",
            1u,
            graphFlow::RouteTable{{3u}},
            secondSource);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        builder.setSource(sourceId, 0u);
        CHECK_THROWS_AS(builder.build(), std::logic_error); // two source nodes
    }
}

TEST_CASE("graph builder rejects invalid route destinations", "[unit][graph][builder]")
{
    {
        graphFlow::GraphBuilder<Device> builder;
        DummyNode source;
        DummyNode sink;
        auto const sourceId = builder.addNode(
            graphFlow::NodeRole::source,
            "source",
            1u,
            graphFlow::RouteTable{{3u}},
            source);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        builder.setSource(sourceId, 0u);
        CHECK_THROWS_AS(builder.build(), std::invalid_argument); // destination out of range
    }
    {
        graphFlow::GraphBuilder<Device> builder;
        DummyNode source;
        DummyNode transform;
        DummyNode sink;
        auto const sourceId = builder.addNode(
            graphFlow::NodeRole::source,
            "source",
            1u,
            graphFlow::RouteTable{{1u}},
            source);
        builder.addNode(
            graphFlow::NodeRole::transform,
            "transform",
            1u,
            graphFlow::RouteTable{{0u}}, // route back to the source is invalid
            transform);
        builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
        builder.setSource(sourceId, 0u);
        CHECK_THROWS_AS(builder.build(), std::invalid_argument);
    }
}

TEST_CASE("graph attaches to a fresh runner through the existing hooks path", "[unit][graph][builder]")
{
    auto device = graphFlow::test::makeCpuDevice();
    graphFlow::ChunkPool<Device> pool{device, 1u, 1u};

    DummyNode source;
    DummyNode transform;
    DummyNode sink;

    graphFlow::GraphBuilder<Device> builder;
    auto const sourceId = builder.addNode(
        graphFlow::NodeRole::source,
        "source",
        1u,
        graphFlow::RouteTable{{1u}},
        source);
    builder.addNode(
        graphFlow::NodeRole::transform,
        "transform",
        1u,
        graphFlow::RouteTable{{2u}},
        transform);
    builder.addNode(graphFlow::NodeRole::sink, "sink", 0u, graphFlow::RouteTable{}, sink);
    builder.setSource(sourceId, 0u, 0u);
    auto const graph = builder.build();

    graphFlow::GraphRunner<Device> runner{device, pool};
    graph.attachTo(runner);

    CHECK(runner.nodeCount() == 3u);
    auto const stats = runner.run();
    CHECK(stats.completedBatches == 0u);
    CHECK_THROWS_AS(graph.attachTo(runner), std::logic_error); // runner must be fresh
}
