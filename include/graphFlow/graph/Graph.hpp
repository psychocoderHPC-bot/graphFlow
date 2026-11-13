/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/graph/NodeBase.hpp"
#include "graphFlow/graph/RouteTable.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graphFlow
{
    template<typename TDevice>
    class GraphBuilder;

    /** Immutable host-side graph topology.
     *
     * A `Graph` owns the topology only. Node objects are caller-owned non-owning references; the
     * caller must keep them alive while the graph is attached and while a runner uses it. The graph
     * stores no device code and no queue/event, so `GraphRunner::run()` remains the only scheduler.
     *
     * The graph is built by `GraphBuilder` and cannot be mutated afterwards. `attachTo()` registers
     * every node in id order with an existing `GraphRunner` through its public `addNode()` and
     * `setSource()` path. Attaching therefore does not change batching, FrameSpec launches, policy
     * semantics, or the example CLI.
     *
     * v1 limits: one source per graph; node ids are stable and equal to insertion order; routes may
     * name an earlier node (a feedback edge), but may not name the source. The builder does not run
     * a global cycle check: a feedback cycle is supported, but the node hooks must terminate it.
     * Runtime mutation, join, broadcast, and multi-owner chunks are not supported.
     */
    template<typename TDevice>
    class Graph
    {
    public:
        using Node = NodeBase<TDevice>;
        using Runner = GraphRunner<TDevice>;

        /** Topology snapshot of one node. `node` is caller-owned and non-owning. */
        struct NodeSpec
        {
            std::string name;
            NodeRole role = NodeRole::transform;
            std::uint32_t bins = 0u;
            RouteTable routes;
            Node* node = nullptr;
        };

        Graph(Graph const&) = default;
        Graph(Graph&&) noexcept = default;
        Graph& operator=(Graph const&) = default;
        Graph& operator=(Graph&&) noexcept = default;
        ~Graph() = default;

        [[nodiscard]] std::uint32_t nodeCount() const noexcept
        {
            return static_cast<std::uint32_t>(m_nodes.size());
        }

        [[nodiscard]] NodeId source() const noexcept
        {
            return m_source;
        }

        [[nodiscard]] std::uint64_t numParticles() const noexcept
        {
            return m_numParticles;
        }

        [[nodiscard]] std::uint32_t liveParticles() const noexcept
        {
            return m_liveParticles;
        }

        [[nodiscard]] NodeRole role(NodeId node) const
        {
            return entry(node).role;
        }

        [[nodiscard]] std::string const& name(NodeId node) const
        {
            return entry(node).name;
        }

        [[nodiscard]] std::uint32_t bins(NodeId node) const
        {
            return entry(node).bins;
        }

        [[nodiscard]] RouteTable const& routes(NodeId node) const
        {
            return entry(node).routes;
        }

        /** Mutable access to the caller-owned node object. */
        [[nodiscard]] Node& node(NodeId id) const
        {
            return *entry(id).node;
        }

        /** Register this topology with a fresh GraphRunner.
         *
         * The runner must not contain nodes yet: ids are assigned in insertion order and must stay
         * stable. `setBatchChunks()`, `setStallTimeout()`, and `run()` remain caller calls on the
         * runner.
         */
        void attachTo(Runner& runner) const
        {
            if(runner.nodeCount() != 0u)
                throw std::logic_error{"graphFlow: Graph::attachTo requires a fresh GraphRunner"};
            for(std::size_t id = 0u; id < m_nodes.size(); ++id)
            {
                NodeSpec const& node = m_nodes[id];
                std::vector<int> route;
                route.reserve(node.routes.size());
                for(NodeId const destination : node.routes.destinations())
                {
                    if(destination > static_cast<NodeId>(std::numeric_limits<int>::max()))
                        throw std::overflow_error{"graphFlow: route destination exceeds int range"};
                    route.push_back(static_cast<int>(destination));
                }
                std::uint32_t const assigned = runner.addNode(
                    node.name,
                    node.role,
                    node.bins,
                    std::move(route),
                    makeNodeHooks(*node.node));
                if(assigned != id)
                    throw std::logic_error{"graphFlow: GraphRunner assigned an unexpected node id"};
            }
            runner.setSource(m_source, m_numParticles, m_liveParticles);
        }

    private:
        Graph(
            std::vector<NodeSpec> nodes,
            NodeId source,
            std::uint64_t numParticles,
            std::uint32_t liveParticles)
            : m_nodes{std::move(nodes)}
            , m_source{source}
            , m_numParticles{numParticles}
            , m_liveParticles{liveParticles}
        {
        }

        [[nodiscard]] NodeSpec const& entry(NodeId id) const
        {
            if(id >= m_nodes.size())
                throw std::out_of_range{"graphFlow: graph node id out of range"};
            return m_nodes[id];
        }

        std::vector<NodeSpec> m_nodes;
        NodeId m_source = 0u;
        std::uint64_t m_numParticles = 0u;
        std::uint32_t m_liveParticles = 0u;

        friend class GraphBuilder<TDevice>;
    };
} // namespace graphFlow
