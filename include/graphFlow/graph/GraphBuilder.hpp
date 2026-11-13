/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/graph/Graph.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graphFlow
{
    /** Builder for the immutable `Graph` topology.
     *
     * Nodes are added in insertion order and receive stable ids `0, 1, ...`. `build()` validates
     * the complete topology and snapshots it into a `Graph`; later builder mutation does not change
     * an already built graph. Node objects are caller-owned references and must outlive the graph
     * and every runner the graph is attached to.
     */
    template<typename TDevice>
    class GraphBuilder
    {
    public:
        using Node = NodeBase<TDevice>;

        /** Add a node and return its stable id.
         *
         * Source and transform nodes must have at least one bin and a route table with exactly
         * `bins` destinations. A sink must have zero bins and an empty route table. Destination
         * range and destination-role checks need the complete node set, so they run in `build()`.
         */
        NodeId addNode(NodeRole role, std::string name, std::uint32_t bins, RouteTable routes, Node& node)
        {
            if(m_nodes.size() >= static_cast<std::size_t>(std::numeric_limits<NodeId>::max()))
                throw std::overflow_error{"graphFlow: graph node count exceeds the 32-bit node id range"};

            switch(role)
            {
            case NodeRole::source:
            case NodeRole::transform:
                if(bins == 0u)
                    throw std::invalid_argument{"graphFlow: a source or transform node needs at least one bin"};
                routes.validateBins(bins);
                break;
            case NodeRole::sink:
                if(bins != 0u)
                    throw std::invalid_argument{"graphFlow: a sink node has zero bins"};
                if(!routes.empty())
                    throw std::invalid_argument{"graphFlow: a sink node has no routes"};
                break;
            default:
                throw std::invalid_argument{"graphFlow: unknown node role"};
            }

            NodeId const id = static_cast<NodeId>(m_nodes.size());
            m_nodes.push_back(Entry{std::move(name), role, bins, std::move(routes), &node});
            return id;
        }

        /** Declare the single source node and its particle-admission capacity P.
         *
         * `liveParticles == 0` disables the particle budget, matching `GraphRunner::setSource()`.
         * The id must already name a source-role node; `build()` additionally requires exactly one
         * source node in the complete graph.
         */
        void setSource(NodeId source, std::uint64_t numParticles, std::uint32_t liveParticles = 0u)
        {
            if(source >= m_nodes.size())
                throw std::out_of_range{"graphFlow: source node id out of range"};
            if(m_nodes[source].role != NodeRole::source)
                throw std::invalid_argument{"graphFlow: setSource requires a source-role node"};
            m_source = source;
            m_numParticles = numParticles;
            m_liveParticles = liveParticles;
        }

        /** Validate the complete topology and return an immutable snapshot. */
        [[nodiscard]] Graph<TDevice> build() const
        {
            if(m_nodes.empty())
                throw std::logic_error{"graphFlow: cannot build an empty graph"};

            std::size_t sourceCount = 0u;
            for(Entry const& node : m_nodes)
            {
                if(node.role == NodeRole::source)
                    ++sourceCount;
                if(node.role != NodeRole::sink)
                {
                    if(node.bins == 0u)
                        throw std::invalid_argument{"graphFlow: a source or transform node needs at least one bin"};
                    node.routes.validateBins(node.bins);
                }
                else
                {
                    if(node.bins != 0u || !node.routes.empty())
                        throw std::invalid_argument{"graphFlow: a sink node has no bins and no routes"};
                }
            }
            if(sourceCount != 1u)
                throw std::logic_error{"graphFlow: a graph needs exactly one source node"};
            if(!m_source.has_value())
                throw std::logic_error{"graphFlow: setSource() was not called"};
            if(m_source.value() >= m_nodes.size())
                throw std::logic_error{"graphFlow: configured source node id out of range"};
            if(m_nodes[m_source.value()].role != NodeRole::source)
                throw std::logic_error{"graphFlow: configured source is not a source-role node"};

            for(std::size_t id = 0u; id < m_nodes.size(); ++id)
            {
                for(NodeId const destination : m_nodes[id].routes.destinations())
                {
                    if(destination >= m_nodes.size())
                        throw std::invalid_argument{"graphFlow: route destination out of range"};
                    if(destination > static_cast<NodeId>(std::numeric_limits<int>::max()))
                        throw std::overflow_error{"graphFlow: route destination exceeds int range"};
                    if(m_nodes[destination].role == NodeRole::source)
                        throw std::invalid_argument{"graphFlow: a route may not target the source node"};
                }
            }

            return Graph<TDevice>{m_nodes, m_source.value(), m_numParticles, m_liveParticles};
        }

        [[nodiscard]] std::uint32_t nodeCount() const noexcept
        {
            return static_cast<std::uint32_t>(m_nodes.size());
        }

    private:
        using Entry = typename Graph<TDevice>::NodeSpec;

        std::vector<Entry> m_nodes;
        std::optional<NodeId> m_source;
        std::uint64_t m_numParticles = 0u;
        std::uint32_t m_liveParticles = 0u;
    };
} // namespace graphFlow
