/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/core/PoolBudget.hpp"
#include "graphFlow/core/Reservation.hpp"
#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/flow/NodeHandle.hpp"
#include "graphFlow/flow/Payload.hpp"
#include "graphFlow/flow/PayloadStorage.hpp"
#include "graphFlow/flow/Ports.hpp"
#include "graphFlow/flow/Tuning.hpp"
#include "graphFlow/graph/RouteTable.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace graphFlow
{
    /** Raised by `Flow::run()` when the topology is valid but the typed execution path is not built.
     *
     * This slice owns registration, handles, ports, topology validation, tuning and pool
     * derivation. The generic execution adapters land in T3; the distinct type keeps the pending
     * execution apart from invalid graph or tuning errors.
     */
    class ExecutionUnavailable : public std::logic_error
    {
    public:
        using std::logic_error::logic_error;
    };

    /** Application-facing graph of one payload type.
     *
     * A `Flow` owns the node topology, the payload storage and the execution runner. Registration
     * returns opaque handles; users declare functors and edges and never see a node id. The
     * low-level `GraphBuilder`/`GraphRunner`/`Hooks` path stays available through its own headers
     * for custom nodes.
     *
     * This implementation slice provides construction, handles, ports, topology validation, tuning
     * and automatic pool derivation. The typed execution adapters are added in T3; `run()` therefore
     * validates the complete graph and then reports that execution is not available yet.
     */
    template<typename TPayload, typename TDevice, typename TStorage = DefaultStorage<TPayload, TDevice>>
    class Flow
    {
        static_assert(Payload<TPayload>, "graphFlow::Flow: the payload type does not satisfy Payload");

    public:
        using PayloadType = TPayload;
        using Device = TDevice;
        using Storage = TStorage;
        using Stats = SchedulerStats;

        static_assert(
            std::constructible_from<TStorage, TDevice, std::uint32_t, std::uint32_t>,
            "graphFlow::Flow: the storage type must be constructible from (device, numChunks, chunkCapacity)");
        static_assert(
            ChunkStorage<TStorage>,
            "graphFlow::Flow: the storage type must satisfy ChunkStorage");
        static_assert(
            requires { { TStorage::slotBytes } -> std::convertible_to<std::uint64_t>; },
            "graphFlow::Flow: the storage type must expose static constexpr slotBytes");

        explicit Flow(TDevice device)
            : m_device{std::move(device)}
            , m_identity{std::make_shared<detail::GraphIdentity>(detail::GraphIdentity{detail::nextGraphIdentity()})}
        {
        }

        Flow(Flow const&) = delete;
        Flow& operator=(Flow const&) = delete;
        Flow(Flow&&) noexcept = default;
        ~Flow() = default;

        /** Replace the optional tuning knobs. Invalid zero values are rejected immediately. */
        void configure(Tuning tuning)
        {
            tuning.validate();
            m_tuning = std::move(tuning);
        }

        [[nodiscard]] Tuning const& tuning() const noexcept
        {
            return m_tuning;
        }

        /** Register the single source node. */
        template<typename TFunctor>
        requires SourceFor<TFunctor, TPayload>
        SourceHandle source(std::string name, TFunctor functor)
        {
            ensureOwner();
            if(m_sourceKey.has_value())
                throw std::logic_error{"graphFlow: a Flow has exactly one source node"};

            auto const key = appendNode(
                std::move(name),
                NodeRole::source,
                1u,
                std::make_shared<TFunctor>(std::move(functor)),
                nullptr);
            m_sourceKey = key;
            return SourceHandle{m_identity, key, 1u, m_nodes[key].name};
        }

        /** Register a classifier node with an optional per-item updater. */
        template<typename TClassifier, typename TUpdate = NoUpdate>
        requires PartitionClassifierFor<TClassifier, TPayload>
                 && UpdateFor<TUpdate, ClassifierBin<TClassifier, TPayload>, TPayload>
        TypedNodeHandle<static_cast<std::uint32_t>(TClassifier::maxBins), ClassifierBin<TClassifier, TPayload>>
        partition(std::string name, TClassifier classifier, TUpdate update = {})
        {
            constexpr std::uint32_t bins = static_cast<std::uint32_t>(TClassifier::maxBins);
            auto const key = appendNode(
                std::move(name),
                NodeRole::transform,
                bins,
                std::make_shared<TClassifier>(std::move(classifier)),
                std::make_shared<TUpdate>(std::move(update)));
            return TypedNodeHandle<bins, ClassifierBin<TClassifier, TPayload>>{
                m_identity,
                key,
                bins,
                m_nodes[key].name};
        }

        /** Register an ordinary single-output transform node. */
        template<typename TFunctor>
        requires TransformFor<TFunctor, TPayload>
        TransformHandle transform(std::string name, TFunctor functor)
        {
            auto const key = appendNode(
                std::move(name),
                NodeRole::transform,
                1u,
                std::make_shared<TFunctor>(std::move(functor)),
                nullptr);
            return TransformHandle{m_identity, key, 1u, m_nodes[key].name};
        }

        /** Register a sink node. */
        template<typename TFunctor>
        requires SinkFor<TFunctor, TPayload>
        NodeHandle sink(std::string name, TFunctor functor)
        {
            auto const key = appendNode(
                std::move(name),
                NodeRole::sink,
                0u,
                std::make_shared<TFunctor>(std::move(functor)),
                nullptr);
            return NodeHandle{m_identity, key, 0u, m_nodes[key].name};
        }

        /** Connect the single output of `from` to `to`. */
        void connect(NodeHandle const& from, NodeHandle const& to)
        {
            connectNodeToNode(from, to);
        }

        /** Compile-time rejection of a bare multi-output node in `connect(node, node)`. */
        template<std::uint32_t MaxBins, typename TBin>
        void connect(TypedNodeHandle<MaxBins, TBin> const& from, NodeHandle const& to)
        {
            static_assert(
                MaxBins == 1u,
                "graphFlow: a multi-output node must be connected via out<Bin>(), not connect(node, node)");
            connectNodeToNode(from, to);
        }

        /** Connect one compile-time output port to a destination node. */
        template<std::uint32_t BinValue>
        void connect(OutputPort<BinValue> const& port, NodeHandle const& to)
        {
            connectPort(port.node(), BinValue, to);
        }

        /** Validate the graph and derive its storage, then report that typed execution is pending.
         *
         * Invalid, missing and duplicated routes are rejected before any device work. The storage
         * and runner members are constructed from the derived configuration so their ownership and
         * destruction order are already fixed for the T3 execution path.
         */
        Stats run(NodeHandle const& source, std::uint64_t itemCount)
        {
            validateForRun(source);

            Config const config = resolveConfig(itemCount);
            m_storage = std::make_unique<TStorage>(m_device, config.numChunks, config.chunkCapacity);
            m_runner = std::make_unique<GraphRunner<TDevice, TStorage>>(m_device, *m_storage);
            m_runner->setBatchChunks(config.batchChunks);

            throw ExecutionUnavailable{
                "graphFlow::Flow::run: graph topology is valid; typed execution is added in T3"};
        }

        Flow& operator=(Flow&& other) noexcept
        {
            if(this != &other)
            {
                // The runner refers to the pool owned by the storage. Destroy it first so a move
                // assignment can never leave a runner pointing at storage that was just reset.
                m_runner.reset();
                m_storage.reset();
                m_device = std::move(other.m_device);
                m_identity = std::move(other.m_identity);
                m_sourceKey = std::move(other.m_sourceKey);
                m_nodes = std::move(other.m_nodes);
                m_tuning = std::move(other.m_tuning);
                m_storage = std::move(other.m_storage);
                m_runner = std::move(other.m_runner);
            }
            return *this;
        }

    private:
        static constexpr NodeId invalidRoute = std::numeric_limits<NodeId>::max();

        struct NodeRecord
        {
            std::string name;
            NodeRole role = NodeRole::transform;
            std::uint32_t bins = 0u;
            std::vector<NodeId> routes;
            std::shared_ptr<void const> functor;
            std::shared_ptr<void const> update;
        };

        void ensureOwner() const
        {
            if(!m_identity)
                throw std::logic_error{"graphFlow: use of a moved-from Flow"};
        }

        [[nodiscard]] NodeId appendNode(
            std::string name,
            NodeRole role,
            std::uint32_t bins,
            std::shared_ptr<void const> functor,
            std::shared_ptr<void const> update)
        {
            ensureOwner();
            if(m_nodes.size() >= static_cast<std::size_t>(std::numeric_limits<NodeId>::max()))
                throw std::overflow_error{"graphFlow: graph node count exceeds the 32-bit node id range"};

            std::vector<NodeId> routes;
            if(role != NodeRole::sink)
                routes.assign(bins, invalidRoute);

            NodeId const key = static_cast<NodeId>(m_nodes.size());
            m_nodes.push_back(
                NodeRecord{std::move(name), role, bins, std::move(routes), std::move(functor), std::move(update)});
            return key;
        }

        void connectNodeToNode(NodeHandle const& from, NodeHandle const& to)
        {
            // Check liveness before the output count so a stale handle always reports the
            // documented destroyed-owner error instead of a shape error.
            (void) from.liveIdentity();
            if(from.bins() != 1u)
                throw std::invalid_argument{
                    "graphFlow: a multi-output node must be connected via out<Bin>(), not connect(node, node)"};
            connectPort(from, 0u, to);
        }

        void connectPort(NodeHandle const& from, std::uint32_t bin, NodeHandle const& to)
        {
            ensureOwner();
            auto const fromIdentity = from.liveIdentity();
            auto const toIdentity = to.liveIdentity();
            if(fromIdentity != m_identity || toIdentity != m_identity)
                throw std::invalid_argument{"graphFlow: node handles belong to a different Flow instance"};

            NodeId const fromKey = from.key();
            NodeId const toKey = to.key();
            if(fromKey >= m_nodes.size() || toKey >= m_nodes.size())
                throw std::logic_error{"graphFlow: malformed node handle"};
            if(bin >= m_nodes[fromKey].bins)
                throw std::out_of_range{"graphFlow: output bin out of range"};

            NodeRecord& producer = m_nodes[fromKey];
            NodeRecord& consumer = m_nodes[toKey];
            if(producer.role == NodeRole::sink)
                throw std::invalid_argument{"graphFlow: a sink node has no output to connect"};
            if(consumer.role == NodeRole::source)
                throw std::invalid_argument{"graphFlow: a route may not target the source node"};
            if(producer.routes[bin] != invalidRoute)
                throw std::invalid_argument{"graphFlow: route for this output is already connected"};
            producer.routes[bin] = toKey;
        }

        void validateForRun(NodeHandle const& source) const
        {
            ensureOwner();
            auto const sourceIdentity = source.liveIdentity();
            if(sourceIdentity != m_identity)
                throw std::invalid_argument{"graphFlow: the run source belongs to a different Flow instance"};

            if(!m_sourceKey.has_value())
                throw std::logic_error{"graphFlow: no source node is registered"};
            if(source.key() != *m_sourceKey)
                throw std::invalid_argument{"graphFlow: run() requires the registered source node"};

            std::size_t sourceCount = 0u;
            for(std::size_t key = 0u; key < m_nodes.size(); ++key)
            {
                NodeRecord const& node = m_nodes[key];
                if(node.role == NodeRole::source)
                    ++sourceCount;

                if(node.role == NodeRole::sink)
                {
                    if(node.bins != 0u || !node.routes.empty())
                        throw std::invalid_argument{"graphFlow: a sink node has no bins and no routes"};
                    continue;
                }

                if(node.bins == 0u)
                    throw std::invalid_argument{"graphFlow: a source or transform node needs at least one bin"};
                if(node.routes.size() != node.bins)
                    throw std::invalid_argument{"graphFlow: route table size must equal the bin count"};

                for(std::size_t bin = 0u; bin < node.routes.size(); ++bin)
                {
                    NodeId const destination = node.routes[bin];
                    if(destination == invalidRoute)
                        throw std::invalid_argument{
                            "graphFlow: missing route for node '" + node.name + "' bin " + std::to_string(bin)};
                    if(destination >= m_nodes.size())
                        throw std::invalid_argument{"graphFlow: route destination out of range"};
                    if(m_nodes[destination].role == NodeRole::source)
                        throw std::invalid_argument{"graphFlow: a route may not target the source node"};
                }
            }
            if(sourceCount != 1u)
                throw std::logic_error{"graphFlow: a graph needs exactly one source node"};
        }

        [[nodiscard]] std::uint32_t maxBins() const noexcept
        {
            std::uint32_t bins = 0u;
            for(NodeRecord const& node : m_nodes)
                bins = std::max(bins, node.bins);
            return bins == 0u ? 1u : bins;
        }

        [[nodiscard]] std::uint32_t transformReserve(std::uint32_t chunkCapacity) const
        {
            std::uint64_t const max = std::numeric_limits<std::uint32_t>::max();
            std::uint64_t total = 0u;
            for(NodeRecord const& node : m_nodes)
            {
                if(node.role != NodeRole::transform)
                    continue;
                std::uint64_t const bound = outputChunkBound(chunkCapacity, chunkCapacity, node.bins);
                if(total > max || bound > max - total)
                    throw std::overflow_error{"graphFlow: transform output reserve exceeds 32 bits"};
                total += bound;
            }
            return static_cast<std::uint32_t>(total);
        }

        [[nodiscard]] std::uint32_t derivePoolSize(
            std::uint32_t liveParticles,
            std::uint32_t chunkCapacity,
            std::uint64_t itemCount,
            std::uint32_t reserve) const
        {
            if(itemCount == 0u)
                return 1u;

            std::uint64_t live = liveParticles;
            if(live == 0u)
            {
                std::uint64_t const cap = 8ull * chunkCapacity;
                live = std::min(itemCount, cap);
            }
            live = std::min(live, itemCount);
            if(live == 0u)
                live = 1u;

            auto const max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
            if(live > max || reserve > max - live)
                throw std::overflow_error{"graphFlow: chunk pool would exceed 2^32 chunks"};
            return static_cast<std::uint32_t>(live + reserve);
        }

        [[nodiscard]] Config resolveConfig(std::uint64_t itemCount) const
        {
            std::uint32_t const bins = maxBins();
            Config config;
            config.numParticles = itemCount;
            config.maxBins = bins;
            config.batchChunks = m_tuning.batchChunks.value_or(1u);
            config.frameCap = m_tuning.frameCap.value_or(defaultFrameCap);
            config.chunkCapacity = m_tuning.chunkCapacity.value_or(config.chunkCapacity);
            config.liveParticles = m_tuning.liveParticles.value_or(config.liveParticles);
            config.tileSize = m_tuning.tileSize.value_or(config.tileSize);
            config.numChunks = m_tuning.numChunks.value_or(config.numChunks);
            config.poolPayloadBudgetBytes = m_tuning.poolPayloadBudgetBytes.value_or(config.poolPayloadBudgetBytes);
            config.policyWaves = m_tuning.policyWaves.value_or(config.policyWaves);
            config.policyTargetBlocksPerSm = m_tuning.policyTargetBlocksPerSm.value_or(config.policyTargetBlocksPerSm);
            config.policyLogicalTileWidth = m_tuning.policyLogicalTileWidth.value_or(config.policyLogicalTileWidth);
            config.policyMaxChunkCapacity = m_tuning.policyMaxChunkCapacity.value_or(config.policyMaxChunkCapacity);
            config.policyLiveParticlesPerChunk
                = m_tuning.policyLiveParticlesPerChunk.value_or(config.policyLiveParticlesPerChunk);
            config.policyMemoryBudgetBytes = m_tuning.policyMemoryBudgetBytes.value_or(config.policyMemoryBudgetBytes);
            config.slotBytes = TStorage::slotBytes;

            DevicePolicyOverrides overrides;
            overrides.chunkCapacity = m_tuning.chunkCapacity;
            overrides.liveParticles = m_tuning.liveParticles;
            overrides.tileSize = m_tuning.tileSize;

            DevicePolicy const policy
                = makeDevicePolicy(m_device.getDeviceProperties(), itemCount, overrides, policyConfigOf(config));
            config.poolOutputReserveChunks = transformReserve(policy.chunkCapacity);
            applyDevicePolicy(config, policy);

            if(m_tuning.poolPayloadBudgetBytes.has_value())
                config.poolPayloadBudgetBytes = *m_tuning.poolPayloadBudgetBytes;
            if(m_tuning.numChunks.has_value())
                config.numChunks = *m_tuning.numChunks;
            else
                config.numChunks = derivePoolSize(
                    config.liveParticles,
                    config.chunkCapacity,
                    itemCount,
                    config.poolOutputReserveChunks);

            config.validate();
            return config;
        }

        TDevice m_device;
        std::shared_ptr<detail::GraphIdentity> m_identity;
        std::optional<NodeId> m_sourceKey;
        std::vector<NodeRecord> m_nodes;
        Tuning m_tuning;
        std::unique_ptr<TStorage> m_storage;
        std::unique_ptr<GraphRunner<TDevice, TStorage>> m_runner;
    };
} // namespace graphFlow
