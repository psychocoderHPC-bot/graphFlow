/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/flow/Concepts.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace graphFlow
{
    template<typename TPayload, typename TDevice, typename TStorage>
    class Flow;

    template<std::uint32_t BinValue>
    class OutputPort;

    namespace detail
    {
        /** Identity token of one owning `Flow` instance.
         *
         * Handles keep a weak reference. The token outlives every node handle but not the Flow, so
         * stale-handle checks never dereference a destroyed owner.
         */
        struct GraphIdentity
        {
            std::uint64_t id = 0u;
        };

        inline std::uint64_t nextGraphIdentity()
        {
            static std::atomic<std::uint64_t> next{0u};
            return next.fetch_add(1u, std::memory_order_relaxed) + 1u;
        }
    } // namespace detail

    /** Opaque, non-owning handle of one node in a `Flow` graph.
     *
     * A handle carries the graph identity, the internal node key and the node label. It stays
     * valid when later nodes are registered. It does not keep the graph, its queues or the copied
     * functors alive. Users never see or derive numeric node ids.
     */
    class NodeHandle
    {
    public:
        NodeHandle() = default;

        /** Whether the owning `Flow` is still alive. Does not dereference the owner. */
        [[nodiscard]] bool valid() const noexcept
        {
            return !m_identity.expired();
        }

        /** Diagnostic label supplied at registration time. */
        [[nodiscard]] std::string const& name() const noexcept
        {
            return m_name;
        }

    protected:
        NodeHandle(
            std::weak_ptr<detail::GraphIdentity const> identity,
            std::uint32_t key,
            std::uint32_t bins,
            std::string name)
            : m_identity{std::move(identity)}
            , m_key{key}
            , m_bins{bins}
            , m_name{std::move(name)}
        {
        }

        /** Lock the graph identity or reject a handle whose owner is gone. */
        [[nodiscard]] std::shared_ptr<detail::GraphIdentity const> liveIdentity() const
        {
            auto identity = m_identity.lock();
            if(!identity)
                throw std::logic_error{"graphFlow: node handle refers to a destroyed Flow"};
            return identity;
        }

        [[nodiscard]] std::uint32_t key() const noexcept
        {
            return m_key;
        }

        [[nodiscard]] std::uint32_t bins() const noexcept
        {
            return m_bins;
        }

    private:
        std::weak_ptr<detail::GraphIdentity const> m_identity;
        std::uint32_t m_key = 0u;
        std::uint32_t m_bins = 0u;
        std::string m_name;

        template<typename TPayload, typename TDevice, typename TStorage>
        friend class Flow;
    };

    /** Node handle carrying the compile-time output bound of its producer.
     *
     * `SourceHandle` and transform handles use one output. A partition handle carries the
     * classifier's `maxBins` and the classifier's bin type, so `out<Bin>()` checks the bin range at
     * compile time.
     */
    template<std::uint32_t MaxBins, typename TBin = std::uint32_t>
    class TypedNodeHandle : public NodeHandle
    {
    public:
        static constexpr std::uint32_t maxBins = MaxBins;
        using bin_type = TBin;

        template<auto BinValue>
        requires PortValue<decltype(BinValue)>
        [[nodiscard]] OutputPort<static_cast<std::uint32_t>(BinValue)> out() const;

    private:
        using NodeHandle::NodeHandle;

        template<typename TPayload, typename TDevice, typename TStorage>
        friend class Flow;
    };

    using SourceHandle = TypedNodeHandle<1u, std::uint32_t>;
    using TransformHandle = TypedNodeHandle<1u, std::uint32_t>;
} // namespace graphFlow

#include "graphFlow/flow/Ports.hpp"
