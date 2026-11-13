/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/flow/NodeHandle.hpp"

#include <cstdint>
#include <utility>

namespace graphFlow
{
    /** Output port of a node.
     *
     * `BinValue` is the compile-time bin index. Use `out<0>()` for a numeric bin or
     * `out<Outcome::feedback>()` for a named enumerator; both denote the same port.
     */
    template<std::uint32_t BinValue>
    class OutputPort
    {
    public:
        static constexpr std::uint32_t bin = BinValue;

        [[nodiscard]] NodeHandle const& node() const noexcept
        {
            return m_node;
        }

    private:
        explicit OutputPort(NodeHandle node) : m_node{std::move(node)}
        {
        }

        NodeHandle m_node;

        template<std::uint32_t MaxBins, typename TBin>
        friend class TypedNodeHandle;
    };

    template<std::uint32_t MaxBins, typename TBin>
    template<auto BinValue>
    requires PortValue<decltype(BinValue)>
    OutputPort<static_cast<std::uint32_t>(BinValue)> TypedNodeHandle<MaxBins, TBin>::out() const
    {
        static_assert(
            static_cast<std::uint32_t>(BinValue) < MaxBins,
            "graphFlow: output bin exceeds the producer's compile-time maxBins");
        (void) this->liveIdentity();
        return OutputPort<static_cast<std::uint32_t>(BinValue)>{*this};
    }
} // namespace graphFlow
