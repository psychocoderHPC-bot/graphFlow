/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace graphFlow
{
    /** Stable graph node identifier. Ids are assigned by GraphBuilder in insertion order. */
    using NodeId = std::uint32_t;

    /** Per-bin destination table of one graph node.
     *
     * `bins()` equals the number of destinations. Every valid output bin in `[0, bins())` maps to
     * exactly one destination node id. An empty table is the only valid table for a sink. A
     * feedback edge is not special: it is an ordinary destination that happens to name an earlier
     * node id.
     *
     * The table only carries topology. GraphBuilder validates the destination range, the role of
     * the destination, and the bin count against the node description.
     */
    class RouteTable
    {
    public:
        RouteTable() = default;

        explicit RouteTable(std::vector<NodeId> destinations)
            : m_destinations{std::move(destinations)}
        {
            if(m_destinations.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
                throw std::overflow_error{"graphFlow: route table exceeds 2^32 bins"};
        }

        RouteTable(std::initializer_list<NodeId> destinations)
            : RouteTable{std::vector<NodeId>{destinations}}
        {
        }

        /** Construct a table and validate it against an explicit bin count. */
        RouteTable(std::uint32_t bins, std::vector<NodeId> destinations)
            : RouteTable{std::move(destinations)}
        {
            validateBins(bins);
        }

        /** Number of bins served by this table. */
        [[nodiscard]] std::uint32_t bins() const noexcept
        {
            return static_cast<std::uint32_t>(m_destinations.size());
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return m_destinations.size();
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return m_destinations.empty();
        }

        [[nodiscard]] std::span<NodeId const> destinations() const noexcept
        {
            return {m_destinations.data(), m_destinations.size()};
        }

        /** Destination of `bin`; throws when the bin is outside the table. */
        [[nodiscard]] NodeId destination(std::uint32_t bin) const
        {
            if(bin >= m_destinations.size())
                throw std::out_of_range{"graphFlow: route bin out of range"};
            return m_destinations[bin];
        }

        /** Unchecked destination lookup for callers that already validated the range. */
        [[nodiscard]] NodeId operator[](std::uint32_t bin) const noexcept
        {
            return m_destinations[bin];
        }

        /** Require this table to serve exactly `bins` bins. */
        void validateBins(std::uint32_t bins) const
        {
            if(m_destinations.size() != bins)
                throw std::invalid_argument{"graphFlow: route table size must equal the bin count"};
        }

    private:
        std::vector<NodeId> m_destinations;
    };
} // namespace graphFlow
