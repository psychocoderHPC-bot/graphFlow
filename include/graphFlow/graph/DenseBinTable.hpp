/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Reservation.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace graphFlow
{
    /** Host-side plan for the dense output table of one transform batch.
     *
     * `binBase[b]` is the index of the first flat output chunk of bin `b`; `fills` holds the fill
     * level of every flat output chunk in bin order. The device table layout is
     * `[binBase[0..B), chunkIndex[0..R)]`, i.e. `B + R` words.
     */
    struct DenseBinPlan
    {
        std::vector<std::uint32_t> binBase; // size B
        std::vector<std::uint32_t> fills;   // size R
        std::uint32_t numOutputChunks = 0u; // R

        [[nodiscard]] std::uint32_t tableWords() const
        {
            return static_cast<std::uint32_t>(binBase.size()) + numOutputChunks;
        }
    };

    /** Build the exact output-chunk plan from per-bin particle totals.
     *
     * Throws when `totals` is empty or when the exact chunk count exceeds the conservative
     * reservation bound of `outputChunkBound`.
     */
    inline DenseBinPlan planDenseBins(std::vector<std::uint32_t> const& totals, std::uint32_t capacity)
    {
        if(totals.empty())
            throw std::invalid_argument{"planDenseBins: at least one bin is required"};
        if(capacity == 0u)
            throw std::invalid_argument{"planDenseBins: capacity must be non-zero"};

        DenseBinPlan plan;
        plan.binBase.resize(totals.size());

        std::uint64_t exact = 0u;
        std::uint64_t particles = 0u;
        for(std::size_t b = 0u; b < totals.size(); ++b)
        {
            plan.binBase[b] = static_cast<std::uint32_t>(exact);
            exact += ceilDivChecked(totals[b], capacity);
            particles += totals[b];
        }
        if(exact > std::uint64_t{std::numeric_limits<std::uint32_t>::max()})
            throw std::overflow_error{"planDenseBins: output chunk count overflow"};

        std::uint64_t const bound = outputChunkBound(particles, capacity, totals.size());
        if(exact > bound)
            throw std::logic_error{"planDenseBins: exact chunk count exceeds the reservation bound"};

        plan.numOutputChunks = static_cast<std::uint32_t>(exact);
        plan.fills.reserve(plan.numOutputChunks);
        for(std::size_t b = 0u; b < totals.size(); ++b)
        {
            for(std::uint32_t offset = 0u; offset < totals[b]; offset += capacity)
            {
                std::uint32_t const remaining = totals[b] - offset;
                plan.fills.push_back(remaining < capacity ? remaining : capacity);
            }
        }
        if(plan.fills.size() != plan.numOutputChunks)
            throw std::logic_error{"planDenseBins: fill list does not match the chunk count"};
        return plan;
    }

    /** Serialise a plan and its chunk indices into the device table layout. */
    inline std::vector<std::uint32_t> buildDenseTable(
        DenseBinPlan const& plan,
        std::vector<std::uint32_t> const& chunkIndices)
    {
        if(chunkIndices.size() != plan.numOutputChunks)
            throw std::invalid_argument{"buildDenseTable: chunk index count does not match the plan"};
        std::vector<std::uint32_t> table;
        table.reserve(plan.tableWords());
        table.insert(table.end(), plan.binBase.begin(), plan.binBase.end());
        table.insert(table.end(), chunkIndices.begin(), chunkIndices.end());
        return table;
    }
} // namespace graphFlow
