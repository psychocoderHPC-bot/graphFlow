/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/flow/Tuning.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace graphFlow::particle
{
    /** Domain knobs of the particle-feedback example.
     *
     * `chunkSize` (C), `liveParticles` (P), `tileSize` (T), `batchChunks` (Q) and `frameCap` map
     * directly to `graphFlow::Tuning`. A zero `liveParticles` keeps the device-policy default.
     */
    struct Parameters
    {
        std::uint64_t numParticles = 0u;
        std::uint32_t chunkSize = 128u;
        std::uint32_t liveParticles = 0u; // P, zero means the device-policy default
        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint32_t tileSize = 128u;
        std::uint32_t deviceIndex = 0u;
        std::uint32_t batchChunks = 1u; // Q, ready input chunks per batch
        std::uint32_t frameCap = graphFlow::defaultFrameCap; // blocks per multiprocessor cap
    };

    /** Map the domain knobs to the optional execution tuning of `Flow`. */
    inline graphFlow::Tuning tuningOf(Parameters const& parameters)
    {
        graphFlow::Tuning tuning;
        tuning.seed = parameters.seed;
        tuning.chunkCapacity = parameters.chunkSize;
        if(parameters.liveParticles != 0u)
            tuning.liveParticles = parameters.liveParticles;
        tuning.tileSize = parameters.tileSize;
        tuning.batchChunks = parameters.batchChunks;
        tuning.frameCap = parameters.frameCap;
        return tuning;
    }

    struct HistogramRow
    {
        std::uint32_t weight = 0u;
        std::uint64_t count = 0u;

        friend bool operator==(HistogramRow const&, HistogramRow const&) = default;
    };

    /** Build the sorted, run-length encoded weight histogram. */
    inline std::vector<HistogramRow> histogramOf(std::vector<std::uint32_t> weights)
    {
        std::sort(weights.begin(), weights.end());
        std::vector<HistogramRow> rows;
        for(std::size_t i = 0u; i < weights.size();)
        {
            std::size_t j = i;
            while(j < weights.size() && weights[j] == weights[i])
                ++j;
            rows.push_back(HistogramRow{weights[i], static_cast<std::uint64_t>(j - i)});
            i = j;
        }
        return rows;
    }
} // namespace graphFlow::particle
