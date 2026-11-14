/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/flow/Tuning.hpp"

#include <algorithm>
#include <cstdint>
#include <string_view>
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

    /** Workload of the example binary.
     *
     * `feedback` is the default public `Flow` graph. `kernel` runs the committed low-level
     * Generate -> Transform -> Sink pipeline so the P03 scatter mode, the dense/hash bin path and
     * the bin-occupancy distributions become selectable from the CLI. The two workloads share
     * C, P, T, Q and the frame cap.
     */
    enum class Workload : std::uint8_t
    {
        feedback,
        kernel
    };

    /** P03 scatter strategy of the dense path, mirrored for the CLI. */
    enum class ScatterChoice : std::uint8_t
    {
        shared, // ScatterMode::SharedStaging
        direct // ScatterMode::Direct
    };

    /** Bin-table path of the low-level transform stage. */
    enum class BinPath : std::uint8_t
    {
        dense,
        hash
    };

    /** Bin-occupancy distribution of the kernel workload.
     *
     * The mapping matches the committed P03 comparison: balanced spreads `id % B`, one-hot sends
     * every id to bin 0, skewed sends every eighth id to `id % B` and the rest to bin 0, and
     * sparse is the balanced spread over a wide bin set (`B >= 1024`).
     */
    enum class Distribution : std::uint8_t
    {
        balanced,
        oneHot,
        skewed,
        sparse
    };

    /** Options of the opt-in kernel workload; the resolver rejects them on the feedback workload. */
    struct KernelOptions
    {
        ScatterChoice scatter = ScatterChoice::shared;
        BinPath binPath = BinPath::dense;
        std::uint32_t bins = 64u;
        Distribution distribution = Distribution::balanced;
    };

    /** Sentinel left in the result buffer when a particle never reaches a sink. */
    inline constexpr std::uint32_t missingWeight = 0xDEAD'BEEFu;

    /** Largest bin count the dense shared table supports (`Config::denseBinLimit`). */
    inline constexpr std::uint32_t maxDenseKernelBins = 1024u;

    /** Compile-time bin counts the kernel workload instantiates.
     *
     * The dense kernel sizes a shared table by `maxBins`, so the value must be a template
     * parameter; the CLI accepts exactly this set. 64 and 1024 cover the dense/hash overlap,
     * 4096 and 65536 cover the many-sparse-bin hash path.
     */
    inline bool isSupportedKernelBins(std::uint32_t bins)
    {
        switch(bins)
        {
        case 64u:
        case 1024u:
        case 4096u:
        case 65536u:
            return true;
        default:
            return false;
        }
    }

    inline std::string_view scatterChoiceName(ScatterChoice scatter)
    {
        return scatter == ScatterChoice::direct ? "direct" : "shared";
    }

    inline std::string_view binPathName(BinPath path)
    {
        return path == BinPath::hash ? "hash" : "dense";
    }

    inline std::string_view distributionName(Distribution distribution)
    {
        switch(distribution)
        {
        case Distribution::oneHot:
            return "one-hot";
        case Distribution::skewed:
            return "skewed";
        case Distribution::sparse:
            return "sparse";
        case Distribution::balanced:
        default:
            return "balanced";
        }
    }

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
