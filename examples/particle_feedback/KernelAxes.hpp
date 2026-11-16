/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleFunctors.hpp"
#include "ParticleTypes.hpp"
#include "graphFlow/core/ScatterMode.hpp"
#include "graphFlow/flow/Flow.hpp"
#include "graphFlow/graph/Classifier.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace graphFlow::particle
{
    /** Occupancy mapping shared by the low-level policy probe and the public classifier.
     *
     * The mapping is the committed P03 comparison: `balanced` spreads `id % B`, `one-hot` sends
     * every id to bin 0, `skewed` sends every eighth id to `id % B` and the rest to bin 0, and
     * `sparse` is the balanced spread over a wide bin set.
     */
    ALPAKA_FN_ACC inline std::uint32_t distributionBin(Distribution distribution, std::uint64_t id, std::uint32_t bins)
    {
        switch(distribution)
        {
        case Distribution::oneHot:
            return 0u;
        case Distribution::skewed:
            return id % 8u == 0u ? static_cast<std::uint32_t>(id % bins) : 0u;
        case Distribution::balanced:
        case Distribution::sparse:
        default:
            return static_cast<std::uint32_t>(id % bins);
        }
    }

    /** Low-level policy view of the distribution mapping, kept for the scatter-mode probe.
     *
     * The measured kernel workload itself runs through `Flow` and uses
     * `KernelDistributionClassifier`; this policy only exposes the same mapping and the
     * `ScatterMode` member that `detail::ScatterModeOf` inspects.
     */
    template<std::uint32_t T_Bins, graphFlow::ScatterMode TMode>
    struct KernelDistributionPolicy
    {
        static constexpr std::uint32_t maxBins = T_Bins;
        static constexpr graphFlow::ScatterMode scatterMode = TMode;

        Distribution distribution = Distribution::balanced;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return distributionBin(distribution, id, T_Bins);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            particle.weight = bin;
        }
    };

    /** Public `Flow` classifier: one bin per particle from the selected distribution. */
    template<std::uint32_t T_Bins>
    struct KernelDistributionClassifier
    {
        static constexpr std::uint32_t maxBins = T_Bins;

        Distribution distribution = Distribution::balanced;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            ParticleRecord const& particle) const
        {
            return distributionBin(distribution, particle[idTag], T_Bins);
        }
    };

    /** Public `Flow` updater: the sink observes the chosen bin as the particle weight. */
    struct KernelBinUpdate
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            ParticleRecord& particle) const
        {
            particle[weightTag] = bin;
            return graphFlow::UpdateResult::success;
        }
    };

    struct KernelAxisResult
    {
        std::vector<std::uint32_t> weights; // chosen bin indexed by particle id
        std::vector<HistogramRow> histogram; // sorted per-bin occupancy
        SchedulerStats stats;
        std::uint32_t liveParticles = 0u;
    };

    namespace detail
    {
        template<typename TDevice, std::uint32_t T_Bins>
        KernelAxisResult runKernelFlow(TDevice device, Parameters const& parameters, KernelOptions const& options)
        {
            if(parameters.numParticles == 0u)
                return KernelAxisResult{};
            if(parameters.liveParticles == 0u)
                throw std::invalid_argument{"graphFlow::particle: kernel workload needs a non-zero P"};

            std::uint64_t const numParticles = parameters.numParticles;
            auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{numParticles});
            {
                auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
                alpaka::onHost::fill(initQueue, result, missingWeight);
            }

            // The measured axes are public graph properties: the distribution is the classifier,
            // the scatter/bin-table choice is Tuning, and Flow derives the pool, tiles, plan and
            // completion. No runner, route table or pool is constructed here.
            graphFlow::SoAFlow<ParticleRecord, TDevice> flow{device};
            auto source = flow.source("generate", GenerateParticles{});
            auto transform = flow.partition(
                "transform",
                KernelDistributionClassifier<T_Bins>{options.distribution},
                KernelBinUpdate{});
            using ResultView = decltype(result.getView());
            auto sink = flow.sink("sink", RecordWeight<ResultView>{result.getView()});

            flow.connect(source, transform);
            // The measured bin counts include 4096 and 65536, so connect the outputs in a loop
            // through the runtime-bin overload instead of enumerating compile-time ports.
            for(std::uint32_t bin = 0u; bin < T_Bins; ++bin)
                flow.connect(transform, bin, sink);

            graphFlow::Tuning tuning = tuningOf(parameters);
            tuning.scatterMode = options.scatter == ScatterChoice::direct ? graphFlow::ScatterMode::Direct
                                                                          : graphFlow::ScatterMode::SharedStaging;
            tuning.binPath = options.binPath == BinPath::hash ? graphFlow::BinPath::hash : graphFlow::BinPath::dense;
            flow.configure(tuning);

            SchedulerStats const stats = flow.run(source, numParticles);

            std::vector<std::uint32_t> host(static_cast<std::size_t>(numParticles), 0u);
            {
                auto queue = device.makeQueue(alpaka::queueKind::blocking);
                alpaka::onHost::memcpy(queue, host, result);
            }

            KernelAxisResult output;
            output.weights = std::move(host);
            output.histogram = histogramOf(output.weights);
            output.stats = stats;
            output.liveParticles = parameters.liveParticles;
            return output;
        }
    } // namespace detail

    /** Run the opt-in kernel workload: Generate -> Transform -> Sink through the public `Flow`.
     *
     * The feedback workload keeps its public `Flow` graph. This workload makes the P03 scatter
     * mode, the dense/hash bin path and the bin-occupancy distribution reachable end to end from
     * the CLI. The sink stores the chosen bin, so `histogram` is the exact per-bin count and its
     * counts sum to N.
     *
     * Unsupported combinations are rejected before any device work: the dense path stops at
     * `Config::denseBinLimit`, the hash path has no direct scatter and is limited to the
     * compile-time hash table width, and `sparse` needs at least 1024 bins.
     */
    template<typename TDevice>
    KernelAxisResult runKernelAxes(TDevice device, Parameters const& parameters, KernelOptions const& options)
    {
        if(!isSupportedKernelBins(options.bins))
            throw std::invalid_argument{"graphFlow::particle: unsupported kernel bin count"};
        if(options.binPath == BinPath::dense && options.bins > maxDenseKernelBins)
            throw std::invalid_argument{
                "graphFlow::particle: the dense kernel path supports at most 1024 bins; use --bin-path hash"};
        if(options.binPath == BinPath::hash && options.scatter == ScatterChoice::direct)
            throw std::invalid_argument{
                "graphFlow::particle: the hash kernel path always stages; --scatter direct is dense-only"};
        if(options.binPath == BinPath::hash && parameters.tileSize > graphFlow::Config::hashBinTableTileSize)
            throw std::invalid_argument{"graphFlow::particle: the hash kernel path supports --tile-size at most 128"};
        if(options.distribution == Distribution::sparse && options.bins < 1024u)
            throw std::invalid_argument{"graphFlow::particle: sparse requires at least 1024 bins"};

        switch(options.bins)
        {
        case 64u:
            return detail::runKernelFlow<TDevice, 64u>(device, parameters, options);
        case 1024u:
            return detail::runKernelFlow<TDevice, 1024u>(device, parameters, options);
        case 4096u:
            return detail::runKernelFlow<TDevice, 4096u>(device, parameters, options);
        case 65536u:
            return detail::runKernelFlow<TDevice, 65536u>(device, parameters, options);
        default:
            throw std::invalid_argument{"graphFlow::particle: unsupported kernel bin count"};
        }
    }
} // namespace graphFlow::particle
