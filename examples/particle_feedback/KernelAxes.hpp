/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleStages.hpp"
#include "ParticleTypes.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/ScatterKernel.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace graphFlow::particle
{
    /** Distribution classifier of the kernel workload, with the P03 scatter-mode knob.
     *
     * The mapping is the committed P03 comparison: `balanced` spreads `id % B`, `one-hot` sends
     * every id to bin 0, `skewed` sends every eighth id to `id % B` and the rest to bin 0, and
     * `sparse` is the balanced spread over a wide bin set. The sink writes the chosen bin into the
     * result buffer, so the host histogram is the exact per-bin occupancy.
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
            switch(distribution)
            {
            case Distribution::oneHot:
                return 0u;
            case Distribution::skewed:
                return id % 8u == 0u ? static_cast<std::uint32_t>(id % T_Bins) : 0u;
            case Distribution::balanced:
            case Distribution::sparse:
            default:
                return static_cast<std::uint32_t>(id % T_Bins);
            }
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

    struct KernelAxisResult
    {
        std::vector<std::uint32_t> weights; // chosen bin indexed by particle id
        std::vector<HistogramRow> histogram; // sorted per-bin occupancy
        SchedulerStats stats;
        std::uint32_t liveParticles = 0u;
    };

    namespace detail
    {
        /** Pool size of the one-transform kernel pipeline.
         *
         * The scheduler reserves one adaptive output credit `R = outputChunkBound(C, C, B)` per
         * in-flight transform batch and needs one free chunk beyond the reserve to dispatch. The
         * live term is P because split routing may fragment the live particles into one chunk each.
         */
        inline std::uint32_t kernelPoolChunks(
            std::uint64_t numParticles,
            std::uint32_t chunkSize,
            std::uint32_t liveParticles,
            std::uint32_t bins)
        {
            if(numParticles == 0u)
                return 1u;
            std::uint64_t const live = std::min<std::uint64_t>(liveParticles, numParticles);
            std::uint64_t const reserve = graphFlow::outputChunkBound(chunkSize, chunkSize, bins);
            std::uint64_t const chunks = live + reserve + 16u;
            if(chunks > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow::particle: kernel pool exceeds 2^32 chunks"};
            return static_cast<std::uint32_t>(chunks);
        }

        template<typename TDevice, typename TPolicy, typename TBinTable>
        KernelAxisResult runKernelPipeline(TDevice device, Parameters const& parameters, TPolicy policy)
        {
            if(parameters.numParticles == 0u)
                return KernelAxisResult{};
            if(parameters.liveParticles == 0u)
                throw std::invalid_argument{"graphFlow::particle: kernel workload needs a non-zero P"};

            std::uint64_t const numParticles = parameters.numParticles;
            std::uint32_t const chunks
                = kernelPoolChunks(numParticles, parameters.chunkSize, parameters.liveParticles, TPolicy::maxBins);
            graphFlow::ChunkPool<TDevice> pool{device, chunks, parameters.chunkSize};
            auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{numParticles});
            auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
            {
                auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
                alpaka::onHost::fill(initQueue, result, missingWeight);
                alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
            }

            GenerateStage<TDevice> generate{device, pool, parameters.seed, parameters.tileSize, parameters.frameCap};
            TransformStage<TDevice, TPolicy, TBinTable> transform{
                device,
                pool,
                policy,
                capped,
                parameters.tileSize,
                parameters.batchChunks,
                parameters.frameCap};
            SinkStage<TDevice> sink{device, pool, result, parameters.tileSize, parameters.frameCap};

            std::vector<int> route(TPolicy::maxBins, 2);
            graphFlow::GraphRunner<TDevice> runner{device, pool};
            auto const source
                = runner.addNode("generate", graphFlow::NodeRole::source, 1u, {1}, hooksOf<TDevice>(generate));
            runner.addNode(
                "transform",
                graphFlow::NodeRole::transform,
                TPolicy::maxBins,
                route,
                hooksOf<TDevice>(transform));
            runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, hooksOf<TDevice>(sink));
            runner.setSource(source, numParticles, parameters.liveParticles);
            runner.setBatchChunks(parameters.batchChunks);
            runner.setStallTimeout(std::chrono::milliseconds{30000});
            SchedulerStats const stats = runner.run();

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

        template<typename TDevice, std::uint32_t T_Bins, BinPath T_Path, ScatterChoice T_Scatter>
        KernelAxisResult runKernelTyped(TDevice device, Parameters const& parameters, Distribution distribution)
        {
            if constexpr(T_Path == BinPath::dense)
            {
                constexpr graphFlow::ScatterMode mode = T_Scatter == ScatterChoice::direct
                                                            ? graphFlow::ScatterMode::Direct
                                                            : graphFlow::ScatterMode::SharedStaging;
                using Policy = KernelDistributionPolicy<T_Bins, mode>;
                return runKernelPipeline<TDevice, Policy, graphFlow::DenseBinTable>(
                    device,
                    parameters,
                    Policy{distribution});
            }
            else
            {
                using Policy = KernelDistributionPolicy<T_Bins, graphFlow::ScatterMode::SharedStaging>;
                // The committed hash table is the 2*T default with T=128. A wider logical tile can
                // hold more distinct bins per tile than the table has slots; reject instead of
                // risking a table-full batch error.
                if(parameters.tileSize > graphFlow::HashBinTable<128u>::tileSize)
                    throw std::invalid_argument{
                        "graphFlow::particle: the hash kernel path supports --tile-size at most 128"};
                return runKernelPipeline<TDevice, Policy, graphFlow::HashBinTable<128u>>(
                    device,
                    parameters,
                    Policy{distribution});
            }
        }

        /** Dispatch one compile-time bin count.
         *
         * `T_AllowDense` must be false above the dense shared-table limit: instantiating the
         * dense kernel for 4096/65536 bins would size a 3*B-word shared table and fail ptxas.
         * The runtime validation already rejects that combination, so the hash-only branch is
         * the only reachable one there.
         */
        template<typename TDevice, std::uint32_t T_Bins, bool T_AllowDense>
        KernelAxisResult dispatchKernelBins(TDevice device, Parameters const& parameters, KernelOptions const& options)
        {
            if constexpr(T_AllowDense)
            {
                if(options.binPath == BinPath::hash)
                    return runKernelTyped<TDevice, T_Bins, BinPath::hash, ScatterChoice::shared>(
                        device,
                        parameters,
                        options.distribution);
                if(options.scatter == ScatterChoice::direct)
                    return runKernelTyped<TDevice, T_Bins, BinPath::dense, ScatterChoice::direct>(
                        device,
                        parameters,
                        options.distribution);
                return runKernelTyped<TDevice, T_Bins, BinPath::dense, ScatterChoice::shared>(
                    device,
                    parameters,
                    options.distribution);
            }
            else
            {
                return runKernelTyped<TDevice, T_Bins, BinPath::hash, ScatterChoice::shared>(
                    device,
                    parameters,
                    options.distribution);
            }
        }
    } // namespace detail

    /** Run the opt-in kernel workload: Generate -> Transform -> Sink with the selected axes.
     *
     * The feedback workload keeps its public `Flow` graph. This workload exists so the P03 scatter
     * mode, the dense/hash bin path and the bin-occupancy distribution are reachable end to end
     * from the CLI. The sink stores the chosen bin, so `histogram` is the exact per-bin count and
     * its counts sum to N.
     */
    template<typename TDevice>
    KernelAxisResult runKernelAxes(TDevice device, Parameters const& parameters, KernelOptions const& options)
    {
        if(!isSupportedKernelBins(options.bins))
            throw std::invalid_argument{"graphFlow::particle: unsupported kernel bin count"};
        if(options.binPath == BinPath::dense && options.bins > maxDenseKernelBins)
            throw std::invalid_argument{"graphFlow::particle: the dense kernel path supports at most 1024 bins"};
        if(options.binPath == BinPath::hash && options.scatter == ScatterChoice::direct)
            throw std::invalid_argument{"graphFlow::particle: the hash kernel path has no direct scatter"};
        if(options.distribution == Distribution::sparse && options.bins < 1024u)
            throw std::invalid_argument{"graphFlow::particle: sparse requires at least 1024 bins"};

        switch(options.bins)
        {
        case 64u:
            return detail::dispatchKernelBins<TDevice, 64u, true>(device, parameters, options);
        case 1024u:
            return detail::dispatchKernelBins<TDevice, 1024u, true>(device, parameters, options);
        case 4096u:
            return detail::dispatchKernelBins<TDevice, 4096u, false>(device, parameters, options);
        case 65536u:
            return detail::dispatchKernelBins<TDevice, 65536u, false>(device, parameters, options);
        default:
            throw std::invalid_argument{"graphFlow::particle: unsupported kernel bin count"};
        }
    }
} // namespace graphFlow::particle
