/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleStages.hpp"

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace graphFlow::particle
{
    /** Bin count shared by Filter, A and B in this example. */
    inline constexpr std::uint32_t FlowBins = 2u;

    /** Number of transform nodes in the particle-feedback graph. */
    inline constexpr std::uint32_t TransformNodes = 3u;

    /** Default source-admission capacity P when the caller does not configure one.
     *
     * P is a working-set knob, not a function of N: capping it at `8*C` means the pool is sized by
     * P and the transform output reserve, independent of arbitrarily large N. The value is
     * documented in README.md and can be overridden with `--live-particles`.
     */
    inline std::uint32_t defaultLiveParticles(std::uint64_t numParticles, std::uint32_t chunkSize)
    {
        std::uint64_t const cap = 8ull * chunkSize;
        std::uint64_t const live = numParticles < cap ? numParticles : cap;
        return static_cast<std::uint32_t>(live == 0u ? 1u : live);
    }

    /** Derive the chunk-pool size from the source-admission capacity P, never from N.
     *
     * The plan's model is `S = ceil(P/C) + K*R`. That is a lower bound, not a safe bound for a
     * split-routing feedback graph: every transform may split one input chunk into one chunk per
     * bin, and repeated feedback can fragment the P live particles into as many as P chunks
     * (one particle per chunk). The pool must therefore hold one chunk per live particle plus the
     * `K*R` output credits of the live transform batches:
     *
     *     S = P + K*R,   R = outputChunkBound(C, C, B),   K = TransformNodes
     *
     * Source admission caps the fill of every admitted chunk at `P - liveParticles`, so the number
     * of live particles (and therefore the number of data-carrying chunks) never exceeds P. The
     * result is independent of N; `P == 0` selects `defaultLiveParticles`. `tests/integration/
     * test_pressure.cpp` pins the N=4097, C=128, P=256 case and records why the smaller formula
     * deadlocks.
     */
    inline std::uint32_t derivePoolSize(
        std::uint64_t numParticles,
        std::uint32_t chunkSize,
        std::uint32_t liveParticles,
        std::uint32_t maxBins)
    {
        if(numParticles == 0u)
            return 1u;
        std::uint64_t live = liveParticles == 0u ? defaultLiveParticles(numParticles, chunkSize) : liveParticles;
        if(live > numParticles)
            live = numParticles;
        if(live == 0u)
            live = 1u;

        std::uint64_t const perTransform = outputChunkBound(chunkSize, chunkSize, maxBins);
        std::uint64_t const chunks = live + static_cast<std::uint64_t>(TransformNodes) * perTransform;
        if(chunks > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error{"graphFlow: chunk pool would exceed 2^32 chunks"};
        return static_cast<std::uint32_t>(chunks == 0u ? 1u : chunks);
    }

    template<typename TDevice, typename TStage>
    typename GraphRunner<TDevice>::Hooks hooksOf(TStage& stage)
    {
        using Runner = GraphRunner<TDevice>;
        return typename Runner::Hooks{
            [&stage](typename Runner::Queue const& queue, typename Runner::Batch const& batch)
            { stage.launch(queue, batch); },
            [&stage](typename Runner::Queue const& queue, ChunkDescriptor const& chunk, std::uint32_t& allocated)
            { return stage.advance(queue, chunk, allocated); },
            [&stage](ChunkDescriptor const& chunk, std::vector<ChunkDescriptor>& outputs)
            { stage.finish(chunk, outputs); }};
    }

    /** Build and run the Generate -> Filter -> {A, B} -> Sink feedback graph. */
    template<typename TDevice>
    FeedbackResult runParticleFeedback(TDevice device, Parameters const& parameters)
    {
        Config config;
        config.chunkCapacity = parameters.chunkSize;
        config.liveParticles = parameters.liveParticles == 0u
                                   ? defaultLiveParticles(parameters.numParticles, parameters.chunkSize)
                                   : parameters.liveParticles;
        if(config.liveParticles > parameters.numParticles && parameters.numParticles != 0u)
            config.liveParticles = static_cast<std::uint32_t>(parameters.numParticles);
        config.numChunks = derivePoolSize(
            parameters.numParticles,
            parameters.chunkSize,
            config.liveParticles,
            FlowBins);
        config.tileSize = parameters.tileSize;
        config.maxBins = FlowBins;
        config.seed = parameters.seed;
        config.maxVisits = parameters.maxVisits;
        config.numParticles = parameters.numParticles;
        config.batchChunks = parameters.batchChunks;
        config.validate();

        std::uint32_t const threads = GraphRunner<TDevice>::threadsPerBlock(device, config.tileSize);
        // The transform stages take the logical tile width and resolve the real block extent
        // themselves; generate and sink stay one block per chunk with the resolved extent.
        ChunkPool<TDevice> pool{device, config.numChunks, config.chunkCapacity};

        // The result buffer is indexed by particle id; the sentinel detects missing deliveries.
        std::uint64_t const resultSize = parameters.numParticles == 0u ? 1u : parameters.numParticles;
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});

        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        FilterPolicy filterPolicy{parameters.seed};
        ProcessAPolicy processAPolicy{parameters.seed, parameters.maxVisits};
        ProcessBPolicy processBPolicy{parameters.seed, parameters.maxVisits};

        GenerateStage<TDevice> generate{pool, parameters.seed, threads};
        TransformStage<TDevice, FilterPolicy> filter{device, pool, filterPolicy, capped, config.tileSize};
        TransformStage<TDevice, ProcessAPolicy> processA{device, pool, processAPolicy, capped, config.tileSize};
        TransformStage<TDevice, ProcessBPolicy> processB{device, pool, processBPolicy, capped, config.tileSize};
        SinkStage<TDevice> sink{pool, result, threads};

        GraphRunner<TDevice> runner{device, pool};
        // Node indices are stable: 0 generate, 1 filter, 2 A, 3 B, 4 sink.
        std::uint32_t const nGenerate
            = runner.addNode("generate", NodeRole::source, 1u, {1}, hooksOf<TDevice>(generate));
        std::uint32_t const nFilter
            = runner.addNode("filter", NodeRole::transform, FlowBins, {2, 3}, hooksOf<TDevice>(filter));
        std::uint32_t const nProcessA
            = runner.addNode("processA", NodeRole::transform, FlowBins, {4, 1}, hooksOf<TDevice>(processA));
        std::uint32_t const nProcessB
            = runner.addNode("processB", NodeRole::transform, FlowBins, {4, 1}, hooksOf<TDevice>(processB));
        std::uint32_t const nSink
            = runner.addNode("sink", NodeRole::sink, 1u, {}, hooksOf<TDevice>(sink));
        alpaka::unused(nFilter, nProcessA, nProcessB, nSink);

        runner.setSource(nGenerate, parameters.numParticles, config.liveParticles);
        runner.setStallTimeout(std::chrono::milliseconds{parameters.stallTimeoutMs});
        SchedulerStats const stats = runner.run();

        std::vector<std::uint32_t> weights(static_cast<std::size_t>(resultSize), 0u);
        auto cappedHost = alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto readQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(readQueue, weights, result);
            alpaka::onHost::memcpy(readQueue, cappedHost, capped);
        }
        std::uint32_t const cappedValue = cappedHost.data()[0u];
        weights.resize(static_cast<std::size_t>(parameters.numParticles));

        FeedbackResult output;
        output.histogram = histogramOf(weights);
        output.weights = std::move(weights);
        output.cappedParticles = cappedValue;
        output.sunkParticles = sink.sunkParticles();
        output.liveParticles = config.liveParticles;
        output.stats = stats;
        return output;
    }
} // namespace graphFlow::particle
