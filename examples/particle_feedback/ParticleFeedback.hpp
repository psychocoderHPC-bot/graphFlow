/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleStages.hpp"

#include <alpaka/alpaka.hpp>

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

    /** Derive a deadlock-free chunk-pool size from the particle count.
     *
     * With feedback, a bin split turns one input chunk into up to two output chunks, so the number
     * of live chunks grows during the run. Live chunks are disjoint in particles, hence the tight
     * worst case is one chunk per particle. With `S >= N + (K + 1) * R + 8` the pool always offers
     * at least `R` unreserved chunks, so a ready transform can always reserve its outputs and the
     * scheduler cannot deadlock on chunk credits. The conservative provisioning from the design
     * document (`ceil(P/C) + K*R`) is not sufficient for split-heavy feedback and is recorded as a
     * follow-up in README.md.
     */
    inline std::uint32_t derivePoolSize(std::uint64_t numParticles, std::uint32_t chunkSize)
    {
        std::uint64_t const reserve = outputChunkBound(chunkSize, chunkSize, FlowBins);
        std::uint64_t const chunks = numParticles + (TransformNodes + 1u) * reserve + 8u;
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
        config.numChunks = derivePoolSize(parameters.numParticles, parameters.chunkSize);
        config.tileSize = parameters.tileSize;
        config.maxBins = FlowBins;
        config.seed = parameters.seed;
        config.maxVisits = parameters.maxVisits;
        config.numParticles = parameters.numParticles;
        config.validate();

        std::uint32_t const threads = GraphRunner<TDevice>::threadsPerBlock(device, config.tileSize);
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
        TransformStage<TDevice, FilterPolicy> filter{device, pool, filterPolicy, capped, threads};
        TransformStage<TDevice, ProcessAPolicy> processA{device, pool, processAPolicy, capped, threads};
        TransformStage<TDevice, ProcessBPolicy> processB{device, pool, processBPolicy, capped, threads};
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

        runner.setSource(nGenerate, parameters.numParticles);
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
        output.stats = stats;
        return output;
    }
} // namespace graphFlow::particle
