/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleFunctors.hpp"
#include "ParticleTypes.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace graphFlow::particle
{
    /** Sentinel left in the result buffer when a particle never reaches a sink. */
    inline constexpr std::uint32_t missingWeight = 0xDEAD'BEEFu;

    struct FeedbackResult
    {
        std::vector<std::uint32_t> weights; // final weight indexed by particle id
        std::vector<HistogramRow> histogram; // sorted by weight, run-length encoded
        std::uint64_t cappedParticles = 0u; // stopped by the --max-visits guard
        std::uint64_t sunkParticles = 0u; // host-side sink conservation diagnostic
        std::uint32_t liveParticles = 0u; // configured source-admission capacity P
        SchedulerStats stats;
    };

    /** Run the Generate -> Filter -> {A, B} -> sink feedback graph through the public `Flow` API.
     *
     * The topology is declared with opaque node handles and named outcome ports; no route table,
     * node id, pool size, or runner configuration is constructed here. `Tuning` carries C, P, T,
     * Q and the frame cap. The pool and the output-credit reserve are derived by `Flow` from the
     * registered bins, so the A/B three-way split reserves one output chunk per outcome.
     *
     * The result views are application-owned. They are initialized before the run and copied back
     * on a blocking queue after `Flow::run()` has retired every queue and pool chunk.
     */
    template<typename TDevice>
    FeedbackResult runParticleFeedback(TDevice device, Parameters const& parameters, graphFlow::Tuning tuning)
    {
        // Index by particle id; the sentinel detects a particle that never reached a sink.
        std::uint64_t const resultSize = parameters.numParticles == 0u ? 1u : parameters.numParticles;
        auto weights = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});
        auto cappedFlags = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, weights, missingWeight);
            alpaka::onHost::memset(initQueue, cappedFlags, std::uint8_t{0});
        }

        graphFlow::Flow<graphFlow::Particle, TDevice> flow{device};
        auto source = flow.source("generate", GenerateParticles{parameters.seed});
        auto filter = flow.partition("filter", FilterByProperty{parameters.seed});
        auto processA = flow.partition(
            "process A",
            SelectOutcome{parameters.seed, rng::purposeProcessA, parameters.maxVisits},
            UpdateWeight{1000u});
        auto processB = flow.partition(
            "process B",
            SelectOutcome{parameters.seed, rng::purposeProcessB, parameters.maxVisits},
            UpdateWeight{1u});
        auto sink = flow.sink("record weight", RecordWeight<decltype(weights.getView())>{weights.getView()});
        auto capped = flow.sink(
            "record capped weight",
            RecordCapped<decltype(weights.getView()), decltype(cappedFlags.getView())>{
                weights.getView(),
                cappedFlags.getView()});

        flow.connect(source, filter);
        flow.connect(filter.template out<0u>(), processA);
        flow.connect(filter.template out<1u>(), processB);
        flow.connect(processA.template out<Outcome::sink>(), sink);
        flow.connect(processB.template out<Outcome::sink>(), sink);
        flow.connect(processA.template out<Outcome::feedback>(), filter);
        flow.connect(processB.template out<Outcome::feedback>(), filter);
        flow.connect(processA.template out<Outcome::capped>(), capped);
        flow.connect(processB.template out<Outcome::capped>(), capped);
        flow.configure(tuning);

        SchedulerStats const stats = flow.run(source, parameters.numParticles);

        std::vector<std::uint32_t> hostWeights(static_cast<std::size_t>(resultSize), 0u);
        std::vector<std::uint32_t> hostCapped(static_cast<std::size_t>(resultSize), 0u);
        {
            auto readQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(readQueue, hostWeights, weights);
            alpaka::onHost::memcpy(readQueue, hostCapped, cappedFlags);
        }

        FeedbackResult output;
        output.weights.assign(
            hostWeights.begin(),
            hostWeights.begin() + static_cast<std::ptrdiff_t>(parameters.numParticles));
        output.histogram = histogramOf(output.weights);
        output.cappedParticles = 0u;
        for(std::size_t id = 0u; id < static_cast<std::size_t>(parameters.numParticles); ++id)
        {
            output.cappedParticles += hostCapped[id];
            if(hostWeights[id] != missingWeight)
                ++output.sunkParticles;
        }
        output.liveParticles = tuning.liveParticles.value_or(parameters.liveParticles);
        if(parameters.numParticles == 0u)
            output.liveParticles = 1u; // keep the historical empty-run admission header
        output.stats = stats;
        return output;
    }

    /** Convenience overload for callers that do not resolve the tuning separately. */
    template<typename TDevice>
    FeedbackResult runParticleFeedback(TDevice device, Parameters const& parameters)
    {
        return runParticleFeedback(std::move(device), parameters, tuningOf(parameters));
    }
} // namespace graphFlow::particle
