/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonFunctors.hpp"
#include "PhotonTypes.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace graphFlow::photon
{
    /** Run the photon transport graph through the public `Flow` API.
     *
     * Topology: source -> propose flight -> advance and route -> scatter -> propose flight, with
     * the four terminal outcome bins of the partition feeding one shared sink. The result views
     * are application-owned and copied back only after `Flow::run()` has retired every queue and
     * pool chunk.
     */
    template<typename TDevice>
    ScatteringResult runPhotonScattering(TDevice device, Parameters const& parameters, graphFlow::Tuning tuning)
    {
        validateParameters(parameters);

        std::uint64_t const resultSize = parameters.numPhotons == 0u ? 1u : parameters.numPhotons;
        auto records = alpaka::onHost::alloc<TerminalRecord>(device, alpaka::Vec{resultSize});
        auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});

        std::uint64_t const traceIds
            = parameters.trace.enabled ? std::min<std::uint64_t>(parameters.numPhotons, parameters.trace.maxPhotons)
                                       : 0u;
        std::uint64_t const traceSlots = traceIds == 0u ? 1u : traceIds;
        std::uint32_t const traceVertices = parameters.trace.enabled ? parameters.trace.maxVertices : 1u;
        std::uint64_t const traceVertexSlots = traceSlots * traceVertices;
        auto traceBuffer = alpaka::onHost::alloc<TraceVertex>(device, alpaka::Vec{traceVertexSlots});
        auto traceCounts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{traceSlots});
        auto traceFlags = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{traceSlots});
        {
            std::vector<TerminalRecord> initial(
                static_cast<std::size_t>(resultSize),
                TerminalRecord{missingEvent, 0u, Vec3{}, Vec3{}});
            std::vector<std::uint32_t> const zero(static_cast<std::size_t>(resultSize), 0u);
            std::vector<std::uint32_t> const traceZero(static_cast<std::size_t>(traceSlots), 0u);
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(initQueue, records, initial);
            alpaka::onHost::memcpy(initQueue, visits, zero);
            alpaka::onHost::memcpy(initQueue, traceCounts, traceZero);
            alpaka::onHost::memcpy(initQueue, traceFlags, traceZero);
        }

        Slab const slab = slabOf(parameters);

        using TraceVertexView = decltype(traceBuffer.getView());
        using TraceCountView = decltype(traceCounts.getView());
        using TraceFlagView = decltype(traceFlags.getView());
        using Trace = TraceWriter<TraceVertexView, TraceCountView, TraceFlagView>;
        Trace const trace{
            traceBuffer.getView(),
            traceCounts.getView(),
            traceFlags.getView(),
            traceVertices,
            static_cast<std::uint32_t>(traceIds)};

        graphFlow::Flow<Photon, TDevice> flow{device};
        auto source = flow.source("generate", GeneratePhotonT<Trace>{trace});
        auto flight = flow.transform("propose flight", ProposeFlight{slab, parameters.maxScatters});
        auto route = flow.partition("advance and route", ClassifyEvent{}, AdvanceToEvent{slab});
        auto scatter = flow.transform("scatter", ScatterT<Trace>{slab, parameters.mode, trace});
        using RecordView = decltype(records.getView());
        using VisitView = decltype(visits.getView());
        auto terminal = flow.sink(
            "terminal",
            RecordTerminal<RecordView, VisitView, Trace>{records.getView(), visits.getView(), trace});

        flow.connect(source, flight);
        flow.connect(flight, route);
        flow.connect(route.template out<Event::scatter>(), scatter);
        flow.connect(scatter, flight);
        flow.connect(route.template out<Event::transmitted>(), terminal);
        flow.connect(route.template out<Event::reflected>(), terminal);
        flow.connect(route.template out<Event::absorbed>(), terminal);
        flow.connect(route.template out<Event::capped>(), terminal);
        flow.configure(tuning);

        ScatteringResult result;
        result.stats = flow.run(source, parameters.numPhotons);

        std::vector<TerminalRecord> hostRecords(static_cast<std::size_t>(resultSize));
        std::vector<std::uint32_t> hostVisits(static_cast<std::size_t>(resultSize), 0u);
        std::vector<std::uint32_t> hostTraceCounts(static_cast<std::size_t>(traceSlots), 0u);
        std::vector<std::uint32_t> hostTraceFlags(static_cast<std::size_t>(traceSlots), 0u);
        std::vector<TraceVertex> hostTraceBuffer(static_cast<std::size_t>(traceVertexSlots));
        {
            auto readQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(readQueue, hostRecords, records);
            alpaka::onHost::memcpy(readQueue, hostVisits, visits);
            if(parameters.trace.enabled)
            {
                alpaka::onHost::memcpy(readQueue, hostTraceCounts, traceCounts);
                alpaka::onHost::memcpy(readQueue, hostTraceFlags, traceFlags);
                alpaka::onHost::memcpy(readQueue, hostTraceBuffer, traceBuffer);
            }
        }

        result.terminal.assign(
            hostRecords.begin(),
            hostRecords.begin() + static_cast<std::ptrdiff_t>(parameters.numPhotons));
        result.terminalVisits.assign(
            hostVisits.begin(),
            hostVisits.begin() + static_cast<std::ptrdiff_t>(parameters.numPhotons));

        std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> histogram;
        for(std::size_t id = 0u; id < result.terminal.size(); ++id)
        {
            TerminalRecord const& record = result.terminal[id];
            bool const delivered = result.terminalVisits[id] == 1u && record.event != missingEvent;
            if(!delivered)
            {
                ++result.inconsistent;
                continue;
            }
            ++result.terminated;
            std::uint32_t const event = static_cast<std::uint32_t>(record.event);
            if(event >= eventBins)
            {
                ++result.inconsistent;
                continue;
            }
            ++result.eventCounts[event];
            if(record.event == Event::capped)
                ++result.capped;
            ++histogram[std::make_pair(event, record.scatterCount)];
        }
        result.histograms.reserve(histogram.size());
        for(auto const& [key, count] : histogram)
            result.histograms.push_back(OutcomeHistogramRow{static_cast<Event>(key.first), key.second, count});

        if(parameters.trace.enabled)
        {
            result.traces.reserve(static_cast<std::size_t>(traceIds));
            for(std::uint64_t id = 0u; id < traceIds; ++id)
            {
                std::uint32_t const count = hostTraceCounts[static_cast<std::size_t>(id)];
                PhotonTrace trace;
                trace.id = id;
                trace.count = count;
                trace.truncated = hostTraceFlags[static_cast<std::size_t>(id)] != 0u;
                trace.vertices.assign(
                    hostTraceBuffer.begin() + static_cast<std::ptrdiff_t>(id * traceVertices),
                    hostTraceBuffer.begin() + static_cast<std::ptrdiff_t>(id * traceVertices + count));
                if(trace.truncated)
                    ++result.traceTruncated;
                result.traces.push_back(std::move(trace));
            }
        }

        result.liveParticles = tuning.liveParticles.value_or(parameters.liveParticles);
        if(parameters.numPhotons == 0u)
            result.liveParticles = 1u;
        return result;
    }

    /** Convenience overload for callers that do not resolve the tuning separately. */
    template<typename TDevice>
    ScatteringResult runPhotonScattering(TDevice device, Parameters const& parameters)
    {
        return runPhotonScattering(std::move(device), parameters, tuningOf(parameters));
    }
} // namespace graphFlow::photon
