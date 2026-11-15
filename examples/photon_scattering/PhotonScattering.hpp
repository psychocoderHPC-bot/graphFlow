/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonFunctors.hpp"
#include "PhotonTypes.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_ONEAPI
#    include <sycl/sycl.hpp>
#endif

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string_view>
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
    template<typename TScalar, typename TDevice>
    ScatteringResultT<TScalar> runPhotonScatteringAs(
        TDevice device,
        Parameters const& parameters,
        graphFlow::Tuning tuning)
    {
        validateParameters(parameters);

        std::uint64_t const resultSize = parameters.numPhotons == 0u ? 1u : parameters.numPhotons;
        auto records = alpaka::onHost::alloc<TerminalRecordT<TScalar>>(device, alpaka::Vec{resultSize});
        auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});

        std::uint64_t const traceIds
            = parameters.trace.enabled ? std::min<std::uint64_t>(parameters.numPhotons, parameters.trace.maxPhotons)
                                       : 0u;
        std::uint64_t const traceSlots = traceIds == 0u ? 1u : traceIds;
        std::uint32_t const traceVertices = parameters.trace.enabled ? parameters.trace.maxVertices : 1u;
        std::uint64_t const traceVertexSlots = traceSlots * traceVertices;
        auto traceBuffer = alpaka::onHost::alloc<TraceVertexT<TScalar>>(device, alpaka::Vec{traceVertexSlots});
        auto traceCounts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{traceSlots});
        auto traceFlags = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{traceSlots});
        {
            std::vector<TerminalRecordT<TScalar>> initial(
                static_cast<std::size_t>(resultSize),
                TerminalRecordT<TScalar>{missingEvent, 0u, Vec3T<TScalar>{}, Vec3T<TScalar>{}});
            std::vector<std::uint32_t> const zero(static_cast<std::size_t>(resultSize), 0u);
            std::vector<std::uint32_t> const traceZero(static_cast<std::size_t>(traceSlots), 0u);
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(initQueue, records, initial);
            alpaka::onHost::memcpy(initQueue, visits, zero);
            alpaka::onHost::memcpy(initQueue, traceCounts, traceZero);
            alpaka::onHost::memcpy(initQueue, traceFlags, traceZero);
        }

        SlabT<TScalar> const slab = slabOf<TScalar>(parameters);

        using TraceVertexView = decltype(traceBuffer.getView());
        using TraceCountView = decltype(traceCounts.getView());
        using TraceFlagView = decltype(traceFlags.getView());
        using Trace = TraceWriter<TraceVertexView, TraceCountView, TraceFlagView, TScalar>;
        Trace const trace{
            traceBuffer.getView(),
            traceCounts.getView(),
            traceFlags.getView(),
            traceVertices,
            static_cast<std::uint32_t>(traceIds)};

        graphFlow::Flow<PhotonT<TScalar>, TDevice> flow{device};
        auto source = flow.source("generate", GeneratePhotonT<Trace, TScalar>{trace});
        auto flight = flow.transform("propose flight", ProposeFlightT<TScalar>{slab, parameters.maxScatters});
        auto route = flow.partition("advance and route", ClassifyEventT<TScalar>{}, AdvanceToEventT<TScalar>{slab});
        auto scatter = flow.transform("scatter", ScatterT<Trace, TScalar>{slab, parameters.mode, trace});
        using RecordView = decltype(records.getView());
        using VisitView = decltype(visits.getView());
        auto terminal = flow.sink(
            "terminal",
            RecordTerminal<RecordView, VisitView, Trace, TScalar>{records.getView(), visits.getView(), trace});

        flow.connect(source, flight);
        flow.connect(flight, route);
        flow.connect(route.template out<Event::scatter>(), scatter);
        flow.connect(scatter, flight);
        flow.connect(route.template out<Event::transmitted>(), terminal);
        flow.connect(route.template out<Event::reflected>(), terminal);
        flow.connect(route.template out<Event::absorbed>(), terminal);
        flow.connect(route.template out<Event::capped>(), terminal);
        flow.configure(tuning);

        ScatteringResultT<TScalar> result;
        result.stats = flow.run(source, parameters.numPhotons);

        std::vector<TerminalRecordT<TScalar>> hostRecords(static_cast<std::size_t>(resultSize));
        std::vector<std::uint32_t> hostVisits(static_cast<std::size_t>(resultSize), 0u);
        std::vector<std::uint32_t> hostTraceCounts(static_cast<std::size_t>(traceSlots), 0u);
        std::vector<std::uint32_t> hostTraceFlags(static_cast<std::size_t>(traceSlots), 0u);
        std::vector<TraceVertexT<TScalar>> hostTraceBuffer(static_cast<std::size_t>(traceVertexSlots));
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
            TerminalRecordT<TScalar> const& record = result.terminal[id];
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
                PhotonTraceT<TScalar> trace;
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

    /** Convenience overload for a caller that does not resolve the tuning separately. */
    template<typename TScalar, typename TDevice>
    ScatteringResultT<TScalar> runPhotonScatteringAs(TDevice device, Parameters const& parameters)
    {
        return runPhotonScatteringAs<TScalar>(std::move(device), parameters, tuningOf(parameters));
    }

    /** Double-precision run used by the CPU, CUDA and HIP backends. */
    template<typename TDevice>
    ScatteringResult runPhotonScattering(TDevice device, Parameters const& parameters, graphFlow::Tuning tuning)
    {
        return runPhotonScatteringAs<double>(std::move(device), parameters, tuning);
    }

    /** Convenience overload for callers that do not resolve the tuning separately. */
    template<typename TDevice>
    ScatteringResult runPhotonScattering(TDevice device, Parameters const& parameters)
    {
        return runPhotonScattering(std::move(device), parameters, tuningOf(parameters));
    }

    /** Convert a single-precision run into the public double-precision result type. */
    inline ScatteringResult toDouble(ScatteringResultT<float> const& source)
    {
        ScatteringResult result;
        result.terminal.reserve(source.terminal.size());
        for(TerminalRecordT<float> const& record : source.terminal)
        {
            result.terminal.push_back(
                TerminalRecord{
                    record.event,
                    record.scatterCount,
                    Vec3{
                        static_cast<double>(record.position.x),
                        static_cast<double>(record.position.y),
                        static_cast<double>(record.position.z)},
                    Vec3{
                        static_cast<double>(record.direction.x),
                        static_cast<double>(record.direction.y),
                        static_cast<double>(record.direction.z)}});
        }
        result.terminalVisits = source.terminalVisits;
        result.histograms = source.histograms;
        for(std::uint32_t event = 0u; event < eventBins; ++event)
            result.eventCounts[event] = source.eventCounts[event];
        result.capped = source.capped;
        result.terminated = source.terminated;
        result.inconsistent = source.inconsistent;
        result.liveParticles = source.liveParticles;
        result.traceTruncated = source.traceTruncated;
        result.stats = source.stats;
        result.traces.reserve(source.traces.size());
        for(PhotonTraceT<float> const& sourceTrace : source.traces)
        {
            PhotonTrace trace;
            trace.id = sourceTrace.id;
            trace.count = sourceTrace.count;
            trace.truncated = sourceTrace.truncated;
            trace.vertices.reserve(sourceTrace.vertices.size());
            for(TraceVertexT<float> const& vertex : sourceTrace.vertices)
            {
                trace.vertices.push_back(
                    TraceVertex{
                        vertex.kind,
                        vertex.scatterCount,
                        Vec3{
                            static_cast<double>(vertex.position.x),
                            static_cast<double>(vertex.position.y),
                            static_cast<double>(vertex.position.z)},
                        Vec3{
                            static_cast<double>(vertex.direction.x),
                            static_cast<double>(vertex.direction.y),
                            static_cast<double>(vertex.direction.z)}});
            }
            result.traces.push_back(std::move(trace));
        }
        return result;
    }

    /** True only when the caller set `GRAPHFLOW_SYCL_FP64_EMULATION=1` in the environment.
     *
     * The variable is read at call time, so a process can query it before it creates a SYCL queue.
     * A value other than the literal `1` leaves the default single-precision path in place.
     */
    inline bool syclFp64EmulationRequested()
    {
        char const* const value = std::getenv("GRAPHFLOW_SYCL_FP64_EMULATION");
        return value != nullptr && std::string_view{value} == "1";
    }

    /** True when the selected oneAPI device reports at least one double floating-point config.
     *
     * The Arc A770 reports an empty `double_fp_config` by default and a non-empty one only when
     * `IGC_EnableDPEmulation=1` and `OverrideDefaultFP64Settings=1` were present when the SYCL
     * runtime created the device. The query is deliberately made on the device rather than
     * trusting the environment alone: `OverrideDefaultFP64Settings=1` on its own does not make
     * double kernels correct. Non-oneAPI devices always return true.
     */
    template<typename TDevice>
    bool syclDeviceSupportsDouble([[maybe_unused]] TDevice const& device)
    {
#if ALPAKA_LANG_ONEAPI
        if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>)
        {
            auto [syclDevice, syclContext] = device.getNativeHandle();
            (void) syclContext;
            return !syclDevice.template get_info<sycl::info::device::double_fp_config>().empty();
        }
#endif
        return true;
    }

    /** Validate an explicit FP64-emulation request or throw instead of falling back to fp32.
     *
     * Both IGC variables must be present when the process starts, because the Intel runtime reads
     * them before the first queue is created. A missing variable or an empty `double_fp_config`
     * is an error: silently running single precision would make the opt-in a lie.
     */
    template<typename TDevice>
    void requireSyclFp64Emulation(TDevice const& device)
    {
        char const* const enable = std::getenv("IGC_EnableDPEmulation");
        if(enable == nullptr || std::string_view{enable} != "1")
            throw std::runtime_error{"GRAPHFLOW_SYCL_FP64_EMULATION=1 also requires IGC_EnableDPEmulation=1"};
        char const* const overrideValue = std::getenv("OverrideDefaultFP64Settings");
        if(overrideValue == nullptr || std::string_view{overrideValue} != "1")
            throw std::runtime_error{"GRAPHFLOW_SYCL_FP64_EMULATION=1 also requires OverrideDefaultFP64Settings=1"};
        if(!syclDeviceSupportsDouble(device))
            throw std::runtime_error{"GRAPHFLOW_SYCL_FP64_EMULATION=1 requested, but the device reports an empty "
                                     "double_fp_config; set both IGC variables before the process starts"};
    }

    /** Double-precision oneAPI photon run, defined in its own translation unit.
     *
     * Keeping the definition in `PhotonScatteringFp64OneApi.cpp` puts the double kernels in a
     * separate SYCL device image from the default fp32 path. If both were instantiated in the
     * caller translation unit, loading the fp32 device image would require the fp64 aspect before
     * `runPhotonScatteringAuto()` can check the runtime opt-in. The specialization is compiled at
     * `-O0` for the icpx 2026.2 FP64-emulation kernel-argument miscompile.
     */
    template<typename TDevice>
    ScatteringResult runPhotonScatteringDoubleOneApi(
        TDevice device,
        Parameters const& parameters,
        graphFlow::Tuning tuning);

#if ALPAKA_LANG_ONEAPI
    /** Concrete oneAPI Intel GPU device type used by the double-path specialization. */
    using OneApiSyclDevice
        = decltype(alpaka::onHost::makeDeviceSelector(
                       alpaka::onHost::DeviceSpec{alpaka::api::OneApi{}, alpaka::deviceKind::intelGpu})
                       .makeDevice(0u));

    template<>
    ScatteringResult runPhotonScatteringDoubleOneApi<OneApiSyclDevice>(
        OneApiSyclDevice device,
        Parameters const& parameters,
        graphFlow::Tuning tuning);
#endif

    /** Run with the precision the selected device supports.
     *
     * Intel GPUs exposed by the oneAPI backend do not advertise the SYCL fp64 aspect by default,
     * so the oneAPI path runs the same physics in single precision and converts the result back to
     * the public double type. Double precision is available only behind an explicit opt-in:
     * `GRAPHFLOW_SYCL_FP64_EMULATION=1` together with `IGC_EnableDPEmulation=1` and
     * `OverrideDefaultFP64Settings=1` in the process environment. A half-configured opt-in throws
     * from `requireSyclFp64Emulation()` rather than running fp32. The CPU, CUDA and HIP paths keep
     * double.
     *
     * The oneAPI photon kernels use block barriers and atomics only, no warp or sub-group
     * collectives. They therefore do not exercise the DG2 sub-group reduction path that can hang
     * with `reqd_sub_group_size` (upstream issue #1036).
     */
    template<typename TDevice>
    ScatteringResult runPhotonScatteringAuto(TDevice device, Parameters const& parameters, graphFlow::Tuning tuning)
    {
        if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>)
        {
            if(syclFp64EmulationRequested())
            {
                requireSyclFp64Emulation(device);
                return runPhotonScatteringDoubleOneApi(std::move(device), parameters, tuning);
            }
            return toDouble(runPhotonScatteringAs<float>(std::move(device), parameters, tuning));
        }
        else
            return runPhotonScattering(std::move(device), parameters, tuning);
    }
} // namespace graphFlow::photon
