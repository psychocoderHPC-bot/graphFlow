/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Cli.hpp"
#include "KernelAxes.hpp"
#include "ParticleFeedback.hpp"
#include "graphFlow/core/Backend.hpp"
#include "graphFlow/detail/QueueProbe.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void printHistogram(
        std::vector<graphFlow::particle::HistogramRow> const& histogram,
        std::uint64_t numParticles,
        std::uint64_t cappedParticles = 0u)
    {
        for(auto const& row : histogram)
            std::cout << row.weight << ',' << row.count << '\n';
        if(numParticles == 0u)
            return;
        if(cappedParticles != 0u)
            std::cout << "# non-terminating " << cappedParticles << '\n';
    }

    template<typename TDevice>
    int run(
        graphFlow::cli::ResolvedOptions const& resolved,
        graphFlow::Backend backend,
        TDevice device,
        std::string const& deviceName)
    {
        graphFlow::particle::Parameters const& parameters = resolved.parameters;
        graphFlow::particle::FeedbackResult const result
            = graphFlow::particle::runParticleFeedback(device, parameters, resolved.tuning);

        std::cout << "# graphFlow_particle_feedback backend=" << graphFlow::backendName(backend)
                  << " device=" << parameters.deviceIndex << " device-name=" << deviceName
                  << " particles=" << parameters.numParticles << " chunk-size=" << parameters.chunkSize
                  << " live-particles=" << result.liveParticles << " tile-size=" << parameters.tileSize
                  << " batch-chunks=" << parameters.batchChunks << " frame-cap=" << parameters.frameCap
                  << " seed=" << parameters.seed << " max-visits=" << parameters.maxVisits << '\n';
        std::cout << "# pool-chunks=" << result.stats.poolChunks << " peak-live-chunks=" << result.stats.peakLiveChunks
                  << " peak-reserved-chunks=" << result.stats.peakReservedChunks << '\n';
        std::cout << "# scheduler " << graphFlow::formatSchedulerStats(result.stats) << '\n';
        std::cout << "# weight,count\n";
        printHistogram(result.histogram, parameters.numParticles, result.cappedParticles);
        return EXIT_SUCCESS;
    }

    template<typename TDevice>
    int runKernel(
        graphFlow::cli::ResolvedOptions const& resolved,
        graphFlow::Backend backend,
        TDevice device,
        std::string const& deviceName)
    {
        graphFlow::particle::Parameters const& parameters = resolved.parameters;
        graphFlow::particle::KernelOptions const& kernel = resolved.kernel;
        graphFlow::particle::KernelAxisResult const result
            = graphFlow::particle::runKernelAxes(device, parameters, kernel);

        std::cout << "# graphFlow_particle_feedback backend=" << graphFlow::backendName(backend)
                  << " device=" << parameters.deviceIndex << " device-name=" << deviceName
                  << " workload=kernel particles=" << parameters.numParticles << " chunk-size=" << parameters.chunkSize
                  << " live-particles=" << result.liveParticles << " tile-size=" << parameters.tileSize
                  << " batch-chunks=" << parameters.batchChunks << " frame-cap=" << parameters.frameCap
                  << " seed=" << parameters.seed
                  << " scatter=" << graphFlow::particle::scatterChoiceName(kernel.scatter)
                  << " bin-path=" << graphFlow::particle::binPathName(kernel.binPath) << " bins=" << kernel.bins
                  << " distribution=" << graphFlow::particle::distributionName(kernel.distribution) << '\n';
        std::cout << "# pool-chunks=" << result.stats.poolChunks << " peak-live-chunks=" << result.stats.peakLiveChunks
                  << " peak-reserved-chunks=" << result.stats.peakReservedChunks << '\n';
        std::cout << "# scheduler " << graphFlow::formatSchedulerStats(result.stats) << '\n';
        std::cout << "# weight,count\n";
        printHistogram(result.histogram, parameters.numParticles);
        return EXIT_SUCCESS;
    }
} // namespace

int main(int argc, char** argv)
{
    graphFlow::cli::Options options;
    try
    {
        options = graphFlow::cli::parseOptions(argc, argv);
    }
    catch(std::exception const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        graphFlow::cli::usage(std::cerr);
        return 2;
    }

    if(options.help)
    {
        graphFlow::cli::usage(std::cout);
        return EXIT_SUCCESS;
    }

    graphFlow::Backend backend;
    try
    {
        backend = graphFlow::requireBackend(options.backend);
    }
    catch(graphFlow::BackendError const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        graphFlow::cli::usage(std::cerr);
        return 2;
    }

    try
    {
        return graphFlow::withBackendDevice(
            backend,
            options.deviceIndex,
            [&](auto device, graphFlow::Backend selected, std::string const& deviceName)
            {
                graphFlow::cli::ResolvedOptions const resolved
                    = graphFlow::cli::resolveOptions(options, device.getDeviceProperties());
                std::cout << "# selected-backend=" << graphFlow::backendName(selected)
                          << " selected-device=" << options.deviceIndex << " device-name=" << deviceName << '\n';
                std::cout << "# policy " << resolved.policy.description << '\n';
                // The per-chunk transform options are accepted for parity with the photon example
                // but cannot take effect here: no graph in this example registers a transform
                // node, so `Flow` never builds a per-chunk lane. Say so instead of silently
                // accepting a benchmark knob that does nothing.
                if(options.chunkKernelSchedule != "off" || options.chunkQueueDepth != 1u
                   || options.chunkLatencyCapUs != 0u)
                    std::cout << "# note: chunk-kernel-schedule, chunk-queue-depth and chunk-latency-cap-us have "
                                 "no effect in this example: its graph has no transform node\n";
                graphFlow::QueueProbeResult const probe = graphFlow::probeQueueIndependence(device);
                std::cout << "# queue-probe overlap=" << graphFlow::toString(probe.overlap)
                          << " queues-alias=" << (probe.queuesAlias ? "yes" : "no")
                          << " second-before-first=" << (probe.secondCompletedWhileFirstInFlight ? "yes" : "no")
                          << " poll-us=" << probe.pollDurationUs << '\n';
                if(resolved.workload == graphFlow::particle::Workload::kernel)
                    return runKernel(resolved, selected, device, deviceName);
                return run(resolved, selected, device, deviceName);
            });
    }
    catch(graphFlow::BackendUnavailable const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        return 2;
    }
    catch(std::out_of_range const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        return 2;
    }
    catch(std::invalid_argument const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        return 2;
    }
    catch(std::exception const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        return 1;
    }
}
