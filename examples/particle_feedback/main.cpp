/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Cli.hpp"
#include "ParticleFeedback.hpp"

#include "graphFlow/core/Backend.hpp"
#include "graphFlow/detail/QueueProbe.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    void printHistogram(graphFlow::particle::FeedbackResult const& result, std::uint64_t numParticles)
    {
        for(auto const& row : result.histogram)
            std::cout << row.weight << ',' << row.count << '\n';
        if(numParticles == 0u)
            return;
        if(result.cappedParticles != 0u)
            std::cout << "# non-terminating " << result.cappedParticles << '\n';
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
            = graphFlow::particle::runParticleFeedback(device, parameters);

        std::cout << "# graphFlow_particle_feedback backend=" << graphFlow::backendName(backend)
                  << " device=" << parameters.deviceIndex << " device-name=" << deviceName
                  << " particles=" << parameters.numParticles
                  << " chunk-size=" << parameters.chunkSize
                  << " live-particles=" << result.liveParticles
                  << " tile-size=" << parameters.tileSize
                  << " batch-chunks=" << parameters.batchChunks
                  << " seed=" << parameters.seed
                  << " max-visits=" << parameters.maxVisits << '\n';
        std::cout << "# pool-chunks=" << result.stats.poolChunks
                  << " peak-live-chunks=" << result.stats.peakLiveChunks
                  << " peak-reserved-chunks=" << result.stats.peakReservedChunks << '\n';
        std::cout << "# weight,count\n";
        printHistogram(result, parameters.numParticles);
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
                          << " selected-device=" << options.deviceIndex
                          << " device-name=" << deviceName << '\n';
                std::cout << "# policy " << resolved.policy.description << '\n';
                graphFlow::QueueProbeResult const probe = graphFlow::probeQueueIndependence(device);
                std::cout << "# queue-probe overlap=" << graphFlow::toString(probe.overlap)
                          << " queues-alias=" << (probe.queuesAlias ? "yes" : "no")
                          << " second-before-first="
                          << (probe.secondCompletedWhileFirstInFlight ? "yes" : "no")
                          << " poll-us=" << probe.pollDurationUs << '\n';
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
