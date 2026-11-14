/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "PhotonCli.hpp"
#include "PhotonScattering.hpp"

#include "graphFlow/core/Backend.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    char const* eventName(graphFlow::photon::Event event)
    {
        using graphFlow::photon::Event;
        switch(event)
        {
        case Event::transmitted:
            return "transmitted";
        case Event::reflected:
            return "reflected";
        case Event::absorbed:
            return "absorbed";
        case Event::capped:
            return "capped";
        case Event::scatter:
            return "scatter";
        }
        return "unknown";
    }

    void printResult(graphFlow::photon::ScatteringResult const& result)
    {
        std::cout << "# event,count\n";
        std::cout << "transmitted," << result.eventCounts[static_cast<std::uint32_t>(
                                             graphFlow::photon::Event::transmitted)]
                  << '\n';
        std::cout << "reflected," << result.eventCounts[static_cast<std::uint32_t>(
                                           graphFlow::photon::Event::reflected)]
                  << '\n';
        std::cout << "absorbed," << result.eventCounts[static_cast<std::uint32_t>(
                                          graphFlow::photon::Event::absorbed)]
                  << '\n';
        std::cout << "capped," << result.eventCounts[static_cast<std::uint32_t>(
                                        graphFlow::photon::Event::capped)]
                  << '\n';
        std::cout << "# outcome,scatter-count,count\n";
        for(auto const& row : result.histograms)
            std::cout << eventName(row.event) << ',' << row.scatterCount << ',' << row.count << '\n';
        std::cout << "# terminal-sum," << result.terminated << '\n';
        std::cout << "# inconsistent," << result.inconsistent << '\n';
    }

    template<typename TDevice>
    int run(
        graphFlow::photon::cli::ResolvedOptions const& resolved,
        graphFlow::Backend backend,
        TDevice device,
        std::string const& deviceName)
    {
        graphFlow::photon::Parameters const& parameters = resolved.parameters;
        graphFlow::photon::ScatteringResult const result
            = graphFlow::photon::runPhotonScattering(device, parameters, resolved.tuning);

        std::cout << "# graphFlow_photon_scattering backend=" << graphFlow::backendName(backend)
                  << " device=" << parameters.deviceIndex << " device-name=" << deviceName
                  << " photons=" << parameters.numPhotons
                  << " thickness=" << parameters.thickness
                  << " sigma-s=" << parameters.sigmaS
                  << " sigma-a=" << parameters.sigmaA
                  << " mode=" << (parameters.mode == graphFlow::photon::ScatterMode::forward ? "forward" : "isotropic")
                  << " max-scatters=" << parameters.maxScatters
                  << " seed=" << parameters.seed
                  << " chunk-size=" << parameters.chunkSize
                  << " live-particles=" << result.liveParticles
                  << " tile-size=" << parameters.tileSize
                  << " batch-chunks=" << parameters.batchChunks
                  << " frame-cap=" << parameters.frameCap << '\n';
        std::cout << "# units length=1 coefficient=1/length\n";
        std::cout << "# pool-chunks=" << result.stats.poolChunks
                  << " peak-live-chunks=" << result.stats.peakLiveChunks
                  << " peak-reserved-chunks=" << result.stats.peakReservedChunks << '\n';
        printResult(result);
        return EXIT_SUCCESS;
    }
} // namespace

int main(int argc, char** argv)
{
    graphFlow::photon::cli::Options options;
    try
    {
        options = graphFlow::photon::cli::parseOptions(argc, argv);
    }
    catch(std::exception const& error)
    {
        std::cerr << "graphFlow_photon_scattering: " << error.what() << '\n';
        graphFlow::photon::cli::usage(std::cerr);
        return 2;
    }

    if(options.help)
    {
        graphFlow::photon::cli::usage(std::cout);
        return EXIT_SUCCESS;
    }

    graphFlow::Backend backend;
    try
    {
        backend = graphFlow::requireBackend(options.backend);
    }
    catch(graphFlow::BackendError const& error)
    {
        std::cerr << "graphFlow_photon_scattering: " << error.what() << '\n';
        graphFlow::photon::cli::usage(std::cerr);
        return 2;
    }

    try
    {
        return graphFlow::withBackendDevice(
            backend,
            options.deviceIndex,
            [&](auto device, graphFlow::Backend selected, std::string const& deviceName)
            {
                graphFlow::photon::cli::ResolvedOptions const resolved
                    = graphFlow::photon::cli::resolveOptions(options, device.getDeviceProperties());
                std::cout << "# selected-backend=" << graphFlow::backendName(selected)
                          << " selected-device=" << options.deviceIndex
                          << " device-name=" << deviceName << '\n';
                std::cout << "# policy " << resolved.policy.description << '\n';
                return run(resolved, selected, device, deviceName);
            });
    }
    catch(graphFlow::BackendUnavailable const& error)
    {
        std::cerr << "graphFlow_photon_scattering: " << error.what() << '\n';
        return 2;
    }
    catch(std::out_of_range const& error)
    {
        std::cerr << "graphFlow_photon_scattering: " << error.what() << '\n';
        return 2;
    }
    catch(std::invalid_argument const& error)
    {
        std::cerr << "graphFlow_photon_scattering: " << error.what() << '\n';
        return 2;
    }
    catch(std::exception const& error)
    {
        std::cerr << "graphFlow_photon_scattering: " << error.what() << '\n';
        return 1;
    }
}
