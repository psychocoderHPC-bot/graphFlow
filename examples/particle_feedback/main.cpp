/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
    struct Options
    {
        std::string backend = "cpu";
        graphFlow::particle::Parameters parameters{};
    };

    void usage(std::ostream& out)
    {
        out << "usage: graphFlow_particle_feedback --backend cpu [--particles N] [--chunk-size C]\n"
               "                                    [--seed S] [--max-visits M] [--device D]\n";
    }

    std::uint64_t parseU64(std::string_view flag, std::string const& value)
    {
        try
        {
            std::size_t consumed = 0u;
            std::uint64_t const parsed = std::stoull(value, &consumed);
            if(consumed != value.size())
                throw std::invalid_argument{"trailing characters"};
            return parsed;
        }
        catch(std::exception const& error)
        {
            throw std::invalid_argument{std::string{flag} + ": invalid value '" + value + "' (" + error.what() + ")"};
        }
    }

    Options parseOptions(int argc, char** argv)
    {
        Options options;
        for(int i = 1; i < argc; ++i)
        {
            std::string const flag = argv[i];
            auto const nextValue = [&](std::string_view name) -> std::string
            {
                if(i + 1 >= argc)
                    throw std::invalid_argument{std::string{name} + ": missing value"};
                return argv[++i];
            };

            if(flag == "--help" || flag == "-h")
            {
                usage(std::cout);
                std::exit(EXIT_SUCCESS);
            }
            else if(flag == "--backend")
            {
                options.backend = nextValue(flag);
            }
            else if(flag == "--particles")
            {
                options.parameters.numParticles = parseU64(flag, nextValue(flag));
            }
            else if(flag == "--chunk-size")
            {
                options.parameters.chunkSize = static_cast<std::uint32_t>(parseU64(flag, nextValue(flag)));
            }
            else if(flag == "--seed")
            {
                options.parameters.seed = parseU64(flag, nextValue(flag));
            }
            else if(flag == "--max-visits")
            {
                options.parameters.maxVisits = static_cast<std::uint32_t>(parseU64(flag, nextValue(flag)));
            }
            else if(flag == "--device")
            {
                options.parameters.deviceIndex = static_cast<std::uint32_t>(parseU64(flag, nextValue(flag)));
            }
            else
            {
                throw std::invalid_argument{"unknown argument '" + flag + "'"};
            }
        }
        return options;
    }

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
    int run(Options const& options, TDevice device)
    {
        graphFlow::particle::FeedbackResult const result
            = graphFlow::particle::runParticleFeedback(device, options.parameters);

        std::cout << "# graphFlow_particle_feedback backend=" << options.backend
                  << " device=" << options.parameters.deviceIndex
                  << " particles=" << options.parameters.numParticles
                  << " chunk-size=" << options.parameters.chunkSize
                  << " seed=" << options.parameters.seed
                  << " max-visits=" << options.parameters.maxVisits << '\n';
        std::cout << "# weight,count\n";
        printHistogram(result, options.parameters.numParticles);
        return EXIT_SUCCESS;
    }
} // namespace

int main(int argc, char** argv)
{
    Options options;
    try
    {
        options = parseOptions(argc, argv);
    }
    catch(std::exception const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        usage(std::cerr);
        return 2;
    }

    if(options.backend != "cpu")
    {
        std::cerr << "graphFlow_particle_feedback: backend '" << options.backend
                  << "' is not available. This build implements the CPU vertical slice only; CUDA and HIP"
                     " execution are not implemented yet (the GRAPHFLOW_ENABLE_CUDA/HIP options exist but no"
                     " device code is compiled).\n";
        return 2;
    }

    try
    {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
        if(!selector.isAvailable() || options.parameters.deviceIndex >= selector.getDeviceCount())
        {
            std::cerr << "graphFlow_particle_feedback: device " << options.parameters.deviceIndex
                      << " is not available (found " << selector.getDeviceCount() << " host devices)\n";
            return 2;
        }
        auto device = selector.makeDevice(options.parameters.deviceIndex);
        return run(options, device);
    }
    catch(std::exception const& error)
    {
        std::cerr << "graphFlow_particle_feedback: " << error.what() << '\n';
        return 1;
    }
}
