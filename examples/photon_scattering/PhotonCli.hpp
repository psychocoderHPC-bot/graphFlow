/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonTypes.hpp"

#include "graphFlow/core/Backend.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace graphFlow::photon::cli
{
    /** Raw command-line values. */
    struct Options
    {
        bool help = false;
        std::string backend = "cpu";
        std::uint64_t numPhotons = 0u;
        std::optional<std::uint32_t> chunkSize; // C, absent means policy
        std::optional<std::uint32_t> liveParticles; // P, absent means policy
        std::optional<std::uint32_t> tileSize; // T, absent means policy
        std::uint32_t batchChunks = 1u; // Q
        std::uint32_t frameCap = graphFlow::defaultFrameCap;
        std::optional<std::uint64_t> poolPayloadBudgetBytes;
        std::optional<std::string> outputDirectory;
        std::uint32_t tracePhotons = 0u;
        std::uint32_t traceVertices = 64u;
        std::uint64_t seed = 0u;
        double thickness = 1.0;
        double sigmaS = 0.0;
        double sigmaA = 0.0;
        std::string mode = "forward";
        std::uint32_t maxScatters = 64u;
        std::uint32_t deviceIndex = 0u;
    };

    /** Resolved run configuration: policy result plus the photon parameters. */
    struct ResolvedOptions
    {
        graphFlow::Backend backend = graphFlow::Backend::cpu;
        graphFlow::photon::Parameters parameters{};
        graphFlow::Tuning tuning{};
        graphFlow::DevicePolicy policy{};
        bool liveParticlesExplicit = false;
    };

    inline void usage(std::ostream& out)
    {
        out << "usage: graphFlow_photon_scattering --backend cpu|cuda|hip [--device D]\n"
               "                                      [--photons N] [--thickness L]\n"
               "                                      [--sigma-s S] [--sigma-a A]\n"
               "                                      [--mode forward|isotropic]\n"
               "                                      [--max-scatters M] [--seed S]\n"
               "                                      [--chunk-size C] [--live-particles P]\n"
               "                                      [--tile-size T] [--batch-chunks Q]\n"
               "                                      [--frame-cap F] [--pool-budget-bytes B]\n"
               "                                      [--output-dir DIR] [--trace-photons M]\n"
               "                                      [--trace-vertices V]\n";
    }

    inline std::uint64_t parseU64(std::string_view flag, std::string const& value)
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
            throw std::invalid_argument{
                std::string{flag} + ": invalid value '" + value + "' (" + error.what() + ")"};
        }
    }

    inline std::uint32_t parseU32(std::string_view flag, std::string const& value, bool allowZero = false)
    {
        std::uint64_t const parsed = parseU64(flag, value);
        if(parsed > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument{std::string{flag} + ": value '" + value + "' exceeds 32 bits"};
        if(!allowZero && parsed == 0u)
            throw std::invalid_argument{std::string{flag} + ": value must be non-zero"};
        return static_cast<std::uint32_t>(parsed);
    }

    inline double parseDouble(std::string_view flag, std::string const& value)
    {
        try
        {
            std::size_t consumed = 0u;
            double const parsed = std::stod(value, &consumed);
            if(consumed != value.size())
                throw std::invalid_argument{"trailing characters"};
            if(!std::isfinite(parsed))
                throw std::invalid_argument{"value is not finite"};
            return parsed;
        }
        catch(std::exception const& error)
        {
            throw std::invalid_argument{
                std::string{flag} + ": invalid value '" + value + "' (" + error.what() + ")"};
        }
    }

    /** Parse the command line without touching a device. */
    inline Options parseOptions(int argc, char const* const* argv)
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
                options.help = true;
                return options;
            }
            else if(flag == "--backend")
                options.backend = nextValue(flag);
            else if(flag == "--photons" || flag == "--N")
                options.numPhotons = parseU64(flag, nextValue(flag));
            else if(flag == "--thickness" || flag == "--L")
                options.thickness = parseDouble(flag, nextValue(flag));
            else if(flag == "--sigma-s" || flag == "--sigmaS")
                options.sigmaS = parseDouble(flag, nextValue(flag));
            else if(flag == "--sigma-a" || flag == "--sigmaA")
                options.sigmaA = parseDouble(flag, nextValue(flag));
            else if(flag == "--mode")
                options.mode = nextValue(flag);
            else if(flag == "--max-scatters" || flag == "--cap")
                options.maxScatters = parseU32(flag, nextValue(flag), true);
            else if(flag == "--seed")
                options.seed = parseU64(flag, nextValue(flag));
            else if(flag == "--chunk-size" || flag == "--C")
                options.chunkSize = parseU32(flag, nextValue(flag));
            else if(flag == "--live-particles" || flag == "--P")
                options.liveParticles = parseU32(flag, nextValue(flag));
            else if(flag == "--tile-size" || flag == "--T")
                options.tileSize = parseU32(flag, nextValue(flag));
            else if(flag == "--batch-chunks" || flag == "--Q")
                options.batchChunks = parseU32(flag, nextValue(flag));
            else if(flag == "--frame-cap")
                options.frameCap = parseU32(flag, nextValue(flag));
            else if(flag == "--pool-budget-bytes")
                options.poolPayloadBudgetBytes = parseU64(flag, nextValue(flag));
            else if(flag == "--output-dir")
                options.outputDirectory = nextValue(flag);
            else if(flag == "--trace-photons")
                options.tracePhotons = parseU32(flag, nextValue(flag), true);
            else if(flag == "--trace-vertices")
                options.traceVertices = parseU32(flag, nextValue(flag), true);
            else if(flag == "--device")
                options.deviceIndex = parseU32(flag, nextValue(flag), true);
            else
                throw std::invalid_argument{"unknown argument '" + flag + "'"};
        }
        return options;
    }

    /** Validate the backend and resolve the device-aware C, P and T. */
    inline ResolvedOptions resolveOptions(
        Options const& options,
        alpaka::onHost::DeviceProperties const& properties,
        graphFlow::DevicePolicyConfig policyConfig = {})
    {
        ResolvedOptions resolved;
        resolved.backend = graphFlow::requireBackend(options.backend);

        graphFlow::DevicePolicyOverrides overrides;
        overrides.chunkCapacity = options.chunkSize;
        overrides.liveParticles = options.liveParticles;
        overrides.tileSize = options.tileSize;

        // The generic payload storage charges sizeof(Photon) per slot, not the particle SoA size.
        policyConfig.slotBytes = sizeof(graphFlow::photon::Photon);
        resolved.policy = graphFlow::makeDevicePolicy(
            properties,
            options.numPhotons,
            overrides,
            policyConfig);
        resolved.liveParticlesExplicit = resolved.policy.liveParticlesExplicit;

        resolved.parameters.numPhotons = options.numPhotons;
        resolved.parameters.chunkSize = resolved.policy.chunkCapacity;
        resolved.parameters.liveParticles = resolved.policy.liveParticles;
        resolved.parameters.tileSize = resolved.policy.tileSize;
        resolved.parameters.batchChunks = options.batchChunks;
        resolved.parameters.frameCap = options.frameCap;
        resolved.parameters.seed = options.seed;
        resolved.parameters.thickness = options.thickness;
        resolved.parameters.sigmaS = options.sigmaS;
        resolved.parameters.sigmaA = options.sigmaA;
        resolved.parameters.maxScatters = options.maxScatters;
        resolved.parameters.deviceIndex = options.deviceIndex;
        resolved.parameters.trace.enabled = options.tracePhotons != 0u;
        resolved.parameters.trace.maxPhotons = options.tracePhotons;
        resolved.parameters.trace.maxVertices = options.traceVertices;
        if(options.poolPayloadBudgetBytes)
        {
            if(*options.poolPayloadBudgetBytes == 0u)
                throw std::invalid_argument{"--pool-budget-bytes: value must be non-zero"};
            resolved.parameters.poolPayloadBudgetBytes = *options.poolPayloadBudgetBytes;
        }
        if(options.outputDirectory && options.outputDirectory->empty())
            throw std::invalid_argument{"--output-dir: value must not be empty"};

        if(options.mode == "forward")
            resolved.parameters.mode = graphFlow::photon::ScatterMode::forward;
        else if(options.mode == "isotropic")
            resolved.parameters.mode = graphFlow::photon::ScatterMode::isotropic;
        else
            throw std::invalid_argument{"--mode: expected 'forward' or 'isotropic', got '" + options.mode + "'"};

        graphFlow::photon::validateParameters(resolved.parameters);
        resolved.tuning = graphFlow::photon::tuningOf(resolved.parameters);
        return resolved;
    }
} // namespace graphFlow::photon::cli
