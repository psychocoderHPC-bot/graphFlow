/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleStages.hpp"

#include "graphFlow/core/Backend.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/core/DevicePolicy.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace graphFlow::cli
{
    /** Raw command-line values.
     *
     * `chunkSize`, `liveParticles` and `tileSize` are optional so the resolver can tell an
     * explicit override from a value derived by the device policy. `batchChunks` is parsed and
     * validated here and forwarded to the transform nodes as the maximum number of ready input
     * chunks per batch.
     */
    struct Options
    {
        bool help = false;
        std::string backend = "cpu";
        std::uint64_t numParticles = 0u;
        std::optional<std::uint32_t> chunkSize;     // C, absent means policy
        std::optional<std::uint32_t> liveParticles; // P, absent means policy
        std::optional<std::uint32_t> tileSize;      // T, absent means policy
        std::uint32_t batchChunks = 1u;             // Q
        std::uint32_t frameCap = graphFlow::defaultFrameCap; // blocks per multiprocessor
        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint32_t deviceIndex = 0u;
    };

    /** Resolved run configuration: the policy result plus the example parameters. */
    struct ResolvedOptions
    {
        graphFlow::Backend backend = graphFlow::Backend::cpu;
        graphFlow::particle::Parameters parameters{};
        graphFlow::DevicePolicy policy{};
        bool liveParticlesExplicit = false; // P came from --live-particles
    };

    inline void usage(std::ostream& out)
    {
        out << "usage: graphFlow_particle_feedback --backend cpu|cuda|hip [--device D]\n"
               "                                    [--particles N] [--chunk-size C]\n"
               "                                    [--live-particles P] [--tile-size T]\n"
               "                                    [--batch-chunks Q] [--frame-cap F]\n"
               "                                    [--seed S] [--max-visits M]\n";
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
            throw std::invalid_argument{std::string{flag} + ": invalid value '" + value + "' (" + error.what() + ")"};
        }
    }

    /** Parse an unsigned 32-bit value, rejecting overflow and optionally zero.
     *
     * The 64-bit parse first avoids the silent truncation of `static_cast<std::uint32_t>` on an
     * oversized literal; the zero check keeps every tunable that reaches the policy non-zero.
     */
    inline std::uint32_t parseU32(std::string_view flag, std::string const& value, bool allowZero = false)
    {
        std::uint64_t const parsed = parseU64(flag, value);
        if(parsed > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument{std::string{flag} + ": value '" + value + "' exceeds 32 bits"};
        if(!allowZero && parsed == 0u)
            throw std::invalid_argument{std::string{flag} + ": value must be non-zero"};
        return static_cast<std::uint32_t>(parsed);
    }

    /** Parse the command line without touching a device.
     *
     * `--help` sets `Options::help` and returns immediately so tests and the wrapper can decide
     * how to print without `std::exit`. Malformed values throw `std::invalid_argument`.
     */
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
            {
                options.backend = nextValue(flag);
            }
            else if(flag == "--particles")
            {
                options.numParticles = parseU64(flag, nextValue(flag));
            }
            else if(flag == "--chunk-size")
            {
                options.chunkSize = parseU32(flag, nextValue(flag));
            }
            else if(flag == "--live-particles")
            {
                options.liveParticles = parseU32(flag, nextValue(flag));
            }
            else if(flag == "--tile-size")
            {
                options.tileSize = parseU32(flag, nextValue(flag));
            }
            else if(flag == "--batch-chunks")
            {
                options.batchChunks = parseU32(flag, nextValue(flag));
            }
            else if(flag == "--frame-cap")
            {
                options.frameCap = parseU32(flag, nextValue(flag));
            }
            else if(flag == "--seed")
            {
                options.seed = parseU64(flag, nextValue(flag));
            }
            else if(flag == "--max-visits")
            {
                options.maxVisits = parseU32(flag, nextValue(flag), true);
            }
            else if(flag == "--device")
            {
                options.deviceIndex = parseU32(flag, nextValue(flag), true);
            }
            else
            {
                throw std::invalid_argument{"unknown argument '" + flag + "'"};
            }
        }
        return options;
    }

    /** Validate the backend and resolve the device-aware `C`, `P` and `T`.
     *
     * An explicit `C` or `T` override is passed through unchanged. An explicit `P` is kept
     * exactly and rejected when it exceeds the payload budget; only the auto-derived P is
     * clamped. `makeDevicePolicy` rejects zero, over-limit and unsatisfiable combinations. With
     * no override the returned values are the policy defaults, so `Parameters` always carries a
     * complete configuration.
     */
    inline ResolvedOptions resolveOptions(
        Options const& options,
        alpaka::onHost::DeviceProperties const& properties,
        DevicePolicyConfig const& policyConfig = policyConfigOf(Config{}))
    {
        ResolvedOptions resolved;
        resolved.backend = requireBackend(options.backend);

        DevicePolicyOverrides overrides;
        overrides.chunkCapacity = options.chunkSize;
        overrides.liveParticles = options.liveParticles;
        overrides.tileSize = options.tileSize;

        resolved.policy = makeDevicePolicy(properties, options.numParticles, overrides, policyConfig);
        resolved.liveParticlesExplicit = resolved.policy.liveParticlesExplicit;

        resolved.parameters.numParticles = options.numParticles;
        resolved.parameters.chunkSize = resolved.policy.chunkCapacity;
        resolved.parameters.liveParticles = resolved.policy.liveParticles;
        resolved.parameters.tileSize = resolved.policy.tileSize;
        resolved.parameters.batchChunks = options.batchChunks;
        resolved.parameters.frameCap = options.frameCap;
        resolved.parameters.seed = options.seed;
        resolved.parameters.maxVisits = options.maxVisits;
        resolved.parameters.deviceIndex = options.deviceIndex;
        // The policy budget bounds the live set P*C; the fixed-capacity pool additionally holds
        // the K*R transform output-credit chunks, so resolve the total pool budget here.
        resolved.parameters.poolPayloadBudgetBytes
            = particle::poolPayloadBudgetBytes(resolved.policy.memoryBudgetBytes, resolved.policy.chunkCapacity);
        return resolved;
    }
} // namespace graphFlow::cli
