/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "ParticleTypes.hpp"
#include "graphFlow/core/Backend.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/DevicePolicy.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"

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
        std::optional<std::uint32_t> chunkSize; // C, absent means policy
        std::optional<std::uint32_t> liveParticles; // P, absent means policy
        std::optional<std::uint32_t> tileSize; // T, absent means policy
        std::uint32_t batchChunks = 1u; // Q
        std::optional<std::uint32_t> occupancyTarget; // D, absent keeps the historical admission
        std::uint32_t frameCap = graphFlow::defaultFrameCap; // blocks per multiprocessor
        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint32_t deviceIndex = 0u;
        graphFlow::PoolMode poolMode = graphFlow::PoolMode::fixed; // chunk-slot layout
        std::optional<std::uint64_t> slotArenaBudgetBytes; // slot-arena payload budget in bytes
        // Opt-in kernel workload. The axis options stay absent so the resolver can reject an
        // axis on the default feedback workload instead of silently ignoring it.
        particle::Workload workload = particle::Workload::feedback;
        std::optional<particle::ScatterChoice> scatter;
        std::optional<particle::BinPath> binPath;
        std::optional<std::uint32_t> bins;
        std::optional<particle::Distribution> distribution;
    };

    /** Resolved run configuration: the policy result plus the example parameters. */
    struct ResolvedOptions
    {
        graphFlow::Backend backend = graphFlow::Backend::cpu;
        graphFlow::particle::Parameters parameters{};
        graphFlow::Tuning tuning{};
        graphFlow::DevicePolicy policy{};
        bool liveParticlesExplicit = false; // P came from --live-particles
        particle::Workload workload = particle::Workload::feedback;
        particle::KernelOptions kernel{};
    };

    inline void usage(std::ostream& out)
    {
        out << "usage: graphFlow_particle_feedback --backend cpu|cuda|hip|sycl [--device D]\n"
               "                                    [--particles N] [--chunk-size C]\n"
               "                                    [--live-particles P] [--tile-size T]\n"
               "                                    [--batch-chunks Q] [--frame-cap F]\n"
               "                                    [--occupancy-target D] [--pool-mode fixed|slotArena]\n"
               "                                    [--slot-arena-budget BYTES]\n"
               "                                    [--seed S] [--max-visits M]\n"
               "                                    [--workload feedback|kernel]\n"
               "       kernel workload only: [--scatter shared|direct]\n"
               "                             [--bin-path dense|hash] [--bins B]\n"
               "                             [--distribution balanced|one-hot|skewed|sparse]\n";
    }

    inline graphFlow::PoolMode parsePoolMode(std::string const& value)
    {
        if(value == graphFlow::poolModeName(graphFlow::PoolMode::fixed))
            return graphFlow::PoolMode::fixed;
        if(value == graphFlow::poolModeName(graphFlow::PoolMode::slotArena))
            return graphFlow::PoolMode::slotArena;
        throw std::invalid_argument{"--pool-mode: expected fixed or slotArena, got '" + value + "'"};
    }

    inline particle::Workload parseWorkload(std::string const& value)
    {
        if(value == "feedback")
            return particle::Workload::feedback;
        if(value == "kernel")
            return particle::Workload::kernel;
        throw std::invalid_argument{"--workload: expected feedback or kernel (got '" + value + "')"};
    }

    inline particle::ScatterChoice parseScatterChoice(std::string const& value)
    {
        if(value == "shared")
            return particle::ScatterChoice::shared;
        if(value == "direct")
            return particle::ScatterChoice::direct;
        throw std::invalid_argument{"--scatter: expected shared or direct (got '" + value + "')"};
    }

    inline particle::BinPath parseBinPath(std::string const& value)
    {
        if(value == "dense")
            return particle::BinPath::dense;
        if(value == "hash")
            return particle::BinPath::hash;
        throw std::invalid_argument{"--bin-path: expected dense or hash (got '" + value + "')"};
    }

    inline particle::Distribution parseDistribution(std::string const& value)
    {
        if(value == "balanced")
            return particle::Distribution::balanced;
        if(value == "one-hot")
            return particle::Distribution::oneHot;
        if(value == "skewed")
            return particle::Distribution::skewed;
        if(value == "sparse")
            return particle::Distribution::sparse;
        throw std::invalid_argument{
            "--distribution: expected balanced, one-hot, skewed or sparse (got '" + value + "')"};
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
            else if(flag == "--occupancy-target")
            {
                // Zero is a defined value: it disables the gate, matching `Config` and `Tuning`,
                // so `--occupancy-target 0` is the explicit off switch.
                options.occupancyTarget = parseU32(flag, nextValue(flag), true);
            }
            else if(flag == "--pool-mode")
            {
                options.poolMode = parsePoolMode(nextValue(flag));
            }
            else if(flag == "--slot-arena-budget")
            {
                // Bytes of the slot-arena payload budget. Zero is a defined value: it keeps the
                // whole pool budget, which is also the absent-value behaviour.
                options.slotArenaBudgetBytes = parseU64(flag, nextValue(flag));
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
            else if(flag == "--workload")
            {
                options.workload = parseWorkload(nextValue(flag));
            }
            else if(flag == "--scatter")
            {
                options.scatter = parseScatterChoice(nextValue(flag));
            }
            else if(flag == "--bin-path")
            {
                options.binPath = parseBinPath(nextValue(flag));
            }
            else if(flag == "--bins")
            {
                options.bins = parseU32(flag, nextValue(flag));
            }
            else if(flag == "--distribution")
            {
                options.distribution = parseDistribution(nextValue(flag));
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
        DevicePolicyConfig const& policyConfigIn = policyConfigOf(Config{}))
    {
        ResolvedOptions resolved;
        resolved.backend = requireBackend(options.backend);

        // Mirror the layout in the CLI-side policy, so an explicit P is checked against the
        // arena formula the run will use. `Flow` re-derives with the exact graph credit K*R*C.
        DevicePolicyConfig policyConfig = policyConfigIn;
        policyConfig.poolMode = options.poolMode;

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
        resolved.parameters.poolMode = options.poolMode;
        if(options.slotArenaBudgetBytes)
        {
            if(options.poolMode != graphFlow::PoolMode::slotArena)
                throw std::invalid_argument{"--slot-arena-budget requires --pool-mode slotArena"};
            resolved.parameters.slotArenaBudgetBytes = *options.slotArenaBudgetBytes;
        }

        resolved.workload = options.workload;
        if(options.workload == particle::Workload::feedback)
        {
            if(options.scatter || options.binPath || options.bins || options.distribution)
                throw std::invalid_argument{
                    "--scatter, --bin-path, --bins and --distribution require --workload kernel"};
        }
        else
        {
            if(options.scatter)
                resolved.kernel.scatter = *options.scatter;
            if(options.binPath)
                resolved.kernel.binPath = *options.binPath;
            if(options.bins)
                resolved.kernel.bins = *options.bins;
            if(options.distribution)
                resolved.kernel.distribution = *options.distribution;

            if(!graphFlow::particle::isSupportedKernelBins(resolved.kernel.bins))
                throw std::invalid_argument{
                    "--bins: the kernel workload instantiates 64, 1024, 4096 and 65536 (got '"
                    + std::to_string(resolved.kernel.bins) + "')"};
            if(resolved.kernel.binPath == particle::BinPath::dense
               && resolved.kernel.bins > graphFlow::particle::maxDenseKernelBins)
                throw std::invalid_argument{
                    "--bin-path dense supports at most 1024 bins; use --bin-path hash above that"};
            if(resolved.kernel.binPath == particle::BinPath::hash
               && resolved.kernel.scatter == particle::ScatterChoice::direct)
                throw std::invalid_argument{"--scatter direct is the dense-path variant; the hash path always stages"};
            if(resolved.kernel.distribution == particle::Distribution::sparse && resolved.kernel.bins < 1024u)
                throw std::invalid_argument{"--distribution sparse requires --bins 1024 or larger"};
        }

        resolved.tuning = particle::tuningOf(resolved.parameters);
        // Opt-in occupancy target. Absent keeps the default tuning unchanged, so the example
        // output is byte-identical to the pre-phase-1 binary.
        if(options.occupancyTarget)
            resolved.tuning.occupancyTargetChunks = options.occupancyTarget;
        return resolved;
    }
} // namespace graphFlow::cli
