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
        std::optional<std::uint32_t> occupancyTarget; // D, absent keeps the historical admission
        std::uint32_t frameCap = graphFlow::defaultFrameCap;
        std::string chunkKernelSchedule = "off"; // off|perChunk
        std::uint32_t chunkQueueDepth = 2u; // measured default, see docs/benchmarks.md
        std::uint32_t chunkLatencyCapUs = 0u; // zero means no cap
        std::optional<std::uint64_t> poolPayloadBudgetBytes;
        graphFlow::PoolMode poolMode = graphFlow::PoolMode::fixed; // chunk-slot layout
        bool poolModeExplicit = false; // --pool-mode was given, so the backend default is off
        std::optional<std::uint64_t> slotArenaBudgetBytes; // slot-arena payload budget in bytes
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
        out << "usage: graphFlow_photon_scattering --backend cpu|cuda|hip|sycl [--device D]\n"
               "                                      [--photons N] [--thickness L]\n"
               "                                      [--sigma-s S] [--sigma-a A]\n"
               "                                      [--mode forward|isotropic]\n"
               "                                      [--max-scatters M] [--seed S]\n"
               "                                      [--chunk-size C] [--live-particles P]\n"
               "                                      [--tile-size T] [--batch-chunks Q]\n"
               "                                      [--occupancy-target D]\n"
               "                                      [--frame-cap F] [--pool-budget-bytes B]\n"
               "                                      [--pool-mode fixed|slotArena]\n"
               "                                      [--slot-arena-budget BYTES]\n"
               "                                      [--chunk-kernel-schedule off|perChunk]\n"
               "                                      [--chunk-queue-depth D] [--chunk-latency-cap-us U]\n"
               "                                      [--output-dir DIR] [--trace-photons M]\n"
               "                                      [--trace-vertices V]\n"
               "       the CPU backend defaults to --pool-mode slotArena with a 1 GiB payload\n"
               "       budget; accelerators default to --pool-mode fixed with the derived device\n"
               "       budget. An explicit --pool-mode or --pool-budget-bytes always wins.\n";
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
            throw std::invalid_argument{std::string{flag} + ": invalid value '" + value + "' (" + error.what() + ")"};
        }
    }

    /** Parse the transform launch schedule of `--chunk-kernel-schedule`. */
    inline graphFlow::ChunkKernelSchedule parseChunkKernelSchedule(std::string_view flag, std::string const& value)
    {
        if(value == "off")
            return graphFlow::ChunkKernelSchedule::off;
        if(value == "perChunk")
            return graphFlow::ChunkKernelSchedule::perChunk;
        throw std::invalid_argument{std::string{flag} + ": expected 'off' or 'perChunk', got '" + value + "'"};
    }

    /** Parse the chunk-slot layout of `--pool-mode`. */
    inline graphFlow::PoolMode parsePoolMode(std::string_view flag, std::string const& value)
    {
        if(value == graphFlow::poolModeName(graphFlow::PoolMode::fixed))
            return graphFlow::PoolMode::fixed;
        if(value == graphFlow::poolModeName(graphFlow::PoolMode::slotArena))
            return graphFlow::PoolMode::slotArena;
        throw std::invalid_argument{std::string{flag} + ": expected fixed or slotArena, got '" + value + "'"};
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
            else if(flag == "--occupancy-target")
                // Zero is a defined value: it disables the gate, matching `Config` and `Tuning`,
                // so `--occupancy-target 0` is the explicit off switch.
                options.occupancyTarget = parseU32(flag, nextValue(flag), true);
            else if(flag == "--chunk-kernel-schedule")
                options.chunkKernelSchedule = nextValue(flag);
            else if(flag == "--chunk-queue-depth")
                options.chunkQueueDepth = parseU32(flag, nextValue(flag));
            else if(flag == "--chunk-latency-cap-us")
                options.chunkLatencyCapUs = parseU32(flag, nextValue(flag), true);
            else if(flag == "--pool-budget-bytes")
                options.poolPayloadBudgetBytes = parseU64(flag, nextValue(flag));
            else if(flag == "--pool-mode")
            {
                options.poolMode = parsePoolMode(flag, nextValue(flag));
                options.poolModeExplicit = true;
            }
            else if(flag == "--slot-arena-budget")
                // Bytes of the slot-arena payload budget. Zero is a defined value: it keeps the
                // whole pool budget, which is also the absent-value behaviour.
                options.slotArenaBudgetBytes = parseU64(flag, nextValue(flag));
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

        // The CPU backend keeps the slot arena with a bounded 1 GiB payload budget. The fixed
        // layout charges `P*C` slots, so the old 16 MiB host budget shrank `C` to 256 and turned a
        // 4M-photon run into hundreds of thousands of tiny batches. Accelerators keep the fixed
        // layout and their derived budget. An explicit flag always wins over these defaults.
        bool const cpuBackend = resolved.backend == graphFlow::Backend::cpu;
        auto const defaultCpuBudget = 1024ull * 1024ull * 1024ull;
        if(options.poolPayloadBudgetBytes && *options.poolPayloadBudgetBytes == 0u)
            throw std::invalid_argument{"--pool-budget-bytes: value must be non-zero"};
        std::uint64_t const budget = options.poolPayloadBudgetBytes.value_or(cpuBackend ? defaultCpuBudget : 0ull);
        graphFlow::PoolMode const mode
            = options.poolModeExplicit ? options.poolMode
                                       : (cpuBackend ? graphFlow::PoolMode::slotArena : graphFlow::PoolMode::fixed);

        graphFlow::DevicePolicyOverrides overrides;
        overrides.chunkCapacity = options.chunkSize;
        overrides.liveParticles = options.liveParticles;
        overrides.tileSize = options.tileSize;

        // The AoS-over-SoA storage charges the aligned attribute widths plus the bin-id scratch
        // word, not the particle SoA size.
        policyConfig.slotBytes = graphFlow::photon::Photon::attribute_list::slotBytes;
        // Mirror the layout in the CLI-side policy, so an explicit P is checked against the
        // arena formula the run will use. `Flow` re-derives with the exact graph credit K*R*C.
        // The budget is applied here, before the policy inverts it: it used to be copied into
        // `Tuning` after `makeDevicePolicy`, so an explicit `--pool-budget-bytes` could not raise
        // the derived `C`.
        policyConfig.poolMode = mode;
        if(budget != 0u)
            policyConfig.memoryBudgetBytes = budget;
        resolved.policy = graphFlow::makeDevicePolicy(properties, options.numPhotons, overrides, policyConfig);
        resolved.liveParticlesExplicit = resolved.policy.liveParticlesExplicit;

        resolved.parameters.numPhotons = options.numPhotons;
        resolved.parameters.chunkSize = resolved.policy.chunkCapacity;
        resolved.parameters.liveParticles = resolved.policy.liveParticles;
        resolved.parameters.tileSize = resolved.policy.tileSize;
        resolved.parameters.batchChunks = options.batchChunks;
        resolved.parameters.frameCap = options.frameCap;
        resolved.parameters.chunkKernelSchedule
            = parseChunkKernelSchedule("--chunk-kernel-schedule", options.chunkKernelSchedule);
        resolved.parameters.chunkQueueDepth = options.chunkQueueDepth;
        resolved.parameters.chunkLatencyCapUs = options.chunkLatencyCapUs;
        resolved.parameters.seed = options.seed;
        resolved.parameters.thickness = options.thickness;
        resolved.parameters.sigmaS = options.sigmaS;
        resolved.parameters.sigmaA = options.sigmaA;
        resolved.parameters.maxScatters = options.maxScatters;
        resolved.parameters.deviceIndex = options.deviceIndex;
        resolved.parameters.trace.enabled = options.tracePhotons != 0u;
        resolved.parameters.trace.maxPhotons = options.tracePhotons;
        resolved.parameters.trace.maxVertices = options.traceVertices;
        resolved.parameters.poolMode = mode;
        if(budget != 0u)
            resolved.parameters.poolPayloadBudgetBytes = budget;
        if(options.slotArenaBudgetBytes)
        {
            if(mode != graphFlow::PoolMode::slotArena)
                throw std::invalid_argument{"--slot-arena-budget requires --pool-mode slotArena"};
            resolved.parameters.slotArenaBudgetBytes = *options.slotArenaBudgetBytes;
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
        // `Flow` re-derives the device policy from `Tuning`. Mirror the budget the CLI used
        // into the policy knob so the re-derivation does not fall back to the 16 MiB host
        // default and reject the C and P that were derived from the 1 GiB CPU default.
        if(budget != 0u)
            resolved.tuning.policyMemoryBudgetBytes = budget;
        // The slot arena is allocated from the pool budget, so the two values have to agree. An
        // explicit `--pool-budget-bytes` is the user's decision and is never rewritten: a pool
        // budget smaller than the arena cannot hold it, so that combination is rejected with both
        // values named. Without an explicit pool budget the derived one, including the 1 GiB CPU
        // default, is raised to the arena size, which is exactly what the arena knob asks for. The
        // device-policy memory cap still sets the derived C and P; only the arena allocation grows.
        if(resolved.tuning.slotArenaBudgetBytes)
        {
            std::uint64_t const arenaBytes = *resolved.tuning.slotArenaBudgetBytes;
            if(options.poolPayloadBudgetBytes)
            {
                if(budget < arenaBytes)
                    throw std::invalid_argument{
                        "--slot-arena-budget (" + std::to_string(arenaBytes) + " bytes) exceeds --pool-budget-bytes ("
                        + std::to_string(budget) + " bytes); the arena is allocated from the pool budget"};
            }
            else if(resolved.policy.memoryBudgetBytes < arenaBytes)
                resolved.tuning.poolPayloadBudgetBytes = arenaBytes;
        }
        // Opt-in occupancy target; absent keeps the default tuning and the output unchanged.
        if(options.occupancyTarget)
            resolved.tuning.occupancyTargetChunks = options.occupancyTarget;
        return resolved;
    }
} // namespace graphFlow::photon::cli
