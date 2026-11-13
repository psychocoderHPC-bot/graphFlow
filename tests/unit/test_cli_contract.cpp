/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Cli.hpp"
#include "TestDevice.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    /** Parse a synthetic argv without launching the example. */
    graphFlow::cli::Options parseArgs(std::vector<std::string> const& args)
    {
        std::vector<char const*> argv;
        argv.reserve(args.size() + 1u);
        argv.push_back("graphFlow_particle_feedback");
        for(auto const& arg : args)
            argv.push_back(arg.c_str());
        return graphFlow::cli::parseOptions(static_cast<int>(argv.size()), argv.data());
    }

    alpaka::onHost::DeviceProperties cpuProperties()
    {
        return graphFlow::test::makeCpuDevice().getDeviceProperties();
    }
} // namespace

TEST_CASE("CLI parses valid backend, device and tunables", "[cli]")
{
    graphFlow::cli::Options const defaults = parseArgs({});
    CHECK_FALSE(defaults.help);
    CHECK(defaults.backend == "cpu");
    CHECK(defaults.deviceIndex == 0u);
    CHECK(defaults.numParticles == 0u);
    CHECK_FALSE(defaults.chunkSize.has_value());
    CHECK_FALSE(defaults.liveParticles.has_value());
    CHECK_FALSE(defaults.tileSize.has_value());
    CHECK(defaults.batchChunks == 1u);
    CHECK(defaults.seed == 0u);
    CHECK(defaults.maxVisits == 64u);

    graphFlow::cli::Options const options = parseArgs(
        {"--backend",
         "cuda",
         "--device",
         "3",
         "--particles",
         "4097",
         "--chunk-size",
         "256",
         "--live-particles",
         "512",
         "--tile-size",
         "64",
         "--batch-chunks",
         "8",
         "--seed",
         "42",
         "--max-visits",
         "7"});
    CHECK_FALSE(options.help);
    CHECK(options.backend == "cuda");
    CHECK(options.deviceIndex == 3u);
    CHECK(options.numParticles == 4097u);
    REQUIRE(options.chunkSize.has_value());
    CHECK(*options.chunkSize == 256u);
    REQUIRE(options.liveParticles.has_value());
    CHECK(*options.liveParticles == 512u);
    REQUIRE(options.tileSize.has_value());
    CHECK(*options.tileSize == 64u);
    CHECK(options.batchChunks == 8u);
    CHECK(options.seed == 42u);
    CHECK(options.maxVisits == 7u);

    graphFlow::cli::ResolvedOptions const resolved
        = graphFlow::cli::resolveOptions(options, cpuProperties());
    CHECK(resolved.backend == graphFlow::Backend::cuda);
    CHECK(resolved.parameters.numParticles == 4097u);
    CHECK(resolved.parameters.chunkSize == 256u);
    CHECK(resolved.parameters.liveParticles == 512u);
    CHECK(resolved.parameters.tileSize == 64u);
    CHECK(resolved.parameters.batchChunks == 8u);
    CHECK(resolved.parameters.seed == 42u);
    CHECK(resolved.parameters.maxVisits == 7u);

    CHECK(parseArgs({"--help"}).help);
}

TEST_CASE("CLI rejects an unknown backend", "[cli]")
{
    graphFlow::cli::Options const options = parseArgs({"--backend", "tpu"});
    CHECK_THROWS_AS(
        graphFlow::cli::resolveOptions(options, cpuProperties()),
        graphFlow::BackendError);
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(options, cpuProperties()),
        Catch::Matchers::ContainsSubstring("unknown backend 'tpu'"));
}

TEST_CASE("CLI rejects zero chunk, live-particle, tile and batch values", "[cli]")
{
    CHECK_THROWS_WITH(
        parseArgs({"--chunk-size", "0"}),
        Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(
        parseArgs({"--live-particles", "0"}),
        Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(
        parseArgs({"--tile-size", "0"}),
        Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(
        parseArgs({"--batch-chunks", "0"}),
        Catch::Matchers::ContainsSubstring("must be non-zero"));
}

TEST_CASE("CLI rejects an oversized tile and capacity", "[cli]")
{
    alpaka::onHost::DeviceProperties const properties = cpuProperties();
    graphFlow::DevicePolicyConfig const policyConfig = graphFlow::policyConfigOf(graphFlow::Config{});

    graphFlow::cli::Options const oversizedTile
        = parseArgs({"--tile-size", std::to_string(properties.maxThreadsPerBlock + 1u)});
    CHECK_THROWS_AS(
        graphFlow::cli::resolveOptions(oversizedTile, properties, policyConfig),
        std::invalid_argument);

    graphFlow::cli::Options const oversizedChunk = parseArgs(
        {"--chunk-size", std::to_string(policyConfig.maxChunkCapacity + 1u)});
    CHECK_THROWS_AS(
        graphFlow::cli::resolveOptions(oversizedChunk, properties, policyConfig),
        std::invalid_argument);

    // Explicit P and C that exceed the payload budget together.
    graphFlow::cli::Options const infeasible = parseArgs(
        {"--particles", "1000000", "--chunk-size", "8192", "--live-particles", "8192"});
    CHECK_THROWS_AS(
        graphFlow::cli::resolveOptions(infeasible, properties),
        std::invalid_argument);
}

TEST_CASE("CLI preserves explicit C, P and T overrides", "[cli]")
{
    graphFlow::cli::Options const options = parseArgs(
        {"--particles", "4097", "--chunk-size", "100", "--live-particles", "123", "--tile-size", "64"});

    graphFlow::cli::ResolvedOptions const resolved
        = graphFlow::cli::resolveOptions(options, cpuProperties());

    CHECK(resolved.policy.chunkCapacity == 100u);
    CHECK(resolved.policy.liveParticles == 123u);
    CHECK(resolved.policy.tileSize == 64u);
    CHECK(resolved.parameters.chunkSize == 100u);
    CHECK(resolved.parameters.liveParticles == 123u);
    CHECK(resolved.parameters.tileSize == 64u);
}

TEST_CASE("CLI allows N=0", "[cli]")
{
    graphFlow::cli::Options const options = parseArgs({"--particles", "0"});
    graphFlow::cli::ResolvedOptions const resolved
        = graphFlow::cli::resolveOptions(options, cpuProperties());

    CHECK(resolved.parameters.numParticles == 0u);
    CHECK(resolved.policy.chunkCapacity > 0u);
    CHECK(resolved.policy.liveParticles == 0u);
    CHECK(resolved.parameters.liveParticles == 0u);
    CHECK_FALSE(resolved.policy.description.empty());
}

TEST_CASE("CLI selects policy defaults when overrides are absent", "[cli]")
{
    graphFlow::cli::Options const options
        = parseArgs({"--particles", "1000000", "--batch-chunks", "8"});
    CHECK_FALSE(options.chunkSize.has_value());
    CHECK_FALSE(options.liveParticles.has_value());
    CHECK_FALSE(options.tileSize.has_value());

    graphFlow::cli::ResolvedOptions const resolved
        = graphFlow::cli::resolveOptions(options, cpuProperties());

    CHECK(resolved.parameters.chunkSize == resolved.policy.chunkCapacity);
    CHECK(resolved.parameters.liveParticles == resolved.policy.liveParticles);
    CHECK(resolved.parameters.tileSize == resolved.policy.tileSize);
    CHECK(resolved.parameters.batchChunks == 8u);
    CHECK(resolved.policy.chunkCapacity > 0u);
    CHECK(resolved.policy.liveParticles > 0u);
    CHECK(resolved.policy.tileSize > 0u);
    CHECK_FALSE(resolved.policy.description.empty());
    CHECK(resolved.policy.description.find("C=") != std::string::npos);
}
