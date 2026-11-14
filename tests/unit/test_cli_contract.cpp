/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Cli.hpp"
#include "ParticleFeedback.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/Backend.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#    include <sys/wait.h>

#    include <cstdio>
#    include <filesystem>
#endif

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

#if !defined(_WIN32)
    /** Locate the example next to the Catch2 binary, which CTest runs from the build tree. */
    std::filesystem::path exampleBinaryPath()
    {
#    if defined(__linux__)
        std::error_code error;
        std::filesystem::path const self = std::filesystem::read_symlink("/proc/self/exe", error);
        if(!error && !self.empty())
            return self.parent_path() / "graphFlow_particle_feedback";
#    endif
        return std::filesystem::current_path() / "graphFlow_particle_feedback";
    }

    std::string shellQuote(std::string const& value)
    {
        std::string quoted{"'"};
        for(char const character : value)
        {
            if(character == '\'')
                quoted += "'\\''";
            else
                quoted += character;
        }
        quoted += '\'';
        return quoted;
    }

    struct ExampleRun
    {
        int exitCode = -1;
        std::string output;
    };

    /** Run the real example binary and capture its combined stdout/stderr and exit status. */
    ExampleRun runExample(std::string const& arguments)
    {
        std::filesystem::path const binary = exampleBinaryPath();
        if(!std::filesystem::exists(binary))
            throw std::runtime_error{"missing example binary '" + binary.string() + "'"};
        std::string const command = shellQuote(binary.string()) + " " + arguments + " 2>&1";
        FILE* pipe = ::popen(command.c_str(), "r");
        if(pipe == nullptr)
            throw std::runtime_error{"popen failed for '" + binary.string() + "'"};
        std::string output;
        char buffer[4096];
        while(std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
            output += buffer;
        int const status = ::pclose(pipe);
        if(status == -1)
            throw std::runtime_error{"pclose failed for '" + binary.string() + "'"};
        if(!WIFEXITED(status))
            throw std::runtime_error{"example '" + binary.string() + "' did not exit normally"};
        return ExampleRun{WEXITSTATUS(status), output};
    }
#endif
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
    CHECK(defaults.frameCap == graphFlow::defaultFrameCap);
    CHECK(defaults.seed == 0u);
    CHECK(defaults.maxVisits == 64u);

    graphFlow::cli::Options const options
        = parseArgs({"--backend",        "cuda", "--device",     "3",  "--particles",    "4097", "--chunk-size", "256",
                     "--live-particles", "512",  "--tile-size",  "64", "--batch-chunks", "8",    "--frame-cap",  "3",
                     "--seed",           "42",   "--max-visits", "7"});
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
    CHECK(options.frameCap == 3u);
    CHECK(options.seed == 42u);
    CHECK(options.maxVisits == 7u);

    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, cpuProperties());
    CHECK(resolved.backend == graphFlow::Backend::cuda);
    CHECK(resolved.parameters.numParticles == 4097u);
    CHECK(resolved.parameters.chunkSize == 256u);
    CHECK(resolved.parameters.liveParticles == 512u);
    CHECK(resolved.parameters.tileSize == 64u);
    CHECK(resolved.parameters.batchChunks == 8u);
    CHECK(resolved.parameters.frameCap == 3u);
    CHECK(resolved.parameters.seed == 42u);
    CHECK(resolved.parameters.maxVisits == 7u);

    CHECK(parseArgs({"--help"}).help);
}

TEST_CASE("CLI rejects an unknown backend", "[cli]")
{
    graphFlow::cli::Options const options = parseArgs({"--backend", "tpu"});
    CHECK_THROWS_AS(graphFlow::cli::resolveOptions(options, cpuProperties()), graphFlow::BackendError);
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(options, cpuProperties()),
        Catch::Matchers::ContainsSubstring("unknown backend 'tpu'"));
}

TEST_CASE("CLI rejects zero chunk, live-particle, tile and batch values", "[cli]")
{
    CHECK_THROWS_WITH(parseArgs({"--chunk-size", "0"}), Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(parseArgs({"--live-particles", "0"}), Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(parseArgs({"--tile-size", "0"}), Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(parseArgs({"--batch-chunks", "0"}), Catch::Matchers::ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(parseArgs({"--frame-cap", "0"}), Catch::Matchers::ContainsSubstring("must be non-zero"));
}

TEST_CASE("CLI rejects an oversized tile, capacity and P", "[cli]")
{
    alpaka::onHost::DeviceProperties const properties = cpuProperties();
    graphFlow::DevicePolicyConfig const policyConfig = graphFlow::policyConfigOf(graphFlow::Config{});

    graphFlow::cli::Options const oversizedTile
        = parseArgs({"--tile-size", std::to_string(properties.maxThreadsPerBlock + 1u)});
    CHECK_THROWS_AS(graphFlow::cli::resolveOptions(oversizedTile, properties, policyConfig), std::invalid_argument);

    graphFlow::cli::Options const oversizedChunk
        = parseArgs({"--chunk-size", std::to_string(policyConfig.maxChunkCapacity + 1u)});
    CHECK_THROWS_AS(graphFlow::cli::resolveOptions(oversizedChunk, properties, policyConfig), std::invalid_argument);

    // Explicit P and C that exceed the payload budget together. Explicit P is rejected, not
    // clamped; only the auto-derived P adapts to the budget.
    graphFlow::cli::Options const infeasible
        = parseArgs({"--particles", "1000000", "--chunk-size", "8192", "--live-particles", "8192"});
    CHECK_THROWS_AS(graphFlow::cli::resolveOptions(infeasible, properties), std::invalid_argument);

    // A budget that cannot hold even one live particle is still a hard error.
    graphFlow::DevicePolicyConfig tiny = policyConfig;
    tiny.memoryBudgetBytes = 1024u;
    CHECK_THROWS_AS(graphFlow::cli::resolveOptions(infeasible, properties, tiny), std::invalid_argument);
}

TEST_CASE("CLI preserves explicit C, P and T overrides", "[cli]")
{
    graphFlow::cli::Options const options
        = parseArgs({"--particles", "4097", "--chunk-size", "100", "--live-particles", "123", "--tile-size", "64"});

    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, cpuProperties());

    CHECK(resolved.policy.chunkCapacity == 100u);
    CHECK(resolved.policy.liveParticles == 123u);
    CHECK(resolved.policy.liveParticlesExplicit);
    CHECK(resolved.liveParticlesExplicit);
    CHECK(resolved.policy.tileSize == 64u);
    CHECK(resolved.parameters.chunkSize == 100u);
    CHECK(resolved.parameters.liveParticles == 123u);
    CHECK(resolved.parameters.tileSize == 64u);
}

TEST_CASE("CLI allows N=0", "[cli]")
{
    graphFlow::cli::Options const options = parseArgs({"--particles", "0"});
    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, cpuProperties());

    CHECK(resolved.parameters.numParticles == 0u);
    CHECK(resolved.policy.chunkCapacity > 0u);
    CHECK(resolved.policy.liveParticles == 0u);
    CHECK(resolved.parameters.liveParticles == 0u);
    CHECK_FALSE(resolved.policy.description.empty());
}

TEST_CASE("CLI selects policy defaults when overrides are absent", "[cli]")
{
    graphFlow::cli::Options const options = parseArgs({"--particles", "1000000", "--batch-chunks", "8"});
    CHECK_FALSE(options.chunkSize.has_value());
    CHECK_FALSE(options.liveParticles.has_value());
    CHECK_FALSE(options.tileSize.has_value());

    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, cpuProperties());

    CHECK(resolved.parameters.chunkSize == resolved.policy.chunkCapacity);
    CHECK(resolved.parameters.liveParticles == resolved.policy.liveParticles);
    CHECK(resolved.parameters.tileSize == resolved.policy.tileSize);
    CHECK(resolved.parameters.batchChunks == 8u);
    CHECK(resolved.policy.chunkCapacity > 0u);
    CHECK(resolved.policy.liveParticles > 0u);
    CHECK_FALSE(resolved.policy.liveParticlesExplicit);
    CHECK_FALSE(resolved.liveParticlesExplicit);
    CHECK(resolved.policy.tileSize > 0u);
    CHECK_FALSE(resolved.policy.description.empty());
    CHECK(resolved.policy.description.find("C=") != std::string::npos);
}

TEST_CASE("CLI default policy fits an A30 usable-memory budget", "[cli][policy]")
{
    // The A30 reports a 1/8 policy budget of 3,171,090,432 B after ECC instead of exactly
    // 3 GiB. The documented default keeps P=2*C=16384 just out of reach; the default must clamp
    // P and still build a valid pool.
    alpaka::onHost::DeviceProperties properties = cpuProperties();
    properties.warpSize = 32u;
    properties.maxThreadsPerBlock = 1024u;
    properties.maxBlocksPerGrid = 0x7FFF'FFFFu;
    properties.multiProcessorCount = 56u;
    properties.globalMemCapacityBytes = 8ull * 3'171'090'432ull;

    graphFlow::cli::Options const options = parseArgs({"--particles", "100000000"});
    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, properties);

    CHECK(resolved.policy.memoryBudgetBytes == 3'171'090'432ull);
    CHECK(resolved.policy.chunkCapacity > 0u);
    CHECK(resolved.policy.chunkCapacity % resolved.policy.tileSize == 0u);
    CHECK(resolved.policy.liveParticles < 16384u);
    CHECK(resolved.policy.liveParticles == 16128u);
    CHECK_FALSE(resolved.policy.liveParticlesExplicit);
    CHECK_FALSE(resolved.liveParticlesExplicit);
    CHECK(
        std::uint64_t{resolved.policy.liveParticles} * resolved.policy.chunkCapacity * graphFlow::payloadBytesPerSlot
        <= resolved.policy.memoryBudgetBytes);

    // The CLI no longer computes a pool: the resolved tuning carries the domain knobs and
    // `Flow` derives the pool and output-credit reserve from the registered bins.
    CHECK(resolved.tuning.chunkCapacity == resolved.parameters.chunkSize);
    CHECK(resolved.tuning.liveParticles == resolved.parameters.liveParticles);
    CHECK(resolved.tuning.tileSize == resolved.parameters.tileSize);
    CHECK(resolved.tuning.batchChunks == resolved.parameters.batchChunks);
    CHECK(resolved.tuning.frameCap == resolved.parameters.frameCap);

    // The largest feasible explicit P passes; the former preset P=16384 is rejected.
    graphFlow::cli::Options const fitting
        = parseArgs({"--particles", "100000000", "--chunk-size", "8192", "--live-particles", "16128"});
    graphFlow::cli::ResolvedOptions const explicitFitting = graphFlow::cli::resolveOptions(fitting, properties);
    CHECK(explicitFitting.policy.liveParticles == 16128u);
    CHECK(explicitFitting.policy.liveParticlesExplicit);
    CHECK(explicitFitting.liveParticlesExplicit);

    graphFlow::cli::Options const overBudget
        = parseArgs({"--particles", "100000000", "--chunk-size", "8192", "--live-particles", "16384"});
    CHECK_THROWS_AS(graphFlow::cli::resolveOptions(overBudget, properties), std::invalid_argument);
}

TEST_CASE("CLI backend and device selection follow the compiled contract", "[cli][backend][device]")
{
    using graphFlow::Backend;
    using graphFlow::BackendUnavailable;

    CHECK(graphFlow::parseBackend("cpu") == Backend::cpu);
    CHECK(graphFlow::parseBackend("cuda") == Backend::cuda);
    CHECK(graphFlow::parseBackend("hip") == Backend::hip);
    CHECK_FALSE(graphFlow::parseBackend("tpu").has_value());
    CHECK_THROWS_AS(graphFlow::requireBackend("tpu"), graphFlow::BackendError);
    CHECK_THROWS_WITH(graphFlow::requireBackend("tpu"), Catch::Matchers::ContainsSubstring("unknown backend 'tpu'"));

    CHECK(graphFlow::isBackendCompiled(Backend::cpu));

    auto const selectFirst = [](Backend backend)
    {
        bool called = false;
        bool namePresent = false;
        graphFlow::withBackendDevice(
            backend,
            0u,
            [&](auto&&, Backend selected, std::string const& deviceName)
            {
                called = selected == backend;
                namePresent = !deviceName.empty();
                return 0;
            });
        return called && namePresent;
    };

    CHECK(selectFirst(Backend::cpu));

    auto const cpuSelector
        = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
    REQUIRE(cpuSelector.isAvailable());
    REQUIRE(cpuSelector.getDeviceCount() > 0u);
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(
            Backend::cpu,
            cpuSelector.getDeviceCount(),
            [](auto&&, Backend, std::string const&) { return 0; }),
        std::out_of_range);

#if ALPAKA_LANG_CUDA
    CHECK(graphFlow::isBackendCompiled(Backend::cuda));
    // A CUDA build must expose a usable device: this throws BackendUnavailable when none is
    // visible, which fails the case instead of skipping the accelerator contract.
    REQUIRE_NOTHROW(selectFirst(Backend::cuda));
    auto const cudaSelector = alpaka::onHost::makeDeviceSelector(
        alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
    REQUIRE(cudaSelector.isAvailable());
    REQUIRE(cudaSelector.getDeviceCount() > 0u);
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(
            Backend::cuda,
            cudaSelector.getDeviceCount(),
            [](auto&&, Backend, std::string const&) { return 0; }),
        std::out_of_range);
#else
    CHECK_FALSE(graphFlow::isBackendCompiled(Backend::cuda));
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(Backend::cuda, 0u, [](auto&&, Backend, std::string const&) { return 0; }),
        BackendUnavailable);
#endif

#if ALPAKA_LANG_HIP
    CHECK(graphFlow::isBackendCompiled(Backend::hip));
    REQUIRE_NOTHROW(selectFirst(Backend::hip));
    auto const hipSelector
        = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{alpaka::api::hip, alpaka::deviceKind::amdGpu});
    REQUIRE(hipSelector.isAvailable());
    REQUIRE(hipSelector.getDeviceCount() > 0u);
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(
            Backend::hip,
            hipSelector.getDeviceCount(),
            [](auto&&, Backend, std::string const&) { return 0; }),
        std::out_of_range);
#else
    CHECK_FALSE(graphFlow::isBackendCompiled(Backend::hip));
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(Backend::hip, 0u, [](auto&&, Backend, std::string const&) { return 0; }),
        BackendUnavailable);
#endif
}

#if !defined(_WIN32)
TEST_CASE("CLI process contract maps usage errors to the documented exit codes", "[cli][exit]")
{
    REQUIRE(std::filesystem::exists(exampleBinaryPath()));

    ExampleRun const help = runExample("--help");
    CHECK(help.exitCode == 0);
    CHECK(help.output.find("usage: graphFlow_particle_feedback") != std::string::npos);

    ExampleRun const unknownBackend = runExample("--backend tpu --particles 0");
    CHECK(unknownBackend.exitCode == 2);
    CHECK(unknownBackend.output.find("unknown backend 'tpu'") != std::string::npos);

    ExampleRun const missingValue = runExample("--particles");
    CHECK(missingValue.exitCode == 2);
    CHECK(missingValue.output.find("missing value") != std::string::npos);

    ExampleRun const unknownArgument = runExample("--bogus");
    CHECK(unknownArgument.exitCode == 2);
    CHECK(unknownArgument.output.find("unknown argument '--bogus'") != std::string::npos);

    ExampleRun const badDevice = runExample("--backend cpu --device 999999 --particles 0");
    CHECK(badDevice.exitCode == 2);
    CHECK(badDevice.output.find("out of range for backend 'cpu'") != std::string::npos);

    // A run-stage overflow is not a usage error and takes the generic exit code 1.
    ExampleRun const overflow = runExample("--backend cpu --particles 0 --max-visits 65536");
    CHECK(overflow.exitCode == 1);
    CHECK(overflow.output.find("max visits exceeds") != std::string::npos);
}

TEST_CASE("CLI process contract runs the selected backend", "[cli][backend][device][exit]")
{
    REQUIRE(std::filesystem::exists(exampleBinaryPath()));

    ExampleRun const cpu = runExample("--backend cpu --particles 0 --chunk-size 128 --seed 42");
    CHECK(cpu.exitCode == 0);
    CHECK(cpu.output.find("# graphFlow_particle_feedback backend=cpu device=0 ") != std::string::npos);
    CHECK(cpu.output.find(" particles=0 ") != std::string::npos);
    CHECK(cpu.output.find("# weight,count") != std::string::npos);

#    if ALPAKA_LANG_CUDA
    ExampleRun const cuda = runExample("--backend cuda --particles 0 --chunk-size 128 --seed 42");
    CHECK(cuda.exitCode == 0);
    CHECK(cuda.output.find("# graphFlow_particle_feedback backend=cuda device=0 ") != std::string::npos);
    CHECK(cuda.output.find(" particles=0 ") != std::string::npos);

    ExampleRun const cudaBadDevice = runExample("--backend cuda --device 999999 --particles 0");
    CHECK(cudaBadDevice.exitCode == 2);
    CHECK(cudaBadDevice.output.find("out of range for backend 'cuda'") != std::string::npos);
#    else
    ExampleRun const cuda = runExample("--backend cuda --particles 0");
    CHECK(cuda.exitCode == 2);
    CHECK(cuda.output.find("not compiled into this build") != std::string::npos);
#    endif

#    if ALPAKA_LANG_HIP
    ExampleRun const hip = runExample("--backend hip --particles 0 --chunk-size 128 --seed 42");
    CHECK(hip.exitCode == 0);
    CHECK(hip.output.find("# graphFlow_particle_feedback backend=hip device=0 ") != std::string::npos);
    CHECK(hip.output.find(" particles=0 ") != std::string::npos);

    ExampleRun const hipBadDevice = runExample("--backend hip --device 999999 --particles 0");
    CHECK(hipBadDevice.exitCode == 2);
    CHECK(hipBadDevice.output.find("out of range for backend 'hip'") != std::string::npos);
#    else
    ExampleRun const hip = runExample("--backend hip --particles 0");
    CHECK(hip.exitCode == 2);
    CHECK(hip.output.find("not compiled into this build") != std::string::npos);
#    endif
}
#endif
