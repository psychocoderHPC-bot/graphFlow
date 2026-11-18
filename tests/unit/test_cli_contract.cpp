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
    CHECK(defaults.chunkKernelSchedule == "off");
    CHECK(defaults.chunkQueueDepth == 1u);
    CHECK(defaults.chunkLatencyCapUs == 0u);
    CHECK_FALSE(defaults.schedulerIdleWait.has_value());

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

TEST_CASE("CLI accepts --occupancy-target 0 as the disabled default", "[cli][admission]")
{
    graphFlow::cli::ResolvedOptions const defaults = graphFlow::cli::resolveOptions(parseArgs({}), cpuProperties());
    CHECK_FALSE(defaults.tuning.occupancyTargetChunks.has_value());

    graphFlow::cli::ResolvedOptions const zero
        = graphFlow::cli::resolveOptions(parseArgs({"--occupancy-target", "0"}), cpuProperties());
    REQUIRE(zero.tuning.occupancyTargetChunks.has_value());
    CHECK(*zero.tuning.occupancyTargetChunks == 0u);
    // A present zero is accepted and means "disabled", the same spelling as `Config`.
    CHECK_NOTHROW(zero.tuning.validate());

    graphFlow::cli::ResolvedOptions const gated
        = graphFlow::cli::resolveOptions(parseArgs({"--occupancy-target", "4"}), cpuProperties());
    REQUIRE(gated.tuning.occupancyTargetChunks.has_value());
    CHECK(*gated.tuning.occupancyTargetChunks == 4u);
    CHECK_NOTHROW(gated.tuning.validate());
}

TEST_CASE("CLI accepts --scheduler-idle-wait auto|on|off", "[cli][scheduler]")
{
    CHECK_FALSE(parseArgs({}).schedulerIdleWait.has_value());

    graphFlow::cli::Options const on = parseArgs({"--scheduler-idle-wait", "on"});
    REQUIRE(on.schedulerIdleWait.has_value());
    CHECK(*on.schedulerIdleWait);

    graphFlow::cli::Options const off = parseArgs({"--scheduler-idle-wait", "off"});
    REQUIRE(off.schedulerIdleWait.has_value());
    CHECK_FALSE(*off.schedulerIdleWait);

    CHECK_FALSE(parseArgs({"--scheduler-idle-wait", "auto"}).schedulerIdleWait.has_value());

    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(on, cpuProperties());
    REQUIRE(resolved.tuning.schedulerIdleWait.has_value());
    CHECK(*resolved.tuning.schedulerIdleWait);
    CHECK_NOTHROW(resolved.tuning.validate());

    CHECK_THROWS_WITH(
        parseArgs({"--scheduler-idle-wait", "sometimes"}),
        Catch::Matchers::ContainsSubstring("expected 'on', 'off' or 'auto'"));
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
    CHECK(graphFlow::parseBackend("sycl") == Backend::sycl);
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

#if ALPAKA_LANG_ONEAPI
    CHECK(graphFlow::isBackendCompiled(Backend::sycl));
    // A SYCL build must expose an Intel GPU device: this throws BackendUnavailable when none is
    // visible, which fails the case instead of skipping the accelerator contract.
    REQUIRE_NOTHROW(selectFirst(Backend::sycl));
    auto const syclSelector = alpaka::onHost::makeDeviceSelector(
        alpaka::onHost::DeviceSpec{alpaka::api::oneApi, alpaka::deviceKind::intelGpu});
    REQUIRE(syclSelector.isAvailable());
    REQUIRE(syclSelector.getDeviceCount() > 0u);
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(
            Backend::sycl,
            syclSelector.getDeviceCount(),
            [](auto&&, Backend, std::string const&) { return 0; }),
        std::out_of_range);
#else
    CHECK_FALSE(graphFlow::isBackendCompiled(Backend::sycl));
    CHECK_THROWS_AS(
        graphFlow::withBackendDevice(Backend::sycl, 0u, [](auto&&, Backend, std::string const&) { return 0; }),
        BackendUnavailable);
#endif
}

TEST_CASE("CLI parses the opt-in kernel workload axes", "[cli][kernel-axes]")
{
    graphFlow::cli::Options const options = parseArgs(
        {"--workload",
         "kernel",
         "--scatter",
         "direct",
         "--bin-path",
         "dense",
         "--bins",
         "1024",
         "--distribution",
         "skewed"});

    CHECK(options.workload == graphFlow::particle::Workload::kernel);
    REQUIRE(options.scatter.has_value());
    CHECK(*options.scatter == graphFlow::particle::ScatterChoice::direct);
    REQUIRE(options.binPath.has_value());
    CHECK(*options.binPath == graphFlow::particle::BinPath::dense);
    REQUIRE(options.bins.has_value());
    CHECK(*options.bins == 1024u);
    REQUIRE(options.distribution.has_value());
    CHECK(*options.distribution == graphFlow::particle::Distribution::skewed);

    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, cpuProperties());
    CHECK(resolved.workload == graphFlow::particle::Workload::kernel);
    CHECK(resolved.kernel.scatter == graphFlow::particle::ScatterChoice::direct);
    CHECK(resolved.kernel.binPath == graphFlow::particle::BinPath::dense);
    CHECK(resolved.kernel.bins == 1024u);
    CHECK(resolved.kernel.distribution == graphFlow::particle::Distribution::skewed);

    graphFlow::cli::ResolvedOptions const defaults
        = graphFlow::cli::resolveOptions(parseArgs({"--workload", "kernel"}), cpuProperties());
    CHECK(defaults.kernel.scatter == graphFlow::particle::ScatterChoice::shared);
    CHECK(defaults.kernel.binPath == graphFlow::particle::BinPath::dense);
    CHECK(defaults.kernel.bins == 64u);
    CHECK(defaults.kernel.distribution == graphFlow::particle::Distribution::balanced);
}

TEST_CASE("CLI rejects axis flags on the feedback workload and invalid combinations", "[cli][kernel-axes]")
{
    using Catch::Matchers::ContainsSubstring;

    CHECK_THROWS_WITH(parseArgs({"--workload", "bogus"}), ContainsSubstring("expected feedback or kernel"));
    CHECK_THROWS_WITH(parseArgs({"--scatter", "staged"}), ContainsSubstring("expected shared or direct"));
    CHECK_THROWS_WITH(parseArgs({"--bin-path", "dense2"}), ContainsSubstring("expected dense or hash"));
    CHECK_THROWS_WITH(
        parseArgs({"--distribution", "uniform"}),
        ContainsSubstring("expected balanced, one-hot, skewed or sparse"));

    // The default feedback workload keeps its exact argument set; an axis without --workload
    // kernel is a usage error instead of a silent no-op.
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(parseArgs({"--scatter", "direct"}), cpuProperties()),
        ContainsSubstring("require --workload kernel"));

    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(parseArgs({"--workload", "kernel", "--bins", "128"}), cpuProperties()),
        ContainsSubstring("instantiates 64, 1024, 4096 and 65536"));
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(parseArgs({"--workload", "kernel", "--bins", "4096"}), cpuProperties()),
        ContainsSubstring("dense supports at most 1024"));
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(
            parseArgs({"--workload", "kernel", "--bin-path", "hash", "--scatter", "direct"}),
            cpuProperties()),
        ContainsSubstring("hash path always stages"));
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(
            parseArgs({"--workload", "kernel", "--distribution", "sparse", "--bins", "64"}),
            cpuProperties()),
        ContainsSubstring("sparse requires --bins 1024 or larger"));
}

#if !defined(_WIN32)
TEST_CASE("CLI process contract maps usage errors to the documented exit codes", "[cli][exit]")
{
    REQUIRE(std::filesystem::exists(exampleBinaryPath()));

    ExampleRun const help = runExample("--help");
    CHECK(help.exitCode == 0);
    CHECK(help.output.find("usage: graphFlow_particle_feedback") != std::string::npos);
    // The usage text names the limitation of the parity flags instead of implying an effect.
    CHECK(help.output.find("accepted for CLI parity and change nothing here") != std::string::npos);

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

    // The per-chunk flags are accepted for parity, and the run states that they cannot take
    // effect: no transform node exists, so the scheduler still reports the off path.
    ExampleRun const inertChunk = runExample(
        "--backend cpu --particles 0 --chunk-kernel-schedule perChunk --chunk-queue-depth 4 "
        "--chunk-latency-cap-us 25");
    CHECK(inertChunk.exitCode == 0);
    CHECK(
        inertChunk.output.find(
            "# note: chunk-kernel-schedule, chunk-queue-depth and chunk-latency-cap-us have no effect")
        != std::string::npos);
    CHECK(inertChunk.output.find("chunkLaunches=0 chunkLimit=none") != std::string::npos);

    // A non-zero particle count proves the dispatch inertness instead of inferring it from an
    // empty particle set: particles are transported while the chunk flags are set, and the
    // scheduler line still reports no per-chunk launch and the off-path clamp.
    ExampleRun const inertChunkNonZero = runExample(
        "--backend cpu --particles 64 --chunk-size 16 --live-particles 64 --seed 42 "
        "--chunk-kernel-schedule perChunk --chunk-queue-depth 4 --chunk-latency-cap-us 25");
    CHECK(inertChunkNonZero.exitCode == 0);
    CHECK(inertChunkNonZero.output.find(" particles=64 ") != std::string::npos);
    CHECK(inertChunkNonZero.output.find("# note: chunk-kernel-schedule") != std::string::npos);
    CHECK(inertChunkNonZero.output.find("chunkLaunches=0 chunkLimit=none") != std::string::npos);

    // The note gate is an OR over the three flags, so pin every flag on its own.
    for(char const* flag : {"--chunk-kernel-schedule perChunk", "--chunk-queue-depth 4", "--chunk-latency-cap-us 25"})
    {
        ExampleRun const oneFlag = runExample(std::string{"--backend cpu --particles 0 "} + flag);
        INFO("single flag: " << flag);
        CHECK(oneFlag.exitCode == 0);
        CHECK(oneFlag.output.find("# note: chunk-kernel-schedule") != std::string::npos);
    }

    // Passing the defaults explicitly changes nothing, so it must stay silent.
    ExampleRun const explicitDefaults = runExample(
        "--backend cpu --particles 0 --chunk-kernel-schedule off --chunk-queue-depth 1 --chunk-latency-cap-us 0");
    CHECK(explicitDefaults.exitCode == 0);
    CHECK(explicitDefaults.output.find("# note: chunk-kernel-schedule") == std::string::npos);

    // Without the flags the note is absent, so the default output is unchanged.
    CHECK(cpu.output.find("# note: chunk-kernel-schedule") == std::string::npos);

    ExampleRun const kernel = runExample(
        "--backend cpu --workload kernel --particles 64 --chunk-size 16 --live-particles 64 "
        "--bins 64 --distribution balanced");
    CHECK(kernel.exitCode == 0);
    CHECK(kernel.output.find("workload=kernel") != std::string::npos);
    CHECK(kernel.output.find("scatter=shared bin-path=dense bins=64 distribution=balanced") != std::string::npos);
    CHECK(kernel.output.find("# weight,count") != std::string::npos);

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

#    if ALPAKA_LANG_ONEAPI
    ExampleRun const sycl = runExample("--backend sycl --particles 0 --chunk-size 128 --seed 42");
    CHECK(sycl.exitCode == 0);
    CHECK(sycl.output.find("# graphFlow_particle_feedback backend=sycl device=0 ") != std::string::npos);
    CHECK(sycl.output.find(" particles=0 ") != std::string::npos);

    ExampleRun const syclBadDevice = runExample("--backend sycl --device 999999 --particles 0");
    CHECK(syclBadDevice.exitCode == 2);
    CHECK(syclBadDevice.output.find("out of range for backend 'sycl'") != std::string::npos);
#    else
    ExampleRun const sycl = runExample("--backend sycl --particles 0");
    CHECK(sycl.exitCode == 2);
    CHECK(sycl.output.find("not compiled into this build") != std::string::npos);
#    endif
}
#endif

TEST_CASE("CLI exposes the slot-arena budget", "[cli][arena]")
{
    graphFlow::cli::Options const defaults = parseArgs({});
    CHECK_FALSE(defaults.slotArenaBudgetBytes.has_value());

    graphFlow::cli::Options const parsed = parseArgs({"--pool-mode", "slotArena", "--slot-arena-budget", "67108864"});
    REQUIRE(parsed.slotArenaBudgetBytes.has_value());
    CHECK(*parsed.slotArenaBudgetBytes == 67'108'864u);

    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(parsed, cpuProperties());
    CHECK(resolved.parameters.poolMode == graphFlow::PoolMode::slotArena);
    CHECK(resolved.parameters.slotArenaBudgetBytes == 67'108'864u);
    REQUIRE(resolved.tuning.slotArenaBudgetBytes.has_value());
    CHECK(*resolved.tuning.slotArenaBudgetBytes == 67'108'864u);

    // Zero is a defined value: it keeps the whole pool budget, like an absent flag.
    graphFlow::cli::ResolvedOptions const zero = graphFlow::cli::resolveOptions(
        parseArgs({"--pool-mode", "slotArena", "--slot-arena-budget", "0"}),
        cpuProperties());
    CHECK(zero.parameters.slotArenaBudgetBytes == 0u);
    CHECK_FALSE(zero.tuning.slotArenaBudgetBytes.has_value());

    // An arena budget above the resolved pool budget raises the pool budget: the arena is the
    // payload, so a larger arena must also raise the cap that bounds it.
    graphFlow::cli::ResolvedOptions const raised = graphFlow::cli::resolveOptions(
        parseArgs({"--pool-mode", "slotArena", "--slot-arena-budget", "67108864"}),
        cpuProperties());
    REQUIRE(raised.tuning.slotArenaBudgetBytes.has_value());
    REQUIRE(raised.tuning.poolPayloadBudgetBytes.has_value());
    CHECK(*raised.tuning.poolPayloadBudgetBytes == 67'108'864u);

    // An arena budget below the pool budget leaves the pool budget to the device policy.
    graphFlow::cli::ResolvedOptions const small = graphFlow::cli::resolveOptions(
        parseArgs({"--pool-mode", "slotArena", "--slot-arena-budget", "4096"}),
        cpuProperties());
    CHECK_FALSE(small.tuning.poolPayloadBudgetBytes.has_value());

    // The budget only applies to the slot arena, so a fixed run rejects it instead of ignoring it.
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(parseArgs({"--slot-arena-budget", "1024"}), cpuProperties()),
        Catch::Matchers::ContainsSubstring("--slot-arena-budget requires --pool-mode slotArena"));

    // An explicit payload budget is never rewritten by the arena. The arena only raises a derived
    // budget, so an explicitly larger pool budget survives and no arena-sized override appears.
    graphFlow::DevicePolicyConfig explicitLarge;
    explicitLarge.memoryBudgetBytes = 134'217'728u;
    graphFlow::cli::ResolvedOptions const kept
        = graphFlow::cli::resolveOptions(parsed, cpuProperties(), explicitLarge);
    CHECK(kept.policy.memoryBudgetBytes == 134'217'728u);
    CHECK_FALSE(kept.tuning.poolPayloadBudgetBytes.has_value());

    // An explicit budget smaller than the arena cannot hold the arena, so the combination is
    // rejected before the run and the message names both byte counts.
    graphFlow::DevicePolicyConfig explicitSmall;
    explicitSmall.memoryBudgetBytes = 1'048'576u;
    std::string budgetError;
    try
    {
        (void) graphFlow::cli::resolveOptions(parsed, cpuProperties(), explicitSmall);
    }
    catch(std::invalid_argument const& error)
    {
        budgetError = error.what();
    }
    CHECK(budgetError.find("(67108864 bytes)") != std::string::npos);
    CHECK(budgetError.find("(1048576 bytes)") != std::string::npos);
}

TEST_CASE("CLI accepts the per-chunk transform flags for parity", "[cli][chunk-schedule]")
{
    using Catch::Matchers::ContainsSubstring;

    // The defaults stay the historical configuration, so an unflagged run is unchanged.
    graphFlow::cli::ResolvedOptions const defaults = graphFlow::cli::resolveOptions(parseArgs({}), cpuProperties());
    CHECK(defaults.tuning.chunkKernelSchedule == graphFlow::ChunkKernelSchedule::off);
    REQUIRE(defaults.tuning.chunkQueueDepth.has_value());
    CHECK(*defaults.tuning.chunkQueueDepth == 1u);
    REQUIRE(defaults.tuning.chunkLatencyCapUs.has_value());
    CHECK(*defaults.tuning.chunkLatencyCapUs == 0u);

    graphFlow::cli::Options const options = parseArgs(
        {"--chunk-kernel-schedule", "perChunk", "--chunk-queue-depth", "4", "--chunk-latency-cap-us", "25"});
    CHECK(options.chunkKernelSchedule == "perChunk");
    CHECK(options.chunkQueueDepth == 4u);
    CHECK(options.chunkLatencyCapUs == 25u);

    // The values are parsed and carried into `Tuning` so a shared command line is accepted by both
    // examples. No graph in this example registers a transform node, so they cannot change the run.
    graphFlow::cli::ResolvedOptions const resolved = graphFlow::cli::resolveOptions(options, cpuProperties());
    CHECK(resolved.tuning.chunkKernelSchedule == graphFlow::ChunkKernelSchedule::perChunk);
    REQUIRE(resolved.tuning.chunkQueueDepth.has_value());
    CHECK(*resolved.tuning.chunkQueueDepth == 4u);
    REQUIRE(resolved.tuning.chunkLatencyCapUs.has_value());
    CHECK(*resolved.tuning.chunkLatencyCapUs == 25u);
    CHECK_NOTHROW(resolved.tuning.validate());

    // Zero latency is the defined "no cap" value and zero depth is a usage error, matching the
    // photon example.
    CHECK_NOTHROW(parseArgs({"--chunk-latency-cap-us", "0"}));
    CHECK_THROWS_WITH(parseArgs({"--chunk-queue-depth", "0"}), ContainsSubstring("must be non-zero"));
    CHECK_THROWS_WITH(
        graphFlow::cli::resolveOptions(parseArgs({"--chunk-kernel-schedule", "batch"}), cpuProperties()),
        ContainsSubstring("expected off or perChunk"));
}
