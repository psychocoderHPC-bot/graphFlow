/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#if !ALPAKA_LANG_CUDA
#    error "test_cuda_kernel.cpp requires a CUDA compilation (GRAPHFLOW_ENABLE_CUDA=ON)"
#endif

namespace
{
    /** Deterministic element-wise kernel driven by the frame launch.
     *
     * Each block owns a strided subset of the tile descriptors, so a block may process several
     * tiles. The threads of a block stride over the slice of the selected tile. The result depends
     * on the block and thread indices, so the host comparison below would fail if the launch did
     * not execute on the selected device.
     */
    struct TransformKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const* input,
            std::uint32_t* output,
            std::uint32_t count,
            std::uint32_t tileSize,
            std::uint32_t numTiles,
            std::uint32_t salt) const
        {
            for(auto [tile] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::blocksInGrid, alpaka::IdxRange{numTiles}))
            {
                std::uint32_t const offset = static_cast<std::uint32_t>(tile) * tileSize;
                std::uint32_t const remaining = count > offset ? count - offset : 0u;
                std::uint32_t const slice = remaining < tileSize ? remaining : tileSize;

                for(auto [i] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{slice}))
                {
                    std::uint32_t const index = offset + static_cast<std::uint32_t>(i);
                    output[index] = input[index] * 3u + index + salt;
                }
            }
        }
    };
} // namespace

TEST_CASE("CUDA kernels execute on the selected NVIDIA GPU", "[cuda]")
{
    using namespace alpaka;

    auto selector = onHost::makeDeviceSelector(api::Cuda{}, deviceKind::nvidiaGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        FAIL("CUDA is compiled in but no NVIDIA GPU is visible; this [cuda] test requires real hardware");

    auto device = selector.makeDevice(0u);
    static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), api::Cuda>);
    static_assert(std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), deviceKind::NvidiaGpu>);

    std::string const deviceName = device.getName();
    std::cout << "# cuda-kernel-test backend=Cuda device-name=" << deviceName << '\n';
    REQUIRE_FALSE(deviceName.empty());

    auto const properties = device.getDeviceProperties();
    std::uint32_t const threadsPerBlock = std::min<std::uint32_t>(properties.maxThreadsPerBlock, 256u);
    REQUIRE(threadsPerBlock > 0u);

    // 4096 tiles are more than any current GPU has multiprocessors, so frameCap=1 forces each
    // block to process several tiles.
    std::uint32_t const count = 1u << 20;
    std::uint32_t const numTiles = (count + threadsPerBlock - 1u) / threadsPerBlock;
    graphFlow::LaunchPolicy const policy = graphFlow::launchPolicy(device, count, threadsPerBlock, 1u);
    CHECK(policy.frameExtent == threadsPerBlock);
    CHECK(policy.numFrames <= properties.maxBlocksPerGrid);
    CHECK(policy.numFrames < numTiles);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(count));
    for(std::uint32_t i = 0u; i < count; ++i)
        input[i] = i * 7u + 11u;

    std::uint32_t const salt = 12345u;
    std::vector<std::uint32_t> expected(static_cast<std::size_t>(count));
    for(std::uint32_t i = 0u; i < count; ++i)
        expected[i] = input[i] * 3u + i + salt;

    auto inputDevice = onHost::alloc<std::uint32_t>(device, Vec{std::uint64_t{count}});
    auto outputDevice = onHost::alloc<std::uint32_t>(device, Vec{std::uint64_t{count}});

    auto queue = device.makeQueue(queueKind::blocking);
    onHost::memcpy(queue, inputDevice, input);
    // Sentinel so a kernel that never ran is observable, not silently equal to zeroed memory.
    onHost::fill(queue, outputDevice, std::uint32_t{0xDEAD'BEEFu});

    queue.enqueue(
        policy.frameSpec(),
        KernelBundle{
            TransformKernel{},
            inputDevice.data(),
            outputDevice.data(),
            count,
            threadsPerBlock,
            numTiles,
            salt});

    std::vector<std::uint32_t> output(static_cast<std::size_t>(count), 0u);
    onHost::memcpy(queue, output, outputDevice);

    CHECK(output == expected);
    CHECK(std::none_of(output.begin(), output.end(), [](std::uint32_t value) { return value == 0xDEAD'BEEFu; }));
}

TEST_CASE("CUDA particle feedback matches the scalar oracle through FrameSpec launches", "[cuda]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        FAIL("CUDA is compiled in but no NVIDIA GPU is visible; this [cuda] test requires real hardware");
    auto device = selector.makeDevice(0u);

    for(std::uint32_t const tileSize : {32u, 128u})
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = 4097u;
        parameters.chunkSize = 128u;
        parameters.tileSize = tileSize;
        parameters.batchChunks = 2u;
        parameters.frameCap = 8u;
        parameters.seed = 42u;
        parameters.maxVisits = 64u;
        parameters.stallTimeoutMs = 120000u;

        CAPTURE(tileSize);
        auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
        auto const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);
        CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
        CHECK(result.weights == reference.weights);
        CHECK(result.sunkParticles == parameters.numParticles);
    }
}

TEST_CASE("CUDA frame launch rejects a tile wider than maxThreadsPerBlock", "[cuda]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        FAIL("CUDA is compiled in but no NVIDIA GPU is visible; this [cuda] test requires real hardware");
    auto device = selector.makeDevice(0u);
    auto const properties = device.getDeviceProperties();

    CHECK_THROWS_AS(
        graphFlow::launchPolicy(device, 1024u, properties.maxThreadsPerBlock + 1u, 8u),
        std::invalid_argument);
}
