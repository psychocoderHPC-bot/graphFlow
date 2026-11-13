/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

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
    /** Deterministic element-wise kernel.
     *
     * Each thread block owns one tile of the flat element range and the block threads stride over
     * that tile. The result depends on both the block and the thread index, so the host comparison
     * below would fail if the launch did not execute on the selected device.
     */
    struct TransformKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const* input,
            std::uint32_t* output,
            std::uint32_t count,
            std::uint32_t tileSize,
            std::uint32_t salt) const
        {
            auto const [tile] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks);
            std::uint32_t const offset = static_cast<std::uint32_t>(tile) * tileSize;
            std::uint32_t const remaining = count > offset ? count - offset : 0u;
            std::uint32_t const slice = remaining < tileSize ? remaining : tileSize;

            for(auto [i] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{slice}))
            {
                std::uint32_t const index = offset + static_cast<std::uint32_t>(i);
                output[index] = input[index] * 3u + index + salt;
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

    std::uint32_t const count = 4096u;
    std::uint32_t const threadsPerBlock
        = std::min<std::uint32_t>(device.getDeviceProperties().maxThreadsPerBlock, 256u);
    REQUIRE(threadsPerBlock > 0u);
    std::uint32_t const numBlocks = (count + threadsPerBlock - 1u) / threadsPerBlock;

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
    onHost::fill(queue, outputDevice, std::uint32_t{0xDEADBEEFu});

    queue.enqueue(
        onHost::ThreadSpec{numBlocks, threadsPerBlock},
        KernelBundle{
            TransformKernel{},
            inputDevice.data(),
            outputDevice.data(),
            count,
            threadsPerBlock,
            salt});

    std::vector<std::uint32_t> output(static_cast<std::size_t>(count), 0u);
    onHost::memcpy(queue, output, outputDevice);

    CHECK(output == expected);
    CHECK(std::none_of(output.begin(), output.end(), [](std::uint32_t value) { return value == 0xDEADBEEFu; }));
}
