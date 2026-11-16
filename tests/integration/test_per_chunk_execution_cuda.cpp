/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "integration/PerChunkSchedule.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <stdexcept>
#include <utility>

#if !ALPAKA_LANG_CUDA
#    error "test_per_chunk_execution_cuda.cpp requires a CUDA compilation (GRAPHFLOW_ENABLE_CUDA=ON)"
#endif

namespace
{
    using CudaSelector
        = decltype(alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu));
    using CudaDevice = decltype(std::declval<CudaSelector&>().makeDevice(0u));

    /** Fail instead of silently falling back to the host when CUDA has no visible device. */
    CudaDevice makeCudaDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{
                "CUDA is compiled in but no NVIDIA GPU is visible; this test requires real hardware"};
        auto device = selector.makeDevice(0u);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Cuda>);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), alpaka::deviceKind::NvidiaGpu>);
        return device;
    }
} // namespace

TEST_CASE(
    "CUDA per-chunk transform output equals the batch-wide output across the C/T/Q matrix",
    "[cuda][chunk-schedule]")
{
    auto device = makeCudaDevice();
    graphFlow::test::perChunk::checkWideTileRejection(device);
    graphFlow::test::perChunk::runStageMatrix(device);
}

TEST_CASE("CUDA per-chunk flow equals the batch-wide flow at Q=1 and Q>=8", "[cuda][chunk-schedule]")
{
    auto device = makeCudaDevice();
    graphFlow::test::perChunk::runFlowMatrix(device);
}
