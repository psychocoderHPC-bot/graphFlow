/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "integration/PerChunkSchedule.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <stdexcept>
#include <utility>

#if !ALPAKA_LANG_HIP
#    error "test_per_chunk_execution_hip.cpp requires a HIP compilation (GRAPHFLOW_ENABLE_HIP=ON)"
#endif

namespace
{
    using HipSelector = decltype(alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu));
    using HipDevice = decltype(std::declval<HipSelector&>().makeDevice(0u));

    /** Fail instead of silently falling back to the host when HIP has no visible device. */
    HipDevice makeHipDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"HIP is compiled in but no AMD GPU is visible; this test requires real hardware"};
        auto device = selector.makeDevice(0u);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Hip>);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), alpaka::deviceKind::AmdGpu>);
        return device;
    }
} // namespace

TEST_CASE(
    "HIP per-chunk transform output equals the batch-wide output across the C/T/Q matrix",
    "[hip][chunk-schedule]")
{
    auto device = makeHipDevice();
    // RDNA reports maxThreadsPerBlock == 256: a 512-wide launch must be rejected, not narrowed.
    graphFlow::test::perChunk::checkWideTileRejection(device);
    graphFlow::test::perChunk::runStageMatrix(device);
}

TEST_CASE("HIP per-chunk flow equals the batch-wide flow at Q=1 and Q>=8", "[hip][chunk-schedule]")
{
    auto device = makeHipDevice();
    graphFlow::test::perChunk::runFlowMatrix(device);
}
