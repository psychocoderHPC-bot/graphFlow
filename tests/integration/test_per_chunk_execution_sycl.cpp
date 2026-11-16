/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "integration/PerChunkSchedule.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <stdexcept>
#include <utility>

#if !ALPAKA_LANG_ONEAPI
#    error "test_per_chunk_execution_sycl.cpp requires a SYCL compilation (GRAPHFLOW_ENABLE_SYCL=ON)"
#endif

namespace
{
    using SyclSelector
        = decltype(alpaka::onHost::makeDeviceSelector(alpaka::api::OneApi{}, alpaka::deviceKind::intelGpu));
    using SyclDevice = decltype(std::declval<SyclSelector&>().makeDevice(0u));

    /** Fail instead of silently falling back to the host when SYCL has no visible device. */
    SyclDevice makeSyclDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::OneApi{}, alpaka::deviceKind::intelGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{
                "SYCL is compiled in but no Intel GPU is visible; this test requires real hardware"};
        auto device = selector.makeDevice(0u);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), alpaka::deviceKind::IntelGpu>);
        return device;
    }
} // namespace

TEST_CASE(
    "SYCL per-chunk transform output equals the batch-wide output across the C/T/Q matrix",
    "[sycl][chunk-schedule]")
{
    auto device = makeSyclDevice();
    graphFlow::test::perChunk::checkWideTileRejection(device);
    graphFlow::test::perChunk::runStageMatrix(device);
}

TEST_CASE("SYCL per-chunk flow equals the batch-wide flow at Q=1 and Q>=8", "[sycl][chunk-schedule]")
{
    auto device = makeSyclDevice();
    graphFlow::test::perChunk::runFlowMatrix(device);
}
