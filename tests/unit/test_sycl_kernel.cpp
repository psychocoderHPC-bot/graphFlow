/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/core/Backend.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#if !ALPAKA_LANG_ONEAPI
#    error "test_sycl_kernel.cpp requires a SYCL compilation (GRAPHFLOW_ENABLE_SYCL=ON)"
#endif

#if !defined(ALPAKA_DISABLE_OneApi_Cpu)
#    error "the SYCL build must disable the OpenCL CPU device kind (alpaka_ONEAPI_Cpu=OFF)"
#endif

namespace
{
    /** Deterministic scalar kernel.
     *
     * The result depends on the global thread index, so a launch that never reached the device
     * cannot match the host loop below.
     */
    struct ScaleAddKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const* input,
            std::uint32_t* output,
            std::uint32_t count,
            std::uint32_t salt) const
        {
            for(auto [i] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{count}))
            {
                std::uint32_t const index = static_cast<std::uint32_t>(i);
                output[index] = input[index] * 3u + index + salt;
            }
        }
    };
} // namespace

TEST_CASE("SYCL backend dispatch selects an Intel GPU", "[sycl][backend]")
{
    using namespace graphFlow;

    CHECK(parseBackend("sycl") == Backend::sycl);
    CHECK(backendName(Backend::sycl) == std::string_view{"sycl"});
    CHECK(isBackendCompiled(Backend::sycl));

    int visited = 0;
    withBackendDevice(
        Backend::sycl,
        0u,
        [&](auto device, Backend selected, std::string const& deviceName)
        {
            // The callback is instantiated for the host branch too, so check the tag types as
            // values instead of a static_assert that would fire for the host device.
            CHECK(selected == Backend::sycl);
            CHECK((std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>) );
            CHECK((std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), alpaka::deviceKind::IntelGpu>) );
            CHECK_FALSE(deviceName.empty());
            ++visited;
            return 0;
        });
    CHECK(visited == 1);
}

TEST_CASE("SYCL scalar kernel executes on the selected Intel GPU", "[sycl]")
{
    using namespace alpaka;

    auto device = graphFlow::test::makeSyclDevice();
    static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), api::OneApi>);
    static_assert(std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), deviceKind::IntelGpu>);

    std::string const deviceName = device.getName();
    std::cout << "# sycl-kernel-test backend=OneApi device-name=" << deviceName << '\n';
    REQUIRE_FALSE(deviceName.empty());

    auto const properties = device.getDeviceProperties();
    std::uint32_t const threadsPerBlock = std::min<std::uint32_t>(properties.maxThreadsPerBlock, 256u);
    REQUIRE(threadsPerBlock > 0u);

    std::uint32_t const count = 1u << 18;
    std::uint32_t const numFrames = (count + threadsPerBlock - 1u) / threadsPerBlock;
    std::uint32_t const salt = 12345u;

    std::vector<std::uint32_t> input(static_cast<std::size_t>(count));
    std::vector<std::uint32_t> expected(static_cast<std::size_t>(count));
    for(std::uint32_t i = 0u; i < count; ++i)
    {
        input[i] = i * 7u + 11u;
        expected[i] = input[i] * 3u + i + salt;
    }

    auto inputDevice = onHost::alloc<std::uint32_t>(device, Vec{std::uint64_t{count}});
    auto outputDevice = onHost::alloc<std::uint32_t>(device, Vec{std::uint64_t{count}});

    auto queue = device.makeQueue(queueKind::blocking);
    onHost::memcpy(queue, inputDevice, input);
    // Sentinel so a kernel that never ran is observable, not silently equal to zeroed memory.
    onHost::fill(queue, outputDevice, std::uint32_t{0xDEAD'BEEFu});

    queue.enqueue(
        onHost::FrameSpec{numFrames, threadsPerBlock},
        KernelBundle{ScaleAddKernel{}, inputDevice.data(), outputDevice.data(), count, salt});

    std::vector<std::uint32_t> output(static_cast<std::size_t>(count), 0u);
    onHost::memcpy(queue, output, outputDevice);

    CHECK(output == expected);
    CHECK(std::none_of(output.begin(), output.end(), [](std::uint32_t value) { return value == 0xDEAD'BEEFu; }));
}
