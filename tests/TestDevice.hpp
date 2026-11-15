/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <stdexcept>

namespace graphFlow::test
{
    /** Create the first host (CPU) device. */
    inline auto makeCpuDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"no host compute device available"};
        return selector.makeDevice(0);
    }

#if defined(ALPAKA_LANG_CUDA) && ALPAKA_LANG_CUDA
    /** Create the first CUDA device or fail; a compiled CUDA build must not skip its tests. */
    inline auto makeCudaDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{
                "CUDA is compiled in but no NVIDIA GPU is visible; this test requires real hardware"};
        return selector.makeDevice(0u);
    }
#endif

#if defined(ALPAKA_LANG_HIP) && ALPAKA_LANG_HIP
    /** Create the first HIP device or fail; a compiled HIP build must not skip its tests. */
    inline auto makeHipDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"HIP is compiled in but no AMD GPU is visible; this test requires real hardware"};
        return selector.makeDevice(0u);
    }
#endif

#if defined(ALPAKA_LANG_ONEAPI) && ALPAKA_LANG_ONEAPI
    /** Create the first oneAPI Intel GPU or fail; a compiled SYCL build must not skip its tests. */
    inline auto makeSyclDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::OneApi{}, alpaka::deviceKind::intelGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{
                "SYCL is compiled in but no Intel GPU is visible; this test requires real hardware"};
        return selector.makeDevice(0u);
    }
#endif
} // namespace graphFlow::test
