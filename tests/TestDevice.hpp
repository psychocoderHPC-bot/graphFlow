/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <stdexcept>

namespace graphFlow::test
{
    /** Create the first host (CPU) device. Tests are CPU-only in this slice. */
    inline auto makeCpuDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"no host compute device available"};
        return selector.makeDevice(0);
    }
} // namespace graphFlow::test
