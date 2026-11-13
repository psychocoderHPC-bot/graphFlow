/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace graphFlow
{
    /** Default upper bound of blocks per multiprocessor used by the frame heuristic. */
    inline constexpr std::uint32_t defaultFrameCap = 8u;

    /** Resolved frame launch of one kernel.
     *
     * `numFrames` is the number of logical frames (thread blocks on CUDA/HIP), `frameExtent` is
     * the physical block extent and `tileSize` is the logical tile width `T` used by the kernels.
     * The host backend keeps `frameExtent == 1` because its executors accept one real thread per
     * block; the kernels therefore never use the block extent as the tile width.
     */
    struct LaunchPolicy
    {
        std::uint32_t numFrames = 0u;
        std::uint32_t frameExtent = 1u;
        std::uint32_t tileSize = 1u;

        [[nodiscard]] auto frameSpec() const
        {
            return alpaka::onHost::FrameSpec{numFrames, frameExtent};
        }
    };

    /** Number of frames for `work` elements of width `tileSize`.
     *
     * `numFrames = min(ceil(work / T), maxBlocksPerGrid, frameCap * multiProcessorCount)`. A zero
     * amount of work returns zero frames and the caller must skip the launch. `T` is validated
     * against `maxThreadsPerBlock` for every backend, including the host backend whose real block
     * extent is one.
     */
    inline std::uint32_t numFramesFor(
        std::uint64_t work,
        std::uint32_t tileSize,
        std::uint32_t frameCap,
        alpaka::onHost::DeviceProperties const& properties)
    {
        if(tileSize == 0u)
            throw std::invalid_argument{"graphFlow::LaunchPolicy: tileSize must be non-zero"};
        if(frameCap == 0u)
            throw std::invalid_argument{"graphFlow::LaunchPolicy: frameCap must be non-zero"};
        if(properties.maxThreadsPerBlock == 0u)
            throw std::runtime_error{"graphFlow::LaunchPolicy: device reports maxThreadsPerBlock == 0"};
        if(tileSize > properties.maxThreadsPerBlock)
            throw std::invalid_argument{
                "graphFlow::LaunchPolicy: tileSize " + std::to_string(tileSize) + " exceeds maxThreadsPerBlock "
                + std::to_string(properties.maxThreadsPerBlock)};
        if(properties.maxBlocksPerGrid == 0u)
            throw std::runtime_error{"graphFlow::LaunchPolicy: device reports maxBlocksPerGrid == 0"};
        if(work == 0u)
            return 0u;

        std::uint64_t const tiles
            = (work + static_cast<std::uint64_t>(tileSize) - 1u) / static_cast<std::uint64_t>(tileSize);
        std::uint64_t const sms = properties.multiProcessorCount == 0u ? 1u : properties.multiProcessorCount;
        std::uint64_t const capped = static_cast<std::uint64_t>(frameCap) * sms;
        std::uint64_t const frames
            = std::min({tiles, static_cast<std::uint64_t>(properties.maxBlocksPerGrid), capped});
        if(frames == 0u)
            throw std::runtime_error{"graphFlow::LaunchPolicy: frame count underflow"};
        return static_cast<std::uint32_t>(frames);
    }

    /** Create a frame launch for `work` elements of logical width `tileSize`.
     *
     * Accelerators use `T` as the physical block extent; the host backend uses one so its
     * serial and OpenMP block executors accept the launch.
     */
    template<typename TDevice>
    LaunchPolicy launchPolicy(
        TDevice const& device,
        std::uint64_t work,
        std::uint32_t tileSize,
        std::uint32_t frameCap = defaultFrameCap)
    {
        auto const properties = device.getDeviceProperties();
        std::uint32_t const frames = numFramesFor(work, tileSize, frameCap, properties);
        if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Host>)
            return LaunchPolicy{frames, 1u, tileSize};
        else
            return LaunchPolicy{frames, tileSize, tileSize};
    }
} // namespace graphFlow
