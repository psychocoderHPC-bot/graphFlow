/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace graphFlow
{
    /** Runtime backend selection.
     *
     * The enum is always complete, independent of the compiled alpaka backends, so the CLI can
     * accept every name and report a configuration error for a backend that was not built. The
     * actual device selection is gated by ALPAKA_LANG_CUDA/ALPAKA_LANG_HIP, which alpaka derives
     * from the compiler in use.
     */
    enum class Backend
    {
        cpu,
        cuda,
        hip
    };

    inline std::string_view backendName(Backend backend)
    {
        switch(backend)
        {
        case Backend::cpu:
            return "cpu";
        case Backend::cuda:
            return "cuda";
        case Backend::hip:
            return "hip";
        }
        return "unknown";
    }

    /** Parse the CLI spelling of a backend; an unknown name yields no value. */
    inline std::optional<Backend> parseBackend(std::string_view name)
    {
        if(name == "cpu")
            return Backend::cpu;
        if(name == "cuda")
            return Backend::cuda;
        if(name == "hip")
            return Backend::hip;
        return std::nullopt;
    }

    class BackendError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    /** Raised when a backend name is known but that backend is absent from this build. */
    class BackendUnavailable : public BackendError
    {
    public:
        using BackendError::BackendError;
    };

    /** Whether the backend was compiled into this translation unit. */
    inline bool isBackendCompiled(Backend backend)
    {
        switch(backend)
        {
        case Backend::cpu:
            return true;
        case Backend::cuda:
#if ALPAKA_LANG_CUDA
            return true;
#else
            return false;
#endif
        case Backend::hip:
#if ALPAKA_LANG_HIP
            return true;
#else
            return false;
#endif
        }
        return false;
    }

    inline Backend requireBackend(std::string_view name)
    {
        auto const parsed = parseBackend(name);
        if(!parsed)
            throw BackendError{"graphFlow: unknown backend '" + std::string{name} + "' (expected cpu, cuda or hip)"};
        return *parsed;
    }

    [[noreturn]] inline void throwBackendUnavailable(Backend backend)
    {
        switch(backend)
        {
        case Backend::cuda:
            throw BackendUnavailable{"graphFlow: backend 'cuda' is not compiled into this build; reconfigure with "
                                     "-DGRAPHFLOW_ENABLE_CUDA=ON and a CUDA toolkit"};
        case Backend::hip:
            throw BackendUnavailable{"graphFlow: backend 'hip' is not compiled into this build; reconfigure with "
                                     "-DGRAPHFLOW_ENABLE_HIP=ON and a HIP toolkit"};
        case Backend::cpu:
            break;
        }
        throw BackendUnavailable{"graphFlow: backend is not available"};
    }

    /** Select one device of an explicit alpaka api/deviceKind pair and run `callback` on it.
     *
     * The selector is created from the exact tags in the pinned `alpaka/tag.hpp`:
     * `api::host`/`deviceKind::cpu`, `api::cuda`/`deviceKind::nvidiaGpu`, and
     * `api::hip`/`deviceKind::amdGpu`. An unavailable backend throws `BackendUnavailable`; an
     * out-of-range device index throws `std::out_of_range` instead of constructing a device.
     */
    template<typename TApi, typename TDeviceKind, typename TCallback>
    decltype(auto) withSelectedDevice(
        Backend backend,
        TApi api,
        TDeviceKind deviceKind,
        std::uint32_t deviceIndex,
        TCallback&& callback)
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{api, deviceKind});
        std::uint32_t const count = selector.getDeviceCount();
        std::string const label{backendName(backend)};
        if(!selector.isAvailable() || count == 0u)
            throw BackendUnavailable{"graphFlow: backend '" + label + "' is compiled but no device is available"};
        if(deviceIndex >= count)
            throw std::out_of_range{
                "graphFlow: device " + std::to_string(deviceIndex) + " is out of range for backend '" + label
                + "' (found " + std::to_string(count) + " device(s))"};
        auto device = selector.makeDevice(deviceIndex);
        std::string deviceName = selector.getDeviceProperties(deviceIndex).getName();
        return std::forward<TCallback>(callback)(std::move(device), backend, deviceName);
    }

    /** Dispatch on the runtime backend, selecting the matching api/deviceKind pair.
     *
     * A backend that was not compiled into this build is rejected before any alpaka device
     * selector is instantiated, so a CPU-only build never references an unavailable backend.
     */
    template<typename TCallback>
    decltype(auto) withBackendDevice(Backend backend, std::uint32_t deviceIndex, TCallback&& callback)
    {
        if(backend == Backend::cpu)
            return withSelectedDevice(
                backend,
                alpaka::api::host,
                alpaka::deviceKind::cpu,
                deviceIndex,
                std::forward<TCallback>(callback));
#if ALPAKA_LANG_CUDA
        if(backend == Backend::cuda)
            return withSelectedDevice(
                backend,
                alpaka::api::cuda,
                alpaka::deviceKind::nvidiaGpu,
                deviceIndex,
                std::forward<TCallback>(callback));
#endif
#if ALPAKA_LANG_HIP
        if(backend == Backend::hip)
            return withSelectedDevice(
                backend,
                alpaka::api::hip,
                alpaka::deviceKind::amdGpu,
                deviceIndex,
                std::forward<TCallback>(callback));
#endif
        throwBackendUnavailable(backend);
    }
} // namespace graphFlow
