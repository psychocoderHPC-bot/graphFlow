/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>

namespace graphFlow
{
    /** Result of the queue-independence probe.
     *
     * `yes`   - the second queue completed while the first queue was still in flight.
     * `no`    - the first queue completed before the second was observed complete, so the two
     *           queues behaved as if serialized; correctness still holds.
     * `unknown` - the probe could not decide (timeout, or both events completed between polls).
     */
    enum class QueueOverlap
    {
        yes,
        no,
        unknown
    };

    inline std::string_view toString(QueueOverlap overlap)
    {
        switch(overlap)
        {
        case QueueOverlap::yes:
            return "yes";
        case QueueOverlap::no:
            return "no";
        case QueueOverlap::unknown:
            return "unknown";
        }
        return "unknown";
    }

    struct QueueProbeResult
    {
        QueueOverlap overlap = QueueOverlap::unknown;
        bool queuesAlias = false;
        bool secondCompletedWhileFirstInFlight = false;
        std::uint64_t delayIterations = 0u;
        std::uint64_t pollDurationUs = 0u;
    };

    namespace detail
    {
        /** Bounded arithmetic work unit used as the "first" queue operation.
         *
         * The result is written so the compiler cannot remove the loop. The kernel is short
         * (milliseconds) but long enough for the host to observe the second queue completing.
         */
        struct QueueProbeDelayKernel
        {
            ALPAKA_FN_ACC void operator()(
                alpaka::onAcc::concepts::Acc auto const& acc,
                std::uint32_t* out,
                std::uint64_t iterations) const
            {
                auto [threadIdx] = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
                std::uint32_t value = static_cast<std::uint32_t>(threadIdx) + 1u;
                for(std::uint64_t i = 0u; i < iterations; ++i)
                    value = value * 1664525u + 1013904223u;
                out[threadIdx] = value;
            }
        };

        /** Host-visible short work unit used as the "second" queue operation. */
        struct QueueProbeNoopKernel
        {
            ALPAKA_FN_ACC void operator()(
                alpaka::onAcc::concepts::Acc auto const& acc,
                std::uint32_t* out) const
            {
                auto [threadIdx] = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
                out[threadIdx] = 0xA5A5A5A5u;
            }
        };

        /** One real thread per block on api::Host, a small block on accelerators. */
        template<typename TDevice>
        std::uint32_t queueProbeThreads(TDevice const& device)
        {
            if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Host>)
            {
                alpaka::unused(device);
                return 1u;
            }
            else
            {
                std::uint32_t const limit = device.getDeviceProperties().maxThreadsPerBlock;
                if(limit == 0u)
                    throw std::runtime_error{"graphFlow: queue probe cannot launch with maxThreadsPerBlock == 0"};
                return std::min<std::uint32_t>(limit, 32u);
            }
        }
    } // namespace detail

    /** Probe whether two non-blocking queues from `device.makeQueue()` actually run in parallel.
     *
     * Two queues are created and each gets one kernel followed by an event. The first kernel is a
     * bounded delay, the second is a no-op. The probe polls the events before waiting: if the
     * second event completes while the first is still incomplete, the queues overlap.
     *
     * `makeQueue()` may return the same underlying handle. The result is therefore advisory: no
     * graph or test correctness depends on `overlap == yes`.
     */
    template<typename TDevice>
    QueueProbeResult probeQueueIndependence(
        TDevice& device,
        std::uint64_t delayIterations = 20'000'000u,
        std::chrono::milliseconds timeout = std::chrono::milliseconds{10'000})
    {
        if(delayIterations == 0u)
            throw std::invalid_argument{"graphFlow: queue probe needs a nonzero delay"};

        QueueProbeResult result;
        result.delayIterations = delayIterations;

        std::uint32_t const threads = detail::queueProbeThreads(device);
        auto const extents = alpaka::Vec{static_cast<std::uint64_t>(threads)};
        auto delayOut = alpaka::onHost::alloc<std::uint32_t>(device, extents);
        auto noopOut = alpaka::onHost::alloc<std::uint32_t>(device, extents);

        auto firstQueue = device.makeQueue(alpaka::queueKind::nonBlocking);
        auto secondQueue = device.makeQueue(alpaka::queueKind::nonBlocking);
        result.queuesAlias = (firstQueue == secondQueue);

        auto firstEvent = device.makeEvent();
        auto secondEvent = device.makeEvent();

        firstQueue.enqueue(
            alpaka::onHost::ThreadSpec{std::uint32_t{1u}, threads},
            alpaka::KernelBundle{detail::QueueProbeDelayKernel{}, delayOut.data(), delayIterations});
        firstQueue.enqueue(firstEvent);

        secondQueue.enqueue(
            alpaka::onHost::ThreadSpec{std::uint32_t{1u}, threads},
            alpaka::KernelBundle{detail::QueueProbeNoopKernel{}, noopOut.data()});
        secondQueue.enqueue(secondEvent);

        auto const start = std::chrono::steady_clock::now();
        auto const deadline = start + timeout;
        bool firstDone = false;
        while(std::chrono::steady_clock::now() < deadline)
        {
            firstDone = firstEvent.isComplete();
            bool const secondDone = secondEvent.isComplete();
            if(secondDone && !firstDone)
            {
                result.overlap = QueueOverlap::yes;
                result.secondCompletedWhileFirstInFlight = true;
                break;
            }
            if(firstDone)
            {
                // The first queue finished before the second was seen complete. Observe the
                // second queue to classify a serialized (or merely starved) pair as `no`.
                result.overlap = QueueOverlap::no;
                break;
            }
            std::this_thread::yield();
        }

        // Drain both queues before the allocations go out of scope, whatever the verdict.
        alpaka::onHost::wait(firstQueue);
        alpaka::onHost::wait(secondQueue);

        auto const end = std::chrono::steady_clock::now();
        result.pollDurationUs
            = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        return result;
    }
} // namespace graphFlow
