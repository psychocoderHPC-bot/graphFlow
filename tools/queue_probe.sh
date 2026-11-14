#!/usr/bin/env bash
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Reproducible queue-overlap probe for the committed
# include/graphFlow/detail/QueueProbe.hpp probe.
#
# The script compiles one header-only probe program against the graphFlow and
# alpaka source trees. It measures two configurations:
#
#   two-queue  - two non-blocking queues, the configuration the scheduler uses
#                when a device hands out independent queues.
#   same-queue - both kernels on one queue, the serialized baseline.
#
# It also runs the committed probe `probeQueueIndependence()` several times so a
# timing-sensitive verdict is visible instead of a single sample.
#
# The probe records event-completion timestamps with a busy poll. That separates
# a real stream serialization (second event completes at the same time as the
# first) from a coarse-poll race (second event completes much earlier but an
# earlier poll missed it). No performance number is produced.

set -euo pipefail

readonly PROG="queue_probe"

die()
{
    printf '%s: error: %s\n' "$PROG" "$*" >&2
    exit 2
}

usage()
{
    cat <<'EOF'
usage: tools/queue_probe.sh --backend cpu|cuda|hip --alpaka-source DIR [options]

required:
  --backend NAME            cpu, cuda, or hip
  --alpaka-source DIR       pinned alpaka source tree (or GRAPHFLOW_ALPAKA_SOURCE_DIR)

options:
  --graphflow-source DIR    graphFlow tree (default: parent of this script)
  --iterations N            delay-kernel loop iterations (default: 20000000)
  --repeats N               repetitions per mode and for the committed probe (default: 5)
  --arch ARCH               CUDA -arch or HIP --offload-arch (default: native)
  --compiler PATH           compiler for the backend (default: $CXX/$NVCC/$HIPCC)
  --workdir DIR             build directory (default: mktemp -d)
  --keep                    keep the generated source and binary
  --help                    show this text

environment:
  ROCM_PATH                 ROCm prefix for the hip backend (default: /opt/rocm)

The script prints the relevant environment variables and the exact compiler
command. It never unsets CUDA_LAUNCH_BLOCKING or CUDA_DEVICE_MAX_CONNECTIONS,
so those knobs stay available for a forced-serialization run.
EOF
}

backend=""
alpaka_source="${GRAPHFLOW_ALPAKA_SOURCE_DIR:-}"
graphflow_source="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iterations=20000000
repeats=5
arch="native"
compiler=""
workdir=""
keep=0

while (($# > 0)); do
    case "$1" in
        --backend) backend="${2-}"; shift 2 ;;
        --alpaka-source) alpaka_source="${2-}"; shift 2 ;;
        --graphflow-source) graphflow_source="${2-}"; shift 2 ;;
        --iterations) iterations="${2-}"; shift 2 ;;
        --repeats) repeats="${2-}"; shift 2 ;;
        --arch) arch="${2-}"; shift 2 ;;
        --compiler) compiler="${2-}"; shift 2 ;;
        --workdir) workdir="${2-}"; shift 2 ;;
        --keep) keep=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument '$1' (try --help)" ;;
    esac
done

case "$backend" in
    cpu|cuda|hip) ;;
    "") die "--backend is required (cpu, cuda, or hip)" ;;
    *) die "--backend must be cpu, cuda, or hip, got '$backend'" ;;
esac
[[ -n "$alpaka_source" ]] || die "--alpaka-source is required"
[[ -d "$alpaka_source/include/alpaka" ]] \
    || die "'$alpaka_source' does not contain include/alpaka"
[[ -f "$graphflow_source/include/graphFlow/graphFlow.hpp" ]] \
    || die "'$graphflow_source' is not a graphFlow source tree"
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || die "--iterations must be a positive integer"
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"

if [[ -z "$workdir" ]]; then
    workdir="$(mktemp -d "${TMPDIR:-/tmp}/graphFlow-queue-probe-XXXXXX")"
else
    mkdir -p "$workdir"
fi
src="$workdir/queue_probe.cpp"
bin="$workdir/queue_probe"

printf '# backend=%s iterations=%s repeats=%s arch=%s graphflow=%s alpaka=%s\n' \
    "$backend" "$iterations" "$repeats" "$arch" "$graphflow_source" "$alpaka_source"
printf '# env CUDA_LAUNCH_BLOCKING=%s CUDA_DEVICE_MAX_CONNECTIONS=%s OMP_NUM_THREADS=%s ROCR_VISIBLE_DEVICES=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "${CUDA_LAUNCH_BLOCKING-<unset>}" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}" \
    "${OMP_NUM_THREADS-<unset>}" "${ROCR_VISIBLE_DEVICES-<unset>}" "${CUDA_VISIBLE_DEVICES-<unset>}"

# Match the graphFlow CMake build: one host executor, no TBB dependency. The
# TBB symbols are otherwise pulled in by alpaka's host executor list on a
# clang-based HIP compile.
include_flags=(-I"$graphflow_source/include" -isystem "$alpaka_source/include"
               -DALPAKA_DISABLE_EXEC_CpuTbbBlocks -DALPAKA_DISABLE_TBB)

case "$backend" in
    cpu)
        cxx="${compiler:-${CXX:-g++}}"
        compile=("$cxx" -x c++ -std=c++20 -O2 -fopenmp "${include_flags[@]}" "$src" -o "$bin"
                 -lhwloc -lpthread)
        ;;
    cuda)
        nvcc="${compiler:-${NVCC:-nvcc}}"
        compile=("$nvcc" -x cu -std=c++20 -O2 "-arch=$arch" --expt-relaxed-constexpr
                 --extended-lambda -Xcompiler -fopenmp "${include_flags[@]}" "$src" -o "$bin"
                 -lhwloc -lcudadevrt -lcudart_static -lrt -lpthread -ldl)
        ;;
    hip)
        rocm_path="${ROCM_PATH:-/opt/rocm}"
        hipcc="${compiler:-${HIPCC:-$rocm_path/bin/amdclang++}}"
        compile=("$hipcc" -x hip -std=c++20 -O2 "--offload-arch=$arch" "--rocm-path=$rocm_path"
                 -D__HIP_ROCclr__=1 "${include_flags[@]}" "$src" -o "$bin"
                 -lhwloc -L"$rocm_path/lib" -lamdhip64)
        ;;
esac

cat >"$src" <<'CPP'
// Generated by tools/queue_probe.sh. Not part of the graphFlow library.
#include "graphFlow/core/Backend.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/detail/QueueProbe.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace graphFlow;
using Clock = std::chrono::steady_clock;

namespace
{
    /** Bounded arithmetic delay; the result is written so it cannot be removed. */
    struct DelayKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t* out,
            std::uint64_t iterations,
            std::uint32_t count) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{count}))
            {
                std::uint32_t value = static_cast<std::uint32_t>(idx) + 1u;
                for(std::uint64_t i = 0u; i < iterations; ++i)
                    value = value * 1'664'525u + 1'013'904'223u;
                out[idx] = value;
            }
        }
    };

    /** Host-visible no-op used as the second queue operation. */
    struct NoopKernel
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t* out,
            std::uint32_t count) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{count}))
                out[idx] = 0xA5A5'A5A5u;
        }
    };

    template<typename TDevice>
    std::uint32_t probeThreads(TDevice const& device)
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
                throw std::runtime_error{"queue_probe: maxThreadsPerBlock == 0"};
            return std::min<std::uint32_t>(limit, 32u);
        }
    }

    struct Sample
    {
        bool queuesAlias = false;
        bool firstDone = false;
        bool secondDone = false;
        bool secondBeforeFirst = false;
        double firstUs = -1.0;
        double secondUs = -1.0;
    };

    template<typename TDevice>
    Sample runTimeline(TDevice& device, std::uint64_t iterations, std::uint32_t threads, bool sameQueue)
    {
        Sample sample;
        LaunchPolicy const policy = launchPolicy(device, threads, threads, defaultFrameCap);
        auto const extents = alpaka::Vec{static_cast<std::uint64_t>(threads)};
        auto firstOut = alpaka::onHost::alloc<std::uint32_t>(device, extents);
        auto secondOut = alpaka::onHost::alloc<std::uint32_t>(device, extents);
        auto const firstQueue = device.makeQueue(alpaka::queueKind::nonBlocking);
        auto const secondQueue = sameQueue ? firstQueue : device.makeQueue(alpaka::queueKind::nonBlocking);
        sample.queuesAlias = (firstQueue == secondQueue);
        auto firstEvent = device.makeEvent();
        auto secondEvent = device.makeEvent();

        firstQueue.enqueue(
            policy.frameSpec(),
            alpaka::KernelBundle{DelayKernel{}, firstOut.data(), iterations, threads});
        firstQueue.enqueue(firstEvent);
        auto const start = Clock::now();
        secondQueue.enqueue(
            policy.frameSpec(),
            alpaka::KernelBundle{NoopKernel{}, secondOut.data(), threads});
        secondQueue.enqueue(secondEvent);

        // Busy poll: a yield can hide a short event behind a long kernel and turn
        // real overlap into a false `no`.
        auto const deadline = start + std::chrono::seconds(60);
        while(Clock::now() < deadline)
        {
            if(!sample.secondDone && secondEvent.isComplete())
            {
                sample.secondDone = true;
                sample.secondUs = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            }
            if(!sample.firstDone && firstEvent.isComplete())
            {
                sample.firstDone = true;
                sample.firstUs = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            }
            if(sample.firstDone && sample.secondDone)
                break;
        }
        sample.secondBeforeFirst = sample.secondDone && sample.firstDone && sample.secondUs <= sample.firstUs;
        alpaka::onHost::wait(firstQueue);
        alpaka::onHost::wait(secondQueue);
        return sample;
    }

    template<typename TDevice>
    void measure(TDevice& device, std::string const& label, std::uint64_t iterations, int repeats)
    {
        std::uint32_t const threads = probeThreads(device);
        std::printf(
            "# backend=%s device=%s threads=%u iterations=%llu repeats=%d\n",
            label.c_str(),
            device.getDeviceProperties().getName().c_str(),
            threads,
            static_cast<unsigned long long>(iterations),
            repeats);

        std::vector<Sample> two;
        std::vector<Sample> same;
        for(int r = 0; r < repeats; ++r)
        {
            two.push_back(runTimeline(device, iterations, threads, false));
            same.push_back(runTimeline(device, iterations, threads, true));
        }
        auto report = [&](char const* mode, std::vector<Sample> const& samples)
        {
            int overlap = 0;
            for(auto const& s : samples)
                overlap += s.secondBeforeFirst ? 1 : 0;
            std::vector<double> first;
            std::vector<double> second;
            for(auto const& s : samples)
            {
                first.push_back(s.firstUs);
                second.push_back(s.secondUs);
            }
            std::sort(first.begin(), first.end());
            std::sort(second.begin(), second.end());
            bool const alias = samples.front().queuesAlias;
            std::printf(
                "# mode=%s queues-alias=%s overlap=%d/%d median-first-us=%.1f median-second-us=%.1f\n",
                mode,
                alias ? "yes" : "no",
                overlap,
                repeats,
                first[first.size() / 2u],
                second[second.size() / 2u]);
        };
        report("two-queue", two);
        report("same-queue", same);

        int yes = 0;
        int no = 0;
        int unknown = 0;
        std::vector<std::uint64_t> poll;
        for(int r = 0; r < repeats; ++r)
        {
            auto const result = probeQueueIndependence(device, iterations, std::chrono::seconds{60});
            if(result.overlap == QueueOverlap::yes)
                ++yes;
            else if(result.overlap == QueueOverlap::no)
                ++no;
            else
                ++unknown;
            poll.push_back(result.pollDurationUs);
        }
        std::sort(poll.begin(), poll.end());
        std::printf(
            "# committed-probe yes=%d no=%d unknown=%d median-poll-us=%llu\n",
            yes,
            no,
            unknown,
            static_cast<unsigned long long>(poll[poll.size() / 2u]));
    }
} // namespace

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::fprintf(stderr, "usage: %s BACKEND ITERATIONS REPEATS\n", argv[0]);
        return 2;
    }
    std::string const backend = argv[1];
    std::uint64_t const iterations = std::strtoull(argv[2], nullptr, 10);
    int const repeats = std::atoi(argv[3]);
    try
    {
        Backend const selected = requireBackend(backend);
        return withBackendDevice(
            selected,
            0u,
            [&](auto device, Backend, std::string const&)
            {
                measure(device, backend, iterations, repeats);
                return 0;
            });
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "queue_probe: %s\n", error.what());
        return 2;
    }
}
CPP

printf '# compile'
printf ' %q' "${compile[@]}"
printf '\n'
"${compile[@]}"

printf '# run %s %s %s\n' "$bin" "$backend" "$iterations"
"$bin" "$backend" "$iterations" "$repeats"

if ((keep == 0)); then
    rm -rf "$workdir"
else
    printf '# workdir=%s\n' "$workdir"
fi
