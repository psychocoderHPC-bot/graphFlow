<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Known limitations

This page records the current limits of the integrated tree. It separates
missing work from platform limits. The full work list is
[docs/plan.md](plan.md) and [docs/missing.md](missing.md).

## Batch depth is unbounded

`--batch-chunks Q` accepts any non-zero 32-bit value. The runner caps the
dispatch at the ready-list length, but the transform's tile-descriptor, table,
and staging buffers are sized for the largest Q-chunk batch. A very large Q can
therefore allocate memory that a short ready list never uses. An optional upper
cap is not implemented.

## Frame launch is implemented

Graph kernels launch with `FrameSpec` from `graphFlow/core/LaunchPolicy.hpp`.
`numFrames` is
`min(ceil(work/T), maxBlocksPerGrid, frameCap*multiProcessorCount)`; `frameCap`
defaults to 8, and the host backend uses a unit frame extent. Blocks stride over
the batch tile list, so several tiles may share a block and no kernel assumes
one block per tile.

## HIP execution is validated on one AMD device

HIP execution runs on `hal` with an AMD Radeon RX 7900 XTX (`gfx1100`) and
ROCm 7.2. The current HIP CTest run passes **307/307** at
`561b9a4c9435e37b164623bb6be60a3051783eb4`; the previous 301/301 at `ffe4074`,
298/298 at `88de14f`, 297/297 at `56c11ac` and the 290/290 at `a610281` are
historical. The
original gate passed 245/245
with 43 HIP cases: 8 typed Flow cases, 24 particle CLI cases, 10 photon CLI
cases, and 1 photon output case, measured on
`b972eaa` and recorded in `/home/agent/final-hip-b972eaa/ctest.log`;
`rocprof` v3 confirms the device kernels. That 245/245 total is historical.
GitHub-hosted HIP remains compile-only because its runners have no AMD GPU, so
no HIP runtime claim is made for GitHub CI.

The HIP hash path, other AMD architectures, and an AMD `N=1e8` acceptance run
remain unrecorded. `hal` reports `sharing=unknown`, so no HIP performance claim
is made. The historical `gfx90a` compile-only gate is not an AMD runtime
result.

## SYCL is validated on one Intel GPU and uses fp32 photons

`GRAPHFLOW_ENABLE_SYCL=ON` adds the `Backend::sycl` dispatcher
(`api::oneApi`/`deviceKind::intelGpu`) and a runtime matrix with typed Flow
execution, failure paths, photon scattering, async/depth, FrameSpec, and
particle feedback/pressure cases. `ctest -L sycl` selects that matrix; the
build registers every device case under the `sycl` label, plus the SYCL CLI
and cross-backend exact-histogram cases: 71 cases total (33 in-process Catch2,
24 particle CLI, 3 cross-backend, 10 photon CLI, 1 photon output). On `hal`
with an Arc A770 (`spir64`, oneAPI 2026.2) the matrix passes 71/71 at current
main `9fb15f1e6ce13f252b31d4cb60cc410862d56511`, and the CPU-only build passes
255/255 on
`fwk394`. The SYCL build disables the OpenCL CPU device kind
(`alpaka_ONEAPI_Cpu=OFF`), so a missing Intel GPU fails the backend contract
instead of falling back to the host. The GitHub-hosted `sycl-compile` job
compiles the target but runs no CTest, so no SYCL runtime claim is made for
GitHub CI. No performance run is recorded and
`hal` reports `sharing=unknown`.

By default the Arc A770 does not advertise the SYCL `fp64` aspect
(`sycl::aspect::fp64` is false on both the Level Zero and OpenCL plugins), so a
double kernel is rejected at submission even though OpenCL reports
`cl_khr_fp64`. The oneAPI path therefore runs the photon physics in single
precision through `runPhotonScatteringAs<float>` and
`runPhotonScatteringAuto`; the CPU, CUDA and HIP paths keep double and the
public result is still `ScatteringResult`. Event counts and the terminal sum
stay exact, while float physics checks use single-precision margins.

Explicit FP64 emulation is opt-in. A fresh process with
`GRAPHFLOW_SYCL_FP64_EMULATION=1`, `IGC_EnableDPEmulation=1` and
`OverrideDefaultFP64Settings=1` exposes `sycl::aspect::fp64` and a non-empty
`double_fp_config`, and `runPhotonScatteringAuto` then runs oneAPI photon
physics in double. `OverrideDefaultFP64Settings=1` alone is not enough: the
runtime aborts on the first double buffer. A half-configured opt-in throws
instead of using fp32, so emulation is never selected silently. The CTest case
`test_photon_scattering_sycl_fp64` supplies all three variables and runs a
bounded case with a 300 s timeout.

DG2 sub-group reductions that combine `reqd_sub_group_size` with subgroup
collectives can hang or produce `DEVICE_LOST` (upstream issue #1036). The
photon kernels use block barriers and atomics only, no subgroup collectives, so
the FP64 case avoids that path; a hang still fails through the CTest timeout.
icpx 2026.2 also miscompiles the double photon functor at `-O1` and above with
FP64 emulation: a 32-byte functor kernel argument (`SlabT<double>` plus a
runtime scalar) arrives corrupted, the transform writes an invalid payload, and
the device can report `DEVICE_LOST`. The double oneAPI photon path is
therefore compiled in its own translation unit,
`PhotonScatteringFp64OneApi.cpp`, at `-O0`; the separate unit also keeps the
double kernels out of the fp32 device image, so a default run never requires
the fp64 aspect. Other translation units and the fp32 SYCL path stay
optimized.

The alpaka pin `eb677e8d7f70d79b35df25d9002f94ba669b6f44` cannot set
relocatable device code for SYCL: its `cmake/alpakaOneApi.cmake` omits the
`target` argument type in that branch. graphFlow keeps
`alpaka_RELOCATABLE_DEVICE_CODE=DEFAULT` and sanitizes the stray `INTERFACE`
argument from the `-fsycl-targets` call in `CMakeLists.txt`; the external
alpaka tree stays unmodified. No upstream `dev_v3.x` commit carries a fix, so
the pin is unchanged. AOT compilation is not used on the validated host:
`ocloc` is present on `hal` and a plain invocation does not crash, but it
rejects double (`-fsycl-targets=spir64_gen`), so the SYCL build stays
JIT-compiled SPIR-V; the first kernel run pays the SPIR-V JIT cost.

## CPU OMP execution

CPU execution with the OpenMP backend works on `rosi_v100` with passive
spinning (`GOMP_SPINCOUNT=0`). The current CPU suite passes 255/255; the
`rosi_v100` 202/202 OMP=ON Release total is historical. CPU OpenMP wall times
at `N=1e5`, `C=128`/`P=1024`, `Q=1`:
OMP=1 2.88 s, OMP=2 2.38 s (best), OMP=4 2.44 s, OMP=8 2.62 s; at `N=1e6`
OMP=2 Q1 23.71 s, Q8 4.46 s. All histogram sums are exact and no hangs were
observed. The previous multi-region pathology is not observed on `rosi_v100`
with passive spinning.

## TSAN is validated on rosi_v100 and runs in hosted CI; hal still cannot run it

The last full integrated TSAN CTest passed 210/210 with 0 reports in 202.83 s
on `rosi_v100` at `f3ef3552c60c0cfe0d202a73a08b808746ad11ce` (non-PIE Debug
 with `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`). That suite
 predates the current 255-case suite, so no full TSAN result exists for
 `561b9a4c9435e37b164623bb6be60a3051783eb4`. Since `gf/p10-sanitizers` at
`cf3bc8a72942b01c4a1ed32e4db7b260e22cb750` (merged as
`2fefe54c6d32e9156be5c08ea497119d0bacc21e`), hosted CI runs a dedicated
`tsan` job on `ubuntu-latest` with the same accepted configuration (non-PIE,
`GRAPHFLOW_ENABLE_OPENMP=OFF`, `alpaka_EXEC_CpuOmpBlocks=OFF`,
`alpaka_DEP_OMP=OFF`). The P10 fwk394 evidence is ASAN+UBSAN 236/236,
error-path leak 1/1, and TSAN 235/235 with 0 reports at `cf3bc8a`; the hosted
`tsan` job runtime was not observed, and no H100 was used. The remaining limitation is configuration
dependent: with the default CpuOmpBlocks plus OpenMP the same suite fails
105/210 with reports inside the alpaka async CPU queue. Those are upstream
alpaka races, not graphFlow races, so the clean gate uses the
OMP-exec-disabled configuration. The hal kernel still aborts ThreadSanitizer at
startup with `FATAL: ThreadSanitizer: unexpected memory mapping`, including for
a non-PIE binary, so hal is not a usable TSAN host; the fwk394 scheduler and
pool subset (14/14 cases) is historical context.

## CUDA 12.5 nvcc internal compiler error

CUDA 12.5 nvcc hits an internal compiler error while compiling
`HashScatterKernel.hpp`. CUDA 13.4 compiles the same tree and is the validated
CUDA compiler. Do not pin CUDA 12.5 for this project until the compiler failure
is resolved.

## CUDA V100 `sm_70` ICE is worked around on `rosi_v100`

nvcc 12.6.85 aborts in `check_name_hiding_by_template_parameters` when a
function template has ten or more abbreviated parameters. The transform,
partition, scatter, and hash-scatter kernels now declare the trailing
`IMdSpan` parameters explicitly, which keeps the abbreviated count below the
threshold; deduction and the instantiated signatures are unchanged. The test
hook adapters keep `GraphRunner::Queue` and `Batch` out of non-template
`std::function` construction, the other nvcc 12.6.85 frontend crash. With the
workaround the historical `rosi_v100` V100 `sm_70` CTest passed 254/254 and
the CPU CTest passed 202/202 in Release; the `f3ef355` runs (CPU 210/210, TSAN
210/210 with 0 reports, and the `N=1e8` CUDA acceptance run) use the same
workaround. The `Debug` build was not re-run after the seven hash cases, so the
measured historical `Debug` totals are the pre-hash 247/247 CUDA and 195/195
 CPU. The current tree is 255/255 CPU, 316/316 CUDA on `hal` A30 `sm_80`, and
 307/307 HIP on `hal` RX 7900 XTX. The RNG main is preserved. The hash-path
rerun remains open; H100 was not used. The historical full device suites at
`f3ef355` were CUDA 264/264 on `hal` A30 `sm_80` (CUDA 13.4, 400.98 s) and
HIP 255/255 on `hal` RX 7900 XTX `gfx1100` (ROCm 7.2, 99.44 s), both on the
shared `hal` host (correctness only).

## Throughput is a shared-host baseline for hal and fwk394

`hal` reports `sharing=unknown` and fwk394 is shared. The historical `N=1e8`
result of 187.5 s, about `5.33e5` particles/s, and the integrated T13
wall times are shared-host baselines, not exclusive-allocation acceptance
numbers. There is no variance study and no frozen regression target. The HIP
run on `hal` is a correctness result only; `hal` sharing is unknown and no HIP
performance number is recorded.

`rosi_v100` is the exclusive validation platform. P07b, merged as `56c11ac`,
re-ran the Q/C matrix on a V100-SXM2-32GB (CUDA 12.6.85, `sm_70`) at `N=1e8`,
`C=8192`, `P=16384`, seed 42, three repetitions: the Q8 median is 12,987,013
particles/s, with 260 rows summing to exactly 100,000,000, and the D2D copy
reference remains 409.733 GB/s. The frozen target is 13,000,000 particles/s at
Q8 with a 10 percent tolerance (floor 11,700,000) and the measured median
passes. P07b also records the kernel-axis end-to-end matrix for scatter,
dense/hash and the four distributions; PR #63 routes those axes through the
public `Flow` (`Tuning::scatterMode` and `Tuning::binPath`), while the recorded
matrix remains kernel-level. The
earlier P07 run at `64d4a1e` (Q8 7.690 s, 1.300e7
particles/s) and the `f3ef355` run (Q1 194.685 s, Q2 9.093 s, Q8 7.676 s) are
historical.

The current test totals are CPU 255/255, CUDA 316/316 on `hal` A30 `sm_80`, and
HIP 307/307 on `hal` RX 7900 XTX; PR #63 adds the public-Flow routed-axis cases
and PR #65 adds the six `[scheduler]` cases on top of the P07b 7/7. The
previous 249/249, 310/310 and 301/301 at `ffe4074`, 247/247, 307/307 and
298/298 at `88de14f`, the 246/246, 306/306 and 297/297 at `56c11ac`, and the
239/299/290 at `a610281` are historical. The `rosi_v100` CPU 210/210 and TSAN
210/210 runs at `f3ef355` and the CPU 202/202 / CUDA 254/254 totals at `2ad7d86` are
historical. The `Debug` build was not re-run after the seven hash cases
(measured historical `Debug` 195/195 CPU and 247/247 CUDA). H100 was not used
for this validation. The historical full CUDA and HIP device suites at
`f3ef355` were CUDA 264/264 on `hal` A30 `sm_80` (CUDA 13.4, 400.98 s) and
HIP 255/255 on `hal` RX 7900 XTX `gfx1100` (ROCm 7.2, 99.44 s), correctness
only.

H100 was not used for this validation; no H100 claim is made and no H100
baseline is recorded. H100 remains preliminary. The earlier JURECA
`jrc0880.jureca` (4x H100 PCIe 80GB, SM 9.0) run observed external load while
the client was stale, so it is preliminary CUDA evidence only.

## Pool memory is budget-capped and clamped to a physical chunk count

The pool size is `S = P + K*R`, where `P` is the source capacity in particles
and `K*R` is the output-credit reserve for the registered transform nodes.
The pool's payload arrays are bounded by `poolPayloadBudgetBytes`;
`Flow` uses the policy memory budget plus the exact registered-node reserve.

The programmatic default is 4 GiB (`defaultPoolPayloadBudgetBytes`). The pool
constructor applies a clamp: a request for more chunks than the budget can hold
is reduced to the largest fitting count, and `numChunks()` reports that physical
count. A request that fits the budget keeps its exact size, so small pools and
their free-list order are unchanged. An explicit `Config` or `Tuning` value for
`poolPayloadBudgetBytes` greater than 4 GiB is honored without any further cap;
the 4 GiB constant is only the default, not a ceiling. The former `C=8192`,
`P=65536` pair (about 12.9 GB of payload storage) is clamped to the largest
chunk count that fits the default budget rather than being allocated as-is.

The chunk generation and the per-chunk reader-lease count are 32-bit counters
that are checked on every increment: the storage rejects the increment that
would wrap either counter. `ChunkPool` enforces this in its own in-class
implementation. The shared `ChunkLedger` (used by `PayloadSlotStorage`) now
applies the same checked increment (PR #58, merged as
`88de14f691a51edda5ad48e153dbbe353bef7c9e`), so its generation and lease
counters raise on 32-bit overflow instead of wrapping after 2^32 increments per
chunk. A slot-based pool
remains the next memory optimization.

## In-flight depth is opt-in

`Tuning::inflightBatches` sets the maximum number of concurrently in-flight
batches per node. When unset the default depth is 1, which reproduces the
historical one-batch-per-node scheduling and uses no extra device memory.

When the depth is above 1 the scheduler submits several independent batches of
the same node before the first completes, overlapping host planning with device
work. Each submitted-depth lane gets its own typed stage and its own event;
`Flow::run` creates one stage instance per lane, so the per-lane device
staging, status and histogram buffers are replicated for every lane. The
replicated buffers are the memory cost of a depth above 1.

A node dispatches new work only while a free lane exists; when every lane is
occupied by an in-flight batch the scheduler stops submitting further batches
for that node until a completion event frees a lane. The accepted work is
therefore bounded by the per-node lane count, the batch size and the chunk pool,
not by an unbounded queue. `Tuning::inflightBatches` rejects a zero value with
`std::invalid_argument`; `GraphRunner` rejects a zero depth for the same reason.

## Queue overlap is advisory and the CUDA probe is timing-sensitive

The missing.md "exploit or document queue overlap" item is closed on the
basis of this evidence (P04, PR #45 at
`e86adb73e41a5f70f4c9839c58895f7a789691ac`); this section stays as the
reference for the probe's timing sensitivity.

`probeQueueIndependence()` creates two non-blocking queues, runs a delay kernel
on the first and a no-op on the second, and polls both events. The scheduler
never depends on the verdict, and correctness does not depend on overlap.

`tools/queue_probe.sh` adds a timed two-queue mode and a same-queue baseline.
Measured at main `dd9ce3c` with 20,000,000 delay iterations and 5 repeats on
the shared `hal` host (A30, CUDA 13.4) and `fwk394` (CPU, g++ 13.3):

| Backend | two queues | one queue | committed probe |
| --- | --- | --- | --- |
| CPU (fwk394) | overlap 5/5 | 0/5 | yes 5/5 |
| CUDA (hal A30) | overlap 4/5 | 0/5 | yes 4/5, no 1/5 |
| HIP (hal RX 7900 XTX) | overlap 5/5 | 0/5 | yes 5/5 |

Reproduce with `tools/queue_probe.sh --help`; the probe takes the backend,
`--alpaka-source`, and the backend architecture (`sm_80` on the A30, `gfx1100`
on the RX 7900 XTX). The timed CUDA run observes the no-op event at a median
10.0 us while the delay kernel completes at 28.6 ms, so the streams do overlap.
The same-queue baseline serializes as expected. The committed probe reported `no` in one of five CUDA
repetitions at 20,000,000 iterations and `yes` in every repetition at
100,000,000 and 400,000,000 iterations. Its `no` verdict is a polling-resolution
artifact, not stream serialization: the loop polls with
`std::this_thread::yield()` and classifies "first event complete" as `no`, which
also happens when both events complete between two polls. A busy shared host can
delay the polling thread past the delay kernel. CPU and HIP use the same probe
and submission pattern, so the CUDA `no` is not caused by the kernels, the
shared A30, or the scheduler. Do not use a single committed-probe `no` as
evidence that CUDA streams are serialized.

At the default depth of 1 the scheduler keeps one in-flight batch per node and
relies on full-chunk pipelining, not cross-queue overlap, for progress. With
`Tuning::inflightBatches` above 1, several independent batches of the same
node overlap and the per-lane events bound the in-flight work. A device whose
queues alias or run sequentially follows the same path.

## Host planning stays; batching amortizes the round trip

This section is the closing evidence for the missing.md "host round-trip
profiling and device-side planning experiment" item (P04, PR #45 at
`e86adb73e41a5f70f4c9839c58895f7a789691ac`).

An `nsys` profile of `graphFlow_particle_feedback` on the shared `hal` A30
(N=1e6, C=1024, P=8192, seed 42, max-visits 64, Release) separates the
per-batch stages from `CUPTI_ACTIVITY_KIND_KERNEL/MEMCPY/MEMSET`. "Host
residual" is classify end to scatter start minus the traced D2H, H2D, and
memset time. It covers event polling, `planDenseBins`/`buildDenseTable`, queue
submission, and API latency.

| Q | partition batches | classify med (total) | scatter med (total) | D2H med | H2D med | memset med | host residual med | round trip med (total) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 171,347 | 2.40 us (431 ms) | 3.26 us (566 ms) | 2.98 us | 3.07 us | 2.72 us | 53.76 us | 62.66 us (10.88 s) |
| 8 | 1,724 | 4.54 us (7.9 ms) | 5.47 us (9.6 ms) | 3.94 us | 4.16 us | 4.19 us | 52.37 us | 66.14 us (116 ms) |

The totals are sums over the three partition streams, not wall time. The
per-batch round trip is dominated by a fixed roughly 52 us host residual. The
plan itself is small: `planDenseBins` plus `buildDenseTable` takes 0.087 us for
the example's 2 and 3 bins and 2.99 us for 1024 bins (200,000 iterations, host
microbenchmark). Q=8 reduces the number of handoffs by about 99x and the total
traced round-trip time from 10.88 s to 116 ms. On this shared host the measured
wall time was 8.08-8.12 s at Q=1 and 0.39-0.41 s at Q=8; both include the 44 ms
queue probe and are shared-host baselines, not acceptance numbers.

Decision: keep host-side planning. Batching already removes most handoffs, and
the remaining cost is event and dispatch latency that device-side planning does
not remove, because the host still reserves output chunks and builds the
scatter table. Revisit device-side planning only with a design that also makes
the output chunk pool device-resident, or with a many-bin workload where the
O(B) plan itself is measurable.

## Generic graph API is host-only

The tree has the `NodeBase`/`Graph`/`GraphBuilder`/`RouteTable` core with
source/transform/sink roles, per-bin routing, and feedback edges as ordinary
routes. Node objects are caller-owned and non-owning, the graph is immutable
after `build()`, and v1 allows one source and no routes into that source.
Runtime graph mutation, join, and broadcast remain open. The particle-feedback
example uses the typed `Flow` API; the lower-level `GraphBuilder` core is not
used by the example.

## CI is hosted and GPU builds are compile-only

CI runs on GitHub-hosted `ubuntu-latest` runners and public containers
(`.github/workflows/ci.yml`, see [ci.md](ci.md)). The `cpu` jobs run the 255
CPU CTest cases in Release and Debug, `sanitizers` runs ASAN+UBSAN plus a
dedicated `leak` step over the error paths, and a separate `tsan` job runs the
host scheduler under TSAN (non-PIE, OMP off, CpuOmpBlocks off). The CUDA,
HIP, and SYCL jobs compile only, because the hosted runners have no NVIDIA,
AMD, or Intel GPU; the `sycl-compile` job builds the Intel-GPU-only SYCL target
in the oneAPI 2025.3 container and runs no CTest. A CUDA, HIP, or SYCL runtime
gate needs a device host. TSAN was validated on the
exclusive `rosi_v100` host (210/210, 0 reports, 202.83 s at `f3ef355`,
 non-PIE with `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`); that
   run is historical and no full TSAN run exists for
  `561b9a4c9435e37b164623bb6be60a3051783eb4`. The P10 fwk394 run
at `cf3bc8a` is ASAN+UBSAN 236/236, leak 1/1, TSAN 235/235 with 0 reports, and
the hosted `tsan` and `leak` runtimes were not observed. No self-hosted
runner, project GPU label, or GitLab pipeline is configured, and no H100 was
used. The local commands in `docs/build.md` remain the documented entry
points.

## Accelerator test coverage is partial

Most Catch2 cases construct the CPU device. The CUDA build adds
`tests/unit/test_cuda_kernel.cpp`, which runs a kernel on the selected NVIDIA
GPU and fails when no GPU is visible. The HIP build adds 8 typed Flow cases
that fail when no AMD GPU is visible. The SYCL build (PR #69) adds the 71-case
device matrix on `hal` Arc A770, whose cases likewise fail instead of skipping
when no Intel GPU is visible. The automated cross-backend histogram
equality cases are `example.particle_feedback.cross_backend.Q{1,2,8}`
(`N=1e6`, `C=128`, `P=1024`, seed 42), added on `gf/cross-backend-equality`:
each compiled backend runs the same seed/N/C/P/Q and the cases require exact
sorted `weight,count` row equality against the first backend plus a count sum
of `N`, so a compiled backend without a usable device fails the case instead
of skipping. The historical CPU total was 210/210 at `751c8a8` with all three Q
cases exact. The device evidence is two build pairs, not one three-backend run:
CUDA and HIP are mutually exclusive in a single build, and the CPU backend
stays enabled in accelerator builds, so the CPU+CUDA pair and the CPU+HIP pair
each ran the three Q cases 3/3 on `hal` (RX 7900 XTX `gfx1100`, ROCm 7.2) with
 the row comparisons exact against the CPU reference. The current full device
  suites are CUDA 316/316 on `hal` A30 `sm_80` (CUDA 13.4) and HIP 307/307 on
  `hal` RX 7900 XTX `gfx1100` (ROCm 7.2), correctness only; the `f3ef355` totals
 264/264 and 255/255 are historical.

## Failure injection is test-only

`GraphRunner::setFailureHook` exists so tests can raise a host exception at a
named failure point (lease acquisition or completion-event enqueue). Only the
tests install the hook; the production `Flow` and runner paths leave it empty.
The hook models control flow around a failed enqueue, not a device or driver
fault. It cannot inject real mid-run CUDA/HIP out-of-memory, ECC, Xid, or
context-loss errors, and the tests do not do so. Pool exhaustion exercises the
pool allocator's own capacity; a full hash table and invalid classifier or
updater results are detected from real device work. Real-driver fault
injection remains an open gap (see
[missing.md](missing.md), asynchronous failure propagation), so the failure
suite does not claim full coverage of asynchronous device errors. The five
 device failure cases were executed at `10c1733770e16ebaecc74513034e6d327924503c`
 on `hal`: CUDA 277/277 on the
A30 `sm_80` and HIP 268/268 on the RX 7900 XTX `gfx1100`, both correctness only
because `hal` is shared.

## Hash path evidence is kernel-level; public routing added

The dense path is the correctness reference. The shared-memory hash path is
exercised by CPU integration tests and by P07b at kernel level on `rosi_v100`
CUDA through the opt-in `--workload kernel` path. PR #63 routes the same axes
through the public `Flow`: `Tuning::scatterMode` and `Tuning::binPath` select
the dense/hash and shared/direct paths, `Config::validate()` rejects an
unsupported axis combination before any device work, and the new CPU and device
correctness cases cover the routed axes including the rejection cases. No
public-Flow performance matrix and no HIP `Direct` execution are recorded, so
the measurement-matrix item stays open for those numbers. The dense limit stays at 1024 bins from the
P03 fwk394 CPU measurement; no end-to-end device crossover is recorded.

## Result storage is explicit v1

The example stores one `uint32_t` final weight per particle, so the result
buffer is `O(N)` and `4 B/weight` (about 0.4 GB at `N=1e8`). On the device the
run keeps `weights` plus `cappedFlags`, both `uint32_t`, for `8 B/particle`
(about 0.8 GB at `N=1e8`, 8 GB at `N=1e9`). The host transient is about
`24 B/particle` (the two host copies, the result vector, and the sort input),
about 2.4 GB at `N=1e8` and 24 GB at `N=1e9`. The histogram is produced with an
`O(N log N)` sort. v1 keeps this explicit buffer; a device histogram is added
only when `8N` bytes plus the payload pool exceeds the device budget.

## Other open items

- The example binaries are registered in CTest. The particle example runs 24
  CLI cases per backend (`N=0,1,17,127,128,4097` crossed with `C=1,31,127,128`)
  plus one `example.public_api` source-contract case. The photon example runs
  its CLI and output cases per backend. A CPU build registers 25 particle
  example tests; a CUDA build registers 49 (24 CPU, 24 CUDA, 1 contract); a HIP
  build adds 24 HIP particle CLI, 10 HIP photon CLI, 1 HIP photon output, and 8
   typed Flow cases. The current CTest totals are 255/255 CPU, 316/316 CUDA on
   `hal` A30 `sm_80`, and 307/307 HIP on `hal` RX 7900 XTX; the previous
   249/249, 310/310 and 301/301 at `ffe4074`, 247/247, 307/307 and 298/298 at
   `88de14f`, the 246/246, 306/306 and 297/297 at `56c11ac`, and the
   239/299/290 at `a610281` are historical. The other historical integrated totals are 194/194 CPU on
   fwk394, 246/246 CUDA on hal, 202/202 CPU
  and 254/254 CUDA on `rosi_v100` V100 `sm_70`, and 245/245 HIP at `b972eaa`.
  The `Debug` build was not re-run after the seven hash cases (measured
  historical `Debug` 195/195 CPU and 247/247 CUDA).
- The historical CPU/CUDA `N=1e6` histogram equality check was a manual diff,
  not an automated test, and was not repeated on the integrated revision; the
  automated cross-backend equality cases above (Q1/Q2/Q8, `N=1e6`) replace the
  manual evidence. The historical full device suites at `f3ef355` were
  CUDA 264/264 on `hal` A30 `sm_80` and HIP 255/255 on `hal` `gfx1100`.
- Release builds do not detect corrupted photon payload state per event. Host
  setup validates the slab and configuration once. Configure
  `-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON` to compile the per-event checks in
  `graphFlow/flow/Debug.hpp` and `PhotonDebug.hpp`; the option defaults to `ON`
  for `Debug` and `OFF` for `Release`/`RelWithDebInfo`.
- The P07b exclusive measurement is recorded in `docs/benchmarks.md`; its frozen
  target is 13,000,000 particles/s at `N=1e8`, Q=8, with a 10 percent
  tolerance, and the measured P07b median 12,987,013 passes. The P07 run and
  the older T13 wall times remain historical/shared-host baselines.
- The result buffer is `O(N)`, `4 B/weight`, with the explicit v1 policy and
  the device/host costs above.
- The `sm_75` RTX 2080 Ti path is a documented build target, not a measured run.
- The hash load factor (`2*T` entries by default) has not been measured.
