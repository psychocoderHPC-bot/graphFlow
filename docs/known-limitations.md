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
ROCm 7.2. The HIP CTest run passes 238/238 with 43 HIP cases: 8 typed Flow
cases, 24 particle CLI cases, 10 photon CLI cases, and 1 photon output case.
`rocprof` v3 confirms the device kernels. GitHub-hosted HIP remains
compile-only because its runners have no AMD GPU, so no HIP runtime claim is
made for GitHub CI.

The HIP hash path, other AMD architectures, and an AMD `N=1e8` acceptance run
remain unrecorded. `hal` reports `sharing=unknown`, so no HIP performance claim
is made. The historical `gfx90a` compile-only gate is not an AMD runtime
result.

## CPU OMP execution

CPU execution with the OpenMP backend works on `rosi_v100` with passive
spinning (`GOMP_SPINCOUNT=0`). The `rosi_v100` CPU CTest passes 195/195 with
OMP=ON Release. CPU OpenMP wall times at `N=1e5`, `C=128`/`P=1024`, `Q=1`:
OMP=1 2.88 s, OMP=2 2.38 s (best), OMP=4 2.44 s, OMP=8 2.62 s; at `N=1e6`
OMP=2 Q1 23.71 s, Q8 4.46 s. All histogram sums are exact and no hangs were
observed. The previous multi-region pathology is not observed on `rosi_v100`
with passive spinning.

## TSAN is blocked on hal

The hal kernel aborts ThreadSanitizer at startup with
`FATAL: ThreadSanitizer: unexpected memory mapping`, including for a non-PIE
binary. The scheduler and pool subset is clean on fwk394 (14/14 cases). Use a
host whose runtime mapping passes the TSAN startup check before treating TSAN as
a gate. An integrated ASAN/UBSAN count is not recorded.

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
workaround the `rosi_v100` V100 `sm_70` CTest passes 247/247 and the CPU CTest
passes 195/195 in Release and Debug; the RNG main is preserved. The hash-path
rerun remains open, and CUDA 13.4/`sm_80`, fwk394, and H100 were not re-run in
this validation.

## Throughput is a shared-host baseline

`hal` reports `sharing=unknown` and fwk394 is shared. The historical `N=1e8`
result of 187.5 s, about `5.33e5` particles/s, and the integrated T13
wall times are shared-host baselines, not exclusive-allocation acceptance
numbers. There is no variance study and no frozen regression target. The HIP
run on `hal` is a correctness result only; `hal` sharing is unknown and no HIP
performance number is recorded.

`rosi_v100` is the exclusive validation platform. Current `rosi_v100`
results: CPU CTest 195/195 and CUDA CTest 247/247 on V100 `sm_70` (OMP=ON and
CUDA 12.6.85, Release and Debug, both with the explicit concept-parameter
workaround); CPU OpenMP wall times recorded (passive, `GOMP_SPINCOUNT=0`).
No frozen regression target exists yet. CUDA 13.4/`sm_80`, fwk394, and H100
were not re-run in this validation.

JURECA `jrc0880.jureca` (4x H100 PCIe 80GB, SM 9.0) is available, but its
gateway sharing flag is unknown. The recorded H100 run observed external load
while the client was stale, so it is preliminary CUDA evidence only: no final
H100 claim is made, and no H100 baseline is recorded. H100 remains
blocked/preliminary.

## Pool memory is budget-capped but still fixed-capacity

The pool is the source capacity `P` plus one output bound per registered node,
and its payload arrays are capped by `poolPayloadBudgetBytes`. `Flow` uses the
policy memory budget plus the exact registered-node reserve; the programmatic
default is 4 GiB. The former `C=8192`, `P=65536` pair is rejected because it
exceeds the budget. The fixed allocation still reserves one chunk per live
particle plus the output credits, so a slot-based pool remains the next memory
optimization.

## One in-flight batch per node

Each node has one in-flight batch. A transform batch can carry up to Q chunks,
but independent batches of the same node do not overlap. A bounded submitted
depth per node is not implemented.

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
(`.github/workflows/ci.yml`, see [ci.md](ci.md)). The `cpu` jobs run the 202
CPU CTest cases in Release and Debug, and `sanitizers` runs ASAN+UBSAN. The
CUDA and HIP jobs compile only, because the hosted runners have no NVIDIA or
AMD GPU; a CUDA or HIP runtime gate needs a device host. TSAN stays a
local/manual check because the hosted runner has no known-good mapping
configuration. No self-hosted runner, project GPU label, or GitLab pipeline is
configured. The local commands in `docs/build.md` remain the documented entry
points.

## Accelerator test coverage is partial

Most Catch2 cases construct the CPU device. The CUDA build adds
`tests/unit/test_cuda_kernel.cpp`, which runs a kernel on the selected NVIDIA
GPU and fails when no GPU is visible. The HIP build adds 8 typed Flow cases
that fail when no AMD GPU is visible. No automated CPU/CUDA/HIP histogram
equality test exists; the equality evidence is manual.

## Hash path evidence is CPU-only (low-level)

The dense path is the correctness reference and the only path the example uses.
The shared-memory hash path is exercised by CPU integration tests (low-level
only). It compiles for CUDA and HIP, but no hash-path accelerator execution is
recorded. The `rosi_v100` hash-path rerun is pending. The dense/hash threshold
is the static 1024-bin limit, not a measured crossover.

## Other open items

- The example binaries are registered in CTest. The particle example runs 24
  CLI cases per backend (`N=0,1,17,127,128,4097` crossed with `C=1,31,127,128`)
  plus one `example.public_api` source-contract case. The photon example runs
  its CLI and output cases per backend. A CPU build registers 25 particle
  example tests; a CUDA build registers 49 (24 CPU, 24 CUDA, 1 contract); a HIP
  build adds 24 HIP particle CLI, 10 HIP photon CLI, 1 HIP photon output, and 8
  typed Flow cases. The historical integrated CTest totals are 194/194 CPU on
  fwk394 and 246/246 CUDA on hal. The `rosi_v100` CPU CTest passes 195/195
  (OMP=ON Release and Debug) and the CUDA CTest on V100 `sm_70` passes 247/247
  in Release and Debug with the explicit concept-parameter workaround. The HIP
  CTest on `hal` (RX 7900 XTX `gfx1100`, ROCm 7.2) passes 238/238 with 43 HIP
  cases.
- The historical CPU/CUDA `N=1e6` histogram equality check was a manual diff,
  not an automated test, and was not repeated on the integrated revision.
- Release builds do not detect corrupted photon payload state per event. Host
  setup validates the slab and configuration once. Configure
  `-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON` to compile the per-event checks in
  `graphFlow/flow/Debug.hpp` and `PhotonDebug.hpp`; the option defaults to `ON`
  for `Debug` and `OFF` for `Release`/`RelWithDebInfo`.
- The integrated T13 benchmark is recorded in `docs/benchmarks.md`; its wall
  times remain shared-host baselines, and no frozen regression target exists
  yet.
- The result buffer is `O(N)` (about 400 MB at `N=1e8`).
- The `sm_75` RTX 2080 Ti path is a documented build target, not a measured run.
- The hash load factor (`2*T` entries by default) has not been measured.
