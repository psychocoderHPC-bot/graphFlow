<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Known limitations

This page records the current limits of the integrated tree. It separates
missing work from platform limits. The full work list is
[docs/plan.md](plan.md) and [docs/missing.md](missing.md).

## Hash path and output reserve fix in progress

The low-level hash path (`HashClassifyKernel`, `HashScatterKernel`) and the
transform output-reserve derivation are under correctness review. The typed
`Flow` partition stage uses the dense path (`DenseBinTable`), which remains the
correctness reference and the only path the examples use. The
hash-path/reserve fix is in progress and is not done.

## CPU path blocked on the H100 host by the OMP executor

The CPU path on `jrc0880.jureca` is blocked by the OpenMP executor; the
investigation is in progress. No CPU result is claimed for the H100 host until
it is resolved.

## H100 evidence is preliminary

The recorded H100 numbers are preliminary: CUDA only, at commit
`aa2c7e11fe8f11625534dee66136d9ab899d29d1` (older than current main), with an
external-load caveat (sibling agent workloads on the same node from about
20:29 UTC 2026-11-13; a long CPU run was SIGKILLed at 925 s). No final H100
acceptance claim is made.

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
the batch tile list, so several tiles may share a block. The former pending
FrameSpec item is closed.

## AMD execution is blocked

No AMD GPU is visible through the available compute targets. HIP compiles the
full tree for `gfx90a` with `amdclang++` from ROCm under `/opt/rocm`, but the
example and the tests do not run. HIP execution, HIP histogram equality, and an
AMD `N=1e8` acceptance run remain open. The compile-only gate is not an AMD
result.

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

## Throughput is a shared-host baseline

`hal` reports `sharing=unknown` and fwk394 is shared. The historical `N=1e8`
result of 187.5 s, about `5.33e5` particles/s, and the integrated T13
wall times are shared-host baselines, not exclusive-allocation acceptance
numbers. There is no variance study and no frozen regression target.

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

## Generic graph core is host-only

The tree has the `NodeBase`/`Graph`/`GraphBuilder`/`RouteTable` core with
source/transform/sink roles, per-bin routing, and feedback edges as ordinary
routes. Node objects are caller-owned and non-owning, the graph is immutable
after `build()`, and v1 allows one source and no routes into that source.
Runtime graph mutation, join, and broadcast remain open. Both examples now use
the typed `Flow` API; the lower-level `GraphBuilder` core is host-only and
unused by the examples.

## No CI runners

There is no `.github/workflows` on main and no CI runner pool. The CUDA, HIP,
and TSAN gates require self-hosted runners with known labels. The local
commands in `docs/build.md` are documented entry points, not a CI run. The CI
workflow exists as draft PR #4 (`gf/t11-ci`, head
`19aba8345ce85fec474573902fdfe775c38ff4b8`) with no runners provisioned. The
repository is public and has no branch-protection or CI badge configuration.

## Accelerator test coverage is partial

Most Catch2 cases construct the CPU device. The CUDA build adds
`tests/unit/test_cuda_kernel.cpp`, which runs a kernel on the selected NVIDIA
GPU and fails when no GPU is visible. HIP has no runtime test. There is no
automated CPU/CUDA/HIP histogram equality test.

## Hash path evidence is CPU-only

The dense path is the correctness reference and the only path the example uses.
The shared-memory hash path is exercised by CPU integration tests. It compiles
for CUDA and HIP, but no hash-path accelerator execution is recorded. The
dense/hash threshold is the static 1024-bin limit, not a measured crossover.

## Other open items

- The particle example binary is registered in CTest as 24 CLI cases per
  backend (`N=0,1,17,127,128,4097` crossed with `C=1,31,127,128`) plus one
  `example.public_api` source-contract case. The photon example adds 10 CLI
  cases per backend (8 forward plus one isotropic and one low-cap forward
  case), one output case, and one `example.photon_public_api` source-contract
  case per build. The current integrated totals are 194/194 CPU and 246/246
  CUDA.
- The historical CPU/CUDA `N=1e6` histogram equality check was a manual diff,
  not an automated test, and was not repeated on the integrated revision.
- The integrated T13 benchmark is recorded in `docs/benchmarks.md`; its wall
  times remain shared-host baselines, and no frozen regression target exists
  yet.
- The result buffer is `O(N)` (about 400 MB at `N=1e8`).
- The `sm_75` RTX 2080 Ti path is a documented build target, not a measured run.
- The hash load factor (`2*T` entries by default) has not been measured.
