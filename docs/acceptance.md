<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow acceptance record

Integrated revision: `68124ad3875e42cbe09247f64bd218a7ac8d58fc` (T1, T2, T3,
T5, T7, T14a). This record uses the supplied integrated acceptance evidence. It
does not claim a local rerun in this worktree.

Environment for the recorded runs: fwk394 for the CPU host and hal for CUDA,
NVIDIA A30 (sm_80, 24 GiB), CUDA 13.4, g++ 13.3, CMake 3.28.3, Ninja 1.11.1,
alpaka `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (alpaka 3.0.0), C++20,
Release. `hal` reports `sharing=unknown`; throughput is a shared-host baseline.

## Builds

| Configuration | Result |
| --- | --- |
| CPU Release on fwk394 | configure and build pass |
| CUDA Release on hal (`CMAKE_CUDA_ARCHITECTURES=80`) | configure and build pass |
| HIP Release (`gfx90a`, `amdclang++`, `/opt/rocm`) | compile-only gate passes |
| ASAN+UBSAN Debug | not recorded for the integrated revision |
| TSAN Debug | fwk394 scheduler/pool subset 14/14; hal blocked by `unexpected memory mapping` |

## Tests

| Configuration | Result |
| --- | --- |
| CPU CTest on fwk394 | 74/74 passed |
| CUDA CTest on hal | 75/75 passed |
| ASAN+UBSAN CTest | not recorded for the integrated revision |
| HIP | compiled; no AMD device was visible, so no execution is claimed |

The CUDA build adds `tests/unit/test_cuda_kernel.cpp`, which runs a kernel on
the selected NVIDIA GPU and fails when no GPU is visible.

## Historical performance baseline

The numbers below are the pre-T5/T7 result. They are not a current integrated
run. They predate device-aware policy resolution, multi-chunk batching, and the
payload budget. The integrated T13 measurement is recorded below; the
historical table stays for comparison.

| Backend | N | Wall time | Histogram sum | Peak live chunks (historical pool) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 1,000,000 | 19.45 s | 1,000,000 | not recorded | maxrss 76.8 MB |
| CUDA | 100,000 | 0.58 s | 100,000 | 514 (65542) | |
| CUDA | 1,000,000 | 2.24 s | 1,000,000 | 857 (65542) | |
| CUDA | 100,000,000 | 187.5 s | 100,000,000 | 877 (65542) | 264 rows; about 5.33e5 particles/s |

The historical `N=1e8` histogram summed to exactly 100,000,000 over 264 rows.
The historical CPU and CUDA `N=1e6` histograms were identical by direct row
diff. That equality check was not repeated on the integrated revision.

The queue probe on CUDA reported `overlap=no queues-alias=no
second-before-first=no`; the queues were distinct, but no overlap was observed.
Correctness does not depend on overlap.

## Integrated T13 measurement

The integrated T13 benchmark is done. The table below is the supplied T13
evidence for this revision, not a fresh run in this worktree. `hal` reports
`sharing=unknown` and fwk394 is shared, so every wall time and derived
particles/s value is a single-run shared-host baseline, not an
exclusive-hardware acceptance claim.

hal A30, CUDA sm_80, `C=8192`, seed 42:

| N | Q | Wall time | Histogram sum | particles/s |
| --- | ---: | ---: | ---: | ---: |
| 100,000,000 | 1 | 218.87 s | 100,000,000 | ~4.57e5 |
| 100,000,000 | 2 | 9.83 s | 100,000,000 | ~1.02e7 |
| 100,000,000 | 8 | 8.50 s | 100,000,000 | ~1.18e7 |
| 1,000,000 | 1 | 2.32 s | 1,000,000 | ~4.31e5 |
| 1,000,000 | 2 | 0.41 s | 1,000,000 | ~2.44e6 |
| 1,000,000 | 8 | 0.35 s | 1,000,000 | ~2.86e6 |

Recorded effects:

- Q effect. Raising the batch-chunk count from Q=1 to Q=2 and Q=8 cut the
  `N=1e8` wall time from 218.87 s to 9.83 s and 8.50 s, and the `N=1e6` wall
  time from 2.32 s to 0.41 s and 0.35 s. The sorted histogram rows are
  identical across Q within each N.
- The `N=1e8` run used `P=16128`, not the policy default `P=16384`: the
  measured A30 payload budget is 3,171,090,432 B, and `C=8192`, `P=16384`
  needs 3,221,225,472 B, so T7 rejects it; `C=8192`, `P=16128`
  (3,170,893,824 B) passes. The default-policy clamp that would keep `P`
  within the device budget is not yet fixed in this revision; the run used an
  explicit `P=16128`.

CPU/CUDA histogram equality, `C=128`, `P=1024`, `Q=1`, seed 42: `rows_equal`
is yes for `N=1e5` (CPU 15.23 s, CUDA 1.96 s) and `N=1e6` (CPU 149.44 s,
CUDA 17.66 s); both histogram sums equal `N`.

fwk394, CPU, `C=1024`, `P=512`, `Q=1`: `N=1e6` took 52.75 s, histogram sum
1,000,000. The fwk394 CUDA sm_75 build fails with an nvcc 12.5 internal
compiler error, so no fwk394 CUDA run is claimed.

## Source admission history

The first implementation admitted source chunks whenever any particle budget
was free. Under `N >> P` this produced one-particle chunks: the historical pool
filled to 8195 of 8198 chunks, and a pre-T5/T7 1e6 CUDA run took about 119 s.
The fix requires a source chunk to carry at least `min(C, P, remaining)`
particles. Consumer ready lists still dispatch partial chunks immediately; the
minimum applies only to source admission.

## Pool and payload budget

The fixed pool remains `S = P + K*R(C)` with
`R(C) = outputChunkBound(C, C, B)` for three transform nodes. A transform batch
reserves the adaptive `outputChunkBound(sum(fill), C, B)` and shrinks its chunk
prefix to the largest one that fits the current credits, so Q does not enlarge
the pool.

The former `C=8192`, `P=65536` pair is no longer a default and is rejected by
the payload budget. Its chunk and payload figures appear only in the historical
table above and are not current defaults. The policy-derived accelerator
defaults are pinned by the unit tests at `C=8192`, `P=16384`, `T=128`; the CLI
adds the exact `K*R(C)` reserve to the policy budget.

A slot-based pool remains future work. The budget caps the fixed-capacity
design without changing the `S = P + K*R(C)` progress bound.

## Repository gates

`.clang-format` (clang-format 20.1.8), `.reuse/dep5`, and
`LICENSES/MPL-2.0.txt` are present. There is no CI workflow yet. Format and
REUSE commands are documented in `docs/build.md`; no CI run is claimed.
