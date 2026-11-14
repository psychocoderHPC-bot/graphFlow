<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow acceptance record

Current integrated revision:
[`ac89ab558c26b938038035b26ca5dbe6d7ce7727`](https://github.com/psychocoderHPC-bot/graphFlow/commit/ac89ab558c26b938038035b26ca5dbe6d7ce7727)
(main, merge of PR #22). The public API chain landed as PR #16
(`f8bedf0d800c4b1beb564c54ff59f99ebbc9bd85`), #17
(`17f9e02fe26b3e8f5df110d767eca164c0c6a646`), #18
(`ed7cf397cf3b36ce4b26f13c749e2423e181fb54`), #19
(`60b566dc7f45369008eddc81219a2ecc07f63a9a`), #20
(`f5254f57aa637f0e14a12d58765a27347628f8c4`), #21
(`449c1ec1ffcd1c72f780c688cf113c82b70bf130`), and #22
(`ac89ab558c26b938038035b26ca5dbe6d7ce7727`). The current CTest totals are
194/194 on fwk394 CPU and 246/246 on hal CUDA `sm_80`, recorded for the
photon rework in the T5 section below. This docs commit adds no new build or
test run.

The integrated T4 revision was
`b138d9a6d5ef0cca18bd634a0a98abd45e13f9bd` (typed Flow execution and the
particle example port); the earlier integrated revision was
`cb0548cedf81a284e6291a39dee1eeca7256d783` (T1, T2, T3, T5, T7, T13, T14a).
Their counts and the historical T13 performance table remain below as
historical evidence.

Environment for the recorded runs: fwk394 for the CPU host and hal for CUDA,
NVIDIA A30 (sm_80, 24 GiB), CUDA 13.4, g++ 13.3, CMake 3.28.3, Ninja 1.11.1,
alpaka `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (alpaka 3.0.0), C++20,
Release. `hal` reports `sharing=unknown`; throughput is a shared-host baseline.

## Builds

| Configuration | Result |
| --- | --- |
| CPU Release on fwk394 | configure and build pass |
| CUDA Release on hal (`CMAKE_CUDA_ARCHITECTURES=80`) | configure and build pass |
| HIP Release (`gfx90a`, `amdclang++`, `/opt/rocm`) | compile-only gate passes; no AMD runtime device on hal, execution blocked |
| ASAN+UBSAN Debug | not recorded for the integrated revision |
| TSAN Debug | fwk394 non-PIE scheduler/pool subset 14/14; hal blocked by `unexpected memory mapping`; no full integrated TSAN run claimed |

## Tests

| Configuration | Result |
| --- | --- |
| CPU CTest on fwk394 | 170/170 passed at T3b |
| CUDA CTest on hal (`sm_80`) | 206/206 passed at T3b |
| ASAN+UBSAN CTest | not recorded for the integrated revision |
| HIP | compiled; no AMD device was visible on hal, execution blocked, so no execution is claimed |

The T3b totals add twelve status/failure cases: the status/failure unit
suite and three typed failure-path integration cases. The CTest totals include
`tests/unit/test_launch_policy.cpp` and
`tests/integration/test_framespec.cpp`, which pin the FrameSpec formula, the
`FrameSpec` grid-stride mapping, the N/C/T/frameCap/Q matrix, and the
`T > maxThreadsPerBlock` rejection. The CUDA build adds
`tests/unit/test_cuda_kernel.cpp`, which runs a kernel on the selected NVIDIA
GPU and fails when no GPU is visible; the FrameSpec matrix also runs on CUDA.

### T4 public Flow example

T4 ports `graphFlow_particle_feedback` to the public `Flow<graphFlow::Particle>`
API. The graph is declared with source/partition/sink handles and named outcome
ports; `ParticleFeedback.hpp` is 121 lines and no longer includes the deprecated
`ParticleStages.hpp`. CTest registers 24 CLI cases per backend
(`N=0,1,17,127,128,4097` crossed with `C=1,31,127,128`) plus one
`example.public_api` source-contract case: 25 example tests in a CPU build and
49 in a CUDA build (24 CPU, 24 CUDA, 1 contract).

Recorded at `b138d9a6d5ef0cca18bd634a0a98abd45e13f9bd` with
`-Dalpaka_COMPILE_PEDANTIC=ON`: **158/158 CTest pass on fwk394 CPU Release**
and **194/194 pass on hal CUDA `sm_80` Release**. A fixed-seed 33-invocation
CLI comparison (the 24 N/C cases plus nine Q/P/T/frameCap and max-visits cases)
matched the pre-change binary on both backends. The example header (including
`live-particles`), policy line, histogram rows, non-terminating count, and
exit status are identical. The derived pool reserve changes from `P+6` to
`P+8` because the A/B partitions now expose three named outcomes; that
diagnostic is not a numeric result.

### T5 photon scattering output and diagnostics

The photon-scattering rework was tested at implementation commit
`7e3a9c9121afaaebcbf149ac7e5d18b4b933e295` on base
`449c1ec1ffcd1c72f780c688cf113c82b70bf130`. The environment is the one above:
fwk394 CPU with g++ 13.3, hal CUDA with an NVIDIA A30 (`sm_80`), CUDA 13.4,
CMake 3.28.3, Ninja 1.11.1, alpaka
`eb677e8d7f70d79b35df25d9002f94ba669b6f44`, C++20 Release, with
`-Dalpaka_COMPILE_PEDANTIC=ON` (`-Werror` confirmed in both build caches).
The remote source trees were uploaded as archives without `.git`, so the
provenance was passed explicitly with
`-DGRAPHFLOW_GIT_COMMIT=7e3a9c9121afaaebcbf149ac7e5d18b4b933e295` and
`-DGRAPHFLOW_ALPAKA_PIN=eb677e8d7f70d79b35df25d9002f94ba669b6f44`.

| Configuration | Command | Result |
| --- | --- | --- |
| CPU fwk394 | `cmake --build photon-rework-build-cpu -j 16` then `ctest --output-on-failure -j 8` | 194/194 passed |
| CUDA hal A30 | `cmake --build photon-rework-build-cuda -j 16` then `ctest --output-on-failure -j 4` | 246/246 passed |
| CPU diagnostics | `./graphFlow_tests '[diagnostics]'` | 2 test cases passed (1,000,019 assertions) |
| CUDA diagnostics | `./graphFlow_tests '[diagnostics]'` | 2 test cases passed (1,000,019 assertions) |
| Label registration | `-DGRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS=ON` then `ctest -N -L diagnostics` | 2 tests registered; default configure registers 0 |

The rework adds `--output-dir` CSV output, bounded deterministic traces,
`tools/plot_photon_scattering.py`, output/trace tests, and the opt-in
diagnostics label. A CPU sample with the final binary wrote
`metadata.csv`, `outcomes.csv`, `scatter_histogram.csv`, `exits.csv`,
`traces.csv` and `trace_summary.csv`; metadata reported the tested commit,
alpaka pin, `sm_80`, and the expected parameters. The plotting script ran on
the forward and isotropic samples and wrote `figure1_paths` through
`figure4_detector` in PNG, SVG and PDF (12 files per sample). The forward
sample had `transmitted=4097`, `reflected=0`, `absorbed=0`, `capped=0`.
The isotropic sample had `transmitted=1946`, `reflected=1066`,
`absorbed=1085`, `capped=0`, summing to 4097. HIP remains compile-only with no
AMD runtime device; no H100 run is claimed.

### PR-A generic graph API core

The additive graph API core was validated at
`60a34302b1c3cf205304916c7233f13118591dd0` on the environment above:
**97/97 CTest passed on fwk394 CPU Release** and **101/101 passed on hal CUDA
`sm_80` Release**, both with `-Dalpaka_COMPILE_PEDANTIC=ON`. The change adds
`NodeBase`, `RouteTable`, `Graph`, and `GraphBuilder` over the existing
`GraphRunner` hooks without changing the example, T5 batching, FrameSpec
launches, or policy semantics. At that revision the particle-feedback example
assembled its graph by hand; the typed `Flow` port landed later (PR #16-#22)
and both examples now run on the public API. No HIP runtime is claimed; the
HIP gate remains compile-only. This documentation commit records the existing
runs and claims no new build or test run.

## Preliminary H100 evidence (not acceptance)

Recorded 2026-11-13 on JURECA `jrc0880.jureca` (4x H100 PCIe 80GB, SM 9.0,
exclusive per operator, gateway sharing flag unknown) at graphFlow commit
[`aa2c7e11fe8f11625534dee66136d9ab899d29d1`](https://github.com/psychocoderHPC-bot/graphFlow/commit/aa2c7e11fe8f11625534dee66136d9ab899d29d1),
older than the current main, with an sm_90 build. CUDA only:

| N | Q | Wall time | Notes |
| ---: | ---: | ---: | --- |
| 100,000,000 | 1 | ~157 s | host-submission bound, one host core busy |
| 100,000,000 | 2 | 6.76 s | |
| 100,000,000 | 8 | 6.52 s | about 1.30e7 particles/s |
| 1,000,000 | 1 | 2.17 s | |
| 1,000,000 | 8 | 0.66 s | same 3.3x step |

`C=8192`, `P=16384`. The sorted histogram is bit-identical across Q at each
N. The CPU run (`C=1024`, `P=512`) matched CUDA at `N=1e6`. External-load
caveat: from about 20:29 UTC 2026-11-13, sibling agent workloads ran on the
same node and a long CPU run was SIGKILLed at 925 s, so node exclusivity was
not verified for the whole window. The CPU path on this host is blocked by
the OMP executor, whose investigation is in progress. This is preliminary
evidence, not an acceptance claim; the final H100 acceptance item stays open.

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
- The `N=1e8` run used `P=16128`, not the documented default `P=16384`: the
  measured A30 payload budget is 3,171,090,432 B, and `C=8192`, `P=16384`
  needs 3,221,225,472 B, so T7 rejects it; `C=8192`, `P=16128`
  (3,170,893,824 B) passes. The auto-derived policy now clamps `P` to 16,128 on
  this device; explicit `P=16384` and `P=65536` are rejected.

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

The fixed pool remains the source capacity plus one output bound per registered
node. `Flow` derives the reserve; the example's two-output filter and two
three-output A/B partitions reserve `2 + 3 + 3 = 8` chunks. A transform batch
reserves the adaptive `outputChunkBound(sum(fill), C, B)` and shrinks its chunk
prefix to the largest one that fits the current credits, so Q does not enlarge
the pool.

The former `C=8192`, `P=65536` pair is no longer a default and is rejected by
the payload budget. Its chunk and payload figures appear only in the historical
table above and are not current defaults. The policy-derived accelerator
defaults are pinned by the unit tests at `C=8192`, `P=16384`, `T=128` on a
nominal 24 GiB device; `Flow` adds the exact registered-node reserve to the policy
budget. The real A30 reports `24,192 MiB` usable after ECC, so its 1/8 budget is
`3,171,090,432 B`. A separate test pins that budget: the policy selects
`P=16128` for the default `C` and for an explicit `C=8192` with auto `P`; an
explicit `P=16384` is rejected. `applyDevicePolicy` + `derivePoolSize` +
`Config::validate` accepts the resulting pool.

A slot-based pool remains future work. The budget caps the fixed-capacity
design without changing the `P` plus registered-node-reserve progress bound.

## Repository gates

`.clang-format` (clang-format 20.1.8), `.reuse/dep5`, and
`LICENSES/MPL-2.0.txt` are present, so the format/REUSE foundation is in main.
The tree-wide format and SPDX sweep (T14b) remains open. The CI workflow
exists as draft PR #4 on branch `gf/t11-ci` (head
`19aba8345ce85fec474573902fdfe775c38ff4b8`); no `.github/workflows` is on
main and no runners are provisioned. The hash-path and output-reserve fix and
the CPU OMP executor investigation are in progress. Format and REUSE commands
are documented in `docs/build.md`; no CI run is claimed.
