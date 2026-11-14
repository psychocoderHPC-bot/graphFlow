<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow acceptance record

Current main is `cb2de187297bb7277b21ea47a6d2a8ed05b909ba` (merge of the
debug-validation work). The T5 photon scattering record is at
`ac89ab558c26b938038035b26ca5dbe6d7ce7727`, building on the typed Flow
execution and particle example port at
`b138d9a6d5ef0cca18bd634a0a98abd45e13f9bd` and the earlier integrated
revision `cb0548cedf81a284e6291a39dee1eeca7256d783` (T1, T2, T3, T5, T7,
T13, T14a). The current final counts and the historical T13 performance table
remain below as evidence.

The HIP runtime validation in this record ran at
`58bb514def3f3c956126ab7c34d0de74a45f679e` on base
`25fa0927a4e042f229d43229734f50dce8277954`. It is the first record with HIP
execution on a real AMD GPU; earlier HIP entries are compile-only and are
labelled historical below.

## Validation platform

`rosi_v100` is the exclusive validation platform. Current `rosi_v100`
results:

| Configuration | Result |
| --- | --- |
| CPU CTest on `rosi_v100` (OMP=ON Release) | **202/202 passed** |
| CPU CTest on `rosi_v100` (OMP=ON Debug) | **195/195 passed** (measured, pre-hash-fix); expected 202/202 after the seven hash cases (not re-run) |
| CUDA CTest on `rosi_v100` (V100 `sm_70`, CUDA 12.6.85 Release) | **254/254 passed** |
| CUDA CTest on `rosi_v100` (V100 `sm_70`, CUDA 12.6.85 Debug) | **247/247 passed** (measured, pre-hash-fix); expected 254/254 after the seven hash cases (not re-run) |
| HIP CTest on `hal` (AMD Radeon RX 7900 XTX `gfx1100`, ROCm 7.2, Release) | **245/245 passed**; 35 HIP cases, measured on `b972eaa` |
| CPU OpenMP on `rosi_v100` (passive, `GOMP_SPINCOUNT=0`) | Works; no hangs. `N=1e5`, `C=128`/`P=1024`, `Q=1`: OMP=1 2.88 s, OMP=2 2.38 s (best), OMP=4 2.44 s, OMP=8 2.62 s. `N=1e6` OMP=2 Q1 23.71 s, Q8 4.46 s. All sums exact. |

The V100 `sm_70` build is unblocked by the explicit concept-parameter
workaround in the transform, partition, scatter, and hash-scatter kernels and
the test hook adapters. The RNG main is preserved. Both totals are one higher
than before the RNG merge: the RNG change removed one `TEST_CASE` and the
debug validation adds two `[photon][debug]` cases. The two cases run in both
build types: Release asserts that the hot path has no per-event state branch,
and Debug exercises the compiled checks.
`GRAPHFLOW_ENABLE_DEBUG_VALIDATION` defaults to `ON` for
`CMAKE_BUILD_TYPE=Debug` and `OFF` for `Release` and `RelWithDebInfo`; the
cache entry stays overridable. Release is recorded below; the `Debug` build
was not re-run on `rosi_v100` after the seven hash cases, so the measured
`rosi_v100` `Debug` totals are the pre-hash 195/195 CPU and 247/247 CUDA, and
202/202 CPU and 254/254 CUDA are the expected (not-yet-re-run) `Debug`
totals. Release has no per-event checks; Debug compiles them in. Host setup validates
the configuration once, and genuine physics failures stay on the `StatusCode`
channel in both builds. The five former `[status]` tests that tripped the RNG
stream-id assertion on Debug builds now pass in both builds.
The hash-path rerun, CI, and the tree-wide format/SPDX sweep (T14b) remain
open. CUDA 13.4/`sm_80`, fwk394, and H100 were not re-run in this validation,
so no claim is updated beyond the recorded `rosi_v100` run.

The previously recorded counts, fwk394 CPU 194/194 and hal CUDA 246/246, are
**historical validation on a shared host** (`hal` reports `sharing=unknown`)
and are **not the final current validation**.

H100: the JURECA `jrc0880.jureca` (4x H100 PCIe 80GB, SM 9.0) run is
preliminary CUDA evidence only: its gateway sharing flag is unknown and the
recorded run observed external load while the client was stale, so no final
H100 claim is made. H100 remains blocked/preliminary. Still open: the
tree-wide format and SPDX sweep (T14b) is CPU-only work. Hosted CI runs the
CPU CTest suite and `reuse lint`, with compile-only CUDA and HIP. This docs
commit records the existing `rosi_v100` and HIP `hal` results; it adds no new
build or test run.

Environment for the historical recorded runs: fwk394 for the CPU host and hal
for CUDA, NVIDIA A30 (sm_80, 24 GiB), CUDA 13.4, g++ 13.3, CMake 3.28.3,
Ninja 1.11.1, alpaka `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (alpaka
3.0.0), C++20, Release. `hal` reports `sharing=unknown`; the historical
throughput is a shared-host baseline. The `rosi_v100` CPU CTest and CPU
OpenMP runs use OMP=ON Release with `GOMP_SPINCOUNT=0` (passive). The
`rosi_v100` CUDA V100 `sm_70` build passes with nvcc 12.6.85 using the
explicit concept-parameter workaround.

## HIP runtime validation on hal

The HIP runtime record is measured on `b972eaa3bd778b81f2dca505d3662df710792bd9`
(HIP runtime merge, PR #35). Platform: `hal`, AMD Radeon RX 7900 XTX
(`gfx1100`), ROCm 7.2.0, `amdclang++` (Clang 22.0.0), offload arch `gfx1100`,
C++20 Release, alpaka `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (3.0.0). The
run is recorded in `/home/agent/final-hip-b972eaa/ctest.log`. The earlier HIP
runtime total of 238/238 at `58bb514def3f3c956126ab7c34d0de74a45f679e` is
historical (pre the final CPU case set).

| Configuration | Result |
| --- | --- |
| HIP CTest (`ROCR_VISIBLE_DEVICES=0`, Release) on `b972eaa` | **245/245 passed** (100%, `ctest.log`) |
| HIP-tagged cases (ctest `hip` label) | **35**: 8 typed Flow, 24 particle CLI, 3 photon CLI; plus 3 photon output cases |
| `rocprof` v3, particle feedback example | 4,901 GPU kernel dispatches |
| `rocprof` v3, typed Flow case | passed, 38,266 assertions |

The eight typed Flow cases are the feedback matrix (`N=0,1,17,128,4097`
crossed with `Q=1,2,8`), a partial tail chunk, a bounded pool, empty and single
source/sink runs, the generic POD chain, one batch-wide transform launch, one
classify plus one scatter partition launch, and Q=8 feedback termination. The
particle and photon CLI/output cases run the real example binaries on the HIP
backend. `rocprof` recorded device kernels from the `api::Hip` path, so the
run is device execution rather than a host fallback.

`hal` reports `sharing=unknown`, so this record makes no HIP performance
claim. Other AMD architectures, an AMD `N=1e8` acceptance run, and the HIP hash
path remain unrecorded. fwk394 and H100 were not re-run. GitHub-hosted HIP
builds remain compile-only because those runners have no AMD GPU; no HIP
runtime claim is made for GitHub CI.

## Builds

The `rosi_v100` CPU Release and Debug builds pass and the CPU CTest suite runs
to completion. The `rosi_v100` CUDA V100 `sm_70` Release and Debug builds pass
with nvcc 12.6.85 using the explicit concept-parameter workaround. The
historical build records below are from the previous platforms.

| Configuration | Result |
| --- | --- |
| `rosi_v100` CPU Release (OMP=ON) | configure and build pass; CTest 202/202 |
| `rosi_v100` CPU Debug (OMP=ON) | configure and build pass; CTest 195/195 (measured, pre-hash-fix); expected 202/202 after the seven hash cases (not re-run) |
| `rosi_v100` CUDA Release (V100 `sm_70`) | configure and build pass; CTest 254/254 |
| `rosi_v100` CUDA Debug (V100 `sm_70`) | configure and build pass; CTest 247/247 (measured, pre-hash-fix); expected 254/254 after the seven hash cases (not re-run) |
| Historical CPU Release on fwk394 | configure and build pass |
| Historical CUDA Release on hal (`CMAKE_CUDA_ARCHITECTURES=80`) | configure and build pass |
| Historical HIP Release (`gfx90a`, `amdclang++`, `/opt/rocm`) | compile-only gate passed; no AMD runtime device was visible, so execution was blocked at that revision |
| HIP Release on hal (`gfx1100`, AMD Radeon RX 7900 XTX, ROCm 7.2) | configure, build, and CTest 245/245 pass (measured on `b972eaa`, `ctest.log`) |
| ASAN+UBSAN Debug (hosted CI) | 202/202 pass on the CPU suite; the `sanitizers` job runs ASAN+UBSAN over the CPU CTest |
| TSAN Debug | fwk394 non-PIE scheduler/pool subset 14/14; hal blocked by `unexpected memory mapping`; no full integrated TSAN run claimed |

## Tests

Current `rosi_v100` CTest: CPU **202/202 passed** and CUDA **254/254 passed**
on V100 `sm_70` (CUDA 12.6.85 Release). The `Debug` build was not re-run on
`rosi_v100` after the seven hash cases, so the measured `Debug` totals are the
pre-hash **195/195** CPU and **247/247** CUDA, with 202/202 and 254/254 the
expected (not-yet-re-run) `Debug` totals. HIP on `hal` (RX 7900 XTX
`gfx1100`, ROCm 7.2) passed **245/245** with 35 HIP cases, measured on
`b972eaa`. The explicit concept-parameter workaround unblocks the nvcc 12.6.85
name-hiding ICE; the RNG main is unchanged. The historical CTest totals below
are from the previous platforms.

| Configuration | Result |
| --- | --- |
| `rosi_v100` CPU CTest (OMP=ON Release) | **202/202 passed** |
| `rosi_v100` CPU CTest (OMP=ON Debug) | **195/195 passed** (measured, pre-hash-fix); expected 202/202 after the seven hash cases (not re-run) |
| `rosi_v100` CUDA CTest (V100 `sm_70`, Release) | **254/254 passed** |
| `rosi_v100` CUDA CTest (V100 `sm_70`, Debug) | **247/247 passed** (measured, pre-hash-fix); expected 254/254 after the seven hash cases (not re-run) |
| Historical CPU CTest on fwk394 | 170/170 passed at T3b; 194/194 on the integrated main (T5) |
| Historical CUDA CTest on hal (`sm_80`) | 206/206 passed at T3b; 246/246 on the integrated main (T5) |
| HIP CTest on hal (`gfx1100`, RX 7900 XTX, ROCm 7.2) | **245/245 passed**; 35 HIP cases (measured on `b972eaa`) |
| Historical HIP compile-only gate (`gfx90a`) | compiled; no AMD device was visible at that revision, so no execution was recorded |
| ASAN+UBSAN CTest | not recorded for the integrated revision |

The historical integrated main totals (194/194 CPU, 246/246 CUDA) are recorded
in the T5 section below. The T3b totals add twelve status/failure cases: the status/failure unit
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

Historically recorded at `b138d9a6d5ef0cca18bd634a0a98abd45e13f9bd` with
`-Dalpaka_COMPILE_PEDANTIC=ON`: **158/158 CTest pass on fwk394 CPU Release**
and **194/194 pass on hal CUDA `sm_80` Release**. A fixed-seed 33-invocation
CLI comparison (the 24 N/C cases plus nine Q/P/T/frameCap and max-visits cases)
matched the pre-change binary on both backends. The example header (including
`live-particles`), policy line, histogram rows, non-terminating count, and
exit status are identical. The derived pool reserve changes from `P+6` to
`P+8` because the A/B partitions now expose three named outcomes; that
diagnostic is not a numeric result. These counts are historical. The
`rosi_v100` CPU CTest passes 202/202 (OMP=ON Release) and the V100
`sm_70` CUDA CTest passes 254/254 in Release with the explicit
concept-parameter workaround; the `rosi_v100` 33-invocation baseline comparison
is not yet recorded.

### T5 photon scattering output and diagnostics

The photon-scattering rework was historically tested at implementation commit
`7e3a9c9121afaaebcbf149ac7e5d18b4b933e295` on base
`449c1ec1ffcd1c72f780c688cf113c82b70bf130`. The historical environment is the
one above: fwk394 CPU with g++ 13.3, hal CUDA with an NVIDIA A30 (`sm_80`),
CUDA 13.4, CMake 3.28.3, Ninja 1.11.1, alpaka
`eb677e8d7f70d79b35df25d9002f94ba669b6f44`, C++20 Release, with
`-Dalpaka_COMPILE_PEDANTIC=ON` (`-Werror` confirmed in both build caches).
The remote source trees were uploaded as archives without `.git`, so the
provenance was passed explicitly with
`-DGRAPHFLOW_GIT_COMMIT=7e3a9c9121afaaebcbf149ac7e5d18b4b933e295` and
`-DGRAPHFLOW_ALPAKA_PIN=eb677e8d7f70d79b35df25d9002f94ba669b6f44`.

| Configuration | Command | Result |
| --- | --- | --- |
| `rosi_v100` CPU (OMP=ON Release) | `ctest --output-on-failure` | **202/202 passed** |
| `rosi_v100` CPU (OMP=ON Debug) | `ctest --output-on-failure` | **195/195 passed** (measured, pre-hash-fix); expected 202/202 after the seven hash cases (not re-run) |
| `rosi_v100` CUDA (V100 `sm_70` Release) | `ctest --output-on-failure` | **254/254 passed** |
| `rosi_v100` CUDA (V100 `sm_70` Debug) | `ctest --output-on-failure` | **247/247 passed** (measured, pre-hash-fix); expected 254/254 after the seven hash cases (not re-run) |
| Historical CPU fwk394 | `cmake --build photon-rework-build-cpu -j 16` then `ctest --output-on-failure -j 8` | 194/194 passed |
| Historical CUDA hal A30 | `cmake --build photon-rework-build-cuda -j 16` then `ctest --output-on-failure -j 4` | 246/246 passed |
| Historical CPU diagnostics | `./graphFlow_tests '[diagnostics]'` | 2 test cases passed (1,000,019 assertions) |
| Historical CUDA diagnostics | `./graphFlow_tests '[diagnostics]'` | 2 test cases passed (1,000,019 assertions) |
| Historical label registration | `-DGRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS=ON` then `ctest -N -L diagnostics` | 2 tests registered; default configure registers 0 |

The rework adds `--output-dir` CSV output, bounded deterministic traces,
`tools/plot_photon_scattering.py`, output/trace tests, and the opt-in
diagnostics label. A historical CPU sample with the final binary wrote
`metadata.csv`, `outcomes.csv`, `scatter_histogram.csv`, `exits.csv`,
`traces.csv` and `trace_summary.csv`; metadata reported the tested commit,
alpaka pin, `sm_80`, and the expected parameters. The plotting script ran on
the forward and isotropic samples and wrote `figure1_paths` through
`figure4_detector` in PNG, SVG and PDF (12 files per sample). The forward
sample had `transmitted=4097`, `reflected=0`, `absorbed=0`, `capped=0`.
The isotropic sample had `transmitted=1946`, `reflected=1066`,
`absorbed=1085`, `capped=0`, summing to 4097. At that historical revision HIP
was compile-only because no AMD runtime device was visible; the current HIP
runtime validation is recorded above. H100: the JURECA `jrc0880.jureca` run is
preliminary CUDA
evidence only (unknown gateway sharing flag; the recorded run observed
external load while the client was stale), so no final H100 claim is made and
H100 remains blocked/preliminary. The `rosi_v100` CPU OpenMP run (passive,
`GOMP_SPINCOUNT=0`) works with no hangs; the histogram sums are exact.

### PR-A generic graph API core

The additive graph API core was validated at
`60a34302b1c3cf205304916c7233f13118591dd0` on the environment above:
**97/97 CTest passed on fwk394 CPU Release** and **101/101 passed on hal CUDA
`sm_80` Release**, both with `-Dalpaka_COMPILE_PEDANTIC=ON`. The change adds
`NodeBase`, `RouteTable`, `Graph`, and `GraphBuilder` over the existing
`GraphRunner` hooks without changing the example, T5 batching, FrameSpec
launches, or policy semantics. That historical record made no HIP runtime
claim; the HIP gate was compile-only at that revision. The current HIP runtime
validation is recorded above. This section records the existing runs and
claims no new build or test run.

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
The tree-wide format and SPDX sweep (T14b) remains open. Hosted CI is
`.github/workflows/ci.yml` and is documented in `docs/ci.md`: CPU CTest and
ASAN+UBSAN run there, CUDA and HIP are compile-only, and `reuse lint` runs. No
format job is wired in until T14b lands.
