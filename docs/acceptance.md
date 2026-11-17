<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow acceptance record

## Current state at `25c73f3`

This is the canonical current-state block. Every older total in this document is
historical. The totals below are the measured suites at
`25c73f31c736b1c5f765003ff8815d2a24c25808`; the 255/255, 316/316, 307/307 and
72/72 totals at `561b9a4c`/`9ace795` are the previous measurements. The AoS
attribute matrix (SoAFlow over the llamaLite record storage), the
registration-site transform attribute subsets, and the record-pipeline example
are included in the totals above: three AoS cases per backend, the attribute
helpers pass 6/6, the registration-site subset case 1/1, and the record
pipeline 2/2. The CUDA suite runs on the RTX 2080 Ti `sm_75`, on
`fwk394`. The SYCL full suite registers 397 cases in the `hal` build; 77 of
them carry the `sycl` label.

| Configuration | Host / device | Toolchain | Result |
| --- | --- | --- | --- |
| CPU Release | `fwk394` | g++ 13.3, CMake 3.28.3, Ninja 1.11.1, alpaka `eb677e8d7f70d79b35df25d9002f94ba669b6f44` | **323/323 passed** |
| CUDA Release | `fwk394`, NVIDIA RTX 2080 Ti `sm_75` | same pins | **389/389 passed** |
| HIP Release | `hal`, AMD Radeon RX 7900 XTX `gfx1100` | ROCm 7.2, hipcc 7.2.26015, same pins | **380/380 passed** |
| SYCL Release | `hal`, Intel Arc A770 `spir64` | oneAPI 2026.2, icpx 2026.2.0, same pins | **397/397 passed** (77 sycl-labelled) |

No test is skipped. CTest reports `100% tests passed, 0 tests failed` for each
suite, and a compiled backend without a usable device fails its cases instead of
skipping. The CPU, CUDA, and HIP totals include the AoS attribute matrix, the
attribute helpers, the registration-site subset case, and the record-pipeline
cases on top of the PR #63 and PR #65 additions; the SYCL full-suite total is
the complete `hal` build, and its `sycl`-labelled subset is 77/77. PR #63 adds the `[config]` bin-path rejection case,
the `[kernel-axes]` rejection case, and the CUDA/HIP typed routed-axis cases on
top of the P07b 7/7; those cases are included in the totals above. PR #65 adds
the six `[scheduler]` cases (five `[scheduler][debug]` and one
`[scheduler][release]` case in `tests/integration/test_scheduler_validation.cpp`,
registered in every backend suite) on top of the PR #63 public-Flow cases;
they are included in the totals above
and pass 6/6 on CPU, CUDA, and HIP. PR #69 registered the 71-case SYCL matrix on
`hal` Arc A770 (33 in-process Catch2 cases plus 24 particle CLI cases, 3
cross-backend exact-histogram cases, 10 photon CLI cases, and 1 photon output
case); the environment-scoped `test_photon_scattering_sycl_fp64` case brought
the sycl-labelled count to 72, which is historical. The AoS-over-SoA matrix,
the registration-site subset, and the record-pipeline cases extend the SYCL
build to the 397-case full suite above. The `ffe4074` totals 249/249, 310/310
and 301/301, the `88de14f` totals 247/247, 307/307 and 298/298, the `56c11ac`
totals 246/246, 306/306 and 297/297, and the earlier `a610281` totals 239/239,
299/299 and 290/290 are historical.
`hal` reports `sharing=unknown` and `fwk394` is shared, so the runs are
correctness results, not performance claims. GitHub-hosted CI runs no SYCL
CTest: the `sycl-compile` job is the compile-only gate and makes no SYCL
runtime claim. H100 was not used and no H100
compile or run is recorded.

## HAL acceptance record

The HAL record covers the `d71935c` device suites. Platform and pins:

| Item | Value |
| --- | --- |
| code main (JUnit run) | `d71935c` (the then-current main `9fb15f1e6ce13f252b31d4cb60cc410862d56511` is newer) |
| alpaka | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (3.0.0) |
| Catch2 | 3.4.0 |
| CMake / Ninja | 3.28.3 / 1.11.1 |
| CPU compiler | g++ 13.3 |
| CUDA | 13.4, `sm_80`, NVIDIA A30 24 GiB on `hal`, driver 615.71.09 |
| ROCm | 7.2, hipcc 7.2.26015, `gfx1100`, AMD Radeon RX 7900 XTX on `hal` |
| oneAPI | 2026.2, icpx 2026.2.0, `spir64`, Intel Arc A770 on `hal` |
| C++ / build | C++20, Release, `-Dalpaka_COMPILE_PEDANTIC=ON` |

Commands:

```sh
# CPU
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=ON -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build-cpu -j 14
ctest --test-dir build-cpu --output-on-failure --no-tests=error

# CUDA, hal A30
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build-cuda -j 14
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-cuda \
  --output-on-failure --no-tests=error

# HIP, hal RX 7900 XTX
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_HIP=ON -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dalpaka_DEP_OMP=OFF -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build-hip -j 14
ROCR_VISIBLE_DEVICES=0 ctest --test-dir build-hip \
  --output-on-failure --no-tests=error

# SYCL, hal Arc A770
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGRAPHFLOW_ENABLE_SYCL=ON -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build-sycl -j 14
ctest --test-dir build-sycl --output-on-failure --no-tests=error -L sycl
```

Counts and no-skip: CPU **323/323**, CUDA **389/389** (`fwk394` RTX 2080 Ti
`sm_75`), HIP **380/380**, SYCL **397/397** full suite on the Arc A770, of
which 77 carry the `sycl` label; each log ends with `100% tests passed, 0
tests failed`. No label or configuration excluded a registered CPU, CUDA, or
HIP case from those runs. The retained JUnit XML reports 246 CPU, 306 CUDA,
and 297 HIP with **0 failures and 0 skips**; these JUnit counts were captured
at `d71935c`, well before the current totals (323/323, 389/389, 380/380,
397/397) of the tree at `25c73f3`, which add the P02b `ChunkLedger` overflow
test, the PR #63 public-Flow cases, the PR #65 scheduler cases, the PR #69
SYCL matrix, the AoS attribute matrix, the registration-site subset, and the
record-pipeline cases on top of them. Raw CTest logs are outside the
repository.

## P07 exclusive performance result

P07 at `64d4a1e11949886c7147c082cc386edde4bdb01f` (merged as `a610281`), one
exclusive `gpu-v100-int` allocation on `rosi_v100` (`gv001`): Tesla
V100-SXM2-32GB, driver 570.158.01, CUDA 12.6.85, `sm_70`, Release, three
repetitions per case, `N=1e8`, `C=8192`, `P=16384`, seed 42.

| Q | median wall s | IQR wall s | median particles/s |
| ---: | ---: | ---: | ---: |
| 1 | 191.990 | 2.315 | 5.209e5 |
| 2 | 8.710 | 0.535 | 1.148e7 |
| 8 | 7.690 | 0.045 | 1.300e7 |

Every run printed 260 histogram rows summing to exactly 100,000,000; the sorted
rows hash to
`21ab0aeb9ec72d380af2ce6e9631f8fa7b9111d24ff7831f272aa9f6a7be9dab`. The
frozen target is 13,000,000 particles/s at Q=8 with a 10 percent tolerance
(floor 11,700,000); the measured median is 13,003,901, so it passes. The
sustained D2D copy reference is 409.733 GB/s at 3,221,225,472 B. No H100 was
used.

## P07b Q/C re-run and kernel axes

P07b, merged as `56c11acbca6d10ea82fd85d213abcb4f9050dbf8`, re-ran the Q/C
matrix and the opt-in kernel-axis matrix on an exclusive `rosi_v100`
allocation. The Q=8 median is `12,987,013` particles/s at `N=1e8`, `C=8192`,
`P=16384`; the 13,000,000 target with its 10 percent floor (11,700,000) still
passes, and all nine Q runs print 260 rows summing to exactly 100,000,000. The
kernel-axis matrix covers `shared|direct` scatter, `dense|hash` bin paths and
the balanced/one-hot/skewed/sparse distributions at kernel level: all 30 runs
exited 0 with histogram sums equal to `N`. The tables and provenance are in
[benchmarks.md](benchmarks.md#p07b-qc-re-run-and-kernel-axes-main-56c11ac).
P07b did not re-measure the P03 CPU kernel timings; those remain the only
isolated-kernel provenance. No H100 was used.

## TSAN status

The last full integrated TSAN CTest is on `rosi_v100` at
`f3ef3552c60c0cfe0d202a73a08b808746ad11ce`: **210/210 passed, 0 reports**,
202.83 s, non-PIE Debug with `alpaka_EXEC_CpuOmpBlocks=OFF`,
`alpaka_DEP_OMP=OFF`, and `GRAPHFLOW_ENABLE_OPENMP=OFF`. With the default
CpuOmpBlocks plus OpenMP the same suite fails 105/210 with reports inside the
upstream alpaka async CPU queue, not graphFlow. That 210-case suite predates the
current 323-case suite, so no full TSAN result exists for
`25c73f31c736b1c5f765003ff8815d2a24c25808`.

The hosted `tsan` job was added at
`2fefe54c6d32e9156be5c08ea497119d0bacc21e` (P10 branch commit
`cf3bc8a72942b01c4a1ed32e4db7b260e22cb750`) and runs the same accepted
configuration. The P10 fwk394 evidence at `cf3bc8a` is ASAN+UBSAN 236/236,
error-path leak 1/1, and TSAN 235/235 with 0 reports; the hosted `tsan` runtime
was not observed. `hal` still aborts ThreadSanitizer at startup with
`FATAL: ThreadSanitizer: unexpected memory mapping`, including for a non-PIE
binary, so `hal` is not a TSAN host. No H100 was used.

## Historical validation totals

These totals are historical and are not the current state:

| Configuration | Result |
| --- | --- |
| CPU CTest at `f3ef355` | **210/210 passed** (OMP=ON Release) |
| TSAN CTest at `f3ef355` | **210/210 passed, 0 reports**, 202.83 s |
| CUDA CTest at `f3ef355` on `hal` A30 `sm_80` | **264/264 passed**, CUDA 13.4 |
| HIP CTest at `f3ef355` on `hal` RX 7900 XTX `gfx1100` | **255/255 passed**, ROCm 7.2 |
| HIP CTest at `b972eaa` | **245/245 passed**; 43 HIP cases |
| HIP CTest at `58bb514` | **238/238 passed** (first real AMD GPU run) |
| CPU CTest at `2ad7d86` on `rosi_v100` | **202/202 passed** (OMP=ON Release) |
| CUDA CTest at `2ad7d86` on `rosi_v100` V100 `sm_70` | **254/254 passed** (CUDA 12.6.85) |
| CPU / CUDA Debug at `rosi_v100`, pre-hash | **195/195** / **247/247** (not re-run after the hash cases) |
| CPU / CUDA / HIP Release at `561b9a4c` | **255/255** / **316/316** (A30 `sm_80`) / **307/307** |
| SYCL sycl-labelled at `9ace795` | **72/72** (`ctest -L sycl`) |
| Device failure cases at `10c1733770e16ebaecc74513034e6d327924503c` | CUDA 277/277 on `hal` A30, HIP 268/268 on `hal` RX 7900 XTX |
| Historical integrated CPU / CUDA on fwk394 / hal | **194/194** / **246/246** |
| FrameSpec revision `327f74f` | CPU 82/82, CUDA 86/86 |
| PR-A graph API core at `60a34302` | CPU 97/97, CUDA 101/101 |
| T3b | CPU 170/170, CUDA 206/206 |

The source-admission/stall/cross-backend branch totals 205/205, 207/207, and
210/210 are historical intermediate counts on merged branches. CPU OpenMP with
passive spinning (`GOMP_SPINCOUNT=0`) at `N=1e5`, `C=128`/`P=1024`, `Q=1`:
OMP=1 2.88 s, OMP=2 2.38 s (best), OMP=4 2.44 s, OMP=8 2.62 s; at `N=1e6`
OMP=2 Q1 23.71 s, OMP=2 Q8 4.46 s. All sums are exact.

The V100 `sm_70` build uses the explicit concept-parameter workaround in the
transform, partition, scatter, and hash-scatter kernels and the test hook
adapters. `GRAPHFLOW_ENABLE_DEBUG_VALIDATION` defaults to `ON` for
`CMAKE_BUILD_TYPE=Debug` and `OFF` for `Release` and `RelWithDebInfo`. Release
has no per-event payload checks; Debug compiles them in. Host setup validates
the configuration once, and genuine physics failures stay on the `StatusCode`
channel in both builds.

H100 was not used. The earlier JURECA `jrc0880.jureca` (4x H100 PCIe 80GB,
SM 9.0) run is preliminary CUDA evidence only (unknown gateway sharing flag;
the recorded run observed external load while the client was stale), so no H100
performance claim is made.

## HIP runtime validation on hal (historical record)

The current HIP total is **380/380** at
`25c73f31c736b1c5f765003ff8815d2a24c25808` (see the current-state block); the
previous **307/307** at `561b9a4c`, **301/301** at `ffe4074`, **298/298** at
`88de14f`, **297/297** at `56c11ac` and the **290/290** at `a610281` are
historical. The record below is the
original HIP runtime gate, measured on
`b972eaa3bd778b81f2dca505d3662df710792bd9` (HIP runtime merge, PR #35).
Platform: `hal`, AMD Radeon RX 7900 XTX (`gfx1100`), ROCm 7.2.0, `amdclang++`
(Clang 22.0.0), offload arch `gfx1100`, C++20 Release, alpaka
`eb677e8d7f70d79b35df25d9002f94ba669b6f44` (3.0.0). The run is recorded in
`/home/agent/final-hip-b972eaa/ctest.log`. The earlier HIP runtime total of
238/238 at `58bb514def3f3c956126ab7c34d0de74a45f679e` is historical (pre the
final CPU case set).

| Configuration | Result |
| --- | --- |
| HIP CTest (`ROCR_VISIBLE_DEVICES=0`, Release) on `b972eaa` | **245/245 passed** (100%, `ctest.log`) |
| HIP cases | **43**: 8 typed Flow, 24 particle CLI, 10 photon CLI, 1 photon output |
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

## Builds (historical)

The current builds are in the current-state block. Historical build records:

| Configuration | Result |
| --- | --- |
| `rosi_v100` CPU Release / Debug (OMP=ON) | configure and build pass; historical CTest 202/202 / 195/195 |
| `rosi_v100` CUDA Release / Debug (V100 `sm_70`) | configure and build pass; historical CTest 254/254 / 247/247 |
| fwk394 CPU Release | configure and build pass; historical CTest 194/194 on the integrated main |
| `hal` CUDA Release (`CMAKE_CUDA_ARCHITECTURES=80`) | configure and build pass; historical CTest 246/246 |
| `hal` HIP Release (`gfx90a`, `amdclang++`, `/opt/rocm`) | compile-only gate passed; no AMD runtime device was visible at that revision |
| `hal` HIP Release (`gfx1100`, RX 7900 XTX, ROCm 7.2) | configure, build, and CTest 245/245 at `b972eaa` |
| ASAN+UBSAN Debug (hosted CI) | historical 210/210 on `gf/cross-backend-equality`; 202/202 at `2ad7d86` |
| TSAN Debug | historical full run 210/210, 0 reports, 202.83 s at `f3ef355`; see the TSAN status section |

## Tests (historical)

The current suites are CPU 323/323, CUDA 389/389, HIP 380/380, and SYCL 397/397 (77 sycl-labelled).
The previous 249/249, 310/310 and 301/301 at `ffe4074`, 247/247, 307/307 and
298/298 at `88de14f`, the 246/246, 306/306 and 297/297 at `56c11ac`, and the
earlier 239/239, 299/299 and 290/290 at `a610281` are historical. The table below records the earlier totals, including the
intermediate branch counts:

| Configuration | Result |
| --- | --- |
| `rosi_v100` CPU CTest (OMP=ON Release) at `f3ef355` | **210/210 passed** |
| `rosi_v100` TSAN CTest at `f3ef355` | **210/210 passed, 0 reports**, 202.83 s |
| `rosi_v100` CPU CTest at `2ad7d86` | **202/202 passed** |
| `rosi_v100` CUDA CTest (V100 `sm_70`, Release) at `2ad7d86` | **254/254 passed** |
| `rosi_v100` CPU / CUDA Debug, pre-hash | **195/195** / **247/247** (not re-run after the hash cases) |
| `f3ef355` CUDA on `hal` A30 `sm_80` | **264/264 passed**, CUDA 13.4, 400.98 s |
| `f3ef355` HIP on `hal` RX 7900 XTX `gfx1100` | **255/255 passed**, ROCm 7.2, 99.44 s |
| `b972eaa` HIP on `hal` | **245/245 passed**; 43 HIP cases |
| `gf/p1-admission-smoke` | CPU 205/205 |
| `gf/stall-diagnostics` | CPU 207/207 |
| `gf/cross-backend-equality` | CPU 210/210 |
| Historical CPU CTest on fwk394 | 170/170 at T3b; 194/194 on the integrated main (T5) |
| Historical CUDA CTest on hal (`sm_80`) | 206/206 at T3b; 246/246 on the integrated main (T5) |
| Historical HIP compile-only gate (`gfx90a`) | compiled; no AMD device was visible |
| ASAN+UBSAN CTest | not recorded for the integrated revision |

The two stall diagnostics cases added on `gf/stall-diagnostics` are the
`[stall]` and `[stats]` cases in `tests/unit/test_flow_execution.cpp`
(`"CPU scheduler stall snapshot is aggregate and topology free"` and
`"CPU completed-run scheduler stats satisfy the peak invariants"`). The first
drives a forced no-progress stall and asserts the thrown
`SchedulerStallError` reports only aggregate key=value counters, with node
names, ids, and routes kept out of the message and limited to the opt-in detail
callback. The second runs a completed source-to-sink graph and asserts the
`SchedulerStats` peak invariants and that the on-demand `snapshot()` mirrors the
completed-run peaks with no aggregate work left. The matching CUDA and HIP
cases register in `tests/integration/test_flow_execution_{cuda,hip}.cpp`.

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
historical `rosi_v100` CPU CTest passed 202/202 (OMP=ON Release) at `2ad7d86`
(210/210 on `gf/cross-backend-equality`, 207/207 on `gf/stall-diagnostics`,
205/205 on `gf/p1-admission-smoke`) and the V100 `sm_70` CUDA CTest passed
254/254 in Release with the explicit concept-parameter workaround (264/264 on
`hal` A30 `sm_80`, CUDA 13.4, 400.98 s, at `f3ef355`); the `rosi_v100`
33-invocation baseline comparison is not recorded.

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
| `rosi_v100` CPU (OMP=ON Release) | `ctest --output-on-failure` | **202/202 passed** at `2ad7d86` (historical); 210/210 on `gf/cross-backend-equality` (207/207 on `gf/stall-diagnostics`, 205/205 on `gf/p1-admission-smoke`) |
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
runtime validation is recorded above. H100 was not used for this validation, so no H100 performance claim
is made and H100 remains preliminary; the earlier JURECA `jrc0880.jureca` run
is preliminary CUDA evidence only (unknown gateway sharing flag; the recorded
run observed external load while the client was stale). The `rosi_v100` CPU OpenMP run (passive,
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
payload budget. The integrated T13 measurement is recorded below as historical;
the current performance record is the P07 result at the top of this document.

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

## Integrated T13 measurement (historical)

The integrated T13 benchmark is historical. The table below is the supplied T13
evidence for that revision, not a fresh run in this worktree. `hal` reports
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

CPU/CUDA/HIP histogram equality, `N=1e6`, `C=128`, `P=1024`, seed 42,
`Q=1/2/8`: automated on `gf/cross-backend-equality`
(`751c8a8cbad9de1a2e71cef57b81147ff071dd51`) by the
`example.particle_feedback.cross_backend.Q{1,2,8}` CTest cases, which run the
real `graphFlow_particle_feedback` binary once per compiled backend with
identical seed/N/C/P/Q, sort the `weight,count` rows, and require exact row
equality against the first backend plus a count sum of N per backend. CUDA and
HIP cases register only when their backend is compiled, and a compiled backend
without a usable device fails the case instead of being skipped. This is the
evidence for P0 cross-backend equality, which is done. It comes from two
device build pairs, not one three-backend run: CUDA and HIP are mutually
exclusive in a single build, and the CPU backend stays enabled in accelerator
builds. The CPU+CUDA build pair and the CPU+HIP build pair each ran the three
Q cases 3/3 with exact row equality against the CPU reference: same seed,
`N`, `C`, `P`, and `Q`, sorted `weight,count` rows identical, and every count
sum equaled `N`. Measured CPU total at `751c8a8`: 210/210 (OMP=ON Release,
pedantic), with all three Q cases passing exact-row equality. The full device
suites were executed at `f3ef355`: CUDA 264/264 on `hal` A30 `sm_80` (CUDA
13.4, 400.98 s) and HIP 255/255 on `hal` RX 7900 XTX `gfx1100` (ROCm 7.2,
99.44 s), both on the shared `hal` host, correctness only.

fwk394, CPU, `C=1024`, `P=512`, `Q=1`: `N=1e6` took 52.75 s, histogram sum
1,000,000. The fwk394 CUDA sm_75 build fails with an nvcc 12.5 internal
compiler error, so no fwk394 CUDA run is claimed.

## N=1e8 acceptance on rosi_v100 (historical)

This is the f3ef355 acceptance run; the current result is P07 at `64d4a1e`
above. The P0 acceptance run is re-recorded on `rosi_v100` at main `f3ef355`.
Platform: V100 SXM2-32GB, CUDA 12.6.85, `sm_70`, Release, with the explicit
concept-parameter workaround. Parameters: `C=8192`, `P=16384`, `T=128`, seed
42, max-visits 64. The `P=16384` payload (3,221,225,472 B) fits the V100
SXM2-32GB budget; the A30 measurement forced `P=16128`. Three repetitions.

| N | Q | Wall time | Histogram sum | particles/s |
| --- | ---: | ---: | ---: | ---: |
| 100,000,000 | 1 | 194.685 s | 100,000,000 | ~5.14e5 |
| 100,000,000 | 2 | 9.093 s | 100,000,000 | ~1.10e7 |
| 100,000,000 | 8 | 7.676 s | 100,000,000 | 1.30e7 |

The sorted histogram has 260 rows, the count column sums to exactly
100,000,000 for every Q, and the sorted `weight,count` rows hash to
SHA-256 `89dc7fb9962e898fe2f1fe0116ad98fe2df9dceb77d33aa890a56467615d02e9`.
Peak memory is 5702 MiB. This is the recorded P0 acceptance run; the hal A30
numbers above stay labelled shared-host baselines. H100 was not used, so no
H100 claim is made.

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
The tree-wide format sweep (T14b) is complete. Hosted CI is
`.github/workflows/ci.yml` and is documented in `docs/ci.md`: CPU CTest and
ASAN+UBSAN run there, CUDA and HIP are compile-only, and `reuse lint` and the
`format` gate run.
