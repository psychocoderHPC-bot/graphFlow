<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Continuous integration

graphFlow runs CI on GitHub Actions with GitHub-hosted `ubuntu-latest`
runners and public container images. The workflow is
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). There are no
self-hosted runners, no project GPU labels, and no GitLab pipeline.

## Triggers and controls

- `push` to `main`, `pull_request` targeting `main`, and `workflow_dispatch`.
  Every job below runs for pushes and pull requests; `workflow_dispatch`
  re-runs the same set manually.
- `permissions: contents: read`; the workflow needs no secrets.
- A `concurrency` group cancels superseded runs for the same pull request or
  ref.
- Every job sets `timeout-minutes`.

## Repository policy

The repository is public and the README carries the CI badge. Changes land
through a feature branch and a pull request targeting `main`; the PR runs the
hosted jobs above and needs review before merge. Branch protection for `main`
is not enabled yet. Enabling it is an operator action; no branch-protection
setting is changed from this repository.

## Hosted matrix

| Job | Runner | Container | Backend | Build | Runs tests |
| --- | --- | --- | --- | --- | --- |
| `cpu` | `ubuntu-latest` | none | CPU | Release and Debug | yes, CTest |
| `sanitizers` | `ubuntu-latest` | none | CPU | Debug | yes, ASAN+UBSAN |
| `leak` | step of `sanitizers` | none | CPU | Debug (ASAN+UBSAN) | yes, `graphflow.error_path_leak_check`, `-L leak` |
| `tsan` | `ubuntu-latest` | none | CPU | Debug, non-PIE | yes, CTest |
| `cuda-compile` | `ubuntu-latest` | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | CUDA | Release, `sm_70` | no, compile only |
| `hip-compile` | `ubuntu-latest` | `rocm/dev-ubuntu-24.04:7.2` | HIP | Release, `gfx90a` | no, compile only |
| `sycl-compile` | `ubuntu-latest` | `intel/oneapi-hpckit:2025.3.0-0-devel-ubuntu24.04` | SYCL (Intel GPU only) | Release | no, compile only |
| `reuse` | `ubuntu-latest` | none | none | none | `reuse lint` |
| `format` | `ubuntu-latest` | none | none | none | `clang-format --dry-run --Werror` |
| `icpx-cpu` | `ubuntu-latest` | `intel/oneapi-hpckit:2025.3.0-0-devel-ubuntu24.04` | CPU | Release | yes, CTest |

The CPU and sanitizer jobs install `cmake`, `ninja-build`, `g++`, and `catch2`
from the Ubuntu 24.04 archive and clone alpaka from source. The CUDA, HIP, and
oneAPI jobs do the same inside their container; they install `git` before
`actions/checkout` because the base images do not guarantee it.

## Test counts

| Target | CTest cases | Where the result comes from |
| --- | ---: | --- |
| CPU, Release and Debug | **270** at `3b90ffd`; 249 at `ffe4074`, 247 at `88de14f`, 246 at `56c11ac`, 239 at `a610281`, 210 at `f3ef355` and 202 at `2ad7d86` are historical | the hosted `cpu` jobs |
| CPU, ASAN+UBSAN Debug | **270** current suite; 249 at `ffe4074`, 247 at `88de14f`, 246 at `56c11ac`, 239 at `a610281`, 210 at `f3ef355` and 202 at `2ad7d86` are historical | the hosted `sanitizers` job |
| CPU, ASAN+UBSAN Debug, fwk394 | 236/236 at `cf3bc8a` on `gf/p10-sanitizers` (historical) | local run; the hosted `sanitizers` runtime was not observed |
| CPU, leak (error-path, `-L leak`) | 1/1 at `cf3bc8a` on `gf/p10-sanitizers` (historical) | local run; the hosted `leak` step runtime was not observed |
| CPU, TSAN Debug non-PIE, no OMP executor | 235/235, 0 reports at `cf3bc8a` (historical); 210/210, 0 reports, 202.83 s at `f3ef355` on `rosi_v100` (historical; no full TSAN at `ffe4074`) | local `fwk394` run; the hosted `tsan` runtime was not observed |
| CPU, icpx Release | 247/247 at `88de14f` on `hal` with icpx 2026.2.0 and 2025.3.3, `OMP_NUM_THREADS=4`; the `ffe4074` re-validation used g++ 13.3 on `fwk394`; the `561b9a4c` re-validation used g++ 13.3 on `fwk394` | local runs; the hosted `icpx-cpu` job runs the same suite |
| CUDA device runtime | **334** at `3b90ffd` on `fwk394` RTX 2080 Ti `sm_75`; 316 at `561b9a4c` on `hal` A30 `sm_80`, 310 at `ffe4074`, 307 at `88de14f`, 306 at `56c11ac`, 299 at `a610281`, 264 at `f3ef355` and 254 at `2ad7d86` are historical | `hal` device run; the hosted job compiles only |
| HIP `gfx1100` | **325** at `3b90ffd` on `hal` RX 7900 XTX, ROCm 7.2; 307 at `561b9a4c`, 301 at `ffe4074`, 298 at `88de14f`, 297 at `56c11ac`, 290 at `a610281`, 255 at `f3ef355` and 245 at `b972eaa` are historical | `hal` device run; the hosted job compiles only |
| SYCL Intel GPU | **342/342** at `3b90ffd` on `hal` Arc A770, oneAPI 2026.2, full suite (75 sycl-labelled); the 72/72 sycl-labelled at `9ace795` and 71/71 at `9fb15f1` are historical | `hal` device run; the hosted `sycl-compile` job builds without a device and produces no CTest result |

The CPU suite registers 270 cases at `3b90ffd`; the 249 count at `ffe4074`, the
247 count at `88de14f` and the 246 count at `56c11ac` are historical. Release compiles the per-event
checks out, Debug compiles them in, and the `[photon][debug]` cases register in
both modes. The opt-in `[diagnostics]` photon cases are excluded from the
default suite. The `fwk394` sanitizer rows are the historical local P10
evidence run at `cf3bc8a` on `gf/p10-sanitizers` (shared host, no H100):
ASAN+UBSAN 236/236, the error-path leak case 1/1, and TSAN 235/235 with 0
reports in the accepted non-PIE, OMP-off, CpuOmpBlocks-off configuration. The 235 TSAN count
and the 236 ASAN+UBSAN count differ because the error-path leak case registers
only when `GRAPHFLOW_ENABLE_ERROR_PATH_LEAK_CHECK` is on (the sanitizer job),
so it is part of the ASAN+UBSAN run but not of the separate TSAN build. The
hosted `sanitizers` (including its `leak` step) and `tsan` job runtimes were
not observed for this branch. The CUDA and HIP counts are device-run totals, not hosted CTest
results: the hosted `cuda-compile` and `hip-compile` jobs produce no CTest
result. The HIP and SYCL runtime totals are the recorded `hal` runs; see
[HIP compile-only](#hip-compile-only) and
[SYCL compile-only](#sycl-compile-only).

The N=1e6 CPU smoke (`example.particle_feedback.smoke.N1000000`) and the two
source-admission cases register unconditionally in the ordinary CTest job, so
hosted CPU CI runs them without a skip; there is no label or configuration that
excludes them. The two stall-diagnostics cases added on `gf/stall-diagnostics`
carry the `[stall]`/`[stats]` and `[diagnostics-check]` tags, which the default
suite does not exclude (only the opt-in photon `[diagnostics]` tag is excluded),
so they run in the same ordinary CPU CTest job without a skip. The three
cross-backend equality cases added on `gf/cross-backend-equality`
(`example.particle_feedback.cross_backend.Q1`, `.Q2`, `.Q8`,
`N=1e6`, `C=128`, `P=1024`, seed 42) carry the
`example;cli;cross-backend;long;cpu` labels (plus `cuda`/`hip` only when those
backends are compiled), which the default suite does not exclude, so they run
in the same ordinary CPU CTest job without a skip; CUDA and HIP cases register
only when their backend is compiled, and a compiled backend without a usable
device fails the case instead of being skipped, so the CPU-only hosted build
still exercises the CPU path and a mismatch fails the job. CUDA and HIP
runtime validation of the equality cases stays a manual
`rosi_v100`/`hal` check.

The two stall-diagnostics cases cover the P1 stall work on CPU: a forced
no-progress stall asserts that the thrown `SchedulerStallError` reports
aggregate key=value counters with no node names, ids, or routes, and a
completed run asserts the `SchedulerStats` peak invariants and the empty
end-of-run aggregate. The same two cases register on CUDA and HIP; the
intermediate 258 and 249 device counts are historical.

T14b re-ran all three suites on the formatted tree at the then-current counts
(historical): CPU Release 202/202 on `rosi_v100` (205/205 with the
source-admission cases and N=1e6 smoke; 207/207 with the stall diagnostics;
210/210 with the cross-backend equality cases), CUDA `sm_70` 254/254 on
`rosi_v100` (258 and 264 on the intermediate branches), and HIP `gfx1100`
245/245 on `hal` (249 and 255 on the intermediate branches). The full CUDA and
HIP device suites were executed at `f3ef355`: CUDA 264/264 on `hal` A30
`sm_80` (CUDA 13.4, 400.98 s) and HIP 255/255 on `hal` RX 7900 XTX `gfx1100`
(ROCm 7.2, 99.44 s), correctness only. The current suites are 270/270,
334/334 and 325/325 at `3b90ffd`, and 342/342 SYCL at `3b90ffd` (75 sycl-labelled); the previous
255/316/307 at `561b9a4c` (72/72 sycl-labelled at `9ace795`),
249/310/301 at
`ffe4074`, 247/307/298 at
`88de14f`, 246/306/297 at `56c11ac`, and the 239/299/290 at `a610281` are
historical.
The format sweep changed whitespace, line breaks, include and `using` order, and
configured numeric separators only.

## Pins

| Component | Pin | Source |
| --- | --- | --- |
| alpaka | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` | `alpaka-group/alpaka`, the `3.0.0` tag |
| Catch2 | 3.4.0 | Ubuntu 24.04 `catch2` package |
| CMake | 3.28.3 | Ubuntu 24.04 `cmake` package |
| Ninja | 1.11.1 | Ubuntu 24.04 `ninja-build` package |
| GCC | 13 | Ubuntu 24.04 `g++` package |
| CUDA | 12.6.3 | `nvidia/cuda:12.6.3-devel-ubuntu24.04` |
| ROCm | 7.2 | `rocm/dev-ubuntu-24.04:7.2` |
| oneAPI | 2025.3 | `intel/oneapi-hpckit:2025.3.0-0-devel-ubuntu24.04` |
| clang-format | 20.1.8 | `pip install clang-format==20.1.8` |
| REUSE | 6.3.0 | `pip install reuse==6.3.0` |

The checkout step verifies that the cloned alpaka `HEAD` equals the pin before
configure runs.

## CPU

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure --no-tests=error \
  --output-junit ctest-cpu-Release.xml
```

The matrix repeats this for `Debug`. CTest reports are uploaded as artifacts.
`--no-tests=error` makes a missing test suite fail the job, and
`alpaka_COMPILE_PEDANTIC=ON` turns on alpaka's strict warning flags.

## Sanitizers

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DGRAPHFLOW_ENABLE_ERROR_PATH_LEAK_CHECK=ON \
  -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure --no-tests=error
ctest --test-dir build --output-on-failure --no-tests=error -L leak
```

The `sanitizers` job runs ASAN+UBSAN over the full host scheduler suite, then a
dedicated `leak` step: `GRAPHFLOW_ENABLE_ERROR_PATH_LEAK_CHECK=ON` registers
`graphflow.error_path_leak_check`, which runs every `[failure]` and `[status]`
case in one process so LeakSanitizer (already enabled by ASAN) reports at exit.
`LSAN_OPTIONS=exitcode=66` turns a leak into a nonzero exit.

A separate `tsan` job runs the host scheduler under ThreadSanitizer on
`ubuntu-latest`: Debug, non-PIE (`-fno-pie`/`-no-pie`),
`GRAPHFLOW_ENABLE_OPENMP=OFF`, `alpaka_EXEC_CpuOmpBlocks=OFF`, and
`alpaka_DEP_OMP=OFF`. That is the accepted clean configuration: with
CpuOmpBlocks plus OpenMP the same suite reports races inside the upstream
alpaka CPU queue, not in graphFlow. `TSAN_OPTIONS=exitcode=66` makes a report
fail the job. The recorded blocker on hal (runtime mapping failure) was
resolved on the exclusive `rosi_v100` host, where the full integrated TSAN
CTest passed 210/210 with 0 reports in 202.83 s at
`f3ef3552c60c0cfe0d202a73a08b808746ad11ce` in this configuration; that run is
historical and the current 270-case suite has no full TSAN run. The P10 run on
`fwk394` at `cf3bc8a` is 235/235 with 0 reports, and the hosted `tsan` job
runtime was not observed. No H100 was used. No full TSAN run exists for the
current 270-case suite.

### Status tests

Current `main` passes the five former `[status]` cases in
`tests/unit/test_flow_failure.cpp` in both Release and Debug. An earlier tree
failed them on Debug and ASAN+UBSAN because an uninitialized payload field
reached the RNG stream-id assertion in
`include/graphFlow/rng/ParticleRng.hpp`; that tree is not the base of this
workflow. The workflow adds no test changes.

## CUDA compile-only

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DGRAPHFLOW_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j "$(nproc)"
```

The job only compiles. It does not call `nvidia-smi`, does not run CTest, and
does not run an example. The hosted runner has no NVIDIA GPU, so a runtime
claim would be false. The compile gate covers nvcc device compilation of the
CUDA backend, the CUDA test translation units, and the CUDA example.

`sm_70` (Volta) is a conservative baseline for the hosted compile gate. The
recorded runtime validation on `rosi_v100` uses V100 `sm_70` with CUDA 12.6.85;
the historical A30 run uses `sm_80` with CUDA 13.4. This job makes no runtime
claim. CUDA 12.5 is not used because it hits an nvcc internal compiler error in
`HashScatterKernel.hpp`.

### `sm_70` compiler workaround

Current `main` contains the explicit-template-parameter workaround for the
nvcc 12.6 `check_name_hiding_by_template_parameters` internal compiler error.
The transform, partition, scatter, and hash-scatter kernels declare the
trailing `IMdSpan` parameters explicitly, and the test hook adapters route
`GraphRunner::Queue` and `Batch` through small templates. The previous hosted
run on the pre-workaround tree failed at `flow/Execution.hpp(385)` while
compiling the examples. The current tree contains the fix, and the hosted CUDA
12.6.3 `sm_70` job compiles it.

## HIP compile-only

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DGRAPHFLOW_ENABLE_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dalpaka_DEP_OMP=OFF
cmake --build build -j "$(nproc)"
```

`alpaka_DEP_OMP=OFF` is required so alpaka does not add an OpenMP dependency
to the HIP build. The hosted runner has no AMD GPU, so the job stops after the
build: no CTest run and no example execution.

HIP runtime validation is separate. On `hal` with an AMD Radeon RX 7900 XTX
(`gfx1100`) and ROCm 7.2, the current HIP CTest run passes **325/325** at
`3b90ffd8962084d59ebc672f7848b093f2d99582`; the previous 307/307 at `561b9a4c`, 301/301 at `ffe4074`,
298/298 at `88de14f`, 297/297 at `56c11ac` and the 290/290 at `a610281` are
historical. The original
245/245 run with 43 HIP cases (8 typed Flow, 24 particle CLI, 10 photon CLI,
and 1 photon output case) was measured on
`b972eaa3bd778b81f2dca505d3662df710792bd9` and recorded in
`/home/agent/final-hip-b972eaa/ctest.log`, with `rocprof` kernel evidence.
`hal` reports `sharing=unknown`, so this is a correctness result and not a
performance claim. The GitHub-hosted job makes no HIP runtime claim.

## REUSE

```sh
python -m pip install --upgrade pip reuse==6.3.0
reuse lint
```

The `reuse` job checks every tracked file against `.reuse/dep5` and
`LICENSES/MPL-2.0.txt`. The blanket `Files: *` entry in `.reuse/dep5` covers
the workflow and documentation files added here; the previous hosted run
passed with every tracked file covered.

## oneAPI icpx CPU

The `icpx-cpu` job is part of the push and pull-request gate. It runs in the
same oneAPI 2025.3 container and with the same CPU-only `icpx` target that the
alpaka pin lists in its supported compiler matrix, and it exercises the CPU
backend with Intel's host compiler:

```sh
# The oneAPI image already initializes PATH, LD_LIBRARY_PATH and
# CMAKE_PREFIX_PATH, so the job does not source setvars.sh.
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure --no-tests=error \
  --output-junit ctest-icpx.xml
```

The CPU job does not set `alpaka_DEP_ONEAPI`; the separate `sycl-compile` job
owns the SYCL compile gate. This job is a host-compiler portability gate for
the CPU backend. `alpaka_COMPILE_PEDANTIC` stays off for this job: it adds `-Werror`,
and icpx (unlike GCC) warns on the brace-initialized `RouteTable{{1u}}` scalars
in the tests. The GCC CPU, sanitizer, and TSAN jobs keep the strict gate.

The job pins `OMP_NUM_THREADS=4`, matching the hosted runner's vCPU count. The
Intel OpenMP runtime (`-fiopenmp`) pays team-creation cost per region, and
graphFlow's host scheduler launches many small regions. With the default team
size of a large host (64 threads on `hal`) the N=1e6 smoke reached its 900 s
driver timeout; with four threads it finished in 59 s and the full suite
passed 247/247 in 506 s on the same loaded host (that run is at `88de14f`; the
`ffe4074` re-validation used g++ 13.3 on `fwk394`). Pinning the team size keeps
the gate stable without changing the tested executor.

## SYCL compile-only

The `sycl-compile` job sets `GRAPHFLOW_ENABLE_SYCL=ON` with `icpx` in the oneAPI
2025.3 container. The graphFlow option enables `alpaka_DEP_ONEAPI=ON` and
`alpaka_ONEAPI_IntelGpu=ON`, and disables the SYCL CPU, NVIDIA, and AMD device
kinds, so only the Intel GPU device kind is compiled. `alpaka_COMPILE_PEDANTIC`
stays off for the same icpx warning on the brace-initialized `RouteTable{{1u}}`
scalars in the tests that keeps it off for `icpx-cpu`.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$PWD/.alpaka" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DGRAPHFLOW_ENABLE_SYCL=ON \
  -Dalpaka_COMPILE_PEDANTIC=OFF
cmake --build build -j "$(nproc)"
```

The job only compiles the SYCL test translation unit and the examples. It does
not use `sycl-ls` as a device gate, does not run CTest, and does not execute an
example. The GitHub-hosted runner has no Intel GPU, so this job makes no SYCL
runtime claim. The validated device run is 342/342 (75 sycl-labelled) at
`3b90ffd8962084d59ebc672f7848b093f2d99582` on `hal` with an Intel Arc A770
(`spir64`, oneAPI 2026.2); the PR #69 matrix (33 in-process, 24 particle
CLI, 3 cross-backend, 10 photon CLI, 1 photon output) plus the
environment-scoped `test_photon_scattering_sycl_fp64` case is historical at
72/72 (`ctest -L sycl` at `9ace795`), and the AoS attribute matrix, the
registration-site subset, and the record-pipeline cases extend the build to the
current full suite. By default the Arc A770 lacks the SYCL `fp64` aspect, so the
oneAPI photon path runs fp32, and the opt-in case runs the double physics. The
hosted job never runs the opt-in case. AOT via `ocloc` is
not used: `ocloc` is present and a plain invocation does not crash, but it
rejects double, so the build stays JIT-compiled SPIR-V. See
[build.md](build.md#sycl-intel-gpu).

## Format

The tree is formatted with clang-format 20.1.8 and the repository
`.clang-format`. The `format` job checks tracked `.cpp` and `.hpp` files under
`include/`, `examples/`, and `tests/`:

```sh
python -m pip install --upgrade pip clang-format==20.1.8
mapfile -t files < <(
  git ls-files 'include/**/*.cpp' 'include/**/*.hpp' \
    'examples/**/*.cpp' 'examples/**/*.hpp' \
    'tests/**/*.cpp' 'tests/**/*.hpp'
)
clang-format --dry-run --Werror "${files[@]}"
```

The job fails on any diff and asserts the installed version string, so a newer
or older clang-format cannot silently change the result. The T14b sweep applied
this configuration to the tree; `main` is clean for the file list above.
Formatting is mechanical: it changes whitespace, line breaks, include and
`using` order, and numeric separators as configured by `.clang-format`. It does
not change program behavior.

## Not included

- Self-hosted runners or labels (`fwk394`, `a30`, `rocm`).
- CUDA, HIP, or SYCL execution; all three jobs are compile-only.
- H100 was not used for the sanitizer evidence, and no H100 claim is made.
- GitLab CI.
- H100 or any other specific GPU target: H100 was not used and no final H100
  run has been recorded, so no H100 performance claim is made.

## Local reproduction

The commands above match `docs/build.md`. To reproduce a job locally, clone
alpaka at the pin, set `GRAPHFLOW_ALPAKA_SOURCE_DIR` to that clone, and run the
same configure, build, and CTest commands. Container jobs need the matching
image and the extra `git` install before checkout.

## Status

The hosted workflow defines `cpu (Release)`, `cpu (Debug)`,
`sanitizers (ASAN+UBSAN)`, `tsan`, `cuda-compile`, `hip-compile`,
`sycl-compile`, `format`, and `reuse`. Each current CPU job runs 270 CTest cases (Release and Debug); the
`sanitizers` job runs ASAN+UBSAN over the same CPU suite plus its `leak` step.
The N=1e6 smoke, source-admission, stall-diagnostics, cross-backend equality,
visit-cap, pool-pressure, CLI-contract, kernel-axis, and in-flight cases run in
the ordinary CTest job without a skip. P07b contributes 7/7 of those cases.
The `format` job enforces the T14b sweep with clang-format 20.1.8. The `sm_70`
workaround clears the nvcc internal compiler error, so `cuda-compile` builds the CUDA sources and the CUDA example without a
GPU. `hip-compile` builds the HIP sources and the HIP example without an AMD
GPU. `sycl-compile` builds the Intel-GPU-only SYCL backend without a device;
it is compile-only and makes no SYCL runtime claim. `icpx-cpu` runs on every push and pull request; the local validation passes
the 270-case CPU suite (g++ 13.3 on `fwk394`; the icpx evidence at
`88de14f` passes the then-current 247-case suite). At `cf3bc8a`
on `gf/p10-sanitizers`, the historical fwk394 evidence is ASAN+UBSAN 236/236,
leak 1/1, and TSAN 235/235 with 0 reports (non-PIE, OMP off, CpuOmpBlocks off);
the hosted `tsan` and `leak` runtimes were not observed, and no H100 was used.
On the exclusive `rosi_v100` host at `f3ef355` the historical CPU CTest is
210/210 and the full integrated TSAN CTest is 210/210 with 0 reports (202.83 s,
non-PIE with `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`). The
current device totals are 334/334 CUDA on `fwk394` RTX 2080 Ti `sm_75`,
325/325 HIP on `hal` RX 7900 XTX, and 342/342 SYCL on `hal` Arc A770 (75
sycl-labelled); see `docs/acceptance.md`.

Container jobs set `shell: bash` because container jobs default to `sh`, which
rejects `set -o pipefail`.

Validation for the icpx job covered YAML parsing, `actionlint`, `bash -n` on
the embedded shell, the container image tag, and the alpaka pin. Validation for
the `sycl-compile` job covered YAML parsing and `actionlint`; no SYCL compile was
run locally and no GitHub-hosted runtime was observed. Locally on
`hal`, the CPU backend configured and built with icpx 2026.2.0 and with icpx
2025.3.3 (the same 2025.3 line as the container) against the pinned alpaka
checkout, and both `ctest` runs passed 247/247 with `OMP_NUM_THREADS=4` (that
run is at `88de14f`). The `ffe4074` re-validation ran the 249-case suite with
g++ 13.3 on `fwk394`; the `3b90ffd` re-validation ran the 270-case suite with
g++ 13.3 on `fwk394`. The hosted container revision 2025.3.0 was not run for
either, and 2025.3.3 is the closest available build of that compiler line. No
GitHub-hosted job runtime was observed for this branch.
