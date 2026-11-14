<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Continuous integration

graphFlow runs CI on GitHub Actions with GitHub-hosted `ubuntu-latest`
runners and public container images. The workflow is
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). There are no
self-hosted runners, no project GPU labels, and no GitLab pipeline.

## Triggers and controls

- `push` to `main`, `pull_request` targeting `main`, and `workflow_dispatch`.
- `permissions: contents: read`; the workflow needs no secrets.
- A `concurrency` group cancels superseded runs for the same pull request or
  ref.
- Every job sets `timeout-minutes`.

## Hosted matrix

| Job | Runner | Container | Backend | Build | Runs tests |
| --- | --- | --- | --- | --- | --- |
| `cpu` | `ubuntu-latest` | none | CPU | Release and Debug | yes, CTest |
| `sanitizers` | `ubuntu-latest` | none | CPU | Debug | yes, ASAN+UBSAN |
| `cuda-compile` | `ubuntu-latest` | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | CUDA | Release, `sm_70` | no, compile only |
| `hip-compile` | `ubuntu-latest` | `rocm/dev-ubuntu-24.04:7.2` | HIP | Release, `gfx90a` | no, compile only |
| `reuse` | `ubuntu-latest` | none | none | none | `reuse lint` |
| `format` | `ubuntu-latest` | none | none | none | `clang-format --dry-run --Werror` |
| `icpx-cpu` | `ubuntu-latest` | `intel/oneapi-hpckit:2025.3.0-0-devel-ubuntu24.04` | CPU | Release | manual dispatch only |

The CPU and sanitizer jobs install `cmake`, `ninja-build`, `g++`, and `catch2`
from the Ubuntu 24.04 archive and clone alpaka from source. The CUDA, HIP, and
oneAPI jobs do the same inside their container; they install `git` before
`actions/checkout` because the base images do not guarantee it.

## Test counts

| Target | CTest cases | Where the result comes from |
| --- | ---: | --- |
| CPU, Release and Debug | 210 on `gf/cross-backend-equality` (202 at `2ad7d86`); 210 = 202 + 2 source-admission + 1 N=1e6 smoke + 2 stall diagnostics + 3 cross-backend equality | the hosted `cpu` jobs |
| CPU, ASAN+UBSAN Debug | 210 on `gf/cross-backend-equality` (202 historical at `2ad7d86`) | the hosted `sanitizers` job |
| CUDA `sm_70` | 254 at `2ad7d86` (historical); 259 = 256 + 3 cross-backend equality cases (256 adds the two `gf/stall-diagnostics` stall diagnostics cases), not re-run on `rosi_v100` | `rosi_v100` device run; the hosted job compiles only |
| HIP `gfx1100` | 245 at `b972eaa` (historical); 250 = 247 + 3 cross-backend equality cases (247 adds the two `gf/stall-diagnostics` stall diagnostics cases), not re-run on `hal` | `hal` device run on `b972eaa`; the hosted job compiles only |

The CPU count is the same in Release and Debug. Release compiles the per-event
checks out, Debug compiles them in, and the `[photon][debug]` cases register in
both modes. The opt-in `[diagnostics]` photon cases are excluded from the
default suite. The CUDA and HIP counts are device-run totals, not hosted CTest
results: the hosted `cuda-compile` and `hip-compile` jobs produce no CTest
result. The HIP runtime total is the recorded `hal` run; see
[HIP compile-only](#hip-compile-only).

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
end-of-run aggregate. The same two cases register on CUDA and HIP, so their
device totals rise by two as well (256 and 247); the recorded device runs above
pre-date the stall work.

T14b re-ran all three suites on the formatted tree: CPU Release 202/202 on
`rosi_v100` (the 202-case historical suite; 205/205 includes the two
source-admission cases and the N=1e6 smoke added on
`gf/p1-admission-smoke`; 207/207 adds the two stall diagnostics cases on
`gf/stall-diagnostics`; 210/210 adds the three cross-backend equality cases on
`gf/cross-backend-equality`, measured on that branch, not re-run on
`rosi_v100`), CUDA `sm_70` 254/254 on `rosi_v100` (256 on
`gf/stall-diagnostics`, 259 on `gf/cross-backend-equality`, not re-run), and
HIP `gfx1100` 245/245 on `hal` (247 on `gf/stall-diagnostics`, 250 on
`gf/cross-backend-equality`, not re-run). The format sweep changed whitespace,
line breaks, include and `using` order, and configured numeric separators only.

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
  -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure --no-tests=error
```

ASAN+UBSAN runs the host scheduler tests. The job does not run TSAN: the
recorded TSAN blocker is a runtime mapping failure, and the hosted runner has
no known-good mapping configuration. TSAN stays a local/manual check until a
runner with a passing startup check is available.

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
(`gfx1100`) and ROCm 7.2, the HIP CTest run passes 245/245 with 43 HIP cases
(8 typed Flow, 24 particle CLI, 10 photon CLI, and 1 photon output case),
measured on `b972eaa3bd778b81f2dca505d3662df710792bd9` and recorded in
`/home/agent/final-hip-b972eaa/ctest.log`, with `rocprof` kernel evidence. `hal` reports `sharing=unknown`, so this is a
correctness result and not a performance claim. The GitHub-hosted job makes no
HIP runtime claim.

## REUSE

```sh
python -m pip install --upgrade pip reuse==6.3.0
reuse lint
```

The `reuse` job checks every tracked file against `.reuse/dep5` and
`LICENSES/MPL-2.0.txt`. The blanket `Files: *` entry in `.reuse/dep5` covers
the workflow and documentation files added here; the previous hosted run
passed with every tracked file covered.

## No SYCL backend

graphFlow has no SYCL backend, so there is no SYCL job and no job pretends to
run one. The optional `icpx-cpu` job compiles and tests the CPU backend with
Intel's host compiler in a oneAPI container. It runs only when a maintainer
starts `workflow_dispatch` with `run_icpx` enabled; it is not part of the
push/PR gate.

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
- CUDA or HIP execution; both jobs are compile-only.
- TSAN on a self-hosted host: blocked by the hal `unexpected memory mapping`
  failure and `rosi_v100` being unreachable; the hosted runner has no known-good
  mapping configuration.
- GitLab CI.
- H100 or any other specific GPU target: `rosi_v100` is unreachable and no
  final H100 run has been recorded, so no H100 performance claim is made.

## Local reproduction

The commands above match `docs/build.md`. To reproduce a job locally, clone
alpaka at the pin, set `GRAPHFLOW_ALPAKA_SOURCE_DIR` to that clone, and run the
same configure, build, and CTest commands. Container jobs need the matching
image and the extra `git` install before checkout.

## Status

On `gf/cross-backend-equality` the hosted jobs pass `cpu (Release)`,
`cpu (Debug)`, `sanitizers (ASAN+UBSAN)`, `cuda-compile`, `hip-compile`,
`format`, and
`reuse`. Each CPU job runs 210 CTest cases (Release and Debug), 202 at
`2ad7d86` plus two source-admission cases, one N=1e6 smoke, two stall
diagnostics cases, and the three cross-backend equality cases; the
`sanitizers` job runs ASAN+UBSAN over the same CPU
suite, and the N=1e6 smoke, the two stall diagnostics cases, and the three
cross-backend equality cases run in the
ordinary CTest job without a skip. The `format` job
enforces the T14b
sweep with clang-format 20.1.8. The `sm_70` workaround clears the nvcc internal
compiler error, so `cuda-compile` builds the CUDA sources and the CUDA example
without a GPU. `hip-compile` builds the HIP sources and the HIP example without
an AMD GPU. `icpx-cpu` is manual and skipped on push and pull requests.

Container jobs set `shell: bash` because container jobs default to `sh`, which
rejects `set -o pipefail`.

Validation for this change covers YAML parsing, `actionlint`, `bash -n` on the
embedded shell, the container image tags, and the alpaka pin. No local project
build or test was run.
