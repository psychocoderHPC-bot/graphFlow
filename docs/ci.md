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
| `icpx-cpu` | `ubuntu-latest` | `intel/oneapi-hpckit:2025.3.0-0-devel-ubuntu24.04` | CPU | Release | manual dispatch only |

The CPU and sanitizer jobs install `cmake`, `ninja-build`, `g++`, and `catch2`
from the Ubuntu 24.04 archive and clone alpaka from source. The CUDA, HIP, and
oneAPI jobs do the same inside their container; they install `git` before
`actions/checkout` because the base images do not guarantee it.

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

### Known pre-existing status-test failures

Five `[status]` unit tests in `tests/unit/test_flow_failure.cpp` fail on the
hosted Debug and ASAN+UBSAN builds:

- "a sink status failure is detected before publication"
- "a classifier status failure rejects the batch before planning"
- "an invalid updater result rejects the planned batch"
- "a successful updater keeps the status word clear"
- "a checked updater overflow fails the planned batch"

Each case acquires an input chunk whose payload slots are never written, so the
classified bin counts depend on allocator contents. The hosted Release build
passes; the hosted Debug build fails, which shows the result depends on heap
layout rather than on the workflow. The workflow adds no test changes, so these
jobs stay red until a follow-up writes the input payloads or zero-initializes
the storage.

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
recorded runtime validation remains the A30 `sm_80` run with CUDA 13.4; this
job makes no runtime claim. CUDA 12.5 is not used because it hits an nvcc
internal compiler error in `HashScatterKernel.hpp`.

### CUDA `sm_70` compiler history

The first hosted run used CUDA 12.6.3 and reproduced the CUDA 12.6.85 nvcc
internal compiler error at `flow/Execution.hpp(385)`
(`check_name_hiding_by_template_parameters`). Main PR #29 made the trailing
`IMdSpan` parameters explicit and routed the test-hook adapters through small
templates. The current hosted run passes the `sm_70` compile gate. CUDA 12.5
remains unused because of the separate `HashScatterKernel.hpp` ICE.

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
build: no CTest run and no example execution. HIP runtime results stay
compile-only until an AMD device is available.

## REUSE

```sh
python -m pip install --upgrade pip reuse==6.3.0
reuse lint
```

The `reuse` job checks every tracked file against `.reuse/dep5` and
`LICENSES/MPL-2.0.txt`. The hosted job passes.

## No SYCL backend

graphFlow has no SYCL backend, so there is no SYCL job and no job pretends to
run one. The optional `icpx-cpu` job compiles and tests the CPU backend with
Intel's host compiler in a oneAPI container. It runs only when a maintainer
starts `workflow_dispatch` with `run_icpx` enabled; it is not part of the
push/PR gate.

## Format is pending T14b

The workflow has no format job. `.clang-format` and the SPDX/REUSE foundation
are in `main`, but the tree-wide format and SPDX sweep (T14b) is still open.
Wiring `clang-format --dry-run --Werror` now would fail on the current tree.
Add the format job after T14b lands, and keep it non-required until the first
clean run.

## Not included

- Self-hosted runners or labels (`fwk394`, `a30`, `rocm`).
- CUDA or HIP execution; both jobs are compile-only.
- TSAN on a self-hosted host.
- GitLab CI.
- H100 or any other specific GPU target.

## Local reproduction

The commands above match `docs/build.md`. To reproduce a job locally, clone
alpaka at the pin, set `GRAPHFLOW_ALPAKA_SOURCE_DIR` to that clone, and run the
same configure, build, and CTest commands. Container jobs need the matching
image and the extra `git` install before checkout.

## Status

The hosted run passes `cpu (Release)`, `reuse`, `hip-compile`, and
`cuda-compile`. It fails the five pre-existing status tests in `cpu (Debug)`
and `sanitizers`.

The workflow YAML parses with `yaml.safe_load`. The first hosted run stopped
the container jobs at the clone step because container jobs default to `sh`
and `set -o pipefail` is not available there. This revision sets `shell: bash`
for the CUDA, HIP, and oneAPI jobs. The final hosted job results are recorded
above.
