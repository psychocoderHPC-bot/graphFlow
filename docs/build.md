<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Building graphFlow

graphFlow builds against a pinned alpaka3 checkout and a local Catch2 3
installation. Use one build directory per backend. CUDA and HIP are mutually
exclusive in one build; the CPU backend stays enabled in accelerator builds so
`--backend cpu` remains available.

## Pinned toolchain

| Component | Pin | Notes |
| --- | --- | --- |
| alpaka | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` | 3.0.0 tag; set `GRAPHFLOW_ALPAKA_SOURCE_DIR` |
| Catch2 | 3.4.0 | local install; `find_package(Catch2 3 REQUIRED)` |
| clang-format | 20.1.8 | format gate |
| reuse | 6.3.0 | license gate |
| CMake | 3.28.3 | `cmake_minimum_required(VERSION 3.25)` |
| Ninja | 1.11.1 | `-G Ninja` |
| CUDA | 13.4 | validated on the A30 with `sm_80`; `sm_75` is the RTX 2080 Ti target |
| ROCm | amdclang++ from `/opt/rocm` | compile-only `gfx90a` gate |
| C++ | C++20 | `CMAKE_CXX_STANDARD 20`, extensions off |

The integrated CPU run used g++ 13.3 on fwk394; the CUDA run used hal. A
different compiler is allowed, but re-run the CPU/CUDA/HIP probes before
recording a new pin.

Catch2 3.4.0 must be visible to `find_package`. When it is not in a default
prefix, add `-DCMAKE_PREFIX_PATH=<catch2-3.4.0-prefix>` or set `Catch2_DIR`.
alpaka is always consumed from the pinned source tree, not downloaded.

## Environment scrub

Clear variables that can change the selected device, the executor, or the host
thread count before configuring and testing:

```sh
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
unset ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE
unset ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE
unset ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE
```

Then expose exactly one device for the run:

```sh
export CUDA_VISIBLE_DEVICES=0   # CUDA runs
# export ROCR_VISIBLE_DEVICES=0 # HIP runs on a host with a visible AMD GPU
```

## CPU

```sh
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF
cmake --build build-cpu -j 14
ctest --test-dir build-cpu --output-on-failure --no-tests=error \
  --output-junit ctest-cpu.xml
```

### Example CLI contract tests

CTest also runs the real `graphFlow_particle_feedback` binary through
`tests/integration/example_cli.cmake`. The matrix is `N=0,1,17,127,128,4097`
crossed with `C=1,31,127,128`, with `--seed 42` and `--max-visits 64`. Every
case requires exit code 0 and a histogram whose counts sum to `N`. The CPU cases
are always registered; a CUDA build adds the same matrix for the CUDA backend.
A missing binary or an unavailable requested backend fails the case instead of
skipping it. The case names follow
`example.particle_feedback.<backend>.N<N>.C<C>`.

`--output-junit <file>` writes the CTest results as JUnit XML. Keep it in the
build directory or an artifact directory; no JUnit file is committed.

Add `-Dalpaka_COMPILE_PEDANTIC=ON` for the CI-equivalent warning gate. CI must
also see warnings such as `-Wsign-compare` and
`-Wimplicit-const-int-float-conversion` as errors; do not disable them in the
project.

## CUDA

A30, `sm_80`:

```sh
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j 14
ctest --test-dir build-cuda --output-on-failure --no-tests=error \
  --output-junit ctest-cuda.xml
```

RTX 2080 Ti, `sm_75`, same tree in a separate build directory:

```sh
cmake -S . -B build-cuda-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda-sm75 -j 14
ctest --test-dir build-cuda-sm75 --output-on-failure --no-tests=error
```

The `sm_75` command is the documented RTX 2080 Ti target. The benchmark record
in `docs/benchmarks.md` was measured on the A30 only.

## HIP, compile-only

The validated HIP gate compiles the full tree for `gfx90a` with `amdclang++`
from ROCm under `/opt/rocm`. No AMD device was visible in the validation
environment, so this gate does not run the example or the tests.

```sh
export ROCM_PATH=/opt/rocm
export PATH="${ROCM_PATH}/bin:${PATH}"
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_HIP=ON \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dalpaka_DEP_OMP=OFF
cmake --build build-hip -j 14
```

`-Dalpaka_DEP_OMP=OFF` is required so alpaka does not add an OpenMP dependency
to the HIP build. Do not run CTest when no AMD device is visible. On an AMD
runner, select one device and run the same test command:

```sh
ROCR_VISIBLE_DEVICES=0 ctest --test-dir build-hip --output-on-failure --no-tests=error
```

Record the GPU model, ROCm version, and `gfx` target with any HIP execution
result. The current `gfx90a` result is compile-only.

## Debug and sanitizers

Debug build:

```sh
cmake -S . -B build-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace
cmake --build build-debug -j 14
ctest --test-dir build-debug --output-on-failure --no-tests=error
```

Per-event payload validation defaults to `ON` when
`CMAKE_BUILD_TYPE=Debug` and to `OFF` for `Release` and `RelWithDebInfo`. Pass
`-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON|OFF` to override the default. The Debug
build above compiles the photon checks in `include/graphFlow/flow/Debug.hpp`
and `PhotonDebug.hpp`.

To run the checks in a Release build, configure with:

```sh
cmake -S . -B build-release-validation -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace
cmake --build build-release-validation -j 14
ctest --test-dir build-release-validation --output-on-failure --no-tests=error
```

The option adds `GRAPHFLOW_DEBUG_VALIDATION=1` to the `graphFlow` interface
target. Release hot loops keep no per-event state branch. The two
`[photon][debug]` CTest cases run in both modes: the Release case asserts that
the hot path has no per-event state branch, and the Debug case exercises the
compiled checks. `rosi_v100` CTest results are **195/195 CPU** and **247/247
CUDA** in the default `Release` build and **195/195 CPU** and **247/247 CUDA**
in the default `Debug` build. Release compiles the per-event checks out; Debug
compiles them in. Host setup validates the run configuration once, and genuine
physics failures stay on the `StatusCode` channel in both builds. The five
former `[status]` tests that tripped the RNG stream-id assertion on Debug
builds now pass in both builds.

ASAN and UBSAN on the host scheduler:

```sh
cmake -S . -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j 14
ctest --test-dir build-asan --output-on-failure --no-tests=error
```

TSAN is blocked on the hal kernel by `FATAL: ThreadSanitizer: unexpected memory
mapping`; see `docs/known-limitations.md`. Use a host whose runtime mapping
passes the TSAN startup check.

## Format and license metadata

The tree contains `.clang-format` (clang-format 20.1.8), `.reuse/dep5`, and
`LICENSES/MPL-2.0.txt`. Run `reuse lint` with reuse 6.3.0 to check the license
metadata. A format check needs clang-format 20.1.8 and the project
`.clang-format`; the exact file list is a CI policy decision. These gates are
not wired into a workflow yet.

## Checks that are not CI yet

The repository has no `.github/workflows` and no CI runners. The commands above
are the documented local entry points. The recorded integrated results cover
the FrameSpec revision `327f74f`: the fwk394 CPU build with CTest 82/82 and the
hal CUDA A30 `sm_80` build with CTest 86/86; the HIP `gfx90a` compile gate also
passes. The later additive PR-A graph API core at
`60a34302b1c3cf205304916c7233f13118591dd0` was validated with **97/97 CPU
CTest** on fwk394 Release and **101/101 CUDA CTest** on hal `sm_80` Release,
both with `-Dalpaka_COMPILE_PEDANTIC=ON`; the counts were recorded at that
revision, and this documentation commit does not claim a new build or test run.
The `sm_75` CUDA build still fails with the CUDA 12.5 nvcc internal compiler
error (`docs/known-limitations.md`); HIP execution and an integrated
ASAN/UBSAN count have no recorded run. Format and REUSE metadata are present;
no CI job runs them yet.
