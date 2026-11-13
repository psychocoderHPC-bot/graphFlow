# Continuous integration

`.github/workflows/ci.yml` is the only CI workflow. It runs eight jobs: CPU
Release, CPU Debug, ASAN+UBSAN, TSAN, CUDA on an A30, HIP compile-only, format,
and REUSE. Every job has a finite timeout, read-only repository permissions, and
concurrency cancellation for superseded runs.

## Job matrix

| Job | Runner | Build | Test evidence | Timeout |
| --- | --- | --- | --- | ---: |
| `cpu` (`Release`, `Debug`) | `ubuntu-latest` | CMake + Ninja, CPU only | CTest, JUnit | 60 min |
| `sanitizers` | `ubuntu-latest` | Debug, `-fsanitize=address,undefined` | CTest, JUnit | 90 min |
| `tsan` | `[self-hosted, linux, fwk394]` | Debug, non-PIE, `-fsanitize=thread` | CTest subset `pool\|dense\|pressure\|scripted\|lease`, JUnit | 60 min |
| `cuda` | `[self-hosted, linux, cuda, a30]` | Release, `CMAKE_CUDA_ARCHITECTURES=80` | CTest + JUnit skip gate + `--backend cuda` example at `N=1e6` | 90 min |
| `hip-compile` | `[self-hosted, linux, rocm]` | Release, HIP `gfx90a`, compile-only | none (build must succeed) | 90 min |
| `format` | `ubuntu-latest` | clang-format 20.1.8 | dry-run over `include/`, `examples/`, `tests/` | 15 min |
| `reuse` | `ubuntu-latest` | reuse 6.3.0 | `reuse lint` | 15 min |

The CUDA, TSAN, and HIP jobs require external self-hosted runners with the exact
labels shown. This workflow neither creates nor emulates those runners. Until the
labels are registered and reachable, treat the three jobs as non-required checks
and do not mark them required in branch protection. `hal` and `fwk394` are the
handover hosts measured for these jobs; that is not proof that a GitHub Actions
runner is registered under any label.

## Pins

| Dependency | Pin | Source |
| --- | --- | --- |
| alpaka | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (alpaka 3.0.0, 2026-10-26) | Recorded in `docs/acceptance.md` and `docs/alpaka-api-map.md` |
| Catch2 | 3.4.0 | Ubuntu 24.04 `noble/universe` package `catch2` |
| clang-format | 20.1.8 | PyPI `clang-format==20.1.8` wheel in a temporary venv |
| reuse | 6.3.0 | PyPI `reuse==6.3.0` in a temporary venv |

The older `4780dc786a2eb1a5452a1c66d94a8b442d411ab8` reference pin is not used:
the repository records the newer validated alpaka tree above. Change the
`ALPAKA_REF` workflow variable only after a full CPU/CUDA/HIP probe and a new
acceptance record. Change the Catch2 pin only if `ubuntu-latest` stops shipping
3.4.0. Only the `cpu` and `sanitizers` apt jobs enforce 3.4.0; the self-hosted
CUDA, TSAN, and HIP jobs use the runner-provided Catch2 3.x.

## Exact commands

All jobs fetch the same alpaka tree:

```sh
rm -rf "$RUNNER_TEMP/alpaka"
git init "$RUNNER_TEMP/alpaka"
git -C "$RUNNER_TEMP/alpaka" remote add origin https://github.com/alpaka-group/alpaka.git
git -C "$RUNNER_TEMP/alpaka" fetch --depth 1 origin "$ALPAKA_REF"
git -C "$RUNNER_TEMP/alpaka" checkout --detach FETCH_HEAD
test "$(git -C "$RUNNER_TEMP/alpaka" rev-parse HEAD)" = "$ALPAKA_REF"
```

CPU Release and Debug:

```sh
sudo apt-get update
sudo apt-get install --yes --no-install-recommends cmake ninja-build catch2

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$RUNNER_TEMP/alpaka" \
  -DGRAPHFLOW_BUILD_TESTS=ON \
  -DGRAPHFLOW_BUILD_EXAMPLES=ON
cmake --build build --parallel
mkdir -p "$RUNNER_TEMP/artifacts"
ctest --test-dir build --output-on-failure --no-tests=error \
  --output-junit "$RUNNER_TEMP/artifacts/cpu-Release-junit.xml"
```

Use `-DCMAKE_BUILD_TYPE=Debug` for the Debug matrix entry.

ASAN+UBSAN:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$RUNNER_TEMP/alpaka" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DGRAPHFLOW_BUILD_TESTS=ON \
  -DGRAPHFLOW_BUILD_EXAMPLES=ON
cmake --build build --parallel
mkdir -p "$RUNNER_TEMP/artifacts"
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build --output-on-failure --no-tests=error \
    --output-junit "$RUNNER_TEMP/artifacts/sanitizers-junit.xml"
```

TSAN on `fwk394`:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$RUNNER_TEMP/alpaka" \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -fno-pie -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread -no-pie" \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -Dalpaka_DEP_OMP=OFF \
  -DGRAPHFLOW_BUILD_TESTS=ON \
  -DGRAPHFLOW_BUILD_EXAMPLES=ON
cmake --build build --parallel
mkdir -p "$RUNNER_TEMP/artifacts"
TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 \
  ctest --test-dir build --output-on-failure --no-tests=error \
    -R 'pool|dense|pressure|scripted|lease' \
    --output-junit "$RUNNER_TEMP/artifacts/tsan-junit.xml"
```

The TSAN job fails if the CTest log or JUnit report contains any
`ThreadSanitizer` string. No suppression file and no `suppressions=` option is
used. `hal` is not a TSAN host: its kernel aborts with `FATAL: ThreadSanitizer:
unexpected memory mapping`. `fwk394` is the compatible host for this subset.

CUDA on the A30 runner:

```sh
export CUDA_VISIBLE_DEVICES=0

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
    -u CPATH -u CPLUS_INCLUDE_PATH \
  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$RUNNER_TEMP/alpaka" \
    -DGRAPHFLOW_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DGRAPHFLOW_BUILD_TESTS=ON \
    -DGRAPHFLOW_BUILD_EXAMPLES=ON
env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
    -u CPATH -u CPLUS_INCLUDE_PATH \
  cmake --build build --parallel
mkdir -p "$RUNNER_TEMP/artifacts"
env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
    -u CPATH -u CPLUS_INCLUDE_PATH \
  ctest --test-dir build --output-on-failure --no-tests=error \
    --output-junit "$RUNNER_TEMP/artifacts/cuda-junit.xml"
env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
    -u CPATH -u CPLUS_INCLUDE_PATH \
  ./build/graphFlow_particle_feedback \
    --backend cuda --particles 1000000 --chunk-size 1024 \
    --live-particles 8192 --seed 42
```

HIP compile-only on the ROCm runner:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR="$RUNNER_TEMP/alpaka" \
  -DGRAPHFLOW_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dalpaka_DEP_OMP=OFF \
  -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DGRAPHFLOW_BUILD_TESTS=ON \
  -DGRAPHFLOW_BUILD_EXAMPLES=ON
cmake --build build --parallel
```

The HIP job stops after the build. It does not run CTest or the example and
makes no HIP runtime claim.

Format:

```sh
sudo apt-get update
sudo apt-get install --yes --no-install-recommends python3-venv
python3 -m venv "$RUNNER_TEMP/clang-format-venv"
"$RUNNER_TEMP/clang-format-venv/bin/pip" install --upgrade pip
"$RUNNER_TEMP/clang-format-venv/bin/pip" install "clang-format==20.1.8"
export PATH="$RUNNER_TEMP/clang-format-venv/bin:$PATH"
clang-format --version

mapfile -t files < <(git ls-files 'include' 'examples' 'tests' | grep -E '\.(cpp|hpp)$')
clang-format --dry-run --Werror "${files[@]}"
```

REUSE:

```sh
sudo apt-get update
sudo apt-get install --yes --no-install-recommends python3-venv
python3 -m venv "$RUNNER_TEMP/reuse-venv"
"$RUNNER_TEMP/reuse-venv/bin/pip" install --upgrade pip
"$RUNNER_TEMP/reuse-venv/bin/pip" install "reuse==6.3.0"
"$RUNNER_TEMP/reuse-venv/bin/reuse" lint
```

## Environment scrub

The CUDA job applies this prefix to configure, build, test, and the example:

```sh
env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CMAKE_PREFIX_PATH \
    -u CPATH -u CPLUS_INCLUDE_PATH
```

Self-hosted images often export library and include paths from other toolchains.
Leaving them set can mix a stale CUDA runtime or a ROCm tree into an A30 build,
and can make a build pass for a reason the repository does not control. The
scrub keeps the job's dependency resolution limited to the runner's default
system paths plus the explicitly passed CMake variables.

The HIP job does not scrub `CMAKE_PREFIX_PATH` because it sets
`-DCMAKE_PREFIX_PATH=/opt/rocm` explicitly. That is the only intended prefix in
that job.

## Host expectations

`hal` is the A30 handover host expected to serve the CUDA job once a runner is
registered:

- Ubuntu 24.04, CUDA 13.4, NVIDIA A30 (`sm_80`), CMake 3.28.3, Ninja 1.11.1,
  g++ 13.3, Catch2 3.4.0.
- ROCm 7.2.0 is installed under `/opt/rocm`, so `hal` can also serve the
  `hip-compile` job, but it has no AMD GPU. `gfx90a` is a compile target only.
- `hal` reports `sharing=unknown`. Timings from this host are shared-host
  baselines, not exclusive-hardware acceptance numbers.
- It cannot run TSAN: the kernel aborts with `FATAL: ThreadSanitizer: unexpected
  memory mapping`, even for a non-PIE binary.

`fwk394` is a second Ubuntu 24.04 handover host:

- 28 CPUs, CUDA 12.5, one RTX 2080 Ti (`sm_75`), one Quadro GV100 (`sm_70`),
  CMake 3.28.3, Ninja 1.11.1, g++ 13.3, Catch2 3.4.0.
- It is the compatible TSAN host for the validated subset
  `-R 'pool|dense|pressure|scripted|lease'`; CTest reports the selected test
  count. The recorded fwk394 baseline is clean (no race or memory-mapping
  report); CI execution is pending runner registration.
- No `/opt/rocm` toolkit or HIP compiler; only AMD runtime libraries are
  installed. It cannot run the `hip-compile` job as written.
- It is not an A30 host. A CUDA compile/run check there must override
  `CMAKE_CUDA_ARCHITECTURES` (for example `75`) and must not be reported as an
  A30 result.

These are handover observations, not proof that GitHub Actions runners are
registered under the labels above. Registration, credentials, and artifact
retention are environment decisions outside the repository.

## GPU test rule

A missing or skipped required GPU test is a failure. The workflow enforces this
in three ways on the CUDA job:

1. `ctest --no-tests=error` fails when CTest discovers no tests.
2. The JUnit report is scanned for `<skipped`; any skipped case fails the job.
3. The `--backend cuda` example must print `# selected-backend=cuda ` and its
   histogram counts must sum to `N`. The example is the runtime proof that a
   CUDA device was selected and that the kernels ran on it.

No CUDA or HIP job sets `continue-on-error`, and no missing device is allowed to
degrade to the CPU backend. Once T3 is merged, the CUDA build will also include
its CUDA-specific CTest case. This branch predates T3, so the `--backend cuda`
example smoke remains the explicit selected-backend and histogram gate.

## Artifacts and retention

| Artifact | Contents |
| --- | --- |
| `junit-cpu-Release`, `junit-cpu-Debug` | CTest JUnit XML |
| `junit-sanitizers` | CTest JUnit XML |
| `tsan-fwk394-artifacts` | compiler version, configure/build/CTest logs, JUnit XML |
| `cuda-a30-artifacts` | `nvidia-smi` output, `nvcc` version, configure/build/CTest logs, JUnit XML, example log |
| `hip-compile-logs` | `amdclang++` version, configure/build logs, runtime-note |

Artifacts are kept for 14 days and uploaded with `if: always()` so a failure
still leaves diagnostics. Logs and JUnit files are written under
`$RUNNER_TEMP/artifacts` and never into the repository.

## Bootstrap prerequisites

Two gates depend on files owned by a follow-up task:

- `format` requires `.clang-format`.
- `reuse` requires `LICENSES/MPL-2.0.txt`, plus the SPDX headers and
  `.reuse/dep5` if needed.

Both jobs fail with an explicit `::error::` when their prerequisite is missing.
They do not skip the check and do not report success. This change deliberately
does not add `.clang-format`, `LICENSES/`, or SPDX headers.

## Known blockers

- **No AMD GPU target.** Neither `hal` nor `fwk394` exposes an AMD device. HIP is
  compile-only; the HIP runtime gate and the AMD acceptance run need a real AMD
  runner.
- **TSAN host split.** `hal` aborts TSAN with `FATAL: ThreadSanitizer:
  unexpected memory mapping`, even for a non-PIE binary. The `tsan` job is
  pinned to `fwk394`, which runs the validated TSAN subset. The `fwk394`
  label must be registered externally. No suppression is used.
- **Shared `hal`.** `hal` reports `sharing=unknown`; performance results from the
  CUDA job are baselines, not exclusive allocations.
- **Self-hosted capacity.** The A30, ROCm, and fwk394 labels are external
  requirements: the matching runners and toolchains must be registered. Until
  then the jobs are non-required checks. A missing runner blocks the job; it
  must not be replaced by a CPU fallback.
- **Bootstrap metadata.** `.clang-format`, `LICENSES/MPL-2.0.txt`, `.reuse/dep5`
  if needed, and SPDX headers are still missing. The format and REUSE jobs
  fail until they land.
- **Pinned commit availability.** The workflow fetches
  `eb677e8d7f70d79b35df25d9002f94ba669b6f44` by SHA. If the upstream repository
  stops serving that object, update the pin deliberately and re-run the probe.

## Manager validation on hal and fwk394

On `hal`, run the CUDA job commands above with `CUDA_VISIBLE_DEVICES=0`,
`CMAKE_CUDA_ARCHITECTURES=80`, and the environment scrub. Confirm `nvidia-smi -L`
shows the A30, the CTest JUnit report has no `<skipped`, and the `N=1e6` example
prints `# selected-backend=cuda ` with a histogram sum of `1000000`. Re-run
the HIP compile with the exact `hip-compile` configure line; a successful build
is the only HIP claim.

On `fwk394`, run the CPU Release and Debug jobs, the sanitizer job, the TSAN
job, and a CUDA compile with `CMAKE_CUDA_ARCHITECTURES=75`. For TSAN, confirm
the selected subset runs and that neither the CTest log nor the JUnit report
contains `ThreadSanitizer`. Do not label that CUDA result as an A30
acceptance run. `fwk394` cannot run the ROCm job without a ROCm install. Do not
run TSAN on `hal`; it aborts with the unexpected-memory-mapping error.
