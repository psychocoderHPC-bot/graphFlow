# Continuous integration

`.github/workflows/ci.yml` is the only CI workflow. It runs seven jobs: CPU
Release, CPU Debug, ASAN+UBSAN, CUDA on an A30, HIP compile-only, format, and
REUSE. Every job has a finite timeout, read-only repository permissions, and
concurrency cancellation for superseded runs.

## Job matrix

| Job | Runner | Build | Test evidence | Timeout |
| --- | --- | --- | --- | ---: |
| `cpu` (`Release`, `Debug`) | `ubuntu-latest` | CMake + Ninja, CPU only | CTest, JUnit | 60 min |
| `sanitizers` | `ubuntu-latest` | Debug, `-fsanitize=address,undefined` | CTest, JUnit | 90 min |
| `cuda` | `[self-hosted, linux, cuda, a30]` | Release, `CMAKE_CUDA_ARCHITECTURES=80` | CTest + JUnit skip gate + `--backend cuda` example at `N=1e6` | 90 min |
| `hip-compile` | `[self-hosted, linux, rocm]` | Release, HIP `gfx90a`, compile-only | none (build must succeed) | 90 min |
| `format` | `ubuntu-latest` | clang-format 20.1.8 | dry-run over `include/`, `examples/`, `tests/` | 15 min |
| `reuse` | `ubuntu-latest` | reuse 6.3.0 | `reuse lint` | 15 min |

The CUDA and HIP jobs require registered self-hosted runners with exactly the
labels shown. The workflow does not emulate those runners, does not fall back to
another host, and does not substitute the CPU backend. `hal` and `fwk394` are
the two hosts used during the handover.

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
3.4.0; the CPU jobs print the installed package version.

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

The CUDA runner applies this prefix to configure, build, test, and the example:

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

`hal` is the A30 host used for the CUDA job:

- Ubuntu 24.04, CUDA 13.4, NVIDIA A30 (`sm_80`), CMake 3.28.3, Ninja 1.11.1,
  g++ 13.3, Catch2 3.4.0.
- ROCm 7.2.0 is installed under `/opt/rocm`, so `hal` can also serve the
  `hip-compile` job, but it has no AMD GPU. `gfx90a` is a compile target only.
- `hal` reports `sharing=unknown`. CI timings from this host are shared-host
  baselines, not exclusive-hardware acceptance numbers.

`fwk394` is a second Ubuntu 24.04 host:

- CUDA 12.5, one RTX 2080 Ti (`sm_75`), one Quadro GV100 (`sm_70`), CMake 3.28.3,
  Ninja 1.11.1, g++ 13.3, Catch2 3.4.0.
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
degrade to the CPU backend. CTest in the CUDA build currently exercises the
CPU-side test suite compiled with the CUDA language; the example supplies the
device-execution evidence until a CUDA-specific CTest case lands.

## Artifacts and retention

| Artifact | Contents |
| --- | --- |
| `junit-cpu-Release`, `junit-cpu-Debug` | CTest JUnit XML |
| `junit-sanitizers` | CTest JUnit XML |
| `cuda-a30-artifacts` | `nvidia-smi` output, `nvcc` version, configure/build/CTest logs, JUnit XML, example log |
| `hip-compile-logs` | `amdclang++` version, configure/build logs, runtime-note |

Artifacts are kept for 14 days and uploaded with `if: always()` so a failure
still leaves diagnostics. Logs and JUnit files are written under
`$RUNNER_TEMP/artifacts` and never into the repository.

## Bootstrap gates

Two gates depend on files that arrive in a follow-up task:

- `format` activates when `.clang-format` exists. Until then it prints a
  warning and a job summary, installs nothing, and passes.
- `reuse` activates when `LICENSES/MPL-2.0.txt` exists. Until then it prints a
  warning and a job summary, installs nothing, and passes.

Both gates become enforcing automatically once their prerequisites land. This
keeps the workflow green during the bootstrap without weakening the eventual
checks. This change deliberately does not add `.clang-format`, `LICENSES/`, or
SPDX headers.

## Known blockers

- **No AMD GPU target.** Neither `hal` nor `fwk394` exposes an AMD device. HIP is
  compile-only; the HIP runtime gate and the AMD acceptance run need a real AMD
  runner.
- **TSAN mapping on `hal`.** The `hal` kernel aborts TSAN with
  `FATAL: ThreadSanitizer: unexpected memory mapping`, even for a non-PIE binary.
  There is no TSAN job until a compatible host is available. A suppression is
  not added without evidence.
- **Shared `hal`.** `hal` reports `sharing=unknown`; performance results from the
  CUDA job are baselines, not exclusive allocations.
- **Self-hosted capacity.** The A30 and ROCm labels must be provided by
  registered runners with CUDA/ROCm toolchains. A missing runner blocks the job;
  it must not be replaced by a CPU fallback.
- **Bootstrap metadata.** `.clang-format`, `LICENSES/MPL-2.0.txt`, `.reuse/dep5`
  if needed, and SPDX headers are still missing. The format and REUSE jobs stay
  inactive until they land.
- **Pinned commit availability.** The workflow fetches
  `eb677e8d7f70d79b35df25d9002f94ba669b6f44` by SHA. If the upstream repository
  stops serving that object, update the pin deliberately and re-run the probe.
- **CUDA-specific CTest case.** The CUDA job proves device execution through the
  example, not through a CUDA-only CTest entry. That entry remains a separate
  task.

## Manager validation on hal and fwk394

On `hal`, run the CUDA job commands above with `CUDA_VISIBLE_DEVICES=0`,
`CMAKE_CUDA_ARCHITECTURES=80`, and the environment scrub. Confirm `nvidia-smi -L`
shows the A30, the CTest JUnit report has no `<skipped`, and the `N=1e6` example
prints `# selected-backend=cuda ` with a histogram sum of `1000000`. Re-run
the HIP compile with the exact `hip-compile` configure line; a successful build
is the only HIP claim.

On `fwk394`, run the CPU Release and Debug jobs, the sanitizer job, and a CUDA
compile with `CMAKE_CUDA_ARCHITECTURES=75`. Do not label that result as an A30
acceptance run. `fwk394` cannot run the ROCm job without a ROCm install.
