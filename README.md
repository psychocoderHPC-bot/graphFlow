# graphFlow

Host-scheduled dataflow graph on [alpaka3](https://github.com/alpaka-group/alpaka)
with chunked ownership, exact output-slot planning, dense/hash classify/scatter
pipelines, and multi-chunk transform batches. The example
`graphFlow_particle_feedback` runs a Generate -> Filter -> {A, B} -> Sink
feedback network and prints the sorted weight histogram.

Documentation:

- [Build and pins](docs/build.md)
- [Architecture](docs/architecture.md)
- [Numerics](docs/numerics.md)
- [Benchmarks](docs/benchmarks.md)
- [Known limitations](docs/known-limitations.md)
- [alpaka3 API map](docs/alpaka-api-map.md)
- [Public API contract](docs/public-api.md)
- [Approved plan](docs/plan.md)
- [Missing work](docs/missing.md)
- [Acceptance record](docs/acceptance.md)

## Status

Current main: `25fa0927a4e042f229d43229734f50dce8277954`. The T5 photon
scattering record is at `ac89ab558c26b938038035b26ca5dbe6d7ce7727` (T1, T2,
T3, T5, T7, T13, T14a). The public API is implemented: the typed
`Flow<TPayload, TDevice, TStorage>` topology, execution, storage, and status
path described in [public API contract](docs/public-api.md). CPU, CUDA, and HIP
execution are validated on real devices. HIP runs on `hal` with an AMD Radeon
RX 7900 XTX (`gfx1100`) and ROCm 7.2: **238/238 CTest passed**, including **43
HIP cases** (8 typed Flow cases plus the particle and photon CLI/output cases),
with `rocprof` recording the GPU kernel dispatches. GitHub-hosted HIP builds
remain compile-only because those runners have no AMD GPU; no HIP runtime claim
is made for GitHub CI. `hal` reports `sharing=unknown`, so the HIP run is a
correctness result, not a performance claim. JURECA `jrc0880.jureca` (4x H100
PCIe 80GB, SM 9.0) is available, but its gateway sharing flag is unknown and the
recorded run observed external load while the client was stale, so it provides
only preliminary CUDA evidence and no final H100 claim. fwk394 and H100 were
not re-run for the HIP validation.

The runner has one non-blocking queue per node, one in-flight batch per node,
device-derived `C`/`P`/`T`, up to `Q` ready input chunks per transform batch, a
source-admission capacity `P`, and a chunk pool bounded by `P` rather than by
`N`. Source admission requires `min(C, P, remaining)` particles per admitted
chunk. Consumer ready lists still dispatch partial chunks immediately.

Transform batches flatten every input chunk into a compact `TileDescriptor`
list and issue one classify launch and one scatter launch per batch. The batch
reserves the adaptive `outputChunkBound(sum(fill), C, B)`; under fragmentation
the largest fitting prefix of the ready list is dispatched. The pool is derived
from the source capacity plus one output bound per registered node, so `Q`
changes launch amortisation and buffer sizing, not the pool bound.

The format/REUSE foundation is in main: `.clang-format` (clang-format 20.1.8),
`.reuse/dep5`, and `LICENSES/MPL-2.0.txt`. The tree-wide format and SPDX sweep
remains open (T14b), and there is no CI workflow yet. See
[known limitations](docs/known-limitations.md).

`rosi_v100` is the exclusive validation platform. Current `rosi_v100`
results at the sm_70 ICE workaround: CPU CTest **195/195 passed** (OMP=ON
Release and Debug, pedantic) and CUDA CTest **247/247 passed** on V100 `sm_70`
(CUDA 12.6.85, Release and Debug). Both totals are one higher than before the
RNG merge: the RNG change removed one `TEST_CASE` and the debug validation adds
two `[photon][debug]` cases. Release compiles the per-event checks out; Debug
compiles them in. Host setup validates the configuration once, and genuine
physics failures stay on the status channel. The five former `[status]` tests
that tripped the RNG stream-id assertion on Debug builds now pass in both
builds. The explicit concept-parameter workaround keeps nvcc 12.6.85 off the
name-hiding ICE and the RNG main is preserved. CPU OpenMP with
`GOMP_SPINCOUNT=0` (passive) at `N=1e5`, `C=128`/`P=1024`, `Q=1`:
OMP=1 2.88 s, OMP=2 2.38 s (best), OMP=4 2.44 s, OMP=8 2.62 s; at `N=1e6`
OMP=2 Q1 23.71 s, OMP=2 Q8 4.46 s. All histogram sums are exact and no
hangs were observed with passive spinning. The previously recorded counts,
fwk394 CPU 194/194 and hal CUDA 246/246, are **historical validation on a
shared host** (`hal` reports `sharing=unknown`) and are **not the final
current validation**. CUDA 13.4/`sm_80` was not re-run in this validation,
and no H100 or fwk394 result is updated here. The CUDA build fails its
`[cuda]` tests when no NVIDIA GPU is visible. The HIP build adds eight typed
Flow cases and the particle and photon CLI/output matrix for the HIP backend;
those cases fail when no AMD GPU is visible. On `hal` with the RX 7900 XTX
(`gfx1100`, ROCm 7.2), the HIP build passed 238/238 with 43 HIP cases.

The public API is implemented and documented in
[public API contract](docs/public-api.md). The typed
`Flow<TPayload, TDevice, TStorage>` execution path runs source, transform,
partition, and sink stages. The particle-feedback example declares its graph
with `Flow<Particle>` handles and named outcome ports in a 121-line user file
(`ParticleFeedback.hpp`). `ParticleStages.hpp` remains only for the older
low-level tests. The photon-scattering example runs the same public API
with slab geometry, five events, `Philox4x32x10` draws, typed failure status,
CSV and trace output, and plots. The RNG API change replaced the old Philox
offset map with keyed engines, so fixed-seed particle and photon baselines
changed once. The earlier 33-invocation CLI comparison predates that change.

## Historical measured baseline

The table below is the pre-T5/T7 result. It is not a current integrated run and
does not include the device-aware policy, batching, or the payload budget. The
integrated T13 result is recorded in [docs/benchmarks.md](docs/benchmarks.md).
The `C=8192`, `P=65536` pair used for those runs is now rejected by the payload
budget.

| Backend | N | Wall time | Histogram sum | Peak live chunks (historical pool) |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1,000,000 | 19.45 s | 1,000,000 | not recorded |
| CUDA | 100,000 | 0.58 s | 100,000 | 514 (65542) |
| CUDA | 1,000,000 | 2.24 s | 1,000,000 | 857 (65542) |
| CUDA | 100,000,000 | 187.5 s | 100,000,000 | 877 (65542) |

The historical `N=1e8` run took about `5.33e5` particles/s over 264 rows with
an exact sum of 100,000,000. `hal` reports `sharing=unknown`, so the wall times
are a shared-host baseline. The historical CPU and CUDA `N=1e6` histograms
matched by direct row diff; that check was not repeated on the integrated main.
Details and the historical invocations are in
[benchmarks](docs/benchmarks.md).

## Layout

```text
include/graphFlow/
  core/    Chunk.hpp, ChunkLedger.hpp, ChunkPool.hpp, ChunkStorage.hpp,
            Config.hpp, DevicePolicy.hpp, LaunchPolicy.hpp,
            PayloadSlotStorage.hpp, PoolBudget.hpp, Reservation.hpp,
            Backend.hpp
  graph/   NodeBase.hpp, RouteTable.hpp, Graph.hpp, GraphBuilder.hpp,
            Classifier.hpp, ClassifyKernel.hpp, ScatterKernel.hpp, Kernels.hpp,
            DenseBinTable.hpp, BinTable.hpp, HashClassifyKernel.hpp,
            HashScatterKernel.hpp, TileDescriptor.hpp
  flow/    Flow.hpp, Execution.hpp, PayloadStorage.hpp, Concepts.hpp,
            NodeHandle.hpp, Payload.hpp, Ports.hpp, Status.hpp, Tuning.hpp
  sched/   GraphRunner.hpp
  rng/     ParticleRng.hpp
  detail/  QueueProbe.hpp
  graphFlow.hpp
examples/particle_feedback/
  Cli.hpp              option parsing and policy resolution
  main.cpp             backend selection, policy output and histogram
  ParticleFunctors.hpp domain functors and named outcome ports
  ParticleTypes.hpp    parameters and histogram formatting
  ParticleFeedback.hpp Flow graph construction and run function
  ParticleStages.hpp   deprecated low-level stages kept for older tests
examples/photon_scattering/
  Cli.hpp              option parsing
  main.cpp             backend selection and output dispatch
  PhotonFunctors.hpp   flight, scatter, and terminal functors
  PhotonOutput.hpp     CSV and trace writers
  PhotonScattering.hpp Flow graph construction and run function
  PhotonTypes.hpp      slab geometry, events, and result types
tests/unit/            reservation, backend/probe, CLI, pool budget, pool/leases,
                        bin table, Philox mapping, device policy, payload storage,
                        CUDA kernel
tests/integration/     particle example, batching, dense and hash pipelines,
                        scripted fixture, pressure, failure paths, FrameSpec,
                        generic DAG, typed Flow execution (CPU, CUDA and HIP),
                        photon scattering (CPU, CUDA and HIP)
tests/reference/       scalar oracle with its own Philox mapping
docs/                  build, architecture, numerics, benchmarks, limitations,
                        API map, plan, acceptance
tools/                 benchmark matrix driver and photon plot script
```

## Build and test

CPU Release:

```sh
cmake -S . -B ./build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace
cmake --build ./build-cpu -j 14
ctest --test-dir ./build-cpu --output-on-failure --no-tests=error
```

CUDA, A30 `sm_80`:

```sh
cmake -S . -B ./build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build ./build-cuda -j 14
ctest --test-dir ./build-cuda --output-on-failure --no-tests=error
```

HIP, AMD Radeon RX 7900 XTX `gfx1100` (runtime; use `gfx90a` only for the
historical compile-only gate):

```sh
cmake -S . -B ./build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_HIP=ON -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dalpaka_DEP_OMP=OFF
cmake --build ./build-hip -j 14
ROCR_VISIBLE_DEVICES=0 ctest --test-dir ./build-hip \
  --output-on-failure --no-tests=error
```

GitHub-hosted runners have no AMD GPU, so a HIP build there is compile-only.
The HIP runtime cases are registered for `GRAPHFLOW_ENABLE_HIP` and fail when
no AMD device is visible; do not treat a compile-only GitHub run as a HIP
runtime result.

CUDA and HIP cannot be enabled in the same build. The CPU backend stays enabled
in accelerator builds. Catch2 3.4.0 must be visible to `find_package`. The exact
pins, the `sm_75` RTX 2080 Ti command, the environment scrub, and the format and
REUSE commands are in [build](docs/build.md).

`GRAPHFLOW_ENABLE_DEBUG_VALIDATION` defaults to `ON` for
`CMAKE_BUILD_TYPE=Debug` and `OFF` for `Release` and `RelWithDebInfo`. Pass
`-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON|OFF` to override it. The two
`[photon][debug]` CTest cases run in both modes: Release asserts that the hot
path has no per-event state branch, and Debug exercises the compiled checks.
Host setup validates the run configuration once. Both default `rosi_v100`
builds pass 195/195 CPU and 247/247 CUDA.

## Run the example

```sh
./build-cpu/graphFlow_particle_feedback --backend cpu --particles 4097 \
  --chunk-size 128 --live-particles 256 --tile-size 128 --batch-chunks 1
```

```text
# selected-backend=cpu selected-device=0 device-name=<host device name>
# policy C=<C> P=<P> T=<T> waves=<waves> blocksPerSm=<blocks> budget=<bytes>B warp=<warp> sms=<sms> cap=<cap>
# queue-probe overlap=yes queues-alias=no second-before-first=yes poll-us=...
# graphFlow_particle_feedback backend=cpu device=0 device-name=<host device name> particles=4097 chunk-size=128 live-particles=256 tile-size=128 batch-chunks=1 seed=42 max-visits=64
# pool-chunks=<S> peak-live-chunks=<peak> peak-reserved-chunks=<reserved>
# weight,count
0,2076
...
```

Arguments: `--backend cpu|cuda|hip` (an uncompiled backend reports a nonzero
configuration error naming the matching `GRAPHFLOW_ENABLE_*` option),
`--device D` (default 0), `--particles N`, `--chunk-size C`, `--live-particles P`,
`--tile-size T`, `--batch-chunks Q` (default 1), `--seed S` (default 0), and
`--max-visits M` (default 64). When `C`, `P` or `T` is absent, `DevicePolicy`
derives it from the selected device. The policy chooses `C` from the wave
estimate, launch limits, and run size, then uses `P = min(N, 2*C)` before budget
clamping. The auto-derived `P` is clamped to the budget, rounded down to a
whole tile when possible; the default path also caps `C` to keep the fixed host
budget small. An explicit `P` is kept exactly and rejected when `P*C` exceeds
the payload budget. `N = 0` prints the header only. Rows are
`weight,count`, strictly increasing in weight, and the counts sum to `N`.

`--batch-chunks Q` is the maximum number of ready input chunks one transform
batch dispatches. Source and sink nodes always dispatch one chunk. The runner
shrinks a transform batch to the largest prefix whose adaptive reserve fits the
current storage credits, so a full `Q` batch is not required for progress.

`--backend` dispatches to `onHost::makeDeviceSelector` with the pinned tags
`api::host`/`deviceKind::cpu`, `api::cuda`/`deviceKind::nvidiaGpu`, and
`api::hip`/`deviceKind::amdGpu`.

## Photon scattering example

`graphFlow_photon_scattering` transports independent photon histories through
a homogeneous plane-parallel slab. The source emits at `(0, 0, 0)` along
`+z`; the graph is source -> propose flight -> advance and route -> scatter ->
propose flight, with transmitted, reflected, absorbed and capped outcomes
feeding one terminal sink. Lengths use one unit and coefficients use inverse
length. The forward mode preserves the direction after a collision; the
isotropic mode samples `mu = 2U-1` and `phi = 2*pi*V`.

Forward run with machine-readable output and a bounded trace:

```sh
mkdir -p /tmp/photon-forward
./build-cpu/graphFlow_photon_scattering --backend cpu --photons 4097 \
  --thickness 1 --sigma-s 3 --sigma-a 0 --mode forward --seed 42 \
  --chunk-size 127 --output-dir /tmp/photon-forward \
  --trace-photons 16 --trace-vertices 256
```

Isotropic run with absorption and the optional detector image:

```sh
mkdir -p /tmp/photon-isotropic
./build-cpu/graphFlow_photon_scattering --backend cpu --photons 4097 \
  --thickness 1 --sigma-s 1 --sigma-a 0.2 --mode isotropic --seed 7 \
  --chunk-size 127 --output-dir /tmp/photon-isotropic \
  --trace-photons 16 --trace-vertices 256
```

`--output-dir` writes `metadata.csv`, `outcomes.csv`,
`scatter_histogram.csv`, `exits.csv`, `traces.csv` and `trace_summary.csv`.
Metadata records the graphFlow commit, alpaka pin, source directories, backend,
device, command line, parameters, units, counts and trace limits. `exits.csv`
contains face exit angles and transverse positions. `traces.csv` stores the
ordered source, scatter and terminal vertices for the deterministic ids
`[0, trace-photons)`. `trace_summary.csv` reports each selected id's stored
vertex count and a `truncated` flag. That flag is independent of the simulation
`--max-scatters` guard: a full trace buffer drops later vertices but the history
still terminates normally or as `capped`.

Plot the saved data (Python and Matplotlib are not C++ or CTest dependencies):

```sh
python3 tools/plot_photon_scattering.py --input /tmp/photon-forward \
  --out /tmp/photon-forward/plots
python3 tools/plot_photon_scattering.py --input /tmp/photon-isotropic \
  --out /tmp/photon-isotropic/plots
```

Each run writes `figure1_paths`, `figure2_scatter_spectrum`,
`figure3_outcomes` and `figure4_detector` as PNG, SVG and PDF. Figure 1 shows
the x-z projection with the scripted three-scatter reflected path from the
unit test. Figure 2 overlays `N*Poisson(sigma_s*L)` in forward mode and shows a
residual panel. Figure 3 shows terminal outcomes and the uncollided fraction
with its binomial error bar against `exp(-sigma_t*L)`. Figure 4 intersects face
exits with finite detector planes and reports hits outside the detector.

Expected forward output for the command above: `transmitted,4097`,
`reflected,0`, `absorbed,0`, `capped,0`, `# terminal-sum,4097` and
`# inconsistent,0`. The collision count follows `Poisson(3)`; about 80.09% of
histories have at least two scatters. For the isotropic command, the expected
uncollided fraction is `exp(-1.2) = 0.3010`. Both runs write one row per
terminal photon to `exits.csv` only for transmitted and reflected outcomes.

Larger statistical diagnostics (`N=200000` forward and `N=100000` isotropic)
are not part of the default CTest suite. Configure with
`-DGRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS=ON` and run
`ctest --test-dir <build> -L diagnostics` to opt in. HIP execution is validated
on `hal` with the RX 7900 XTX (`gfx1100`, ROCm 7.2): the HIP photon CLI and
output cases are part of the 43 HIP cases in the 238/238 CTest run. CUDA
execution was historically validated on `hal` with `sm_80`; the `rosi_v100`
CUDA V100 `sm_70` build passes 247/247 with the explicit concept-parameter
workaround. CUDA 13.4/`sm_80` was not re-run in this validation.

## Semantics

Fields per particle: `id: uint64`, `property: uint32`, `weight: uint32`,
`visit: uint32`. All decisions are integer only.

```text
Generate: property = engine(id, event 0, source)(); weight = 0; visit = 0
Filter:   engine(id, visit, filter)() >= property ? bin 1 : bin 0
A:        engine(id, visit, A)() >= 0x80000000 ? weight += 1000, visit += 1, feedback : sink
B:        engine(id, visit, B)() >= 0x80000000 ? weight += 1,    visit += 1, feedback : sink
Sink:     result[id] = weight
```

`--seed` sets `Tuning::seed`. graphFlow packs `(id, event, purpose)`, mixes the
position with SplitMix64, and constructs `Philox4x32x10(key, 0, 0)` for each
logical position. The functor receives the initialized engine; it never sees a
seed, offset, or purpose count. Weight addition is checked; overflow and an
out-of-range classifier result are hard batch errors. A particle that would feed
back but already reached `--max-visits` is routed to the sink without a mutation
and counted as non-terminating.

The packed key avoids the pinned `Philox4x32x10` constructor-offset defect
without an advance loop. Draw positions depend only on seed, id, event, and the
internal stage purpose, so batching and scatter order do not change them. The
engine-based mapping replaces the old offset map, so fixed-seed baselines change
once. See [numerics](docs/numerics.md).

## Pipeline

1. `GraphRunner` selects a transform batch prefix of up to `Q` ready chunks and
   reserves the adaptive `outputChunkBound(sum(fill), C, B)`.
2. `TransformStage` flattens the batch into one `TileDescriptor` per tile
   (`chunkIndex`, `tileOffset`, `count`), copies the list to the device once,
   and launches one classify kernel whose blocks stride over the descriptor list.
   `ClassifyKernel<Policy, B>` zeroes its shared `counts[B]` table, classifies
   its tile slice, writes `binIds[slot]`, accumulates with block-scope atomics,
   and flushes nonempty bins into the batch-wide `counts[bin]` with device-scope
   atomics. `counts[B]` is the status word.
3. The host copies `counts[B + 1]` to an `allocMapped` staging buffer and the
   queue event covers that copy. After polling the event, it allocates exactly
   `sum_b ceil(total_b / C)` output chunks for the whole batch, builds
   `binBase[B]` plus the flat output chunk list, copies that table to the device,
   zeroes `next[B]` on the same queue, and launches one scatter kernel whose
   blocks stride over the descriptor list.
4. `ScatterKernel<Policy, B>` rebuilds per-tile counts from `binIds`, reserves
   one device-scope range per nonempty bin with `atomicAdd(next[bin], count)`,
   assigns local ranks through a shared cursor, applies the payload mutation, and
   writes the payload to `outChunk * C + offset`. A range may cross any number
   of chunk boundaries.
5. After the scatter event the host verifies `status == 0` and
   `next[bin] == total[bin]`, publishes the output descriptors, releases the
   input chunks and leases, and routes the outputs by bin.

A shared-memory open-addressing hash table (`HashBinTable`, `2*T` entries by
default) covers large bin counts. A full table is a hard batch error, not a
dense fallback. The example uses the dense path.

Every kernel launch uses `FrameSpec` from `graphFlow/core/LaunchPolicy.hpp`.
The frame count is
`min(ceil(work/T), maxBlocksPerGrid, frameCap*multiProcessorCount)` with
`frameCap` defaulting to 8. On CUDA/HIP the frame extent is `T`; on the host
backend it is 1 because the host executors require one real thread per block.
Kernels stride over the batch-wide tile descriptor list with
`worker::blocksInGrid`, so one block may process several tiles. Polling is a
bounded spin followed by `std::this_thread::yield()`; the scheduler never uses
`enqueueHostFn`.

## Queue independence

`device.makeQueue()` may return the same handle, and enqueuing to two queues
never guarantees parallel execution. `graphFlow::probeQueueIndependence`
creates two non-blocking queues, enqueues a bounded delay kernel and a no-op
kernel with events, and records whether the second completed while the first
was still in flight. The example prints the probe as
`# queue-probe overlap=... queues-alias=...`.

The CPU probe reports overlap. The CUDA probe reports distinct queues with no
observed overlap. Correctness never depends on overlap: the scheduler keeps
per-queue FIFO order and one event per batch, so aliased or serialized queues
only change timing. The measured CUDA speedup comes from full-chunk pipelining,
not cross-queue overlap.

## Working set, budget, and ownership

- `P` is the source-admission capacity. When the CLI derives it, the policy
  uses `P = min(N, 2*C)` before budget clamping. A source chunk is admitted only
  while fewer than `P` particles are between source and sink, and its fill is
  capped at `P - live`.
- `Flow` derives the pool as `P` data chunks plus one output bound per
  registered node. The example has a two-output filter and two three-output A/B
  partitions, so the reserve is `2 + 3 + 3 = 8` chunks. `P` data chunks cover
  the live particles because split routing can fragment them down to one
  particle per chunk. `Q` does not enlarge the pool.
- `poolPayloadBudgetBytes` bounds the fixed-capacity pool payload arrays:
  `S*C*payloadBytesPerSlot` must fit. `Flow` derives the budget from the device
  policy plus the exact registered-node reserve, and `Config::validate` rejects
  a pool over budget before allocation.
- A transform batch carries up to `Q` input chunks. It leases every selected
  input from launch until the completion event of its last device stage, and the
  runner releases all leases and inputs after that event.
- The runner has an `abort` hook for a batch that acquired output chunks but
  never published them, for example after a completion-event enqueue failure.
  Teardown drains the node queue, calls `abort`, releases leases, releases owned
  inputs and ready chunks, and rethrows the original error.
- A stall detector reports generated, sunk, and live particles, free and
  reserved chunks, and per-node ready/in-flight depth after a configurable quiet
  period.

The full design is in [architecture](docs/architecture.md).

## Current limitations

- `--batch-chunks Q` has no upper cap beyond the 32-bit parse limit. Large `Q`
  sizes tile-descriptor, table, and staging buffers even when the ready list is
  short. An optional cap is not implemented.
- The frame launch is implemented. `--frame-cap F` bounds the grid to
  `F * multiProcessorCount` blocks (default 8); kernels stride the tile list and
  the host backend uses a unit frame extent.
- HIP execution is validated on one AMD device: `hal` with an AMD Radeon RX
  7900 XTX (`gfx1100`, ROCm 7.2), 238/238 CTest with 43 HIP cases. Other AMD
  architectures and the hash path are not run on HIP. `hal` reports
  `sharing=unknown`, so there is no HIP performance claim. GitHub-hosted HIP
  remains compile-only because its runners have no AMD GPU.
- CUDA 12.5 nvcc hits an internal compiler error in `HashScatterKernel.hpp`.
  CUDA 13.4 compiled `sm_80` historically; nvcc 12.6.85 now builds V100
  `sm_70` with the explicit concept-parameter workaround.
- TSAN is blocked on hal by `unexpected memory mapping`; the non-PIE
  scheduler/pool subset is clean on fwk394 (14/14). No full integrated TSAN run
  is claimed.
- `hal` is shared. The historical throughput is a baseline, not an exclusive
  acceptance number. JURECA `jrc0880.jureca` (4x H100 PCIe 80GB, SM 9.0) is
  available and exclusive per operator, but its gateway sharing flag is unknown
  and no H100 baseline is recorded yet.
- The fixed-capacity pool is budget-capped, but a slot-based pool remains future
  work.
- One in-flight batch per node.
- The generic graph API core is host-only and not yet used by the example.
- CPU OpenMP works with passive spinning (`GOMP_SPINCOUNT=0`); the previous
  multi-region pathology is not observed. CPU CTest on `rosi_v100` passes
  195/195 with OMP=ON at the sm_70 ICE workaround in Release and Debug.
- CUDA V100 `sm_70` on `rosi_v100` passes 247/247 (CUDA 12.6.85 Release and
  Debug) with the explicit concept-parameter workaround.
- Release builds have no per-event payload checks; the Debug default compiles
  them in. Genuine physics failures stay on the `StatusCode` channel in both
  builds.
- CPU-only work that remains open: the tree-wide format and SPDX sweep (T14b).
- No CI runners or `.github/workflows` yet (PR #4 is a draft).
- The hash path is CPU-tested only (low-level); a `rosi_v100` hash-path rerun
  is pending. The dense/hash threshold is static.
- The integrated performance run is recorded in
  [docs/benchmarks.md](docs/benchmarks.md). The numbers in this README are
  historical and pre-T5/T7.
- The result buffer is `O(N)` (about 400 MB at `N=1e8`).

## Verified findings

1. **Philox constructor offset is not equivalent to advancing an engine.** In
   the pinned tree, `PhiloxSingle::skip()` sets the within-block position and
   `advanceState()` resets it, and `skip4()` derives carry words from
   `counter[0] < temp[1]` / `counter[0] < temp[2]`. For a nonzero subsequence
   even `skip4(0)` perturbs counter word 3. `graphFlow::rng::makeEngine` packs
   the position into the key and constructs with subsequence and offset zero.
   See `tests/unit/test_philox.cpp` and [numerics](docs/numerics.md).
2. **The plan's pool formula `ceil(P/C) + K*R` is not sound for split
   routing.** A transform can split one chunk into one chunk per bin, and
   repeated feedback fragments live particles until they occupy one chunk each.
   The pool must hold `P` data chunks plus one output credit per registered
   output. `Q` is an adaptive batch cap and does not change `S`.
   `tests/integration/test_pressure.cpp`, `tests/integration/test_batching.cpp`,
   and `tests/unit/test_pool_budget.cpp` pin the bound and the budget.
3. **The host backend requires a block extent of one.** `api/host/exec/Serial.hpp`
   throws when `numThreads.product() != 1`, so `launchPolicy` sets the frame
   extent to 1 on `api::Host` and to `T` on an accelerator. `numFrames` still
   follows the `T`-based formula, and `T > maxThreadsPerBlock` is rejected before
   the launch.
4. **Histogram multiset equality is the acceptance gate.** Scatter order is
   unspecified, so the 64-bin test compares sorted multisets. Per-particle
   comparisons remain a diagnostic for lost or duplicated ids.
5. **Oracle independence.** `tests/reference/ScalarModel.hpp` and
   `tests/reference/ScalarTransport.hpp` do not include the production RNG
   header. They re-implement the packed stream mapping and construct their own
   zero-offset engines; pinned literals in `tests/unit/test_philox.cpp` fix the
   keyed `(seed, id, event, purpose)` outputs so a shared mapping bug cannot
   hide.
