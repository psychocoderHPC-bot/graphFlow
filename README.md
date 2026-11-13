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
- [Approved plan](docs/plan.md)
- [Missing work](docs/missing.md)
- [Acceptance record](docs/acceptance.md)

## Status

Integrated main: `d5fa45f3cb9298f5e288e2b750d326b56f8ba47b` (T1, T2, T3, T5,
T7, T13, T14a). CPU and CUDA execution are validated on the recorded integrated
runs. HIP compiles the full tree for `gfx90a` with `amdclang++`, but no AMD
device is visible, so HIP execution is not claimed.

The runner has one non-blocking queue per node, one in-flight batch per node,
device-derived `C`/`P`/`T`, up to `Q` ready input chunks per transform batch, a
source-admission capacity `P`, and a chunk pool bounded by `P` rather than by
`N`. Source admission requires `min(C, P, remaining)` particles per admitted
chunk. Consumer ready lists still dispatch partial chunks immediately.

Transform batches flatten every input chunk into a compact `TileDescriptor`
list and issue one classify launch and one scatter launch per batch. The batch
reserves the adaptive `outputChunkBound(sum(fill), C, B)`; under fragmentation
the largest fitting prefix of the ready list is dispatched. The pool stays
`S = P + K*R(C)`, so `Q` changes launch amortisation and buffer sizing, not the
pool bound.

The format/REUSE foundation is in main: `.clang-format` (clang-format 20.1.8),
`.reuse/dep5`, and `LICENSES/MPL-2.0.txt`. The tree-wide format and SPDX sweep
remains open (T14b), and there is no CI workflow yet. See
[known limitations](docs/known-limitations.md).

Recorded integrated tests: fwk394 CPU 74/74, hal CUDA 75/75. The totals come
from the supplied integrated acceptance evidence; this document does not claim
a local run. The CUDA build adds a kernel execution test (`[cuda]`) that fails
when no NVIDIA GPU is visible.

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
  core/    Chunk.hpp, ChunkPool.hpp, Config.hpp, DevicePolicy.hpp, PoolBudget.hpp,
           Reservation.hpp, Backend.hpp
  graph/   Classifier.hpp, ClassifyKernel.hpp, ScatterKernel.hpp, Kernels.hpp,
           DenseBinTable.hpp, BinTable.hpp, HashClassifyKernel.hpp,
           HashScatterKernel.hpp, TileDescriptor.hpp
  sched/   GraphRunner.hpp
  rng/     ParticleRng.hpp
  detail/  QueueProbe.hpp
examples/particle_feedback/
  Cli.hpp              option parsing and policy resolution
  main.cpp             backend selection, policy output and histogram
  ParticleStages.hpp   generate, batched transform, and sink stages
  ParticleFeedback.hpp graph assembly and run function
tests/unit/            reservation, backend/probe, CLI, pool budget, pool/leases,
                       bin table, Philox mapping, device policy, CUDA kernel
tests/integration/     particle example, batching, dense and hash pipelines,
                       scripted fixture, pressure, failure paths
tests/reference/       scalar oracle with its own Philox mapping
docs/                  build, architecture, numerics, benchmarks, limitations,
                       API map, plan, acceptance
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

HIP, compile-only for `gfx90a`:

```sh
cmake -S . -B ./build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_HIP=ON -DGRAPHFLOW_ENABLE_OPENMP=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dalpaka_DEP_OMP=OFF
cmake --build ./build-hip -j 14
```

CUDA and HIP cannot be enabled in the same build. The CPU backend stays enabled
in accelerator builds. Catch2 3.4.0 must be visible to `find_package`. The exact
pins, the `sm_75` RTX 2080 Ti command, the environment scrub, and the format and
REUSE commands are in [build](docs/build.md).

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
clamping. The budget caps `P`; the default path also caps `C` to keep the fixed
host budget small. The clamp rounds down to a whole tile when possible. Explicit overrides are preserved when they fit the device
limits and the budget; an over-budget `P` is reduced to the largest feasible
value. `N = 0` prints the header only. Rows are
`weight,count`, strictly increasing in weight, and the counts sum to `N`.

`--batch-chunks Q` is the maximum number of ready input chunks one transform
batch dispatches. Source and sink nodes always dispatch one chunk. The runner
shrinks a transform batch to the largest prefix whose adaptive reserve fits the
current storage credits, so a full `Q` batch is not required for progress.

`--backend` dispatches to `onHost::makeDeviceSelector` with the pinned tags
`api::host`/`deviceKind::cpu`, `api::cuda`/`deviceKind::nvidiaGpu`, and
`api::hip`/`deviceKind::amdGpu`.

## Semantics

Fields per particle: `id: uint64`, `property: uint32`, `weight: uint32`,
`visit: uint32`. All decisions are integer only.

```text
Generate: property = Philox(seed, id, offset 0); weight = 0; visit = 0
Filter:   draw(seed, id, visit, Filter) >= property ? bin 1 : bin 0
A:        draw(seed, id, visit, A) >= 0x80000000 ? weight += 1000, visit += 1, feedback : sink
B:        draw(seed, id, visit, B) >= 0x80000000 ? weight += 1,    visit += 1, feedback : sink
Sink:     result[id] = weight
```

Decision offsets are `4 * (1 + 3 * visit + purpose)` with purpose 0 for Filter,
1 for A, and 2 for B. The property draw uses offset 0. Weight addition is
checked; overflow and an out-of-range classifier result are hard batch errors. A
particle that would feed back but already reached `--max-visits` is routed to the
sink without a mutation and counted as non-terminating.

The pinned `Philox4x32x10` constructor offset is defective. `rng::drawAt`
constructs at offset zero and advances the public engine. Draw positions depend
only on seed, particle id, visit, and purpose, so batching and scatter order do
not change them. See [numerics](docs/numerics.md).

## Pipeline

1. `GraphRunner` selects a transform batch prefix of up to `Q` ready chunks and
   reserves the adaptive `outputChunkBound(sum(fill), C, B)`.
2. `TransformStage` flattens the batch into one `TileDescriptor` per tile
   (`chunkIndex`, `tileOffset`, `count`), copies the list to the device once,
   and launches one classify kernel with one block per descriptor.
   `ClassifyKernel<Policy, B>` zeroes its shared `counts[B]` table, classifies
   its tile slice, writes `binIds[slot]`, accumulates with block-scope atomics,
   and flushes nonempty bins into the batch-wide `counts[bin]` with device-scope
   atomics. `counts[B]` is the status word.
3. The host copies `counts[B + 1]` to an `allocMapped` staging buffer and the
   queue event covers that copy. After polling the event, it allocates exactly
   `sum_b ceil(total_b / C)` output chunks for the whole batch, builds
   `binBase[B]` plus the flat output chunk list, copies that table to the device,
   zeroes `next[B]` on the same queue, and launches one scatter kernel with one
   block per descriptor.
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

The current launch mapping is one `ThreadSpec` with one block per
`TileDescriptor`. A `FrameSpec` refactor is still pending. Polling is a bounded
spin followed by `std::this_thread::yield()`; the scheduler never uses
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
- The pool is `S = P + K*R(C)` chunks, with `K = 3` transform nodes and
  `R(C) = outputChunkBound(C, C, B)`. `P` data chunks cover the live particles
  because split routing can fragment them down to one particle per chunk;
  `K*R(C)` is the transform output reserve. `Q` does not enlarge the pool.
- `poolPayloadBudgetBytes` bounds the fixed-capacity pool payload arrays:
  `S*C*payloadBytesPerSlot` must fit. The CLI uses the policy memory budget plus
  the exact `K*R(C)` reserve. `Config::validate` rejects a pool over budget
  before allocation.
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
- The launch still uses `ThreadSpec` with one block per tile descriptor. The
  `FrameSpec` refactor is pending.
- No AMD GPU is visible on hal. HIP is compile-only for `gfx90a`; HIP execution
  is blocked until an AMD runtime device is available; no HIP runtime result is
  claimed.
- CUDA 12.5 nvcc hits an internal compiler error in `HashScatterKernel.hpp`.
  CUDA 13.4 is the validated compiler.
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
- No generic graph API yet.
- No CI runners or `.github/workflows` yet.
- The hash path is CPU-tested only. The dense/hash threshold is static.
- The integrated performance run is recorded in
  [docs/benchmarks.md](docs/benchmarks.md). The numbers in this README are
  historical and pre-T5/T7.
- The result buffer is `O(N)` (about 400 MB at `N=1e8`).

## Verified findings

1. **Philox constructor offset is not equivalent to advancing an engine.** In
   the pinned tree, `PhiloxSingle::skip()` sets the within-block position and
   `advanceState()` resets it, and `skip4()` derives carry words from
   `counter[0] < temp[1]` / `counter[0] < temp[2]`. For a nonzero subsequence
   even `skip4(0)` perturbs counter word 3. `graphFlow::rng::drawAt` constructs
   at offset zero and advances the public engine. See
   `tests/unit/test_philox.cpp` and [numerics](docs/numerics.md).
2. **The plan's pool formula `ceil(P/C) + K*R` is not sound for split
   routing.** A transform can split one chunk into one chunk per bin, and
   repeated feedback fragments live particles until they occupy one chunk each.
   The pool must hold `P` data chunks plus the `K*R` output credits. `Q` is an
   adaptive batch cap and does not change `S`. `tests/integration/
   test_pressure.cpp`, `tests/integration/test_batching.cpp`, and
   `tests/unit/test_pool_budget.cpp` pin the bound and the budget.
3. **The host backend requires a block extent of one.** `api/host/exec/Serial.hpp`
   throws when `numThreads.product() != 1`, so `ThreadSpec{1, T}` is only valid
   for accelerators; `GraphRunner::threadsPerBlock` maps it to 1 for `api::Host`.
   The accelerator branch validates against `maxThreadsPerBlock` and throws
   rather than passing an invalid extent.
4. **Histogram multiset equality is the acceptance gate.** Scatter order is
   unspecified, so the 64-bin test compares sorted multisets. Per-particle
   comparisons remain a diagnostic for lost or duplicated ids.
5. **Oracle independence.** `tests/reference/ScalarModel.hpp` does not include
   the production RNG header. It constructs `Philox4x32x10(seed, id, 0)` and
   applies the draw-index formula locally; pinned literals in
   `tests/unit/test_philox.cpp` fix the `(seed, id, visit, purpose)` mapping so a
   shared offset bug cannot hide.
