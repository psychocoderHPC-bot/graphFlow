# graphFlow

Host-scheduled dataflow graph on [alpaka3](https://github.com/alpaka-group/alpaka) with
chunked ownership, exact output-slot planning, and dense/hash classify/scatter pipelines. The
example `graphFlow_particle_feedback` runs a Generate -> Filter -> {A, B} -> Sink feedback
network and prints the sorted weight histogram.

This slice has a validated CPU path and a runtime backend selector for CPU, CUDA and HIP:
dense bins plus a large-bin shared-memory hash table, one input chunk per batch, one in-flight
batch per node, a configured source-admission capacity P, a chunk pool bounded by P rather than
by the total particle count N, and a startup queue-independence probe. Source admission requires
`min(C, P, remaining)` particles per admitted chunk, so a full classify/plan/scatter round trip is
amortised; consumer ready lists still dispatch partial chunks immediately.

## Measured on hal

A30, seed 42, `C=8192`, `P=65536`: `N=1e6` in 2.24 s and `N=1e8` in 187.5 s
(about `5.33e5` particles/s), with exact histogram sums and a peak of 877 live chunks in a
65542-chunk pool. `hal` reports `sharing=unknown`, so the throughput is a recorded baseline, not
an exclusive-hardware acceptance claim.
CPU `N=1e6` takes 19.45 s. CPU and CUDA CTest pass 42/42; ASAN/UBSAN pass 42/42. HIP compiles the
full tree for `gfx90a`, but no AMD device is visible, so HIP execution is not claimed. The CUDA
queue probe reports distinct queues with no observed overlap; the measured speedup comes from
full-chunk pipelining, not cross-queue overlap. See `docs/acceptance.md` for the full record.

## Layout

```text
include/graphFlow/
  core/    Chunk.hpp, ChunkPool.hpp, Config.hpp, Reservation.hpp, Backend.hpp
  graph/   Classifier.hpp, ClassifyKernel.hpp, ScatterKernel.hpp, Kernels.hpp, DenseBinTable.hpp,
           BinTable.hpp, HashClassifyKernel.hpp, HashScatterKernel.hpp
  sched/   GraphRunner.hpp
  rng/     ParticleRng.hpp
  detail/  QueueProbe.hpp
examples/particle_feedback/
  main.cpp             CLI, backend selection and histogram output
  ParticleStages.hpp   generate, transform and sink stages
  ParticleFeedback.hpp graph assembly and run function
tests/unit/            reservation, backend/probe, pool/leases, bin table, Philox mapping
tests/integration/     particle example, 64-bin dense and hash pipelines, scripted fixture, pressure, failure paths
tests/reference/       scalar oracle with its own Philox mapping
docs/alpaka-api-map.md alpaka3 API selection and verification record
```

## Configure, build, test

CPU build:

```sh
cmake -S /tmp/graphFlow -B /tmp/graphFlow/build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace
cmake --build /tmp/graphFlow/build-cpu -j 14
ctest --test-dir /tmp/graphFlow/build-cpu --output-on-failure --no-tests=error
```

CUDA and HIP are separate build directories. Each option enables the matching CMake language
and sets `alpaka_DEP_CUDA=ON`/`alpaka_CUDA_NvidiaGpu=ON` or
`alpaka_DEP_HIP=ON`/`alpaka_HIP_AmdGpu=ON` before `add_subdirectory(alpaka)`, so
`alpaka_finalize()` compiles the same example and test sources for the accelerator.
Configuration fails with an explicit message when the toolkit is missing.

```sh
# CUDA, e.g. hal/A30
cmake -S /tmp/graphFlow -B /tmp/graphFlow/build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build /tmp/graphFlow/build-cuda -j 14
ctest --test-dir /tmp/graphFlow/build-cuda --output-on-failure --no-tests=error

# HIP, e.g. an AMD runner (set the runner's gfx target)
cmake -S /tmp/graphFlow -B /tmp/graphFlow/build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build /tmp/graphFlow/build-hip -j 14
ctest --test-dir /tmp/graphFlow/build-hip --output-on-failure --no-tests=error
```

The CPU backend stays enabled in accelerator builds, so `--backend cpu` remains available.
CUDA and HIP cannot be enabled in the same build.

CMake options:

| Option | Default | Meaning |
| --- | --- | --- |
| `GRAPHFLOW_ALPAKA_SOURCE_DIR` | `/workspace` | pinned alpaka3 source tree |
| `GRAPHFLOW_ENABLE_CUDA` | `OFF` | adds CUDA language plus `alpaka_DEP_CUDA`/`alpaka_CUDA_NvidiaGpu` |
| `GRAPHFLOW_ENABLE_HIP` | `OFF` | adds HIP language plus `alpaka_DEP_HIP`/`alpaka_HIP_AmdGpu` |
| `GRAPHFLOW_ENABLE_OPENMP` | `OFF` | links `OpenMP::OpenMP_CXX` when found |
| `GRAPHFLOW_BUILD_TESTS` | `ON` | Catch2 test target `graphFlow_tests` |
| `GRAPHFLOW_BUILD_EXAMPLES` | `ON` | `graphFlow_particle_feedback` |

Tests use the locally installed Catch2 3 (`find_package(Catch2 3 REQUIRED)`) and
`catch_discover_tests`.

## Run the example

```sh
./build-cpu/graphFlow_particle_feedback --backend cpu --particles 4097 --chunk-size 128 \
  --live-particles 256 --seed 42
```

```text
# selected-backend=cpu selected-device=0 device-name=<host device name>
# queue-probe overlap=yes queues-alias=no second-before-first=yes poll-us=...
# graphFlow_particle_feedback backend=cpu device=0 device-name=<host device name> particles=4097 chunk-size=128 live-particles=256 seed=42 max-visits=64
# pool-chunks=262 peak-live-chunks=259 peak-reserved-chunks=6
# weight,count
0,2076
...
```

Arguments: `--backend cpu|cuda|hip` (a backend that was not compiled reports a nonzero
configuration error naming the matching `GRAPHFLOW_ENABLE_*` option), `--device D` (default 0),
`--particles N`, `--chunk-size C` (default 128), `--live-particles P` (default
`min(N, 8*C)`), `--seed S` (default 0), `--max-visits M` (default 64). `N = 0` prints the
header only. Rows are `weight,count`, strictly increasing in weight, and the counts sum to
`N`. The header reports the selected backend, device index and device name, the derived pool
size, and the peak live/reserved chunk counts measured by the pool.

`--backend` dispatches to `onHost::makeDeviceSelector` with the pinned tags
`api::host`/`deviceKind::cpu`, `api::cuda`/`deviceKind::nvidiaGpu`, or
`api::hip`/`deviceKind::amdGpu`. The selected device and backend are printed before the
histogram.

## Semantics

Fields per particle: `id: uint64`, `property: uint32`, `weight: uint32`, `visit: uint32`.
All decisions are integer only.

```text
Generate: property = Philox(seed, id, offset 0); weight = 0; visit = 0
Filter:   draw(seed, id, visit, Filter) >= property ? bin 1 : bin 0
A:        draw(seed, id, visit, A) >= 0x80000000 ? weight += 1000, visit += 1, feedback : sink
B:        draw(seed, id, visit, B) >= 0x80000000 ? weight += 1,    visit += 1, feedback : sink
Sink:     result[id] = weight
```

Decision offsets are `4 * (1 + 3 * visit + purpose)` with purpose 0 for Filter, 1 for A and
2 for B; the property draw uses offset 0. Weight addition is checked; overflow and an
out-of-range classifier result are hard batch errors. A particle that would feed back but
already reached `--max-visits` is routed to the sink without a mutation and counted as
non-terminating.

## Pipeline

1. `ClassifyKernel<Policy, B>` launches one block per tile of a chunk
   (`ceil(fill / T)` blocks). Each block zeroes its shared `counts[B]` table, classifies its
   `[tile*T, min(fill, (tile+1)*T))` slice, writes `binIds[slot]`, accumulates with
   block-scope atomics, and flushes nonempty bins into `counts[bin]` with device-scope
   atomics. `counts[B]` is the status word.
2. The host copies `counts[B + 1]` to an `allocMapped` staging buffer and the queue event
   covers that copy. After polling the event, it allocates exactly
   `sum_b ceil(total_b / C)` output chunks, builds `binBase[B]` plus the flat output chunk
   list, copies that table to a per-node device buffer, zeroes `next[B]` with
   `onHost::memset` on the same queue, and enqueues `ScatterKernel`.
3. `ScatterKernel<Policy, B>` rebuilds per-tile counts from `binIds`, reserves one
   device-scope range per nonempty bin with `atomicAdd(next[bin], count)`, assigns local
   ranks through a shared cursor, applies the policy payload mutation, and writes the
   payload to `outChunk * C + offset`.
4. After the scatter event the host verifies `status == 0` and `next[bin] == total[bin]`,
   publishes the output descriptors, releases the input chunk and routes the outputs by
   bin.

`GraphRunner` owns one non-blocking queue and one event per node, one in-flight batch and
one host ready deque per node, sink-first dispatch, fair round-robin otherwise, and
output-chunk reservations in chunk credits. Polling is a bounded spin followed by
`std::this_thread::yield()`; the scheduler never uses `enqueueHostFn`.

### Queue independence

`device.makeQueue()` may return the same handle, and enqueuing to two queues never guarantees
parallel execution. `graphFlow::probeQueueIndependence` therefore creates two non-blocking
queues, enqueues a bounded delay kernel and a no-op kernel with events, and records whether the
second completed while the first was still in flight. `QueueProbeResult` reports
`overlap` = `yes`/`no`/`unknown`, `queuesAlias`, `secondCompletedWhileFirstInFlight`,
`delayIterations`, and `pollDurationUs`. The example prints the probe as
`# queue-probe overlap=... queues-alias=...`; `tests/unit/test_backend.cpp` runs it on the CPU
backend.

The probe is advisory. Correctness never depends on overlap: if queues alias, the scheduler's
fair round-robin submission still relies on per-queue FIFO order and events, not on parallel
execution. Feedback loops may therefore run slower, but they produce the same histogram.

Working-set control:

- `P` (`--live-particles`, default `min(N, 8*C)`) is the source-admission capacity. A source
  chunk is admitted only while fewer than `P` particles are between source and sink, and its
  fill is capped at `P - live`. Source admission never consults the raw free-list length.
- The pool is `S = P + K*R` chunks, with `K = 3` transform nodes and
  `R = outputChunkBound(C, C, B)`. `P` data chunks cover the live particles (split routing
  can fragment them down to one particle per chunk) and `K*R` is the transform output
  reserve. The plan's `ceil(P/C) + K*R` is a lower bound that deadlocks under split
  feedback; the N=4097, C=128, P=256 run peaks at 259 live chunks, so the P term is
  necessary.
- Non-source input chunks carry a pool reader lease from launch until the last stage's
  completion event. `ChunkPool::release` rejects a release while a lease is outstanding, so
  release-after-last-use is enforced rather than assumed.
- A stall detector reports generated/sunk/live particles, free and reserved chunks, and
  per-node ready/in-flight depth after a configurable quiet period instead of spinning
  forever. Exceptions from a hook drain in-flight batches, release every owned chunk once,
  and rethrow the original error.

## Simplifications in this slice

- The dense path is the correctness reference. A shared-memory open-addressing hash table
  (`HashBinTable`, `2*T` entries by default) covers large bin counts; a chunk whose distinct
  bins exceed the table entries is a hard error, not a dense fallback.
- One input chunk per classify batch; one in-flight batch per node.
- CPU and CUDA are validated on `hal` (A30). HIP compiles the full tree for `gfx90a`, but
  no AMD device is visible, so HIP execution is not claimed.
- `N = 1e8` and the measured throughput are recorded in `docs/acceptance.md`; the number is a
  shared-`hal` baseline, not an exclusive-hardware acceptance claim.
- No join/broadcast, no multi-GPU, no device-side scheduling. There is a test-only
  Philox-free scripted routing fixture for the plan section 9.3 rows.
- The host backend executes one real thread per block, so the CPU kernels run with a block
  extent of 1 while keeping the tile and barrier structure that accelerators need.
  `threadsPerBlock` rejects a zero extent and, on accelerators, an extent above
  `maxThreadsPerBlock`.

## Verified findings

1. **Philox constructor offset is not equivalent to advancing an engine.** In the pinned
   tree, `PhiloxSingle::skip()` sets the within-block position and `advanceState()` resets
   it, and `skip4()` derives carry words from `counter[0] < temp[1]` / `counter[0] <
   temp[2]`. For a nonzero subsequence even `skip4(0)` perturbs counter word 3.
   `graphFlow::rng::drawAt` therefore constructs at offset zero and advances the public
   engine, which matches "offset counts individual 32 bit outputs". See
   `tests/unit/test_philox.cpp` and `docs/alpaka-api-map.md`.
2. **The plan's pool formula `ceil(P/C) + K*R` is not sound for split routing.** A
   transform can split one chunk into one chunk per bin, and repeated feedback fragments
   live particles until they occupy one chunk each. The pool must hold `P` data chunks plus
   the `K*R` output credits. With `ceil(P/C)+K*R = 8` at `P=256, C=128` the scheduler
   deadlocks with free chunks stranded in ready lists; `P + K*R = 262` completes and peaks
   at 259 live chunks. `tests/integration/test_pressure.cpp` pins both the completion and the
   bound. This contradicts the plan and is recorded as such.
3. **The host backend requires a block extent of one.** `api/host/exec/Serial.hpp` throws
   when `numThreads.product() != 1`, so `ThreadSpec{1, T}` is only valid for accelerators;
   `GraphRunner::threadsPerBlock` maps it to 1 for `api::Host`. The accelerator branch
   validates against `maxThreadsPerBlock` and throws rather than passing an invalid extent.
4. **Split routing also invalidates a per-particle acceptance gate.** The acceptance contract
   remains histogram multiset equality; the 64-bin test compares sorted multisets so a
   different valid scatter order cannot fail.
5. **Oracle independence.** `tests/reference/ScalarModel.hpp` no longer includes the
   production RNG header. It constructs `Philox4x32x10(seed, id, 0)` and applies the
   draw-index formula locally; pinned literals in `tests/unit/test_philox.cpp` fix the
   `(seed, id, visit, purpose)` mapping so a shared offset bug cannot hide.

## Remaining gaps

1. The hash-table load factor (`2*T` entries by default) has not been measured; a chunk with
   more distinct bins than entries fails by design.
2. CUDA and HIP execution: the build path and runtime probe exist, but no accelerator run has
   been performed in this environment; the `cuda` and `hip` CI jobs still have to execute
   them.
3. Multi-batch admission and multiple in-flight batches per node for throughput.
4. `N = 1e6` CI smoke and `N = 1e8` manual acceptance with a frozen throughput target.
5. ASAN/UBSAN and TSAN runs for the host scheduler and kernels.
6. Guarded in-kernel stable grouping or sort if a scan/sort variant is required.
7. Smarter, histogram-driven leases could lower the `P + K*R` bound; it is already
   independent of `N`.
