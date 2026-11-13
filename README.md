# graphFlow

Host-scheduled dataflow graph on [alpaka3](https://github.com/alpaka-group/alpaka) with
chunked ownership, exact output-slot planning, and a dense classify/scatter pipeline. The
example `graphFlow_particle_feedback` runs a Generate -> Filter -> {A, B} -> Sink feedback
network and prints the sorted weight histogram.

This is the CPU vertical slice: dense bins, one input chunk per batch, one in-flight batch
per node, a configured source-admission capacity P, and a chunk pool bounded by P rather
than by the total particle count N.

## Layout

```text
include/graphFlow/
  core/    Chunk.hpp, ChunkPool.hpp, Config.hpp, Reservation.hpp
  graph/   Classifier.hpp, ClassifyKernel.hpp, ScatterKernel.hpp, Kernels.hpp, DenseBinTable.hpp
  sched/   GraphRunner.hpp
  rng/     ParticleRng.hpp
examples/particle_feedback/
  main.cpp             CLI and histogram output
  ParticleStages.hpp   generate, transform and sink stages
  ParticleFeedback.hpp graph assembly and run function
tests/unit/            reservation, pool/leases, bin table, Philox mapping
tests/integration/     particle example, 64-bin dense pipeline, scripted fixture, pressure
tests/reference/       scalar oracle with its own Philox mapping
docs/alpaka-api-map.md alpaka3 API selection and verification record
```

## Configure, build, test

```sh
cmake -S /tmp/graphFlow -B /tmp/graphFlow/build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace
cmake --build /tmp/graphFlow/build-cpu -j 14
ctest --test-dir /tmp/graphFlow/build-cpu --output-on-failure --no-tests=error
```

CMake options:

| Option | Default | Meaning |
| --- | --- | --- |
| `GRAPHFLOW_ALPAKA_SOURCE_DIR` | `/workspace` | pinned alpaka3 source tree |
| `GRAPHFLOW_ENABLE_CUDA` | `OFF` | option only; no CUDA device code exists in this slice |
| `GRAPHFLOW_ENABLE_HIP` | `OFF` | option only; no HIP device code exists in this slice |
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
# graphFlow_particle_feedback backend=cpu device=0 particles=4097 chunk-size=128 live-particles=256 seed=42 max-visits=64
# pool-chunks=262 peak-live-chunks=259 peak-reserved-chunks=6
# weight,count
0,2076
...
```

Arguments: `--backend cpu` (CUDA/HIP are rejected with an explanatory message),
`--particles N`, `--chunk-size C` (default 128), `--live-particles P` (default
`min(N, 8*C)`), `--seed S` (default 0), `--max-visits M` (default 64), `--device D`
(default 0). `N = 0` prints the header only. Rows are `weight,count`, strictly
increasing in weight, and the counts sum to `N`. The header also reports the derived pool
size and the peak live/reserved chunk counts measured by the pool.

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

1. `ClassifyKernel<Policy, B>` processes one chunk per block. Per tile it zeroes a shared
   `counts[B]` table, classifies, writes `binIds[slot]`, accumulates with block-scope
   atomics, and flushes nonempty bins into `counts[bin]` with device-scope atomics.
   `counts[B]` is the status word.
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

- Dense bins only. No shared-memory hash path.
- One input chunk per classify batch; one in-flight batch per node.
- CPU backend only. `GRAPHFLOW_ENABLE_CUDA` and `GRAPHFLOW_ENABLE_HIP` exist but no device
  code is compiled, and the CLI rejects those backends.
- No `N = 1e8` run and no performance gate. Only functional correctness is measured.
- No join/broadcast, no multi-GPU, no device-side scheduling. There is a test-only
  Philox-free scripted routing fixture for the plan section 9.3 rows.
- CUDA/HIP execution is not enabled. `threadsPerBlock` clamps the non-host branch to
  `maxThreadsPerBlock` but no accelerator code path is compiled in this slice.
- The host backend executes one real thread per block, so the CPU kernels run with a block
  extent of 1 while keeping the tile and barrier structure that accelerators need.

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
   `GraphRunner::threadsPerBlock` maps it to 1 for `api::Host` and clamps the non-host branch
   to `maxThreadsPerBlock`.
4. **Split routing also invalidates a per-particle acceptance gate.** The acceptance contract
   remains histogram multiset equality; the 64-bin test compares sorted multisets so a
   different valid scatter order cannot fail.
5. **Oracle independence.** `tests/reference/ScalarModel.hpp` no longer includes the
   production RNG header. It constructs `Philox4x32x10(seed, id, 0)` and applies the
   draw-index formula locally; pinned literals in `tests/unit/test_philox.cpp` fix the
   `(seed, id, visit, purpose)` mapping so a shared offset bug cannot hide.

## Remaining gaps

1. Shared-memory hash path for large bin counts, with table-full as a hard error.
2. CUDA and HIP execution, including the queue-independence probe; only the CPU path runs.
3. Multi-batch admission and multiple in-flight batches per node for throughput.
4. `N = 1e6` CI smoke and `N = 1e8` manual acceptance with a frozen throughput target.
5. ASAN/UBSAN and TSAN runs for the host scheduler and kernels.
6. Guarded in-kernel stable grouping or sort if a scan/sort variant is required.
7. Smarter, histogram-driven leases could lower the `P + K*R` bound; it is already
   independent of `N`.
