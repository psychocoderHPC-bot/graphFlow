# graphFlow

Host-scheduled dataflow graph on [alpaka3](https://github.com/alpaka-group/alpaka) with
chunked ownership, exact output-slot planning, and a dense classify/scatter pipeline. The
example `graphFlow_particle_feedback` runs a Generate -> Filter -> {A, B} -> Sink feedback
network and prints the sorted weight histogram.

This is the first vertical slice: CPU execution, dense bins, one input chunk per batch, one
in-flight batch per node.

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
tests/unit/            reservation, pool, bin table, Philox
tests/integration/     particle example and 64-bin dense pipeline
tests/reference/       independent scalar oracle
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
./build-cpu/graphFlow_particle_feedback --backend cpu --particles 4097 --chunk-size 128 --seed 42
```

```text
# graphFlow_particle_feedback backend=cpu device=0 particles=4097 chunk-size=128 seed=42 max-visits=64
# weight,count
0,2011
...
# non-terminating 0
```

Arguments: `--backend cpu` (CUDA/HIP are rejected with an explanatory message),
`--particles N`, `--chunk-size C` (default 128), `--seed S` (default 0), `--max-visits M`
(default 64), `--device D` (default 0). `N = 0` prints the header only. Rows are
`weight,count`, strictly increasing in weight, and the counts sum to `N`.

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
one host ready deque per node, fair round-robin dispatch, and output-chunk reservations in
chunk credits. Polling is a bounded spin followed by `std::this_thread::yield()`; the
scheduler never uses `enqueueHostFn`. Input chunks are released only after the last device
access of their batch.

## Simplifications in this slice

- Dense bins only. No shared-memory hash path.
- One input chunk per classify batch; one in-flight batch per node.
- CPU backend only. `GRAPHFLOW_ENABLE_CUDA` and `GRAPHFLOW_ENABLE_HIP` exist but no device
  code is compiled, and the CLI rejects those backends.
- No `N = 1e8` run and no performance gate. Only functional correctness is measured.
- No scripted (Philox-free) routing fixture, no join/broadcast, no multi-GPU, no
  device-side scheduling.
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
2. **Chunk provisioning from the design document is not sufficient for split-heavy
   feedback.** With two bins a batch can turn one chunk into two, so live chunks grow up to
   one chunk per particle. `derivePoolSize` uses `N + (K + 1) * R + 8`, which keeps at least
   `R` unreserved chunks available so a ready transform can always reserve. The plan's
   `ceil(P/C) + K*R` deadlocked at `N = 512, C = 128` with free chunks stranded in ready
   lists.
3. **The host backend requires a block extent of one.** `api/host/exec/Serial.hpp` throws
   when `numThreads.product() != 1`, so `ThreadSpec{1, T}` is only valid for accelerators;
   `GraphRunner::threadsPerBlock` maps it to 1 for `api::Host`.

## Follow-ups

1. Hash path for large bin counts, with table-full as a hard error.
2. CUDA and HIP execution, plus the queue-independence probe.
3. Multi-batch admission and multiple in-flight batches per node for throughput.
4. Smarter pool provisioning (exact histogram-driven leases instead of the conservative
   bound) to keep the working set independent of `N`.
5. Guarded in-kernel stable grouping or sort if a scan/sort variant is required.
6. A scripted routing fixture (no Philox) and sanitizer runs (ASAN/UBSAN, TSAN).
7. `N = 1e6` smoke and `N = 1e8` acceptance with a frozen throughput target.
