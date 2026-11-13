# alpaka3 API map

Every accelerator primitive used by graphFlow was checked against the pinned alpaka3 tree
under `/workspace` before use. Pinned revision:
`eb677e8d7f70d79b35df25d9002f94ba669b6f44` (2026-10-26), alpaka 3.0.0.

Process: search `include/alpaka`, `example`, `test` and `docs/snippets` for the operation,
read the implementation and one usage, then select the public API. The snippets
`docs/snippets/example/050_kernel.cpp`, `120_sharedMemory.cpp`, `150_atomics.cpp`,
`160_events.cpp`, `170_random.cpp` and `example/tutorial/src/05_kernel.cpp` were read before
writing the kernels. Only public `alpaka::onHost` and `alpaka::onAcc` interfaces are used.

| Capability | Searched symbols | Pinned path | Selected API | Tested backends | Notes |
| --- | --- | --- | --- | --- | --- |
| Queue creation | `makeQueue`, `queueKind`, `QueuePolicyList` | `onHost/Device.hpp:85-131`, `tag.hpp:100-125`, `api/host/Device.hpp:143-163` | `device.makeQueue(alpaka::queueKind::nonBlocking)` | CPU | One queue per node. The host backend creates a fresh `cpu::Queue` with its own callback thread per call, so queues are independent in practice. Independence is not guaranteed by the API. |
| Backend selection | `makeDeviceSelector`, `DeviceSpec`, `api`, `deviceKind`, `ALPAKA_LANG_CUDA`, `ALPAKA_LANG_HIP` | `tag.hpp:190-288`, `api/api.hpp:36-48`, `onHost/DeviceSelector.hpp:16-70`, `onHost/trait.hpp:166-171` | `onHost::makeDeviceSelector(DeviceSpec{api::host, deviceKind::cpu})`, `{api::cuda, deviceKind::nvidiaGpu}`, `{api::hip, deviceKind::amdGpu}` | CPU (CUDA/HIP paths compile-gated, not executed here) | `ALPAKA_LANG_*` is derived from the compiler; a CPU-only build must not instantiate the CUDA/HIP `DeviceSpec` because `DeviceSelector` static_asserts invalid combinations. `Backend.hpp` therefore guards each branch with `#if ALPAKA_LANG_CUDA`/`#if ALPAKA_LANG_HIP` and reports an uncompiled backend as a configuration error. |
| Queue-independence probe | `makeQueue`, `makeEvent`, `enqueue(event)`, `isComplete`, `wait`, `alloc`, `ThreadSpec`, `KernelBundle` | `onHost/Device.hpp:85-170`, `onHost/Queue.hpp:129-275`, `onHost/Event.hpp:99-138` | Two non-blocking queues, a bounded delay kernel plus a no-op kernel, one event per queue, `isComplete()` polling, then `onHost::wait` | CPU | `probeQueueIndependence` records `overlap=yes/no/unknown`, `queuesAlias`, and `secondCompletedWhileFirstInFlight`. It is advisory: the scheduler only requires per-queue FIFO order and completion events, so aliasing queues are correct but may serialize. |
| Events | `makeEvent`, `enqueue(event)`, `isComplete`, `waitFor` | `onHost/Event.hpp:99-138`, `onHost/Queue.hpp:245-270`, `api/host/Event.hpp` | `device.makeEvent()`, `queue.enqueue(event)`, `event.isComplete()` | CPU | The scheduler never uses `enqueueHostFn`; it polls `isComplete()` with bounded spin plus `std::this_thread::yield()`. The event is enqueued after the last operation it covers, including the metadata copy. |
| Device memory | `alloc`, `allocLike` | `onHost/Device.hpp:215-260` | `onHost::alloc<T>(device, Vec{numChunks * C})` | CPU | Pool payload arrays are allocated once. |
| Mapped staging | `allocMapped` | `onHost/Device.hpp:300-320`, `api/host/Device.hpp:277-284` | `onHost::allocMapped<T>(device, Vec{...})` | CPU | Host-visible staging for histogram totals, status words, range counters and the output table. Buffers live as long as their node. |
| Copies and fills | `memcpy`, `memset`, `fill` | `onHost/Queue.hpp:309-455`, `api/host/Queue.hpp:517-682` | `onHost::memcpy(queue, dst, src)`, `onHost::memset(queue, view, 0)`, `onHost::fill(queue, view, value)` | CPU | All are queue tasks, so they order with kernels and are covered by the node event. `next[B]` is zeroed with `onHost::memset` on the same queue before scatter. Host sources for async copies are persistent mapped views, never temporaries. |
| Kernel launch | `ThreadSpec`, `FrameSpec`, `KernelBundle`, `enqueue` | `onHost/ThreadSpec.hpp`, `onHost/FrameSpec.hpp`, `onHost/Queue.hpp:129-200`, `KernelBundle.hpp` | `queue.enqueue(ThreadSpec{numTiles, blockThreads}, KernelBundle{kernel, args...})` | CPU | `ThreadSpec` gives exact block/thread control, which the one-block-per-tile design needs. `numTiles = ceil(fill/T)` (no block for `fill == 0`), so a chunk larger than `T` is spread over several blocks instead of occupying one SM. `FrameSpec` is intentionally not used because it does not guarantee block boundaries. |
| Block extent on the host backend | `cpu::Serial`, `cpu::OmpBlocks`, `maxThreadsPerBlock` | `api/host/exec/Serial.hpp:36-44`, `api/host/exec/OmpBlocks.hpp:38-46`, `onHost/DeviceProperties.hpp:43-56` | `ThreadSpec{numTiles, GraphRunner::threadsPerBlock(device, T)}`; `threadsPerBlock` returns 1 for `api::Host`; accelerators validate `T <= maxThreadsPerBlock` | CPU | The host executors throw unless `numThreads.product() == 1`. The launch therefore distinguishes the logical tile width `T` (used by the kernels for `tileOffset` and `count`) from the real block extent (1 on the host, up to `T` on an accelerator). Kernels stride inside the tile with `makeIdxMap(..., worker::threadsInBlock, IdxRange{count})`, so they are correct for 1 or `T` real threads, and every thread reaches every barrier including inactive tail lanes. A zero extent, or an accelerator extent above `maxThreadsPerBlock`, throws instead of reaching `ThreadSpec`. |
| Index mapping | `makeIdxMap`, `worker::threadsInBlock`, `IdxRange` | `onAcc/interface.hpp:40-170`, `onAcc/WorkerGroup.hpp:118-140`, `onAcc/range.hpp` | `onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{n})` | CPU | Used for every strided loop: shared table zeroing, flush, and particle loops. No hardcoded block size. |
| Atomics | `atomicAdd`, `atomicExch`, `atomicCas`, `scope::Block`, `scope::Device` | `onAcc/atomic.hpp:25-190`, `onAcc/scope.hpp`, `api/host/atomic.hpp:211-235` | `onAcc::atomicAdd(acc, ptr, 1u, onAcc::scope::block)`, `onAcc::atomicCas(acc, ptr, compare, value, onAcc::scope::block)`, `scope::device` | CPU | Shared per-tile histogram and the hash table key/count use block scope; `counts[bin]`, `next[bin]` and the capped counter use device scope. `atomicCas` is the only writer of a hash key (`0xFFFFFFFF` to bin). `scope::system` is rejected by `static_assert` and is not used. |
| Shared memory | `declareSharedVar`, `declareSharedMdArray`, `getDynSharedMem`, `syncBlockThreads` | `onAcc/Acc.hpp:132-200`, `mem/MdSpanArray.hpp:45-218` | `onAcc::declareSharedMdArray<uint32_t, uniqueId()>(acc, CVec<uint32_t, B>{})`, `onAcc::declareSharedMdArray<HashBinEntry, uniqueId()>(acc, CVec<uint32_t, 2*T>{})`, `onAcc::declareSharedVar<uint32_t, uniqueId()>(acc)`, `onAcc::syncBlockThreads(acc)` | CPU | `declareSharedMdArray` is used with `uint32_t` and with the trivially copyable `HashBinEntry` struct. The hash table is a compile-time-sized array (`2*T`, optionally `4*T`) and is reinitialised by the block. Dynamic shared memory (`getDynSharedMem`) is available but not needed for this fixed size; the dense path keeps its three `B`-word tables. The return value is stored by value. |
| RNG | `Philox4x32x10`, `PhiloxSingle`, `skip`, `advanceState` | `rand/engine/philox/philox.hpp:31-75`, `PhiloxSingle.hpp:60-140`, `PhiloxBaseCommon.hpp:55-90` | `alpaka::rand::engine::Philox4x32x10(seed, subsequence, offset)` and `operator()` | CPU | `graphFlow::rng::drawAt` constructs at offset zero and advances, because the constructor offset is not equivalent to advancing (see below). No project RNG algorithm. |
| In-kernel scan/sort | `scan`, `reduce`, `transform`, `sort` | `onHost/algo/scan.hpp`, `onHost/algo/reduce.hpp`, `onHost/algo/transform.hpp` | **Not used** | n/a | These enqueue kernels; they are not in-kernel collectives. The dense path and the large-bin hash path need no in-kernel scan and no sort: the hash path groups by linear probing with `atomicCas` and reserves contiguous ranges with `atomicAdd`. Ballot/shuffle are not used either, so nothing depends on a hardcoded warp size. Any future device-side stable grouping is project code. |

## Project-owned code and why

| Project code | Reason |
| --- | --- |
| `ChunkPool`, `ChunkHandle`, generations, free list | alpaka3 has no chunk ownership or reuse-token concept. |
| `outputChunkBound`, `planDenseBins`, `buildDenseTable` | Graph policy: exact output slot assignment belongs to the graph, not to alpaka. |
| `GraphRunner` (ready deques, routing, credits, termination) | alpaka3 has no graph, DAG or scheduler. |
| `rng::decisionOffset`, `rng::drawAt` | Encodes the particle-id to Philox-offset mapping and normalises the pinned constructor-offset behaviour. |
| `DenseBinTable`, `HashBinTable`, `HashBinEntry`, `hashInsert`, `hashFind` | Compile-time bin-table policy and the open-addressing helpers used by the large-bin path. |
| `ClassifyKernel`, `ScatterKernel`, `HashClassifyKernel`, `HashScatterKernel`, `GenerateKernel`, `SinkKernel` | Typed kernels for the graph; they use only alpaka primitives for launch, indexing, atomics and shared memory. |
| Scalar reference model (tests only) | Independent oracle; no device code. |

## Pinned-behaviour notes

1. **Philox constructor offset.** In `PhiloxSingle`, `skip()` computes the within-block
   position, then `advanceState()` resets `position` to zero, and `skip4()` compares
   `counter[0]` against `temp[1]` and `temp[2]` instead of the corresponding words. With a
   nonzero subsequence, `skip4(0)` changes counter word 3, so `Philox4x32x10(seed, id, k)()`
   differs from advancing a fresh engine even when `k` is a multiple of four.
   `graphFlow::rng::drawAt` composes the public engine instead: construct at offset 0 and
   call it `offset + 1` times. `tests/unit/test_philox.cpp` pins both the wrapper contract
   and the observed pinned behaviour.
2. **Host block extent and tile-per-block launch.** `cpu::Serial`, `cpu::OmpBlocks` and
   `cpu::TbbBlocks` require a thread block extent of one. The transform launch still uses one
   block per tile, `ThreadSpec{ceil(fill/T), threadsPerBlock(device, T)}`, so the grid grows
   with the chunk while the host block extent stays 1. The kernels receive the logical `T`
   separately from the block extent and use it for `tileOffset = tile*T` and
   `count = min(T, fill - tileOffset)`; the inner stride comes from
   `makeIdxMap(..., worker::threadsInBlock, IdxRange{count})`, which uses the real block
   extent internally. A zero `T`, or an accelerator `T` above
   `DeviceProperties::maxThreadsPerBlock`, throws instead of reaching `ThreadSpec`. The
   CUDA/HIP branch compiles when the backend is enabled but was not executed in this
   environment.
3. **No in-kernel scan or sort.** The dense histogram uses shared atomics per tile and a
   device-scope range reservation per nonempty bin. Any scan or sort variant must be
   implemented as project code and measured.
4. **Queue independence is not guaranteed.** `makeQueue()` may return the same handle.
   `probeQueueIndependence` records `overlap=yes/no/unknown` (`QueueProbeResult` also reports
   `queuesAlias`, `secondCompletedWhileFirstInFlight`, `delayIterations`, `pollDurationUs`).
   Correctness never depends on the verdict: the scheduler keeps per-queue FIFO order, one
   event per batch, and fair round-robin submission, so aliased queues only serialize and
   change timing. The probe runs on CPU in `tests/unit/test_backend.cpp`; CUDA/HIP builds
   compile it but no accelerator run was performed here.
5. **Philox workaround is verified in both directions.** The production wrapper
   (`rng::drawAt`) and the scalar oracle (`tests/reference/ScalarModel.hpp`, which does not
   include the production RNG header) each construct `Philox4x32x10(seed, id, 0)` and advance
   locally. `tests/unit/test_philox.cpp` pins literal draws for
   `(seed=42, id=7, visit, purpose)` and compares both implementations, so an offset-mapping
   defect cannot be shared.
6. **Bounded pool.** The pool is `S = P + K*R`, not the plan's `ceil(P/C) + K*R`. Source
   admission is gated by the `P`-particle budget and caps each source fill at `P - live`, so
   the number of data-carrying chunks never exceeds `P`. Split routing fragments chunks, so
   the `ceil(P/C)` input term is only a lower bound. Measured at `N=4097, C=128, P=256`:
   `S=262`, peak live 259, peak reserved 6.
7. **Reader leases.** `ChunkPool` counts reader leases. The runner leases a non-source input
   from launch until the last stage's completion event and `release` rejects a release while a
   lease is outstanding. This makes release-after-last-use an enforced invariant.
8. **Exception teardown.** A throwing hook drains in-flight batches, releases each owned
   input and queued chunk once, and rethrows. `TransformStage` releases output chunks it
   acquired before a plan or finalise error.
9. **Large-bin hash path.** `HashBinTable<T, factor>` sizes its shared table at the next power of
   two of `factor * T` entries (`factor = 2` by default). `HashClassifyKernel` inserts with
   `atomicCas` on the key and `atomicAdd` on the count, flushes each occupied entry to the global
   `total[bin]`, and uses `scope::block` for shared memory and `scope::device` for `total[bin]`.
   `HashScatterKernel` rebuilds the table from the stored `binIds`, reserves
   `next[bin] = atomicAdd(next[bin], count, device)` per entry, and hands out `localRank` with a
   block-scope `atomicAdd`. Both kernels process one tile per block and keep a private shared
   table for that tile, so the table must have at least as many entries as the tile's distinct
   bins (at most `T`); `HashBinTable<T>` sizes it at `2*T` entries. Tiles of the same chunk
   reserve disjoint global ranges from `next[bin]`, so `j = base + rank` spans chunks exactly as
   before. A full table is a hard `BatchError`; there is no dense fallback. The dense path remains
   the correctness reference and is compared against the hash path in
   `tests/integration/test_hash_bins.cpp`.
