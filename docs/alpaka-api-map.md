<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# alpaka3 API map

Every accelerator primitive used by graphFlow was checked against the pinned
alpaka3 tree under `/workspace` before use. Pinned revision:
`eb677e8d7f70d79b35df25d9002f94ba669b6f44` (2026-10-26), alpaka 3.0.0.

Process: search `include/alpaka`, `example`, `test` and `docs/snippets` for the
operation, read the implementation and one usage, then select the public API.
The snippets `docs/snippets/example/050_kernel.cpp`, `120_sharedMemory.cpp`,
`150_atomics.cpp`, `160_events.cpp`, `170_random.cpp` and
`example/tutorial/src/05_kernel.cpp` were read before writing the kernels. Only
public `alpaka::onHost` and `alpaka::onAcc` interfaces are used.

The HIP build compiles every selected API for `gfx90a`. No AMD device was
visible, so HIP execution is not claimed. Most Catch2 cases construct the CPU
device. The CUDA build adds `tests/unit/test_cuda_kernel.cpp`, which runs a
kernel on the selected NVIDIA GPU. The `--backend cuda` example remains the
device-run evidence for the full particle path.

| Capability | Searched symbols | Pinned path | Selected API | Tested backends | Notes |
| --- | --- | --- | --- | --- | --- |
| Backend selection | `makeDeviceSelector`, `DeviceSpec`, `api`, `deviceKind`, `ALPAKA_LANG_CUDA`, `ALPAKA_LANG_HIP` | `tag.hpp:190-288`, `api/api.hpp:36-48`, `onHost/DeviceSelector.hpp:16-70`, `onHost/trait.hpp:166-171` | `onHost::makeDeviceSelector(DeviceSpec{api::host, deviceKind::cpu})`, `{api::cuda, deviceKind::nvidiaGpu}`, `{api::hip, deviceKind::amdGpu}` | CPU; CUDA example; HIP compiles | `ALPAKA_LANG_*` is derived from the compiler. A CPU-only build must not instantiate the CUDA/HIP `DeviceSpec` because `DeviceSelector` static_asserts invalid combinations. `Backend.hpp` guards each branch with `#if ALPAKA_LANG_CUDA`/`#if ALPAKA_LANG_HIP` and reports an uncompiled backend as a configuration error. |
| Queue creation | `makeQueue`, `queueKind`, `QueuePolicyList` | `onHost/Device.hpp:85-131`, `tag.hpp:100-125`, `api/host/Device.hpp:143-163` | `device.makeQueue(alpaka::queueKind::nonBlocking)` | CPU tests; CUDA example; HIP compiles | One non-blocking queue per node. The host backend creates a fresh `cpu::Queue` with its own callback thread per call, so queues are independent in practice. Independence is not guaranteed by the API. |
| Queue-independence probe | `makeQueue`, `makeEvent`, `enqueue(event)`, `isComplete`, `wait`, `alloc`, `FrameSpec`, `KernelBundle` | `onHost/Device.hpp:85-170`, `onHost/Queue.hpp:129-275`, `onHost/Event.hpp:99-138` | Two non-blocking queues, a bounded delay kernel plus a no-op kernel, one event per queue, `isComplete()` polling, then `onHost::wait` | CPU tests; CUDA probe; HIP compiles | `probeQueueIndependence` records `overlap=yes/no/unknown`, `queuesAlias`, and `secondCompletedWhileFirstInFlight`. The CPU probe reports overlap. The CUDA probe reports distinct queues with `overlap=no`. It is advisory: the scheduler only requires per-queue FIFO order and completion events. |
| Events | `makeEvent`, `enqueue(event)`, `isComplete`, `waitFor` | `onHost/Event.hpp:99-138`, `onHost/Queue.hpp:245-270`, `api/host/Event.hpp` | `device.makeEvent()`, `queue.enqueue(event)`, `event.isComplete()`, `queue.waitFor(event)` | CPU tests; CUDA example; HIP compiles | The scheduler never uses `enqueueHostFn`; it polls `isComplete()` with bounded spin plus `std::this_thread::yield()`. The event is enqueued after the last operation it covers, including the metadata copy. |
| Device memory | `alloc`, `allocLike` | `onHost/Device.hpp:215-260` | `onHost::alloc<T>(device, Vec{numChunks * C})` | CPU tests; CUDA example; HIP compiles | Pool payload arrays, the result buffer, and the per-batch tile-descriptor buffer are allocated once. The pool is bounded by `S = P + K*R(C)`, not by `N` or `Q`. |
| Mapped staging | `allocMapped` | `onHost/Device.hpp:300-320`, `api/host/Device.hpp:277-284` | `onHost::allocMapped<T>(device, Vec{...})` | CPU tests; CUDA example; HIP compiles | Host-visible staging for histogram totals, status words, range counters, the output table, the capped counter, and the `TileDescriptor` list. Buffers live as long as their node. Host sources for async copies are persistent mapped views, never temporaries. |
| Pool payload budget | `payloadBytesPerSlot`, `requiredPayloadBytes`, `validatePayloadBudget`, `payloadBudgetWithReserve` | `core/PoolBudget.hpp`, `core/Config.hpp`, `core/DevicePolicy.hpp` | Project-owned checked 64-bit arithmetic over `onHost::alloc` extents; `Config::validate` rejects an over-budget pool | CPU unit tests; CUDA/HIP compile | alpaka exposes allocation but no pool or working-set budget policy. The budget covers `S*C*payloadBytesPerSlot`; `Flow` adds the exact registered-node reserve to the policy budget. |
| Copies and fills | `memcpy`, `memset`, `fill` | `onHost/Queue.hpp:309-455`, `api/host/Queue.hpp:517-682` | `onHost::memcpy(queue, dst, src)`, `onHost::memset(queue, view, 0)`, `onHost::fill(queue, view, value)` | CPU tests; CUDA example; HIP compiles | All are queue tasks, so they order with kernels and are covered by the node event. `next[B]` is zeroed with `onHost::memset` on the same queue before scatter. |
| Kernel launch | `FrameSpec`, `LaunchPolicy`, `KernelBundle`, `enqueue` | `onHost/FrameSpec.hpp`, `onHost/Queue.hpp:129-200`, `KernelBundle.hpp`, `core/LaunchPolicy.hpp` | `queue.enqueue(LaunchPolicy::frameSpec(), KernelBundle{kernel, args...})`; `numFrames = min(ceil(work/T), maxBlocksPerGrid, frameCap*multiProcessorCount)` | CPU tests; CUDA example; HIP compiles | `FrameSpec{numFrames, frameExtent}` is the only graph launch specification. On CUDA/HIP `numFrames` is the block count and `frameExtent` is `T`; on `api::Host` the extent is 1 and `numFrames` still follows the `T`-based formula. `frameCap` defaults to 8 and is a `Config`/`Parameters`/CLI knob. Kernels stride over the batch-wide tile descriptor list with `worker::blocksInGrid`, so a block may process several tiles and no kernel assumes `blockIdx == tile`. |
| Multi-chunk batch | `Batch`, `batchChunks`, `setBatchChunks`, `TileDescriptor` | `sched/GraphRunner.hpp`, `graph/TileDescriptor.hpp`, `flow/Execution.hpp` | Project-owned `Batch` vector of chunk descriptors plus a flat `TileDescriptor` list; one classify and one scatter launch per batch | CPU integration tests; CUDA compiles; CUDA example | alpaka has no batch scheduler or tile-descriptor list. `Q` is a cap; the runner shrinks the prefix to the largest adaptive reserve that fits. `S = P + K*R(C)` does not grow with `Q`. |
| Graph topology | `Flow`, `NodeHandle`, `TypedNodeHandle`, `OutputPort`, `NodeBase`, `RouteTable`, `Graph`, `GraphBuilder`, `Hooks`, `Batch` | `flow/Flow.hpp`, `flow/NodeHandle.hpp`, `flow/Ports.hpp`, `graph/NodeBase.hpp`, `graph/RouteTable.hpp`, `graph/Graph.hpp`, `graph/GraphBuilder.hpp`, `sched/GraphRunner.hpp` | Public `Flow<TPayload, TDevice, TStorage>` with `source`/`partition`/`transform`/`sink` handles, `connect`, `configure(Tuning)`, `run`; the lower-level host-only `NodeBase::submit/poll/complete/retire` maps to `Hooks::launch/advance/finish/abort`; `RouteTable` maps each bin to one node id; immutable `Graph` attaches through `GraphRunner::addNode/setSource` | CPU unit/integration tests; CUDA/HIP compile; both examples use the `Flow` API | alpaka3 has no graph model or scheduler. The graph layer stores topology only, never device function pointers, and leaves `GraphRunner::run()`, batching, and `FrameSpec` untouched. Nodes are caller-owned and non-owning; a feedback edge is an ordinary route to an earlier node. |
| Block extent on the host backend | `cpu::Serial`, `cpu::OmpBlocks`, `FrameSpec`, `maxThreadsPerBlock` | `api/host/Device.hpp:350-383`, `api/unifiedCudaHip/Device.hpp:403-435`, `onHost/DeviceProperties.hpp:43-56` | `launchPolicy(device, work, T, frameCap)` returns `FrameSpec{frames, 1}` on `api::Host` and `FrameSpec{frames, T}` on an accelerator; `T <= maxThreadsPerBlock` is validated before the launch | CPU tests; CUDA example; HIP compiles | The host executors throw unless the block extent is 1, so the frame extent is 1 there while `T` stays the logical tile width. Kernels stride inside each descriptor with `makeIdxMap(..., worker::threadsInBlock, IdxRange{count})`, so they are correct for 1 or `T` real threads, and every thread reaches every barrier including inactive tail lanes. |
| Index mapping | `makeIdxMap`, `worker::threadsInBlock`, `worker::blocksInGrid`, `worker::threadsInGrid`, `IdxRange` | `onAcc/interface.hpp:40-170`, `onAcc/WorkerGroup.hpp:118-140`, `onAcc/range.hpp` | `onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles})`, `worker::threadsInBlock` per tile, and `worker::threadsInGrid` for the generate/sink element ranges | CPU tests; CUDA example; HIP compiles | Blocks stride over the batch-wide tile list, and each tile's particle loop is block-local. When a capped launch has fewer frames than `T`-sized tiles, the remaining tiles are picked up by later frame iterations instead of being dropped. No hardcoded block size or `blockIdx == tile` assumption. |
| Typed views | `IMdSpan`, `getExtents`, `getPitches`, `getSubView` | `mem/MdSpan.hpp:228-236`, `mem/View.hpp:201-291` | `alpaka::concepts::IMdSpan` kernel parameters, `view.data()`, `view.getExtents()` | CPU tests; CUDA example; HIP compiles | Kernels take non-owning typed views. The host keeps ownership in `ChunkPool` and per-node buffers. No raw device pointer arithmetic beyond `data()` for shared-memory tables. |
| Atomics | `atomicAdd`, `atomicExch`, `atomicCas`, `scope::Block`, `scope::Device` | `onAcc/atomic.hpp:25-190`, `onAcc/scope.hpp`, `api/host/atomic.hpp:211-235` | `onAcc::atomicAdd(acc, ptr, 1u, onAcc::scope::block)`, `onAcc::atomicExch(acc, ptr, 1u, onAcc::scope::device)`, `onAcc::atomicCas(acc, ptr, compare, value, onAcc::scope::block)`, `scope::device` | CPU tests; CUDA example dense path; HIP compiles | Shared per-tile histogram and hash-table key/count use block scope; `counts[bin]`, `next[bin]`, the capped counter, and the status word use device scope. `atomicCas` is the only writer of a hash key (`0xFFFFFFFF` to bin). `scope::system` is rejected by `static_assert` and is not used. |
| Shared memory | `declareSharedVar`, `declareSharedMdArray`, `getDynSharedMem`, `syncBlockThreads` | `onAcc/Acc.hpp:132-200`, `mem/MdSpanArray.hpp:45-218` | `onAcc::declareSharedMdArray<uint32_t, uniqueId()>(acc, CVec<uint32_t, B>{})`, `onAcc::declareSharedMdArray<HashBinEntry, uniqueId()>(acc, CVec<uint32_t, 2*T>{})`, `onAcc::declareSharedVar<uint32_t, uniqueId()>(acc)`, `onAcc::syncBlockThreads(acc)` | CPU tests; CUDA example dense path; HIP compiles | `declareSharedMdArray` is used with `uint32_t` and with the trivially copyable `HashBinEntry` struct. The hash table is a compile-time-sized array (`2*T`, optionally `4*T`) and is reinitialised by the block. Dynamic shared memory is available but not used for this fixed size; the dense path keeps its three `B`-word tables. |
| Warp and intrinsics | `warp::ballot`, `any`, `all`, `activemask`, `shfl`, `getSize`; `popcount`, `ffs`, `clz` | `onAcc/warp.hpp:32-131`, `onAcc/intrinsic.hpp:17-46` | **Not used** | n/a | The dense and hash paths need no ballot, shuffle, or warp-level scan. No control decision uses `shfl`, and nothing hardcodes a warp size. A future small-dense-bin optimization must use `warp::getSize(acc)` and the ballot return type instead of assuming 32. |
| Hash table | `atomicCas`, `atomicAdd`, `scope::Block`, `scope::Device`, `declareSharedMdArray` | `graphFlow/graph/BinTable.hpp`, `HashClassifyKernel.hpp`, `HashScatterKernel.hpp`; alpaka `onAcc/atomic.hpp:25-190`, `onAcc/Acc.hpp:132-200` | `HashBinTable<T, factor>` with `nextPowerOfTwo(factor*T)` entries; `atomicCas` insert, `atomicAdd` count, linear probing, per-entry `base`/`localRank` | CPU integration tests; CUDA/HIP compile only for the hash path | No alpaka hash container exists. The project table avoids a sort or scan. The default factor is 2 (`2*T` entries); the factor is a measurement knob. A full table is a hard batch error. The example uses the dense path, so no accelerator hash execution is recorded. |
| RNG engine | `Philox4x32x10`, `operator()` | `rand/engine/philox/philox.hpp:31-75`, `PhiloxSingle.hpp:60-140`, `PhiloxBaseCommon.hpp:55-90` | `alpaka::rand::engine::Philox4x32x10(key, 0, 0)` and `operator()` | CPU tests; CUDA example; HIP compiles | `graphFlow::rng::makeEngine` packs `(id, event, purpose)` into the engine key and never passes a nonzero subsequence or offset, avoiding the pinned constructor-offset defect. No project RNG algorithm. |
| RNG distributions | `UniformReal`, `NormalReal`, `interval::CO` | `rand/distribution/UniformReal.hpp`, `NormalReal.hpp`, `interval.hpp` | `alpaka::rand::distribution::UniformReal<double>{}(engine)`; integer decisions use `engine()` | CPU tests; CUDA example; HIP compiles | Photon flight and angular draws use the alpaka distributions. The exponential sample maps the `[0, 1)` draw with `1 - U` instead of a local word-to-float conversion. `NormalReal` keeps Box-Muller state, so each use owns a local object. |
| Device properties and policy | `multiProcessorCount`, `warpSize`, `sharedMemPerBlockBytes`, `maxThreadsPerBlock`, `maxBlocksPerGrid`, `globalMemCapacityBytes` | `onHost/DeviceProperties.hpp:36-81` | `DeviceProperties` fields; `graphFlow::makeDevicePolicy`; CLI resolution in `examples/particle_feedback/Cli.hpp` | CPU unit tests; CUDA/HIP compile | The policy derives `C`, `P`, and `T` from the device, caps the payload budget, and preserves explicit overrides. Project code is needed because alpaka exposes the properties but no chunk, working-set, or occupancy policy; the wave estimate uses `targetBlocksPerSm` as a knob. The CLI wires policy defaults and overrides into the example. |
| Host element-wise algorithms | `transform`, `concurrent`, `LinearizedIdxGenerator` | `onHost/algo/transform.hpp`, `onHost/algo/concurrent.hpp`, `mem/LinearizedIdxGenerator.hpp` | **Evaluated, not used** for Generate, Sink, classify, and scatter | n/a | `transform` writes one output per launch. `concurrent` can write several SoA outputs, but Generate also needs a per-lane key-based Philox draw, and the typed stage already shares one `FrameSpec` launch with the batch descriptor mapping. Sink is an id-indexed gather, not an element-wise map. Classify and scatter need per-block shared memory, block/device atomics, and uniform block barriers, so an element-wise algorithm cannot express them. The typed kernels stay. |
| In-kernel scan/sort | `scan`, `reduce`, `sort` | `onHost/algo/scan.hpp`, `onHost/algo/reduce.hpp` | **Not used** | n/a | These enqueue kernels; they are not in-kernel collectives. The dense path and the large-bin hash path need no in-kernel scan and no sort: the hash path groups by linear probing with `atomicCas` and reserves contiguous ranges with `atomicAdd`. Any future device-side stable grouping is project code. |

## Project-owned code and why

| Project code | Reason |
| --- | --- |
| `ChunkPool`, `ChunkHandle`, generations, leases, free list | alpaka3 has no chunk ownership, reuse-token, or reader-lease concept. |
| `outputChunkBound`, `planDenseBins`, `buildDenseTable` | Graph policy: exact output slot assignment belongs to the graph, not to alpaka. |
| `GraphRunner` (ready deques, routing, credits, termination) | alpaka3 has no graph, DAG, or scheduler. |
| `NodeBase`, `RouteTable`, `Graph`, `GraphBuilder` | alpaka3 has no host-side graph topology or node lifecycle. The layer is topology only: nodes are caller-owned and non-owning, a graph is immutable after `build()`, and attachment reuses the existing `GraphRunner` hook path. |
| `Flow<TPayload, TDevice, TStorage>`, `NodeHandle`, `TypedNodeHandle`, `OutputPort`, `Status`, `Tuning`, `PayloadStorage` | alpaka3 has no typed application-facing graph API. The project layer stores handles and ports only, derives the pool from the registered nodes, and exposes a status channel that the host checks before publishing output. |
| `Batch`, `batchChunks`, `TileDescriptor`, adaptive prefix | alpaka3 has no multi-chunk batch scheduler or compact tile descriptor. The project flattens a batch into one descriptor list and one classify/scatter pair. |
| `PoolBudget`, `payloadBytesPerSlot`, `validatePayloadBudget` | alpaka3 has no pool or working-set budget policy. The project checks `S*C*payloadBytesPerSlot` before allocation. |
| `QueueProbe` / `probeQueueIndependence` | alpaka does not promise independent queues. The probe records the platform verdict; correctness never depends on it. |
| `rng::Engine`, `makeEngine`, `StreamTraits`, `checkedStreamKey` | Encodes the logical `(id, event, purpose)` stream mapping, the packed-key limits, and the zero-offset engine construction that avoids the pinned constructor-offset defect. |
| `DenseBinTable`, `HashBinTable`, `HashBinEntry`, `hashInsert`, `hashFind` | Compile-time bin-table policy and the open-addressing helpers used by the large-bin path. |
| `ClassifyKernel`, `ScatterKernel`, `HashClassifyKernel`, `HashScatterKernel`, `GenerateKernel`, `SinkKernel` | Typed kernels for the graph; they use only alpaka primitives for launch, indexing, atomics, and shared memory. |
| `DevicePolicy`, `makeDevicePolicy`, `payloadFitsBudget`, CLI resolution | Device-aware `C`/`P`/`T` and memory-budget policy. alpaka exposes the properties but no chunk or working-set policy. |
| Scalar reference model (tests only) | Independent oracle; no device code. |

## Pinned-behaviour notes

1. **Philox constructor offset.** In `PhiloxSingle`, `skip()` computes the
   within-block position, then `advanceState()` resets `position` to zero, and
   `skip4()` compares `counter[0]` against `temp[1]` and `temp[2]` instead of the
   corresponding words. With a nonzero subsequence, `skip4(0)` changes counter
   word 3, so `Philox4x32x10(seed, id, k)()` differs from advancing a fresh
   engine even when `k` is a multiple of four. `graphFlow::rng::makeEngine`
   therefore puts the complete packed position in the key and constructs with
   subsequence and offset zero. `tests/unit/test_philox.cpp` pins the keyed
   construction, the packed-key bijection, the limits, and the zero-offset
   equivalence.
2. **Frame launch and tile-per-descriptor mapping.** Every launch uses
   `FrameSpec{numFrames, frameExtent}` from `launchPolicy`. On CUDA/HIP
   `numFrames` is the block count and `frameExtent` is `T`; on `api::Host` the
   frame extent is 1 because the host executors require one real thread per
   block. `numFrames` is `min(ceil(work/T), maxBlocksPerGrid,
   frameCap*multiProcessorCount)` with `frameCap` defaulting to 8. The kernels
   stride over the batch-wide `TileDescriptor` list with
   `makeIdxMap(..., worker::blocksInGrid, IdxRange{numTiles})`, so one block may
   process several tiles and a capped grid never drops the remaining work. The
   inner tile slice uses `makeIdxMap(..., worker::threadsInBlock,
   IdxRange{count})` and the real block extent. A zero `T`, a zero `frameCap`, or
   a `T` above `maxThreadsPerBlock` throws before the launch.
3. **No in-kernel scan or sort.** The dense histogram uses shared atomics per
   tile and a device-scope range reservation per nonempty bin. Any scan or sort
   variant must be implemented as project code and measured.
4. **Queue independence is not guaranteed.** `makeQueue()` may return the same
   handle. `probeQueueIndependence` records `overlap=yes/no/unknown`
   (`QueueProbeResult` also reports `queuesAlias`,
   `secondCompletedWhileFirstInFlight`, `delayIterations`, `pollDurationUs`).
   Correctness never depends on the verdict: the scheduler keeps per-queue FIFO
   order, one event per batch, and fair round-robin submission, so aliased queues
   only serialize and change timing. The probe runs on CPU in
   `tests/unit/test_backend.cpp`; the CUDA example runs it on the A30 and reports
   distinct queues with no observed overlap.
5. **Key-based streams are verified in both directions.** The production
   wrapper (`rng::makeEngine`) and the scalar oracles
   (`tests/reference/ScalarModel.hpp`, `tests/reference/ScalarTransport.hpp`,
   which do not include the production RNG header) each construct a zero-offset
   engine from the packed position. `tests/unit/test_philox.cpp` pins literal
   keyed outputs and compares both implementations, so a mapping defect cannot
   be shared.
6. **Bounded pool.** The pool is `S = P + K*R(C)`, not the plan's
   `ceil(P/C) + K*R`. Source admission is gated by the `P`-particle budget and
   caps each source fill at `P - live`, so the number of data-carrying chunks
   never exceeds `P`. Split routing fragments chunks, so the `ceil(P/C)` input
   term is only a lower bound. A transform batch reserves the adaptive
   `outputChunkBound(sum(fill), C, B)` and shrinks its prefix to fit; Q does not
   enlarge `S`. `tests/unit/test_pool_budget.cpp` and
   `tests/integration/test_batching.cpp` pin the budget and the prefix rule.
7. **Reader leases.** `ChunkPool` counts reader leases. The runner leases a
   non-source input from launch until the last stage's completion event and
   `release` rejects a release while a lease is outstanding. This makes
   release-after-last-use an enforced invariant.
8. **Exception teardown.** A throwing hook drains the node queue, calls the
   stage's `abort` hook, releases every outstanding lease, releases each owned
   input and queued chunk once, and rethrows. `TransformStage::abort` releases
   output chunks it acquired before a plan or event-enqueue failure.
9. **Large-bin hash path.** `HashBinTable<T, factor>` sizes its shared table at
   the next power of two of `factor * T` entries (`factor = 2` by default).
   `HashClassifyKernel` inserts with `atomicCas` on the key and `atomicAdd` on
   the count, flushes each occupied entry to the global `total[bin]`, and uses
   `scope::block` for shared memory and `scope::device` for `total[bin]`.
   `HashScatterKernel` rebuilds the table from the stored `binIds`, reserves
   `next[bin] = atomicAdd(next[bin], count, device)` per entry, and hands out
   `localRank` with a block-scope `atomicAdd`. Both kernels use one shared table
   per block and reinitialize it for each tile the block processes, so the table
   must have at least as many entries as that tile's distinct bins (at most `T`);
   `HashBinTable<T>` sizes it at `2*T` entries. A block may process several tiles
   from the batch-wide descriptor list. Tiles of the same chunk reserve
   disjoint global ranges from `next[bin]`, so `j = base + rank` spans chunks
   exactly as before. A full table is a hard `BatchError`; there is no dense
   fallback. The dense path remains the correctness reference and is compared
   against the hash path in `tests/integration/test_hash_bins.cpp`.
10. **Device-aware policy is wired into the CLI.** `makeDevicePolicy` is
    unit-tested against CPU `DeviceProperties` and against accelerator-shaped
    public fields. `Cli.hpp` resolves absent `--chunk-size`, `--live-particles`,
    and `--tile-size` from the policy and preserves explicit overrides. The
    accelerator-shaped test pins `C=8192`, `P=16384`, `T=128` on a nominal 24 GiB
    device. The measured A30 usable budget selects `P=16128` for the auto default
    and for explicit `C=8192` with auto P; explicit `P=16384` and `P=65536` are
    rejected, while an explicit in-budget P is kept exactly.
11. **Batching and payload budget are project policy.** alpaka has no
    multi-chunk batch scheduler, compact tile descriptor, or pool budget. The
    project flattens a batch into `TileDescriptor` values, uses one classify and
    one scatter launch per batch, and checks `S*C*payloadBytesPerSlot` against
    `poolPayloadBudgetBytes` before allocation. Both launches share the batch's
    `LaunchPolicy` frame spec.
12. **The generic graph layer is host-only.** `NodeBase` is a virtual host
    interface; `Graph` and `GraphBuilder` hold non-owning node references. The
    graph snapshots topology and node ids at `build()`, then registers the nodes
    with a fresh `GraphRunner` in id order. `RouteTable` carries no feedback-edge
    flag: a feedback edge is an ordinary destination that names an earlier node.
    v1 allows one source, rejects routes to the source, and has no runtime graph
    mutation. The particle and photon examples use the typed `Flow` API, not
    the lower-level generic core.
