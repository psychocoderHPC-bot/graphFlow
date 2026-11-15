<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# alpaka3 API map

Every accelerator helper and API used by graphFlow, mapped to the pinned alpaka3
revision `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (the 3.0.0 tag, 2026-10-26).
Paths are relative to `include/alpaka/` in that revision; line numbers are from
that revision.

The map is derived from the graphFlow sources in this tree. Each row gives the
symbols used, the pinned path, the exact form used in graphFlow, the backend
evidence, and the selection or portability reason. Only public alpaka interfaces
are used: `onHost`, `onAcc`, `math`, `rand`, the value types `Vec`, `CVec`,
`IdxRange`, `KernelBundle`, `DeviceProperties`, and the annotation and assertion
macros. Project policy lives under `graphFlow/` and is listed separately.

## Backend evidence

| Label | Meaning |
| --- | --- |
| CPU runtime | Hosted `cpu` and `sanitizers` jobs build and run CTest. This is the current runtime gate. |
| CUDA compile-only + ancestor runtime | The hosted `cuda-compile` job builds `sm_70` with no device. The last device runs are recorded in `docs/ci.md` at ancestor `f3ef355` (V100 `sm_70`, A30 `sm_80`); they were not repeated at this commit. |
| HIP compile-only + ancestor runtime | The hosted `hip-compile` job builds `gfx90a` with no device. The last device runs are recorded in `docs/ci.md` at ancestors `f3ef355` and `b972eaa` (RX 7900 XTX `gfx1100`); they were not repeated at this commit. |
| CUDA/HIP compile-gated + failure-test instantiation | A CUDA or HIP translation unit instantiates the path, so the hosted compile gate compiles it. The only device runtime is the typed full-table failure test, which reaches the recorded error path; it is not a successful-path run. |
| CPU runtime only | The path is instantiated and tested on the CPU device only. No CUDA/HIP translation unit in this tree instantiates it, so the hosted compile gate is not evidence for it. |
| Not used | The API was evaluated and is not called. |

A compiled CUDA or HIP backend without a visible device fails its tests instead
of skipping. CUDA and HIP are mutually exclusive in one build. The default
example and the successful-path CUDA and HIP typed tests use the dense scatter
path. The hash path is compiled in CUDA and HIP builds through the typed failure
tests: in this tree `tests/integration/test_failure_paths_cuda.cpp:26-30` and
`test_failure_paths_hip.cpp:26-30` call `tests/integration/FailurePaths.hpp:271-288`,
which instantiates `TransformStage<Device, ManyBinsPolicy, HashBinTable<4u>>`.
Those cases execute only the device classify full-table abort. P07b's opt-in
`--workload kernel` example path additionally runs the hash path end to end on
CUDA (`rosi_v100`, `sm_70`, shared scatter); HIP and the public `Flow` graph have
no recorded hash execution on a device. This map makes no performance claim.

## alpaka APIs

| Capability | Symbol(s) used or evaluated | Pinned path | Selected API (actual code) | Backend evidence | Rationale / portability |
| --- | --- | --- | --- | --- | --- |
| Backend detection | `ALPAKA_LANG_CUDA`, `ALPAKA_LANG_HIP` | `core/config.hpp:231-260` | `#if ALPAKA_LANG_CUDA` and `#if ALPAKA_LANG_HIP` in `core/Backend.hpp`; `#if !ALPAKA_LANG_CUDA` and `#if !ALPAKA_LANG_HIP` error guards in the device test files | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | The macros are derived from the compiler. The `Backend` enum stays complete so the CLI can accept all three names and report `BackendUnavailable` for an uncompiled one. A CPU-only build never instantiates a CUDA or HIP device selector. |
| Device selection | `makeDeviceSelector`, `DeviceSpec`, `api::host`, `api::cuda`, `api::hip`, `api::Host`, `api::Cuda`, `api::Hip`, `deviceKind::cpu`, `deviceKind::nvidiaGpu`, `deviceKind::amdGpu`, `deviceKind::NvidiaGpu`, `deviceKind::AmdGpu` | `onHost/DeviceSelector.hpp:15-85`, `onHost/DeviceSpec.hpp:28-70`, `tag.hpp:236-274`, `api/host/Api.hpp:19-39`, `api/cuda/Api.hpp:21-41`, `api/hip/Api.hpp:21-41` | `alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{api, deviceKind})`; `static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Cuda>)`; `alpaka::api::Host` in `if constexpr` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `DeviceSelector` static_asserts `DeviceSpec::isValid()`. Invalid api and device-kind pairs must not be instantiated, so `core/Backend.hpp` guards each branch with the matching language macro. Type tags are used in `static_assert` and `if constexpr`; value tags build the `DeviceSpec`. |
| Device discovery | `getDeviceCount`, `isAvailable`, `makeDevice`, `getDeviceProperties`, `getName`, `getApi`, `getDeviceKind` | `onHost/DeviceSelector.hpp:42-66`, `onHost/Device.hpp:65,179-195`, `onHost/DeviceProperties.hpp:28-33` | `selector.isAvailable()`, `selector.getDeviceCount()`, `selector.makeDevice(idx)`, `selector.getDeviceProperties(idx).getName()`, `device.getName()`, `device.getApi()`, `device.getDeviceKind()` | CPU runtime; CUDA compile-only + ancestor runtime (the ancestor CUDA kernel test prints the device name); HIP compile-only + ancestor runtime | The selector checks availability and the device index before construction. A compiled backend with no visible device throws `BackendUnavailable` or fails the test; it does not fall back to the CPU device. |
| Queue creation | `makeQueue`, `queueKind::nonBlocking`, `queueKind::blocking`, `operator==` | `onHost/Device.hpp:99-131`, `tag.hpp:108-131`, `api/host/Device.hpp:143-163`, `onHost/Queue.hpp:96-101` | `device.makeQueue(alpaka::queueKind::nonBlocking)` for scheduler lanes; `device.makeQueue(alpaka::queueKind::blocking)` for example initialization, readback, and test drains; `firstQueue == secondQueue` in the advisory probe | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | One non-blocking queue per node lane. Blocking queues are confined to bounded setup and readback. `makeQueue()` may return an aliased handle; correctness uses per-queue FIFO order and events, not independence. |
| Queue-independence probe (advisory) | `makeQueue`, `makeEvent`, `enqueue(event)`, `isComplete`, `onHost::wait`, `alloc`, `FrameSpec`, `KernelBundle`, `operator==` | `onHost/Device.hpp:99-170`, `onHost/Queue.hpp:96-101,135-175,245-270`, `onHost/Event.hpp:99-103`, `onHost/interface.hpp:150-159` | Two non-blocking queues, a bounded delay kernel plus a no-op kernel, one event per queue, `isComplete()` polling, then `onHost::wait` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `probeQueueIndependence` records `overlap=yes/no/unknown`, `queuesAlias`, and `secondCompletedWhileFirstInFlight`. The verdict is advisory and timing-sensitive (see `docs/known-limitations.md`). The scheduler only requires per-queue FIFO order and completion events. |
| Events and completion | `makeEvent`, `enqueue(event)`, `isComplete`, `waitFor` | `onHost/Device.hpp:137-165`, `onHost/Queue.hpp:245-270`, `onHost/Event.hpp:99-103` | `device.makeEvent()`, `queue.enqueue(event)`, `event.isComplete()`, `queue.waitFor(event)` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | The event is enqueued after the last operation it covers, including the metadata copy. The scheduler polls `isComplete()` with a bounded spin and `std::this_thread::yield()`. |
| Queue drain | `onHost::wait` | `onHost/interface.hpp:150-159` | `alpaka::onHost::wait(queue)` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | Used for teardown, the queue probe, and test drains. A queue wait complements the completion event; it does not replace event-based accounting. |
| Host task enqueue (tests only) | `enqueueHostFn` | `onHost/Queue.hpp:176-193` | `queue.enqueueHostFn([]{ ... })` in `tests/integration/test_async_execution.cpp` | CPU runtime only | The async execution test uses a short host sleep to keep one batch in flight. The scheduler and the production stages never enqueue host functions. `enqueueHostFnDeferred` and `enqueueNativeFn` are not used. A host task must not block while waiting for another host task on a queue. |
| Frame launch | `FrameSpec`, `getNumFrames`, `getFrameExtents`, `KernelBundle`, `queue.enqueue` | `onHost/FrameSpec.hpp:45-97`, `KernelBundle.hpp:58-100`, `onHost/Queue.hpp:135-175` | `queue.enqueue(policy.frameSpec(), alpaka::KernelBundle{kernel, args...})` with `alpaka::onHost::FrameSpec{frames, frameExtent}` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `numFrames` is the block count and `frameExtent` is the threads-per-block extent. The host backend requires extent 1; accelerators use `T`. `ThreadSpec` is not used. Kernel functors and arguments must satisfy the trivially-copyable `KernelBundle` constraints. |
| Block extent on the host backend | `cpu::Serial`, `cpu::OmpBlocks`, `FrameSpec`, `maxThreadsPerBlock`, `maxBlocksPerGrid` | `api/host/exec/Serial.hpp:26`, `api/host/exec/OmpBlocks.hpp:31`, `api/host/Device.hpp:345-382`, `api/unifiedCudaHip/Device.hpp:402-436`, `onHost/DeviceProperties.hpp:56,76` | `launchPolicy(device, work, T, frameCap)` returns `FrameSpec{frames, 1}` on `api::Host` and `FrameSpec{frames, T}` on an accelerator; `T <= maxThreadsPerBlock` is validated before the launch | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | The host executors require a block extent of 1, so the frame extent is 1 there while `T` stays the logical tile width. Kernels derive the tile width from `T` and stride inside each descriptor with `makeIdxMap(..., worker::threadsInBlock, IdxRange{count})`, so they are correct for 1 or `T` real threads and every thread reaches every barrier, including inactive tail lanes. |
| Device annotation and assertion | `ALPAKA_FN_ACC`, `ALPAKA_ASSERT_ACC`, `ALPAKA_TYPEOF`, `alpaka::unused` | `core/common.hpp:43-56,201`, `core/Assert.hpp:31-53`, `unused.hpp:15-18` | `ALPAKA_FN_ACC` on kernel functors and device helpers; `ALPAKA_ASSERT_ACC` for the RNG key preconditions; `ALPAKA_TYPEOF` for deduced view, queue, and device types; `alpaka::unused(device)` in host-only `if constexpr` branches | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `ALPAKA_FN_ACC` expands to `__device__ __host__` for CUDA and HIP and to nothing on the host. `ALPAKA_ASSERT_ACC` is the device-visible assertion. `alpaka::unused` keeps host-only branches warning-free under `alpaka_COMPILE_PEDANTIC=ON`. |
| Device memory | `alloc` | `onHost/Device.hpp:215-222` | `alpaka::onHost::alloc<T>(device, alpaka::Vec{n})` for pool payload, ids, properties, weights, visits, bin ids, tile descriptors, terminal records, and trace vertices | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | One owning allocation per array. The pool is bounded by `S = P + K*R(C)`, not by `N` or `Q`. The allocation is a typed view; kernels never receive owning pointers. |
| Mapped staging | `allocMapped` | `onHost/Device.hpp:280-305`, `api/host/Device.hpp:277-284` | `alpaka::onHost::allocMapped<T>(device, alpaka::Vec{n})` for status words, counts, next, table, tile list, capped counter, and the output table | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | Host-visible staging. Host sources for async copies are persistent mapped views, never temporaries. Buffers live as long as their node. Device access to mapped memory goes over the interconnect; graphFlow uses it for host-written metadata and copies, not as the particle payload path. |
| Copies and fills | `memcpy`, `memset`, `fill` | `onHost/Queue.hpp:310-454`, `api/host/Queue.hpp:517-682` | `alpaka::onHost::memcpy(queue, dst, src)` and `memcpy(queue, dst, src, alpaka::Vec{n})`; `alpaka::onHost::memset(queue, view, std::uint8_t{0})`; `alpaka::onHost::fill(queue, view, value)` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | All are queue tasks, so they order with kernels and are covered by the node event. `memset` is byte-wise and `fill` is element-wise. `next[B]` is zeroed on the same queue before scatter. |
| Vector and view helpers | `Vec`, `CVec`, `getExtents`, `getPitches`, `getSubView`, `data` | `CVec.hpp:31`, `Vec.hpp:118-170`, `mem/MdSpan.hpp:104-236`, `mem/View.hpp:201-290` | `alpaka::Vec{std::uint64_t{n}}`, `alpaka::CVec<T, N>{}`, `view.getExtents().x()`, `view.getPitches()`, `view.getSubView(alpaka::Vec{n})`, `view.data()` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `Vec` carries runtime extents; `CVec` carries compile-time extents for shared arrays. `getSubView` limits the batch-wide tile list to the live prefix. `.data()` is used for host metadata access and for flat pointers to the probe kernels; production kernels take typed views. |
| Typed kernel views | `concepts::IMdSpan`, `onAcc::concepts::Acc` | `mem/concepts/IMdSpan.hpp:91`, `onAcc/Acc.hpp:111-122` | `alpaka::concepts::IMdSpan` parameters; `alpaka::onAcc::concepts::Acc auto const& acc` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | Kernels take non-owning typed views; the host keeps ownership in `ChunkPool` and per-node buffers. The `Acc` concept constrains the kernel signature. No pointer arithmetic outside indexed access to shared and global arrays. |
| Index mapping | `makeIdxMap`, `worker::threadsInBlock`, `worker::blocksInGrid`, `worker::threadsInGrid`, `IdxRange` | `onAcc/interface.hpp:57-170`, `onAcc/WorkerGroup.hpp:103-112`, `mem/IdxRange.hpp:22-51` | `alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::blocksInGrid, alpaka::IdxRange{n})`; `worker::threadsInBlock` per tile; `worker::threadsInGrid` for generate and sink ranges | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | Blocks stride over the batch-wide descriptor list, and each tile's particle loop is block-local. No hardcoded block size and no `blockIdx == tile` assumption. `onAcc::range::*` is not used; graphFlow passes `worker::*` groups explicitly. |
| Shared memory and barriers | `declareSharedVar`, `declareSharedMdArray`, `syncBlockThreads`, `uniqueId`, `CVec` | `onAcc/Acc.hpp:132-200`, `mem/MdSpanArray.hpp:43-66`, `UniqueId.hpp:57` | `alpaka::onAcc::declareSharedMdArray<uint32_t, uniqueId()>(acc, alpaka::CVec<uint32_t, B>{})`; `declareSharedVar<uint32_t, uniqueId()>(acc)`; `alpaka::onAcc::syncBlockThreads(acc)`; `declareSharedMdArray<HashBinEntry, uniqueId()>` in the hash path | Dense path: CPU runtime, CUDA compile-only + ancestor runtime, HIP compile-only + ancestor runtime. Hash `HashBinEntry` table: CPU runtime and P07b CUDA kernel-workload runtime on `rosi_v100`; CUDA/HIP compile-gated + typed failure-test instantiation. The typed failure test reaches only the full-table classify abort; HIP successful hash execution is not recorded. | Static shared memory only; `getDynSharedMem` is available but not used. A distinct `uniqueId()` per declaration avoids name collisions. The dense scatter keeps three `B`-word arrays; the hash path sizes its array from the compile-time tile width. |
| Atomics | `atomicAdd`, `atomicExch`, `atomicCas`, `scope::block`, `scope::device`, `scope::system` | `onAcc/atomic.hpp:25-190`, `onAcc/scope.hpp:45-93`, `api/host/atomic.hpp:211-235` | `atomicAdd(..., scope::block)` for shared counts; `atomicAdd(..., scope::device)` for global counts, next, and local ranks; `atomicExch(..., scope::block)` and `scope::device` for status flags; `atomicCas(..., scope::block)` for hash-key insertion | Dense `atomicAdd` and `atomicExch`: CPU runtime, CUDA compile-only + ancestor runtime, HIP compile-only + ancestor runtime. `atomicCas` and the hash-key exchange: CPU runtime and P07b CUDA kernel-workload runtime on `rosi_v100`; CUDA/HIP compile-gated + typed failure-test instantiation. The typed failure test reaches only the full-table classify abort; HIP successful hash execution is not recorded. | Block scope for per-block shared tables; device scope for batch-wide totals, range reservation, and status. `scope::system` is rejected by `static_assert` and is unused. The default `SharedStaging` scatter is the accelerator-tested path; P07b also runs `graphFlow::ScatterMode::Direct` on CUDA at kernel level, and HIP `Direct` execution is not recorded. |
| Large-bin hash path | `HashBinTable`, `hashInsert`, `hashFind`, `HashBinEntry` (project code) on top of `atomicCas`, `atomicAdd`, and `declareSharedMdArray` | `graphFlow/graph/BinTable.hpp`, `HashClassifyKernel.hpp`, `HashScatterKernel.hpp`; alpaka `onAcc/atomic.hpp:25-190`, `onAcc/Acc.hpp:132-200` | `HashBinTable<T, factor>` with `nextPowerOfTwo(factor*T)` entries; `atomicCas` insert, `atomicAdd` count, linear probing, per-entry `base` and `localRank` | CPU runtime and P07b CUDA kernel-workload runtime on `rosi_v100`; CUDA/HIP compile-gated + typed failure-test instantiation. The 299-case CUDA and 290-case HIP full runs at ancestor `ad7208f` executed and passed `CUDA failure: a full hash table aborts without a partial sink write` (CTest #216) and `HIP failure: a full hash table aborts without a partial sink write` (#211); both reach `tests/integration/FailurePaths.hpp:271-288`. Those cases execute the device `HashClassifyKernel` up to the full-table abort. P07b's opt-in `--workload kernel` path runs `HashClassifyKernel` and `HashScatterKernel` end to end on CUDA with shared scatter; HIP hash scatter and the public `Flow` hash path have no device run. | No alpaka hash container exists. Open addressing avoids a sort or scan. The default factor is 2 (`2*T` entries). A full table is a hard batch error; there is no dense fallback. The default example uses the dense path; P07b's opt-in kernel workload records CUDA hash execution, but the public `Flow` graph has no accelerator hash run. |
| RNG engine | `Philox4x32x10`, `operator()` | `rand/engine/philox/philox.hpp:31-75`, `PhiloxSingle.hpp:26-125`, `PhiloxBaseCommon.hpp:26-101` | `alpaka::rand::engine::Philox4x32x10(key, 0, 0)` and `operator()` | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `graphFlow::rng::makeEngine` packs `(id, event, purpose)` into the engine key and never passes a nonzero subsequence or offset. This avoids the pinned constructor-offset defect below. No project RNG algorithm. |
| RNG distributions | `UniformReal`, `interval::CO`, `NormalReal` | `rand/distribution/UniformReal.hpp:258-335`, `rand/distribution/interval.hpp:33-38`, `rand/distribution/NormalReal.hpp` | `alpaka::rand::distribution::UniformReal<double>{}(engine)`; test-only `UniformReal<double, alpaka::rand::interval::CO>{}(engine)` | `UniformReal<double>`: CPU runtime, CUDA compile-only + ancestor runtime, HIP compile-only + ancestor runtime. The explicit `CO` spelling appears in the host-only `test_philox.cpp`. `NormalReal` is not used. | The default interval of `UniformReal<double>` is already `interval::CO`; the test spells it out. Photon flight and angular draws use `UniformReal`. The exponential sample maps `[0, 1)` with `1 - U` so `log` receives a positive value. `NormalReal` keeps Box-Muller state, so each use would need a local object; graphFlow does not use it. |
| Math | `math::constants::pi`, `math::sqrt`, `math::cos`, `math::sin`, `math::max`, `math::log`, `math::isfinite` | `math/constants.hpp:24-55`, `math.hpp:21,63,159,178,208,278` | function-local `constexpr double pi = alpaka::math::constants::pi;`; `alpaka::math::sqrt`, `cos`, `sin`, `max`, `log`, and `isfinite` | `pi`, `sqrt`, `cos`, `sin`, `max`, and `log`: CPU runtime, CUDA compile-only + ancestor runtime, HIP compile-only + ancestor runtime. `isfinite`: CPU runtime when `GRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON` (the default for `CMAKE_BUILD_TYPE=Debug`); device instantiation follows `graphFlow::debug::checksEnabled` and the hosted CUDA and HIP gate is Release, so no device runtime is claimed. | A function-local `constexpr` is device-visible under nvcc; a namespace-scope variable is not. Scalar test oracles stay on `std::` math so a defect cannot be shared with production code. |
| Device properties | `DeviceProperties`, `multiProcessorCount`, `warpSize`, `sharedMemPerBlockBytes`, `maxThreadsPerBlock`, `maxBlocksPerGrid`, `globalMemCapacityBytes` | `onHost/DeviceProperties.hpp:28-81`, `onHost/Device.hpp:179` | `device.getDeviceProperties()`; fields consumed by `LaunchPolicy`, `DevicePolicy`, and the example metadata | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | `LaunchPolicy` uses `maxThreadsPerBlock`, `maxBlocksPerGrid`, and `multiProcessorCount`. `DevicePolicy` uses `warpSize` to distinguish the host profile, `sharedMemPerBlockBytes` for the dense and hash shared-memory cap, and `globalMemCapacityBytes` for the payload budget. `warpSize` is never used as a hardcoded warp width in a loop. |
| Pool payload budget (project) | `payloadBytesPerSlot`, `requiredPayloadBytes`, `validatePayloadBudget`, `payloadBudgetWithReserve` | `core/PoolBudget.hpp`, `core/Config.hpp`, `core/DevicePolicy.hpp` | checked 64-bit arithmetic over the `alpaka::onHost::alloc` extents; `Config::validate` rejects an over-budget pool | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | alpaka exposes allocation but no pool or working-set budget policy. The budget covers `S*C*payloadBytesPerSlot`; `Flow` adds the exact registered-node reserve. |
| Graph topology (project) | `Flow`, `NodeHandle`, `TypedNodeHandle`, `OutputPort`, `NodeBase`, `RouteTable`, `Graph`, `GraphBuilder`, `Hooks`, `Batch` | `flow/Flow.hpp`, `flow/NodeHandle.hpp`, `flow/Ports.hpp`, `graph/NodeBase.hpp`, `graph/RouteTable.hpp`, `graph/Graph.hpp`, `graph/GraphBuilder.hpp`, `sched/GraphRunner.hpp` | Public `Flow<TPayload, TDevice, TStorage>` with `source`, `partition`, `transform`, and `sink` handles, `connect`, `configure(Tuning)`, and `run`; the lower-level host-only `NodeBase::submit/poll/complete/retire` maps to `Hooks::launch/advance/finish/abort`; `RouteTable` maps each bin to one node id; immutable `Graph` attaches through `GraphRunner::addNode/setSource` | CPU runtime for the generic tests; CUDA compile-only + ancestor runtime and HIP compile-only + ancestor runtime for the typed Flow and example runs | alpaka3 has no graph model or scheduler. The graph layer stores topology only, never device function pointers, and leaves `GraphRunner::run()`, batching, and `FrameSpec` untouched. Nodes are caller-owned and non-owning; a feedback edge is an ordinary route to an earlier node. |
| Multi-chunk batch (project) | `Batch`, `batchChunks`, `setBatchChunks`, `TileDescriptor` | `sched/GraphRunner.hpp`, `graph/TileDescriptor.hpp`, `flow/Execution.hpp` | project-owned `Batch` vector of chunk descriptors plus a flat `TileDescriptor` list; one classify and one scatter launch per batch | CPU runtime; CUDA compile-only + ancestor runtime; HIP compile-only + ancestor runtime | alpaka has no batch scheduler or tile-descriptor list. `Q` is a cap; the runner shrinks the prefix to the largest adaptive reserve that fits. `S = P + K*R(C)` does not grow with `Q`. |
| Warp and intrinsics | `warp::ballot`, `any`, `all`, `activemask`, `shfl`, `getSize`; `popcount`, `ffs`, `clz` | `onAcc/warp.hpp:32-266`, `intrinsic.hpp` | **Not used** | n/a (not used) | The dense and hash paths need no ballot, shuffle, or warp-level scan. No control decision uses `shfl`, and nothing hardcodes a warp size. A future small-dense-bin optimization must use `warp::getSize(acc)` and the ballot return type instead of assuming 32. |
| Host element-wise algorithms | `transform`, `concurrent`, `LinearizedIdxGenerator` | `onHost/algo/transform.hpp`, `onHost/algo/concurrent.hpp`, `mem/LinearizedIdxGenerator.hpp` | **Evaluated, not used** for generate, sink, classify, and scatter | n/a (not used) | `transform` writes one output per launch. `concurrent` can write several SoA outputs, but generate also needs a per-lane key-based Philox draw, and the typed stage already shares one `FrameSpec` launch with the batch descriptor mapping. Sink is an id-indexed gather, not an element-wise map. Classify and scatter need per-block shared memory, block and device atomics, and uniform block barriers, so an element-wise algorithm cannot express them. |
| In-kernel scan and sort | `scan`, `reduce`, `sort` | `onHost/algo/scan.hpp`, `onHost/algo/reduce.hpp` | **Not used** | n/a (not used) | These enqueue kernels; they are not in-kernel collectives. The dense path and the large-bin hash path need no in-kernel scan and no sort: the hash path groups by linear probing with `atomicCas` and reserves contiguous ranges with `atomicAdd`. Any future device-side stable grouping is project code. |

**Rule:** use `alpaka::math` functions and `alpaka::math::constants` where the
semantics match. Scalar test oracles stay independent of production headers and
may keep `std::` math so a shared defect cannot pass unnoticed.

## Project-owned code and why

| Project code | Reason |
| --- | --- |
| `ChunkPool`, `ChunkHandle`, generations, leases, free list | alpaka3 has no chunk ownership, reuse-token, or reader-lease concept. |
| `outputChunkBound`, `planDenseBins`, `buildDenseTable` | Graph policy: exact output slot assignment belongs to the graph, not to alpaka. |
| `GraphRunner` (ready deques, routing, credits, termination) | alpaka3 has no graph, DAG, or scheduler. |
| `NodeBase`, `RouteTable`, `Graph`, `GraphBuilder` | alpaka3 has no host-side graph topology or node lifecycle. The layer is topology only: nodes are caller-owned and non-owning, a graph is immutable after `build()`, and attachment reuses the existing `GraphRunner` hook path. |
| `Flow<TPayload, TDevice, TStorage>`, `NodeHandle`, `TypedNodeHandle`, `OutputPort`, `Status`, `Tuning`, `PayloadStorage` | alpaka3 has no typed application-facing graph API. The project layer stores handles and ports only, derives the pool from the registered nodes, and exposes a status channel that the host checks before publishing output. |
| `Batch`, `batchChunks`, `TileDescriptor`, adaptive prefix | alpaka3 has no multi-chunk batch scheduler or compact tile descriptor. The project flattens a batch into one descriptor list and one classify and scatter pair. |
| `PoolBudget`, `payloadBytesPerSlot`, `validatePayloadBudget` | alpaka3 has no pool or working-set budget policy. The project checks `S*C*payloadBytesPerSlot` before allocation. |
| `QueueProbe` and `probeQueueIndependence` | alpaka does not promise independent queues. The probe records the platform verdict; correctness never depends on it. |
| `rng::Engine`, `makeEngine`, `StreamTraits`, `checkedStreamKey` | Encodes the logical `(id, event, purpose)` stream mapping, the packed-key limits, and the zero-offset engine construction that avoids the pinned constructor-offset defect. |
| `DenseBinTable`, `HashBinTable`, `HashBinEntry`, `hashInsert`, `hashFind` | Compile-time bin-table policy and the open-addressing helpers used by the large-bin path. |
| `ClassifyKernel`, `ScatterKernel`, `HashClassifyKernel`, `HashScatterKernel`, `GenerateKernel`, `SinkKernel` | Typed kernels for the graph; they use only alpaka primitives for launch, indexing, atomics, and shared memory. |
| `DevicePolicy`, `makeDevicePolicy`, `payloadFitsBudget`, CLI resolution | Device-aware `C`, `P`, and `T` and memory-budget policy. alpaka exposes the properties but no chunk or working-set policy. |
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
   `FrameSpec{numFrames, frameExtent}` from `launchPolicy`. On CUDA and HIP
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
   handle. `probeQueueIndependence` records `overlap=yes/no/unknown`,
   `queuesAlias`, and `secondCompletedWhileFirstInFlight`. The verdict is
   advisory and timing-sensitive (see `docs/known-limitations.md`); correctness
   only requires per-queue FIFO order, one event per batch, and fair round-robin
   submission.
5. **Key-based streams are verified in both directions.** The production wrapper
   (`rng::makeEngine`) and the scalar oracles
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
   from the batch-wide descriptor list. Tiles of the same chunk reserve disjoint
   global ranges from `next[bin]`, so `j = base + rank` spans chunks exactly as
   before. A full table is a hard `BatchError`; there is no dense fallback. The
   dense path remains the correctness reference and is compared against the hash
   path in `tests/integration/test_hash_bins.cpp` on the CPU device. The CUDA
   and HIP typed failure tests instantiate the hash path and execute only its
   full-table classify abort. P07b's opt-in kernel workload runs both hash
   kernels end to end on CUDA; HIP and the public `Flow` graph have no successful
   hash run on a device.
10. **Device-aware policy is wired into the CLI.** `makeDevicePolicy` is
    unit-tested against CPU `DeviceProperties` and against accelerator-shaped
    public fields. `Cli.hpp` resolves absent `--chunk-size`, `--live-particles`,
    and `--tile-size` from the policy and preserves explicit overrides. The
    accelerator-shaped test pins `C=8192`, `P=16384`, and `T=128` on a nominal
    24 GiB device. The A30 usable budget selects `P=16128` for the auto default
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
13. **Backend evidence is split by path.** CPU CTest is the only current
    runtime gate. CUDA and HIP device runs are recorded at ancestors and in the
    P07b record, and were not repeated at this commit. The CUDA and HIP typed
    failure tests instantiate the hash path and execute only its full-table
    classify abort. The hash success path and `graphFlow::ScatterMode::Direct`
    have CPU coverage plus the P07b CUDA kernel-workload run; HIP hash and
    `Direct` execution have no device run. The hosted CUDA and HIP jobs compile
    only, so they make no execution claim.
