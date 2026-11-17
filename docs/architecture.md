<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Architecture

graphFlow is a host-scheduled dataflow runner for chunked particle payloads.
The host owns topology, admission, routing, and completion. Device code owns
per-tile histogram, hash, and scatter work. The public API is the typed
`Flow<TPayload, TDevice, TStorage>` class in
`include/graphFlow/flow/Flow.hpp`; the particle-feedback and photon-scattering
examples both declare their graphs through `Flow` handles and named outcome
ports. The lower-level `GraphBuilder`/`GraphRunner`/`Hooks` path stays
available through its own headers for custom nodes.

## Public Flow API

`Flow<TPayload, TDevice, TStorage>` in
`include/graphFlow/flow/Flow.hpp` is the application-facing entry point. It
owns the node topology, the payload storage, and the execution runner. Users
register functors through `source`, `partition`, `transform`, and `sink`;
registration returns opaque handles. Users declare edges with `connect` and
launch with `configure(Tuning)` and `run(source, itemCount)`. The `Payload`
concept rejects a payload that is not trivially copyable and default-constructible.
`TStorage` defaults to `DefaultStorage<TPayload, TDevice>`: the canonical
`Particle` payload selects the SoA `ParticleStorage<TDevice>`, every other
payload selects the generic `PayloadSlotStorage<TPayload, TDevice>`. Both have a
statically known `slotBytes`. `SoAStorage<TDevice, Fields...>` is an explicit
opt-in for record payloads and is never selected by the default alias.

`run()` validates the topology, derives the configuration, constructs the typed
stages and the scheduler, and returns `SchedulerStats` only after every queue
and pool chunk has been retired, or it propagates an execution error without
publishing a partial output. The typed execution adapters live in
`flow/Execution.hpp`; the status channel lives in `flow/Status.hpp`.

## Attribute records and column storage

`flow/Attributes.hpp` wraps the vendored llamaLite descriptors (`TagBase`,
`Field`, `Set`) and defines the `Record` adapter with graphFlow validation and
accounting.
`Record<Fields...>` is a value payload backed by a one-element
`llama_lite::SoA`; it is trivially copyable and copying it copies every
attribute. `RecordProxy<Access, Set<Tags...>>` narrows `get`/`set` per node at
compile time, but `load`, conversion to the record type, and `store` still move
the whole record. The proxy narrows access, not the layout.

`SoAStorage<TDevice, Fields...>` is the array-of-structs-over-SoA storage. It
allocates one device column per attribute and one `uint32_t` bin-id column,
each of `numChunks * chunkCapacity` elements. There is no contiguous record
array: `base(handle)` is `handle.index * C` and `slotIndex(handle, row)` is
`base + row`. `loadRecord`/`storeRecord` gather or scatter one record, and
`proxy<Set>(slot)` returns a per-slot proxy. Chunk lifecycle, generations,
reader leases, and the free list stay in the shared `detail::ChunkLedger`, so
the scheduler sees the same `PayloadStorage`/`ChunkStorage` contract as for
`ParticleStorage` and `PayloadSlotStorage`.

Slot accounting matches the other storages:

```text
padded(T)      = ceil(sizeof(T) / alignof(T)) * alignof(T)
attributeBytes = sum of padded(value_type) over all attributes
slotBytes      = attributeBytes + sizeof(std::uint32_t)   // bin-id scratch
```

The canonical particle attribute set accounts 20 + 4 = 24 bytes, the existing
`payloadBytesPerSlot`, so pool sizing, `maxPayloadChunks`, and the budget checks
keep their values. `slotBytes` is a budget and view size; the columns allocate
their own value types and the scratch column is separate.

The typed stages reach the columns through
`detail::StorageAccess<SoAStorage<...>, Record<...>>`, which binds one view per
attribute in declaration order. The launch helpers expand the view tuple into
separate top-level kernel arguments, because alpaka rewrites only top-level
view arguments for the device. `load`/`store` then gather or scatter one
record attribute by attribute.

`SoAStorage` is never a default. `DefaultStorage` still selects
`ParticleStorage` for the canonical `Particle` payload and
`PayloadSlotStorage` for every other payload; `ParticleStorage` remains the
migration source for the deprecated low-level stages. The particle and photon
examples opt into `SoAFlow` with a `Record` payload and matching `SoAStorage`.

## Host scheduler

`GraphRunner` in `include/graphFlow/sched/GraphRunner.hpp` runs one loop on the
calling thread. That thread owns queue submission, event retirement, routing,
the pool free list, staging, and source admission. The runner creates no
scheduler thread and never blocks on a queue in the steady state.

Each node has:

- one non-blocking queue;
- one completion event;
- one host ready deque;
- at most one in-flight batch;
- a `launch` hook, an `advance` hook, a `finish` hook, and an `abort` hook.

A batch holds a vector of chunk descriptors and the first particle id. A source
or sink batch carries one chunk. A transform batch carries up to
`batchChunks` (`Q`) ready input chunks. Q defaults to 1.

The loop order is:

1. Retire completed events and run `advance` or `finish`.
2. Admit one source chunk when the particle budget and the free-list
   precondition allow it.
3. Dispatch sinks first, then nodes in fair round-robin order.
4. Check termination.
5. Back off or report a stall when no progress is possible.

Sinks run first because they release chunks without reserving outputs. A
transform scans the largest requested prefix of its ready deque, computes the
adaptive reserve for that prefix, and shrinks the prefix until the reserve fits
the current storage credits. The selected prefix is removed from `ready` only
after the launch and event enqueue succeed, so a throwing hook cannot leave the
same chunk owned twice.

Termination requires all of: the source is exhausted, every ready deque and
in-flight slot is empty, and no output reservation remains. After the loop the
runner drains every queue and checks that the pool is empty. A quiet period
raises a `SchedulerStallError` with aggregate key=value counters: source
particles/generated/sunk/live, chunk capacity, free, reserved, ready, in-flight
and event-slot aggregates, and their peaks. The `what()` text never contains
node names, node ids, or routes. Per-node ready/in-flight detail is built only
when `Tuning::diagnostics` is enabled and is delivered through an optional
detail callback and `SchedulerStallError::detail()`.

## Chunk registry (Phase 0)

`GraphRunner` owns a `ChunkRegistry` (`core/ChunkRegistry.hpp`) with one record
per physical pool slot for its current incarnation: `{handle, producing node,
owner node, destination node, phase, fill, bin}`. The phases are `free`,
`reserved`, `ready`, `inflight`, and `retired`. The registry is filled at
transitions that already exist: acquire and release are reported by the storage
(`ChunkPool`/`ChunkLedger` optional hooks), routing sets the destination from
`node.route[bin]`, dispatch moves a chunk to its lane, and a sink arrival marks
it retired before the release returns it to the free list. Nothing in the
scheduler reads the registry in Phase 0, so the record can never change a
decision. The constructor seeds every record from the pool's current
generations, and the run-start reset re-seeds in place without allocating.
Every mutator is `noexcept`: an invariant violation records a sticky fault and
leaves the pool untouched, and the run continues. `validateChunkRegistry()`
reports that fault first, then enforces exactly one phase per index, a live
reader lease for every non-source inflight chunk, an empty free-list
intersection with leases, generation agreement with the pool, phase/owner
consistency, and the acquisition/release live-count balance. `SchedulerStats`
exposes the aggregate transition counters, fault count and phase peaks;
`snapshot()` exposes the current phase counts and live chunk count without node
ids.

## Pool, batch reserve, and admission

`ChunkPool` stores fixed-capacity SoA payload arrays. A chunk handle is
`{chunkIndex, generation}`. The generation increments on every release, so a
stale handle is rejected instead of aliasing a reused chunk. Particle `i` of
chunk `k` lives at slot `k*C + i`.

The fixed pool is sized as

```text
S = P + sum over nodes of outputChunkBound(C, C, node.maxBins)
```

`Flow` derives the reserve from the registered nodes. The particle example has
a two-output filter and two three-output A/B partitions, so the reserve is
`2 + 3 + 3 = 8` chunks. `P` is the source-admission capacity in live particles.
`P` data chunks cover the live particles because split routing can fragment
them down to one particle per chunk. The plan's `ceil(P/C) + K*R` is a lower
bound and deadlocks under split feedback. Q does not enlarge `S`.

A transform batch reserves the adaptive amount

```text
outputChunkBound(sum(fill over the batch), C, B)
```

for the chunks it actually takes. The scheduler starts with the largest prefix
of up to Q ready chunks and removes the last chunk until the reserve fits the
available credits. `outputChunkBound` is monotone in the particle count, so the
fitting prefixes are contiguous and the loop finds the largest one. A
fragmented ready list therefore dispatches a smaller batch instead of stalling.
The transform stage's table and descriptor buffers are sized for the largest
Q-chunk batch, but the runtime reserve follows the actual prefix.

The pool payload budget is explicit. `poolPayloadBudgetBytes` bounds
`S*C*payloadBytesPerSlot`, with `payloadBytesPerSlot = 24` bytes for the id,
property, weight, visit, and bin-id arrays. That value equals the
`AttributeList::slotBytes` of the id, property, weight, and visit fields, so an
`SoAStorage` over that field set is budgeted identically. The policy memory
budget covers the `P*C` live working set; `Flow` adds the exact registered-node
output-credit reserve. `Config::validate` rejects a pool over budget before any
allocation. The former `C=8192`, `P=65536` pair exceeds the default budget and
is rejected.

Source admission is gated by particles, not by the raw free-list length. A
source chunk is admitted only when it can carry

```text
min(C, P, remaining)
```

particles. A tail chunk gets the remaining particles and a deliberately small
`P` gets a smaller chunk. A one-particle chunk therefore occurs only when `P=1`
or the tail has one particle. Consumer ready lists still dispatch partial chunks
immediately, so the no-full-chunk rule applies to source generation only.

A non-source input chunk carries a reader lease from launch until the
completion event of the batch's last device stage has fired. `ChunkPool::release`
rejects a release while a lease is outstanding. This enforces
release-after-last-use rather than assuming it. Source chunks are handed
downstream and are not leased. A transform batch leases every selected input
chunk.

## Batching and tile descriptors

`TileDescriptor` in `include/graphFlow/graph/TileDescriptor.hpp` is

```text
{chunkIndex, tileOffset, count}
```

`chunkIndex` selects the chunk in the pool, `tileOffset` is the first particle
slot of the tile within the chunk, and `count` is the number of particles the
tile owns. The host flattens every input chunk of a batch into one descriptor
list and copies it to the device once. Classify and scatter each launch one
kernel whose blocks stride over the descriptor list, so a batch is two launches
regardless of how many chunks or tiles it carries. `ChunkPool` rejects
`numChunks*C > UINT32_MAX`, so the descriptor index arithmetic stays in 32 bits.

The launch mapping is a `FrameSpec` from `launchPolicy`. `numFrames` is
`min(ceil(work/T), maxBlocksPerGrid, frameCap*multiProcessorCount)`, with
`frameCap` defaulting to 8. On CUDA/HIP the frame extent is `T`; on `api::Host`
it is 1. The blocks stride over the descriptor list with
`worker::blocksInGrid`, so a block may process several tiles and a capped grid
never drops the remaining descriptors.

## Classify and scatter

Both dense and hash stages launch one `FrameSpec` per batch. Each block strides
over the batch-wide tile descriptor list with
`onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles})`, so a
block may process several tiles. `T` is passed separately from the real block
extent. On `api::Host` the block extent is one real thread; on an accelerator it
is `T`, rejected when `T` exceeds `maxThreadsPerBlock`. Within a tile the threads
stride with `onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock,
IdxRange{count})`, so the same code is correct for one or for `T` real threads.
Every thread reaches every barrier, including inactive tail lanes.

### Dense path

`ClassifyKernel<Policy, B>` is the correctness reference.

1. Zero a shared `counts[B]` table and synchronize.
2. Classify each particle in the tile slice and store its bin in `binIds`.
3. Accumulate into `counts[bin]` with block-scope atomics.
4. Flush every nonempty bin into the batch-wide device-scope `counts[bin]`
   total.

`counts[B]` is the status word. An out-of-range classifier result sets it and
drops the particle instead of indexing outside the table.

`ScatterKernel<Policy, B>` rebuilds per-tile counts from `binIds`, reserves one
device-scope range per nonempty bin with `atomicAdd(next[bin], count)`, assigns
local ranks through a shared cursor, applies the payload mutation, and writes
the payload to `outChunk*C + offset`. `next[bin]` is shared by every descriptor
of the batch, so ranges of different tiles and chunks are disjoint and a range
may cross any number of output chunk boundaries.

### Hash path

`HashBinTable<T, factor>` sizes a shared open-addressing table at the next power
of two of `factor*T` entries. The default is `2*T`. Each entry is
`{key, count, base, localRank}` and `0xFFFFFFFF` marks an empty slot.

`HashClassifyKernel` inserts with `atomicCas` on the key and `atomicAdd` on the
count, then flushes occupied entries into the global totals. `HashScatterKernel`
rebuilds the table from `binIds`, reserves one range per occupied entry, and
hands out local ranks. A full table is a hard batch error. There is no dense
fallback and no sort or scan.

The default feedback example flow has two bins and uses the dense path; the
opt-in `--workload kernel` path can select the hash table. The hash path is
selected by instantiating `TransformStage` with a `HashBinTable`. The
dense/hash threshold is the static `Config::denseBinLimit = 1024`, not a
measured crossover.

## Exact host output planning

For each dense batch, the device writes `total[B]` plus a status word. The host
copies `B+1` words through an `allocMapped` staging buffer, and the node event
covers that copy. After polling the event the host:

1. Rejects a nonzero status word.
2. Builds `binBase[B]` and the exact fill list with `planDenseBins` over the
   batch-wide totals.
3. Allocates exactly `sum_b ceil(total_b/C)` output chunks and releases the
   surplus credits.
4. Copies the table `[binBase[0..B), chunkIndex[0..R)]` to the device.
5. Zeroes `next[B]` on the same queue.
6. Launches the scatter kernel and copies `total`, `next`, and the status word
   back.
7. Verifies `status == 0` and `next[bin] == total[bin]` for every bin.

Output descriptors are published only after the scatter event completes. The
scheduler then routes each descriptor by its bin. The source is not a transform:
it returns its own chunk downstream. Every other node releases its input chunks
only after its last device stage has completed.

## Validation split

The host validates the run configuration once before the first launch. The
photon example's `validateParameters` checks the slab (`thickness`, `sigmaS`,
`sigmaA`, and their finite total), the RNG stream limits, and the execution
knobs. Device stages assume a valid slab and do not re-check the coefficients
or the position range per event.

Per-event payload checks live in `include/graphFlow/flow/Debug.hpp` and the
photon `PhotonDebug.hpp` helper. A build with
`-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON` validates a finite position and
direction, `z` inside `[0, L]`, a finite non-negative proposal distance, and
proposal/event agreement. A failed check reports `StatusCode::invalid`, so the
host fails the batch before it publishes output. The option defaults to `ON` for
`CMAKE_BUILD_TYPE=Debug` and `OFF` for `Release` and `RelWithDebInfo`; pass
`-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON|OFF` to override. Release builds compile
the helper to a constant `true` and keep no per-event state branch.

The release path keeps the physics errors: a vacuum ray with no reachable face,
an invalid proposal or classifier bin, counter and weight overflow, and
duplicate terminal delivery.

## Ownership, leases, and teardown

A batch owns its input descriptors, output leases, staging buffers, event, and
error word. A transform batch acquires a reader lease for every selected input
before launch. On the normal path the runner releases all leases and input
chunks after `finish` and after the scatter event has fired.

If a hook throws, the runner drains the affected queue, calls the node's
`abort` hook, releases every outstanding lease, releases each owned input and
queued chunk once, and rethrows the original error. A transform's `abort`
releases output chunks that were acquired but never published, for example when
the completion-event enqueue failed after planning. A failed `TransformStage`
is single-use because a device stage may have been partially enqueued.

## Queue independence

`device.makeQueue()` may return the same handle and does not promise parallel
execution. `probeQueueIndependence` creates two non-blocking queues, enqueues a
bounded delay kernel on the first and a no-op kernel on the second, and polls
both events. It reports `overlap=yes|no|unknown`, `queuesAlias`, and whether the
second completed while the first was in flight.

The CUDA probe reports distinct queues with no observed overlap. The CPU probe
reports overlap. Correctness does not depend on the verdict: the scheduler keeps
per-queue FIFO order and one event per batch, so aliased or serialized queues
only change timing. The historical CUDA speedup came from full-chunk pipelining,
not cross-queue overlap.

## Device-aware policy

`include/graphFlow/core/DevicePolicy.hpp` derives `C`, `P`, and `T` from the
public `DeviceProperties` (`multiProcessorCount`, `warpSize`,
`sharedMemPerBlockBytes`, `maxThreadsPerBlock`, `maxBlocksPerGrid`, and
`globalMemCapacityBytes`). It caps the payload working set by a budget, keeps
explicit overrides, and rejects impossible combinations.

The CLI resolves the policy in `examples/particle_feedback/Cli.hpp`. An absent
`--chunk-size`, `--live-particles`, or `--tile-size` takes the policy value; an
explicit value is passed through as an override. An explicit `P` is rejected
when its payload exceeds the budget; only the auto-derived default `P` is
clamped. The policy is unit-tested with CPU properties and accelerator-shaped
public fields. The accelerator-shaped test pins `C=8192`, `P=16384`, `T=128` on
a nominal 24 GiB device. A device with the ECC-reduced A30 usable memory
(`3,171,090,432 B` at a 1/8 budget) selects `P=16128`: an explicit `C=8192`
with auto P also selects `P=16128`, while the default path budget-caps `C` to
`8064` with the same P. Explicit `P=16384` and `P=65536` are over budget and are
rejected.

## Current scheduler limits

- Q is unbounded apart from the 32-bit parse limit. An optional cap is not
  implemented.
- One in-flight batch per node.
- Each batch uses one `FrameSpec` launch for classify and one for scatter. The
  frame count is capped, so a block may process several tiles.
- A single transform output plan per batch.
- The lower-level `NodeBase`/`Graph`/`GraphBuilder`/`RouteTable` core is
  host-only; the example uses the typed `Flow` API, not the generic core.
