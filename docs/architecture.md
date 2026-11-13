<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Architecture

graphFlow is a host-scheduled dataflow runner for chunked particle payloads.
The host owns topology, admission, routing, and completion. Device code owns
per-tile histogram, hash, and scatter work. The current tree is a
particle-feedback example built from reusable runner primitives, not a generic
graph API.

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
raises a stall error that reports generated, sunk, and live particles, free and
reserved chunks, and per-node ready/in-flight depth.

## Pool, batch reserve, and admission

`ChunkPool` stores fixed-capacity SoA payload arrays. A chunk handle is
`{chunkIndex, generation}`. The generation increments on every release, so a
stale handle is rejected instead of aliasing a reused chunk. Particle `i` of
chunk `k` lives at slot `k*C + i`.

The fixed pool is sized as

```text
S = P + K*R(C)
R(C) = outputChunkBound(C, C, B)
K = number of transform nodes (3 in the particle example)
```

`P` is the source-admission capacity in live particles. `R(C)` is the
conservative output-chunk bound for one single-chunk transform. `P` data chunks
cover the live particles because split routing can fragment them down to one
particle per chunk; `K*R(C)` covers the output credits of the live transform
nodes. The plan's `ceil(P/C) + K*R` is a lower bound and deadlocks under split
feedback. Q does not enlarge `S`.

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
property, weight, visit, and bin-id arrays. The policy memory budget covers the
`P*C` live working set; the example adds the exact `K*R(C)` output-credit
reserve. `Config::validate` rejects a pool over budget before any allocation.
The former `C=8192`, `P=65536` pair exceeds the default budget and is rejected.

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

The example always uses the dense path because its flow has two bins. The hash
path is selected by instantiating `TransformStage` with a `HashBinTable`. The
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
- The generic `NodeBase`/`Graph`/`GraphBuilder`/`RouteTable` core is host-only; the example still assembles its graph by hand.
