<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Public API contract

This document freezes the application-facing API of the simplified graphFlow
interface. The public API is implemented: the typed `Flow` class lives in
`include/graphFlow/flow/Flow.hpp`, the execution adapters in
`flow/Execution.hpp`, and the status channel in `flow/Status.hpp`. The
particle-feedback example declares its graph with `SoAFlow` handles and named
outcome ports, while `ParticleStages.hpp` remains only for the older low-level
tests. The photon-scattering example uses the same API for slab transport. The
API is additive and replaces no existing public type.

## Payload

`graphFlow::Flow<TPayload, TDevice, TStorage>` owns one graph for one payload
type. `TPayload` is a trivially copyable, default-constructible record whose
fields and meanings belong to the application. graphFlow copies payloads by
value and may send them through host and device memory. Functors and result
views are device-copyable named C++20 structs; accelerator functors use
`ALPAKA_FN_ACC` and may take the accelerator as their first argument.

`TStorage` is the chunk storage and defaults to
`DefaultStorage<TPayload, TDevice>`. For the canonical `Particle` payload that
alias selects the SoA `ParticleStorage<TDevice>`; every other payload selects
`PayloadSlotStorage<TPayload, TDevice>`, a generic array-of-struct storage with
`slotBytes == sizeof(TPayload)`. `TDevice` is the alpaka device type the graph
runs on. The `Payload<TPayload>` concept rejects a payload that is not
trivially copyable and default-constructible. An application may pass
`SoAStorage<TDevice, Fields...>` explicitly for a record payload; it is never
selected by default.

## Attribute records and AoS-over-SoA storage

`graphFlow/flow/Attributes.hpp`, included by `graphFlow/graphFlow.hpp`,
re-exports the llamaLite tag, field, and set descriptors and adds the record
and storage adapters. The record and column types build on the vendored
llamaLite descriptors under `include/graphFlow/thirdparty/llamaLite`;
graphFlow adds validation, slot accounting, and the chunk-storage contract, not
a second record implementation.

| Type | Role |
| --- | --- |
| `TagBase` | Empty base of every tag; one tag selects one attribute. |
| `Field<Tag, T>` | Compile-time attribute descriptor; `T` must be trivially copyable. |
| `Set<Tags...>` | Per-node read/write subset with a compile-time `contains<Tag>`. |
| `AttributeList<Fields...>` | Validated, ordered list; source of truth for names, order, value types, and slot accounting. |
| `Record<Fields...>` | Value record with `get<Tag>()`, `set<Tag>()`, `operator[](Tag)`, and value semantics. |
| `RecordProxy<TAccess, TSet>` | Narrow view whose `get`, `set`, and `operator[]` reject tags outside its set at compile time. |
| `SoAStorage<TDevice, Fields...>` | Payload storage with one device column per attribute plus a bin-id scratch column. |

A tag is an empty type derived from `graphFlow::TagBase`; two attributes with
the same tag cannot coexist in one `AttributeList`. A `Field` is a type
descriptor only: values live in a `Record` or in an `SoAStorage` column. An
`AttributeList` requires at least one field and rejects a field whose value
type is not trivially copyable.

`GRAPHFLOW_TAG(Id, idTag)` declares `IdTag` and a `constexpr` value `idTag`
that names it; the one-argument form `GRAPHFLOW_TAG(Id)` names the value
`IdTag_v`. Passing a tag value to `operator[]` adds no runtime state: the tag
is empty and only selects the attribute at compile time. `GRAPHFLOW_ATTRIBUTE`
and hand-written tag structs keep working unchanged.

```cpp
GRAPHFLOW_TAG(Id, idTag);
GRAPHFLOW_TAG(Weight, weightTag);

using Id = graphFlow::Field<IdTag, std::uint64_t>;
using Weight = graphFlow::Field<WeightTag, std::uint32_t>;

using ItemRecord = graphFlow::Record<Id, Weight>;
using ItemStorage = graphFlow::SoAStorage<TDevice, Id, Weight>;

graphFlow::Flow<ItemRecord, TDevice, ItemStorage> flow{device};
```

`Record<Fields...>` is trivially copyable and default constructible, so it
satisfies `Payload`. Its value storage is a one-element `llama_lite::SoA`, so
copying a record copies every attribute. `record[idTag]` is the subscript
spelling of `record.get<IdTag>()` and, on a mutable record, returns the value
reference needed for `record[idTag] = value` or `record[idTag] += value`.
`record.proxy<Set<Tags...>>()` returns a `RecordProxy` whose `get<Tag>()`,
`set<Tag>()`, and `operator[]` reject tags outside that set at compile time.
`load()`, the conversion to `record_type`, and `store()` operate on the whole
record: copying a proxy into a `Record` copies every attribute, including
attributes outside the set. The proxy narrows access, not the record layout or
the copy. `transform(name, functor)` passes the full `Record` payload.
`flow.transform<Set<Tags...>>(name, functor)` is an additive overload that hands
the functor a `RecordProxy` limited to the set; the scheduler still loads and
stores the complete record, so attributes outside the set keep their values.

`SoAStorage<TDevice, Fields...>` implements `PayloadStorage` and
`ChunkStorage`. It allocates one `onHost::alloc<value_type>` column per
attribute plus one `std::uint32_t` column; there is no contiguous record array.
Chunk `k` owns row `k` of every column:

```text
base(handle)          = handle.index * chunkCapacity
slotIndex(handle,row) = base(handle) + row
```

`loadRecord(slot)` and `storeRecord(slot, record)` gather or scatter one
logical record, `proxy<Set<Tags...>>(slot)` returns a per-slot `RecordProxy`,
`column<Tag>()` exposes one attribute column, and `binIds()` exposes the
scratch column for custom classify/scatter stages. The constructor rejects a
zero chunk count or chunk capacity and clamps a larger chunk count to the
payload budget, exactly like the other storages; `numChunks()` reports the
physical count.

`slotBytes` is the accounting size of one logical record, not the size of one
contiguous object:

```text
padded(T)      = ceil(sizeof(T) / alignof(T)) * alignof(T)
attributeBytes = sum of padded(value_type) over all attributes
scratchBytes   = sizeof(std::uint32_t)
slotBytes      = attributeBytes + scratchBytes
```

For the canonical particle fields (`uint64_t` id and three `uint32_t` fields)
`attributeBytes` is 20 and `slotBytes` is 24, equal to `payloadBytesPerSlot`.
The payload budget keeps its existing meaning:
`numChunks * chunkCapacity * slotBytes` is what `maxPayloadChunks`,
`requiredPayloadBytes`, and `validatePayloadBudget` account for. Each column
allocates its own value type, so `slotBytes` sizes the budget and views, not
one array.

### Device access

`detail::StorageAccess<SoAStorage<TDevice, Fields...>, Record<Fields...>>`
binds one view per attribute in declaration order. `views()` returns them as a
tuple, and the launch helpers `enqueueKernelWithViews` and
`enqueueKernelWithInOutViews` expand the tuple into separate top-level kernel
arguments. The expansion is required because alpaka rewrites only top-level
view arguments for the device; packing the columns into one struct would leave
the device with host views. `load(slot, views...)` gathers a record and
`store(slot, record, views...)` scatters it, attribute by attribute.

### Defaults and migration from `ParticleStorage`

The defaults do not change:

| `TPayload` | `DefaultStorage<TPayload, TDevice>` | `slotBytes` |
| --- | --- | --- |
| `graphFlow::Particle` | `ParticleStorage<TDevice>` | `payloadBytesPerSlot` (24) |
| any other payload | `PayloadSlotStorage<TPayload, TDevice>` | `sizeof(TPayload)` |

A `Record` is also a valid payload with the default `PayloadSlotStorage`; the
record type and the column layout are separate choices. The storage defaults
never change because a type happens to be a `Record`.

`ParticleStorage` keeps its existing `ids`, `properties`, `weights`, and
`visits` arrays for the deprecated low-level particle stages. The particle and
photon examples opt into `SoAFlow` and `SoAStorage` for their `Record` payloads.
An application that wants attribute columns migrates only its own payload:

1. Declare one tag derived from `graphFlow::TagBase` and one `Field<Tag, T>` per
   attribute, then name the ordered `AttributeList` or `Record`. Attribute order
   fixes the column and kernel-argument order.
2. Pass `SoAStorage<TDevice, Fields...>` as the third `Flow` template argument.
   The record type is the first argument and the two field lists must match.
3. Replace field access (`payload.id`) with `payload[idTag]` (or
   `payload.get<IdTag>()`) and assignments with `payload[idTag] = value` (or
   `payload.set<IdTag>(value)`); source functors construct the record from the
   declared fields. A node may call `record.proxy<Set<Tags...>>()` or
   `storage.proxy<Set<Tags...>>(slot)`, but a record copy still carries every
   attribute.
4. Use `binIds()` only when implementing a custom classify/scatter stage; the
   built-in stages allocate their own bin-id scratch view.

The column layout carries no timing or throughput claim. This document
describes the contract, not a measured comparison with `ParticleStorage`.

## Flow operations

`Flow<TPayload, TDevice, TStorage>` is constructed with the device it runs on
(`Flow<TPayload, TDevice> flow{device}`); the device must outlive the flow. The
calls below are the complete minimal surface.

| Call | Contract |
| --- | --- |
| `source(name, functor)` | Map a logical item index to one initial payload. The node has one output. |
| `partition(name, classifier[, update])` | `classifier` declares `static constexpr std::uint32_t maxBins` and returns one bin for an immutable payload. The optional `update` receives the selected bin and a mutable copy and runs exactly once per emitted item. |
| `transform(name, functor)` | Update each payload and forward it on one output. No histogram or bin scatter is involved. The first version uses an out-of-place path or proves exclusive ownership for in-place reuse. `transform<Set<Tags...>>(name, functor)` keeps the same path and hands the functor a `RecordProxy` limited to the set. |
| `sink(name, functor)` | Consume each payload once. Application result views are allowed and own their storage. |
| `connect(output, node)` | Connect a node or output port to a node. Fan-in and feedback are allowed. Each item follows exactly one successor. |
| `configure(tuning)` | Set the optional `graphFlow::Tuning` value for the run. |
| `run(source, itemCount)` | Return `SchedulerStats` after completion and resource retirement, or propagate an execution error. |

`Flow` also accepts an optional `graphFlow::Tuning` value outside these calls
through `configure`. `Tuning::seed` is a plain `uint64_t` (default zero).
Every other `Tuning` field is optional and empty by default, so the device
policy supplies the derived value: batching (`batchChunks`,
`chunkCapacity`), storage and pool limits (`liveParticles`, `tileSize`,
`numChunks`, `poolPayloadBudgetBytes`), the launch frame cap (`frameCap`),
device-policy knobs, the opt-in `diagnostics` flag, the measured partition
axes (`scatterMode` and `binPath`), and an optional `stallTimeout`
(`std::chrono::milliseconds`). It never changes payload semantics or routes;
tuning is not an argument of the minimal calls above.

`Tuning::chunkKernelSchedule` selects the transform launch schedule; absence
keeps `ChunkKernelSchedule::off`, the historical batch-wide launch, and
`ChunkKernelSchedule::perChunk` is the opt-in that issues one kernel launch per
input chunk. The schedule changes launch geometry only: the storage chunk
capacity, the pool size, the payload layout and the host-side output-position
planning stay what the device policy derived. It affects
`Flow::transform` nodes only; source, sink and partition nodes keep the batch
path, because the per-chunk queue pools are allocated only for transform nodes.
`Tuning::chunkQueueDepth` (default 1) is the number of non-blocking device
queues a per-chunk transform node owns; lane `i` enqueues on queue
`i % chunkQueueDepth`, so the chunks of one node run and retire independently.
Zero is rejected when the schedule is `perChunk`. `Tuning::chunkLatencyCapUs`
(default 0, meaning no cap) is the per-chunk launch latency cap in
microseconds; it clamps only the capacity and wave count that the per-chunk
launch-size policy reports, at a fixed 10 us per wave, and never changes the
storage chunk capacity. The `photon_scattering` CLI exposes the three knobs as
`--chunk-kernel-schedule off|perChunk` (default `off`), `--chunk-queue-depth D`
(default 1) and `--chunk-latency-cap-us U` (default 0).

`scatterMode` selects `ScatterMode::SharedStaging` (the default) or
`ScatterMode::Direct` on the dense path. `binPath` is `BinPath::automatic` by
default: it selects dense up to `Config::denseBinLimit` bins and the
shared-memory hash table above it. A run is rejected before device work when an
explicit dense path exceeds the limit, when direct scatter is combined with the
hash path, or when a hash run requests a tile wider than
`Config::hashBinTableTileSize`.

`run()` returns `SchedulerStats` with aggregate peaks (`peakReadyChunks`,
`peakReadyNodes`, `maxReadyDepth`, `peakInflightBatches`, `peakInflightChunks`,
`peakReservedChunks`, `peakEventSlotsInUse`) and source counters
(`sourceParticles`, `sourceGenerated`, `sourceSunk`, `sourceLiveParticles`,
`chunkCapacity`). The schedule adds `chunkLaunches`, the number of enqueued
per-chunk transform kernels, and `chunkLimit`, the clamp name
(`wave`, `maximum`, `grid`, `memory` or `latency`) that set the per-chunk
launch-size policy. The off path reports `chunkLaunches=0 chunkLimit=none`. A
Phase 0 chunk registry adds `registryAcquisitions`, `registryRoutings`,
`registryRetirements` and the phase peaks `peakRegistryReady`,
`peakRegistryInflight`, `peakRegistryReserved`; these counters observe the
pipeline and never feed a scheduling decision. A stall throws
`SchedulerStallError`; `snapshot()` on the runner and on the error exposes the
same aggregate state, including the current registry phase counts, without
topology. The low-level `GraphRunner` additionally exposes
`chunkRegistry()` (one record per pool slot), `chunkRegistrySummary()` and
`validateChunkRegistry()`.

## Quickstart: `SoAFlow<Particle>`

The snippet below declares an application record named `Particle` and runs it
with column storage. It is a custom record, not `graphFlow::Particle`; the
canonical particle type keeps the `ParticleStorage` default.
`graphFlow::SoAFlow<Record, Device>` is the public shorthand for
`Flow<Record, Device, SoAStorage<Device, Fields...>>`; the record attribute
pack is written once.

```cpp
GRAPHFLOW_TAG(Id, idTag);
GRAPHFLOW_TAG(Value, valueTag);

using Id = graphFlow::Field<IdTag, std::uint64_t>;
using Value = graphFlow::Field<ValueTag, std::uint32_t>;
using Particle = graphFlow::Record<Id, Value>;

using Device = decltype(device);
graphFlow::SoAFlow<Particle, Device> flow{device};

auto source = flow.source("source", Generate{});
auto bump = flow.transform("bump", Bump{});
auto split = flow.partition("split", Mod2{}, AddBin{});
auto sink = flow.sink("record", RecordValue<decltype(values.getView())>{values.getView()});
flow.connect(source, bump);
flow.connect(bump, split);
flow.connect(split.template out<0u>(), sink);
flow.connect(split.template out<1u>(), sink);
flow.configure(graphFlow::Tuning{});
flow.run(source, count);
```

`Generate` returns a `Particle` from an item index, `Bump` mutates the value
through `particle[valueTag]`, `Mod2` declares `maxBins = 2` and reads
`particle[idTag]`, `AddBin` adds the selected bin to `particle[valueTag]`, and
`RecordValue` writes the result into an application-owned view. Every functor
is a named, device-copyable struct that uses `ALPAKA_FN_ACC`. The complete source is
`examples/record_pipeline/main.cpp`; it uses the public
`graphFlow::SoAFlow<Particle, Device>` alias.

Build and run it:

```sh
cmake --build build-cpu --target graphFlow_record_pipeline
./build-cpu/graphFlow_record_pipeline
# record_pipeline: 10 12 12 14 14 16 16 18 sum=112
```

The output is a correctness check, not a performance result.

## Migrating from the low-level API

The legacy path assembled numeric nodes with `GraphBuilder`, registered hooks,
and sized a `ChunkPool`/`GraphRunner` by hand. The public path replaces that
setup:

1. Keep the domain functors. Give each a stable name and mark accelerator code
   with `ALPAKA_FN_ACC`.
2. Replace `addNode`/`setSource` calls with `source`, `transform`, `partition`,
   and `sink`. Store the returned handles.
3. Replace route vectors and numeric `NodeId` values with `connect`. A
   multi-output node connects through `out<Bin>()`, or through
   `connect(node, bin, node)` when the bin is only known at runtime.
4. Replace manual `ChunkPool`/`GraphRunner` construction with
   `configure(Tuning)` and `run(source, count)`. `Flow` constructs and retires
   the storage, stages, queues, and events.
5. Keep result views in application-owned buffers; they must outlive `run()`.

### From `ParticleStorage` to `Record` and `SoAStorage`

`graphFlow::Particle` with its default `ParticleStorage` stays the canonical
low-level particle path. The particle-feedback example migrates its public
payload to a `ParticleRecord` with `idTag`, `propertyTag`, `weightTag` and
`visitTag`. An application that wants attribute columns changes only its payload
and storage:

1. Declare one tag derived from `graphFlow::TagBase` and one `Field<Tag, T>` per
   attribute. List the fields in the `Record` and the `SoAStorage` in the same
   order.
2. Pass `Record<Fields...>` as the `Flow` payload and
   `SoAStorage<TDevice, Fields...>` as the third template argument, or use
   `SoAFlow<Record<Fields...>, TDevice>`, which deduces the storage from the
   record. The two field lists must match. A custom record with only two
   template arguments still selects `DefaultStorage`, which is
   `PayloadSlotStorage` for any payload other than `graphFlow::Particle`; it
   does not give per-attribute columns.
3. Replace `payload.id` with `payload[idTag]` (or `payload.get<IdTag>()`) and
   assignments with `payload[idTag] = value` (or `payload.set<IdTag>(value)`).
4. Use `binIds()` only in custom classify/scatter stages. The built-in stages
   manage their own scratch column.

`slotBytes` is the accounting size (padded attributes plus one `uint32_t`
scratch word), not one contiguous array. The payload budget and pool sizing
keep their meaning.

## Compile errors and fixes

The type-level contracts fail at compile time. Read the first `static_assert`
or concept diagnostic.

| Diagnostic | Cause and fix |
| --- | --- |
| `graphFlow::Flow: the payload type does not satisfy Payload` | The payload is not trivially copyable or not default-constructible. Use a `Record` or a trivial POD. |
| `no matching function for call to` on `source`, `transform`, `partition`, or `sink` | The functor signature does not match its role. A source returns the payload, a transform takes `Payload&` or returns the payload, a classifier declares `static constexpr std::uint32_t maxBins` and returns a bin, and a sink takes `Payload const&`. |
| `graphFlow: a multi-output node must be connected via out<Bin>(), not connect(node, node)` | The partition handle was connected directly. Connect each output with `out<Bin>()` or with the runtime-bin overload. |
| `graphFlow::Record: get<Tag>() names an unknown attribute` | The tag is not one of the record fields. Add the `Field<Tag, T>` to the record or use the correct tag. |
| `graphFlow::RecordProxy: get<Tag>() is outside the node's attribute subset` | The proxy was created with a `Set<Tags...>` that does not contain the requested tag. Widen the set or use a tag in it. |
| `graphFlow::SoAStorage: proxy Set contains a tag that is not an attribute` | The `Set` passed to `proxy` names a tag outside the storage field list. |
| `graphFlow::Flow: the storage type must be constructible from (device, numChunks, chunkCapacity)` | A custom storage does not provide the constructor `Flow` uses. Add it or use `SoAStorage`/`DefaultStorage`. |
| `graphFlow: an attribute list must contain at least one field`, `every attribute must be a graphFlow::Field<Tag, T>`, or `attribute value types must be trivially copyable` | The `Record`/`SoAStorage` field list is empty, contains a non-field type, or stores a non-trivially-copyable value. |
| `graphFlow: a partition needs at least one bin` | The classifier declares `maxBins = 0`. Use a positive bin count. |

A functor that omits `ALPAKA_FN_ACC` can compile for the host backend and then
fail when the same file is compiled for CUDA, HIP, or SYCL. Mark every functor,
classifier, and updater with `ALPAKA_FN_ACC` or `ALPAKA_FN_HOST_ACC` as
required. `Tuning` is a runtime object; it is not a template argument and does
not change functor signatures.

## Failure status

A functor may take an optional `Status&` argument after the optional
accelerator argument:

| Functor | Signatures with status |
| --- | --- |
| source | `(acc, status, id)`, `(status, id)` |
| transform | `(acc, status, payload)`, `(status, payload)` |
| classifier | `(acc, status, payload)`, `(status, payload)` |
| updater | `(acc, status, bin, payload)`, `(status, bin, payload)` |
| sink | `(acc, status, payload)`, `(status, payload)` |

`Status` carries one status word per batch. A functor calls
`status.set(graphFlow::StatusCode::invalid)` or
`status.set(graphFlow::StatusCode::overflow)` when it rejects an item.
`invalid` covers non-finite or impossible geometry, a vacuum ray with no
reachable face, a classifier result outside `maxBins`, and other checked input
failures. `overflow` covers a checked counter or capacity overflow.

Application setup validates the configuration and geometry once on the host.
Release kernels do not repeat per-event payload checks. Configure a build with
`-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON` to compile the checks in
`graphFlow/flow/Debug.hpp` and `PhotonDebug.hpp`; a failed debug check reports
`StatusCode::invalid` through the same channel and fails the batch before it
publishes output. The option defaults to `ON` for `CMAKE_BUILD_TYPE=Debug` and
`OFF` for `Release` and `RelWithDebInfo`, and an explicit
`-DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON|OFF` overrides it. It never replaces a physics
error: a vacuum ray with no reachable face, an invalid proposal or bin, a
counter or weight overflow, and duplicate terminal delivery stay release-path
failures.

The device adapter copies the status word to the host after the metadata copy
and before the batch event is observed. The host rejects the batch with a
`BatchError` before it plans or publishes outputs, and the abort path returns
the planned output chunks, leases, and pool credits exactly once. A failed
stage is single-use. A capped history is a routed outcome, not a failure, so a
classifier may still send it to a named terminal port.

An updater reports its result through `UpdateResult`:

| Value | Meaning |
| --- | --- |
| `success` | The update completed. |
| `invalid` | The update rejected the item; the host fails the batch. |
| `overflow` | A checked counter or capacity overflowed; the host fails the batch. |

## Handles and ports

A node handle is an opaque value carrying a graph identity and an internal node
key. It stays valid when more nodes are registered. Users never pass, compare,
or derive numeric node IDs. A handle does not keep the graph, queues, or copied
functors alive.

An output port names a route of the producing node. `out<0>()` is bin `0` and
`out<Outcome::feedback>()` is the same port by a named enumerator. The bin range
is checked at compile time for static ports. Several bins may target the same
node. Registration order is free; connections may be declared in any order.

Before launch the implementation rejects cross-graph connections, duplicated
routes, missing routes, incompatible payloads, incompatible directions, and
static ports outside the classifier's `maxBins`.

## Ownership and lifetime

The graph owns node state and copies of user functors. Handles are non-owning.
The execution engine owns queues and events and retires them before `run()`
returns. Buffers, copied functors, and result views stay alive until the last
dependent queue event; a sink result view must outlive the run but owns no
internal graphFlow storage.

The source drains until `itemCount` items are admitted and all ready and
in-flight work is retired. Outputs are published only after scatter completion,
and a feedback edge is scheduled like any other edge. The run ends with no
in-flight batch and no live pool chunk, or reports an execution error without
publishing a partial output.

Using a handle after its owning `Flow` has been destroyed is rejected with
`std::logic_error`; the check uses the handle's graph identity and does not
dereference the destroyed owner. Using a handle with a different live graph is
rejected with `std::invalid_argument`. Neither case silently aliases node state.

## RNG and engine injection

`Tuning::seed` is the only run-level RNG input. graphFlow derives one
initialized `graphFlow::rng::Engine` per logical `(id, event, purpose)` stream
and passes it to an opting-in functor after the optional accelerator argument:

```cpp
ALPAKA_FN_ACC Output operator()(auto const& acc, graphFlow::rng::Engine& engine, Input const& in) const
{
    double u = alpaka::rand::distribution::UniformReal<double>{}(engine);
    // integer decisions use engine() directly
}
```

The same optional argument works in `source`, `partition` classifiers, `update`
updaters, `transform`, and `sink` functors. Existing signatures without an
engine keep working through `if constexpr` dispatch. A partition updater receives
its own stream. Payloads expose `id` and `visit` or `scatterCount`; a custom
payload can specialise `graphFlow::rng::StreamTraits`. Users never construct a
seed, offset, or purpose.

## Backend validation

The contract is exercised on CPU, CUDA, HIP, and SYCL. The current suite
counts at `3b90ffd8962084d59ebc672f7848b093f2d99582` are
**270/270 CPU**, **334/334 CUDA** on `fwk394` RTX 2080 Ti `sm_75`,
**325/325 HIP** on `hal` RX 7900 XTX, and **342/342 SYCL** on `hal` Arc A770
(75 sycl-labelled). The last full recorded run before the AoS additions
passed 255/255, 316/316 and 307/307 at
`561b9a4c9435e37b164623bb6be60a3051783eb4`. The AoS-over-SoA matrix
adds three cases per backend (3/3 each); the attribute declaration helpers
pass 6/6, the registration-site transform attribute-subset case 1/1, and the
record-pipeline example 2/2. PR #63 adds the
public-Flow routed-axis and rejection cases and PR #65 adds the six
`[scheduler]` cases on top of the P07b 7/7. The `561b9a4c` totals 255/255,
316/316 and 307/307, the `9ace795` 72/72 sycl-labelled SYCL run, and the
`ffe4074` totals 249/249,
310/310 and 301/301, the 247/247, 307/307 and 298/298 at `88de14f`, the
246/246, 306/306 and 297/297 at `56c11ac` and the 239/299/290 at `a610281`
are historical. The original HIP gate at `b972eaa` ran 245/245 with
43 HIP cases, including 8 typed Flow cases through the public `Flow` API
plus the particle and photon example CLI/output cases; that total is
historical and recorded in `/home/agent/final-hip-b972eaa/ctest.log`. The typed
cases cover the feedback matrix,
partial tails, bounded pools, empty and single runs, the generic POD chain,
one batch-wide transform launch, one classify plus one scatter partition
launch, and Q=8 feedback termination. `rocprof` confirms the device kernels.

The public `Flow` API routes both the dense and the shared-memory hash bin path
through `Tuning::binPath`; the dense scatter mode is selected by
`Tuning::scatterMode`. Wide partitions connect their outputs through the
additive runtime-bin `connect(node, bin, node)` overload so 4096 and 65536-bin
graphs do not need a compile-time port per bin. The tree-wide format and SPDX
sweep (T14b) has landed and is enforced by the hosted `format` job.

GitHub-hosted HIP builds are compile-only because those runners have no AMD
GPU, so they are not a runtime result. `hal` reports `sharing=unknown`; the HIP
run is a correctness record and makes no performance claim.

## Examples

The three example binaries are the executable contract of this API.

### Record pipeline

`examples/record_pipeline` is the minimal AoS-over-SoA pipeline from the
quickstart. It uses `Record<Id, Value>`, `SoAStorage<Device, Id, Value>`, one
source, one transform, a two-bin partition, and one sink. It prints the eight
result values and their sum:

```sh
./build-cpu/graphFlow_record_pipeline
# record_pipeline: 10 12 12 14 14 16 16 18 sum=112
```

CTest registers `example.record_pipeline.api` for the source-level contract and
`example.record_pipeline.run` for the binary output when the example target is
built. The run is a correctness check, not a performance result.

### Particle feedback

`examples/particle_feedback` runs a Generate -> Filter -> {A, B} -> Sink
feedback network. The user graph lives in a 121-line file
(`ParticleFeedback.hpp`); the functors and domain types are separate user
files. The RNG API change replaced the offset map with keyed engines, so the
fixed-seed baselines changed once; the earlier 33-invocation comparison
predates it. Minimal shape:

```cpp
graphFlow::Flow<graphFlow::Particle, TDevice> flow{device};
auto source   = flow.source("generate", GenerateParticles{});
auto filter   = flow.partition("filter", FilterByProperty{});
auto processA = flow.partition("process A", SelectOutcome{maxVisits}, UpdateWeight{1000u});
auto processB = flow.partition("process B", SelectOutcome{maxVisits}, UpdateWeight{1u});
auto sink     = flow.sink("record weight", RecordWeight<decltype(weights.getView())>{weights.getView()});

flow.connect(source, filter);
flow.connect(filter.template out<0u>(), processA);
flow.connect(filter.template out<1u>(), processB);
flow.connect(processA.template out<Outcome::sink>(), sink);
flow.connect(processA.template out<Outcome::feedback>(), filter);
flow.configure(tuning);
SchedulerStats stats = flow.run(source, numParticles);
```

Run it with:

```sh
./build-cpu/graphFlow_particle_feedback --backend cpu --particles 4097 \
  --chunk-size 128 --live-particles 256 --tile-size 128 --batch-chunks 1
```

### Photon scattering

`examples/photon_scattering` transports independent photon histories through a
homogeneous plane-parallel slab. The graph is source -> propose flight ->
advance and route -> scatter -> propose flight, with transmitted, reflected,
absorbed, and capped outcomes feeding one terminal sink. The payload is a
`graphFlow::Record` over constexpr id, position, direction, scatter-count and
flight-proposal tags, and the run uses `graphFlow::SoAFlow` so
`SoAStorage` keeps one column per attribute. Functors receive the same
initialized `graphFlow::rng::Engine`; floating-point draws use
`UniformReal<double>` (and `NormalReal<double>` where needed). Functors that
reject an item set the typed `Status`; the host rejects the batch before it
publishes output.

```sh
mkdir -p /tmp/photon-forward
./build-cpu/graphFlow_photon_scattering --backend cpu --photons 4097 \
  --thickness 1 --sigma-s 3 --sigma-a 0 --mode forward --seed 42 \
  --chunk-size 127 --output-dir /tmp/photon-forward \
  --trace-photons 16 --trace-vertices 256
python3 tools/plot_photon_scattering.py --input /tmp/photon-forward \
  --out /tmp/photon-forward/plots
```

`--chunk-kernel-schedule off|perChunk` (default `off`) selects the transform
launch schedule described above; a user opts in when the device has enough
capacity for the per-chunk launch pattern to pay off, for example the
measured runs in [benchmarks](benchmarks.md#per-chunk-transform-schedule-photon-scattering).
`--chunk-queue-depth D` (default 1) sets the queue pool of each per-chunk
transform node, and `--chunk-latency-cap-us U` (default 0, no cap) clamps the
capacity and waves that launch-size policy reports without touching the storage
chunk capacity. With `perChunk` the scheduler line ends in
`chunkLaunches=<count> chunkLimit=<name>`; the off path ends in
`chunkLaunches=0 chunkLimit=none`.

`--output-dir` writes `metadata.csv`, `outcomes.csv`,
`scatter_histogram.csv`, `exits.csv`, `traces.csv` and `trace_summary.csv`.
The plot script writes `figure1_paths` through `figure4_detector` as PNG, SVG
and PDF. Larger statistical diagnostics are opt-in through
`-DGRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS=ON` and `ctest -L diagnostics`.
