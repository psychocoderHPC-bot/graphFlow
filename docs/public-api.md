<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Public API contract

This document freezes the application-facing API of the simplified graphFlow
interface. The public API is implemented: the typed `Flow` class lives in
`include/graphFlow/flow/Flow.hpp`, the execution adapters in
`flow/Execution.hpp`, and the status channel in `flow/Status.hpp`. The
particle-feedback example declares its graph with `Flow` handles and named
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
`DefaultStorage<TPayload, TDevice>`, a generic payload-slot storage with
statically known `slotBytes`. `TDevice` is the alpaka device type the graph
runs on. The `Payload<TPayload>` concept rejects a payload that is not
trivially copyable and default-constructible.

## Flow operations

`Flow<TPayload, TDevice, TStorage>` is constructed with the device it runs on
(`Flow<TPayload, TDevice> flow{device}`); the device must outlive the flow. The
calls below are the complete minimal surface.

| Call | Contract |
| --- | --- |
| `source(name, functor)` | Map a logical item index to one initial payload. The node has one output. |
| `partition(name, classifier[, update])` | `classifier` declares `static constexpr std::uint32_t maxBins` and returns one bin for an immutable payload. The optional `update` receives the selected bin and a mutable copy and runs exactly once per emitted item. |
| `transform(name, functor)` | Update each payload and forward it on one output. No histogram or bin scatter is involved. The first version uses an out-of-place path or proves exclusive ownership for in-place reuse. |
| `sink(name, functor)` | Consume each payload once. Application result views are allowed and own their storage. |
| `connect(output, node)` | Connect a node or output port to a node. Fan-in and feedback are allowed. Each item follows exactly one successor. |
| `configure(tuning)` | Set the optional `graphFlow::Tuning` value for the run. |
| `run(source, itemCount)` | Return `SchedulerStats` after completion and resource retirement, or propagate an execution error. |

`Flow` also accepts an optional `graphFlow::Tuning` value outside these calls
through `configure`. `Tuning` fields are optional and are empty by default, so
the device policy supplies the derived value: batching (`batchChunks`,
`chunkCapacity`), storage and pool limits (`liveParticles`, `tileSize`,
`numChunks`, `poolPayloadBudgetBytes`), the launch frame cap (`frameCap`),
device-policy knobs, and the opt-in `diagnostics` flag. It never changes
payload semantics or routes; tuning is not an argument of the minimal calls
above.

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

## Examples

The two example binaries are the executable contract of this API.

### Particle feedback

`examples/particle_feedback` runs a Generate -> Filter -> {A, B} -> Sink
feedback network. The user graph lives in a 121-line file
(`ParticleFeedback.hpp`); the functors and domain types are separate user
files. A fixed-seed 33-invocation CLI comparison matched the pre-change binary
on CPU and CUDA. Minimal shape:

```cpp
graphFlow::Flow<graphFlow::Particle, TDevice> flow{device};
auto source   = flow.source("generate", GenerateParticles{seed});
auto filter   = flow.partition("filter", FilterByProperty{seed});
auto processA = flow.partition("process A", SelectOutcome{seed, ...}, UpdateWeight{1000u});
auto processB = flow.partition("process B", SelectOutcome{seed, ...}, UpdateWeight{1u});
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
absorbed, and capped outcomes feeding one terminal sink. Random draws use
`Philox4x32x10` through the same key-based mapping as the particle example.
Functors that reject an item set the typed `Status`; the host rejects the batch
before it publishes output.

```sh
mkdir -p /tmp/photon-forward
./build-cpu/graphFlow_photon_scattering --backend cpu --photons 4097 \
  --thickness 1 --sigma-s 3 --sigma-a 0 --mode forward --seed 42 \
  --chunk-size 127 --output-dir /tmp/photon-forward \
  --trace-photons 16 --trace-vertices 256
python3 tools/plot_photon_scattering.py --input /tmp/photon-forward \
  --out /tmp/photon-forward/plots
```

`--output-dir` writes `metadata.csv`, `outcomes.csv`,
`scatter_histogram.csv`, `exits.csv`, `traces.csv` and `trace_summary.csv`.
The plot script writes `figure1_paths` through `figure4_detector` as PNG, SVG
and PDF. Larger statistical diagnostics are opt-in through
`-DGRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS=ON` and `ctest -L diagnostics`.
