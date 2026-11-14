<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Public API

Status: implemented. The application-facing API of the simplified graphFlow
interface lives in `include/graphFlow/flow/`. `graphFlow::Flow<TPayload,
TDevice, TStorage = DefaultStorage<TPayload, TDevice>>` owns one graph for one
payload type; the typed execution path is in `flow/Execution.hpp`, the storage
adapters in `flow/PayloadStorage.hpp` (the SoA `ParticleStorage` over the
existing `ChunkPool` for the canonical `Particle` payload, and the generic AoS
`PayloadSlotStorage` for every other trivially-copyable POD payload), the
failure channel in `flow/Status.hpp`, and the four-purpose RNG mapping in
`rng/ParticleRng.hpp`. The particle-feedback and photon-scattering examples
declare their graphs with `Flow` handles and named outcome ports;
`ParticleStages.hpp` remains only for the older low-level tests. The API is
additive and replaces no existing public type.

## Payload

A payload is a trivially copyable, default-constructible record. Its fields and
meanings belong to the application. graphFlow copies payloads by value and may
send them through host and device memory. `DefaultStorage` binds the canonical
`Particle` payload to the particle SoA pool and every other POD payload to the
slot storage, so a photon record is never reinterpreted as particle fields.
The storage parameter can be specialized for other layouts. Functors and
result views are device-copyable named C++20 structs; accelerator functors use
`ALPAKA_FN_ACC` and may take the accelerator as their first argument.

## Flow operations

`Flow` is constructed with the device it runs on
(`Flow<Particle, TDevice> flow{device}`); the device must outlive the flow. The
calls below are the complete minimal surface, all implemented.

| Call | Contract |
| --- | --- |
| `source(name, functor)` | Map a logical item index to one initial payload. The node has one output. Exactly one source per graph. |
| `partition(name, classifier[, update])` | `classifier` declares `static constexpr std::uint32_t maxBins` and returns one bin for an immutable payload. The optional `update` receives the selected bin and a mutable copy and runs exactly once per emitted item. |
| `transform(name, functor)` | Update each payload and forward it on one output. The functor either mutates the payload in place or returns its replacement. No histogram or bin scatter is involved. |
| `sink(name, functor)` | Consume each payload once. Application result views are allowed and own their storage. |
| `connect(node, node)` | Connect the single output of a node to a node. A multi-output node must be connected through its ports. Fan-in and feedback are allowed. Each item follows exactly one successor. |
| `connect(out<Bin>(), node)` | Connect one compile-time output port of a partition to a destination node. Several bins may target the same node. |
| `configure(tuning)` | Replace the optional `graphFlow::Tuning` value. |
| `run(source, itemCount)` | Return after completion and resource retirement, or propagate an execution error. Returns the scheduler `Stats`. |

`Tuning` covers batching, storage and pool limits, and diagnostics. It never
changes payload semantics or routes; tuning is not an argument of the minimal
calls above.

## Examples

Particle feedback (Generate -> Filter -> {A, B} -> sink, with the feedback
outcome looping back into the filter):

```cpp
graphFlow::Flow<graphFlow::Particle, TDevice> flow{device};
auto source   = flow.source("generate", GenerateParticles{parameters.seed});
auto filter   = flow.partition("filter", FilterByProperty{parameters.seed});
auto processA = flow.partition(
    "process A",
    SelectOutcome{parameters.seed, graphFlow::rng::purposeProcessA, parameters.maxVisits},
    UpdateWeight{1000u});
auto processB = flow.partition(
    "process B",
    SelectOutcome{parameters.seed, graphFlow::rng::purposeProcessB, parameters.maxVisits},
    UpdateWeight{1u});
auto sink = flow.sink("record weight", RecordWeight{weights.getView()});
auto capped = flow.sink("record capped weight", RecordCapped{...});

flow.connect(source, filter);
flow.connect(filter.template out<0u>(), processA);
flow.connect(filter.template out<1u>(), processB);
flow.connect(processA.template out<Outcome::sink>(), sink);
flow.connect(processA.template out<Outcome::feedback>(), filter);
flow.connect(processA.template out<Outcome::capped>(), capped);
// ... processB ports likewise
flow.configure(tuning);
auto stats = flow.run(source, parameters.numParticles);
```

The full file is
[ParticleFeedback.hpp](https://github.com/psychocoderHPC-bot/graphFlow/blob/ac89ab558c26b938038035b26ca5dbe6d7ce7727/examples/particle_feedback/ParticleFeedback.hpp)
(121 lines).

Photon scattering (source -> propose flight -> advance and route -> scatter ->
back to propose flight, with the four terminal event ports feeding one shared
terminal sink):

```cpp
graphFlow::Flow<Photon, TDevice> flow{device};
auto source = flow.source("generate", GeneratePhotonT{trace});
auto flight = flow.transform("propose flight", ProposeFlight{slab, seed, maxScatters});
auto route  = flow.partition("advance and route", ClassifyEvent{}, AdvanceToEvent{slab});
auto scatter = flow.transform("scatter", ScatterT{parameters.mode, seed, trace});
auto terminal = flow.sink("terminal", RecordTerminal{records.getView(), visits.getView(), trace});

flow.connect(source, flight);
flow.connect(flight, route);
flow.connect(route.template out<Event::scatter>(), scatter);
flow.connect(scatter, flight);
flow.connect(route.template out<Event::transmitted>(), terminal);
flow.connect(route.template out<Event::reflected>(), terminal);
flow.connect(route.template out<Event::absorbed>(), terminal);
flow.connect(route.template out<Event::capped>(), terminal);
flow.configure(tuning);
auto stats = flow.run(source, parameters.numPhotons);
```

The full file is
[PhotonScattering.hpp](https://github.com/psychocoderHPC-bot/graphFlow/blob/ac89ab558c26b938038035b26ca5dbe6d7ce7727/examples/photon_scattering/PhotonScattering.hpp).

## Failure status

A functor may take an optional `Status&` argument after the optional
accelerator argument. Both examples use it: the particle stages report a
weight-overflow or out-of-range classifier result, and the photon stages
report an invalid geometry or proposal before any output is published. The
random draws behind both examples use the four-purpose-compatible mapping of
the RNG section below.

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

## RNG mapping

Random draws come from the public `alpaka::rand::engine::Philox4x32x10`; the
project wrapper `graphFlow::rng` (in `include/graphFlow/rng/ParticleRng.hpp`)
encodes a key-based position per `(seed, id, visit, purpose)` so batching and
scatter order never change a draw. The particle example keeps the three-purpose
mapping `4 * (1 + 3 * visit + purpose)` (`purposesPerVisit = 3`). The photon
example reserves four purposes per completed scatter count
(`photonPurposesPerVisit = 4`): `purposeFlight`, `purposeCollisionKind`,
`purposeCosine`, `purposeAzimuth`. The offset formula generalizes to
`4 * (1 + purposesPerVisit * visit + purpose)`.

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
