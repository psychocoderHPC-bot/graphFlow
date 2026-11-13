<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Public API contract

This document freezes the application-facing API of the simplified graphFlow
interface. The typed execution path is implemented in `flow/Execution.hpp`. The
particle-feedback example declares its graph with `Flow` handles and named
outcome ports, while `ParticleStages.hpp` remains only for the older low-level
tests. The API is additive and replaces no existing public type.

## Payload

`graphFlow::Flow<Payload>` owns one graph for one payload type. A payload is a
trivially copyable, default-constructible record. Its fields and meanings belong
to the application. graphFlow copies payloads by value and may send them through
host and device memory. Functors and result views are device-copyable named C++20
structs; accelerator functors use `ALPAKA_FN_ACC` and may take the accelerator as
their first argument.

## Flow operations

`Flow<Payload>` is constructed with the device it runs on (`Flow<Payload> flow{device}`);
the device must outlive the flow. The calls below are the complete minimal surface.

| Call | Contract |
| --- | --- |
| `source(name, functor)` | Map a logical item index to one initial payload. The node has one output. |
| `partition(name, classifier[, update])` | `classifier` declares `static constexpr std::uint32_t maxBins` and returns one bin for an immutable payload. The optional `update` receives the selected bin and a mutable copy and runs exactly once per emitted item. |
| `transform(name, functor)` | Update each payload and forward it on one output. No histogram or bin scatter is involved. The first version uses an out-of-place path or proves exclusive ownership for in-place reuse. |
| `sink(name, functor)` | Consume each payload once. Application result views are allowed and own their storage. |
| `connect(output, node)` | Connect a node or output port to a node. Fan-in and feedback are allowed. Each item follows exactly one successor. |
| `run(source, itemCount)` | Return after completion and resource retirement, or propagate an execution error. |

`Flow` also accepts an optional `graphFlow::Tuning` value outside these calls.
`Tuning` covers batching, storage and pool limits, and diagnostics. It never
changes payload semantics or routes; tuning is not an argument of the minimal
calls above.

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
