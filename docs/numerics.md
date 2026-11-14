<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Numerics

The particle example is integer-only. The photon example is floating-point.
Random draws use the public `alpaka::rand::engine::Philox4x32x10`; graphFlow
does not implement an RNG algorithm. graphFlow builds one initialized engine
per logical `(id, event, purpose)` position and passes it to an opting-in
functor. The functor never receives a seed, an offset, or a purpose count.

## Stream initialization

The root seed comes from `--seed` and is stored in `Tuning::seed`. graphFlow
derives every item stream from that seed and a packed logical position:

```text
packed = id[31:0] | event[15:0] << 32 | purpose[15:0] << 48
key    = mix64(packed ^ mix64(baseSeed))
engine = Philox4x32x10(key, subsequence = 0, offset = 0)
```

`mix64` is the SplitMix64 finalizer: three xor-shift permutations and two odd
64-bit multiplications. Every step is invertible, so `packed -> key` is a
bijection for a fixed base seed. The engine is always constructed with
subsequence and offset zero. That avoids the pinned constructor-offset defect
below and needs no `O(event)` advance loop.

`id` is the logical item index (particle id or photon id) and must be unique
within a run. `event` is the particle visit or the photon scatter count.
`purpose` is internal: Flow assigns `2 * nodeKey` to a node functor and
`2 * nodeKey + 1` to an optional partition updater. Registration order is part
of the reproducibility contract; users do not choose purposes.

Streams are disjoint when `(id, event, purpose)` differs. They do not depend on
chunk, tile, thread, batch, backend, or scheduling order, so a change to `Q` or
`C` cannot change a draw. Reclassified, capped, or retried items keep their
`(id, event)` position and a different stage purpose, so they cannot advance
another item's stream.

Limits are explicit: 32 bits for the id, 16 bits for the event, and 16 bits for
the purpose. `checkedStreamKey()` rejects an out-of-range position on the host;
`makeEngine()` asserts the same precondition on the device. The particle and
photon examples validate their visit/scatter caps against the event range.

## Philox constructor-offset defect

The pinned alpaka tree has a defect in the single-value `Philox4x32x10`
constructor offset path:

- `PhiloxSingle::skip()` computes the within-block position, then
  `advanceState()` resets the position to zero.
- `skip4()` derives carry words from `counter[0] < temp[1]` and
  `counter[0] < temp[2]` instead of the corresponding counter words.
- With a nonzero subsequence, even `skip4(0)` changes counter word 3.

graphFlow therefore never passes a nonzero subsequence or offset. The packed
key carries the complete logical position and the engine starts at its first
output. `tests/unit/test_philox.cpp` pins the keyed construction, the uniqueness
of packed keys, the limits, and the zero-offset equivalence.

## Stream positions and draw order

| Draw | Stream position |
| --- | --- |
| Particle property | `(id, event 0, source purpose)` |
| Filter at visit `v` | `(id, v, filter purpose)` |
| Process A at visit `v` | `(id, v, A purpose)` |
| Process B at visit `v` | `(id, v, B purpose)` |
| Photon flight and collision kind | `(id, scatterCount, flight purpose)`, two consecutive draws |
| Photon cosine and azimuth | `(id, scatterCount, scatter purpose)`, two consecutive draws |

The particle property and the Filter decision are separate engines, so they
cannot alias even though both are the first draw of their stream. Photon flight
uses `UniformReal<double, interval::CO>`; the exponential sample uses `1 - U`
so `log` receives a positive value. The collision kind uses a second `[0, 1)`
draw from the same propose-stage engine. Isotropic scattering maps a `[0, 1)`
cosine draw to `2U - 1` and uses a second `[0, 1)` draw for the azimuth. No
matrix is shared between items or stages.

`tests/reference/ScalarModel.hpp` and `tests/reference/ScalarTransport.hpp` do
not include the production RNG header. They re-implement the packing and mixer
and construct their own zero-offset engines. `tests/unit/test_philox.cpp` pins
literal keyed outputs for `(seed=42, id=7, event, purpose)`, so a shared mapping
mistake cannot hide behind the oracle.

## Baseline change

The engine-based streams replace the old `4 * (1 + 3 * visit + purpose)`
Philox offset map and the photon midpoint word conversion. Fixed-seed particle
weights, photon outcomes, and histograms therefore change once. The new
baselines are the values produced at the RNG API change; tests compare against
the independent oracle and no longer pin the old offset literals.

## Integer decisions

```text
Generate: property = engine(id, 0, source)()
Filter:   engine(id, visit, filter)() >= property ? bin 1 : bin 0
A:        engine(id, visit, A)() >= 0x80000000 ? weight += 1000, visit += 1, feedback
B:        engine(id, visit, B)() >= 0x80000000 ? weight += 1,    visit += 1, feedback
```

Every comparison is on `uint32_t`. There are no floating-point thresholds and
no backend-dependent rounding.

## Checked weight addition

The A and B stages add 1000 and 1. Before the mutation each stage checks

```text
weight <= UINT32_MAX - increment
```

On overflow it writes the device status word and leaves the particle
unchanged. The status word is a hard batch error: the host aborts the batch and
does not publish partial output. The scatter's `next[bin] == total[bin]` check
runs only after the status word is clean.

## Max visits

A particle that would feed back but already has `visit == maxVisits` is routed
to the sink without a mutation and counted as non-terminating. The classification
is `visit < maxVisits ? feedback : sink` after the survival draw. A particle
that fails the survival draw routes to the sink without a mutation. A particle
that survives but is capped is counted through the device-scope capped counter.

## Histogram acceptance

Acceptance is the sorted, run-length encoded histogram of final weights:

- rows are `weight,count`;
- weights are strictly increasing;
- every count is positive;
- counts sum to `N`.

Scatter order is unspecified, so per-particle order is not a gate. Batching
changes the launch and scatter order, not the draw positions. The integration
tests additionally compare the weight buffer element-wise against the scalar
oracle as a diagnostic for lost or duplicated ids. The acceptance contract
stays histogram multiset equality.

## CPU and CUDA equality

The historical pre-T5/T7 CPU and CUDA `N=1e6` runs with seed 42, `C=8192`, and
`P=65536` produced identical histograms by direct row diff. That check was not
repeated on the integrated revision, where the pair is rejected by the payload
budget. HIP compiles the same code path for `gfx90a`, but no AMD device was
visible, so no HIP numeric result is claimed.
