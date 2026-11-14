<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Numerics

The particle example is integer-only. The photon example is floating-point.
Random draws come from the public `alpaka::rand::engine::Philox4x32x10`
(RNG4); there is no project RNG algorithm. The draw position is key-based: it
depends only on seed, item id, visit, and purpose, so batching and scatter
order do not change the draws.

## Philox constructor-offset defect

The pinned alpaka tree has a defect in the single-value `Philox4x32x10`
constructor offset path:

- `PhiloxSingle::skip()` computes the within-block position, then
  `advanceState()` resets the position to zero.
- `skip4()` derives carry words from `counter[0] < temp[1]` and
  `counter[0] < temp[2]` instead of the corresponding counter words.
- The remainder `offset % 4` is discarded.

As a result, `Philox4x32x10(seed, subsequence, offset)` is not equivalent to
advancing a fresh engine by `offset` draws. With a nonzero subsequence even
`skip4(0)` changes counter word 3, so a multiple-of-four offset is not safe
either.

The project workaround is `graphFlow::rng::drawAt` in
`include/graphFlow/rng/ParticleRng.hpp`:

1. Construct `Philox4x32x10(seed, particleId, 0)`.
2. Call the engine once for the first output.
3. Call it `offset` more times.
4. Return the last value.

The draw position is key-based. It depends only on `(seed, particleId, offset)`
and not on chunk, tile, thread, batch, backend, or scheduling order. The public
engine stays the only RNG implementation. The workaround is pinned by
`tests/unit/test_philox.cpp`.

## Draw offsets

The particle mapping reserves `purposesPerVisit = 3` draws per visit; the
photon mapping reserves four independent draws per completed scatter count.

| Draw | Offset |
| --- | --- |
| Property (particle) | `0` |
| Filter at visit `v` | `4 * (1 + 3*v + 0)` |
| Process A at visit `v` | `4 * (1 + 3*v + 1)` |
| Process B at visit `v` | `4 * (1 + 3*v + 2)` |
| Photon flight, kind, cosine, azimuth | `4 * (1 + 4*visit + purpose)` for purposes 0 through 3 |

Each decision owns a block of four 32-bit outputs. The property draw uses offset
zero and the decision blocks start at offset four, so the ranges never overlap.
`graphFlow::rng::decisionOffset(visit, purpose, purposesPerVisit)` implements
the mapping.

`tests/reference/ScalarModel.hpp` does not include the production RNG header. It
constructs the engine at offset zero, advances locally, and applies the same
formula. `tests/unit/test_philox.cpp` pins literal draws for
`(seed=42, id=7, visit, purpose)`, so a shared offset mistake cannot hide behind
the oracle.

## Integer decisions

```text
Generate: property = draw(seed, id, 0)
Filter:   draw(seed, id, visit, Filter) >= property ? bin 1 : bin 0
A:        draw(seed, id, visit, A) >= 0x80000000 ? weight += 1000, visit += 1, feedback
B:        draw(seed, id, visit, B) >= 0x80000000 ? weight += 1,    visit += 1, feedback
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
