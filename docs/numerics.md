<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Numerics

The particle example is integer-only. Random draws come from the public
`alpaka::rand::engine::Philox4x32x10`; there is no project RNG algorithm. The
photon example uses the same public engine with a four-purpose mapping and
double-precision sampling.

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

| Draw | Offset |
| --- | --- |
| Property | `0` |
| Filter at visit `v` | `4 * (1 + 3*v + 0)` |
| Process A at visit `v` | `4 * (1 + 3*v + 1)` |
| Process B at visit `v` | `4 * (1 + 3*v + 2)` |

Each decision owns a block of four 32-bit outputs. The property draw uses offset
zero and the decision blocks start at offset four, so the ranges never overlap.
`graphFlow::rng::decisionOffset(visit, purpose)` implements the mapping.

`tests/reference/ScalarModel.hpp` does not include the production RNG header. It
constructs the engine at offset zero, advances locally, and applies the same
formula. `tests/unit/test_philox.cpp` pins literal draws for
`(seed=42, id=7, visit, purpose)`, so a shared offset mistake cannot hide behind
the oracle.

## Photon scattering RNG (RNG4)

The photon example reserves four independent draws per completed scatter count
(`photonPurposesPerVisit = 4`): `purposeFlight`, `purposeCollisionKind`,
`purposeCosine`, `purposeAzimuth`. The offset formula generalizes the particle
mapping to

```text
4 * (1 + 4 * visit + purpose)
```

where `visit` is the photon's `scatterCount`. The draw positions are key-based
and depend only on `(seed, id, scatterCount, purpose)`, not on chunk, tile,
thread, backend, or scheduling order.

Sampling uses the open-interval uniform `(word + 0.5) * 2^-32`, strictly inside
`(0, 1)`, so `log(U)` is finite and the exponential free-flight sample is always
positive:

| Draw | Purpose | Use |
| --- | --- | --- |
| `purposeFlight` | flight | `sampledDistance = -log(U) / sigmaTotal` |
| `purposeCollisionKind` | collision kind | competition between scatter and absorption |
| `purposeCosine` | cosine | `mu = 2U - 1` (isotropic mode) |
| `purposeAzimuth` | azimuth | `phi = 2*pi*V` (isotropic mode) |

Forward mode reuses the incoming direction and does not consume the cosine and
azimuth draws. The collision count follows `Poisson(sigma_s * L)` in the
expected value; absorption competes through `sigma_a`.

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

## Photon terminal accounting

Each photon receives exactly one terminal record. The host pass checks that the
terminal visit count equals one and that the recorded event is within the five
event bins (`scatter`, `transmitted`, `reflected`, `absorbed`, `capped`); a
record that fails either check is counted as inconsistent and does not enter
the outcome counts. The per-event counts sum to `N`, which is what the CLI
tests and the `# terminal-sum` line check. A capped history is a routed
outcome, not a failure, so it still terminates normally and is counted.

## CPU and CUDA equality

The particle port to the `Flow` API (T4) was validated against a fixed-seed
33-invocation CLI baseline (the 24 N/C cases plus nine Q/P/T/frameCap and
max-visits cases); the example header, policy line, histogram rows,
non-terminating count, and exit status are identical to the pre-change binary
on both backends. The historical pre-T5/T7 CPU and CUDA `N=1e6` runs with seed
42, `C=8192`, and `P=65536` produced identical histograms by direct row diff;
that check was not repeated on the integrated revision, where the pair is
rejected by the payload budget. HIP compiles the same code path for `gfx90a`,
but no AMD device was visible, so no HIP numeric result is claimed.
