<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Benchmarks

This page contains the historical pre-T5/T7 baseline only. It is not a current
integrated run. The integrated performance matrix is pending in T13, and this
page does not invent a post-T5/T7 number.

## Historical environment

| Item | Value |
| --- | --- |
| Host | hal container, `sharing=unknown` |
| GPU | NVIDIA A30, `sm_80`, 24 GiB |
| CUDA | 13.4 |
| CPU compiler | g++ 13.3 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| alpaka | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` (3.0.0) |
| Build | Release |

## Historical invocations

These commands use the pre-T5/T7 settings `C=8192`, `P=65536`. The current
policy rejects that pair because its payload exceeds the budget. Keep them only
as the invocation record for the numbers below.

CPU:

```sh
./build-cpu/graphFlow_particle_feedback --backend cpu --particles 1000000 \
  --chunk-size 8192 --live-particles 65536 --seed 42 --max-visits 64
```

CUDA:

```sh
./build-cuda/graphFlow_particle_feedback --backend cuda --particles 100000 \
  --chunk-size 8192 --live-particles 65536 --seed 42 --max-visits 64
./build-cuda/graphFlow_particle_feedback --backend cuda --particles 1000000 \
  --chunk-size 8192 --live-particles 65536 --seed 42 --max-visits 64
./build-cuda/graphFlow_particle_feedback --backend cuda --particles 100000000 \
  --chunk-size 8192 --live-particles 65536 --seed 42 --max-visits 64
```

## Historical results

| Backend | N | Wall time | Histogram sum | Peak live chunks (pool) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 1,000,000 | 19.45 s | 1,000,000 | not recorded | maxrss 76.8 MB |
| CUDA | 100,000 | 0.58 s | 100,000 | 514 (65542) | |
| CUDA | 1,000,000 | 2.24 s | 1,000,000 | 857 (65542) | |
| CUDA | 100,000,000 | 187.5 s | 100,000,000 | 877 (65542) | 264 rows; about 5.33e5 particles/s |

The historical `N=1e8` histogram summed to exactly 100,000,000 over 264 rows.
The historical CPU and CUDA `N=1e6` histograms were identical by direct row
diff.

## Historical full-chunk source admission

Before full-chunk admission, the scheduler admitted source chunks whenever any
particle budget was free. Under `N >> P` that produced one-particle chunks.
The historical `N=1e6` CUDA run took about 119 s and peaked at 8195 of 8198
chunks.

After the change a source chunk must carry at least `min(C, P, remaining)`
particles. The same historical `N=1e6` run took 2.24 s and peaked at 857 of
65542 chunks. Consumer ready lists still dispatch partial chunks immediately;
the minimum applies only to source generation.

## Historical correctness checks

- The header reported the selected backend, device, and pool settings.
- Every histogram row had a strictly increasing weight and a positive count.
- The count column summed to `N`.
- The `N=1e8` result had an exact sum of 100,000,000.
- The CPU and CUDA `N=1e6` rows matched by direct diff.

The acceptance contract is histogram multiset equality, not per-particle order.

## Current state and caveats

- The integrated main adds device-aware `C`/`P`/`T`, multi-chunk batching with
  an adaptive prefix, and the pool payload budget. The numbers above predate
  those changes.
- The T13 integrated benchmark is pending. No current integrated wall time,
  particles-per-second value, or peak count is claimed here.
- The former `C=8192`, `P=65536` pool is rejected by the payload budget. Its
  historical 65542-chunk peak is not a current default.
- `hal` is shared, so the historical wall times and the derived `5.33e5`
  particles/s are a baseline, not an exclusive-hardware acceptance claim.
- No AMD GPU was visible. HIP is compile-only for `gfx90a`, so there is no HIP
  benchmark.
- No RTX 2080 Ti run is recorded. The `sm_75` build is a documented target.
- There is no sustained copy-bandwidth reference for the same device and data
  size yet.
- No CI job runs these benchmarks.
