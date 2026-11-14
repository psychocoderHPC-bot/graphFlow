<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Benchmarks

This page contains the historical pre-T5/T7 baseline and the integrated T13
result. The integrated numbers are the recorded T13 evidence for main
`68124ad3875e42cbe09247f64bd218a7ac8d58fc`, supplied from the run on that
revision. They are not a fresh run in this worktree, and every number on this
page states its provenance. The full matrix, budget rejections, and raw-row
provenance live in [tools/BENCH_TABLE.md](../tools/BENCH_TABLE.md) (PR #8).

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

## Integrated T13 result

Integrated main `68124ad3875e42cbe09247f64bd218a7ac8d58fc` (T5 batching, T7
pool payload budget). The numbers are the supplied T13 evidence for that
revision, not a fresh run in this worktree.

hal A30, CUDA sm_80, `C=8192`, seed 42:

| N | Q | Wall time | Histogram sum | particles/s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 100,000,000 | 1 | 218.87 s | 100,000,000 | ~4.57e5 | 264 rows |
| 100,000,000 | 2 | 9.83 s | 100,000,000 | ~1.02e7 | 264 rows |
| 100,000,000 | 8 | 8.50 s | 100,000,000 | ~1.18e7 | 264 rows |
| 1,000,000 | 1 | 2.32 s | 1,000,000 | ~4.31e5 | 150 rows |
| 1,000,000 | 2 | 0.41 s | 1,000,000 | ~2.44e6 | 150 rows |
| 1,000,000 | 8 | 0.35 s | 1,000,000 | ~2.86e6 | 150 rows |

These runs used `P=16128`, which leaves the documented 196,608 B of headroom
under the measured A30 budget (see the budget row below). The auto-derived
default now clamps to the same P; an explicit `P=16384` is rejected on this
device. The sorted histogram rows are identical
across Q within each N (identical SHA-256 per N; the `N=1e8` and `N=1e6` row
sets differ, as expected for different particle counts).

CPU/CUDA histogram equality, `C=128`, `P=1024`, `Q=1`, seed 42:

| N | CPU wall s | CUDA wall s | Histogram sum | rows_equal |
| ---: | ---: | ---: | ---: | --- |
| 100,000 | 15.23 | 1.96 | 100,000 | yes |
| 1,000,000 | 149.44 | 17.66 | 1,000,000 | yes |

fwk394, CPU, `C=1024`, `P=512`, `Q=1`: `N=1,000,000` took 52.75 s
(~1.90e4 particles/s), histogram sum 1,000,000. The fwk394 CUDA sm_75 build
fails with an nvcc 12.5 internal compiler error, so no fwk394 CUDA run is
claimed.

Payload budget against the measured 3,171,090,432 B A30 budget:

| Pair | Payload B | Result |
| --- | ---: | --- |
| `C=8192`, `P=16128` | 3,170,893,824 | passes |
| `C=8192`, `P=16384` | 3,221,225,472 | rejected |
| `C=8192`, `P=65536` | 12,884,901,888 | rejected |

## Current state and caveats

- The integrated main adds device-aware `C`/`P`/`T`, multi-chunk batching with
  an adaptive prefix, and the pool payload budget. The historical numbers above
  predate those changes.
- The T13 integrated benchmark is recorded in the section above. No number
  here goes beyond that evidence, and no new measurement was taken for this
  page.
- The former `C=8192`, `P=65536` pool is rejected by the payload budget. Its
  historical 65542-chunk peak is not a current default.
- `hal` reports `sharing=unknown` and fwk394 is a shared host, so the
  historical wall times and the derived `5.33e5` particles/s are a baseline,
  not an exclusive-hardware acceptance claim. The same caveat applies to the
  integrated T13 table: all of its wall times and derived particles/s values
  are single-run baselines on shared hosts, not exclusive-hardware acceptance
  numbers.
- HIP execution is validated for correctness on `hal` (AMD Radeon RX 7900 XTX,
  `gfx1100`, ROCm 7.2) with 245/245 CTest (43 HIP cases), but no HIP benchmark
  is recorded.
  `hal` reports `sharing=unknown`, so no HIP performance number is claimed.
- No RTX 2080 Ti run is recorded. The `sm_75` build is a documented target.
- There is no sustained copy-bandwidth reference for the same device and data
  size yet.
- No CI job runs these benchmarks.
