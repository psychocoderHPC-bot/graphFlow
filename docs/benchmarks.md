<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Benchmarks

This page contains the P07 exclusive measurement merged into `a610281`, the
P07b Q/C re-run and kernel-axis matrix merged at `56c11ac`, the
historical pre-T5/T7 baseline, and the historical integrated T13 result. Every
number states its provenance. The full matrix, budget rejections, copy
reference, kernel matrix, and raw-row provenance live in
[tools/BENCH_TABLE.md](../tools/BENCH_TABLE.md).

## P07 measurement matrix and frozen target (code `64d4a1e`)

Provenance: code `64d4a1e11949886c7147c082cc386edde4bdb01f` (merged as
`a61028192884bb92e0107b32f20e12b5c9ca76c9`), one exclusive
`gpu-v100-int` allocation on `rosi_v100` (`gv001`), Tesla V100-SXM2-32GB,
driver 570.158.01, CUDA 12.6.85, `sm_70`, Release, three repetitions per case.
The tables report median and interquartile range across the three runs. No H100
was used.

Q sweep at `N=100,000,000`, `C=8192`, `P=16384`, seed 42:

| Q | median wall s | IQR wall s | median particles/s | IQR particles/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 191.990 | 2.315 | 5.209e5 | 6.18e3 |
| 2 | 8.710 | 0.535 | 1.148e7 | 6.59e5 |
| 8 | 7.690 | 0.045 | 1.300e7 | 7.56e4 |

Every run printed 260 histogram rows summing to exactly `100,000,000`; the
sorted rows are identical across Q, C, P, and repetitions and hash to
`21ab0aeb9ec72d380af2ce6e9631f8fa7b9111d24ff7831f272aa9f6a7be9dab`.

C-wave sweep at `P=8192`, Q=8 (1, 2, and 4 resident `C`-chunks):

| C | P/C | median wall s | IQR wall s | median particles/s |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 4 | 11.010 | 0.045 | 9.083e6 |
| 4096 | 2 | 11.330 | 0.025 | 8.826e6 |
| 8192 | 1 | 22.980 | 0.090 | 4.352e6 |

The frozen target is `13,000,000` particles/s at `N=1e8`, `C=8192`, `P=16384`,
Q=8, with a 10 percent tolerance (floor `11,700,000`). The measured median is
`13,003,901` particles/s, so the gate passes. The nominal value is rounded from
the earlier exclusive `rosi_v100` Q=8 run at `f3ef355` (7.676 s, 1.303e7
particles/s). `tools/bench_matrix.sh --target-pps ...` enforces the gate and
writes `target.md`; no CI job invokes it yet.

Sustained copy-bandwidth reference on the same device and payload
(3,221,225,472 B, 10 device-to-device copies): `409.733` GB/s, 7.862 ms per
copy. This is a D2D copy-engine reference, not a scatter-bandwidth claim.

## P07b Q/C re-run and kernel axes (main `56c11ac`)

This run closes the P07 kernel-level gap. An opt-in `--workload kernel` path in
the example selects the P03 `ScatterMode` (`shared|direct`), the dense/hash bin
table (`--bin-path`), the bin count (`--bins 64|1024|4096|65536`) and the
occupancy (`--distribution balanced|one-hot|skewed|sparse`). The default
feedback workload and its output are unchanged. The kernel workload is a
separate Generate -> Transform -> Sink pipeline, so its particles/s is not
comparable with the feedback Q/C numbers above.

Provenance: main `a61028192884bb92e0107b32f20e12b5c9ca76c9` plus the P07b
change set on branch `gf/p07b-e2e`, merged as
`56c11acbca6d10ea82fd85d213abcb4f9050dbf8`, exclusive `rosi_v100` (`gv001`),
Tesla V100-SXM2-32GB, driver 570.158.01, CUDA 12.6.85, `sm_70`, Release,
three repetitions. Binary
SHA-256 `23c2bbeed6b9b7b0216ffaf865eaa081a80176b44d14ab05f163ba8a749775b4`.
No H100 was used.

The same run re-ran the Q/C matrix at `N=1e8`, `C=8192`, `P=16384`. The Q=8
median is `12,987,013` particles/s, so the `13,000,000` target with its 10
percent floor (`11,700,000`) passes. All nine Q runs print 260 rows summing to
exactly `100,000,000` with SHA-256
`21ab0aeb9ec72d380af2ce6e9631f8fa7b9111d24ff7831f272aa9f6a7be9dab`. The C
wave at `P=8192`, Q=8 takes 11.12 s (C=2048), 11.36 s (C=4096) and 23.01 s
(C=8192). The full tables are in
[tools/BENCH_TABLE.md](../tools/BENCH_TABLE.md#p07b-qc-re-run-and-kernel-axis-e2e).

| scatter | bin path | bins | distribution | N | median wall s | IQR wall s | median particles/s |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| shared | hash | 4096 | sparse | 262144 | 2.030 | 0.010 | 1.2913e5 |
| shared | hash | 65536 | sparse | 262144 | 2.170 | 0.005 | 1.2080e5 |
| direct | dense | 64 | balanced | 1000000 | 0.340 | 0.000 | 2.9412e6 |
| shared | dense | 64 | balanced | 1000000 | 0.340 | 0.005 | 2.9412e6 |
| shared | dense | 64 | one-hot | 1000000 | 0.210 | 0.005 | 4.7619e6 |
| shared | dense | 64 | skewed | 1000000 | 0.230 | 0.010 | 4.3478e6 |
| direct | dense | 1024 | balanced | 1000000 | 2.000 | 0.000 | 5.0000e5 |
| shared | dense | 1024 | balanced | 1000000 | 2.000 | 0.005 | 5.0000e5 |
| shared | hash | 64 | balanced | 1000000 | 0.340 | 0.010 | 2.9412e6 |
| shared | hash | 1024 | balanced | 1000000 | 2.000 | 0.020 | 5.0000e5 |

All 30 runs exited 0 and their histogram sums equal N. The sorted rows are
identical across repetitions, scatter modes and dense/hash at the same bin
count; the kernel-axes histogram SHA-256 values are in
[tools/BENCH_TABLE.md](../tools/BENCH_TABLE.md#p07b-qc-re-run-and-kernel-axis-e2e).
The 4096/65536 sparse cases use N=262144 because per-bin planning produces one
sink chunk per particle there. The wall timer has 10 ms resolution, so the
sub-second cases are quantized. `sparse` is the P03 many-bin balanced mapping
over B>=1024; the committed tests have no separate sparse classifier. `direct`
is dense-only, and the hash path assumes the 2*T table with T=128.

The committed CPU kernel timings remain the P03 measurement on fwk394: shared
staging versus direct 5.48 vs 15.85 ms (balanced), 2.13 vs 48.31 ms (one-hot),
2.21 vs 37.25 ms (skewed), 4.19 vs 6.62 ms (sparse), and dense 4.06 ms versus
5.11/4.90 ms for 2*T/4*T at B=1024. P07b did not re-measure those. The `Direct`
scatter path was measured on CPU by P03 and on CUDA at kernel level by P07b; no
HIP `Direct` performance run is recorded. P07b closes the kernel-level
scatter/dense/distribution gap and exercises the dense `direct` mode on
`rosi_v100` CUDA. PR #63 routes the same axes through the public `Flow`, but no
public-Flow performance matrix is recorded, so the measurement-matrix item
stays open for those numbers.

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

## N=1e8 acceptance on rosi_v100 (V100 SXM2-32GB)

The P0 acceptance run is re-recorded on `rosi_v100` at main `f3ef355`
(V100 SXM2-32GB, CUDA 12.6.85, `sm_70`, Release), three repetitions.
Parameters: `C=8192`, `P=16384`, `T=128`, seed 42, max-visits 64. The
`P=16384` payload (3,221,225,472 B) fits the V100 SXM2-32GB budget; the A30
run needed `P=16128` against the smaller measured A30 budget.

| N | Q | Wall time | Histogram sum | particles/s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 100,000,000 | 1 | 194.685 s | 100,000,000 | ~5.14e5 | 260 rows |
| 100,000,000 | 2 | 9.093 s | 100,000,000 | ~1.10e7 | 260 rows |
| 100,000,000 | 8 | 7.676 s | 100,000,000 | 1.30e7 | 260 rows |

The sorted histogram has 260 rows and the count column sums to exactly
100,000,000 for every Q; the sorted `weight,count` rows hash to SHA-256
`89dc7fb9962e898fe2f1fe0116ad98fe2df9dceb77d33aa890a56467615d02e9`. Peak
memory is 5702 MiB. This is the recorded P0 acceptance run; the hal A30
numbers above stay labelled shared-host baselines. H100 was not used, so no
H100 claim is made.

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
  numbers. The P07 table at the top is the exception: it is the recorded
  exclusive-platform run on `rosi_v100` (V100 SXM2-32GB). The older `f3ef355`
  acceptance table is historical.
 - HIP execution is validated for correctness on `hal` (AMD Radeon RX 7900 XTX,
   `gfx1100`, ROCm 7.2); the current suite is 307/307 at
   `561b9a4c9435e37b164623bb6be60a3051783eb4` and the previous 301/301 at
   `ffe4074`, 298/298 at `88de14f`, 297/297 at `56c11ac`, the 290/290 at
   `a610281`, and the 245/245 run at `b972eaa` are historical.
  No HIP benchmark is recorded.
  `hal` reports `sharing=unknown`, so no HIP performance number is claimed.
- No RTX 2080 Ti run is recorded. The `sm_75` build is a documented target.
- The P07 section records a sustained D2D copy reference (`409.733` GB/s at
  3,221,225,472 B) on the V100. It is a context number, not a scatter
  bandwidth. The P07b section adds end-to-end kernel-axis numbers for scatter,
  dense/hash and the four distributions; the P03 CPU kernel timings remain the
  only isolated-kernel provenance.
- No CI job runs these benchmarks. The frozen target is implemented in
  `tools/bench_matrix.sh` and passes on the P07 run; wiring it into CI is open.
