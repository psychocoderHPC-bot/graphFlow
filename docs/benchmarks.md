<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Benchmarks

This page contains the P07 exclusive measurement merged into `a610281`, the
P07b Q/C re-run and kernel-axis matrix merged at `56c11ac`, the
historical pre-T5/T7 baseline, the historical integrated T13 result, and the
chunk-pipeline measured results (occupancy target, default-path parity, the
frozen p07 gate on reference hardware, and the slot arena). Every number states
its provenance. The full matrix, budget rejections, copy reference, kernel
matrix, and raw-row provenance live in
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
(~1.90e4 particles/s), histogram sum 1,000,000. That CPU run predates the
current CUDA suite on this host, so no wall-time comparison is claimed; the
per-chunk transform schedule section below records the fwk394 RTX 2080 Ti
`sm_75` measurements.

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

## Measured AoS vs default (main `3ee436d`)

Provenance: main `3ee436dfcdbf28999538d165517b3565902f0d07`, N=1e7, C=8192,
P=16384, Q=8, T=128, seed 42. Three processes with five repetitions each per
configuration; the first repetition of each process is a warmup, so every
median below covers the four steady repetitions of its process (12 samples per
configuration). Every process prints the full value histogram; its count column
equals N exactly and the value digest `f0362124faddd01a` (258 rows) matches
between the two storages. Both configurations run the same topology over the
same logical payload with the same 8 GiB pool budget; the AoS slot is 16 B
against the 24 B default slot, so the wall-time difference reflects the storage
layout, not a different workload. The default storage selection is unchanged.
No H100 was used. No claim beyond this setup is made.

| Backend | AoS wall s | default wall s | AoS faster by |
| --- | ---: | ---: | ---: |
| CPU | 1.3075 | 1.3211 | 1.0% |
| CUDA (A30) | 0.2797 | 0.3036 | 7.9% |
| HIP (gfx1100) | 0.4601 | 0.4773 | 3.6% |
| SYCL (A770) | 3.5111 | 3.6457 | 3.7% |

The AoS median is at or below the default median in every row, so no
configuration shows storage-attributable overhead. The HIP row has overlapping
interquartile ranges and a 7% spread across its three processes; the SYCL row
is bimodal, with one AoS process at 2.59 s. The three device rows ran on `hal`,
which reports `sharing=unknown`, so they are baselines, not acceptance numbers.

## Current state and caveats

- The integrated main adds device-aware `C`/`P`/`T`, multi-chunk batching with
  an adaptive prefix, and the pool payload budget. The historical numbers above
  predate those changes.
- The T13 integrated benchmark is recorded in the T13 section above. No number
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
   `gfx1100`, ROCm 7.2); the current suite is 380/380 at
   `25c73f31c736b1c5f765003ff8815d2a24c25808` and the previous 307/307 at
   `561b9a4c`, 301/301 at
   `ffe4074`, 298/298 at `88de14f`, 297/297 at `56c11ac`, the 290/290 at
   `a610281`, and the 245/245 run at `b972eaa` are historical.
   The measured AoS section records HIP wall times on `hal`; because `hal`
   reports `sharing=unknown`, those are baselines, not an acceptance claim.
- The current CUDA suite (389/389 at
  `25c73f31c736b1c5f765003ff8815d2a24c25808`) runs on the RTX 2080 Ti
  `sm_75`, on `fwk394`, correctness only. The exclusive performance record
  (P07b) remains the V100 `sm_70` measurement at `56c11ac`, which predates the
  AoS-over-SoA additions. No benchmark was taken at `3b90ffd`; the measured
  AoS section above is the only record that compares the default and opt-in
  storages, and it is at `3ee436d`.
- The P07 section records a sustained D2D copy reference (`409.733` GB/s at
  3,221,225,472 B) on the V100. It is a context number, not a scatter
  bandwidth. The P07b section adds end-to-end kernel-axis numbers for scatter,
  dense/hash and the four distributions; the P03 CPU kernel timings remain the
  only isolated-kernel provenance.
- No CI job runs these benchmarks. The frozen target is implemented in
  `tools/bench_matrix.sh` and passes on the P07 run; wiring it into CI is open.

## Per-chunk transform schedule (photon scattering)

Provenance: code `c7ff8e21e509733c17c90968f8d6656dce03466c`
(branch `issue-83-per-chunk-schedule`), one process per arm, interleaved
repetitions, median across repetitions. Harness, identical for both arms and
every card:

```sh
./graphFlow_photon_scattering --backend cuda --photons 4000000 --thickness 4 \
  --sigma-s 0.5 --sigma-a 0.05 --mode isotropic --max-scatters 32 --seed 7 \
  --batch-chunks 8 [--chunk-kernel-schedule perChunk --chunk-queue-depth 4]
```

N=4M photons, Q=8, seed 7, Release. The off arm is the default schedule; the
perChunk arm uses queue depth 4. The physics output is byte-identical across
arms: md5 `9cbb15ab113a85ff23f1fa406dc0eda8` in every accepted repetition on
every card.

| Card | off wall s | perChunk (depth 4) wall s | delta |
| --- | ---: | ---: | ---: |
| A30 `sm_80` | 1.745 | 1.655 | -4.61% |
| RTX 2080 Ti `sm_75` | 1.5452 | 1.4072 | -8.90% |
| Quadro GV100 `sm_70` | 1.2698 | 1.0910 | -13.67% |

The off path against the unmodified binary in the same window is +0.37% on the
A30, -0.62% on the RTX 2080 Ti and -0.31% on the GV100, so the off path shows
no regression within noise.

Mechanism, from a controlled constant-threads-per-block experiment on hal A30
(code `c7ff8e21e509733c17c90968f8d6656dce03466c`, experiment-only override of
the threads-per-block setting, otherwise identical harness and flags): the
gain is the per-chunk dispatch itself. The batch-wide transform derives its
grid from the particle count while assigning one block per chunk, so with two
particles per chunk at most about 2 of up to 66 blocks do any work; a
per-chunk launch spreads each chunk across up to 33 blocks. At constant 128
threads per block, transform GPU time falls from 281.8 ms to 73.6 ms, and the
paired decomposition of the same workload (12 interleaved accepted reps,
storage C=4224 P=8448 T=128 in every run, physics md5
`9cbb15ab113a85ff23f1fa406dc0eda8` in all 84 accepted runs) attributes
-91.0 ms (-5.19%, bootstrap 95% CI [-5.85, -4.93]) to the dispatch alone and
the shipped 512-thread default to only +12.0 ms (+0.73%, CI [0.00, +1.30],
512 faster in 3 of 12 paired reps). Active-CTA-per-SM counters appear
identical across variants, which is why that metric must not be used to judge
this change. The per-chunk path multiplies the sink and partition launches,
costing +19.3 ms and +14.4 ms on this card and workload, which returns 16-22%
of the transform saving. The 512-thread per-chunk default was not tuned and
128 threads per block measured about 0.7% faster, with the confidence
interval touching zero, so the default was left unchanged.
The whole-process wall time includes a 0.31-0.34 s start-up floor on fwk394,
so the steady-phase improvement is larger than the wall percentage. The
magnitudes are card-specific and must not be extrapolated, and the queue depth
was not tuned. No H100 was used.

## Chunk occupancy target, phase 1 (main `54ba63f`)

Provenance: the opt-in chunk occupancy target merged in
`54ba63f8d3696993bc65670b7c1d9f8b8d5d3016` (PR #87). One exclusive
4x V100-SXM2-32GB node; the photon transform harness from the per-chunk
transform schedule section at 4M photons, `C=4224`, `P=8448`, `Q=8`, seed 7,
no output directory. Eight accepted repetitions per arm; arms interleaved and
the ordering alternated.

| D | delta vs baseline | minimum detectable effect |
| ---: | ---: | ---: |
| 5 | +0.43% | 1.05% |
| 8 | +0.06% | 2.14% |
| 16 | +0.45% | 0.99% |
| 32 | +0.15% | 1.61% |

Baseline median `1.5190` s at `2,633,398` photons/s, coefficient of variation
0.62%. Every confidence interval includes zero. The target is throughput-neutral
within that resolution; a sub-1% effect is not excluded.
`--occupancy-target 4` is rejected before launch because the registered lane
count is 5 on that graph.

## Default-path throughput parity, phase 1

Provenance: the phase 1 change set versus the previous `main` on the same
exclusive 4x V100-SXM2-32GB node.

| Harness | delta | 95% CI |
| --- | ---: | --- |
| frozen p07 gate, Q=8 | +0.23% | [-29,207, +89,183] particles/s |
| photon harness | -0.19% | [-10,296, +1,607] photons/s |

A raw ABAB ordering suggested +0.80% and the reversed order -0.87%, which is
why the ordering was alternated and an order term fitted.

## Frozen p07 gate on reference hardware

Provenance: the same exclusive 4x V100-SXM2-32GB node, `N=1e8`, `C=8192`,
`P=16384`, seed 42.

| Q | median particles/s |
| ---: | ---: |
| 8 | 13,210,040 |
| 2 | 11,641,444 |
| 1 | 520,800 (median) |

Q=8 passes the `13,000,000` target, 1.6% above nominal and 12.9% above the
`11,700,000` floor. Q=1 is about 25 times slower than Q=8. Earlier measurements
on other hardware (A30 `10.76e6`, RTX 2080 Ti `11.59e6` at `C=4096`, GV100
`10.12e6`) were hardware provenance rather than regressions; the absolute
target must not be used as a pass or fail on other cards.

## Slot arena, phase 2 (main `ca26235`)

Provenance: the slot arena merged in
`ca262351ce71c67aa7e46e3693889b9542418c47` (PR #94). Measured on `hal` with an
A30 for CUDA and the same host for CPU; 24 bytes per slot. The fixed-layout
bytes are identical for both backends; no CUDA-specific fixed value was
recorded. No throughput
improvement is claimed, because chunk width was measured throughput-neutral at
fixed depth: equal-depth controls gave +0.3% at 4M photons, +1.1% at 40M and
+1.2% on a GV100 when the chunk size changed from `8,192` to `100,000`.

| Configuration | Layout | CPU bytes | CUDA bytes |
| --- | --- | ---: | ---: |
| `C=4224`, `P=8448` | fixed | 857,235,456 | |
| | arena | 12,367,872 (69.3x) | 13,787,136 (62.2x) |
| `C=8192`, `P=16384`, `N=100000` | fixed | 3,222,798,336 | |
| | arena | 42,270,720 (76.2x) | 41,287,680 (78.1x) |

At a 64 MiB budget the fixed policy admits `P=640` at `C=4224` and `P=256` at
`C=8192`, while the arena runs 8,448 and 16,384. The physics output is
byte-identical between the two layouts on CPU and CUDA.

An earlier design figure of 817 MiB to 0.81 MB, about 1057x, is retracted as
unreachable. It used a credit factor of 6 where the transform reserve is 8, and
it assumed the arena need only cover the credit bound, whereas the live
output-range peak, measured between 129 and 214 depending on configuration,
dominates. Credit bounds reserved ranges rather than live ranges.

## Measurement method, chunk pipeline runs

`--output-dir` must be omitted for throughput runs: it writes a roughly 489 MiB
`exits.csv` and costs about 9 seconds of host CPU per run, which inflated a 1.5 s
run to about 10.6 s and made an earlier sweep under-powered. Arms must be
interleaved with alternating order: position bias alone produced a spurious
+0.8% effect that reversed under reversed ordering.
