# P07 measurement matrix, copy reference, and frozen target

Provenance: graphFlow main `64d4a1e11949886c7147c082cc386edde4bdb01f`
(P01 in-flight batching and P03 scatter/dense-threshold merged). Host `gv001`,
one exclusive `gpu-v100-int` allocation, Tesla V100-SXM2-32GB, driver
570.158.01, CUDA 12.6.85, `sm_70`, Release. CPU: Xeon Gold 6136, 48 threads.
Binary SHA-256:
`450c4093f7238e88b5a6a3f920db9e10663e65cf0833f180ab8ba587c6e85bea`.

Every case ran three independent repetitions. The tables report the median and
the interquartile range (inclusive halves, `Q3 - Q1`). Wall time includes
process start, CUDA initialisation, and the startup queue probe. No H100 was
used.

Build:

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHFLOW_ALPAKA_SOURCE_DIR=<alpaka-eb677e8> \
  -DGRAPHFLOW_ALPAKA_PIN=eb677e8d7f70d79b35df25d9002f94ba669b6f44 \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build-cuda -j 24
```

## Q sweep and frozen target

Command:

```sh
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-cuda/graphFlow_particle_feedback --out /tmp/gf-p07-run-q \
    --matrix p07 --reps 3 --device 0 \
    --kernel-tests build-cuda/graphFlow_tests --copy-reference \
    --target-pps 13000000 --target-n 100000000 --target-c 8192 \
    --target-q 8 --target-tolerance 10
```

`N=100000000`, `C=8192`, `P=16384`, seed 42, `T=128`, pool 16392 chunks.

| Q | median wall s | IQR wall s | median particles/s | IQR particles/s | peak live | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 191.990 | 2.315 | 5.209e5 | 6.18e3 | 236–237 | 6 |
| 2 | 8.710 | 0.535 | 1.148e7 | 6.59e5 | 24–26 | 6–9 |
| 8 | 7.690 | 0.045 | 1.300e7 | 7.56e4 | 12–13 | 7 |

The Q=8 median is `13,003,901` particles/s. The frozen target is
`13,000,000` particles/s, rounded from the earlier exclusive `rosi_v100` Q=8
run at `f3ef355` (7.676 s, 1.303e7 particles/s), with a 10 percent tolerance.
The floor is `11,700,000` particles/s; this run passes. The driver enforces the
gate and writes `target.md`; no CI job invokes it yet.

## C-wave sweep

Command:

```sh
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-cuda/graphFlow_particle_feedback --out /tmp/gf-p07-run-c \
    --backend cuda --n 100000000 --chunk-sizes 2048,4096,8192 \
    --live-particles 8192 --batch-chunks 8 --seeds 42 --reps 3 --timeout 1800
```

`P=8192`, Q=8, so `P/C` is 4, 2, and 1 resident full chunks. The `waves=1`
field in the policy header is the launch-wave field and is unrelated.

| C | P/C | median wall s | IQR wall s | median particles/s | IQR particles/s | peak live |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 4 | 11.010 | 0.045 | 9.083e6 | 3.70e4 | 15 |
| 4096 | 2 | 11.330 | 0.025 | 8.826e6 | 1.95e4 | 13 |
| 8192 | 1 | 22.980 | 0.090 | 4.352e6 | 1.72e4 | 11–12 |

One resident chunk stalls the pipeline: 22.98 s against 11.01 s at four
chunks. Two and four chunks differ by about 3 percent. The target configuration
above uses `P=16384` (two resident chunks of `C=8192`) and reaches 7.69 s, so
the live-particle capacity matters as well as the wave count.

## Exact histogram

Every one of the 18 runs printed 260 sorted rows and a count column summing to
exactly `100,000,000`. The sorted `weight,count` rows are identical across Q,
C, P, and the repetitions. All groups hash to SHA-256
`21ab0aeb9ec72d380af2ce6e9631f8fa7b9111d24ff7831f272aa9f6a7be9dab`.

| Group | runs | rows_equal | SHA-256 |
| --- | ---: | --- | --- |
| Q run, `C=8192`, `P=16384` | 9 | yes | `21ab0aeb…be9dab` |
| C=2048, `P=8192` | 3 | yes | `21ab0aeb…be9dab` |
| C=4096, `P=8192` | 3 | yes | `21ab0aeb…be9dab` |
| C=8192, `P=8192` | 3 | yes | `21ab0aeb…be9dab` |

## Copy-bandwidth reference

The driver generated and compiled a device-to-device `cudaMemcpy` reference at
the largest selected payload (`P*C*24 = 3,221,225,472 B`), then copied it 10
times on device 0:

```text
copy_bandwidth_gbytes_per_s=409.733 bytes=3221225472 iters=10 ms_per_copy=7.862 device=0
```

This is a sustained D2D copy-engine reference for the same device and payload
size, not a scatter-bandwidth claim. The example does not report classify,
scatter, or host round-trip timings at this revision, so no scatter bandwidth
is derived from it.

## Kernel matrix: scatter, dense/hash, distributions

The example CLI exposes only N, C, P, Q, and seed. Scatter mode, dense versus
hash, and the balanced/one-hot/skewed/sparse distributions are selectable only
through the committed integration tests, so end-to-end throughput for those
axes is blocked at this revision. The driver ran the test cases and recorded
their durations:

```sh
build-cuda/graphFlow_tests --durations yes --order decl '[scatter]'
build-cuda/graphFlow_tests --durations yes --order decl '[hash]'
```

| Axis | Test case | Duration s |
| --- | --- | ---: |
| Scatter default | scatter mode defaults to shared staging | 0.000 |
| Scatter, all distributions | direct and shared-staging scatter agree on all distributions | 1.360 |
| Dense/hash overlap | hash and dense pipelines agree at overlapping bin counts | 1.042 |
| 2*T vs 4*T | 2*T and 4*T hash tables agree with dense at overlapping bins | 1.447 |
| Sparse bins | hash pipeline handles sparse 4096 and 65536 bin distributions | 0.729 |

Both commands exited 0. The durations cover the whole Catch2 case, including
device setup, not an isolated kernel. The distribution cases are balanced,
one-hot, skewed, and sparse (1024 bins at about two particles per bin in the
scatter comparison; 4096 and 65536 bins in the sparse case). No P07
kernel-level throughput number is claimed.

The earlier P03 measurement on fwk394 (Xeon W-2175, CPU, `C=1024`, `T=128`,
`N=1<<20`, 14 OpenMP threads) is the only committed kernel-timing provenance:
shared staging versus direct was 5.48 vs 15.85 ms (balanced), 2.13 vs
48.31 ms (one-hot), 2.21 vs 37.25 ms (skewed), and 4.19 vs 6.62 ms (sparse,
B=1024). At B=1024 the dense table took 4.06 ms against 5.11 ms (2*T) and
4.90 ms (4*T). Those numbers were not re-measured here.

Artifacts from the run: `/tmp/gf-p07-run-q/` and `/tmp/gf-p07-run-c/`
(`results.csv`, `aggregates.csv`, `summary.md`, `copy-reference.txt`,
`kernel-matrix.md`, `target.md`, `histogram-stability.tsv`, per-run logs). These
are outside the repository.

---

# Integrated-main benchmark evidence

Provenance: integrated main `68124ad3875e42cbe09247f64bd218a7ac8d58fc`
(T5 batching + T7 budget), run with `tools/bench_matrix.sh` from T13. The
numbers are the recorded T13 evidence, not new measurements. Wall time includes
device initialisation and the queue probe. `hal` reports `sharing=unknown`;
fwk394 is shared. These are single-run baselines, not exclusive-hardware
acceptance claims.

## hal A30, CUDA sm_80

C=8192, seed=42. The measured policy budget is 3,171,090,432 B. The 24,192 MiB
usable figure is rounded; `nvidia-smi` reports 24,576 MiB because ECC reserves
memory. P=16128 uses `8192 * 16128 * 24 B = 3,170,893,824 B`, leaving 196,608 B
of exact headroom. P=16384 (3,221,225,472 B) and P=65536 (12,884,901,888 B)
are rejected by T7.

| N | C | P | Q | wall s | particles/s | hist sum | hist rows | peak live |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100000000 | 8192 | 16128 | 1 | 218.87 | 4.57e5 | 100000000 | 264 | 218 |
| 100000000 | 8192 | 16128 | 2 | 9.83 | 1.02e7 | 100000000 | 264 | 23 |
| 100000000 | 8192 | 16128 | 8 | 8.50 | 1.18e7 | 100000000 | 264 | 14 |
| 1000000 | 8192 | 16128 | 1 | 2.32 | 4.31e5 | 1000000 | 150 | 214 |
| 1000000 | 8192 | 16128 | 2 | 0.41 | 2.44e6 | 1000000 | 150 | 19 |
| 1000000 | 8192 | 16128 | 8 | 0.35 | 2.86e6 | 1000000 | 150 | 15 |

The sorted histogram rows are identical across Q within each N. SHA-256:
`d06a3bf93b00345b148fb848019921288d625d0198ee6b3f2793d25c02c40de0`
for the 1e8 rows and
`14dfc8f1137ca9b86f0026c0d19e1f615bca632a877caa8b20d39bb586b2d85e`
for the 1e6 rows. Q=2/Q=8 are 22.3x/25.7x faster than Q=1 at 1e8 and
5.7x/6.6x faster at 1e6.

## CPU/CUDA equality

| N | C | P | Q | CPU wall s | CUDA wall s | hist sum | rows_equal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100000 | 128 | 1024 | 1 | 15.23 | 1.96 | 100000 | yes |
| 1000000 | 128 | 1024 | 1 | 149.44 | 17.66 | 1000000 | yes |

## fwk394, CPU

The CUDA sm_75 build fails with the nvcc 12.5 ICE at
`HashScatterKernel.hpp:153`; no CUDA run is claimed. The CPU host budget is
16 MiB, so C=4096/P=8192 and C=4096/P=16384 are rejected.

| N | C | P | Q | wall s | particles/s | hist sum | hist rows | peak live |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000000 | 1024 | 512 | 1 | 52.75 | 1.90e4 | 1000000 | 150 | 46 |

An N=1e8 CPU acceptance was not run; a linear projection from 52.75 s per 1e6
is about 88 minutes.

## Budget rejections

| Host | Pair | Payload B | Budget B | Result |
| --- | --- | ---: | ---: | --- |
| hal A30 | C=8192, P=16384 | 3,221,225,472 | 3,171,090,432 | rejected |
| hal A30 | C=8192, P=65536 | 12,884,901,888 | 3,171,090,432 | rejected |
| hal A30 | C=8192, P=16128 | 3,170,893,824 | 3,171,090,432 | passes |
| fwk394 CPU | C=4096, P=8192 | 805,306,368 | 16,777,216 | rejected |
| fwk394 CPU | C=4096, P=16384 | 1,610,612,736 | 16,777,216 | rejected |
| fwk394 CPU | C=1024, P=512 | 12,582,912 | 16,777,216 | passes |

Rejected cases exit 2 with
`graphFlow::DevicePolicy: liveParticles and chunkCapacity exceed the payload memory budget`.
The driver records them as `fail`, not as passes.

## Harness state

The integrated run exposed two T13 harness defects: backend names were sent
through the numeric list validator, and a tab-delimited empty `failure` field
collapsed and shifted successful CSV rows. Both are fixed in this tree and
covered by `tools/bench_matrix.sh --self-test`.
