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
