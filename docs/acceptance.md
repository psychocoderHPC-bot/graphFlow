# graphFlow acceptance record

Environment: hal container, NVIDIA A30 (sm_80, 24 GiB), CUDA 13.4, g++ 13.3,
CMake 3.28.3, Ninja 1.11.1, alpaka `eb677e8d7f70d79b35df25d9002f94ba669b6f44`
(alpaka 3.0.0), C++20, Release. `hal` reports `sharing=unknown`; the throughput
numbers are a recorded baseline, not an exclusive-hardware acceptance claim.

## Builds

| Configuration | Result |
| --- | --- |
| CPU Release | configure and build pass |
| CUDA Release (`CMAKE_CUDA_ARCHITECTURES=80`) | configure and build pass |
| HIP Release (`gfx90a`, `amdclang++`) | compile-only gate passes |
| ASAN+UBSAN Debug | build and tests pass |
| TSAN Debug | blocked by the hal kernel: `FATAL: ThreadSanitizer: unexpected memory mapping`; not a project defect |

## Tests

| Configuration | Result |
| --- | --- |
| CPU CTest | 42/42 passed, 39.86 s |
| CUDA CTest | 42/42 passed, 39.00 s |
| ASAN+UBSAN CTest | 42/42 passed, 75.62 s; no Address/LSanitizer/UBSan report |
| HIP | compiled; no AMD device was visible, so no execution is claimed |

## Particle feedback

Common settings: `--seed 42 --chunk-size 8192 --live-particles 65536 --max-visits 64`.

| Backend | N | Wall time | Histogram sum | Peak live chunks | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 1,000,000 | 19.45 s | 1,000,000 | not recorded | maxrss 76.8 MB |
| CUDA | 100,000 | 0.58 s | 100,000 | 514 / 65542 | |
| CUDA | 1,000,000 | 2.24 s | 1,000,000 | 857 / 65542 | |
| CUDA | 100,000,000 | 187.5 s | 100,000,000 | 877 / 65542 | 264 histogram rows; about 5.33e5 particles/s |

The queue probe on CUDA reports `overlap=no queues-alias=no
second-before-first=no`; the queues are distinct, but no overlap was observed.
Correctness and the measured speedup come from full-chunk pipelining, not from
cross-queue overlap.

## Before and after source admission

The first implementation admitted source chunks whenever any particle budget
was free. Under `N >> P` this produced one-particle chunks: the pool filled to
8195 of 8198 chunks, a 1e6 CUDA run took about 119 s, and the profile showed
about 7 kernel launches and 12 copies per particle. The fix requires a source
chunk to carry at least `min(C, P, remaining)` particles, so one full
classify/plan/scatter round trip is amortised. The same 1e6 run then took 2.24 s
with a peak of 857 live chunks. Consumer ready lists still dispatch partial
chunks immediately; the minimum applies only to source admission.

## Memory note

With `C=8192` and `P=65536`, the plan's safe bound `S = P + K*R` gives 65542
chunks and about 12.9 GB of payload storage. Observed peak live is under 1000
chunks. A slot-based pool or a smaller configured `P` is the next memory
optimisation; it does not affect the measured throughput at these settings.
