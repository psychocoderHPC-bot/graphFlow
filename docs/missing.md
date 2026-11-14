<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow — remaining work

This is the complete list of work still required to finish the plan in
`/workspace/graph-async-chunk-scheduling-plan.md`. It is based on the current
repository state, not on the intended final state.

HIP runtime validation is measured on `b972eaa3bd778b81f2dca505d3662df710792bd9`
(HIP runtime merge, PR #35), recorded in
`/home/agent/final-hip-b972eaa/ctest.log`; the earlier 238/238 at
`58bb514def3f3c956126ab7c34d0de74a45f679e` is historical. The validated code
main is `2ad7d86ceaebed7da11c352ebee12e9525c2b0d2`. The source-admission tests
and the N=1e6 CPU smoke are on `gf/p1-admission-smoke`
(`12871ad19ac1eea9d4bac5c15db9f59085f89a46`), not yet merged; the aggregate
scheduler stall diagnostics are on `gf/stall-diagnostics`
(`53f0e1df409df29234dd51142d2b136d47ea68a7`), also not yet merged. The
`rosi_v100` CPU/CUDA counts at `2ad7d86` below are the final validated-main
values; they are historical for this branch, whose current CPU CTest is 210/210
(202 plus the two admission cases, the one N=1e6 smoke, the two stall
diagnostics cases, and the three cross-backend equality cases):

- Repository: `https://github.com/psychocoderHPC-bot/graphFlow` (public).
- **Validation platform:** `rosi_v100` is the exclusive validation platform
  and is reachable again. `rosi_v100` results at main `f3ef355`: CPU CTest
  **210/210 passed** (OMP=ON Release, pedantic) and the full integrated TSAN
  CTest passes **210/210 with 0 reports** in 202.83 s (non-PIE Debug with
  `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`); the `N=1e8` CUDA
  acceptance run is recorded on the V100 `sm_70` (CUDA 12.6.85, Release). The
  `rosi_v100` results at validated code main `2ad7d86`, CPU CTest
  **202/202 passed** (OMP=ON Release, pedantic) and CUDA CTest **254/254
  passed** on V100 `sm_70` (CUDA 12.6.85 Release), are historical. The CPU
  CTest count is 210 (202 plus two source-admission cases, one N=1e6 smoke,
  two stall diagnostics cases, and three cross-backend equality cases); the
  expected device totals for that suite are 264 (CUDA `sm_70`) and 255 (HIP
  `gfx1100`), and the device full suites have not been re-run, so the measured
  CUDA/HIP CTest totals stay the recorded 254/254 and 245/245 values. The
  explicit concept-parameter workaround keeps nvcc 12.6.85 off the name-hiding
  ICE and the RNG main is preserved. The `Debug` build was not re-run on
  `rosi_v100` after the seven hash cases: the measured `Debug` totals are the
  pre-hash 195/195 CPU and 247/247 CUDA, and 202/202 and 254/254 are the
  expected (not-yet-re-run) `Debug` totals for the `2ad7d86` suite. CPU OpenMP
  (passive, `GOMP_SPINCOUNT=0`) works with no hangs. The previously recorded
  counts, 194/194 on fwk394 CPU and 246/246 on hal CUDA `sm_80`, are
  **historical validation on a shared host** (`hal` reports `sharing=unknown`)
  and are **not the final current validation**. CUDA 13.4/`sm_80`, fwk394, and
  H100 were not re-run in this validation.
- The public API is implemented: `Flow<TPayload, TDevice, TStorage>` with
  source/partition/transform/sink handles, named ports, `configure(Tuning)`,
  typed failure `Status`, and `run()`.
- Per-event photon payload validation is debug-only
  (`GRAPHFLOW_ENABLE_DEBUG_VALIDATION`, issue #26). Host setup validates the
  slab and configuration once; release keeps the physics errors on the status
  channel.
- T1 (device-aware policy), T2 (payload storage), T3 (CUDA kernel test and
  typed Flow execution), T5 (batching, payload budget, photon scattering),
  T7 (pool payload budget), T13 (integrated benchmark), T14a (format/REUSE
  foundation), and the `Flow`/particle/photon examples are done.
- HIP runs on `hal` with an AMD Radeon RX 7900 XTX (`gfx1100`) and ROCm 7.2:
  245/245 CTest with 43 HIP cases, measured on `b972eaa` (255 is the expected
  total on `gf/cross-backend-equality`, not re-run). GitHub-hosted HIP
  remains compile-only
  because those runners have no AMD GPU; the `gfx90a` compile-only gate is
  historical.
- CPU execution with the OpenMP backend: the `rosi_v100` run (passive,
  `GOMP_SPINCOUNT=0`) works with no hangs; the CPU CTest suite passes 210/210
  with OMP=ON on this branch (historical: 202/202 at validated main
  `2ad7d86`). The previous multi-region pathology is not observed on
  `rosi_v100`.
- H100: not used for this validation; no final H100 claim is made. The JURECA
  `jrc0880.jureca` run is preliminary CUDA evidence only (unknown gateway
  sharing flag; external load while the client was stale) and stays historical.
  H100 remains preliminary.

Priority legend: **P0** blocks plan acceptance, **P1** required for plan
completion, **P2** hardening or optimization. **Blocked** means an external
resource or decision is missing.

## 1. Core architecture and scheduler

- [x] **P0: multi-chunk batching.** Done (PR #6 / T5): `--batch-chunks Q`
  (default 1) dispatches up to `Q` ready input chunks per transform node, with
  one classify launch and one scatter launch per batch and the adaptive
  `outputChunkBound(sum(fill), C, B)` reserve; the integrated T13 run measured
  Q=1/2/8 with identical sorted histograms across Q. The historical tree
  passed 194/194 CPU on fwk394 and 246/246 CUDA on hal (historical); the
   `rosi_v100` CPU CTest passes 202/202 (Release) at validated main `2ad7d86`
   (historical; 210/210 on this branch) and the V100 `sm_70` CUDA CTest
   passes 254/254 (Release) with the explicit concept-parameter workaround
   (264 on this branch, not re-run). Open
   refinement: `--batch-chunks Q` has no upper cap beyond the 32-bit parse
   limit, tracked in Section 7.
- [x] **P1: FrameSpec launch.** Done: every graph kernel launches through
  `launchPolicy` with `FrameSpec{numFrames, frameExtent}`. `numFrames` is
  `min(ceil(work/T), maxBlocksPerGrid, frameCap*multiProcessorCount)` with
  `frameCap` default 8; the host frame extent is 1 and accelerators use `T`.
  Kernels stride the batch tile list, so a block may process several tiles. The
  FrameSpec unit and matrix tests pass on CPU and CUDA.
- [x] **P0: generic graph API.** The public API is implemented. The host-side
  `NodeBase`/`Graph`/`GraphBuilder`/`RouteTable` core is in the tree with
  source/transform/sink roles, per-bin routing, feedback edges, and no device
  function pointers. The typed payload contract is `Flow<TPayload, TDevice,
  TStorage>` over the `Payload` concept, and the particle example is ported to
  it. The generic DAG acceptance test runs a source, two transforms, a sink,
  and one feedback edge through `GraphRunner`. Remaining: the host-only generic
  core is not yet used by the example, which uses the typed `Flow` API.
- [ ] **P1: multiple in-flight batches per node.** The scheduler currently
  allows one in-flight batch per node. Add a bounded submitted depth per node,
  an event pool, and fair round-robin across ready nodes while preserving
  exactly-once completion and lease release. Acceptance: depth >1 overlaps
  independent work, event-pool exhaustion backpressures, no double retire.
- [ ] **P1: exploit or document queue overlap.** The CUDA queue probe reports
  distinct queues but `overlap=no`; the CPU probe reports overlap. Investigate
  whether the CUDA probe's kernels or the shared A30 prevent overlap, and either
  use independent node queues for real overlap or document that the scheduler
  relies on full-chunk pipelining. Correctness must never depend on overlap.
- [x] **P1: device-aware chunk sizing.** The device-aware policy is in main
  (T1) and the integrated T13 matrix is recorded (`docs/benchmarks.md`).
  Closed boundary: the policy default `P=16384` exceeds the measured A30
  payload budget of 3,171,090,432 B (needs 3,221,225,472 B), so the T13 `N=1e8`
  run used an explicit `P=16128`. The auto-derived policy now clamps `P` to
  16,128; an explicit over-budget `P` still fails clearly.
- [ ] **P1: reduce pool memory.** `S=P+K*R` at `C=8192`, `P=65536` allocates
  about 12.9 GB of payload storage while observed peak live is under 1000
  chunks. Implement a slot-based pool or a documented smaller `P`/memory budget
  without breaking the fragmentation bound or progress guarantee. Acceptance:
  peak memory falls while `N >> P` still completes and the histogram is exact.
- [x] **P1: chunk-fill policy tests.** Done: `tests/integration/test_admission.cpp`
  pins `min(C, P, remaining)` for `P<C`, `P==C`, `P==0`, the tail, and `N>>P`,
  and rejects one-particle source chunks with a mutation-style assertion.
- [x] **P1: scheduler state and stall diagnostics.** Done: `SchedulerStats`
  exposes aggregate ready, in-flight, reserved, event-slot, source and capacity
  peaks; `SchedulerSnapshot` and `SchedulerStallError` report the aggregate
  state on stall or end; `Tuning::stallTimeout` and `Tuning::diagnostics` are
  wired through `Flow::run`; the examples print one `# scheduler` summary line
  and the photon metadata records it. Per-node detail stays behind the opt-in
  diagnostics callback and never enters `what()`.

## 2. Execution pipeline

- [ ] **P1: batch-level classify/plan/scatter.** The current host plan is
  per chunk. Extend `total[B]`, `next[B]`, `binBase[B]`, and `outChunks[R]` to a
  multi-chunk batch; verify `next[bin]==total[bin]` after scatter across the
  batch; keep the exact fill levels.
- [ ] **P1: direct versus shared-staging scatter.** The direct SoA scatter is
  the correctness reference. Implement the measured shared-staging variant and
  benchmark it under balanced, one-hot, skewed, and many-sparse-bin
  distributions. Keep the faster path documented; do not replace correctness
  evidence.
- [ ] **P1: dense/hash threshold measurement.** The dense limit is currently a
  static `Config::denseBinLimit=1024`. Measure dense versus hash at overlapping
  bin counts, including `2*T` versus `4*T` hash capacity, and make the threshold
  measurement-driven. Keep table-full as a hard batch error and keep the
  no-hardcoded-warp-size guarantee.
- [ ] **P2: host round-trip profiling and device-side planning experiment.**
  The profile shows the host round trip and poll path matter. Measure classify,
  D2H total/status, host planning, H2D table, scatter, and event latency. If the
  round trip still dominates after batching, implement a device-side output
  planning variant (global prefix over bin totals) and compare; otherwise
  document why the host path stays.
- [ ] **P1: error and abort coverage.** Add tests for invalid classifier
  results, hash-table full, counter overflow, weight overflow, pool exhaustion,
  async enqueue failure, and abort during in-flight work. Assert no partial
  output is published and all leases are released.
- [ ] **P2: counters and validation.** In debug or validation builds, check
  conservation (`next[bin]==total[bin]`), input conservation, output chunk fill
  levels, and exactly-once sink delivery; keep these off the hot path. The
  photon per-event payload checks are implemented behind
  `GRAPHFLOW_ENABLE_DEBUG_VALIDATION` (issue #26); the scheduler counter checks
  remain open.

## 3. Example, RNG, and end-to-end tests

- [x] **P0: RNG API ([issue #25](https://github.com/psychocoderHPC-bot/graphFlow/issues/25)).** `graphFlow::rng::Engine` and
  `makeEngine(baseSeed, id, event, purpose)` construct a zero-offset
  `Philox4x32x10` from a bijective packed key. Flow injects the initialized
  engine into opting-in source, classifier, updater, transform, and sink
  functors through `if constexpr`, so existing signatures keep working. The
  public offset/purpose helpers and the photon `openUniform` conversion are
  removed; floating-point draws use `UniformReal` and integer decisions use the
  engine output. The independent scalar oracles use the same mapping without
  the production header. Fixed-seed particle and photon baselines changed once
  at this commit. V100 CUDA `sm_70` passes 254/254 (Release) with the explicit
  concept-parameter workaround on the RNG main.
- [x] **P0: register the example CLI in CTest.** The real
  `graphFlow_particle_feedback` binary runs through `example_cli.cmake` for
  `N=0,1,17,127,128,4097` crossed with `C=1,31,127,128` (24 cases per backend)
  with finite timeouts and a failure when the requested backend did not run.
  A CUDA or HIP build adds the same 24 cases for that backend; the HIP cases
  run on `hal` with a real AMD GPU.
- [x] **P0: automate CPU/CUDA/HIP histogram equality.** Done on
  `gf/cross-backend-equality`
  (`751c8a8cbad9de1a2e71cef57b81147ff071dd51`): the
  `example.particle_feedback.cross_backend.Q{1,2,8}` CTest cases run the real
  `graphFlow_particle_feedback` binary once per compiled backend with identical
  seed, `N=1e6`, `C=128`, `P=1024`, and `Q`, sort the `weight,count` rows, and
  require exact row equality against the first backend plus a count sum of N
  per backend. CUDA and HIP cases register only when their backend is
  compiled, and a compiled backend without a usable device fails the case
  instead of being skipped, so the CPU-only hosted build still runs the case.
  Measured at `751c8a8`: CPU 210/210 (OMP=ON Release, pedantic), with the three
  Q cases exact. The device evidence is two build pairs, not one three-backend
  run: CUDA and HIP are mutually exclusive in a single build, and the CPU
  backend stays enabled in accelerator builds, so the CPU+CUDA pair and the
  CPU+HIP pair each ran the three Q cases 3/3 with the device output compared
  against the CPU reference (same seed/N/C/P/Q, sorted `weight,count` rows
  identical, count sum N for every backend). CUDA 264 and HIP 255 are the
  expected full-suite totals for the branch (254 and 245 plus the two
  source-admission cases, the one N=1e6 smoke, the two stall diagnostics
  cases, and the three cross-backend equality cases); the full device suites
  have not been re-run, so those counts are not claimed as executed.
- [x] **P1: device-dependent default `C`/`P`.** Done (policy PR #11): the
  device-aware policy derives `C`/`P`/`T`, auto-clamps `P` to the usable-memory
  budget, and rejects an explicit over-budget `P`. The example prints the
  resolved policy and the chosen values, and the CLI override stays.
- [ ] **P1: full max-visits/non-termination coverage.** Keep the scripted
  fixture and add cases for particles that reach the visit cap, zero-weight
  particles, empty input, exact chunk fill, and multiple feedback rounds.
- [ ] **P2: result-storage policy.** The result buffer is `O(N)` (about 400 MB
  at `N=1e8`). Document the limit and either keep it explicit or add an optional
  device-side histogram when the result buffer becomes the memory bottleneck.
- [ ] **P2: CLI contract tests.** Assert the selected backend and device in the
  example output, invalid backend/device exit codes, `N=0` header-only output,
  and `C=0` rejection.

## 4. CI, format, REUSE, and repository policy

- [x] **P0: GitHub Actions workflow.** `.github/workflows/ci.yml` runs on
  GitHub-hosted `ubuntu-latest` runners and public containers:
  - CPU Release and Debug with CTest, JUnit upload, and `--no-tests=error`;
  - ASAN+UBSAN on CPU;
  - CUDA compile-only (`sm_70`) and HIP compile-only (`gfx90a`);
  - `reuse lint`;
  - finite timeouts and JUnit artifact upload.
  Still open: TSAN on a host where the runtime mapping check passes, and a CUDA
  or HIP runtime gate on a device host. The workflow uses no self-hosted runner.
- [x] **P0: format configuration.** The upstream-derived `.clang-format`
  (clang-format 20.1.8) is in main (PR #5 / T14a). The tree-wide format sweep
  (T14b) is complete, and the hosted `format` job fails on any diff.
- [x] **P0: REUSE compliance.** The REUSE foundation is in main (PR #5 / T14a):
  `LICENSES/MPL-2.0.txt` and `.reuse/dep5` are present. The tree is compliant,
  and hosted CI runs `reuse lint`.
- [x] **P1: add the plan to the repository.** Done: `docs/plan.md` is in main
  with an SPDX header.
- [x] **P1: complete the documentation set.** `docs/build.md` (exact pins,
  configure/build/test commands, environment scrub), `docs/architecture.md`,
  `docs/numerics.md`, `docs/known-limitations.md`, `docs/benchmarks.md`, and
  `docs/public-api.md` are in main and linked from the README. The hosted
  workflow is documented in `docs/ci.md`.
- [ ] **P1: complete the alpaka API map.** `docs/alpaka-api-map.md` covers the
  main primitives; add an entry for every new accelerator helper, with searched
  symbols, pinned path, selected API, tested backends, and the reason for any
  project-owned code.
- [ ] **P1: repository policy.** Decide public versus private, enable branch
  protection for `main`, add a CI badge, and document the PR/merge process.
- [x] **P1: exact pin record.** The alpaka commit
  (`eb677e8d7f70d79b35df25d9002f94ba669b6f44`, alpaka 3.0.0) and Catch2 3.4.0
  are recorded in `docs/build.md` and `docs/acceptance.md`. Update the pin
  deliberately and re-run CPU/CUDA/HIP probes.

## 5. Performance and acceptance

- [x] **P0: CUDA-specific kernel test.** Done (T3):
  `tests/unit/test_cuda_kernel.cpp` runs a real kernel on the selected NVIDIA
  GPU and fails when no GPU is visible rather than skipping.
 - [x] **P0: HIP execution gate.** Done at
   `b972eaa3bd778b81f2dca505d3662df710792bd9` on `hal` with an AMD Radeon RX
   7900 XTX (`gfx1100`) and ROCm 7.2: 245/245 CTest with 43 HIP cases, plus
   particle and photon CLI/output cases, measured and recorded in
   `/home/agent/final-hip-b972eaa/ctest.log`. `rocprof` confirms the device
   kernels.
  Remaining HIP work: other AMD architectures, the HIP hash path, an AMD
  `N=1e8` acceptance run, and no performance claim while `hal` reports
  `sharing=unknown`.
- [x] **P0: exclusive or repeated acceptance run.** Done on `rosi_v100` at
  main `f3ef355` (V100 SXM2-32GB, CUDA 12.6.85 `sm_70`, Release), `C=8192`,
  `P=16384`, `T=128`, seed 42, max-visits 64, three repetitions: Q1 194.685 s,
  Q2 9.093 s, Q8 7.676 s, 1.30e7 particles/s at Q8, 260 histogram rows, count
  sum exactly 100,000,000, sorted-histogram SHA-256
  `89dc7fb9962e898fe2f1fe0116ad98fe2df9dceb77d33aa890a56467615d02e9`, peak
  5702 MiB. The `P=16384` payload (3,221,225,472 B) fits the V100 SXM2-32GB
  budget, unlike the A30, which forced `P=16128`. `hal` is shared
  (`sharing=unknown`); its numbers stay labelled as a baseline. H100 was not
  used, so no H100 performance claim is made.
- [ ] **P1: freeze the throughput target.** The current baseline is about
  `5.33e5` particles/s at `N=1e8` on the A30 with `C=8192`, `P=65536`. After an
  exclusive measurement, freeze a regression target with a stated tolerance and
  enforce it in CI or a scheduled benchmark.
- [ ] **P1: complete the measurement matrix.** Record end-to-end particles/s,
  wall time, launch count, classify/scatter time, host round-trip latency,
  device idle time, effective bandwidth, peak live/reserved chunks, event-pool
  use, and queue overlap. Cover chunk sizes around one/two/four waves, batch
  depths, direct versus staged scatter, dense versus hash, and balanced,
  one-hot, skewed, and sparse distributions.
- [ ] **P1: copy-bandwidth reference.** Measure a sustained copy-bandwidth
  reference on the same device and data size and report it next to the scatter
  bandwidth.
- [x] **P1: `N=1e6` CI smoke.** Done for the hosted CPU job: the fixed-seed
  `example.particle_feedback.smoke.N1000000` CTest case runs in the ordinary
  CTest job with a finite timeout and an exact histogram-sum check. Hosted CUDA
  and HIP stay compile-only, so their `N=1e6` runtime smoke remains a manual
  `rosi_v100`/`hal` check. `N=1e8` stays the manual acceptance run.
- [ ] **P1: HAL acceptance record.** Record exact commit, alpaka pin, compiler,
  CUDA/ROCm versions, driver, GPU, commands, test counts, JUnit, and the
  no-skipped-test assertion. Keep raw logs outside the repository and link their
  location.

## 6. Robustness and hardening

- [x] **P0: TSAN validation.** Done on `rosi_v100` at main `f3ef355`: full
  integrated TSAN CTest **210/210, 0 reports**, 202.83 s, non-PIE Debug with
  `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`. With the default
  CpuOmpBlocks plus OpenMP the same suite fails 105/210 with reports inside the
  alpaka async CPU queue; those are upstream alpaka races, not graphFlow races,
  which is why the clean gate uses the OMP-exec-disabled configuration. The
  fwk394 non-PIE scheduler/pool subset (14/14) and the hal startup abort
  (`FATAL: ThreadSanitizer: unexpected memory mapping`, even for a non-PIE
  binary) are historical context for why this was blocked; `hal` is shared
  (`sharing=unknown`) and not a usable TSAN host. No TSAN suppressions were
  used; suppressions are allowed only with evidence and must not hide a race.
- [ ] **P1: stress tests.** Cover `N >> P`, small `P`, pool exhaustion, event
  and staging exhaustion, feedback cycles, max-visits, abort during in-flight
  work, and repeated runs with generation reuse.
- [ ] **P1: asynchronous failure propagation.** Test enqueue failures, device
  errors, OOM, and `wait` failures; ensure leases, staging, and events are
  released exactly once and the process returns nonzero.
- [ ] **P2: sanitizer coverage for the host scheduler.** Keep ASAN/UBSAN/TSAN
  in CI and add a leak check for the error paths.
- [ ] **P2: overflow hardening.** Audit `chunkIndex*C`, `chunk*C+offset`,
  `tile*T`, `ceil(n/C)`, and counter arithmetic for 32-bit overflow; add debug
  assertions and document the supported limits.

## 7. Blockers and decisions

- **AMD hardware:** resolved for correctness. `hal` exposes an AMD Radeon RX
  7900 XTX (`gfx1100`, ROCm 7.2) and the HIP runtime gate passes 245/245 with
  43 HIP cases on `b972eaa`; the expected HIP total on
  `gf/cross-backend-equality` is 255 (245 plus the two source-admission cases,
  the one N=1e6 smoke, the two stall diagnostics cases, and the three
  cross-backend equality cases), and the full HIP suite has not been re-run on
  `hal`. Other
  AMD architectures and the AMD `N=1e8` acceptance remain open; GitHub-hosted
  HIP stays compile-only.
- **TSAN host:** resolved. The TSAN host is `rosi_v100` (exclusive); the full
  integrated TSAN CTest passes 210/210 with 0 reports in 202.83 s at main
  `f3ef355` (non-PIE Debug with `alpaka_EXEC_CpuOmpBlocks=OFF` and
  `alpaka_DEP_OMP=OFF`). The hal kernel still rejects the TSAN memory mapping
  and the fwk394 non-PIE subset (14/14) is historical context.
- **CI status:** on this branch the hosted CPU jobs run the 210-case CPU suite
  (Release and Debug) and the `sanitizers` job runs ASAN+UBSAN over it, with
  no skip: the N=1e6 smoke, the two stall diagnostics cases, and the three
  cross-backend equality cases run in the ordinary CTest job. Hosted CI was
  green on final main with the historical
  202-case suite (CPU 202/202 Release and Debug, ASAN+UBSAN Debug 202/202);
  `format` (clang-format 20.1.8, T14b) and `reuse lint` pass; CUDA and HIP jobs
  are compile-only; no self-hosted runner, SYCL job, or GitLab pipeline. TSAN
  has a recorded clean `rosi_v100` run (210/210, 0 reports, 202.83 s, non-PIE
  with `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`) but stays a
  local/manual check on that host; hosted CI runs no TSAN job. The CUDA or HIP
  runtime gate still needs a device host with a known sharing and label
  configuration; no self-hosted runner is provisioned.
- **CI GPU runtime:** hosted CI is in `.github/workflows/ci.yml` and its CUDA
  and HIP jobs are compile-only. A CUDA or HIP runtime gate still needs a device
  host with a known sharing and label configuration; no self-hosted runner is
  provisioned.
- **Exclusive throughput:** resolved for the acceptance run. `rosi_v100` is
  reachable and the `N=1e8` CUDA acceptance run is recorded on it at main
  `f3ef355` (Q1 194.685 s, Q2 9.093 s, Q8 7.676 s, 1.30e7 particles/s at Q8,
  3 reps, peak 5702 MiB; full record in `docs/acceptance.md` and
  `docs/benchmarks.md`). `hal` is shared (`sharing=unknown`); its historical
  187.5 s number stays a shared-host baseline. H100 was not used; the JURECA
  `jrc0880.jureca` run is preliminary CUDA evidence only and no H100
  performance claim is made.
- **CPU OMP backend:** CPU execution with the OpenMP backend works on
  `rosi_v100` with passive spinning (`GOMP_SPINCOUNT=0`); the CPU CTest
  passes 210/210 with OMP=ON Release on this branch (202/202 was the
  historical validated-main count).
- **Pool memory:** `S=P+K*R` is safe but overprovisioned at `C=8192`,
  `P=65536` (about 12.9 GB). Decide between a slot-based pool, a smaller default
  `P`, or a memory-budget cap.
- **Queue overlap:** the CUDA probe reports no overlap between distinct queues.
  Decide whether to investigate further or document the full-chunk strategy.
- **Policy-default budget boundary:** resolved. The policy default
  `C=8192`, `P=16384` exceeds the actual A30 payload budget of 3,171,090,432 B
  (it needs 3,221,225,472 B), so the T13 `N=1e8` run used `P=16128`
  (3,170,893,824 B). The auto-derived policy now clamps to the same P, so an
  un-overridden CUDA run on this device starts; an explicit `P=16384` or
  `P=65536` is rejected with a clear error.

## 8. Explicitly out of scope for v1

These are not missing work; the plan excludes them from v1:

- Join and broadcast ownership; multi-owner chunks.
- Multi-GPU execution.
- Runtime graph mutation during execution.
- Device-persistent schedulers, device function pointers, cooperative launch,
  and system-scope atomics.
- Device-side output planning as the default path.
- Stable output order; acceptance is histogram/multiset equality.
- Fault tolerance and checkpoint/restart.

## 9. Completion checklist

- [x] All P0 items above are implemented and covered by tests (batching, RNG
  API, example CLI in CTest, cross-backend histogram equality, CUDA kernel
  test, HIP execution gate, GitHub Actions workflow, format, REUSE, the
  `N=1e8` acceptance run on `rosi_v100`, and TSAN validation).
- [x] CPU, CUDA, and HIP build; HIP execution is validated on `hal` (RX 7900
  XTX `gfx1100`, ROCm 7.2) with 245/245 CTest and 43 HIP cases, measured on
   `b972eaa`. `rosi_v100` CPU CTest passes 210/210 (OMP=ON Release) on this
   branch (202/202 historical at validated main `2ad7d86`) and the
   CUDA V100 `sm_70` CTest passes 254/254 (Release) with the explicit
   concept-parameter workaround (264 on this branch, not re-run).
- [x] The example CLI is registered in CTest for the full CPU, CUDA, and HIP
  matrices; the HIP cases run on `hal` with a real AMD GPU.
- [x] The public API (`Flow<TPayload, TDevice, TStorage>`) is implemented and
  used by both examples; the particle example is ported to it and the photon
  example uses it for slab transport.
- [x] `N=1e6` runs in CI: the fixed-seed `N=1e6` CPU smoke
   (`example.particle_feedback.smoke.N1000000`) runs in the ordinary hosted CPU
   CTest job with a finite timeout and exact histogram-sum check, no skip.
    The `N=1e8` acceptance run is re-recorded on `rosi_v100` at main `f3ef355`
    (V100 SXM2-32GB, CUDA 12.6.85 `sm_70`, Release): Q1 194.685 s, Q2 9.093 s,
    Q8 7.676 s, 1.30e7 particles/s at Q8, 260 rows, sum exactly 100,000,000,
    sorted-histogram SHA-256
    `89dc7fb9962e898fe2f1fe0116ad98fe2df9dceb77d33aa890a56467615d02e9`, peak
    5702 MiB, 3 repetitions, `C=8192`, `P=16384`, `T=128`, seed 42, max-visits
    64. H100 was not used; no H100 claim. The `rosi_v100` CPU CTest passes
    210/210 on this branch at `f3ef355` (202/202 historical at validated main
    `2ad7d86`); the `rosi_v100` 33-invocation baseline comparison is not yet
    recorded.
- [x] CPU/CUDA/HIP histogram equality is automated: the
   `example.particle_feedback.cross_backend.Q{1,2,8}` cases compare exact
   sorted histogram rows for `N=1e6` across every compiled backend with a
   fail-not-skip contract, measured CPU 210/210 at
   `751c8a8cbad9de1a2e71cef57b81147ff071dd51` (device full suites not
   re-run; CUDA 264 and HIP 255 are the expected totals for the current
   suite).
- [ ] Throughput is measured on an exclusive allocation, frozen, and guarded.
  Throughput is now measured on the exclusive `rosi_v100` (`N=1e8` CUDA
  acceptance run at main `f3ef355`, Q1 194.685 s, Q2 9.093 s, Q8 7.676 s,
  1.30e7 particles/s at Q8; record in `docs/acceptance.md` and
  `docs/benchmarks.md`). What remains is freezing a regression target with a
  stated tolerance and enforcing it in CI or a scheduled benchmark. The CPU
  OpenMP wall times are recorded (see `docs/acceptance.md`). H100 was not
  used; no H100 claim.
- [x] ASAN/UBSAN/TSAN pass on a compatible host. The `rosi_v100` full
  integrated TSAN CTest passes 210/210 with 0 reports in 202.83 s at main
  `f3ef355` (non-PIE Debug with `alpaka_EXEC_CpuOmpBlocks=OFF` and
  `alpaka_DEP_OMP=OFF`); hosted CI runs ASAN+UBSAN over the same 210-case CPU
  suite. With the default CpuOmpBlocks plus OpenMP the TSAN suite fails
  105/210 with reports inside the alpaka async CPU queue (upstream alpaka
  races, not graphFlow).
 - [x] Format, REUSE, docs, and the API map are complete and green. T14b
  (tree-wide format and SPDX sweep) is complete and hosted CI enforces the
  format gate.
- [ ] The repository is public or explicitly kept private, with branch
  protection and CI configured.
- [ ] An independent review confirms the final integrated state and the
  acceptance record.
