<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow — remaining work

This is the complete list of work still required to finish the plan in
`/workspace/graph-async-chunk-scheduling-plan.md`. It is based on the current
repository state at code main `9fb15f1e6ce13f252b31d4cb60cc410862d56511` (PR
#69, the full SYCL test matrix), not on the intended final state.

- Repository: `https://github.com/psychocoderHPC-bot/graphFlow` (public).
  Changes land through a feature branch and a PR targeting `main`; branch
  protection is pending operator action.
- **Current validation:** CPU **255/255** (fwk394 Release, g++ 13.3), CUDA
  **316/316** on `hal` A30 `sm_80` (CUDA 13.4), HIP **307/307** on `hal`
  RX 7900 XTX `gfx1100` (ROCm 7.2), and SYCL **71/71** on `hal` Arc A770
  `spir64` (oneAPI 2026.2, PR #69 matrix, fp32 photons). No case is skipped or
  filtered; PR #63
  adds the public-Flow routed-axis and rejection cases, PR #65 adds the six
  `[scheduler]` cases (6/6 on CPU, CUDA and HIP), and PR #69 registers the 71-case
  SYCL device matrix, all on top of the P07b 7/7. The
  `ffe4074` totals 249/310/301, the `88de14f` totals 247/307/298, the `56c11ac`
  totals 246/306/297, and the earlier `a610281` totals 239/299/290 are
  historical. The full record, commands, and pins are in
  [acceptance.md](acceptance.md).
- **P07b performance:** exclusive `rosi_v100` Q/C re-run merged as `56c11ac`:
  `N=1e8`, `C=8192`, `P=16384`, Q8 median 12,987,013 particles/s, 260 rows
  summing to 100,000,000. The frozen target is 13,000,000 particles/s with a
  10 percent tolerance (floor 11,700,000) and passes. P07b also records the
  kernel-axis end-to-end matrix for scatter, dense/hash and the four
  distributions. The earlier P07 run at `64d4a1e` (Q8 7.690 s and 1.300e7
  particles/s) is historical. Details in [benchmarks.md](benchmarks.md).
- **TSAN:** the last full integrated TSAN CTest is 210/210 with 0 reports at
  `f3ef3552c60c0cfe0d202a73a08b808746ad11ce`; it predates the current suite,
  so no full TSAN result exists for
  `561b9a4c9435e37b164623bb6be60a3051783eb4`. The hosted `tsan` job exists
  since `2fefe54c6d32e9156be5c08ea497119d0bacc21e` (P10 commit
  `cf3bc8a72942b01c4a1ed32e4db7b260e22cb750`); its runtime was not observed.
  `hal` cannot run TSAN.
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
  **307/307** at `561b9a4c9435e37b164623bb6be60a3051783eb4`; the previous
  301/301 at `ffe4074`, 298/298 at `88de14f`, 297/297 at `56c11ac`, the
  290/290 at `a610281`, the original 245/245 gate with 43 HIP cases at
  `b972eaa`, and the 255/255 full suite at `f3ef355` are historical.
  GitHub-hosted HIP remains compile-only; the `gfx90a` compile-only gate is
  historical.
- CPU execution with the OpenMP backend works with passive spinning
  (`GOMP_SPINCOUNT=0`) and no hangs; the current CPU suite is 255/255.
- H100: not used for any current validation; no final H100 claim is made. The
  JURECA `jrc0880.jureca` run is preliminary CUDA evidence only (unknown
  gateway sharing flag; external load while the client was stale) and stays
  historical.

Priority legend: **P0** blocks plan acceptance, **P1** required for plan
completion, **P2** hardening or optimization. **Blocked** means an external
resource or decision is missing.

## 1. Core architecture and scheduler

- [x] **P0: multi-chunk batching.** Done (PR #6 / T5): `--batch-chunks Q`
  (default 1) dispatches up to `Q` ready input chunks per transform node, with
  one classify launch and one scatter launch per batch and the adaptive
  `outputChunkBound(sum(fill), C, B)` reserve; the integrated T13 run measured
  Q=1/2/8 with identical sorted histograms across Q. The current suites are
  255/255 CPU, 316/316 CUDA on `hal` A30 `sm_80`, and 307/307 HIP on `hal`
  RX 7900 XTX. The historical totals (249/249 CPU, 310/310 CUDA, 301/301 HIP
  at `ffe4074`; 247/247 CPU, 307/307 CUDA, 298/298 HIP
  at `88de14f`; 246/246 CPU, 306/306 CUDA, 297/297 HIP at `56c11ac`; 239/239,
  299/299 and 290/290 at `a610281`; 194/194 CPU and
  246/246 CUDA; 202/202 and 254/254 on `rosi_v100`;
  264/264 and 255/255 at `f3ef355`) stay labelled historical. Open refinement:
  `--batch-chunks Q` has no upper cap beyond the 32-bit parse limit, tracked in
  Section 7.
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
- [x] **P1: multiple in-flight batches per node.** Done (PR #46):
  `Tuning::inflightBatches` sets a bounded submitted depth per node; the default
  of 1 preserves the historical one-batch-per-node scheduling. The scheduler
  keeps fair round-robin across ready nodes, one event per submitted-depth lane,
  and backpressures by refusing to submit when every lane of a node is in flight.
  `Flow::run` creates one typed stage per lane, so per-lane device staging is
  replicated at a depth above 1. Acceptance tests (`tests/integration/test_async_execution.cpp`)
  cover depth-1 and depth-4 runs with a forced 50 ms in-flight delay: `maxInflight
  == depth`, `duplicateDeliveries == 0`, `maxReadyDepth >= 1` (lane exhaustion
  backpressures), and the full particle-feedback histogram is identical across
  every `(batchChunks, inflightBatches)` pair in `{1,2,8} x {1,2,4}`.
- [x] **P1: exploit or document queue overlap.** Resolved by documentation
  (P04, PR #45 at `e86adb73e41a5f70f4c9839c58895f7a789691ac`): the
  `tools/queue_probe.sh` timed two-queue run with the same-queue baseline
  measured overlap 4/5 on the shared `hal` A30, 5/5 on CPU and HIP, and the
  committed probe's single `no` verdict at 20,000,000 delay iterations was
  shown to be a `std::this_thread::yield()` polling-resolution artifact, not
  stream serialization; it is `yes` at 100,000,000 and 400,000,000
  iterations. The scheduler keeps one in-flight batch per node at the default
  depth of 1 and relies on full-chunk pipelining; with
  `Tuning::inflightBatches` above 1, independent per-node batches overlap.
  The verdict stays advisory and correctness never depends on overlap
  (see [known-limitations.md](known-limitations.md#queue-overlap-is-advisory-and-the-cuda-probe-is-timing-sensitive)).
- [x] **P1: device-aware chunk sizing.** The device-aware policy is in main
  (T1) and the integrated T13 matrix is recorded (`docs/benchmarks.md`).
  Closed boundary: the policy default `P=16384` exceeds the measured A30
  payload budget of 3,171,090,432 B (needs 3,221,225,472 B), so the T13 `N=1e8`
  run used an explicit `P=16128`. The auto-derived policy now clamps `P` to
  16,128; an explicit over-budget `P` still fails clearly.
- [x] **P1: reduce pool memory (ChunkLedger wrap closed).** `ChunkPool` and
  `PayloadSlotStorage` clamp a requested chunk count that exceeds the
  `poolPayloadBudgetBytes` (default 4 GiB) to the largest fitting physical
  count; `numChunks()` reports the clamped value. An explicit budget above
  4 GiB is honored. `Config::validate` still rejects an explicitly over-budget
  pool size with a clear error. The pool also rejects a total slot count that
  exceeds `UINT32_MAX`, so the 32-bit kernel slot index cannot wrap. The
  `ChunkPool` in-class generation and lease counters are checked on every
  increment and raise on 32-bit overflow. PR #58 (merged as
  `88de14f691a51edda5ad48e153dbbe353bef7c9e`) applies the same checked
  increment to the shared `ChunkLedger` used by `PayloadSlotStorage`: its
  generation and lease counters now raise on 32-bit overflow instead of
  wrapping, with a matching overflow test. A slot-based pool also remains
  future work.
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

- [x] **P1: batch-level classify/plan/scatter.** Covered by the P01
  in-flight-depth work (PR #46) and the T5 batching work (PR #6): a batch of
  up to `Q` input chunks runs one classify and one scatter launch, the host
  accumulates the batch-level `total[B]` from the per-chunk counts
  (`m_totals` in `PartitionStage`), and `planDenseBins` builds `binBase[B]`
  and the exact `outChunks[R]` fills over the summed totals. `next[bin]` is
  shared by every tile descriptor of the batch
  (`include/graphFlow/graph/ScatterKernel.hpp`), and the scatter stage
  verifies `next[bin] == total[bin]` for every bin after the batch scatter
   (`include/graphFlow/flow/Execution.hpp`), throwing on a mismatch. The
  multi-chunk cases pass on CPU, CUDA, and HIP (255/255, 316/316, 307/307).
- [ ] **P1: direct versus shared-staging scatter (GPU Direct open).** P03 at
  `ad7208f80fbb6a07d9a87767e947abc057c00e84` added the `Direct` dense-scatter
  alternative beside the shared-staging default and measured both on fwk394 CPU
  (`5.48` vs `15.85 ms` balanced, `2.13` vs `48.31 ms` one-hot, `2.21` vs
  `37.25 ms` skewed, `4.19` vs `6.62 ms` sparse). The committed integration
  cases compare the two paths on all four distributions. P07b runs the
  kernel-workload `direct` mode on `rosi_v100` CUDA at kernel level (see
  [benchmarks.md](benchmarks.md)). PR #63 routes `Direct` and the bin path
  through the public `Flow` (`Tuning::scatterMode` and `Tuning::binPath`) with
  `Config::validate()` rejecting unsupported combinations before any device
  work; only HIP `Direct` and the public-Flow performance numbers are missing,
  so this item stays open.
- [x] **P1: dense/hash threshold measurement.** P03 at
  `ad7208f80fbb6a07d9a87767e947abc057c00e84` made the threshold
  measurement-backed from the fwk394 CPU run: dense beat the `2*T` and `4*T`
  hash tables through `B=1024`, so `denseBinLimit` stays 1024 and the hash
  capacity stays `2*T`. The added cases pin direct/shared-staging agreement,
  `2*T`/`4*T` agreement at overlapping bins, and the dense-limit helper.
  Table-full remains a hard batch error. End-to-end scatter/dense/distribution
  throughput is still blocked under the measurement-matrix item below.
- [x] **P2: host round-trip profiling and device-side planning experiment.**
  Resolved by measurement and the keep-host-side-planning decision (P04, PR
  #45 at `e86adb73e41a5f70f4c9839c58895f7a789691ac`): an `nsys` profile of
  `graphFlow_particle_feedback` on the `hal` A30 (`N=1e6`, `C=1024`,
  `P=8192`) records classify, D2H total/status, host planning, H2D table,
  scatter, memset, and the per-batch event/dispatch round trip; Q=8 batching
  removes about 99x of the handoffs and drops the total traced round trip
  from 10.88 s to 116 ms. The remaining cost is event and dispatch latency
  that device-side planning does not remove, because the host still reserves
  output chunks and builds the scatter table. Device-side planning stays
  out of scope for v1 and is revisited only with a device-resident output
  chunk pool or a many-bin workload where the O(B) plan is measurable
  (see [known-limitations.md](known-limitations.md#host-planning-stays-batching-amortizes-the-round-trip)).
- [ ] **P1: error and abort coverage (real-driver injection open).** Host tests
  cover invalid classifier and updater results, counter and weight overflow, a
   full hash table, pool exhaustion, an injected completion-event enqueue
   failure, and abort during in-flight work. The CUDA/HIP cases added at
   `10c1733770e16ebaecc74513034e6d327924503c`
   (`tests/integration/FailurePaths.hpp`) cover invalid classifier and
   updater results, a full hash table, pool exhaustion, and an injected enqueue
  failure, and they fail rather than skip when no device is visible; the PR #69
  SYCL matrix adds the same failure paths on the Arc A770. The
  current suites are 255/255 CPU, 316/316 CUDA, 307/307 HIP and 71/71 SYCL at
  `9fb15f1e6ce13f252b31d4cb60cc410862d56511` and include these failure
  cases. The injection hook is test-only: no production path calls
  `GraphRunner::setFailureHook`, and the hook raises a host exception at a named
  scheduler point instead of injecting a driver fault. Missing artifact: real
  mid-run CUDA/HIP out-of-memory, ECC, Xid, context-loss, or `wait`-failure
  injection. That gap is why this item stays open (see
  [known-limitations.md](known-limitations.md#failure-injection-is-test-only)).
  The covered paths assert no partial output is published and every lease is
  released.
- [x] **P2: counters and validation.** The photon per-event payload checks are
  implemented behind `GRAPHFLOW_ENABLE_DEBUG_VALIDATION` (issue #26), and P03
  added debug-only input-conservation counters in the dense and hash scatter
  kernels. `GraphRunner::run()` now adds the scheduler-level checks under the
  same flag: source/transform batches must publish exactly their input
  particles (the host-side form of `next[bin]==total[bin]`), every published
  fill must lie in `[1, C]`, and a chunk incarnation may be scheduled for a
  sink only once. The checks sit behind `if constexpr(debug::checksEnabled)`
  and the empty-base release state keeps them out of the hot path; the
  `[scheduler][debug]` cases exercise each violation and the
  `[scheduler][release]` case proves a lossy transform completes unchecked in
  Release. At `561b9a4c9435e37b164623bb6be60a3051783eb4` the six
  `[scheduler]` cases pass 6/6 on CPU, CUDA and HIP, and the Release
  compile-out keeps the CPU `GraphRunner` at 288 bytes (352 bytes with
  `GRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON`); no diagnostic strings or check
  symbols for the new code are present in the Release binary.

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
   at this commit. The historical V100 CUDA `sm_70` total is 254/254 (Release)
   with the explicit concept-parameter workaround; the current CUDA suite is
   316/316 on `hal` A30 `sm_80`.
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
  Measured at `751c8a8`: historical CPU 210/210 (OMP=ON Release, pedantic),
  with the three Q cases exact. The device evidence is two build pairs, not one
  three-backend run: CUDA and HIP are mutually exclusive in a single build, and
  the CPU backend stays enabled in accelerator builds, so the CPU+CUDA pair and
  the CPU+HIP pair each ran the three Q cases 3/3 with the device output
   compared against the CPU reference (same seed/N/C/P/Q, sorted `weight,count`
   rows identical, count sum N for every backend). The current full device
   suites are CUDA 316/316 on `hal` A30 `sm_80` (CUDA 13.4) and HIP 307/307 on
   `hal` RX 7900 XTX `gfx1100` (ROCm 7.2); the `f3ef355` totals 264/264 and
  255/255 are historical.
- [x] **P1: device-dependent default `C`/`P`.** Done (policy PR #11): the
  device-aware policy derives `C`/`P`/`T`, auto-clamps `P` to the usable-memory
  budget, and rejects an explicit over-budget `P`. The example prints the
  resolved policy and the chosen values, and the CLI override stays.
- [x] **P1: full max-visits/non-termination coverage.** P06 at
  `a4c523ae6693a9c71e07b804dffd6787d2e8966e` added the missing cases:
  `max-visits cap bounds non-terminating feedback`, `zero-weight particles are
  sunk and form the first histogram row`, `empty particle input completes
  without launching work`, `exact chunk and admission fill conserves every
  particle`, and the bounded/small-P/tight-pool pressure cases. The existing
  scripted fixture and the Q=8 feedback matrix cover multiple feedback rounds.
- [x] **P2: result-storage policy.** The v1 policy is explicit: `O(N)`,
  `4 B/weight`, device `weights` plus `cappedFlags` at `8 B/particle`, and a
  host transient of about `24 B/particle` (about 2.4 GB at `N=1e8`, 24 GB at
  `N=1e9`). The histogram uses an `O(N log N)` sort. A device histogram is
  added only when `8N` bytes plus the pool payload exceeds the device budget.
  The policy is recorded in [known-limitations.md](known-limitations.md) and
  the README.
- [x] **P2: CLI contract tests.** Done (P06, PR #51 at
  `a4c523ae6693a9c71e07b804dffd6787d2e8966e`): `tests/unit/test_cli_contract.cpp`
  asserts the selected backend and device in the example output, pins the real
  `graphFlow_particle_feedback` exit codes for help, usage errors, out-of-range
  devices, run-stage overflow, and the compiled/uncompiled backend split, and
   the `example_cli.cmake` matrix covers `N=0` header-only output and `C=0`
   rejection. The cases run in the 255/255 CPU, 316/316 CUDA, and 307/307 HIP
   suites.

## 4. CI, format, REUSE, and repository policy

- [x] **P0: GitHub Actions workflow.** `.github/workflows/ci.yml` runs on
  GitHub-hosted `ubuntu-latest` runners and public containers:
  - CPU Release and Debug with CTest, JUnit upload, and `--no-tests=error`;
   - ASAN+UBSAN on CPU plus a dedicated `leak` step over the `[failure]` and
     `[status]` error-path cases (`GRAPHFLOW_ENABLE_ERROR_PATH_LEAK_CHECK=ON`);
   - a hosted `tsan` job on `ubuntu-latest` (non-PIE Debug, OMP off,
     CpuOmpBlocks off);
   - CUDA compile-only (`sm_70`), HIP compile-only (`gfx90a`), and
     SYCL compile-only (Intel GPU only, oneAPI 2025.3 container);
   - `reuse lint`;
   - finite timeouts and JUnit artifact upload.
   Still open: a CUDA, HIP, or SYCL runtime gate on a device host. The workflow
   uses no self-hosted runner. The P10 fwk394 evidence at `cf3bc8a` is ASAN+UBSAN
   236/236, leak 1/1, TSAN 235/235 with 0 reports; the hosted `tsan`, `leak`,
   and SYCL runtimes were not observed, and no H100 was used.
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
- [x] **P1: complete the alpaka API map.** Done (P08, PR #55 at
  `82dd2cec795357ea748bd7782144df74e71bf1f2`): `docs/alpaka-api-map.md` now
  covers every accelerator helper graphFlow uses, with the searched symbols,
  the pinned path at alpaka revision `eb677e8d7f70d79b35df25d9002f94ba669b6f44`,
  the exact API form, the backend evidence per capability, and the reason for
  each project-owned code path. The new accelerator helpers added since the
  first map (frame launch, queue probe, scatter mode) are all listed.
- [ ] **P1: repository policy (branch protection pending operator).** The
  repository is public, the README carries the CI badge, and the PR flow is
  documented in [ci.md](ci.md) (feature branch -> PR to `main`, CI jobs plus
  review before merge). Missing artifact: branch protection for `main`, which is
  an operator action. No branch-protection setting is changed from the
  repository.
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
    kernels. The current HIP suite is **307/307** at
    `561b9a4c9435e37b164623bb6be60a3051783eb4`; the previous 301/301 at
    `ffe4074`, 298/298 at `88de14f`, 297/297 at `56c11ac` and the 290/290 at
    `a610281` are historical.
  Remaining HIP work: other AMD architectures, the HIP hash path, an AMD
  `N=1e8` acceptance run, and no performance claim while `hal` reports
  `sharing=unknown`.
- [x] **P0: SYCL execution gate.** Done at
  `9fb15f1e6ce13f252b31d4cb60cc410862d56511` (PR #69) on `hal` with an Intel
  Arc A770 (`spir64`, oneAPI 2026.2, icpx 2026.2.0): **71/71** CTest via
  `ctest -L sycl` (33 in-process Catch2 cases, 24 particle CLI, 3
  cross-backend exact-histogram, 10 photon CLI, 1 photon output). The Arc
  A770 lacks the SYCL `fp64` aspect, so the oneAPI photon path runs fp32;
  AOT compilation with `ocloc` (`-fsycl-targets=spir64_gen`) crashes on
  `hal`, so the build stays JIT-compiled SPIR-V. GitHub-hosted SYCL remains
  compile-only in the `sycl-compile` job; no SYCL runtime claim is made for
  GitHub CI, and no SYCL performance run is recorded while `hal` reports
  `sharing=unknown`. Remaining SYCL work: a second Intel device and an
  `N=1e8` acceptance run.
- [x] **P0: exclusive or repeated acceptance run.** P07 on `rosi_v100` at
  `64d4a1e11949886c7147c082cc386edde4bdb01f` (merged as `a610281`; V100
  SXM2-32GB, CUDA 12.6.85 `sm_70`, Release), `C=8192`, `P=16384`, seed 42:
  Q1 median 191.990 s, Q2 8.710 s, Q8 7.690 s, 1.300e7 particles/s at Q8, 260
  histogram rows, count sum exactly 100,000,000, sorted-row SHA-256
  `21ab0aeb9ec72d380af2ce6e9631f8fa7b9111d24ff7831f272aa9f6a7be9dab`. The
  older `f3ef355` run (Q1 194.685 s, Q2 9.093 s, Q8 7.676 s, hash
  `89dc7fb9962e898fe2f1fe0116ad98fe2df9dceb77d33aa890a56467615d02e9`) is
  historical. The later P07b Q/C re-run merged as `56c11ac` measures a Q8
  median of 12,987,013 particles/s and also records the kernel-axis matrix.
  `hal` is shared (`sharing=unknown`); its numbers stay labelled as a
  baseline. H100 was not used, so no H100 performance claim is made.
- [ ] **P1: freeze the throughput target (enforcement open).** P07 froze the
  target at 13,000,000 particles/s for `N=1e8`, `C=8192`, `P=16384`, Q=8, with
  a 10 percent tolerance (floor 11,700,000). The measured P07b median
  12,987,013 passes in `tools/bench_matrix.sh`. Missing artifact: a CI or
  scheduled job that invokes the gate; no workflow job runs it yet.
- [ ] **P1: complete the measurement matrix (public Flow matrix open).**
  P07 records wall time, particles/s, the copy reference, and the frozen target
  for the default dense example path. P07b adds kernel-level end-to-end numbers
  for `shared|direct` scatter, `dense|hash` bin paths and the
  balanced/one-hot/skewed/sparse distributions through the opt-in
  `--workload kernel` path, including dense `direct` on `rosi_v100` CUDA. PR
  #63 routes the same axes through the public `Flow` (`Tuning::scatterMode` and
  `Tuning::binPath`) with `Config::validate()` rejecting unsupported
  combinations, but no public-Flow performance matrix and no HIP `direct`
  execution are recorded, so this item stays open. Missing artifact:
  public-Flow measurements for those axes.
- [x] **P1: copy-bandwidth reference.** P07 measured 409.733 GB/s over
  3,221,225,472 B (10 device-to-device copies, 7.862 ms per copy) on the
  exclusive `rosi_v100`. It is a copy-engine reference, not a scatter
  bandwidth, and is recorded next to the scatter discussion in
  [benchmarks.md](benchmarks.md).
- [x] **P1: `N=1e6` CI smoke.** Done for the hosted CPU job: the fixed-seed
  `example.particle_feedback.smoke.N1000000` CTest case runs in the ordinary
  CTest job with a finite timeout and an exact histogram-sum check. Hosted CUDA
  and HIP stay compile-only, so their `N=1e6` runtime smoke remains a manual
  `rosi_v100`/`hal` check. `N=1e8` stays the manual acceptance run.
- [x] **P1: HAL acceptance record.** The record is in
  [acceptance.md](acceptance.md): exact code commit, alpaka/Catch2/CMake/Ninja
  pins, g++/CUDA/ROCm versions, A30/RX 7900 XTX devices, the `hal` driver
  (NVIDIA 615.71.09), commands, counts, the no-skipped-test assertion, the
  P07b result, and the TSAN status with exact commits. The retained JUnit XML
  reports 246 CPU, 306 CUDA, and 297 HIP with no failures or skips; those
  JUnit counts were captured at `d71935c` and predate the final 249/310/301
  suite, so the record is the `d71935c` HAL run, not the final totals. Raw
  logs stay outside the repository.

## 6. Robustness and hardening

- [x] **P0: TSAN validation.** Done on `rosi_v100` at
  `f3ef3552c60c0cfe0d202a73a08b808746ad11ce`: full integrated TSAN CTest
  **210/210, 0 reports**, 202.83 s, non-PIE Debug with
  `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`. This is the last
  full TSAN run; it predates the current 255-case suite, so the current tree
  has no full TSAN result. With the default CpuOmpBlocks plus OpenMP the same
  suite fails 105/210 with reports inside the alpaka async CPU queue; those are
  upstream alpaka races, not graphFlow races, which is why the clean gate uses
  the OMP-exec-disabled configuration. Since `gf/p10-sanitizers` at
  `cf3bc8a72942b01c4a1ed32e4db7b260e22cb750` (merged as
  `2fefe54c6d32e9156be5c08ea497119d0bacc21e`) the same configuration runs as a
  hosted `tsan` job on `ubuntu-latest`; the P10 fwk394 run is TSAN 235/235 with
  0 reports, and the hosted job runtime was not observed (no H100 used). The
  fwk394 non-PIE scheduler/pool subset (14/14) and the hal startup abort
  (`FATAL: ThreadSanitizer: unexpected memory mapping`, even for a non-PIE
  binary) are historical context; `hal` is shared (`sharing=unknown`) and not a
  usable TSAN host. No TSAN suppressions were used; suppressions are allowed
  only with evidence and must not hide a race.
- [x] **P1: stress tests.** Done (P06, PR #51 at
  `a4c523ae6693a9c71e07b804dffd6787d2e8966e`, plus the error/abort coverage
  at PR #47/PR #48): `tests/integration/test_pressure.cpp` covers `N >> P`
  across `Q`, `P` below `C`, and a tight pool that reaches its physical
  capacity and reuses every chunk; `tests/integration/test_particle_feedback.cpp`
   covers the max-visits cap, feedback cycles, and the zero-weight and empty
   input cases; the failure-path cases (PR #48 at
   `10c1733770e16ebaecc74513034e6d327924503c`) cover pool
   exhaustion and abort during in-flight work. The cases pass on CPU, CUDA,
   and HIP (255/255, 316/316, 307/307).
- [ ] **P1: asynchronous failure propagation.** Enqueue-failure and abort paths
   are exercised by the host failure tests and by the CUDA/HIP failure cases
   added at `10c1733770e16ebaecc74513034e6d327924503c` (invalid classifier and
   updater results, full hash table,
  pool exhaustion, injected event-enqueue failure). The injection hook is
  test-only and cannot raise real mid-run CUDA/HIP OOM, ECC, or context-loss
  errors from the device or runtime; pool exhaustion exercises the pool
  allocator's capacity, not a driver allocation failure. `wait` failures and
  real-driver faults remain an open gap and no fault-injection driver is wired
  in, so this item does not claim full coverage. On the covered paths leases,
  staging, and events are released exactly once and the process returns
  nonzero.
- [x] **P2: sanitizer coverage for the host scheduler.** Closed on
   `gf/p10-sanitizers` (`cf3bc8a`): hosted CI keeps ASAN+UBSAN, adds a
   dedicated `leak` step (`graphflow.error_path_leak_check`, the `[failure]`
   and `[status]` cases in one process under LeakSanitizer), and adds a hosted
   `tsan` job (non-PIE, OMP off, CpuOmpBlocks off). P10 fwk394 evidence:
   ASAN+UBSAN 236/236, leak 1/1, TSAN 235/235 with 0 reports; the hosted
   `tsan` and `leak` runtimes were not observed, and no H100 was used.
- [ ] **P2: overflow hardening.** Audit `chunkIndex*C`, `chunk*C+offset`,
  `tile*T`, `ceil(n/C)`, and counter arithmetic for 32-bit overflow; add debug
  assertions and document the supported limits.

## 7. Blockers and decisions

- **AMD hardware:** resolved for correctness. `hal` exposes an AMD Radeon RX
  7900 XTX (`gfx1100`, ROCm 7.2) and the current HIP suite passes **307/307**
  at `561b9a4c9435e37b164623bb6be60a3051783eb4`; the previous 301/301 at
  `ffe4074`, 298/298 at `88de14f`, 297/297 at `56c11ac` and the 290/290 at
  `a610281` are historical. The 245/245 gate at `b972eaa` and the
  255/255 suite at `f3ef355` are historical. Other
  AMD architectures and the AMD `N=1e8` acceptance remain open; GitHub-hosted
  HIP stays compile-only.
- **TSAN host:** resolved. The TSAN host is `rosi_v100` (exclusive); the last
  full integrated TSAN CTest passed 210/210 with 0 reports in 202.83 s at
  `f3ef3552c60c0cfe0d202a73a08b808746ad11ce` (non-PIE Debug with
  `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`). Since
  `gf/p10-sanitizers` at
  `cf3bc8a72942b01c4a1ed32e4db7b260e22cb750` (merged as
  `2fefe54c6d32e9156be5c08ea497119d0bacc21e`) a hosted `tsan` job on
  `ubuntu-latest` runs the same configuration; the P10 fwk394 run is 235/235
  with 0 reports, the hosted job runtime was not observed, and no H100 was
  used. The current 255-case suite has no full TSAN run; the hal kernel still
  rejects the TSAN memory mapping and the fwk394 non-PIE subset (14/14) is
  historical context.
- **CI status:** the hosted CPU jobs run the 255-case CPU suite (Release and
  Debug) and the `sanitizers` job runs ASAN+UBSAN over it plus a dedicated
  `leak` step, with no skip. A hosted `tsan` job runs the host scheduler under
  TSAN (non-PIE, OMP off, CpuOmpBlocks off). The historical 202-case final main
  was green (CPU 202/202 Release and Debug, ASAN+UBSAN Debug 202/202);
  `format` (clang-format 20.1.8, T14b) and `reuse lint` pass; the
  `sycl-compile` job builds the Intel-GPU-only SYCL target and, with
  `cuda-compile` and `hip-compile`, is compile-only; the SYCL device matrix
  (71/71 at `9fb15f1` on `hal` Arc A770) has no hosted runtime job, and there
  is no self-hosted runner or GitLab pipeline. The
  P10 fwk394 run at `cf3bc8a` is ASAN+UBSAN 236/236, leak 1/1, TSAN 235/235
  with 0 reports, and the hosted `tsan`/`leak` runtimes were not observed. The
  CUDA, HIP, or SYCL runtime gate still needs a device host with a known
  sharing and
  label configuration; no self-hosted runner is provisioned. No H100 was used.
- **CI GPU runtime:** hosted CI is in `.github/workflows/ci.yml` and its CUDA,
  HIP, and SYCL jobs are compile-only (`sycl-compile` builds the Intel-GPU-only
  SYCL target in the oneAPI 2025.3 container and runs no CTest). A CUDA, HIP,
  or SYCL runtime gate still needs a device host with a known sharing and
  label configuration; no self-hosted runner is
  provisioned.
- **Exclusive throughput:** resolved for the acceptance run. The P07b Q/C
  re-run merged as `56c11ac` on `rosi_v100` records a Q8 median of 12,987,013
  particles/s and 260 rows summing to 100,000,000; the frozen target passes.
  P07b also records the kernel-axis end-to-end matrix. The earlier P07 run at
  `64d4a1e` (Q8 7.690 s, 1.300e7 particles/s) and the `f3ef355` run are
  historical. `hal` is shared (`sharing=unknown`); its 187.5 s number stays a
  shared-host baseline. H100 was not used; the JURECA `jrc0880.jureca` run is
  preliminary CUDA evidence only and no H100 performance claim is made.
- **CPU OMP backend:** CPU execution with the OpenMP backend works on
  `rosi_v100` with passive spinning (`GOMP_SPINCOUNT=0`); the current CPU suite
  is 255/255. The `rosi_v100` 210/210 and 202/202 totals are historical.
- **Pool memory:** the payload budget and chunk clamp are implemented, and the
  `ChunkLedger` generation/lease increment is now checked (PR #58). A
  slot-based pool remains an optional optimization.
- **Queue overlap:** resolved. The timed two-queue probe (P04, PR #45 at
  `e86adb73e41a5f70f4c9839c58895f7a789691ac`) shows real overlap on the
  `hal` A30, CPU, and HIP; the single committed-probe `no` verdict at
  20,000,000 delay iterations is a polling-resolution artifact. The scheduler
  documents its full-chunk pipelining strategy at the default depth of 1 and
  uses per-node batch overlap only when `Tuning::inflightBatches` is above 1.
  Correctness never depends on overlap.
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

The plan section 14 acceptance checklist remains open: the throughput guard, the
branch-protection operator action, and the independent final review are not
closed.

- [x] All P0 items above are implemented and covered by tests (batching, RNG
  API, example CLI in CTest, cross-backend histogram equality, CUDA kernel
  test, HIP execution gate, GitHub Actions workflow, format, REUSE, the
  `N=1e8` acceptance run on `rosi_v100`, and TSAN validation).
- [x] CPU, CUDA, and HIP build. The current suites are 255/255 CPU, 316/316
  CUDA on `hal` A30 `sm_80` (CUDA 13.4), and 307/307 HIP on `hal` RX 7900 XTX
  (ROCm 7.2). PR #63 adds the public-Flow routed-axis cases and PR #65 adds
  the six `[scheduler]` cases on top of the P07b 7/7, included above. The
  previous 249/249, 310/310 and 301/301 at `ffe4074`, 247/247, 307/307 and
  298/298 at `88de14f`, the 246/246, 306/306 and 297/297 at `56c11ac`, the
  239/239, 299/299 and 290/290 at `a610281`, and 245/245 HIP at `b972eaa`,
  210/210 CPU and 254/254 CUDA at `f3ef355`/`2ad7d86`, and 264/264 / 255/255
  at `f3ef355` are historical.
- [x] The example CLI is registered in CTest for the full CPU, CUDA, and HIP
  matrices; the HIP cases run on `hal` with a real AMD GPU.
- [x] The public API (`Flow<TPayload, TDevice, TStorage>`) is implemented and
  used by both examples; the particle example is ported to it and the photon
  example uses it for slab transport.
- [x] `N=1e6` runs in CI: the fixed-seed `N=1e6` CPU smoke
   (`example.particle_feedback.smoke.N1000000`) runs in the ordinary hosted CPU
   CTest job with a finite timeout and exact histogram-sum check, no skip.
    The current `N=1e8` acceptance record is the P07b Q/C re-run merged as
    `56c11ac` (V100 SXM2-32GB, CUDA 12.6.85 `sm_70`, Release): Q8 median
    12,987,013 particles/s, 260 rows, sum exactly 100,000,000, sorted-row
    SHA-256 `21ab0aeb9ec72d380af2ce6e9631f8fa7b9111d24ff7831f272aa9f6a7be9dab`,
    and a 409.733 GB/s copy reference. P07b also records the kernel-axis
    end-to-end matrix. The earlier P07 run at `64d4a1e` (Q8 7.690 s, 1.300e7
    particles/s) and the `f3ef355` run are historical. H100 was not used; no
    H100 claim.
- [x] CPU/CUDA/HIP histogram equality is automated: the
   `example.particle_feedback.cross_backend.Q{1,2,8}` cases compare exact
   sorted histogram rows for `N=1e6` across every compiled backend with a
   fail-not-skip contract, measured historical CPU 210/210 at
     `751c8a8cbad9de1a2e71cef57b81147ff071dd51`; the current full device suites
     are 316/316 CUDA on `hal` A30 `sm_80` (CUDA 13.4) and 307/307 HIP on
     `hal` `gfx1100` (ROCm 7.2). The `f3ef355` 264/264 and 255/255 totals are
     historical.
- [ ] Throughput is measured on an exclusive allocation, frozen, and guarded.
  P07b measures 12,987,013 particles/s at Q8 and freezes the 13,000,000
  particles/s target with a 10 percent tolerance; the measured median passes.
  Missing artifact: a CI or scheduled benchmark job that invokes
  `tools/bench_matrix.sh`. The CPU OpenMP wall times are recorded in
  `docs/acceptance.md`. H100 was not used; no H100 claim.
- [x] ASAN/UBSAN/TSAN pass on a compatible host. The historical `rosi_v100`
   full integrated TSAN CTest passed 210/210 with 0 reports in 202.83 s at
   `f3ef3552c60c0cfe0d202a73a08b808746ad11ce` (non-PIE Debug with
   `alpaka_EXEC_CpuOmpBlocks=OFF` and `alpaka_DEP_OMP=OFF`); the current
   255-case suite has no full TSAN run. Hosted CI runs ASAN+UBSAN over the
   current CPU suite, a dedicated `leak` step, and a hosted `tsan` job in the
   same configuration. P10 fwk394 evidence at `cf3bc8a`: ASAN+UBSAN 236/236,
   leak 1/1, TSAN 235/235 with 0 reports; the hosted `tsan`/`leak` runtimes
   were not observed, and no H100 was used. With the default CpuOmpBlocks plus
   OpenMP the TSAN suite fails 105/210 with reports inside the alpaka async
   CPU queue (upstream alpaka races, not graphFlow).
- [x] Format, REUSE, docs, and the API map are complete and green. T14b
   (tree-wide format and SPDX sweep) is complete and hosted CI enforces the
   format gate. The alpaka API map was completed in P08 (PR #55 at
   `82dd2cec795357ea748bd7782144df74e71bf1f2`) and covers every accelerator
   helper graphFlow uses.
- [ ] The repository is public with CI configured; branch protection for
  `main` is pending operator action.
- [ ] An independent review confirms the final integrated state and the
  acceptance record.
