<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow — remaining work

This is the complete list of work still required to finish the plan in
`/workspace/graph-async-chunk-scheduling-plan.md`. It is based on the current
repository state, not on the intended final state.

Current validated state at integrated main
`ac89ab558c26b938038035b26ca5dbe6d7ce7727` (T5):

- Repository: `https://github.com/psychocoderHPC-bot/graphFlow` (public).
- CTest totals: 194/194 on fwk394 CPU, 246/246 on hal CUDA `sm_80`, per
  `docs/acceptance.md`. The totals include the photon-scattering cases and
  the `[cuda]` kernel and FrameSpec matrix cases.
- The public API is implemented: `Flow<TPayload, TDevice, TStorage>` with
  source/partition/transform/sink handles, named ports, `configure(Tuning)`,
  typed failure `Status`, and `run()`.
- T1 (device-aware policy), T2 (payload storage), T3 (CUDA kernel test and
  typed Flow execution), T5 (batching, payload budget, photon scattering),
  T7 (pool payload budget), T13 (integrated benchmark), T14a (format/REUSE
  foundation), and the `Flow`/particle/photon examples are done.
- HIP compiles the full tree for `gfx90a`; no AMD device is visible on hal, so
  HIP execution is blocked and no HIP execution is claimed.
- CPU execution with the OpenMP backend is blocked and not claimed.
- H100: the JURECA `jrc0880.jureca` run is preliminary CUDA evidence only
  (unknown gateway sharing flag; external load while the client was stale), so
  no final H100 claim is made.
- `hal` reports `sharing=unknown`; the throughput is a shared-host baseline.

Priority legend: **P0** blocks plan acceptance, **P1** required for plan
completion, **P2** hardening or optimization. **Blocked** means an external
resource or decision is missing.

## 1. Core architecture and scheduler

- [x] **P0: multi-chunk batching.** Done (PR #6 / T5): `--batch-chunks Q`
  (default 1) dispatches up to `Q` ready input chunks per transform node, with
  one classify launch and one scatter launch per batch and the adaptive
  `outputChunkBound(sum(fill), C, B)` reserve; the integrated T13 run measured
  Q=1/2/8 with identical sorted histograms across Q. The current tree passes
  194/194 CPU on fwk394 and 246/246 CUDA on hal. Open refinement:
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
- [ ] **P1: chunk-fill policy tests.** Pin `min(C, P, remaining)` admission
  for `P<C`, `P==0`, the tail, and `N>>P`. Add a regression test that would fail
  if one-particle source chunks return.
- [ ] **P1: scheduler state and stall diagnostics.** The stall timeout and state
  dump exist; expose the remaining counters (ready depth per node, in-flight
  depth, reserved credits, event-pool use, source budget) in the example output
  and in failure messages.

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
  levels, and exactly-once sink delivery; keep these off the hot path.

## 3. Example, RNG, and end-to-end tests

- [x] **P0: register the example CLI in CTest.** The real
  `graphFlow_particle_feedback` binary runs through `example_cli.cmake` for
  `N=0,1,17,127,128,4097` crossed with `C=1,31,127,128` (24 cases per backend)
  with finite timeouts and a failure when the requested backend did not run.
  A CUDA build adds the same 24 cases for CUDA; HIP cases remain unavailable
  without an AMD runtime device.
- [ ] **P0: automate CPU/CUDA/HIP histogram equality.** The manual `N=1e6`
  CPU/CUDA diff is empty. Make it a test that compares sorted histogram rows for
  the same `(N, seed, C)` on every compiled backend; add HIP when an AMD device
  is available.
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

- [ ] **P0: GitHub Actions workflow.** Add `.github/workflows/ci.yml` with:
  - CPU Release and Debug on `ubuntu-latest`, CTest with JUnit and
    `--no-tests=error`;
  - ASAN+UBSAN on CPU;
  - TSAN on a host where the runtime mapping check passes;
  - CUDA on a self-hosted A30 runner with `CMAKE_CUDA_ARCHITECTURES=80`,
    environment scrub, and no skipped GPU tests;
  - HIP compile-only on the current host and HIP execution on an AMD runner when
    available;
  - format and `reuse lint` jobs;
  - finite timeouts and artifact upload for logs/JUnit.
- [x] **P0: format configuration.** The upstream-derived `.clang-format`
  (clang-format 20.1.8) is in main (PR #5 / T14a). The tree-wide format sweep
  (T14b) and the CI gate that fails on a diff remain open; T14b is CPU-only
  work.
- [x] **P0: REUSE compliance.** The REUSE foundation is in main (PR #5 / T14a):
  `LICENSES/MPL-2.0.txt` and `.reuse/dep5` are present. The tree-wide SPDX sweep
  (T14b) and the `reuse lint` CI gate remain open; T14b is CPU-only work.
- [x] **P1: add the plan to the repository.** Done: `docs/plan.md` is in main
  with an SPDX header.
- [x] **P1: complete the documentation set.** `docs/build.md` (exact pins,
  configure/build/test commands, environment scrub), `docs/architecture.md`,
  `docs/numerics.md`, `docs/known-limitations.md`, `docs/benchmarks.md`, and
  `docs/public-api.md` are in main and linked from the README. Remaining:
  `docs/ci.md` is part of the open CI workflow (PR #4 is a draft).
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
- [ ] **P0: HIP execution gate. Blocked.** No AMD runtime device is visible on
  hal; HIP is compile-only for `gfx90a`. Run the example and the relevant tests
  on an AMD GPU. Record the GPU model, ROCm version, `gfx` target, and results.
- [ ] **P0: exclusive or repeated acceptance run.** `hal` is shared
  (`sharing=unknown`). JURECA `jrc0880.jureca` (4x H100 PCIe 80GB, SM 9.0) is
  available and exclusive per operator, but its gateway sharing flag is unknown
  and no H100 baseline is recorded yet. Re-run `N=1e8` on an exclusive
  allocation or repeat the baseline enough times to report variance; keep the
  shared-host numbers labelled as a baseline.
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
- [ ] **P1: `N=1e6` CI smoke.** Run `N=1e6` on CPU and CUDA in CI (and HIP when
  available) with a finite timeout; keep `N=1e8` as the manual acceptance run.
- [ ] **P1: HAL acceptance record.** Record exact commit, alpaka pin, compiler,
  CUDA/ROCm versions, driver, GPU, commands, test counts, JUnit, and the
  no-skipped-test assertion. Keep raw logs outside the repository and link their
  location.

## 6. Robustness and hardening

- [ ] **P0: TSAN validation.** The non-PIE scheduler/pool subset is clean on
  fwk394 (14/14); hal is blocked by `unexpected memory mapping` even for a
  non-PIE binary, and no full integrated TSAN run is claimed. Run TSAN on a
  compatible host; add a suppression only with evidence and do not hide a race.
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

- **AMD hardware:** no AMD GPU is visible through `computeMCP_targets`; HIP is
  compile-only. A real AMD runner is required for the HIP runtime gate and the
  AMD `N=1e8` acceptance.
- **TSAN host:** the hal kernel rejects the TSAN memory mapping. A compatible
  host or a different runtime/kernel is required.
- **CI runners:** the workflow needs self-hosted CUDA and AMD runners with known
  labels, credentials, and artifact retention. PR #4 is a draft and no runners
  are provisioned; the repository is public and has no `.github/workflows` on
  main.
- **Exclusive throughput:** `hal` is shared; the current 187.5 s is a baseline,
  not an exclusive-hardware acceptance number. The JURECA `jrc0880.jureca`
  H100 run is preliminary CUDA evidence only: its gateway sharing flag is
  unknown and the recorded run observed external load while the client was
  stale, so no final H100 claim is made.
- **CPU OMP backend:** CPU execution with the OpenMP backend is blocked and
  not claimed; only the serial CPU backend has a recorded run.
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

- [ ] All P0 items above are implemented and covered by tests.
- [ ] CPU, CUDA, and HIP build; HIP execution is validated on an AMD GPU or the
  gate is explicitly recorded as blocked. CPU OMP execution is blocked and not
  claimed.
- [x] The example CLI is registered in CTest for the full CPU/CUDA matrix;
  HIP remains blocked by the missing AMD runtime device.
- [x] The public API (`Flow<TPayload, TDevice, TStorage>`) is implemented and
  used by both examples; the particle example is ported to it and the photon
  example uses it for slab transport.
- [ ] `N=1e6` runs in CI; `N=1e8` passes on A30 and AMD with exact histograms.
- [ ] Throughput is measured on an exclusive allocation, frozen, and guarded.
  The H100 run is preliminary CUDA evidence only (external load, stale client);
  no final claim.
- [ ] ASAN/UBSAN/TSAN pass on a compatible host.
- [ ] Format, REUSE, docs, and the API map are complete and green. T14b
  (tree-wide format and SPDX sweep) is CPU-only work and remains open.
- [ ] The repository is public or explicitly kept private, with branch
  protection and CI configured.
- [ ] An independent review confirms the final integrated state and the
  acceptance record.
