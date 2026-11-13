<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# graphFlow — remaining work

This is the complete list of work still required to finish the plan in
`/workspace/graph-async-chunk-scheduling-plan.md`. It is based on the current
repository state, not on the intended final state.

Current validated state:

- Repository: `https://github.com/psychocoderHPC-bot/graphFlow` (public),
  branch `main`, HEAD `2877c8578047702506eb71fc024570e5ba052790`.
- Last recorded validation on the integrated revision: 74/74 CTest cases on
  fwk394 CPU, 75/75 on hal CUDA, per `docs/acceptance.md`; no new test run is
  claimed here.
- HIP compiles the full tree for `gfx90a`; no AMD device is visible, so no HIP
  execution is claimed.
- CUDA example: `N=1e5` 0.58 s, `N=1e6` 2.24 s, `N=1e8` 187.5 s with an exact
  histogram sum of 100,000,000.
- CPU example: `N=1e6` 19.45 s.
- The CPU and CUDA `N=1e6` histograms are identical by direct row diff.
- `hal` reports `sharing=unknown`; the throughput is a shared-host baseline.

Priority legend: **P0** blocks plan acceptance, **P1** required for plan
completion, **P2** hardening or optimization. **Blocked** means an external
resource or decision is missing.

## 1. Core architecture and scheduler

- [ ] **P0: multi-chunk batching.** A consumer currently processes exactly one
  input chunk per batch. Add a configurable batch of ready chunks per node,
  build a compact tile descriptor list over all chunks, run one classify launch
  and one scatter launch per batch, and plan output chunks across the whole
  batch. Add `--batch-chunks Q` (default sane value, `Q=1` fallback). Acceptance:
  Q=1,2,8 produce identical histograms for `C=1024` and `C=8192`; batch-level
  output planning preserves exact fill levels and chunk-boundary crossings;
  host round-trip count drops measurably.
- [ ] **P0: generic graph API.** The current runner is a particle-feedback
  runner with reusable primitives. Add the planned host-side
  `NodeBase`/`Graph`/`GraphBuilder`/`RouteTable` model with typed payload
  contracts, source/transform/sink roles, per-bin routing, feedback edges, and
  no device function pointers. Port the particle example to it. Acceptance:
  an arbitrary small DAG (source, two transforms, sink, one feedback edge) runs
  through the same scheduler without particle-specific code in the core.
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
  Open boundary: the policy default `P=16384` exceeds the measured A30 payload
  budget of 3,171,090,432 B (needs 3,221,225,472 B), so the T13 `N=1e8` run
  used an explicit `P=16128`; the default-policy clamp to the actual device
  budget is in progress, tracked in Section 7.
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

- [ ] **P0: register the example CLI in CTest.** The Catch2 tests cover the
  matrix, but the real `graphFlow_particle_feedback` binary is not a CTest
  entry. Register `N=0,1,17,127,128,4097`, `C=1,31,127,128`, fixed seeds, and
  the feedback cases for CPU and CUDA (HIP when available), with finite
  timeouts, JUnit output, and a failure when a required backend did not run.
- [ ] **P0: automate CPU/CUDA/HIP histogram equality.** The manual `N=1e6`
  CPU/CUDA diff is empty. Make it a test that compares sorted histogram rows for
  the same `(N, seed, C)` on every compiled backend; add HIP when an AMD device
  is available.
- [ ] **P1: device-dependent default `C`/`P`.** Replace the fixed default with
  the device-aware policy from Section 1, print the policy and the chosen
  values, and keep the CLI override.
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
- [ ] **P0: format configuration.** Add the upstream-derived `.clang-format`
  (clang-format 20.1.8), run it over the tree, and make the CI job fail on a
  diff.
- [ ] **P0: REUSE compliance.** Add `LICENSES/MPL-2.0.txt`, `.reuse/dep5` if
  needed, SPDX headers for every source and documentation file, and a
  `reuse lint` gate.
- [ ] **P1: add the plan to the repository.** Copy the approved plan to
  `docs/plan.md` with an SPDX header, or link it explicitly. The repository
  currently has README and `docs/acceptance.md`, but not the plan.
- [ ] **P1: complete the documentation set.** Add `docs/build.md` (exact pins,
  configure/build/test commands, environment scrub), `docs/ci.md`,
  `docs/architecture.md`, `docs/numerics.md`, `docs/known-limitations.md`, and
  `docs/benchmarks.md`. Link them from the README.
- [ ] **P1: complete the alpaka API map.** `docs/alpaka-api-map.md` covers the
  main primitives; add an entry for every new accelerator helper, with searched
  symbols, pinned path, selected API, tested backends, and the reason for any
  project-owned code.
- [ ] **P1: repository policy.** Decide public versus private, enable branch
  protection for `main`, add a CI badge, and document the PR/merge process.
- [ ] **P1: exact pin record.** Record the alpaka commit and Catch2 version used
  for validation in `docs/build.md` and the acceptance record; update the pin
  deliberately and re-run CPU/CUDA/HIP probes.

## 5. Performance and acceptance

- [ ] **P0: CUDA-specific kernel test.** The CUDA build's CTest cases do not
  prove the kernels ran on the GPU; the example `--backend cuda` does. Add a
  CUDA test that runs a real kernel on the device, asserts the selected backend
  and device name, and fails when hardware is missing rather than skipping.
- [ ] **P0: HIP execution gate.** Run the example and the relevant tests on an
  AMD GPU. Record the GPU model, ROCm version, `gfx` target, and results. The
  current `gfx90a` result is compile-only.
- [ ] **P0: exclusive or repeated acceptance run.** `hal` is shared
  (`sharing=unknown`). Re-run `N=1e8` on an exclusive allocation or repeat the
  baseline enough times to report variance; keep the shared-host numbers
  labelled as a baseline.
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

- [ ] **P0: TSAN validation.** The hal kernel aborts TSAN with `unexpected
  memory mapping` even for a non-PIE binary. Run TSAN on a compatible host and
  record the scheduler/pool subset; add a suppression only with evidence and do
  not hide a race.
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
  labels, credentials, and artifact retention. The repository is currently
  private and has no `.github/workflows`.
- **Exclusive throughput:** `hal` is shared; the current 187.5 s is a baseline,
  not an exclusive-hardware acceptance number.
- **Pool memory:** `S=P+K*R` is safe but overprovisioned at `C=8192`,
  `P=65536` (about 12.9 GB). Decide between a slot-based pool, a smaller default
  `P`, or a memory-budget cap.
- **Queue overlap:** the CUDA probe reports no overlap between distinct queues.
  Decide whether to investigate further or document the full-chunk strategy.
- **Policy-default budget boundary:** the policy default `C=8192`, `P=16384`
  exceeds the actual A30 payload budget of 3,171,090,432 B (it needs
  3,221,225,472 B), so the T13 `N=1e8` run had to use an explicit `P=16128`
  (3,170,893,824 B) to pass. The default-policy clamp to the measured device
  budget is in progress; until it lands, an un-overridden CUDA run on this
  device is rejected.

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
  gate is explicitly recorded as blocked.
- [ ] The example CLI is registered in CTest for the full matrix and every
  required backend.
- [ ] `N=1e6` runs in CI; `N=1e8` passes on A30 and AMD with exact histograms.
- [ ] Throughput is measured on an exclusive allocation, frozen, and guarded.
- [ ] ASAN/UBSAN/TSAN pass on a compatible host.
- [ ] Format, REUSE, docs, and the API map are complete and green.
- [ ] The repository is public or explicitly kept private, with branch
  protection and CI configured.
- [ ] An independent review confirms the final integrated state and the
  acceptance record.
