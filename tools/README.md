# Benchmark matrix driver

`tools/bench_matrix.sh` runs `graphFlow_particle_feedback` over a repeatable
matrix and writes a machine-readable CSV plus a Markdown summary. The driver
never builds the project, never runs `N = 1e8` unless the `acceptance` preset is
selected, and never invents a bandwidth number. This document contains no
measured result; every number in a report comes from a real run of the binary.

## Requirements

- bash 4.3 or newer (`local -n`, associative arrays)
- GNU coreutils `timeout` (a missing `timeout` is recorded and the limit is
  skipped)
- `/usr/bin/time` for wall time and max RSS; the driver falls back to shell
  `SECONDS` and reports `NA` for RSS when it is absent
- an already built `graphFlow_particle_feedback` and an already existing output
  directory. The driver fails if either is missing and does not create the
  output directory.

## Options

| Option | Meaning |
| --- | --- |
| `--binary PATH` | example executable, required |
| `--out DIR` | existing writable output directory, required |
| `--backend LIST` | `cpu`, `cuda`, `hip`, or a comma-separated list |
| `--device N` | device index passed to every run (default `0`) |
| `--n LIST` | particle counts `N` |
| `--chunk-sizes LIST` | chunk sizes `C` |
| `--live-particles LIST` | source-admission capacities `P` |
| `--batch-chunks LIST` | `Q` values |
| `--seeds LIST` | seeds `S` (also accepts `0`) |
| `--matrix smoke\|acceptance\|full` | preset, default `smoke` |
| `--timeout S` | per-run wall-clock limit in seconds |
| `--copy-reference [CMD]` | optional copy-bandwidth hook, see below |

Every list overrides one dimension of the selected preset. A comma-separated
`--backend` list is the normal way to produce CPU and CUDA runs in one report,
so the histogram comparison is computed automatically. Lists can be repeated
across dimensions; the driver runs the full Cartesian product.

## Presets

| Preset | Backends | N | C | P | Q | Seeds | Timeout | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke` | `cpu` | 4097, 100000 | 128 | 1024 | 1 | 42 | 120 s | default; small and local-safe |
| `acceptance` | `cuda` | 100000000 | 8192 | 65536 | 1 | 42 | 1800 s | the documented A30 acceptance shape; never the default |
| `full` | `cpu,cuda` | 100000, 1000000 | 1024, 8192 | 2048, 8192 | 1, 2, 8 | 42, 7 | 900 s | long sweep; stays below the acceptance working set |

The `acceptance` preset is sized for a 24 GiB A30. The pool bound is
`S = P + K*R` with `K = 3` and `R = 2`, so `C = 8192, P = 65536` allocates
about 12.9 GB of payload storage. That does not fit an 11 GiB RTX 2080 Ti.
Reduce `P` and `C` on that card, or run `smoke`. The `full` preset stops at
`P = 8192` so its largest pool is about 1.6 GB; it never uses the `P = 65536`
acceptance working set. A CPU-only build needs `--backend cpu` with `full`,
because the preset also schedules CUDA cases.

## Outputs

| Path | Content |
| --- | --- |
| `results.csv` | one row per case, see schema below |
| `summary.md` | Markdown case table, pair comparison, failure list |
| `logs/<case>.log` | full example stdout for the case |
| `logs/<case>.err` | example stderr |
| `logs/<case>.time` | `/usr/bin/time` output |
| `histograms/<case>.csv` | the case histogram rows, sorted by weight |
| `histograms/<case>.raw.csv` | rows in emission order |
| `pairs/<key>.diff` | `diff -u` of a CPU/CUDA pair whose rows differ |
| `results.tsv`, `pairs.tsv`, `pair-results.tsv` | driver-internal tables |
| `environment.txt` | host, kernel, binary SHA-256, git commit, timer tool, GPU list, scrub record |
| `copy-reference.txt` | raw output of `--copy-reference CMD`, when supplied |

CSV columns:

```text
case,backend,selected_backend,device,n,chunk_size,live_particles,batch_chunks,seed,
status,failure,exit_status,wall_s,max_rss_kb,particles_per_s,pool_chunks,
peak_live_chunks,peak_reserved_chunks,histogram_rows,histogram_sum,rows_equal,policy,log
```

Wall time is the whole process, including device init and the queue probe.
`particles_per_s = N / wall_s`. `max_rss_kb` comes from `/usr/bin/time` `%M`.
A case fails on a nonzero exit, a missing `# selected-backend=` header, a
missing `# pool-chunks=` header, a missing `# weight,count` header, or a
histogram sum that differs from `N`. The driver continues through the matrix so
the report is complete, marks each failed case, and exits `1` at the end. Exit
`2` is a usage, path, or validation error.

Each case is checked for the selected backend, the pool counters, the policy
text, and the histogram evidence before it can be marked `ok`.

## CPU/CUDA comparison

For a key `(N, C, seed, Q)` with exactly one successful CPU run and one
successful CUDA run, the driver sorts both histogram files and compares them
byte for byte. It records `rows_equal` as `yes` or `no` and writes a unified
diff on a mismatch. A key without a counterpart is `-`. A key with several runs
on either side is `ambiguous`; this happens in `full` because `P` is swept while
the contract key deliberately leaves `P` out. Only successful runs enter the
comparison.

## Copy-bandwidth reference

`--copy-reference CMD` runs `CMD` under the same timeout, stores its raw output
in `copy-reference.txt`, and records the exit status in `environment.txt`. No
number is parsed from that output. With a bare `--copy-reference`, the report
only lists the documented commands. A copy-reference failure never changes the
benchmark verdict, and the report never claims a bandwidth value.

Documented commands, to be used only when the tool is installed:

```sh
nvbandwidth -t device_to_device_memcpy_read_ce
bandwidthTest --device=0 --mode=quick
```

## hal (A30, sm_80)

```sh
cd <graphFlow checkout>
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGRAPHFLOW_BUILD_TESTS=OFF
cmake --build build-cuda -j 14

mkdir -p /tmp/gf-bench-hal
env -u CUDA_LAUNCH_BLOCKING -u CUDA_DEVICE_MAX_CONNECTIONS \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-cuda/graphFlow_particle_feedback \
    --backend cuda --device 0 \
    --matrix acceptance --timeout 1800 \
    --out /tmp/gf-bench-hal

# cheap check of the same build
env CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-cuda/graphFlow_particle_feedback \
    --backend cuda --matrix smoke --out /tmp/gf-bench-hal-smoke
```

## fwk394 (RTX 2080 Ti, sm_75, 11 GiB)

The default acceptance working set does not fit 11 GiB. Use the smoke preset,
or a 1e8 run with a reduced working set as shown. The reduced run is not the
A30 acceptance configuration and must be labelled as such in any report.

```sh
cd <graphFlow checkout>
cmake -S . -B build-fwk394 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGRAPHFLOW_BUILD_TESTS=OFF
cmake --build build-fwk394 -j 14

mkdir -p /tmp/gf-bench-fwk394
# pick an idle GPU from nvidia-smi; this example uses index 0
env -u CUDA_LAUNCH_BLOCKING -u CUDA_DEVICE_MAX_CONNECTIONS \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-fwk394/graphFlow_particle_feedback \
    --backend cuda --device 0 \
    --matrix smoke --out /tmp/gf-bench-fwk394-smoke

# 1e8 with a working set that fits 11 GiB (about 1.6 GB payload)
env -u CUDA_LAUNCH_BLOCKING -u CUDA_DEVICE_MAX_CONNECTIONS \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-fwk394/graphFlow_particle_feedback \
    --backend cuda --device 0 --matrix acceptance \
    --chunk-sizes 4096 --live-particles 16384 \
    --timeout 1800 --out /tmp/gf-bench-fwk394
```

`CUDA_VISIBLE_DEVICES` is the only supported device selector. Set it to the
idle card's index as reported after the ordering is pinned, and pass the same
logical index to `--device`. The driver sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` for
every child and forwards `CUDA_VISIBLE_DEVICES` unchanged.

## Environment scrub

Each child runs through `env` with these variables removed, because any one of
them can serialise kernels or change thread placement:

```text
CUDA_LAUNCH_BLOCKING
CUDA_DEVICE_MAX_CONNECTIONS
CUDA_INJECTION64_PATH
OMP_NUM_THREADS
OMP_PROC_BIND
OMP_PLACES
KMP_AFFINITY
MKL_NUM_THREADS
```

The driver exports `LC_ALL=C` and `LANG=C` for itself and every child so
`/usr/bin/time`, `sort`, and `awk` use a dot decimal. The
outer `env -u ...` in the commands above is belt and braces for the shell that
starts the driver; the driver repeats the scrub for each run. `nvidia-smi` is
queried once, outside the timed region, and its output is stored in
`environment.txt`.

## Shared-host caveat

`hal` reports `sharing=unknown`, and fwk394 may host other jobs. Wall time
includes device initialisation and the queue probe, and neighbouring tenants,
clocks, thermals, and MIG partitions can move it. Compare repeats from the same
host and the same environment, report the median and spread, and never present a
single shared-host run as an exclusive-hardware acceptance result. A report
generated on a shared host is a recorded baseline.

## Batching dependency (T5)

The example parses `--batch-chunks Q` but the current runner still processes one
chunk per batch (`Cli.hpp` and `ParticleStages.hpp` say so). Until T5 lands,
`Q = 1, 2, 8` execute the same path and a Q sweep only repeats work. A
`rows_equal=yes` across Q values proves that the flag is accepted; it does not
prove batch-level planning, output fill levels, or exact chunk-boundary
crossings. The pair key keeps Q so the same driver becomes meaningful once T5
wires the value into the scheduler.
