# Benchmark matrix driver

`tools/bench_matrix.sh` runs `graphFlow_particle_feedback` over a repeatable
matrix and writes a machine-readable CSV plus a Markdown summary. The driver
never builds the project, never runs `N = 1e8` unless the `acceptance` or
`fwk394` preset is selected, and never invents a bandwidth number. This
document contains no measured result; every number in a report comes from a real
run of the binary.

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
| `--matrix smoke\|acceptance\|fwk394\|full` | preset, default `smoke` |
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
| `acceptance` | `cuda` | 100000000 | 8192 | 16384 | 1 | 42 | 1800 s | A30 acceptance shape; exactly 3.0 GiB |
| `fwk394` | `cuda` | 100000000 | 4096 | 16384 | 1 | 42 | 1800 s | requested 2080 Ti pair; over budget on 11 GiB, see below |
| `full` | `cpu,cuda` | 100000, 1000000 | 1024, 8192 | 2048, 8192 | 1, 2, 8 | 42, 7 | 900 s | long sweep; over-budget cases are recorded as negatives |

## Payload budget (T7)

T7 derives the accelerator policy budget as `min(device memory / 8, 3 GiB)`.
`makeDevicePolicy` rejects a pair whose `P*C*24 bytes` exceeds that budget with:

```text
graphFlow::DevicePolicy: liveParticles and chunkCapacity exceed the payload memory budget
```

The example then adds the `K*R` output-credit reserve to the pool budget, and
`Config::validate` checks `S*C*24` against it. A pool overrun uses the
`graphFlow::Config: ... exceeding the payload budget of ... bytes` message. The
example prints either message after `graphFlow_particle_feedback: ` and exits
2. The driver records `status=fail`, `exit_status=2`, and the first stderr line
in the CSV `failure` column; the full text stays in `logs/<case>.err`. A matrix
that contains a negative case exits 1 at the end.

| Host | Policy budget | Pair | Payload | Result |
| --- | ---: | --- | ---: | --- |
| A30 24 GiB | 3 GiB | C=8192, P=16384 | 3 GiB | passes at the boundary |
| A30 24 GiB | 3 GiB | C=8192, P=65536 | ~12.9 GB | rejected, canonical negative case |
| fwk394 11 GiB | 1.375 GiB | C=4096, P=16384 | 1.5 GiB | rejected |
| fwk394 11 GiB | 1.375 GiB | C=4096, P=8192 | 0.75 GiB | passes |

`--matrix acceptance` is the A30 pair. The former `P=65536` pair remains
reachable through `--live-particles 65536`; run it deliberately to record the
rejection:

```sh
mkdir -p /tmp/gf-bench-negative
env CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-cuda/graphFlow_particle_feedback \
    --backend cuda --device 0 --matrix acceptance \
    --live-particles 65536 --out /tmp/gf-bench-negative
```

Expect the case to exit 2 with the DevicePolicy message and the driver to exit
1. That is a successful negative test, not a working-set result.

The CPU backend has a 16 MiB host policy budget. GPU-sized explicit pairs in
`full` therefore fail on CPU and are recorded as T7 negatives. Use
`--backend cuda` on a device whose 1/8 budget covers the pair, or narrow the
C/P lists, for an all-pass sweep.

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
the report is complete, marks each failed case, and exits `1` at the end. A
nonzero example exit appends the first stderr line to the `failure` column, so
a T7 budget rejection carries the DevicePolicy message in the CSV. The example
exits `2` for invalid arguments including that rejection; driver-level usage
and path errors also exit `2`, but before any case runs.

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

T7 gives this card a 1.375 GiB policy budget. The `fwk394` preset is
C=4096, P=16384 (about 1.6 GB, 1.5 GiB), so that pair is the documented
over-budget negative case on this card. The passing 1e8 run overrides P to 8192
(about 0.75 GiB). Neither value is the A30 acceptance configuration, so label
any fwk394 report accordingly.

```sh
cd <graphFlow checkout>
cmake -S . -B build-fwk394 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGRAPHFLOW_ALPAKA_SOURCE_DIR=/workspace \
  -DGRAPHFLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGRAPHFLOW_BUILD_TESTS=OFF
cmake --build build-fwk394 -j 14

# pick an idle GPU from nvidia-smi; this example uses index 0
mkdir -p /tmp/gf-bench-fwk394-smoke
env -u CUDA_LAUNCH_BLOCKING -u CUDA_DEVICE_MAX_CONNECTIONS \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-fwk394/graphFlow_particle_feedback \
    --backend cuda --device 0 \
    --matrix smoke --out /tmp/gf-bench-fwk394-smoke

# requested fwk394 pair: expected T7 negative case on 11 GiB
mkdir -p /tmp/gf-bench-fwk394-negative
env -u CUDA_LAUNCH_BLOCKING -u CUDA_DEVICE_MAX_CONNECTIONS \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-fwk394/graphFlow_particle_feedback \
    --backend cuda --device 0 \
    --matrix fwk394 --out /tmp/gf-bench-fwk394-negative

# passing 1e8 run on 11 GiB: same C, P reduced to 8192
mkdir -p /tmp/gf-bench-fwk394
env -u CUDA_LAUNCH_BLOCKING -u CUDA_DEVICE_MAX_CONNECTIONS \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash tools/bench_matrix.sh \
    --binary build-fwk394/graphFlow_particle_feedback \
    --backend cuda --device 0 \
    --matrix fwk394 --live-particles 8192 \
    --out /tmp/gf-bench-fwk394
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
`Q = 1, 2, 8` execute the same path and a Q sweep only repeats work. `Q > 1`
rows are not batching evidence yet. A `rows_equal=yes` across Q values proves
that the flag is accepted; it does not prove batch-level planning, output fill
levels, or exact chunk-boundary crossings. The pair key keeps Q so the same driver becomes meaningful once T5
wires the value into the scheduler.
