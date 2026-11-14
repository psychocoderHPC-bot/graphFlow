#!/usr/bin/env bash
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Reproducible benchmark driver for graphFlow_particle_feedback.
#
# The driver never runs the example at build time and never invents a bandwidth
# number. It records exactly what the binary printed and exits nonzero when a
# case fails. See tools/README.md for the hal and fwk394 invocations.

set -euo pipefail

# Deterministic number formatting, sorting, and /usr/bin/time output.
export LC_ALL=C
export LANG=C

readonly PROG="bench_matrix"

die()
{
    printf '%s: error: %s\n' "$PROG" "$*" >&2
    exit 2
}

usage()
{
    cat <<'EOF'
usage: tools/bench_matrix.sh --binary PATH --out DIR [options]

required:
  --binary PATH            graphFlow_particle_feedback executable
  --out DIR                existing, writable output directory

run selection:
  --backend LIST           cpu,cuda,hip; comma- or space-separated (default: preset)
  --device N               device index passed to every run (default: 0)
  --n LIST                 particle counts N (comma-separated)
  --chunk-sizes LIST       chunk sizes C
  --live-particles LIST    source-admission capacities P
  --batch-chunks LIST      Q values
  --seeds LIST             seeds S
  --reps R                 independent repetitions per case (default: preset,
                           p07 uses 3); the report adds median and IQR
  --matrix PRESET          smoke (default), acceptance, fwk394, full, or p07
  --timeout S              per-run wall-clock limit in seconds (default: preset)
  --kernel-tests PATH      optional graphFlow_tests binary; runs the committed
                           scatter and dense/hash integration cases and records
                           their Catch2 durations
  --self-test, --dry-run   run the harness self-test and exit

frozen target:
  --target-pps PPS         nominal particles/s regression target
  --target-n N             target particle count (default 100000000)
  --target-c C             target chunk size (default 8192)
  --target-q Q             target batch depth (default 8)
  --target-tolerance PCT   allowed shortfall in percent (default 10)
                           The verdict is PASS, FAIL, or BLOCKED when no
                           matching successful case exists.

reference hook:
  --copy-reference [CMD]   run CMD and capture its output; with no CMD run a
                           generated sustained device-to-device copy reference
                           at the largest selected payload, or report BLOCKED
                           when no CUDA compiler is available

Every list may be overridden independently of the preset. Numeric LIST values
are positive integers, except --seeds, which also accepts 0. --backend accepts
backend names.
EOF
}

# Internal tables use an ASCII unit separator instead of a tab. A tab is IFS
# whitespace, so `read` collapses empty fields and shifts later columns.
SEP=$'\x1f'

tsv_join()
{
    local -a fields=("$@")
    local out="" i field
    for i in "${!fields[@]}"; do
        field="${fields[i]}"
        field="${field//$SEP/ }"
        if (( i > 0 )); then
            out+="$SEP"
        fi
        out+="$field"
    done
    printf '%s\n' "$out"
}

csv_quote()
{
    local text="${1-}"
    text="${text//\"/\"\"}"
    printf '"%s"' "$text"
}

# Read newline-separated numbers from stdin and print "median iqr" with six
# decimals. The quartiles use inclusive halves (the lower and upper halves both
# contain the median for an odd count), and the sort is a local insertion sort
# so this does not depend on gawk's asort().
quantile_stats()
{
    awk '
        function median(a, count,    mid) {
            if (count == 0) return 0
            mid = int((count + 1) / 2)
            if (count % 2 == 1) return a[mid]
            return (a[count / 2] + a[count / 2 + 1]) / 2
        }
        { values[++n] = $1 + 0 }
        END {
            for (i = 2; i <= n; ++i) {
                v = values[i]
                j = i - 1
                while (j >= 1 && values[j] > v) {
                    values[j + 1] = values[j]
                    --j
                }
                values[j + 1] = v
            }
            med = median(values, n)
            if (n >= 2) {
                half = int((n + 1) / 2)
                for (i = 1; i <= half; ++i)
                    lower[i] = values[i]
                start = n - half + 1
                for (i = start; i <= n; ++i)
                    upper[i - start + 1] = values[i]
                iqr = median(upper, half) - median(lower, half)
            } else {
                iqr = 0
            }
            printf "%.6f %.6f\n", med, iqr
        }
    '
}

# Backend names accept a comma- or space-separated list. This must not go
# through split_list(), which validates unsigned integers.
split_backends()
{
    local input="$1" value
    IFS=', ' read -r -a SPLIT_VALUES <<< "$input" || true
    (( ${#SPLIT_VALUES[@]} > 0 )) || die "--backend list is empty"
    for value in "${SPLIT_VALUES[@]}"; do
        [[ -n "$value" ]] || die "--backend list '$input' contains an empty element"
        case "$value" in
            cpu|cuda|hip) ;;
            *) die "--backend accepts cpu, cuda, or hip (got '$value')" ;;
        esac
    done
}

# Main-path backend initialisation: split, validate, and de-duplicate in order.
init_backends()
{
    local input="$1" value seen duplicate
    split_backends "$input"
    backends=()
    for value in "${SPLIT_VALUES[@]}"; do
        duplicate="0"
        for seen in "${backends[@]:-}"; do
            [[ "$seen" == "$value" ]] && duplicate="1"
        done
        [[ "$duplicate" == "1" ]] || backends+=("$value")
    done
}

run_self_test()
{
    local -a expected fields
    local line

    init_backends "cpu,cuda hip"
    expected=(cpu cuda hip)
    if [[ "${backends[*]}" != "${expected[*]}" ]]; then
        printf 'self-test FAIL: backend comma/space split, got [%s]\n' "${backends[*]}" >&2
        return 1
    fi

    init_backends "cuda"
    expected=(cuda)
    if [[ "${backends[*]}" != "${expected[*]}" ]]; then
        printf 'self-test FAIL: single backend, got [%s]\n' "${backends[*]}" >&2
        return 1
    fi

    init_backends "cpu,cuda cpu"
    expected=(cpu cuda)
    if [[ "${backends[*]}" != "${expected[*]}" ]]; then
        printf 'self-test FAIL: backend de-duplication, got [%s]\n' "${backends[*]}" >&2
        return 1
    fi

    # 24 fields with the failure field (index 11) empty. A tab-delimited reader
    # would drop that empty field and shift every later column.
    line="$(tsv_join case_0001 cpu cpu 0 100000 128 1024 1 42 1 ok "" 0 1.23 456 789 1030 12 6 90 100000 "C=128 P=1024" /tmp/log /tmp/hist)"
    IFS="$SEP" read -r -a fields <<< "$line"
    if (( ${#fields[@]} != 24 )); then
        printf 'self-test FAIL: expected 24 TSV fields, got %d\n' "${#fields[@]}" >&2
        return 1
    fi
    if [[ "${fields[9]}" != "1" ]]; then
        printf 'self-test FAIL: rep field is [%s]\n' "${fields[9]}" >&2
        return 1
    fi
    if [[ -n "${fields[11]}" ]]; then
        printf 'self-test FAIL: empty failure field became [%s]\n' "${fields[11]}" >&2
        return 1
    fi
    if [[ "${fields[12]}" != "0" ]]; then
        printf 'self-test FAIL: exit_status shifted to [%s]\n' "${fields[12]}" >&2
        return 1
    fi
    if [[ "${fields[21]}" != "C=128 P=1024" || "${fields[22]}" != "/tmp/log" || "${fields[23]}" != "/tmp/hist" ]]; then
        printf 'self-test FAIL: tail columns shifted after empty failure\n' >&2
        return 1
    fi
    if [[ "$(csv_quote "${fields[11]}")" != '""' ]]; then
        printf 'self-test FAIL: empty failure is not quoted empty in CSV\n' >&2
        return 1
    fi

    # Median and Tukey-hinge IQR of 1..5 are 3 and 4-2=2.
    stats="$(printf '1\n2\n3\n4\n5\n' | quantile_stats)"
    if [[ "$stats" != "3.000000 2.000000" ]]; then
        printf 'self-test FAIL: median/IQR of 1..5 is [%s]\n' "$stats" >&2
        return 1
    fi

    printf 'self-test: PASS (backend split; 24 TSV columns; empty failure preserved; CSV quoting; median/IQR)\n'
}

matrix="smoke"
binary=""
out_dir=""
device="0"
backend_list=""
n_list=""
c_list=""
p_list=""
q_list=""
seeds_list=""
timeout_s=""
reps="1"
kernel_tests=""
target_pps=""
target_n="100000000"
target_c="8192"
target_q="8"
target_tolerance="10"
copy_reference_set="0"
copy_reference=""
self_test="0"

backend_set="0"
device_set="0"
n_set="0"
c_set="0"
p_set="0"
q_set="0"
seeds_set="0"
timeout_set="0"
reps_set="0"

while (( $# > 0 )); do
    arg="$1"
    shift
    inline="0"
    case "$arg" in
        --*=*)
            opt="${arg%%=*}"
            value="${arg#*=}"
            inline="1"
            ;;
        *)
            opt="$arg"
            value=""
            ;;
    esac

    case "$opt" in
        --help|-h)
            usage
            exit 0
            ;;
        --self-test|--dry-run)
            self_test="1"
            continue
            ;;
        --binary|--out|--backend|--device|--n|--chunk-sizes|--live-particles|--batch-chunks|--seeds|--reps|--matrix|--timeout|--kernel-tests|--target-pps|--target-n|--target-c|--target-q|--target-tolerance)
            if [[ "$inline" == "0" ]]; then
                if (( $# == 0 )); then
                    die "$opt requires a value"
                fi
                value="$1"
                shift
            fi
            ;;
        --copy-reference)
            if [[ "$inline" == "0" ]]; then
                if (( $# > 0 )) && [[ "$1" != -* ]]; then
                    value="$1"
                    shift
                else
                    value=""
                fi
            fi
            copy_reference_set="1"
            copy_reference="$value"
            continue
            ;;
        *)
            die "unknown option '$opt' (try --help)"
            ;;
    esac

    case "$opt" in
        --binary) binary="$value" ;;
        --out) out_dir="$value" ;;
        --backend) backend_list="$value"; backend_set="1" ;;
        --device) device="$value"; device_set="1" ;;
        --n) n_list="$value"; n_set="1" ;;
        --chunk-sizes) c_list="$value"; c_set="1" ;;
        --live-particles) p_list="$value"; p_set="1" ;;
        --batch-chunks) q_list="$value"; q_set="1" ;;
        --seeds) seeds_list="$value"; seeds_set="1" ;;
        --reps) reps="$value"; reps_set="1" ;;
        --matrix) matrix="$value" ;;
        --timeout) timeout_s="$value"; timeout_set="1" ;;
        --kernel-tests) kernel_tests="$value" ;;
        --target-pps) target_pps="$value" ;;
        --target-n) target_n="$value" ;;
        --target-c) target_c="$value" ;;
        --target-q) target_q="$value" ;;
        --target-tolerance) target_tolerance="$value" ;;
    esac
done

if [[ "$self_test" == "1" ]]; then
    run_self_test
    exit 0
fi

[[ -n "$binary" ]] || die "missing required --binary PATH"
[[ -n "$out_dir" ]] || die "missing required --out DIR"

case "$matrix" in
    smoke|acceptance|fwk394|full|p07) ;;
    *) die "--matrix must be smoke, acceptance, fwk394, full, or p07 (got '$matrix')" ;;
esac

# Presets only fill values the caller did not override. Smoke is deliberately
# small; acceptance is the one documented 1e8 run and is never the default.
case "$matrix" in
    smoke)
        [[ "$backend_set" == "1" ]] || backend_list="cpu"
        [[ "$n_set" == "1" ]] || n_list="4097,100000"
        [[ "$c_set" == "1" ]] || c_list="128"
        [[ "$p_set" == "1" ]] || p_list="1024"
        [[ "$q_set" == "1" ]] || q_list="1"
        [[ "$seeds_set" == "1" ]] || seeds_list="42"
        [[ "$timeout_set" == "1" ]] || timeout_s="120"
        ;;
    acceptance)
        # Measured A30 usable budget: 3,171,090,432 B (24,192 MiB usable).
        # 8192 * 16128 * 24 B == 3,170,893,824 B, the largest 128-aligned
        # value under that budget. P=16384 is rejected on this device.
        [[ "$backend_set" == "1" ]] || backend_list="cuda"
        [[ "$n_set" == "1" ]] || n_list="100000000"
        [[ "$c_set" == "1" ]] || c_list="8192"
        [[ "$p_set" == "1" ]] || p_list="16128"
        [[ "$q_set" == "1" ]] || q_list="1"
        [[ "$seeds_set" == "1" ]] || seeds_list="42"
        [[ "$timeout_set" == "1" ]] || timeout_s="1800"
        ;;
    fwk394)
        # Requested 2080 Ti pair. On an 11 GiB card T7's 1/8 policy budget is
        # 1.375 GiB, so this ~1.5 GiB pair is the documented GPU negative case.
        # C=4096/P=8192 (768 MiB) fits the GPU budget but the sm_75 CUDA build
        # is blocked by the nvcc 12.5 ICE. On fwk394 CPU the 16 MiB host budget
        # rejects C=4096/P=8192; the passing CPU pair is C=1024/P=512.
        [[ "$backend_set" == "1" ]] || backend_list="cuda"
        [[ "$n_set" == "1" ]] || n_list="100000000"
        [[ "$c_set" == "1" ]] || c_list="4096"
        [[ "$p_set" == "1" ]] || p_list="16384"
        [[ "$q_set" == "1" ]] || q_list="1"
        [[ "$seeds_set" == "1" ]] || seeds_list="42"
        [[ "$timeout_set" == "1" ]] || timeout_s="1800"
        ;;
    full)
        [[ "$backend_set" == "1" ]] || backend_list="cpu,cuda"
        [[ "$n_set" == "1" ]] || n_list="100000,1000000"
        [[ "$c_set" == "1" ]] || c_list="1024,8192"
        [[ "$p_set" == "1" ]] || p_list="2048,8192"
        [[ "$q_set" == "1" ]] || q_list="1,2,8"
        [[ "$seeds_set" == "1" ]] || seeds_list="42,7"
        [[ "$timeout_set" == "1" ]] || timeout_s="900"
        ;;
    p07)
        # P07 target/Q matrix. P=16384/C=8192 is exactly the 3 GiB accelerator
        # cap (3,221,225,472 B) and fits the V100 SXM2-32GB. The C-wave sweep
        # is a separate invocation with --chunk-sizes 2048,4096,8192
        # --live-particles 8192, which keeps P/C at 4, 2, and 1 waves.
        [[ "$backend_set" == "1" ]] || backend_list="cuda"
        [[ "$n_set" == "1" ]] || n_list="100000000"
        [[ "$c_set" == "1" ]] || c_list="8192"
        [[ "$p_set" == "1" ]] || p_list="16384"
        [[ "$q_set" == "1" ]] || q_list="1,2,8"
        [[ "$seeds_set" == "1" ]] || seeds_list="42"
        [[ "$timeout_set" == "1" ]] || timeout_s="1800"
        [[ "$reps_set" == "1" ]] || reps="3"
        ;;
esac

[[ "$device" =~ ^[0-9]+$ ]] || die "--device must be an unsigned integer (got '$device')"
[[ "$timeout_s" =~ ^[0-9]+$ ]] || die "--timeout must be an unsigned integer (got '$timeout_s')"
(( 10#$timeout_s > 0 )) || die "--timeout must be greater than zero"
[[ "$reps" =~ ^[0-9]+$ ]] || die "--reps must be an unsigned integer (got '$reps')"
(( 10#$reps > 0 )) || die "--reps must be greater than zero"
for target_value in "$target_n" "$target_c" "$target_q" "$target_tolerance"; do
    [[ "$target_value" =~ ^[0-9]+$ ]] || die "target values must be unsigned integers (got '$target_value')"
done
if [[ -n "$target_pps" ]]; then
    [[ "$target_pps" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--target-pps must be a positive number (got '$target_pps')"
fi
(( 10#$target_tolerance <= 100 )) || die "--target-tolerance must be at most 100"

SPLIT_VALUES=()
split_list()
{
    local name="$1" input="$2" allow_zero="$3" value
    IFS=',' read -r -a SPLIT_VALUES <<< "$input" || true
    (( ${#SPLIT_VALUES[@]} > 0 )) || die "$name list is empty"
    for value in "${SPLIT_VALUES[@]}"; do
        [[ -n "$value" ]] || die "$name list '$input' contains an empty element"
        [[ "$value" =~ ^[0-9]+$ ]] || die "$name value '$value' is not an unsigned integer"
        if [[ "$allow_zero" != "1" && "$value" =~ ^0+$ ]]; then
            die "$name value must be greater than zero"
        fi
    done
}

init_backends "$backend_list"

split_list "--n" "$n_list" "0"
ns=("${SPLIT_VALUES[@]}")
split_list "--chunk-sizes" "$c_list" "0"
cs=("${SPLIT_VALUES[@]}")
split_list "--live-particles" "$p_list" "0"
ps=("${SPLIT_VALUES[@]}")
split_list "--batch-chunks" "$q_list" "0"
qs=("${SPLIT_VALUES[@]}")
split_list "--seeds" "$seeds_list" "1"
seeds=("${SPLIT_VALUES[@]}")

# Canonical decimal values keep the CSV and the histogram-sum comparison simple.
canonicalise()
{
    local -n target="$1"
    local i
    for i in "${!target[@]}"; do
        target[i]="$(( 10#${target[i]} ))"
    done
}
canonicalise ns
canonicalise cs
canonicalise ps
canonicalise qs
canonicalise seeds
device="$(( 10#$device ))"
timeout_s="$(( 10#$timeout_s ))"
reps="$(( 10#$reps ))"
target_n="$(( 10#$target_n ))"
target_c="$(( 10#$target_c ))"
target_q="$(( 10#$target_q ))"
target_tolerance="$(( 10#$target_tolerance ))"

# Path checks. Both are required inputs: the driver never guesses a build tree
# and never creates the output directory behind the operator's back.
if [[ ! -e "$binary" ]]; then
    die "binary '$binary' does not exist"
fi
if [[ ! -f "$binary" ]]; then
    die "binary '$binary' is not a regular file"
fi
if [[ ! -x "$binary" ]]; then
    die "binary '$binary' is not executable"
fi
case "$binary" in
    /*) ;;
    *) binary="$PWD/$binary" ;;
esac
if [[ ! -d "$out_dir" ]]; then
    die "output directory '$out_dir' does not exist (create it first)"
fi
if [[ ! -w "$out_dir" ]]; then
    die "output directory '$out_dir' is not writable"
fi
case "$out_dir" in
    /*) ;;
    *) out_dir="$PWD/$out_dir" ;;
esac

log_dir="$out_dir/logs"
hist_dir="$out_dir/histograms"
pair_dir="$out_dir/pairs"
mkdir -p "$log_dir" "$hist_dir" "$pair_dir"

results_tsv="$out_dir/results.tsv"
pairs_tsv="$out_dir/pairs.tsv"
pair_results_tsv="$out_dir/pair-results.tsv"
csv_path="$out_dir/results.csv"
summary_path="$out_dir/summary.md"
env_path="$out_dir/environment.txt"
: > "$results_tsv"
: > "$pairs_tsv"
: > "$pair_results_tsv"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"
git_commit="unknown"
if [[ -n "$repo_dir" ]]; then
    git_commit="$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
binary_sha="unavailable"
if command -v sha256sum >/dev/null 2>&1; then
    binary_sha="$(sha256sum "$binary" | awk '{print $1}')"
fi
have_time="0"
if [[ -x /usr/bin/time ]]; then
    have_time="1"
fi
have_timeout="0"
if command -v timeout >/dev/null 2>&1; then
    have_timeout="1"
fi

# Environment recorded once. Per-run variables are scrubbed below so an
# inherited tuning knob cannot silently change one case.
{
    printf 'generated: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host: %s\n' "$(uname -n)"
    printf 'kernel: %s\n' "$(uname -sr)"
    printf 'machine: %s\n' "$(uname -m)"
    printf 'matrix: %s\n' "$matrix"
    printf 'binary: %s\n' "$binary"
    printf 'binary_sha256: %s\n' "$binary_sha"
    printf 'git_commit: %s\n' "$git_commit"
    printf 'git_branch: %s\n' "$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    printf 'wall_timer: %s\n' "$([[ "$have_time" == "1" ]] && echo /usr/bin/time || echo shell-seconds-fallback)"
    printf 'timeout_tool: %s\n' "$([[ "$have_timeout" == "1" ]] && echo timeout || echo unavailable)"
    printf 'per_run_cuda_device_order: PCI_BUS_ID\n'
    printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
} > "$env_path"

if command -v nvidia-smi >/dev/null 2>&1; then
    {
        printf '\nnvidia-smi:\n'
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
    } >> "$env_path"
fi

# Child environment: drop knobs that serialise or reorder execution, and pin
# the device ordering so --device means the same card on every host.
run_env=(env)
for scrub in CUDA_LAUNCH_BLOCKING CUDA_DEVICE_MAX_CONNECTIONS CUDA_INJECTION64_PATH \
             OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY MKL_NUM_THREADS; do
    run_env+=(-u "$scrub")
done
run_env+=("CUDA_DEVICE_ORDER=PCI_BUS_ID" "LC_ALL=C" "LANG=C")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    run_env+=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

case_failed=0
case_index=0

run_case()
{
    local backend="$1" n="$2" c="$3" p="$4" q="$5" seed="$6" rep="$7"
    local name log err timing hist_raw hist_sorted
    local rc="0" wall="NA" maxrss="NA" pps="NA" timing_line=""
    local selected="NA" policy="NA" pool="NA" peak_live="NA" peak_reserved="NA"
    local hist_rows="0" hist_sum="0" status failure stderr_first=""
    local has_hist="0" pool_line=""

    case_index=$(( case_index + 1 ))
    name="$(printf 'case_%04d_%s_n%s_c%s_p%s_q%s_s%s_r%s' "$case_index" "$backend" "$n" "$c" "$p" "$q" "$seed" "$rep")"
    log="$log_dir/$name.log"
    err="$log_dir/$name.err"
    timing="$log_dir/$name.time"
    hist_raw="$hist_dir/$name.raw.csv"
    hist_sorted="$hist_dir/$name.csv"

    local -a binary_cmd=("$binary" --backend "$backend" --device "$device" --particles "$n" \
        --chunk-size "$c" --live-particles "$p" --batch-chunks "$q" --seed "$seed")
    local -a child=()

    : > "$timing"
    if [[ "$have_time" == "1" ]]; then
        # Time runs closest to the binary so %M is the example's RSS. timeout
        # wraps time and kills the whole process group on expiry; a timed-out
        # case is failed and may then have an empty timing file.
        child=(/usr/bin/time -f '%e %M' -o "$timing" "${binary_cmd[@]}")
        if [[ "$have_timeout" == "1" ]]; then
            child=(timeout --signal=TERM "${timeout_s}s" "${child[@]}")
        fi
        child=("${run_env[@]}" "${child[@]}")
        LC_ALL=C "${child[@]}" > "$log" 2> "$err" || rc=$?
        # GNU time may add a status line to the timing file; take the last
        # purely numeric "seconds rss-kib" line.
        timing_line="$(awk '/^[0-9]+([.][0-9]+)?[[:space:]]+[0-9]+$/{line=$0} END{print line}' "$timing")"
        if [[ -n "$timing_line" ]]; then
            read -r wall maxrss <<< "$timing_line"
        fi
    else
        child=("${binary_cmd[@]}")
        if [[ "$have_timeout" == "1" ]]; then
            child=(timeout --signal=TERM "${timeout_s}s" "${child[@]}")
        fi
        child=("${run_env[@]}" "${child[@]}")
        local started="$SECONDS"
        "${child[@]}" > "$log" 2> "$err" || rc=$?
        wall="$(( SECONDS - started ))"
    fi
    [[ -n "$wall" ]] || wall="NA"
    [[ -n "$maxrss" ]] || maxrss="NA"

    selected="$(grep -m1 '^# selected-backend=' "$log" 2>/dev/null || true)"
    if [[ -n "$selected" ]]; then
        selected="${selected#\# selected-backend=}"
        selected="${selected%% *}"
    else
        selected="NA"
    fi
    policy="$(grep -m1 '^# policy ' "$log" 2>/dev/null | sed 's/^# policy //' || true)"
    [[ -n "$policy" ]] || policy="NA"
    pool_line="$(grep -m1 '^# pool-chunks=' "$log" 2>/dev/null || true)"
    if [[ -n "$pool_line" ]]; then
        pool="$(awk -F'[= ]' '{print $3}' <<< "$pool_line")"
        peak_live="$(awk -F'[= ]' '{print $5}' <<< "$pool_line")"
        peak_reserved="$(awk -F'[= ]' '{print $7}' <<< "$pool_line")"
    fi
    [[ -n "$pool" ]] || pool="NA"
    [[ -n "$peak_live" ]] || peak_live="NA"
    [[ -n "$peak_reserved" ]] || peak_reserved="NA"

    if grep -q '^# weight,count' "$log" 2>/dev/null; then
        has_hist="1"
        awk '/^# weight,count/{seen=1;next} seen && /^[0-9]+,[0-9]+$/{print}' "$log" > "$hist_raw"
    else
        : > "$hist_raw"
    fi
    sort -t, -k1,1n -k2,2n "$hist_raw" > "$hist_sorted"
    hist_rows="$(wc -l < "$hist_raw" | tr -d ' ')"
    hist_sum="$(awk -F, '{s+=$2} END{printf "%.0f", s+0}' "$hist_raw")"

    failure=""
    if (( rc != 0 )); then
        if (( rc == 124 || rc == 137 )); then
            failure="exit=$rc (timeout ${timeout_s}s)"
        else
            # Prefer the example's graphFlow:: exception line. A T7 budget
            # rejection is an std::invalid_argument that exits 2 and prints
            # "graphFlow_particle_feedback: graphFlow::DevicePolicy: ...".
            # Otherwise keep the first non-blank stderr line and skip the
            # abnormal-termination lines added by /usr/bin/time.
            stderr_first="$(grep -m1 'graphFlow::' "$err" 2>/dev/null || true)"
            if [[ -z "$stderr_first" ]]; then
                stderr_first="$(grep -m1 -v -e '^[[:space:]]*$' \
                    -e '^Command exited with non-zero status' \
                    -e '^Command terminated by signal' "$err" 2>/dev/null || true)"
            fi
            stderr_first="${stderr_first#graphFlow_particle_feedback: }"
            stderr_first="${stderr_first//$'\t'/ }"
            failure="exit=$rc"
            if [[ -n "$stderr_first" ]]; then
                failure="${failure}: ${stderr_first:0:200}"
            fi
        fi
    fi
    if [[ "$selected" == "NA" ]]; then
        failure="${failure:+$failure; }missing selected-backend header"
    fi
    if [[ -z "$pool_line" ]]; then
        failure="${failure:+$failure; }missing pool-chunks header"
    fi
    if [[ "$has_hist" == "0" ]]; then
        failure="${failure:+$failure; }missing histogram header"
    fi
    if [[ "$policy" == "NA" ]]; then
        failure="${failure:+$failure; }missing policy header"
    fi
    if [[ -n "$pool_line" ]] && ! [[ "$pool" =~ ^[0-9]+$ && "$peak_live" =~ ^[0-9]+$ && "$peak_reserved" =~ ^[0-9]+$ ]]; then
        failure="${failure:+$failure; }malformed pool-chunks header"
    fi
    if (( hist_sum != n )); then
        failure="${failure:+$failure; }histogram sum $hist_sum != N $n"
    fi
    if [[ -n "$failure" ]]; then
        status="fail"
        case_failed=$(( case_failed + 1 ))
        printf '[%04d] FAIL %-4s n=%-10s c=%-5s p=%-6s q=%-2s seed=%-3s rep=%-3s %s\n' \
            "$case_index" "$backend" "$n" "$c" "$p" "$q" "$seed" "$rep" "$failure" >&2
    else
        status="ok"
        printf '[%04d] ok   %-4s n=%-10s c=%-5s p=%-6s q=%-2s seed=%-3s rep=%-3s wall=%ss\n' \
            "$case_index" "$backend" "$n" "$c" "$p" "$q" "$seed" "$rep" "$wall"
    fi

    if [[ "$wall" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        pps="$(awk -v n="$n" -v w="$wall" 'BEGIN { if (w + 0 > 0) printf "%.3f", n / w; else print "NA" }')"
    fi

    # Internal table order matches README.md. SEP is non-whitespace so empty
    # fields, such as a successful case's empty failure field, survive.
    tsv_join "$name" "$backend" "$selected" "$device" "$n" "$c" "$p" "$q" "$seed" "$rep" \
        "$status" "$failure" "$rc" "$wall" "$maxrss" "$pps" "$pool" "$peak_live" \
        "$peak_reserved" "$hist_rows" "$hist_sum" "$policy" "$log" "$hist_sorted" \
        >> "$results_tsv"

    if [[ "$status" == "ok" && ( "$backend" == "cpu" || "$backend" == "cuda" ) ]]; then
        tsv_join "$n" "$c" "$seed" "$q" "$rep" "$backend" "$hist_sorted" "$name" >> "$pairs_tsv"
    fi
}

for backend in "${backends[@]}"; do
    for n in "${ns[@]}"; do
        for c in "${cs[@]}"; do
            for p in "${ps[@]}"; do
                for q in "${qs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for (( rep = 1; rep <= reps; ++rep )); do
                            run_case "$backend" "$n" "$c" "$p" "$q" "$seed" "$rep"
                        done
                    done
                done
            done
        done
    done
done

# CPU/CUDA histogram comparison. The contract key is (N, C, seed, Q, rep); P is
# not part of it. Repetition `r` pairs CPU run r with CUDA run r. A key with
# more than one run on either side is reported as ambiguous instead of
# guessing which pair to compare.
declare -A cpu_count=() cuda_count=()
declare -A cpu_hist=() cuda_hist=() cpu_names=() cuda_names=()
declare -A pair_keys=()
while IFS="$SEP" read -r n c seed q rep backend hist name; do
    key="$n|$c|$seed|$q|$rep"
    pair_keys["$key"]="1"
    case "$backend" in
        cpu)
            cpu_count["$key"]=$(( ${cpu_count["$key"]:-0} + 1 ))
            cpu_hist["$key"]="$hist"
            cpu_names["$key"]="${cpu_names["$key"]:-}${name} "
            ;;
        cuda)
            cuda_count["$key"]=$(( ${cuda_count["$key"]:-0} + 1 ))
            cuda_hist["$key"]="$hist"
            cuda_names["$key"]="${cuda_names["$key"]:-}${name} "
            ;;
    esac
done < "$pairs_tsv"

declare -A rows_equal_by_case=()
if (( ${#pair_keys[@]} > 0 )); then
for key in "${!pair_keys[@]}"; do
    IFS='|' read -r n c seed q rep <<< "$key"
    cc="${cpu_count[$key]:-0}"
    uc="${cuda_count[$key]:-0}"
    cname="${cpu_names[$key]:-}"
    uname_="${cuda_names[$key]:-}"
    cname="${cname% }"
    uname_="${uname_% }"
    if (( cc == 0 || uc == 0 )); then
        eq="-"
        diff_path="-"
        [[ -n "$cname" ]] || cname="-"
        [[ -n "$uname_" ]] || uname_="-"
    elif (( cc > 1 || uc > 1 )); then
        eq="ambiguous"
        diff_path="-"
        [[ -n "$cname" ]] || cname="-"
        [[ -n "$uname_" ]] || uname_="-"
        for nm in $cname $uname_; do
            rows_equal_by_case["$nm"]="$eq"
        done
    else
        if cmp -s "${cpu_hist[$key]}" "${cuda_hist[$key]}"; then
            eq="yes"
            diff_path="-"
        else
            eq="no"
            safe_key="${key//[^A-Za-z0-9_.-]/_}"
            diff_path="$pair_dir/$safe_key.diff"
            diff -u "${cpu_hist[$key]}" "${cuda_hist[$key]}" > "$diff_path" 2>&1 || true
        fi
        rows_equal_by_case["$cname"]="$eq"
        rows_equal_by_case["$uname_"]="$eq"
    fi
    tsv_join "$n" "$c" "$seed" "$q" "$rep" "$eq" "$cname" "$uname_" "$diff_path" >> "$pair_results_tsv"
done
fi

# --copy-reference is informational for the benchmark verdict: a failure is
# recorded but never changes the case result. The generated reference prints a
# machine-readable GB/s line that the report copies verbatim; it never invents
# a number for a reference that did not run.
copy_reference_note="Not run."
copy_reference_number=""
if [[ "$copy_reference_set" == "1" ]]; then
    if [[ -n "$copy_reference" ]]; then
        ref_rc="0"
        if [[ "$have_timeout" == "1" ]]; then
            timeout --signal=TERM "${timeout_s}s" bash -c "$copy_reference" > "$out_dir/copy-reference.txt" 2>&1 || ref_rc=$?
        else
            bash -c "$copy_reference" > "$out_dir/copy-reference.txt" 2>&1 || ref_rc=$?
        fi
        {
            printf '\ncopy_reference_cmd: %s\n' "$copy_reference"
            printf 'copy_reference_exit: %s\n' "$ref_rc"
        } >> "$env_path"
        copy_reference_note="Ran \`$copy_reference\` (exit $ref_rc); raw output is in \`copy-reference.txt\`."
        copy_reference_number="$(grep -m1 '^copy_bandwidth_gbytes_per_s=' "$out_dir/copy-reference.txt" 2>/dev/null || true)"
    else
        # Same payload as the largest selected case so the reference covers the
        # data size the scatter actually moves.
        copy_bytes="0"
        for p in "${ps[@]}"; do
            for c in "${cs[@]}"; do
                candidate=$(( p * c * 24 ))
                (( candidate > copy_bytes )) && copy_bytes="$candidate"
            done
        done
        (( copy_bytes > 0 )) || copy_bytes=$(( 1024 * 1024 * 1024 ))
        nvcc_bin=""
        if command -v nvcc >/dev/null 2>&1; then
            nvcc_bin="$(command -v nvcc)"
        elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
            nvcc_bin="/usr/local/cuda/bin/nvcc"
        fi
        if [[ -n "$nvcc_bin" ]]; then
            copy_dir="$out_dir/copy-reference"
            mkdir -p "$copy_dir"
            cat > "$copy_dir/copy-reference.cu" <<'CUDA_SRC'
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                                         \
    do                                                                                           \
    {                                                                                            \
        cudaError_t const status = (call);                                                       \
        if(status != cudaSuccess)                                                                \
        {                                                                                        \
            std::fprintf(stderr, "%s\n", cudaGetErrorString(status));                            \
            return 2;                                                                            \
        }                                                                                        \
    } while(false)

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::fprintf(stderr, "usage: copy-reference BYTES ITERS [DEVICE]\n");
        return 2;
    }
    std::size_t const bytes = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    int const iters = std::atoi(argv[2]);
    int const device = argc > 3 ? std::atoi(argv[3]) : 0;
    CHECK_CUDA(cudaSetDevice(device));
    void* src = nullptr;
    void* dst = nullptr;
    CHECK_CUDA(cudaMalloc(&src, bytes));
    CHECK_CUDA(cudaMalloc(&dst, bytes));
    CHECK_CUDA(cudaMemset(src, 1, bytes));
    CHECK_CUDA(cudaMemset(dst, 0, bytes));
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < iters; ++i)
        CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    double const msPerCopy = static_cast<double>(elapsedMs) / static_cast<double>(iters);
    double const gbytesPerSecond = (static_cast<double>(bytes) / 1.0e9) / (msPerCopy / 1.0e3);
    std::printf(
        "copy_bandwidth_gbytes_per_s=%.3f bytes=%zu iters=%d ms_per_copy=%.3f device=%d\n",
        gbytesPerSecond,
        bytes,
        iters,
        msPerCopy,
        device);
    CHECK_CUDA(cudaFree(src));
    CHECK_CUDA(cudaFree(dst));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
CUDA_SRC
            ref_rc="0"
            {
                printf 'copy_reference_mode: generated-cuda-d2d\n'
                printf 'copy_reference_bytes: %s\n' "$copy_bytes"
                printf 'copy_reference_compiler: %s\n' "$nvcc_bin"
            } >> "$env_path"
            if "$nvcc_bin" -O3 -o "$copy_dir/copy-reference" "$copy_dir/copy-reference.cu" \
                > "$copy_dir/compile.log" 2>&1; then
                if [[ "$have_timeout" == "1" ]]; then
                    timeout --signal=TERM "${timeout_s}s" \
                        "$copy_dir/copy-reference" "$copy_bytes" 10 "$device" \
                        > "$out_dir/copy-reference.txt" 2>&1 || ref_rc=$?
                else
                    "$copy_dir/copy-reference" "$copy_bytes" 10 "$device" \
                        > "$out_dir/copy-reference.txt" 2>&1 || ref_rc=$?
                fi
                if (( ref_rc == 0 )); then
                    copy_reference_note="Ran the generated device-to-device copy reference over $copy_bytes B; raw output is in \`copy-reference.txt\`."
                    copy_reference_number="$(grep -m1 '^copy_bandwidth_gbytes_per_s=' "$out_dir/copy-reference.txt" 2>/dev/null || true)"
                else
                    copy_reference_note="BLOCKED: the generated copy reference exited $ref_rc; see \`copy-reference.txt\`."
                fi
            else
                ref_rc="$?"
                printf 'copy_bandwidth_gbytes_per_s=unavailable\n' > "$out_dir/copy-reference.txt"
                copy_reference_note="BLOCKED: the generated copy reference did not compile (nvcc exit $ref_rc); see \`copy-reference/compile.log\`."
            fi
            printf 'copy_reference_exit: %s\n' "$ref_rc" >> "$env_path"
        else
            ref_rc="127"
            printf 'copy_bandwidth_gbytes_per_s=unavailable\n' > "$out_dir/copy-reference.txt"
            {
                printf 'copy_reference_mode: blocked\n'
                printf 'copy_reference_exit: %s\n' "$ref_rc"
            } >> "$env_path"
            copy_reference_note="BLOCKED: no nvcc was found for the generated device-to-device copy reference."
        fi
    fi
fi

# Per-case aggregation across repetitions. The group key is the measured
# configuration without the repetition; failures still count toward `runs` but
# are excluded from the quantiles. Median and inclusive-halves IQR are printed
# for both wall time and end-to-end particles/s.
aggregates_tsv="$out_dir/aggregates.tsv"
aggregates_csv="$out_dir/aggregates.csv"
hist_stability_tsv="$out_dir/histogram-stability.tsv"
: > "$aggregates_tsv"
: > "$hist_stability_tsv"

awk -F"$SEP" -v OFS="$SEP" '
    function median(a, count,    mid) {
        if (count == 0) return 0
        mid = int((count + 1) / 2)
        if (count % 2 == 1) return a[mid]
        return (a[count / 2] + a[count / 2 + 1]) / 2
    }
    function sort(a, count,    i, j, v) {
        for (i = 2; i <= count; ++i) {
            v = a[i]
            j = i - 1
            while (j >= 1 && a[j] > v) {
                a[j + 1] = a[j]
                --j
            }
            a[j + 1] = v
        }
    }
    function iqr_of(a, count,    half, i, start, lower, upper) {
        if (count < 2) return 0
        half = int((count + 1) / 2)
        for (i = 1; i <= half; ++i)
            lower[i] = a[i]
        start = count - half + 1
        for (i = start; i <= count; ++i)
            upper[i - start + 1] = a[i]
        return median(upper, half) - median(lower, half)
    }
    {
        key = $2 "|" $3 "|" $5 "|" $6 "|" $7 "|" $8 "|" $9
        runs[key]++
        if ($11 == "ok") {
            slot = ++ok[key]
            wall[key SUBSEP slot] = $14 + 0
            pps[key SUBSEP slot] = $16 + 0
        }
    }
    END {
        for (key in runs) {
            split(key, field, "|")
            count = ok[key] + 0
            med_wall = "NA"
            iqr_wall = "NA"
            med_pps = "NA"
            iqr_pps = "NA"
            if (count > 0) {
                for (i = 1; i <= count; ++i) {
                    wall_values[i] = wall[key SUBSEP i]
                    pps_values[i] = pps[key SUBSEP i]
                }
                sort(wall_values, count)
                sort(pps_values, count)
                med_wall = sprintf("%.6f", median(wall_values, count))
                iqr_wall = sprintf("%.6f", iqr_of(wall_values, count))
                med_pps = sprintf("%.3f", median(pps_values, count))
                iqr_pps = sprintf("%.3f", iqr_of(pps_values, count))
            }
            print field[1], field[2], field[3], field[4], field[5], field[6], field[7], \
                runs[key], count, med_wall, iqr_wall, med_pps, iqr_pps
        }
    }
' "$results_tsv" | sort -t"$SEP" -k3,3n -k4,4n -k6,6n -k1,1 > "$aggregates_tsv"

{
    printf 'backend,selected_backend,n,chunk_size,live_particles,batch_chunks,seed,runs,ok_runs,median_wall_s,iqr_wall_s,median_particles_per_s,iqr_particles_per_s\n'
    while IFS="$SEP" read -r backend selected n c p q seed runs ok_runs med_wall iqr_wall med_pps iqr_pps; do
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$backend" "$selected" "$n" "$c" "$p" "$q" "$seed" "$runs" "$ok_runs" \
            "$med_wall" "$iqr_wall" "$med_pps" "$iqr_pps"
    done < "$aggregates_tsv"
} > "$aggregates_csv"

# Sorted histogram rows must depend only on (backend, N, C, seed): P, Q, and
# the repetition must not change the multiset. Compare every successful run in
# a group against the first and record the SHA-256 of the sorted rows.
declare -A hist_first=() hist_stable=() hist_runs=()
while IFS="$SEP" read -r name backend selected dev n c p q seed rep status failure rc wall maxrss pps pool peak_live peak_reserved hist_rows hist_sum policy log hist_sorted; do
    [[ "$status" == "ok" ]] || continue
    key="$backend|$n|$c|$seed"
    hist_runs["$key"]=$(( ${hist_runs["$key"]:-0} + 1 ))
    if [[ -z "${hist_first[$key]:-}" ]]; then
        hist_first["$key"]="$hist_sorted"
        hist_stable["$key"]="yes"
    elif ! cmp -s "${hist_first[$key]}" "$hist_sorted"; then
        hist_stable["$key"]="no"
    fi
done < "$results_tsv"
for key in "${!hist_runs[@]}"; do
    IFS='|' read -r backend n c seed <<< "$key"
    sha="unavailable"
    if command -v sha256sum >/dev/null 2>&1; then
        sha="$(sha256sum "${hist_first[$key]}" | awk '{print $1}')"
    fi
    tsv_join "$backend" "$n" "$c" "$seed" "${hist_runs[$key]}" "${hist_stable[$key]}" "$sha" >> "$hist_stability_tsv"
done
sort -t"$SEP" -k2,2n -k3,3n -k1,1 "$hist_stability_tsv" -o "$hist_stability_tsv"

# Kernel-level matrix. The example binary at this revision exposes Q, C, P, and
# N only; scatter mode, dense versus hash, and the balanced/one-hot/skewed/
# sparse distributions are selectable only through the committed integration
# tests. Run those test cases and record the exact names and Catch2 durations
# instead of claiming an end-to-end throughput axis that the CLI does not have.
kernel_note="Not run; pass --kernel-tests PATH to execute the kernel matrix."
kernel_failed="0"
if [[ -n "$kernel_tests" ]]; then
    if [[ ! -x "$kernel_tests" ]]; then
        die "--kernel-tests '$kernel_tests' is not an executable file"
    fi
    kernel_dir="$out_dir/kernel-tests"
    mkdir -p "$kernel_dir"
    kernel_rcs=""
    for spec in scatter hash; do
        case "$spec" in
            scatter) tag="[scatter]" ;;
            hash) tag="[hash]" ;;
        esac
        kernel_rc="0"
        if [[ "$have_timeout" == "1" ]]; then
            timeout --signal=TERM "${timeout_s}s" \
                "$kernel_tests" --durations yes --order decl "$tag" \
                > "$kernel_dir/$spec.txt" 2>&1 || kernel_rc=$?
        else
            "$kernel_tests" --durations yes --order decl "$tag" \
                > "$kernel_dir/$spec.txt" 2>&1 || kernel_rc=$?
        fi
        kernel_rcs="${kernel_rcs}${spec}=${kernel_rc} "
        if (( kernel_rc != 0 )); then
            kernel_failed="1"
            case_failed=$(( case_failed + 1 ))
        fi
    done
    {
        printf '# Kernel matrix (test-case durations)\n\n'
        printf 'Binary: `%s`\n\n' "$kernel_tests"
        printf 'Commands: `graphFlow_tests --durations yes --order decl [scatter]` and `[hash]`.\n'
        printf 'Exit status: `%s`\n\n' "${kernel_rcs% }"
        for spec in scatter hash; do
            printf '## %s\n\n' "$spec"
            printf '```text\n'
            grep -E '^[0-9]+([.][0-9]+)? s: ' "$kernel_dir/$spec.txt" 2>/dev/null || true
            printf '```\n\n'
        done
        printf 'The durations cover each whole Catch2 test case (device setup included), not an\n'
        printf 'isolated kernel. End-to-end throughput for these axes is not selectable through\n'
        printf 'the example CLI at this revision, so no such number is claimed.\n'
    } > "$out_dir/kernel-matrix.md"
    kernel_note="Ran the scatter and hash integration cases; see \`kernel-matrix.md\`."
fi

# Frozen throughput target. The verdict compares the best successful aggregate
# for (N, C, Q) against the nominal target reduced by the allowed shortfall.
target_state="BLOCKED"
target_line=""
if [[ -n "$target_pps" ]]; then
    target_line="$(
        awk -F"$SEP" -v n="$target_n" -v c="$target_c" -v q="$target_q" -v target="$target_pps" \
            -v tolerance="$target_tolerance" '
            $3 == n && $4 == c && $6 == q && $9 + 0 > 0 {
                if (selected == "" || $12 + 0 > best) {
                    best = $12 + 0
                    selected = $0
                }
            }
            END {
                if (selected != "") {
                    floor = target * (1.0 - tolerance / 100.0)
                    state = (best >= floor) ? "PASS" : "FAIL"
                    print selected "|" state "|" sprintf("%.3f", floor)
                }
            }' "$aggregates_tsv"
    )"
    if [[ -z "$target_line" ]]; then
        target_state="BLOCKED"
        target_reason="no successful aggregate matched N=$target_n C=$target_c Q=$target_q"
    else
        IFS='|' read -r t_line target_state target_floor <<< "$target_line"
        IFS="$SEP" read -r t_backend t_selected t_n t_c t_p t_q t_seed t_runs t_ok \
            t_med_wall t_iqr_wall t_med_pps t_iqr_pps <<< "$t_line"
        target_reason="matched backend=$t_backend P=$t_p seed=$t_seed ok_runs=$t_ok"
    fi
fi
target_failed="0"
[[ "$target_state" == "FAIL" ]] && target_failed="1"
{
    printf '# Frozen throughput target\n\n'
    if [[ -z "$target_pps" ]]; then
        printf 'No target requested. Pass `--target-pps` to enable the regression gate.\n'
    else
        printf -- '- Nominal target: `%s` particles/s\n' "$target_pps"
        printf -- '- Tolerance: `%s` percent\n' "$target_tolerance"
        printf -- '- Match key: `N=%s, C=%s, Q=%s`\n' "$target_n" "$target_c" "$target_q"
        if [[ "$target_state" == "BLOCKED" ]]; then
            printf -- '- Verdict: **BLOCKED** (%s)\n' "$target_reason"
        else
            printf -- '- Floor: `%s` particles/s\n' "$target_floor"
            printf -- '- Matched case: `%s` (%s), median `%s` particles/s\n' \
                "$t_backend" "$target_reason" "$t_med_pps"
            printf -- '- Verdict: **%s**\n' "$target_state"
        fi
    fi
} > "$out_dir/target.md"

{
    printf 'case,backend,selected_backend,device,n,chunk_size,live_particles,batch_chunks,seed,rep,status,failure,exit_status,wall_s,max_rss_kb,particles_per_s,pool_chunks,peak_live_chunks,peak_reserved_chunks,histogram_rows,histogram_sum,rows_equal,policy,log\n'
    while IFS="$SEP" read -r name backend selected dev n c p q seed rep status failure rc wall maxrss pps pool peak_live peak_reserved hist_rows hist_sum policy log hist_sorted; do
        eq="${rows_equal_by_case[$name]:--}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$name" "$backend" "$selected" "$dev" "$n" "$c" "$p" "$q" "$seed" "$rep" \
            "$status" "$(csv_quote "$failure")" "$rc" "$wall" "$maxrss" "$pps" \
            "$pool" "$peak_live" "$peak_reserved" "$hist_rows" "$hist_sum" "$eq" \
            "$(csv_quote "$policy")" "$(csv_quote "$log")"
    done < "$results_tsv"
} > "$csv_path"

{
    printf '# graphFlow particle-feedback benchmark\n\n'
    printf -- '- Generated: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- Host: `%s` (%s)\n' "$(uname -n)" "$(uname -sr)"
    printf -- '- Matrix: `%s`\n' "$matrix"
    printf -- '- Binary: `%s`\n' "$binary"
    printf -- '- Binary SHA-256: `%s`\n' "$binary_sha"
    printf -- '- Git commit: `%s`\n' "$git_commit"
    printf -- '- Backends: `%s`\n' "${backends[*]}"
    printf -- '- Device index: `%s` (CUDA_DEVICE_ORDER=PCI_BUS_ID)\n' "$device"
    printf -- '- CUDA_VISIBLE_DEVICES: `%s`\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf -- '- Per-run timeout: `%ss`\n' "$timeout_s"
    printf -- '- Repetitions per case: `%s`\n' "$reps"
    printf -- '- Wall timer: `%s`\n' "$([[ "$have_time" == "1" ]] && echo /usr/bin/time || echo shell-seconds-fallback)"
    printf -- '- CPU/CUDA pair key: `(N, C, seed, Q, rep)`\n'
    printf -- '- Aggregate quantiles: median and inclusive-halves IQR\n\n'
    printf 'The script records only what was executed. Treat a shared host as a\n'
    printf 'baseline; a target verdict is acceptance-grade only on an exclusive\n'
    printf 'allocation.\n\n'

    printf '## Aggregates (median and IQR)\n\n'
    printf '| backend | N | C | P | Q | seed | runs | ok | median wall s | IQR wall s | median particles/s | IQR particles/s |\n'
    printf '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    while IFS="$SEP" read -r backend selected n c p q seed runs ok_runs med_wall iqr_wall med_pps iqr_pps; do
        printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
            "$backend" "$n" "$c" "$p" "$q" "$seed" "$runs" "$ok_runs" \
            "$med_wall" "$iqr_wall" "$med_pps" "$iqr_pps"
    done < "$aggregates_tsv"

    printf '\n## Sorted-histogram stability\n\n'
    printf 'Every successful run in a `(backend, N, C, seed)` group must produce the\n'
    printf 'same sorted rows across P, Q, and the repetitions. The SHA-256 covers the\n'
    printf 'sorted `weight,count` rows of the first run in the group.\n\n'
    printf '| backend | N | C | seed | runs | rows_equal | SHA-256 |\n'
    printf '| --- | ---: | ---: | ---: | ---: | --- | --- |\n'
    while IFS="$SEP" read -r backend n c seed runs stable sha; do
        printf '| %s | %s | %s | %s | %s | %s | `%s` |\n' \
            "$backend" "$n" "$c" "$seed" "$runs" "$stable" "$sha"
    done < "$hist_stability_tsv"

    printf '\n## Cases\n\n'
    printf '| case | backend | N | C | P | Q | seed | rep | status | wall_s | particles/s | max RSS KiB | pool | peak live | peak res | hist rows | hist sum | rows_equal |\n'
    printf '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n'
    while IFS="$SEP" read -r name backend selected dev n c p q seed rep status failure rc wall maxrss pps pool peak_live peak_reserved hist_rows hist_sum policy log hist_sorted; do
        eq="${rows_equal_by_case[$name]:--}"
        printf '| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
            "$name" "$backend" "$n" "$c" "$p" "$q" "$seed" "$rep" "$status" "$wall" "$pps" \
            "$maxrss" "$pool" "$peak_live" "$peak_reserved" "$hist_rows" "$hist_sum" "$eq"
    done < "$results_tsv"

    printf '\n## CPU/CUDA histogram comparison\n\n'
    printf 'Sorted histogram rows are compared only when the pair key has exactly\n'
    printf 'one CPU and one CUDA run. `-` means no counterpart in this run;\n'
    printf '`ambiguous` means the key matched several runs (for example a P sweep).\n\n'
    printf '| N | C | seed | Q | rep | rows_equal | CPU case | CUDA case | diff |\n'
    printf '| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |\n'
    if [[ -s "$pair_results_tsv" ]]; then
        sort -t"$SEP" -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n "$pair_results_tsv" |
        while IFS="$SEP" read -r n c seed q rep eq cname uname_ diff_path; do
            printf '| %s | %s | %s | %s | %s | %s | `%s` | `%s` | `%s` |\n' \
                "$n" "$c" "$seed" "$q" "$rep" "$eq" "$cname" "$uname_" "$diff_path"
        done
    else
        printf '| - | - | - | - | - | - | - | - | - |\n'
    fi

    printf '\n## Kernel matrix (scatter, dense/hash, distributions)\n\n'
    printf '%s\n\n' "$kernel_note"

    printf '\n## Copy-bandwidth reference\n\n'
    printf '%s\n\n' "$copy_reference_note"
    if [[ -n "$copy_reference_number" ]]; then
        printf '```text\n%s\n```\n\n' "$copy_reference_number"
    fi
    printf 'Documented fallback commands (none executed unless passed through `--copy-reference`):\n\n'
    printf '```sh\n'
    printf 'nvbandwidth -t device_to_device_memcpy_read_ce\n'
    printf 'bandwidthTest --device=%s --mode=quick\n' "$device"
    printf '```\n\n'
    if [[ -z "$copy_reference_number" ]]; then
        printf 'No copy-bandwidth number was measured by this report.\n'
    fi

    printf '\n## Frozen throughput target\n\n'
    if [[ -z "$target_pps" ]]; then
        printf 'No target requested. See `target.md`.\n'
    elif [[ "$target_state" == "BLOCKED" ]]; then
        printf '**BLOCKED**: %s. See `target.md`.\n' "$target_reason"
    else
        printf 'Nominal `%s` particles/s at `N=%s, C=%s, Q=%s`, tolerance `%s%%`; ' \
            "$target_pps" "$target_n" "$target_c" "$target_q" "$target_tolerance"
        printf 'verdict **%s** (median `%s` particles/s). See `target.md`.\n' \
            "$target_state" "$t_med_pps"
    fi

    if (( case_failed > 0 )); then
        printf '\n## Failures\n\n'
        while IFS="$SEP" read -r name backend selected dev n c p q seed rep status failure rc wall maxrss pps pool peak_live peak_reserved hist_rows hist_sum policy log hist_sorted; do
            if [[ "$status" == "fail" ]]; then
                printf -- '- `%s`: %s (log: `%s`)\n' "$name" "$failure" "$log"
            fi
        done < "$results_tsv"
    fi
} > "$summary_path"

printf '\nCSV:        %s\n' "$csv_path"
printf 'Aggregates: %s\n' "$aggregates_csv"
printf 'Summary:    %s\n' "$summary_path"
printf 'Logs:       %s\n' "$log_dir"
if [[ -n "$target_pps" ]]; then
    printf 'Target:      %s\n' "$target_state"
fi
if (( case_failed > 0 )); then
    printf '%s: %d of %d cases failed\n' "$PROG" "$case_failed" "$case_index" >&2
    exit 1
fi
if (( target_failed > 0 )); then
    printf '%s: frozen throughput target FAILED\n' "$PROG" >&2
    exit 1
fi
printf '%s: all %d cases passed\n' "$PROG" "$case_index"
exit 0
