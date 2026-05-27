#!/usr/bin/env bash
# scripts/bench_strong_scaling.sh
#
# Strong-scaling sweep for the MPI trainer: hold the *total* per-iter
# compute work fixed across rank counts. Specifically:
#
#   total_iters_in_window = TOTAL_ITERS / np     (rounded down)
#
# So np=1 measures 100 iters of "1 camera/iter", np=2 measures 50 iters of
# "2 cameras/iter", np=4 measures 25 iters of "4 cameras/iter". Total
# camera-iterations is constant (TOTAL_ITERS).
#
# Per-iter compute per rank is the same regardless of np; the change
# between configurations is the allreduce overhead per iter plus the
# wall-time effect of doing 1/np fewer iters. Perfect-strong-scaling would
# give wall_np / wall_1 == 1/np (because the global iter count drops 1/np
# while per-iter time stays constant).
#
# All runs append per-bucket rows to logs/perf/strong_scaling.csv tagged
# np=<n>, plus a one-line per-run summary to logs/perf/strong_summary.csv:
#
#   np,iters,profile_iters,wall_sec,iters_per_sec,host_stage
#
# Usage:
#   bash scripts/bench_strong_scaling.sh                   # default sweep {1,2,4}
#   NP_LIST="1 2" TOTAL_ITERS=120 bash scripts/bench_strong_scaling.sh
#
# Plot with: python3 scripts/bench_comm_breakdown.py logs/perf/strong_scaling.csv

set -u
set -o pipefail

cd "$(dirname "$0")/.."

NP_LIST=${NP_LIST:-"1 2 4"}
DATA=${DATA:-data/360_v2/garden}
IMAGES=${IMAGES:-images_4}
TOTAL_ITERS=${TOTAL_ITERS:-200}           # camera-iterations to measure across the window
WARMUP_ITERS=${WARMUP_ITERS:-50}          # iters before the profile window opens
SEED=${SEED:-42}
MPIRUN=${MPIRUN:-mpirun}
OVERSUB=${OVERSUB:---oversubscribe}
MPI_RUNTIME_FLAGS=${MPI_RUNTIME_FLAGS:---mpi-host-stage}

# NEXEL=1 enables the Nexels appearance path (--use-nexel + a smaller
# hash table to keep the per-iter allreduce payload sensible on this
# hardware). All output CSVs get a "_nexel" suffix so a follow-up SH
# run does not clobber them.
NEXEL_FLAGS=""
NEXEL_SUFFIX=""
if [ "${NEXEL:-0}" = "1" ]; then
    NEXEL_FLAGS="--use-nexel --hash-log-table-size ${HASH_LOG2:-17}"
    NEXEL_SUFFIX="_nexel"
    echo "(NEXEL=1: --use-nexel pipeline; CSVs suffixed _nexel)"
fi

mkdir -p logs/perf
BREAKDOWN_CSV=logs/perf/strong_scaling${NEXEL_SUFFIX}.csv
SUMMARY_CSV=logs/perf/strong_summary${NEXEL_SUFFIX}.csv
RUN_LOG_DIR=logs/perf/strong_logs${NEXEL_SUFFIX}
mkdir -p "$RUN_LOG_DIR"

# Clean slate so the python plotter sees only this sweep.
rm -f "$BREAKDOWN_CSV" "$SUMMARY_CSV"
printf 'np,profile_iters,total_iters_window,wall_sec,iters_per_sec,host_stage\n' > "$SUMMARY_CSV"

# Quick check: train_mpi present?
test -x ./build/train_mpi || { echo "build/train_mpi missing — run: make build/train_mpi" >&2; exit 1; }

for NP in $NP_LIST; do
    # Profile window = ceil(TOTAL_ITERS / np). Train slightly longer to give
    # the window a warmup runway.
    PROF_ITERS=$(( (TOTAL_ITERS + NP - 1) / NP ))
    RUN_ITERS=$(( WARMUP_ITERS + PROF_ITERS + 5 ))
    TAG="np${NP}"
    LOG="${RUN_LOG_DIR}/${TAG}.log"

    echo "==> [strong] np=$NP  prof_iters=$PROF_ITERS  total_iters=$RUN_ITERS  flags='$MPI_RUNTIME_FLAGS'"

    # /usr/bin/time -f gives us elapsed seconds on stderr in a parseable form.
    # Use SECONDS as a fallback so the script works without GNU time.
    START=$SECONDS
    $MPIRUN -np "$NP" $OVERSUB ./build/train_mpi \
        --data "$DATA" --images "$IMAGES" \
        --iters "$RUN_ITERS" \
        --profile-start-iter "$((WARMUP_ITERS + 1))" --profile-iters "$PROF_ITERS" \
        --profile-csv "$BREAKDOWN_CSV" --profile-tag "$TAG" \
        --profile-exit \
        --densify-every 0 --opacity-reset-every 0 \
        --seed "$SEED" --log-every 1000 \
        $MPI_RUNTIME_FLAGS $NEXEL_FLAGS \
        > "$LOG" 2>&1
    STATUS=$?
    WALL=$(( SECONDS - START ))
    if [ $STATUS -ne 0 ]; then
        echo "  FAILED (status=$STATUS, see $LOG)" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi

    if [ "$PROF_ITERS" -gt 0 ] && [ "$WALL" -gt 0 ]; then
        IPS=$(python3 -c "print(f'{${PROF_ITERS}/${WALL}:.3f}')")
    else
        IPS="0"
    fi
    HOST_STAGE_NOTE="no"
    [[ "$MPI_RUNTIME_FLAGS" == *--mpi-host-stage* ]] && HOST_STAGE_NOTE="yes"
    printf '%d,%d,%d,%d,%s,%s\n' "$NP" "$PROF_ITERS" "$TOTAL_ITERS" "$WALL" "$IPS" "$HOST_STAGE_NOTE" >> "$SUMMARY_CSV"
    echo "  done in ${WALL}s  →  ${IPS} it/s  (logs: $LOG)"
done

echo
echo "── strong scaling summary ─────────────────────────────────────"
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo
echo "Per-bucket detail: $BREAKDOWN_CSV"
echo "Plot:              python3 scripts/bench_comm_breakdown.py $BREAKDOWN_CSV"
