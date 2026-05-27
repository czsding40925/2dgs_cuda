#!/usr/bin/env bash
# scripts/bench_weak_scaling.sh
#
# Weak-scaling sweep for the MPI trainer: hold the per-rank work fixed.
# Every rank does the same iter count regardless of np, so the *total*
# work scales linearly with np. Per-iter wall time should stay near
# constant if the implementation is scaling well — communication overhead
# (allreduce + RNG synchronisation + load imbalance) is what makes it grow.
#
# All runs append per-bucket rows to logs/perf/weak_scaling.csv tagged
# np=<n>, plus a one-line per-run summary to logs/perf/weak_summary.csv:
#
#   np,iters_per_rank,wall_sec,iters_per_sec,host_stage
#
# Usage:
#   bash scripts/bench_weak_scaling.sh                  # default sweep {1,2,4}
#   NP_LIST="1 2" PROFILE_ITERS=40 bash scripts/bench_weak_scaling.sh
#
# Plot: python3 scripts/bench_comm_breakdown.py logs/perf/weak_scaling.csv

set -u
set -o pipefail

cd "$(dirname "$0")/.."

NP_LIST=${NP_LIST:-"1 2 4"}
DATA=${DATA:-data/360_v2/garden}
IMAGES=${IMAGES:-images_4}
WARMUP_ITERS=${WARMUP_ITERS:-50}
PROFILE_ITERS=${PROFILE_ITERS:-50}    # iters per rank in the measurement window
SEED=${SEED:-42}
MPIRUN=${MPIRUN:-mpirun}
OVERSUB=${OVERSUB:---oversubscribe}
MPI_RUNTIME_FLAGS=${MPI_RUNTIME_FLAGS:---mpi-host-stage}

NEXEL_FLAGS=""
NEXEL_SUFFIX=""
if [ "${NEXEL:-0}" = "1" ]; then
    NEXEL_FLAGS="--use-nexel --hash-log-table-size ${HASH_LOG2:-17}"
    NEXEL_SUFFIX="_nexel"
    echo "(NEXEL=1: --use-nexel pipeline; CSVs suffixed _nexel)"
fi

mkdir -p logs/perf
BREAKDOWN_CSV=logs/perf/weak_scaling${NEXEL_SUFFIX}.csv
SUMMARY_CSV=logs/perf/weak_summary${NEXEL_SUFFIX}.csv
RUN_LOG_DIR=logs/perf/weak_logs${NEXEL_SUFFIX}
mkdir -p "$RUN_LOG_DIR"

rm -f "$BREAKDOWN_CSV" "$SUMMARY_CSV"
printf 'np,iters_per_rank,wall_sec,iters_per_sec,host_stage\n' > "$SUMMARY_CSV"

test -x ./build/train_mpi || { echo "build/train_mpi missing — run: make build/train_mpi" >&2; exit 1; }

RUN_ITERS=$(( WARMUP_ITERS + PROFILE_ITERS + 5 ))

for NP in $NP_LIST; do
    TAG="np${NP}"
    LOG="${RUN_LOG_DIR}/${TAG}.log"

    echo "==> [weak] np=$NP  profile_iters=$PROFILE_ITERS  total_iters=$RUN_ITERS  flags='$MPI_RUNTIME_FLAGS'"

    START=$SECONDS
    $MPIRUN -np "$NP" $OVERSUB ./build/train_mpi \
        --data "$DATA" --images "$IMAGES" \
        --iters "$RUN_ITERS" \
        --profile-start-iter "$((WARMUP_ITERS + 1))" --profile-iters "$PROFILE_ITERS" \
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
    if [ "$PROFILE_ITERS" -gt 0 ] && [ "$WALL" -gt 0 ]; then
        IPS=$(python3 -c "print(f'{${PROFILE_ITERS}/${WALL}:.3f}')")
    else
        IPS="0"
    fi
    HOST_STAGE_NOTE="no"
    [[ "$MPI_RUNTIME_FLAGS" == *--mpi-host-stage* ]] && HOST_STAGE_NOTE="yes"
    printf '%d,%d,%d,%s,%s\n' "$NP" "$PROFILE_ITERS" "$WALL" "$IPS" "$HOST_STAGE_NOTE" >> "$SUMMARY_CSV"
    echo "  done in ${WALL}s  →  ${IPS} it/s  (logs: $LOG)"
done

echo
echo "── weak scaling summary ──────────────────────────────────────"
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo
echo "Per-bucket detail: $BREAKDOWN_CSV"
echo "Plot:              python3 scripts/bench_comm_breakdown.py $BREAKDOWN_CSV"
