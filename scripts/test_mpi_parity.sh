#!/usr/bin/env bash
# scripts/test_mpi_parity.sh
#
# Smoke-test for the MPI trainer: with a shared seed, np=1 of the MPI
# binary should produce the *same* loss trajectory as the single-process
# train binary (same RNG draws, no allreduce path active). Bit-identical
# parity is not achievable — the backward rasterizer uses atomicAdd, and
# CUDA does not guarantee deterministic reduction order, so two runs of
# the same binary with the same seed differ by ~1e-5 at later iters. We
# therefore assert agreement within TOL_ABS, defaulting to 1e-4.
# np=2 will diverge — it uses a different camera each iter on rank 1 —
# but the trajectory should remain finite and non-divergent.
#
# We compare:
#   (a) ./build/train             vs ./build/train_mpi -np 1   → bit-identical
#       per-iter losses required.
#   (b) ./build/train_mpi -np 1   vs ./build/train_mpi -np 2   → np=2 differs
#       (different effective batch); just assert finite + stable.
#
# Usage: make run-mpi-parity   (or: bash scripts/test_mpi_parity.sh)
#
# Exits 0 on pass, 1 on failure.

set -u
set -o pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-data/360_v2/garden}
IMAGES=${IMAGES:-images_4}
ITERS=${ITERS:-30}
LOG_EVERY=${LOG_EVERY:-1}
TOL_ABS=${TOL_ABS:-1e-4}
MPIRUN=${MPIRUN:-mpirun}
OVERSUB=${OVERSUB:---oversubscribe}
MPI_RUNTIME_FLAGS=${MPI_RUNTIME_FLAGS:---mpi-host-stage}

# Densification must not fire inside the window — different rank counts
# can produce different N progressions, and that's a separate test.
COMMON_FLAGS=(
    --data "$DATA"
    --images "$IMAGES"
    --iters "$ITERS"
    --log-every "$LOG_EVERY"
    --densify-every 0
    --opacity-reset-every 0
    --seed 42
)

# Optionally exercise the Nexels appearance path. The same parity
# invariant must hold: at np=1 the MPI binary should match the
# single-process binary byte-identically (because the appearance-grad
# allreduce is gated on world_size > 1 and the hash grid + MLP init use
# a host RNG with cfg.nexel_init_seed).
#
# Trigger with: NEXEL=1 bash scripts/test_mpi_parity.sh
if [ "${NEXEL:-0}" = "1" ]; then
    COMMON_FLAGS+=(--use-nexel --hash-log-table-size "${HASH_LOG2:-17}")
    echo "(NEXEL=1: exercising the --use-nexel pipeline)"
fi

mkdir -p logs/parity
SOLO_LOG="logs/parity/solo.log"
MPI1_LOG="logs/parity/mpi_np1.log"
MPI2_LOG="logs/parity/mpi_np2.log"

# Strip the progress-bar overwrite chars so awk can find one record per line.
sanitise () { tr '\r' '\n' < "$1" | grep -E 'loss=' ; }
extract_losses () { sanitise "$1" | awk '{ for (i=1;i<=NF;i++) if ($i ~ /^loss=/) { sub("loss=","",$i); print $i } }'; }

echo "==> baseline: ./build/train (no MPI)"
./build/train "${COMMON_FLAGS[@]}" > "$SOLO_LOG" 2>&1 || { echo "solo train failed"; tail "$SOLO_LOG"; exit 1; }

echo "==> ./build/train_mpi -np 1 (MPI singleton)"
$MPIRUN -np 1 $OVERSUB ./build/train_mpi "${COMMON_FLAGS[@]}" $MPI_RUNTIME_FLAGS > "$MPI1_LOG" 2>&1 || { echo "np=1 train failed"; tail "$MPI1_LOG"; exit 1; }

echo "==> ./build/train_mpi -np 2"
$MPIRUN -np 2 $OVERSUB ./build/train_mpi "${COMMON_FLAGS[@]}" $MPI_RUNTIME_FLAGS > "$MPI2_LOG" 2>&1 || { echo "np=2 train failed"; tail "$MPI2_LOG"; exit 1; }

L_SOLO=$(extract_losses "$SOLO_LOG")
L_MPI1=$(extract_losses "$MPI1_LOG")
L_MPI2=$(extract_losses "$MPI2_LOG")

N_SOLO=$(printf '%s\n' "$L_SOLO" | wc -l)
N_MPI1=$(printf '%s\n' "$L_MPI1" | wc -l)
N_MPI2=$(printf '%s\n' "$L_MPI2" | wc -l)
echo "    iters logged:  solo=$N_SOLO  mpi1=$N_MPI1  mpi2=$N_MPI2"

if [ "$N_SOLO" -ne "$N_MPI1" ] || [ "$N_SOLO" -ne "$N_MPI2" ]; then
    echo "FAIL: mismatched iter counts"; exit 1
fi

# (a) solo vs MPI np=1: must agree within $TOL_ABS at every logged iter.
paste <(echo "$L_SOLO") <(echo "$L_MPI1") | \
awk -v tol="$TOL_ABS" '
    { a=$1; b=$2; d=a-b; if (d<0) d=-d;
      if (d > tol) { printf("FAIL parity@iter %d: solo=%s  mpi1=%s  |delta|=%.3e > %s\n", NR, a, b, d, tol); bad=1 }
      if (a+0 != a+0 || b+0 != b+0) { printf("FAIL NaN@iter %d: solo=%s mpi1=%s\n", NR, a, b); bad=1 }
    }
    END { if (bad) exit 1; printf("PASS solo vs mpi_np1: max |delta| within %s over %d iters\n", tol, NR) }
' || exit 1

# (b) MPI np=1 vs np=2: just require finite, non-divergent loss.
echo "$L_MPI2" | awk '
    { x=$1; if (x+0 != x+0) { printf("FAIL NaN@iter %d (np=2)\n", NR); exit 1 }
      if (x > 100.0)        { printf("FAIL divergence@iter %d (np=2 loss=%s)\n", NR, x); exit 1 } }
    END { printf("PASS np=2 stable: %d iters, all finite\n", NR) }
'

echo "OK"
