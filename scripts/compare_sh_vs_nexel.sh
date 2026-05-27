#!/usr/bin/env bash
# scripts/compare_sh_vs_nexel.sh
#
# Train the same scene for ITERS steps with two appearance pipelines
# back-to-back:
#
#   1. SH  : the legacy spherical-harmonics path (default).
#   2. Nexels: --use-nexel (hash grid + shared MLP).
#
# Both runs share --seed, --densify-*, --iters so the only difference is
# the colour source. Per-iter logs are saved verbatim, plus a 2-row
# summary CSV with the best PSNR and the final-iter values, plus a final
# preview from each run for side-by-side inspection.
#
# Usage:
#   bash scripts/compare_sh_vs_nexel.sh
#   ITERS=2000 bash scripts/compare_sh_vs_nexel.sh
#
# Output:
#   logs/quality/sh.log
#   logs/quality/nexel.log
#   logs/quality/summary.csv
#   previews/quality_sh_final.png
#   previews/quality_nexel_final.png

set -u
set -o pipefail

cd "$(dirname "$0")/.."

DATA=${DATA:-data/360_v2/garden}
IMAGES=${IMAGES:-images_4}
ITERS=${ITERS:-5000}
LOG_EVERY=${LOG_EVERY:-50}
SEED=${SEED:-42}
HASH_LOG2=${HASH_LOG2:-17}      # T = 2^17 = 128K cells per level

# Densification window scaled to the iter budget.
DENSIFY_START=${DENSIFY_START:-250}
DENSIFY_EVERY=${DENSIFY_EVERY:-100}
DENSIFY_STOP=${DENSIFY_STOP:-$(( ITERS * 3 / 4 ))}

mkdir -p logs/quality previews

COMMON=(
    --data "$DATA"
    --images "$IMAGES"
    --iters "$ITERS"
    --log-every "$LOG_EVERY"
    --seed "$SEED"
    --densify-start "$DENSIFY_START"
    --densify-every "$DENSIFY_EVERY"
    --densify-stop  "$DENSIFY_STOP"
    --opacity-reset-every 0
    --preview-every "$((ITERS))"          # one preview at the final iter
    --preview-out previews/__quality
)

test -x ./build/train || { echo "build/train missing — run: make build/train"; exit 1; }

run_one () {
    local TAG="$1"; shift
    local LOG="logs/quality/${TAG}.log"
    local PREVIEW="previews/quality_${TAG}_final.png"
    local PREVIEW_RAW="previews/__quality_$(printf '%06d' "$ITERS").png"

    echo "==> running tag='$TAG'  iters=$ITERS  densify=[$DENSIFY_START,$DENSIFY_STOP] every=$DENSIFY_EVERY"
    rm -f "$PREVIEW_RAW"
    /usr/bin/time -f '%e' -o "logs/quality/${TAG}.wall" \
        ./build/train "${COMMON[@]}" "$@" > "$LOG" 2>&1 || {
        echo "  FAILED — tail:"; tail -10 "$LOG"; return 1;
    }
    if [ -f "$PREVIEW_RAW" ]; then mv "$PREVIEW_RAW" "$PREVIEW"; fi
    echo "  log : $LOG"
    echo "  png : $PREVIEW"
}

# 1. SH baseline.
run_one sh
# 2. Nexels — per-Gaussian RGB.
run_one nexel --use-nexel --hash-log-table-size "$HASH_LOG2"
# 3. Nexels — view-dependent SH-basis (D_out=48).
run_one nexel_vd --use-nexel --hash-log-table-size "$HASH_LOG2" --nexel-view-dep

# Build a small summary CSV from the per-iter PSNR values logged each
# log_every iters. Cols: tag,iter,loss,l1,dssim,psnr.
TAGS=(sh nexel nexel_vd)
echo "tag,iter,loss,l1,dssim,psnr" > logs/quality/per_iter.csv
for TAG in "${TAGS[@]}"; do
    tr '\r' '\n' < "logs/quality/${TAG}.log" \
        | grep -E '\] +[0-9]+/[0-9]+ ' \
        | awk -v tag="$TAG" '{
            for (i=1; i<=NF; i++) {
                if ($i ~ /[0-9]+\/[0-9]+/ && $i !~ /^sh=/) { split($i, a, "/"); iter=a[1]+0 }
                if ($i ~ /^loss=/)  { sub("loss=","",$i);  loss=$i }
                if ($i ~ /^l1=/)    { sub("l1=","",$i);    l1=$i }
                if ($i ~ /^dssim=/) { sub("dssim=","",$i); dssim=$i }
                if ($i ~ /^psnr=/)  { sub("psnr=","",$i);  psnr=$i }
            }
            printf("%s,%d,%s,%s,%s,%s\n", tag, iter, loss, l1, dssim, psnr)
        }' >> logs/quality/per_iter.csv
done

# Final-iter + best-PSNR summary.
echo "tag,wall_sec,final_iter,final_loss,final_l1,final_dssim,final_psnr,best_psnr,best_psnr_iter" > logs/quality/summary.csv
for TAG in "${TAGS[@]}"; do
    WALL=$(cat "logs/quality/${TAG}.wall" 2>/dev/null || echo "?")
    awk -v tag="$TAG" -v wall="$WALL" -F, '
        $1 == tag && NR > 1 {
            last_iter = $2; last_loss = $3; last_l1 = $4; last_dssim = $5; last_psnr = $6;
            if ($6 + 0 > best_psnr + 0) { best_psnr = $6; best_iter = $2; }
        }
        END {
            printf("%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                   tag, wall, last_iter, last_loss, last_l1, last_dssim, last_psnr,
                   best_psnr, best_iter)
        }
    ' logs/quality/per_iter.csv >> logs/quality/summary.csv
done

echo
echo "── quality comparison summary ──────────────────────────────────"
column -t -s, logs/quality/summary.csv 2>/dev/null || cat logs/quality/summary.csv
echo
echo "Full per-iter trace: logs/quality/per_iter.csv"
echo "Previews:            previews/quality_sh_final.png  previews/quality_nexel_final.png"
