#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data/360_v2/garden}"
IMAGES_DIR="${IMAGES_DIR:-images_4}"
ITERS="${ITERS:-30000}"
LOG_EVERY="${LOG_EVERY:-50}"
PREVIEW_EVERY="${PREVIEW_EVERY:-500}"
PREVIEW_OUT="${PREVIEW_OUT:-previews/garden_script/iter}"
SAVE_PLY="${SAVE_PLY:-checkpoints/garden_script_latest.ply}"
SAVE_PLY_EVERY="${SAVE_PLY_EVERY:-500}"

DENSIFY_START="${DENSIFY_START:-250}"
DENSIFY_EVERY="${DENSIFY_EVERY:-50}"
DENSIFY_STOP="${DENSIFY_STOP:-15000}"
OPACITY_RESET_EVERY="${OPACITY_RESET_EVERY:-3000}"
DENSIFY_PRUNE_ALPHA="${DENSIFY_PRUNE_ALPHA:-0.05}"
DENSIFY_GROW_SCALE3D="${DENSIFY_GROW_SCALE3D:-0.01}"
DENSIFY_PRUNE_SCALE3D="${DENSIFY_PRUNE_SCALE3D:-0.1}"

DIST_LAMBDA="${DIST_LAMBDA:-1e-2}"
DIST_START_ITER="${DIST_START_ITER:-3000}"
NORMAL_LAMBDA="${NORMAL_LAMBDA:-5e-2}"
NORMAL_START_ITER="${NORMAL_START_ITER:-7000}"

mkdir -p "$(dirname "$PREVIEW_OUT")" "$(dirname "$SAVE_PLY")"

make build/train

exec ./build/train \
  --data "$DATA_DIR" \
  --images "$IMAGES_DIR" \
  --iters "$ITERS" \
  --log-every "$LOG_EVERY" \
  --preview-every "$PREVIEW_EVERY" \
  --preview-out "$PREVIEW_OUT" \
  --save-ply "$SAVE_PLY" \
  --save-ply-every "$SAVE_PLY_EVERY" \
  --densify-start "$DENSIFY_START" \
  --densify-every "$DENSIFY_EVERY" \
  --densify-stop "$DENSIFY_STOP" \
  --opacity-reset-every "$OPACITY_RESET_EVERY" \
  --densify-prune-alpha "$DENSIFY_PRUNE_ALPHA" \
  --densify-grow-scale3d "$DENSIFY_GROW_SCALE3D" \
  --densify-prune-scale3d "$DENSIFY_PRUNE_SCALE3D" \
  --dist-lambda "$DIST_LAMBDA" \
  --dist-start-iter "$DIST_START_ITER" \
  --normal-lambda "$NORMAL_LAMBDA" \
  --normal-start-iter "$NORMAL_START_ITER" \
  "$@"
