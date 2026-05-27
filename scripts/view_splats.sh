#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PLY="${PLY:-checkpoints/gamma_patchB_full.ply}"
PORT="${PORT:-8080}"
RESOLUTION="${RESOLUTION:-1024}"

make build/train

exec python3 python/viser_splat_viewer.py \
  --ply "$PLY" \
  --port "$PORT" \
  --resolution "$RESOLUTION" \
  "$@"
