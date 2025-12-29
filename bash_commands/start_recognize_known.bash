#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="$REPO_ROOT/build"
OUT_DIR="$REPO_ROOT/output_batch/tmp"

cd "$BUILD_DIR"

rm -rf "$OUT_DIR"

./app --dir "$REPO_ROOT/data/test_videos/known" \
      --out "$OUT_DIR" \
      --train_web "" \
      --train_auto "$REPO_ROOT/data/train_auto_pred/clusters_clean" \
      --step 5 \
      --conf 0.60 \
      --nms 0.40 \
      --topk 200 \
      --min_face 45 \
      --max_faces 25 \
      --unk_th 0.4 \
      --margin 0.01 \
      --strong 0.98