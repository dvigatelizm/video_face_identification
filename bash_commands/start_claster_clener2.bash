#!/usr/bin/env bash
set -e

cd ~/video_face_identification/tools/build_cleaner

OUT=../../data/train_auto_pred/clusters_clean

rm -rf "$OUT"
mkdir -p "$OUT"

./cluster_cleaner \
  --in  ../../data/train_auto_pred/clusters \
  --out "$OUT" \
  --yunet ../../models/face_detection_yunet_2023mar.onnx \
  --sface ../../models/face_recognition_sface.onnx \
  --min_face 70 \
  --keep_sim 0.54 \
  --drop_sim 0.44 \
  --min_keep 4 \
  --max_keep 60 \
  --align 1 \
  --report 1

echo "[OK] clusters cleaned into: $OUT"
echo "[NOTE] main.cpp must read from $OUT/*/aligned/"