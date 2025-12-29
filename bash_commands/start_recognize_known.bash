#!/usr/bin/env bash
set -e

cd ~/video_face_identification/build

rm -rf ../output_batch/tmp

./app --dir ../data/test_videos/known \
      --out ../output_batch/tmp \
      --train_web "" \
      --train_auto ../data/train_auto_pred/clusters_clean \
      --step 5 \
      --conf 0.60 \
      --nms 0.40 \
      --topk 200 \
      --min_face 45 \
      --max_faces 25 \
      --unk_th 0.4 \
      --margin 0.01 \
      --strong 0.98