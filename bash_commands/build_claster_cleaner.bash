#!/usr/bin/env bash
set -e

cd ~/video_face_identification/tools
rm -rf build_cleaner
mkdir -p build_cleaner
cd build_cleaner

cmake ..
make -j"$(nproc)"

echo "[OK] cleaner build done: $(pwd)/cluster_cleaner"