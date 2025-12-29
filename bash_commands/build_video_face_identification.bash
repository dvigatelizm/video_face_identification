#!/usr/bin/env bash
set -e

cd ~/video_face_identification
rm -rf build
mkdir -p build
cd build

cmake ..
make -j"$(nproc)"

echo "[OK] build done: $(pwd)/app"