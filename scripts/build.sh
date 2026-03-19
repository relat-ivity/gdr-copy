#!/usr/bin/env bash
# build.sh  —  configure and compile gdr_copy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"

echo "[build] Root: ${ROOT}"
echo "[build] Build dir: ${BUILD}"

cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -G Ninja 2>/dev/null \
    || cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build "${BUILD}" --parallel $(nproc)

echo ""
echo "Build complete."
echo "  Demo:  sudo ${BUILD}/demo [gpu_id] [nic_name]"
echo "  Bench: sudo ${BUILD}/bench [gpu_id] [nic_name]"
