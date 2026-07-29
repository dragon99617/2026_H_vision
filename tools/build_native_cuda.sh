#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${PROJECT_DIR}/native/ball_cuda.cu"
OUTPUT="${PROJECT_DIR}/native/libball_cuda.so"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

if [ ! -x "${NVCC}" ]; then
    echo "nvcc not found: ${NVCC}" >&2
    exit 1
fi

"${NVCC}" \
    -O3 \
    --use_fast_math \
    -Xcompiler=-fPIC \
    -shared \
    -gencode arch=compute_72,code=sm_72 \
    "${SOURCE}" \
    -o "${OUTPUT}"

echo "Built ${OUTPUT}"
