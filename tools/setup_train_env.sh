#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv-train"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DOWNLOAD_DIR="${PROJECT_DIR}/.cache/wheels"
PORT_FILE="$(mktemp)"
PROXY_PID=""

cleanup() {
  if [ -n "${PROXY_PID}" ]; then
    kill "${PROXY_PID}" 2>/dev/null || true
    wait "${PROXY_PID}" 2>/dev/null || true
  fi
  rm -f "${PORT_FILE}"
}
trap cleanup EXIT

"${PYTHON_BIN}" -m venv --clear --without-pip --system-site-packages "${VENV_DIR}"
mkdir -p "${DOWNLOAD_DIR}"

"${PYTHON_BIN}" "${PROJECT_DIR}/tools/curl_pypi_proxy.py" \
  --port-file "${PORT_FILE}" &
PROXY_PID=$!
for _ in $(seq 1 100); do
  if [ -s "${PORT_FILE}" ]; then
    break
  fi
  sleep 0.1
done
if [ ! -s "${PORT_FILE}" ]; then
  echo "local PyPI bridge failed to start" >&2
  exit 1
fi
PYPI_PORT="$(tr -d '\n' <"${PORT_FILE}")"
PIP_INDEX="http://127.0.0.1:${PYPI_PORT}/simple"
export PIP_INDEX_URL="${PIP_INDEX}"
export PIP_TRUSTED_HOST="127.0.0.1"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
PIP=("${VENV_DIR}/bin/python" -m pip)

TORCH_WHEEL="${DOWNLOAD_DIR}/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
VISION_WHEEL="${DOWNLOAD_DIR}/torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl"
if [ ! -s "${TORCH_WHEEL}" ]; then
  curl -fL --retry 5 --max-time 1800 \
    -o "${TORCH_WHEEL}" \
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
fi
if [ ! -s "${VISION_WHEEL}" ]; then
  curl -fL --retry 5 --max-time 1800 \
    -o "${VISION_WHEEL}" \
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl"
fi

"${PIP[@]}" install \
  "filelock" "sympy==1.13.3" "networkx==3.1" "jinja2==3.1.6" "fsspec==2025.3.0"
"${PIP[@]}" install --no-deps "${TORCH_WHEEL}"
"${PIP[@]}" install --no-deps "${VISION_WHEEL}"

# Keep JetPack's GStreamer-enabled OpenCV by intentionally omitting the
# opencv-python dependency from the isolated training environment.
"${PIP[@]}" install --no-deps "ultralytics==8.4.102"
"${PIP[@]}" install \
  "numpy==1.24.4" \
  "matplotlib==3.7.5" \
  "pillow==9.5.0" \
  "requests==2.31.0" \
  "scipy==1.10.1" \
  "psutil==5.9.8" \
  "pandas==2.0.3" \
  "seaborn==0.13.2" \
  "tqdm==4.67.1" \
  "nvidia-ml-py==12.575.51" \
  "polars==0.20.31" \
  "ultralytics-thop==2.0.18" \
  "onnx==1.14.1" \
  "onnxruntime==1.16.3" \
  "protobuf==5.29.6"

YOLO_AUTOINSTALL=false "${VENV_DIR}/bin/python" - <<'PY'
import cv2
import torch
import ultralytics

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch verification failed")
PY

echo "Training environment ready: ${VENV_DIR}"
