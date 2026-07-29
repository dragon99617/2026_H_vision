#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${PROJECT_DIR}/models/yolo26s.pt"
EXPECTED_SHA256="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"

mkdir -p "$(dirname "${DESTINATION}")"
if [ ! -s "${DESTINATION}" ]; then
    curl -fL --retry 5 --max-time 1800 -o "${DESTINATION}" "${URL}"
fi

ACTUAL_SHA256="$(sha256sum "${DESTINATION}" | awk '{print $1}')"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo "YOLO26s SHA-256 mismatch: ${ACTUAL_SHA256}" >&2
    exit 1
fi
echo "YOLO26s weights verified: ${DESTINATION}"
