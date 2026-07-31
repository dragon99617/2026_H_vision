#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.env"

echo "== Network =="
ip -br addr || true
echo

echo "== USB =="
lsusb | grep -i -E '2bc5|orbbec' || echo "No Orbbec USB device found in lsusb output."
echo

echo "== Video devices =="
for name in /sys/class/video4linux/video*/name; do
    [ -e "$name" ] || continue
    printf '%s: ' "$name"
    cat "$name"
done
echo

echo "== Python OpenCV =="
if python3 -c 'import cv2; print(cv2.__version__)' >/dev/null 2>&1; then
    python3 -c 'import cv2; print("OK  OpenCV", cv2.__version__)'
else
    echo "MISS python3 OpenCV"
fi
echo

echo "== Selected =="
echo "Camera: $CAMERA_DEVICE"
echo "Mode:   ${WIDTH}x${HEIGHT}@${FPS}"
echo "URL:    http://192.168.88.1:${HTTP_PORT}/"
