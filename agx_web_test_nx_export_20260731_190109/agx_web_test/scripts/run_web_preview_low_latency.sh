#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.env"

SERVER="$ROOT_DIR/tools/low_latency_mjpeg_server.py"

echo "Starting low-latency AGX camera web preview"
echo "  camera:  $CAMERA_DEVICE"
echo "  mode:    ${WIDTH}x${HEIGHT}@${FPS}"
echo "  quality: $JPEG_QUALITY"
echo "  url:     http://<AGX_IP>:${HTTP_PORT}/"
echo
echo "This tries raw V4L2 MJPEG first, then falls back to OpenCV low-buffer mode."
echo "Stop with Ctrl-C."

exec python3 "$SERVER" \
    --device "$CAMERA_DEVICE" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    --quality "$JPEG_QUALITY" \
    --buffers 2
