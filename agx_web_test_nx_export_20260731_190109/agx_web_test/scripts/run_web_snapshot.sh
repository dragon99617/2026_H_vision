#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.env"

SNAPSHOT_INTERVAL_MS="${SNAPSHOT_INTERVAL_MS:-50}"
TEST_NAME="${TEST_NAME:-run}"
RECORD_ARGS=()
if [ "$RECORD" = "1" ]; then
    RECORD_ARGS=(
        --record
        --record-dir "$ROOT_DIR/$RECORD_DIR"
        --record-prefix "$RECORD_PREFIX"
        --record-fps "$RECORD_FPS"
        --mp4-bitrate-kbps "$MP4_BITRATE_KBPS"
        --test-name "$TEST_NAME"
    )
fi

echo "Starting stable-latency AGX web preview"
echo "  camera:   $CAMERA_DEVICE"
echo "  mode:     ${WIDTH}x${HEIGHT}@${FPS}"
echo "  quality:  $JPEG_QUALITY"
echo "  interval: ${SNAPSHOT_INTERVAL_MS} ms"
echo "  control:  ${CONTROL_SOCKET}"
echo "  record:   ${RECORD}"
if [ "$RECORD" = "1" ]; then
    echo "  mp4:      ${MP4_BITRATE_KBPS} kbps after stop recording"
fi
echo "  url:      http://<AGX_IP>:${HTTP_PORT}/"
echo
echo "This mode always fetches the latest frame and drops old frames."
echo "It is usually steadier on phone browsers than MJPEG streaming."
echo "Stop with Ctrl-C."

exec python3 "$ROOT_DIR/tools/snapshot_server.py" \
    --device "$CAMERA_DEVICE" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" \
    --quality "$JPEG_QUALITY" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    --interval-ms "$SNAPSHOT_INTERVAL_MS" \
    --control-socket "$CONTROL_SOCKET" \
    --control-timeout-ms "$CONTROL_TIMEOUT_MS" \
    "${RECORD_ARGS[@]}"
