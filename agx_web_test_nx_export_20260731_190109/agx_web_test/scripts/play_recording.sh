#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.env"

echo "Open the web playback page on your phone/browser:"
echo
echo "  http://<AGX_IP>:${HTTP_PORT}/records"
echo
echo "Example from your current phone-hotspot network:"
echo "  http://192.168.43.9:${HTTP_PORT}/records"
echo
echo "This project records frame folders, not normal video files."
echo "Use the web page to play/pause and drag the progress slider."
