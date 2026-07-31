#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECORD="${RECORD:-0}"
exec "$ROOT_DIR/scripts/run_web_snapshot.sh"
