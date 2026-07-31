#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.env"

record_root="$ROOT_DIR/$RECORD_DIR"

if [ ! -d "$record_root" ]; then
    echo "No record directory yet: $record_root"
    exit 0
fi

for dir in "$record_root"/*; do
    [ -d "$dir" ] || continue
    frames=$(find "$dir" -maxdepth 1 -name '*.jpg' | wc -l)
    printf '%s frames  %s\n' "$frames" "$dir"
done | sort
