#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${1:-$PARENT_DIR/${PROJECT_NAME}_nx_export_${STAMP}.tar.gz}"

tar \
    --exclude="$PROJECT_NAME/records" \
    --exclude="$PROJECT_NAME/tools/__pycache__" \
    --exclude="$PROJECT_NAME/.pytest_cache" \
    -czf "$OUTPUT" \
    -C "$PARENT_DIR" \
    "$PROJECT_NAME"

echo "$OUTPUT"
