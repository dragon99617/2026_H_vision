#!/usr/bin/env python3
"""Check that the Gemini 336L RGB MJPEG V4L2 node is usable."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from ball_runtime.camera import discover_orbbec_rgb_device


def main() -> int:
    try:
        device = discover_orbbec_rgb_device()
    except RuntimeError as exc:
        print("camera check failed: %s" % exc, file=sys.stderr)
        return 1
    print("Orbbec RGB MJPEG capture node: %s" % device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
