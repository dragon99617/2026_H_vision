#!/usr/bin/env python3
from ball_runtime.bootstrap import ensure_gstreamer_tls_preload

ensure_gstreamer_tls_preload()

from ball_runtime.cli import run_main


if __name__ == "__main__":
    raise SystemExit(run_main(position_mode_default="rgb-contour"))
