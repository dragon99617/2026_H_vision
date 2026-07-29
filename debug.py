#!/usr/bin/env python3
from ball_runtime.bootstrap import ensure_gstreamer_tls_preload

ensure_gstreamer_tls_preload()

from ball_runtime.cli import debug_main


if __name__ == "__main__":
    raise SystemExit(debug_main())
