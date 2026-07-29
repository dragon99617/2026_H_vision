from __future__ import annotations

import os
import sys
from typing import Optional


_GLDISPATCH_CANDIDATES = (
    "/lib/aarch64-linux-gnu/libGLdispatch.so.0",
    "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0",
)


def _find_gldispatch() -> Optional[str]:
    for path in _GLDISPATCH_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _already_preloaded(library: str, value: str) -> bool:
    library_realpath = os.path.realpath(library)
    library_name = os.path.basename(library)
    for entry in value.split(":"):
        if not entry:
            continue
        if os.path.basename(entry) == library_name:
            return True
        if os.path.exists(entry) and os.path.realpath(entry) == library_realpath:
            return True
    return False


def ensure_gstreamer_tls_preload() -> None:
    """Restart once with libGLdispatch preloaded before OpenCV/TensorRT imports."""
    library = _find_gldispatch()
    if library is None:
        return

    current_preload = os.environ.get("LD_PRELOAD", "")
    if _already_preloaded(library, current_preload):
        return

    environment = os.environ.copy()
    environment["LD_PRELOAD"] = (
        library if not current_preload else "%s:%s" % (library, current_preload)
    )
    os.execve(
        sys.executable,
        [sys.executable] + sys.argv,
        environment,
    )
