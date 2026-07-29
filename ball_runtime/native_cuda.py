from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY = PROJECT_DIR / "native/libball_cuda.so"


class NativeCudaLibrary:
    def __init__(self, path: Optional[str] = None) -> None:
        library_path = Path(path) if path is not None else DEFAULT_LIBRARY
        if not library_path.is_file():
            raise FileNotFoundError(
                "native CUDA helper not found: %s; run tools/build_native_cuda.sh"
                % library_path
            )
        self.path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.ball_launch_preprocess.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.ball_launch_preprocess.restype = ctypes.c_int

    def launch_preprocess(
        self,
        source_device,
        source_width: int,
        source_height: int,
        source_stride: int,
        output_device,
        output_width: int,
        output_height: int,
        stream,
    ) -> None:
        status = self.lib.ball_launch_preprocess(
            source_device,
            source_width,
            source_height,
            source_stride,
            output_device,
            output_width,
            output_height,
            stream,
        )
        if status != 0:
            raise RuntimeError("CUDA image preprocessing failed with status %d" % status)
