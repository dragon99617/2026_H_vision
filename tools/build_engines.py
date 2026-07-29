#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts/engine_build"
TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
CANDIDATES = ((768, 480), (640, 416))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(item) for item in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build both target-specific FP16 engines")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    args = parser.parse_args()
    if not TRTEXEC.is_file():
        raise FileNotFoundError(TRTEXEC)
    import tensorrt

    manifest = {"tensorrt": tensorrt.__version__, "candidates": []}
    for width, height in CANDIDATES:
        onnx = MODELS_DIR / (
            "ball_yolo26s_%dx%d_end2end.onnx" % (width, height)
        )
        if not onnx.is_file():
            raise FileNotFoundError(onnx)
        engine = MODELS_DIR / (
            "ball_yolo26s_%dx%d_fp16.engine" % (width, height)
        )
        cache = Path(str(engine) + ".cache")
        with tempfile.TemporaryDirectory(prefix="ball-yolo26-") as temp:
            patched = Path(temp) / onnx.name
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "tools/patch_trt84_onnx.py"),
                    str(onnx),
                    str(patched),
                ],
                check=True,
            )
            build_command = [
                TRTEXEC,
                "--onnx=%s" % patched,
                "--saveEngine=%s" % engine,
                "--fp16",
                "--memPoolSize=workspace:%d" % args.workspace_mib,
                "--buildOnly",
                "--minTiming=1",
                "--avgTiming=1",
                "--timingCacheFile=%s" % cache,
            ]
            run_logged(
                build_command,
                ARTIFACTS_DIR / ("build_%dx%d.log" % (width, height)),
            )
        manifest["candidates"].append(
            {
                "width": width,
                "height": height,
                "onnx": str(onnx),
                "onnx_sha256": sha256(onnx),
                "engine": str(engine),
                "engine_bytes": engine.stat().st_size,
                "engine_sha256": sha256(engine),
                "timing_cache": str(cache),
            }
        )
    output = PROJECT_DIR / "artifacts/engine_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
