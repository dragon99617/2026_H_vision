#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models/ball_yolo26s_best.pt"
CANDIDATES = ((480, 768), (416, 640))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export static YOLO26s ONNX candidates")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    from ultralytics import YOLO

    outputs = []
    for height, width in CANDIDATES:
        model = YOLO(str(args.model))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=(height, width),
                batch=1,
                dynamic=False,
                simplify=False,
                opset=args.opset,
                end2end=True,
                device=0,
            )
        )
        destination = (
            PROJECT_DIR
            / "models"
            / ("ball_yolo26s_%dx%d_end2end.onnx" % (width, height))
        )
        if exported.resolve() != destination.resolve():
            shutil.move(str(exported), str(destination))
        outputs.append(
            {
                "height": height,
                "width": width,
                "onnx": str(destination),
                "bytes": destination.stat().st_size,
            }
        )
    manifest = {"source_model": str(args.model.resolve()), "candidates": outputs}
    output = PROJECT_DIR / "artifacts/export_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
