#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "dataset/data.yaml"
DEFAULT_PROJECT = PROJECT_DIR / "artifacts/training"
DEFAULT_PRETRAINED = PROJECT_DIR / "models/yolo26s.pt"

# JetPack's OpenCV is built with GStreamer. Do not let Ultralytics replace it
# with the PyPI wheel while resolving optional dependencies.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def train_once(args, batch: int):
    import torch
    from ultralytics import YOLO

    from ball_runtime.jetson_compat import install_torchvision_nms_fallback

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the training environment")
    if install_torchvision_nms_fallback():
        print(
            "Xavier SM72: enabled CPU fallback for Ultralytics validation warmup NMS",
            file=sys.stderr,
        )
    model = YOLO(args.model)
    return model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=0,
        workers=args.workers,
        amp=True,
        patience=30,
        close_mosaic=15,
        seed=2026,
        deterministic=True,
        cache=False,
        project=str(args.project),
        name="yolo26s_ball_batch%d" % batch,
        exist_ok=True,
        optimizer="auto",
        hsv_h=0.02,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.10,
        scale=0.35,
        shear=2.0,
        perspective=0.0002,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.9,
        mixup=0.05,
        plots=True,
        verbose=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO26s on the ball dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            "dataset config not found; run prepare_dataset.py first: %s" % args.data
        )
    if not args.model.is_file():
        raise FileNotFoundError(
            "pretrained weights not found: %s (download yolo26s.pt first)"
            % args.model
        )

    selected_batch = args.batch
    try:
        results = train_once(args, selected_batch)
    except RuntimeError as exc:
        if selected_batch != 2 or "out of memory" not in str(exc).lower():
            raise
        print("CUDA OOM at batch=2; retrying once with batch=1", file=sys.stderr)
        import torch

        torch.cuda.empty_cache()
        selected_batch = 1
        results = train_once(args, selected_batch)

    save_dir = Path(results.save_dir)
    best = save_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError("training completed without best.pt: %s" % best)
    destination = PROJECT_DIR / "models/ball_yolo26s_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    summary = {
        "model": str(destination),
        "model_sha256": sha256(destination),
        "pretrained_model": str(args.model.resolve()),
        "pretrained_sha256": sha256(args.model),
        "source_best": str(best),
        "batch": selected_batch,
        "imgsz": args.imgsz,
        "epochs_requested": args.epochs,
        "save_dir": str(save_dir),
    }
    (PROJECT_DIR / "artifacts/training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
