#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from ball_runtime.metrics import (
    ImagePredictions,
    load_yolo_ground_truth,
    select_threshold,
    summarize,
)
from ball_runtime.types import Detection

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models/ball_yolo26s_best.pt"
DEFAULT_DATASET = PROJECT_DIR / "dataset"
CANDIDATES = ((480, 768), (416, 640))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def collect(model, dataset: Path, images: Sequence[Path], imgsz) -> list:
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=imgsz,
        conf=0.001,
        device=0,
        verbose=False,
        stream=False,
    )
    records = []
    for image, result in zip(images, results):
        predictions = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            for box, score in zip(xyxy, confidence):
                predictions.append(
                    Detection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(score),
                        class_id=0,
                    )
                )
        label = dataset / "labels" / image.parent.name / (image.stem + ".txt")
        records.append(
            ImagePredictions(
                image=image,
                ground_truth=load_yolo_ground_truth(image, label),
                predictions=tuple(predictions),
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PT model and select confidence")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    from ultralytics import YOLO

    from ball_runtime.jetson_compat import install_torchvision_nms_fallback

    install_torchvision_nms_fallback()
    model = YOLO(str(args.model))
    val_images = sorted((args.dataset / "images/val").glob("*.jpg"))
    test_images = sorted((args.dataset / "images/test").glob("*.jpg"))
    validation = collect(model, args.dataset, val_images, args.imgsz)
    selected = select_threshold(validation, target_recall=0.97)
    test = collect(model, args.dataset, test_images, args.imgsz)
    candidate_tests = {}
    for height, width in CANDIDATES:
        records = collect(model, args.dataset, test_images, (height, width))
        candidate_tests["%dx%d" % (width, height)] = summarize(
            records,
            selected["threshold"],
        )
    report = {
        "model": str(args.model.resolve()),
        "imgsz": args.imgsz,
        "selected_confidence": selected,
        "validation": summarize(validation, selected["threshold"]),
        "test": summarize(test, selected["threshold"]),
        "test_candidates": candidate_tests,
    }
    output = PROJECT_DIR / "artifacts/pytorch_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (PROJECT_DIR / "models/default_conf.json").write_text(
        json.dumps(
            {
                "confidence": selected["threshold"],
                "selection_rule": selected["selection_rule"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
