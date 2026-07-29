#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ball_runtime.metrics import ImagePredictions, load_yolo_ground_truth, summarize
from ball_runtime.tensorrt_detector import parse_end2end_output, preprocess

PROJECT_DIR = Path(__file__).resolve().parent
CANDIDATES = ((768, 480), (640, 416))


def read_confidence() -> float:
    path = PROJECT_DIR / "models/default_conf.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return float(json.loads(path.read_text(encoding="utf-8"))["confidence"])


def evaluate(model_path: Path, confidence: float) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("expected exactly one ONNX input and one output")
    shape = tuple(int(value) for value in inputs[0].shape)
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise RuntimeError("unexpected ONNX input shape: %s" % (shape,))
    input_height, input_width = shape[2:]
    records = []
    timings = []
    images = sorted((PROJECT_DIR / "dataset/images/test").glob("*.jpg"))
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("cannot read %s" % image_path)
        blob, scale, pad_x, pad_y = preprocess(
            image,
            input_width=input_width,
            input_height=input_height,
            dtype=np.float32,
        )
        started = time.monotonic()
        raw = session.run([outputs[0].name], {inputs[0].name: blob})[0]
        timings.append((time.monotonic() - started) * 1000.0)
        detections = parse_end2end_output(
            raw,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence=0.001,
        )
        label = PROJECT_DIR / "dataset/labels/test" / (image_path.stem + ".txt")
        records.append(
            ImagePredictions(
                image=image_path,
                ground_truth=load_yolo_ground_truth(image_path, label),
                predictions=tuple(detections),
            )
        )
    metrics = summarize(records, confidence)
    metrics["input_shape"] = list(shape)
    metrics["cpu_inference_mean_ms"] = float(np.mean(timings))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate static end-to-end ONNX models")
    parser.parse_args()
    confidence = read_confidence()
    report = {"confidence": confidence, "candidates": {}}
    for width, height in CANDIDATES:
        key = "%dx%d" % (width, height)
        model = PROJECT_DIR / "models" / (
            "ball_yolo26s_%s_end2end.onnx" % key
        )
        if not model.is_file():
            raise FileNotFoundError(model)
        report["candidates"][key] = {
            "onnx": str(model.relative_to(PROJECT_DIR)),
            "metrics": evaluate(model, confidence),
        }
    output = PROJECT_DIR / "artifacts/onnx_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
