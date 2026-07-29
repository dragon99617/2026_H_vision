#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import cv2

from ball_runtime.metrics import (
    ImagePredictions,
    load_yolo_ground_truth,
    summarize,
)
from ball_runtime.tensorrt_detector import TensorRTDetector
from ball_runtime.types import FramePacket

PROJECT_DIR = Path(__file__).resolve().parent
TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
CANDIDATES = ((768, 480), (640, 416))
THROUGHPUT_PATTERN = re.compile(r"Throughput:\s+([0-9.]+)\s+qps")
GPU_MEAN_PATTERN = re.compile(
    r"GPU Compute Time:.*?mean = ([0-9.]+) ms"
)


def read_confidence() -> float:
    path = PROJECT_DIR / "models/default_conf.json"
    if not path.is_file():
        return 0.25
    return float(json.loads(path.read_text(encoding="utf-8"))["confidence"])


def benchmark(engine: Path, duration: int) -> dict:
    command = [
        str(TRTEXEC),
        "--loadEngine=%s" % engine,
        "--warmUp=1000",
        "--duration=%d" % duration,
        "--useCudaGraph",
        "--noDataTransfers",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    throughput = THROUGHPUT_PATTERN.search(completed.stdout)
    gpu_mean = GPU_MEAN_PATTERN.search(completed.stdout)
    if throughput is None or gpu_mean is None:
        raise RuntimeError("failed to parse trtexec output")
    return {
        "pure_fps": float(throughput.group(1)),
        "gpu_mean_ms": float(gpu_mean.group(1)),
        "command": command,
    }


def evaluate(engine: Path, confidence: float) -> dict:
    detector = TensorRTDetector(str(engine), confidence=0.001)
    try:
        images = sorted((PROJECT_DIR / "dataset/images/test").glob("*.jpg"))
        records = []
        for index, image_path in enumerate(images, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("cannot read %s" % image_path)
            result = detector.detect(
                FramePacket(
                    frame_id=index,
                    captured_monotonic=time.monotonic(),
                    image=image,
                )
            )
            label = (
                PROJECT_DIR
                / "dataset/labels/test"
                / (image_path.stem + ".txt")
            )
            records.append(
                ImagePredictions(
                    image=image_path,
                    ground_truth=load_yolo_ground_truth(image_path, label),
                    predictions=result.detections,
                )
            )
        metrics = summarize(records, confidence)
        metrics["input_shape"] = list(detector.input_shape)
        metrics["fast_path"] = detector.fast_path
        return metrics
    finally:
        detector.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark and evaluate both FP16 engines")
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()
    confidence = read_confidence()
    pytorch_report = json.loads(
        (PROJECT_DIR / "artifacts/pytorch_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    onnx_report = json.loads(
        (PROJECT_DIR / "artifacts/onnx_metrics.json").read_text(encoding="utf-8")
    )
    report = {"confidence": confidence, "candidates": []}
    for width, height in CANDIDATES:
        key = "%dx%d" % (width, height)
        engine = PROJECT_DIR / "models" / (
            "ball_yolo26s_%dx%d_fp16.engine" % (width, height)
        )
        if not engine.is_file():
            raise FileNotFoundError(engine)
        pytorch_metrics = pytorch_report["test_candidates"][key]
        onnx_metrics = onnx_report["candidates"][key]["metrics"]
        tensorrt_metrics = evaluate(engine, confidence)
        item = {
            "width": width,
            "height": height,
            "engine": str(engine.relative_to(PROJECT_DIR)),
            "benchmark": benchmark(engine, args.duration),
            "pytorch_metrics": pytorch_metrics,
            "onnx_metrics": onnx_metrics,
            "metrics": tensorrt_metrics,
            "comparison": {
                "onnx_recall_drop_vs_pytorch": (
                    pytorch_metrics["recall"] - onnx_metrics["recall"]
                ),
                "tensorrt_recall_drop_vs_pytorch": (
                    pytorch_metrics["recall"] - tensorrt_metrics["recall"]
                ),
            },
        }
        live_path = PROJECT_DIR / "artifacts" / (
            "live_%dx%d.json" % (width, height)
        )
        if not live_path.is_file():
            smoke_path = PROJECT_DIR / "artifacts" / (
                "rgbd_smoke%s.json"
                % ("" if (width, height) == (768, 480) else "_%dx%d" % (width, height))
            )
            if smoke_path.is_file():
                live_path = smoke_path
        if live_path.is_file():
            item["live"] = json.loads(live_path.read_text(encoding="utf-8"))
        report["candidates"].append(item)
    output = PROJECT_DIR / "artifacts/candidate_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
