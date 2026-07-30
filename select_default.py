#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = PROJECT_DIR / "artifacts/candidate_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible(candidate: dict) -> bool:
    metrics = candidate["metrics"]
    pytorch_metrics = candidate["pytorch_metrics"]
    onnx_metrics = candidate["onnx_metrics"]
    benchmark = candidate["benchmark"]
    all_metrics = (pytorch_metrics, onnx_metrics, metrics)
    if any(
        item["precision"] < 0.95
        or item["recall"] < 0.95
        or item["map50"] < 0.95
        for item in all_metrics
    ):
        return False
    if (
        candidate["comparison"]["tensorrt_recall_drop_vs_pytorch"] > 0.01
        or benchmark["pure_fps"] < 75.0
    ):
        return False
    live = candidate.get("live")
    if live is not None:
        return (
            live.get("capture_median_fps", 0.0) >= 59.0
            and live.get("inference_average_fps", 0.0) >= 58.0
            and live.get("latency_p95_ms", float("inf")) <= 35.0
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the highest-resolution accepted engine")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    candidates = sorted(
        report["candidates"],
        key=lambda item: item["width"] * item["height"],
        reverse=True,
    )
    accepted = [item for item in candidates if eligible(item)]
    if not accepted:
        raise RuntimeError("no engine meets the accuracy and performance gates")
    selected = accepted[0]
    provisional = "live" not in selected
    live = selected.get("live", {})
    engine_path = PROJECT_DIR / selected["engine"]
    config = {
        "engine": selected["engine"],
        "engine_sha256": sha256(engine_path),
        "confidence": report["confidence"],
        "input_shape": selected["metrics"]["input_shape"],
        "precision": selected["metrics"]["precision"],
        "recall": selected["metrics"]["recall"],
        "map50": selected["metrics"]["map50"],
        "tensorrt_recall_drop_vs_pytorch": selected["comparison"][
            "tensorrt_recall_drop_vs_pytorch"
        ],
        "pure_fps": selected["benchmark"]["pure_fps"],
        "provisional_without_live_camera": provisional,
        "provisional_without_10min_live": (
            live.get("duration_seconds", 0.0) < 600.0
        ),
        "provisional_without_tube_scene": (
            live.get("tube_pose_valid_frames", 0) == 0
        ),
        "selection_rule": (
            "highest resolution meeting P/R/mAP50 >= 0.95, pure FPS >= 75, "
            "and live gates when live metrics exist"
        ),
    }
    if live:
        config["live"] = live
    output = PROJECT_DIR / "models/default.json"
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
