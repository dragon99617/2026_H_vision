#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(mode: str, metrics: dict) -> list:
    failures = []
    duration = float(metrics.get("duration_seconds", 0.0))
    if mode == "performance":
        requirements = (
            ("duration_seconds", duration, 600.0, ">="),
            (
                "capture_median_fps",
                float(metrics.get("capture_median_fps", 0.0)),
                59.0,
                ">=",
            ),
            (
                "inference_average_fps",
                float(metrics.get("inference_average_fps", 0.0)),
                58.0,
                ">=",
            ),
        )
        for name, actual, target, operator in requirements:
            if actual < target:
                failures.append("%s %.3f %s %.3f" % (name, actual, operator, target))
        latency = float(metrics.get("latency_p95_ms", float("inf")))
        if latency > 35.0:
            failures.append("latency_p95_ms %.3f <= 35.000" % latency)
    elif mode == "motion":
        if duration < 300.0:
            failures.append("duration_seconds %.3f >= 300.000" % duration)
        measured = float(metrics.get("measured_detection_coverage", 0.0))
        effective = float(metrics.get("effective_output_rate", 0.0))
        if measured < 0.97:
            failures.append("measured_detection_coverage %.5f >= 0.97000" % measured)
        if effective < 0.99:
            failures.append("effective_output_rate %.5f >= 0.99000" % effective)
        streak = int(metrics.get("max_predicted_streak_frames", 999999))
        if streak > 2:
            failures.append("max_predicted_streak_frames %d <= 2" % streak)
    else:
        if duration < 60.0:
            failures.append("duration_seconds %.3f >= 60.000" % duration)
        jitter = float(
            metrics.get("center_jitter_p95_radius_px", float("inf"))
        )
        if jitter > 2.0:
            failures.append("center_jitter_p95_radius_px %.3f <= 2.000" % jitter)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check live NX acceptance metrics")
    parser.add_argument(
        "--mode",
        choices=("performance", "motion", "static"),
        required=True,
    )
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    failures = check(args.mode, metrics)
    report = {
        "mode": args.mode,
        "passed": not failures,
        "failures": failures,
        "metrics": str(args.metrics),
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
