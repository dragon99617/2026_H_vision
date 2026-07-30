#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ball_runtime.orbbec_sdk import OrbbecSdkCapture
from ball_runtime.tube_geometry import (
    TubeGeometryConfig,
    TubePoseEstimator,
    segment_white_tube,
)


def version_tuple(value: str):
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Gemini 336L SDK RGB-D profiles, calibration and rates"
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument(
        "--check-tube",
        action="store_true",
        help="also run white-tube segmentation and 3D pose diagnostics",
    )
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")

    capture = OrbbecSdkCapture()
    color_times = []
    depth_times = []
    previous_depth_id = 0
    depth_valid_ratios = []
    depth_medians_m = []
    tube_reasons = Counter()
    tube_valid_poses = 0
    tube_contour_lengths = []
    pose_estimator = TubePoseEstimator(TubeGeometryConfig())
    try:
        for _ in range(args.frames):
            color, depth, depth_id, dts, scale, cal = capture.read(1000)
            now = time.monotonic()
            color_times.append(now)
            if depth_id and depth_id != previous_depth_id:
                depth_times.append(now)
                previous_depth_id = depth_id
                valid = depth > 0
                depth_valid_ratios.append(
                    float(valid.sum()) / float(valid.size)
                )
                if valid.any():
                    depth_medians_m.append(
                        float(statistics.median(depth[valid])) * scale
                    )
                if args.check_tube:
                    contour = segment_white_tube(
                        color, pose_estimator.config
                    )
                    tube_contour_lengths.append(
                        contour.projected_length_px
                    )
                    pose = pose_estimator.update(
                        depth,
                        scale,
                        depth_id,
                        dts,
                        cal,
                        contour,
                    )
                    if pose.valid:
                        tube_valid_poses += 1
                        tube_reasons["valid"] += 1
                    else:
                        tube_reasons[pose.reason] += 1
            if color.shape != (800, 1280, 3):
                raise RuntimeError("unexpected RGB shape: %r" % (color.shape,))
            if depth is not None and depth.shape != (400, 640):
                raise RuntimeError("unexpected depth shape: %r" % (depth.shape,))
        color_intervals = [
            right - left
            for left, right in zip(color_times, color_times[1:])
            if right > left
        ]
        depth_intervals = [
            right - left
            for left, right in zip(depth_times, depth_times[1:])
            if right > left
        ]
        report = {
            "device": capture.device_info,
            "decoder": capture.decoder_backend,
            "color_shape": [800, 1280, 3],
            "depth_shape": [400, 640],
            "color_frames": len(color_times),
            "depth_frames": len(depth_times),
            "color_fps": (
                (len(color_times) - 1)
                / (color_times[-1] - color_times[0])
                if len(color_times) >= 2
                else 0.0
            ),
            "depth_fps": (
                (len(depth_times) - 1)
                / (depth_times[-1] - depth_times[0])
                if len(depth_times) >= 2
                else 0.0
            ),
            "color_interval_p50_ms": (
                statistics.median(color_intervals) * 1000.0
                if color_intervals
                else 0.0
            ),
            "depth_interval_p50_ms": (
                statistics.median(depth_intervals) * 1000.0
                if depth_intervals
                else 0.0
            ),
            "depth_scale_m": scale,
            "depth_global_valid_ratio_p50": (
                statistics.median(depth_valid_ratios)
                if depth_valid_ratios
                else 0.0
            ),
            "depth_global_median_distance_m": (
                statistics.median(depth_medians_m)
                if depth_medians_m
                else None
            ),
            "calibration": {
                "color": capture.calibration.color.__dict__,
                "depth": capture.calibration.depth.__dict__,
                "depth_to_color_rotation": (
                    capture.calibration.depth_to_color_rotation
                ),
                "depth_to_color_translation_m": (
                    capture.calibration.depth_to_color_translation_m
                ),
            },
        }
        if args.check_tube:
            report["tube_diagnostic"] = {
                "pose_attempts": len(depth_times),
                "valid_poses": tube_valid_poses,
                "valid_rate": (
                    tube_valid_poses / float(len(depth_times))
                    if depth_times
                    else 0.0
                ),
                "contour_length_px_p50": (
                    statistics.median(tube_contour_lengths)
                    if tube_contour_lengths
                    else 0.0
                ),
                "pose_reasons": dict(tube_reasons),
            }
        print(json.dumps(report, indent=2))
        if report["device"]["connection"] not in ("USB3.0", "USB3.1", "USB3.2"):
            print("WARNING: camera is not reporting a USB 3.x connection")
        firmware = version_tuple(report["device"]["firmware"])
        if firmware and firmware < (1, 2, 20):
            print("ERROR: Gemini 336L firmware must be at least 1.2.20")
            return 2
        if firmware and firmware < (1, 6, 0):
            print("WARNING: firmware 1.6.00 or newer is recommended")
        return 0
    finally:
        capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
