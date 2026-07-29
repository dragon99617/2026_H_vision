#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ball_runtime.bootstrap import ensure_gstreamer_tls_preload

ensure_gstreamer_tls_preload()

import cv2

from ball_runtime.cli import load_default_confidence, load_default_engine
from ball_runtime.orbbec_sdk import OrbbecSdkCapture
from ball_runtime.tensorrt_detector import TensorRTDetector
from ball_runtime.tube_geometry import (
    TubeGeometryConfig,
    TubePoseEstimator,
    segment_white_tube,
)
from ball_runtime.types import FramePacket


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yolo_line(detection, width: int, height: int) -> str:
    center_x, center_y = detection.center
    return "0 %.8f %.8f %.8f %.8f\n" % (
        center_x / width,
        center_y / height,
        detection.width / width,
        detection.height / height,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect continuous RGB-D half-pipe segments for retraining"
    )
    parser.add_argument("--segment", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--mode", choices=("positive", "negative"), required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "tube_data/raw"
    )
    parser.add_argument("--engine", default=load_default_engine())
    parser.add_argument("--conf", type=float, default=min(0.5, load_default_confidence()))
    parser.add_argument("--no-depth-save", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.sample_fps <= 0:
        parser.error("--count and --sample-fps must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.segment):
        parser.error("--segment may contain only letters, digits, _ and -")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be in 0..1")
    segment_dir = args.output / args.segment
    if segment_dir.exists():
        raise RuntimeError(
            "segment already exists; choose a new --segment: %s" % segment_dir
        )
    image_dir = segment_dir / "images"
    label_dir = segment_dir / "labels"
    depth_dir = segment_dir / "depth"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    if not args.no_depth_save:
        depth_dir.mkdir(parents=True)

    detector = (
        TensorRTDetector(args.engine, args.conf)
        if args.mode == "positive"
        else None
    )
    capture = OrbbecSdkCapture()
    pose_estimator = TubePoseEstimator(TubeGeometryConfig())
    records = []
    next_sample = time.monotonic()
    captured = 0
    latest_depth_id = 0
    try:
        while captured < args.count:
            color, depth, depth_id, depth_ts, depth_scale, calibration = capture.read(
                1000
            )
            now = time.monotonic()
            if now < next_sample:
                if args.preview:
                    cv2.imshow("tube dataset collection", color)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                continue
            next_sample = now + 1.0 / args.sample_fps
            stem = "%s_%06d" % (args.segment, captured)
            image_path = image_dir / (stem + ".jpg")
            label_path = label_dir / (stem + ".txt")
            if not cv2.imwrite(
                str(image_path), color, (cv2.IMWRITE_JPEG_QUALITY, 95)
            ):
                raise RuntimeError("failed to save %s" % image_path)

            detection = None
            if detector is not None:
                result = detector.detect(
                    FramePacket(
                        frame_id=captured + 1,
                        captured_monotonic=now,
                        image=color,
                    )
                )
                if result.detections:
                    detection = max(
                        result.detections,
                        key=lambda item: item.confidence,
                    )
            if detection is not None:
                label_path.write_text(
                    yolo_line(detection, color.shape[1], color.shape[0]),
                    encoding="utf-8",
                )
            else:
                label_path.write_text("", encoding="utf-8")

            depth_path = None
            if depth is not None and not args.no_depth_save:
                depth_path = depth_dir / (stem + ".png")
                if not cv2.imwrite(str(depth_path), depth):
                    raise RuntimeError("failed to save %s" % depth_path)
            contour = segment_white_tube(color, pose_estimator.config)
            pose = None
            if (
                depth is not None
                and depth_id > 0
                and depth_id != latest_depth_id
            ):
                latest_depth_id = depth_id
                pose = pose_estimator.update(
                    depth,
                    depth_scale,
                    depth_id,
                    depth_ts,
                    calibration,
                    contour,
                    detection,
                )
            records.append(
                {
                    "image": str(image_path.relative_to(segment_dir)),
                    "label": str(label_path.relative_to(segment_dir)),
                    "depth": (
                        str(depth_path.relative_to(segment_dir))
                        if depth_path is not None
                        else None
                    ),
                    "image_sha256": sha256(image_path),
                    "depth_frame_id": depth_id,
                    "auto_ball_label": detection is not None,
                    "ball_confidence": (
                        detection.confidence if detection is not None else 0.0
                    ),
                    "tube_contour_valid": contour.valid,
                    "tube_pose_valid": bool(pose is not None and pose.valid),
                    "tube_pose_reason": (
                        pose.reason if pose is not None else "reused depth frame"
                    ),
                    "pitch_degrees": (
                        pose.pitch_degrees
                        if pose is not None and pose.valid
                        else None
                    ),
                }
            )
            captured += 1
            print(
                "\r%d/%d auto-label=%s tube=%s"
                % (
                    captured,
                    args.count,
                    "yes" if detection is not None else "NO",
                    "valid"
                    if pose is not None and pose.valid
                    else "invalid",
                ),
                end="",
                flush=True,
            )
            if args.preview:
                preview = color.copy()
                if detection is not None:
                    cv2.rectangle(
                        preview,
                        (int(detection.x1), int(detection.y1)),
                        (int(detection.x2), int(detection.y2)),
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow("tube dataset collection", preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        print()
        capture.close()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()

    manifest = {
        "segment": args.segment,
        "split": args.split,
        "mode": args.mode,
        "continuous_segment": True,
        "captured_images": len(records),
        "auto_labels_require_manual_review": args.mode == "positive",
        "device": capture.device_info,
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
        "records": records,
    }
    (segment_dir / "segment.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("segment", "split", "mode", "captured_images")}))
    return 0 if len(records) == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
