from __future__ import annotations

import argparse
import csv
import json
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera import CameraWorker
from .datagram_writer import DatagramWriter, parse_udp_endpoint
from .inference import InferenceWorker
from .latest import LatestValue
from .rgb_tube_position_detector import RgbTubePositionDetector
from .rgb_tube_worker import RgbTubeWorker
from .serial_protocol import encode_result
from .serial_writer import SerialWriter
from .single_ball_tracker import SingleBallTrackingDetector
from .tube_geometry import TubeGeometryConfig
from .tube_position_detector import TubePositionDetector
from .tube_pose_worker import TubePoseWorker
from .types import DetectionResult, FramePacket, RgbTubeState, TubeState
from .visualize import draw_debug

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "models/default.json"
DEFAULT_ENGINE = PROJECT_DIR / "models/ball_yolo26s_768x480_fp16.engine"


def load_default_engine() -> str:
    if DEFAULT_CONFIG.is_file():
        try:
            data = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            raw = Path(str(data["engine"]))
            path = raw if raw.is_absolute() else PROJECT_DIR / raw
            return str(path)
        except (OSError, KeyError, TypeError, ValueError):
            pass
    return str(DEFAULT_ENGINE)


def load_default_confidence() -> float:
    if DEFAULT_CONFIG.is_file():
        try:
            data = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            return float(data["confidence"])
        except (OSError, KeyError, TypeError, ValueError):
            pass
    confidence_path = PROJECT_DIR / "models/default_conf.json"
    if confidence_path.is_file():
        try:
            data = json.loads(confidence_path.read_text(encoding="utf-8"))
            return float(data["confidence"])
        except (OSError, KeyError, TypeError, ValueError):
            pass
    return 0.25


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def confidence_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def unit_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def add_common_arguments(
    parser: argparse.ArgumentParser,
    serial_default: str,
    position_mode_default: str = "rgbd",
) -> None:
    parser.add_argument("--engine", default=load_default_engine())
    parser.add_argument("--device", default="auto", help="V4L2 path or auto")
    parser.add_argument(
        "--camera-backend",
        choices=("sdk", "v4l2"),
        default="sdk",
        help="sdk provides RGB-D; v4l2 is color-only compatibility mode",
    )
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=800)
    parser.add_argument("--fps", type=positive_int, default=60)
    parser.add_argument("--depth-width", type=positive_int, default=640)
    parser.add_argument("--depth-height", type=positive_int, default=400)
    parser.add_argument("--depth-fps", type=positive_int, default=30)
    parser.add_argument(
        "--position-mode",
        choices=("rgbd", "rgb-contour"),
        default=position_mode_default,
        help="3D RGB-D axis or 2D RGB contour projection",
    )
    parser.add_argument(
        "--color-auto-exposure",
        action="store_true",
        help="keep color auto exposure enabled instead of fixed manual exposure",
    )
    parser.add_argument(
        "--color-exposure",
        type=positive_int,
        default=None,
        help="manual Orbbec color exposure in SDK units",
    )
    parser.add_argument(
        "--color-exposure-scale",
        type=float,
        default=0.30,
        help="manual exposure as a fraction of the SDK default when --color-exposure is omitted",
    )
    parser.add_argument(
        "--color-gain",
        type=nonnegative_int,
        default=None,
        help="optional fixed Orbbec color gain in SDK units",
    )
    parser.add_argument(
        "--conf",
        type=confidence_value,
        default=load_default_confidence(),
    )
    parser.add_argument("--serial", default=serial_default, help="path, auto, or off")
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--baud", type=positive_int, default=921600)
    parser.add_argument(
        "--control-udp",
        default="off",
        help="Send timestamped tube-v3 to the NX controller at HOST:PORT",
    )
    parser.add_argument("--stats-interval", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Write live performance metrics on exit",
    )
    parser.add_argument(
        "--position-csv",
        type=Path,
        default=None,
        help="Record every output for tube-position validation",
    )
    parser.add_argument("--reference-position-cm", type=float, default=None)
    parser.add_argument("--reference-pitch-deg", type=float, default=None)
    parser.add_argument("--track-distance", type=float, default=80.0)
    parser.add_argument("--track-alpha", type=unit_value, default=0.75)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument(
        "--protocol",
        choices=("tube-v3", "tube-v2", "pixel-v1"),
        default="tube-v2",
    )
    parser.add_argument("--tube-length-cm", type=float, default=25.0)
    parser.add_argument("--tube-pose-max-age-ms", type=float, default=100.0)
    parser.add_argument(
        "--positive-end",
        choices=("image-right",),
        default="image-right",
    )
    parser.add_argument("--tube-min-projection-px", type=float, default=400.0)
    parser.add_argument("--tube-min-depth-ratio", type=float, default=0.60)
    parser.add_argument("--tube-max-rms-mm", type=float, default=4.0)
    parser.add_argument("--tube-white-min-gray", type=int, default=105)
    parser.add_argument("--tube-white-max-saturation", type=int, default=150)
    parser.add_argument("--rgb-tube-alpha", type=unit_value, default=0.35)


def read_power_mode() -> str:
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            timeout=2.0,
        )
        for line in result.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":", 1)[-1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


class Runtime:
    def __init__(self, args) -> None:
        self.args = args
        self.stop_event = threading.Event()
        self.frames: LatestValue[FramePacket] = LatestValue()
        self.results: LatestValue[DetectionResult] = LatestValue()
        self.tube_states: LatestValue[TubeState] = LatestValue()
        self.rgb_tube_states: LatestValue[RgbTubeState] = LatestValue()
        self.detector = None
        self.camera: Optional[CameraWorker] = None
        self.inference: Optional[InferenceWorker] = None
        self.serial_writer: Optional[SerialWriter] = None
        self.control_writer: Optional[DatagramWriter] = None
        self.tube_pose_worker: Optional[TubePoseWorker] = None
        self.rgb_tube_worker: Optional[RgbTubeWorker] = None
        self.power_mode = read_power_mode()
        self.position_csv_handle = None
        self.position_csv_writer = None
        self.position_csv_rows = 0
        self.position_csv_armed = False

    def start(self) -> None:
        ball_detector = SingleBallTrackingDetector(
            engine_path=self.args.engine,
            confidence=self.args.conf,
            match_distance=self.args.track_distance,
            smooth_alpha=self.args.track_alpha,
            hold_frames=self.args.hold_frames,
        )
        if self.args.protocol in ("tube-v2", "tube-v3"):
            geometry_config = TubeGeometryConfig(
                tube_length_cm=self.args.tube_length_cm,
                min_projected_length_px=self.args.tube_min_projection_px,
                min_valid_bin_ratio=self.args.tube_min_depth_ratio,
                max_rms_m=self.args.tube_max_rms_mm / 1000.0,
                pose_max_age_ms=self.args.tube_pose_max_age_ms,
                white_min_gray=self.args.tube_white_min_gray,
                white_max_saturation=self.args.tube_white_max_saturation,
            )
            if self.args.position_mode == "rgb-contour":
                self.detector = RgbTubePositionDetector(
                    ball_detector,
                    geometry_config,
                    smoothing_alpha=self.args.rgb_tube_alpha,
                    tube_states=self.rgb_tube_states,
                )
                self.rgb_tube_worker = RgbTubeWorker(
                    self.frames,
                    self.rgb_tube_states,
                    self.stop_event,
                    geometry_config,
                    smoothing_alpha=self.args.rgb_tube_alpha,
                )
            else:
                self.detector = TubePositionDetector(
                    ball_detector,
                    geometry_config,
                    tube_states=self.tube_states,
                )
                self.tube_pose_worker = TubePoseWorker(
                    self.frames,
                    self.results,
                    self.tube_states,
                    self.stop_event,
                    geometry_config,
                )
        else:
            self.detector = ball_detector
        print(
            "engine loaded: input=%s output=%s conf=%.3f path=%s"
            % (
                self.detector.input_shape,
                self.detector.output_shape,
                self.args.conf,
                self.args.engine,
            ),
            file=sys.stderr,
            flush=True,
        )
        warmup = np.full(
            (self.args.height, self.args.width, 3),
            114,
            dtype=np.uint8,
        )
        result = self.detector.detect(
            FramePacket(0, time.monotonic(), warmup)
        )
        print(
            "warmup complete: %.1f ms fast_path=%s power=%s"
            % (result.inference_ms, self.detector.fast_path, self.power_mode),
            file=sys.stderr,
            flush=True,
        )

        if self.args.position_csv is not None:
            self.args.position_csv.parent.mkdir(parents=True, exist_ok=True)
            self.position_csv_handle = self.args.position_csv.open(
                "w", encoding="utf-8", newline=""
            )
            fieldnames = [
                "host_monotonic",
                "frame_id",
                "status",
                "position_cm",
                "ball_confidence",
                "tube_confidence",
                "pose_pitch_deg",
                "depth_age_ms",
                "fit_rms_mm",
                "valid_bin_ratio",
                "position_source",
                "projected_tube_length_px",
                "reference_position_cm",
                "reference_pitch_deg",
            ]
            self.position_csv_writer = csv.DictWriter(
                self.position_csv_handle, fieldnames=fieldnames
            )
            self.position_csv_writer.writeheader()

        serial_mode = "off" if self.args.no_serial else self.args.serial
        if serial_mode.lower() != "off":
            self.serial_writer = SerialWriter(serial_mode, self.args.baud)
            self.serial_writer.start()
        if self.args.control_udp.lower() != "off":
            self.control_writer = DatagramWriter(self.args.control_udp)

        def submit(result: DetectionResult) -> None:
            if self.serial_writer is not None:
                self.serial_writer.submit(
                    encode_result(
                        result,
                        self.args.width,
                        self.args.height,
                        protocol=self.args.protocol,
                    )
                )
            if self.control_writer is not None:
                self.control_writer.submit(encode_result(result, protocol="tube-v3"))
            if self.position_csv_writer is not None:
                if not self.position_csv_armed:
                    if not result.has_valid_position:
                        return
                    self.position_csv_armed = True
                detection = result.detection
                pose = result.tube_pose
                self.position_csv_writer.writerow(
                    {
                        "host_monotonic": "%.9f" % time.monotonic(),
                        "frame_id": result.frame_id,
                        "status": int(result.status),
                        "position_cm": (
                            "%.5f" % result.position_cm
                            if result.position_cm is not None
                            else ""
                        ),
                        "ball_confidence": (
                            "%.6f" % detection.confidence
                            if detection is not None
                            else ""
                        ),
                        "tube_confidence": "%.6f"
                        % result.tube_confidence,
                        "pose_pitch_deg": (
                            "%.5f" % pose.pitch_degrees
                            if pose is not None
                            else ""
                        ),
                        "depth_age_ms": "%.5f" % result.depth_age_ms,
                        "fit_rms_mm": (
                            "%.5f" % (pose.rms_m * 1000.0)
                            if pose is not None
                            else ""
                        ),
                        "valid_bin_ratio": (
                            "%.6f" % pose.valid_bin_ratio
                            if pose is not None
                            else ""
                        ),
                        "position_source": result.position_source,
                        "projected_tube_length_px": (
                            "%.3f" % result.tube_contour.projected_length_px
                            if result.tube_contour is not None
                            else ""
                        ),
                        "reference_position_cm": (
                            self.args.reference_position_cm
                            if self.args.reference_position_cm is not None
                            else ""
                        ),
                        "reference_pitch_deg": (
                            self.args.reference_pitch_deg
                            if self.args.reference_pitch_deg is not None
                            else ""
                        ),
                    }
                )
                self.position_csv_rows += 1
                if self.position_csv_rows % 60 == 0:
                    self.position_csv_handle.flush()

        on_result = (
            submit
            if self.serial_writer is not None
            or self.control_writer is not None
            or self.position_csv_writer is not None
            else None
        )

        self.camera = CameraWorker(
            self.frames,
            self.stop_event,
            device=self.args.device,
            width=self.args.width,
            height=self.args.height,
            fps=self.args.fps,
            backend=self.args.camera_backend,
            depth_width=self.args.depth_width,
            depth_height=self.args.depth_height,
            depth_fps=self.args.depth_fps,
            enable_depth=(
                self.args.protocol in ("tube-v2", "tube-v3")
                and self.args.position_mode == "rgbd"
            ),
            color_auto_exposure=self.args.color_auto_exposure,
            color_exposure=self.args.color_exposure,
            color_exposure_scale=self.args.color_exposure_scale,
            color_gain=self.args.color_gain,
        )
        self.inference = InferenceWorker(
            self.detector,
            self.frames,
            self.results,
            self.stop_event,
            on_result=on_result,
        )
        self.camera.start()
        if self.tube_pose_worker is not None:
            self.tube_pose_worker.start()
        if self.rgb_tube_worker is not None:
            self.rgb_tube_worker.start()
        self.inference.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.camera is not None:
            self.camera.join()
        if self.inference is not None:
            self.inference.join()
        if self.tube_pose_worker is not None:
            self.tube_pose_worker.join()
        if self.rgb_tube_worker is not None:
            self.rgb_tube_worker.join()
        if self.serial_writer is not None:
            self.serial_writer.stop()
        if self.control_writer is not None:
            self.control_writer.close()
            self.control_writer = None
        if self.position_csv_handle is not None:
            self.position_csv_handle.flush()
            self.position_csv_handle.close()
            self.position_csv_handle = None
            self.position_csv_writer = None
        if self.detector is not None:
            self.detector.close()

    def stats_line(self) -> str:
        if self.camera is None or self.inference is None:
            return "runtime not started"
        serial_device = "off"
        serial_fps = 0.0
        replaced = 0
        if self.serial_writer is not None:
            serial_device = self.serial_writer.connected_device or "waiting"
            serial_fps = self.serial_writer.write_fps()
            replaced = self.serial_writer.replaced_packets
        _, latest = self.results.get()
        source = latest.position_source if latest else ""
        age_text = (
            "n/a"
            if source == "rgb-contour"
            else "%.1fms"
            % (latest.depth_age_ms if latest else float("inf"))
        )
        return (
            "capture=%.1fFPS depth=%.1fFPS tube=%.1fFPS infer=%.1fFPS frame=%d result=%d skipped=%d "
            "pre=%.2fms gpu=%.2fms post=%.2fms total=%.2fms "
            "geometry=%.2fms position=%s tube=%.3f source=%s age=%s "
            "latency=%.2fms p95=%.2fms serial=%s %.1fHz replaced=%d"
            % (
                self.camera.capture_fps(),
                self.camera.depth_fps_value(),
                (
                    self.tube_pose_worker.pose_fps()
                    if self.tube_pose_worker is not None
                    else (
                        self.rgb_tube_worker.contour_fps()
                        if self.rgb_tube_worker is not None
                        else 0.0
                    )
                ),
                self.inference.inference_fps(),
                self.camera.frame_count,
                self.inference.result_count,
                self.inference.skipped_frames,
                latest.preprocess_ms if latest else 0.0,
                latest.gpu_ms if latest else 0.0,
                latest.postprocess_ms if latest else 0.0,
                latest.inference_ms if latest else 0.0,
                latest.geometry_ms if latest else 0.0,
                (
                    "%.2fcm" % latest.position_cm
                    if latest and latest.position_cm is not None
                    else "invalid"
                ),
                latest.tube_confidence if latest else 0.0,
                source or "none",
                age_text,
                latest.latency_ms if latest else 0.0,
                self.inference.latency_p95_ms(),
                serial_device,
                serial_fps,
                replaced,
            )
        )


def install_signal_handlers(runtime: Runtime) -> None:
    def stop(_signum, _frame) -> None:
        runtime.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def run_main(position_mode_default: str = "rgbd") -> int:
    parser = argparse.ArgumentParser(
        description="Headless YOLO26s TensorRT single-ball detector"
    )
    add_common_arguments(
        parser,
        serial_default="auto",
        position_mode_default=position_mode_default,
    )
    args = parser.parse_args()
    if args.track_distance <= 0 or args.hold_frames < 0:
        parser.error("tracking parameters are invalid")
    validate_tube_arguments(parser, args)
    runtime = Runtime(args)
    install_signal_handlers(runtime)
    try:
        runtime.start()
        started = time.monotonic()
        next_stats = started
        next_sample = started + 3.0
        capture_samples = []
        inference_samples = []
        while not runtime.stop_event.wait(0.1):
            now = time.monotonic()
            if now >= next_sample:
                if runtime.camera is not None:
                    capture_samples.append(runtime.camera.capture_fps())
                if runtime.inference is not None:
                    inference_samples.append(runtime.inference.inference_fps())
                next_sample = now + 0.5
            if args.stats_interval > 0 and now >= next_stats:
                print(runtime.stats_line(), file=sys.stderr, flush=True)
                next_stats = now + args.stats_interval
            if args.max_seconds > 0 and now - started >= args.max_seconds:
                break
        exit_code = 1 if runtime.inference and runtime.inference.last_error else 0
        if args.metrics_json is not None:
            metrics = {
                "engine": args.engine,
                "position_mode": args.position_mode,
                "duration_seconds": max(0.0, time.monotonic() - started),
                "capture_median_fps": (
                    statistics.median(capture_samples) if capture_samples else 0.0
                ),
                "inference_average_fps": (
                    statistics.mean(inference_samples) if inference_samples else 0.0
                ),
                "latency_p95_ms": (
                    runtime.inference.latency_p95_ms()
                    if runtime.inference is not None
                    else 0.0
                ),
                "capture_frames": (
                    runtime.camera.frame_count if runtime.camera is not None else 0
                ),
                "depth_frames": (
                    runtime.camera.depth_frame_count
                    if runtime.camera is not None
                    else 0
                ),
                "inference_results": (
                    runtime.inference.result_count
                    if runtime.inference is not None
                    else 0
                ),
                "skipped_frames": (
                    runtime.inference.skipped_frames
                    if runtime.inference is not None
                    else 0
                ),
                "serial_hz": (
                    runtime.serial_writer.write_fps()
                    if runtime.serial_writer is not None
                    else 0.0
                ),
                "tube_pose_frames": (
                    runtime.tube_pose_worker.processed_count
                    if runtime.tube_pose_worker is not None
                    else 0
                ),
                "tube_pose_valid_frames": (
                    runtime.tube_pose_worker.valid_count
                    if runtime.tube_pose_worker is not None
                    else 0
                ),
                "tube_pose_skipped_depth_frames": (
                    runtime.tube_pose_worker.skipped_depth_frames
                    if runtime.tube_pose_worker is not None
                    else 0
                ),
                "tube_pose_hz": (
                    runtime.tube_pose_worker.pose_fps()
                    if runtime.tube_pose_worker is not None
                    else (
                        runtime.rgb_tube_worker.contour_fps()
                        if runtime.rgb_tube_worker is not None
                        else 0.0
                    )
                ),
                "tube_geometry_ms": (
                    runtime.tube_pose_worker.last_geometry_ms
                    if runtime.tube_pose_worker is not None
                    else (
                        runtime.rgb_tube_worker.last_geometry_ms
                        if runtime.rgb_tube_worker is not None
                        else 0.0
                    )
                ),
                "rgb_tube_frames": (
                    runtime.rgb_tube_worker.processed_count
                    if runtime.rgb_tube_worker is not None
                    else 0
                ),
                "rgb_tube_valid_frames": (
                    runtime.rgb_tube_worker.valid_count
                    if runtime.rgb_tube_worker is not None
                    else 0
                ),
            }
            if runtime.inference is not None:
                metrics.update(runtime.inference.tracking_stats.snapshot())
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(
                json.dumps(metrics, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(metrics), file=sys.stderr, flush=True)
        return exit_code
    finally:
        runtime.stop()
        print(runtime.stats_line(), file=sys.stderr, flush=True)


def debug_main(position_mode_default: str = "rgbd") -> int:
    parser = argparse.ArgumentParser(
        description="YOLO26s TensorRT single-ball detector with debug window"
    )
    add_common_arguments(
        parser,
        serial_default="off",
        position_mode_default=position_mode_default,
    )
    parser.add_argument("--window", default="YOLO26s ball debug")
    parser.add_argument("--preview-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.preview_scale <= 0:
        parser.error("--preview-scale must be positive")
    if args.track_distance <= 0 or args.hold_frames < 0:
        parser.error("tracking parameters are invalid")
    validate_tube_arguments(parser, args)

    runtime = Runtime(args)
    install_signal_handlers(runtime)
    result_sequence = 0
    try:
        runtime.start()
        started = time.monotonic()
        next_stats = started
        while not runtime.stop_event.is_set():
            result_sequence, result = runtime.results.wait_after(
                result_sequence,
                timeout=0.2,
            )
            if result is None:
                cv2.waitKey(1)
                continue
            capture_frame_id = runtime.camera.frame_count if runtime.camera else 0
            serial_text = "off"
            serial_fps = 0.0
            if runtime.serial_writer is not None:
                serial_text = runtime.serial_writer.connected_device or "waiting"
                serial_fps = runtime.serial_writer.write_fps()
            canvas = draw_debug(
                result.source_image,
                result,
                capture_frame_id,
                runtime.camera.capture_fps() if runtime.camera else 0.0,
                runtime.camera.depth_fps_value() if runtime.camera else 0.0,
                (
                    runtime.tube_pose_worker.pose_fps()
                    if runtime.tube_pose_worker is not None
                    else (
                        runtime.rgb_tube_worker.contour_fps()
                        if runtime.rgb_tube_worker is not None
                        else 0.0
                    )
                ),
                runtime.inference.inference_fps() if runtime.inference else 0.0,
                runtime.inference.skipped_frames if runtime.inference else 0,
                runtime.inference.latency_p95_ms() if runtime.inference else 0.0,
                serial_text,
                serial_fps,
                runtime.detector.input_shape if runtime.detector else (),
                runtime.detector.fast_path if runtime.detector else "unknown",
                (
                    runtime.camera.device
                    if runtime.camera and runtime.camera.device
                    else "waiting"
                ),
                runtime.power_mode,
                args.protocol,
                (
                    runtime.camera.color_controls_text()
                    if runtime.camera is not None
                    else "exposure unavailable"
                ),
            )
            if abs(args.preview_scale - 1.0) > 1e-6:
                canvas = cv2.resize(
                    canvas,
                    (
                        max(1, int(round(canvas.shape[1] * args.preview_scale))),
                        max(1, int(round(canvas.shape[0] * args.preview_scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(args.window, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            now = time.monotonic()
            if args.stats_interval > 0 and now >= next_stats:
                print(runtime.stats_line(), file=sys.stderr, flush=True)
                next_stats = now + args.stats_interval
            if args.max_seconds > 0 and now - started >= args.max_seconds:
                break
        return 1 if runtime.inference and runtime.inference.last_error else 0
    finally:
        runtime.stop_event.set()
        runtime.stop()
        cv2.destroyAllWindows()
        print(runtime.stats_line(), file=sys.stderr, flush=True)


def validate_tube_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.control_udp.lower() != "off":
        try:
            parse_udp_endpoint(args.control_udp)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.protocol == "pixel-v1":
            parser.error("--control-udp requires tube-v2 or tube-v3 position mode")
    if args.tube_length_cm <= 0:
        parser.error("--tube-length-cm must be positive")
    if args.tube_pose_max_age_ms <= 0:
        parser.error("--tube-pose-max-age-ms must be positive")
    if args.tube_min_projection_px <= 0:
        parser.error("--tube-min-projection-px must be positive")
    if not 0.0 < args.tube_min_depth_ratio <= 1.0:
        parser.error("--tube-min-depth-ratio must be in (0, 1]")
    if args.tube_max_rms_mm <= 0:
        parser.error("--tube-max-rms-mm must be positive")
    if not 0 <= args.tube_white_min_gray <= 255:
        parser.error("--tube-white-min-gray must be in 0..255")
    if not 0 <= args.tube_white_max_saturation <= 255:
        parser.error("--tube-white-max-saturation must be in 0..255")
    if args.color_exposure_scale <= 0:
        parser.error("--color-exposure-scale must be positive")
    if (
        args.protocol in ("tube-v2", "tube-v3")
        and args.position_mode == "rgbd"
        and args.camera_backend != "sdk"
    ):
        parser.error("RGB-D tube protocol requires --camera-backend sdk with depth")
    if (
        args.reference_position_cm is not None
        and not -args.tube_length_cm / 2.0
        <= args.reference_position_cm
        <= args.tube_length_cm / 2.0
    ):
        parser.error("--reference-position-cm is outside the physical tube")
    if (
        args.reference_pitch_deg is not None
        and not -90.0 <= args.reference_pitch_deg <= 90.0
    ):
        parser.error("--reference-pitch-deg must be in -90..90")
