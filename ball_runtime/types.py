from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class TrackStatus(IntEnum):
    LOST = 0
    MEASURED = 1
    PREDICTED = 2


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Tuple[float, ...] = ()


@dataclass(frozen=True)
class RgbdCalibration:
    color: CameraIntrinsics
    depth: CameraIntrinsics
    depth_to_color_rotation: Tuple[float, ...]
    depth_to_color_translation_m: Tuple[float, float, float]


@dataclass(frozen=True)
class TubeContour:
    valid: bool
    mask: object = None
    contour: object = None
    endpoint_negative_px: Optional[Tuple[float, float]] = None
    endpoint_positive_px: Optional[Tuple[float, float]] = None
    center_px: Optional[Tuple[float, float]] = None
    axis_px: Optional[Tuple[float, float]] = None
    projected_length_px: float = 0.0
    width_px: float = 0.0
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class TubePose:
    valid: bool
    captured_monotonic: float
    center_m: Optional[Tuple[float, float, float]] = None
    direction: Optional[Tuple[float, float, float]] = None
    endpoint_negative_m: Optional[Tuple[float, float, float]] = None
    endpoint_positive_m: Optional[Tuple[float, float, float]] = None
    endpoint_negative_px: Optional[Tuple[float, float]] = None
    endpoint_positive_px: Optional[Tuple[float, float]] = None
    observed_length_m: float = 0.0
    fitted_length_m: float = 0.0
    rms_m: float = 0.0
    valid_bin_ratio: float = 0.0
    confidence: float = 0.0
    pitch_degrees: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class TubeState:
    depth_frame_id: int
    processed_monotonic: float
    contour: TubeContour
    pose: TubePose
    geometry_ms: float


@dataclass(frozen=True)
class RgbTubeState:
    frame_id: int
    captured_monotonic: float
    processed_monotonic: float
    contour: TubeContour
    geometry_ms: float


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_monotonic: float
    image: object
    depth: object = None
    depth_frame_id: int = 0
    depth_captured_monotonic: float = 0.0
    depth_scale_m: float = 0.001
    calibration: Optional[RgbdCalibration] = None
    camera_backend: str = "v4l2-color"


@dataclass(frozen=True)
class DetectionResult:
    frame_id: int
    captured_monotonic: float
    completed_monotonic: float
    inference_ms: float
    detections: Tuple[Detection, ...]
    source_image: object = None
    status: TrackStatus = TrackStatus.LOST
    preprocess_ms: float = 0.0
    gpu_ms: float = 0.0
    postprocess_ms: float = 0.0
    missed_frames: int = 0
    ball_status: Optional[TrackStatus] = None
    tube_contour: Optional[TubeContour] = None
    tube_pose: Optional[TubePose] = None
    position_cm: Optional[float] = None
    position_projected_px: Optional[Tuple[float, float]] = None
    tube_confidence: float = 0.0
    geometry_ms: float = 0.0
    depth_age_ms: float = float("inf")
    position_source: str = ""

    @property
    def latency_ms(self) -> float:
        return max(
            0.0,
            (self.completed_monotonic - self.captured_monotonic) * 1000.0,
        )

    @property
    def detection(self):
        return self.detections[0] if self.detections else None

    @property
    def effective_ball_status(self) -> TrackStatus:
        return self.ball_status if self.ball_status is not None else self.status

    @property
    def has_valid_position(self) -> bool:
        return (
            self.position_cm is not None
            and self.status != TrackStatus.LOST
            and (
                self.position_source == "rgb-contour"
                or (
                    self.tube_pose is not None
                    and self.tube_pose.valid
                )
            )
        )
