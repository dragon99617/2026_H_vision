from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .tube_geometry import (
    TubeGeometryConfig,
    contour_axis_position_cm,
    point_in_tube,
    segment_white_tube,
)
from .types import DetectionResult, FramePacket, TrackStatus, TubeContour


def smooth_rgb_tube_contour(
    previous: TubeContour,
    contour: TubeContour,
    smoothing_alpha: float,
) -> TubeContour:
    if (
        previous is None
        or not previous.valid
        or previous.endpoint_negative_px is None
        or previous.endpoint_positive_px is None
    ):
        return contour
    current_negative = np.asarray(
        contour.endpoint_negative_px, dtype=np.float64
    )
    current_positive = np.asarray(
        contour.endpoint_positive_px, dtype=np.float64
    )
    previous_negative = np.asarray(
        previous.endpoint_negative_px, dtype=np.float64
    )
    previous_positive = np.asarray(
        previous.endpoint_positive_px, dtype=np.float64
    )
    current_center = (current_negative + current_positive) * 0.5
    previous_center = (previous_negative + previous_positive) * 0.5
    current_length = float(
        np.linalg.norm(current_positive - current_negative)
    )
    if (
        current_length < 1.0
        or np.linalg.norm(current_center - previous_center)
        > current_length * 0.20
    ):
        return contour
    alpha = float(smoothing_alpha)
    negative = alpha * current_negative + (1.0 - alpha) * previous_negative
    positive = alpha * current_positive + (1.0 - alpha) * previous_positive
    axis = positive - negative
    length = float(np.linalg.norm(axis))
    if length < 1.0:
        return contour
    axis /= length
    return replace(
        contour,
        endpoint_negative_px=(float(negative[0]), float(negative[1])),
        endpoint_positive_px=(float(positive[0]), float(positive[1])),
        center_px=(
            float((negative[0] + positive[0]) * 0.5),
            float((negative[1] + positive[1]) * 0.5),
        ),
        axis_px=(float(axis[0]), float(axis[1])),
        projected_length_px=length,
    )


class RgbTubePositionDetector:
    """Estimate signed centimetres from the 2D tube contour only."""

    def __init__(
        self,
        ball_detector,
        config: TubeGeometryConfig,
        smoothing_alpha: float = 0.35,
        tube_states=None,
    ) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.ball_detector = ball_detector
        self.config = config
        self.smoothing_alpha = float(smoothing_alpha)
        self.last_contour = None
        self.tube_states = tube_states

    @property
    def input_shape(self):
        return self.ball_detector.input_shape

    @property
    def output_shape(self):
        return self.ball_detector.output_shape

    @property
    def fast_path(self):
        return "%s+tube-rgb-contour" % self.ball_detector.fast_path

    def close(self) -> None:
        self.ball_detector.close()

    def _smooth_contour(self, contour: TubeContour) -> TubeContour:
        smoothed = smooth_rgb_tube_contour(
            self.last_contour,
            contour,
            self.smoothing_alpha,
        )
        self.last_contour = smoothed
        return smoothed

    def detect(self, packet: FramePacket) -> DetectionResult:
        raw = self.ball_detector.detect(packet)
        geometry_started = time.monotonic()
        external_geometry_ms = 0.0
        if self.tube_states is not None:
            _, state = self.tube_states.get()
            if state is None:
                contour = TubeContour(
                    valid=False,
                    reason="waiting for first RGB tube contour",
                )
            else:
                contour = state.contour
                external_geometry_ms = state.geometry_ms
                age_ms = (
                    packet.captured_monotonic - state.captured_monotonic
                ) * 1000.0
                if age_ms > self.config.pose_max_age_ms:
                    contour = replace(
                        contour,
                        valid=False,
                        reason="RGB tube contour expired",
                    )
        else:
            contour = segment_white_tube(packet.image, self.config)
            if contour.valid:
                contour = self._smooth_contour(contour)
            else:
                self.last_contour = None

        detection = raw.detection
        if detection is not None and not point_in_tube(
            contour, detection.center
        ):
            self.ball_detector.reset()
            detection = None
            raw = replace(
                raw,
                detections=(),
                status=TrackStatus.LOST,
                missed_frames=0,
            )

        ball_status = raw.status
        output_status = TrackStatus.LOST
        position_cm = None
        projected = None
        if detection is not None and contour.valid:
            position_cm, projected = contour_axis_position_cm(
                detection.center,
                contour,
                self.config.tube_length_cm,
            )
            output_status = ball_status

        completed = time.monotonic()
        local_geometry_ms = (completed - geometry_started) * 1000.0
        return DetectionResult(
            frame_id=raw.frame_id,
            captured_monotonic=raw.captured_monotonic,
            completed_monotonic=completed,
            inference_ms=raw.inference_ms + local_geometry_ms,
            detections=raw.detections,
            source_image=raw.source_image,
            status=output_status,
            preprocess_ms=raw.preprocess_ms,
            gpu_ms=raw.gpu_ms,
            postprocess_ms=raw.postprocess_ms,
            missed_frames=raw.missed_frames,
            ball_status=ball_status,
            tube_contour=contour,
            position_cm=position_cm,
            position_projected_px=projected,
            tube_confidence=contour.confidence if contour.valid else 0.0,
            geometry_ms=(
                external_geometry_ms
                if self.tube_states is not None
                else local_geometry_ms
            ),
            position_source="rgb-contour",
        )
