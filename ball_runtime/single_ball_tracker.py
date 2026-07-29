from __future__ import annotations

import math
import time
from typing import Optional, Sequence, Tuple

from .tensorrt_detector import TensorRTDetector
from .types import Detection, DetectionResult, FramePacket, TrackStatus


def detection_iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = left.width * left.height
    right_area = right.width * right.height
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def shifted(detection: Detection, dx: float, dy: float) -> Detection:
    return Detection(
        x1=detection.x1 + dx,
        y1=detection.y1 + dy,
        x2=detection.x2 + dx,
        y2=detection.y2 + dy,
        confidence=detection.confidence,
        class_id=0,
    )


def blended(
    predicted: Detection,
    observed: Detection,
    observation_weight: float,
) -> Detection:
    old_weight = 1.0 - observation_weight
    return Detection(
        x1=predicted.x1 * old_weight + observed.x1 * observation_weight,
        y1=predicted.y1 * old_weight + observed.y1 * observation_weight,
        x2=predicted.x2 * old_weight + observed.x2 * observation_weight,
        y2=predicted.y2 * old_weight + observed.y2 * observation_weight,
        confidence=observed.confidence,
        class_id=0,
    )


def clipped(detection: Detection, width: int, height: int) -> Optional[Detection]:
    x1 = min(width - 1.0, max(0.0, detection.x1))
    y1 = min(height - 1.0, max(0.0, detection.y1))
    x2 = min(width - 1.0, max(0.0, detection.x2))
    y2 = min(height - 1.0, max(0.0, detection.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=detection.confidence,
        class_id=0,
    )


class SingleBallTrackingDetector:
    """Keep one stable target and bridge at most two missed inference frames."""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        confidence: float = 0.25,
        match_distance: float = 80.0,
        smooth_alpha: float = 0.75,
        hold_frames: int = 2,
        velocity_alpha: float = 0.5,
        base_detector=None,
    ) -> None:
        if match_distance <= 0.0:
            raise ValueError("match_distance must be positive")
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0, 1]")
        if hold_frames < 0:
            raise ValueError("hold_frames cannot be negative")
        if base_detector is None:
            if engine_path is None:
                raise ValueError("engine_path is required without base_detector")
            base_detector = TensorRTDetector(engine_path, confidence)
        self.base_detector = base_detector
        self.match_distance = float(match_distance)
        self.smooth_alpha = float(smooth_alpha)
        self.hold_frames = int(hold_frames)
        self.velocity_alpha = float(velocity_alpha)
        self.tracked: Optional[Detection] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_frame_id = 0
        self.missed_frames = 0

    @property
    def input_shape(self):
        return self.base_detector.input_shape

    @property
    def output_shape(self):
        return self.base_detector.output_shape

    @property
    def fast_path(self):
        return getattr(self.base_detector, "fast_path", "unknown")

    def close(self) -> None:
        self.base_detector.close()

    def reset(self) -> None:
        self._clear()
        self.last_frame_id = 0

    def detect(self, packet: FramePacket) -> DetectionResult:
        started = time.monotonic()
        raw = self.base_detector.detect(packet)
        width = int(packet.image.shape[1])
        height = int(packet.image.shape[0])
        frame_delta = max(1, packet.frame_id - self.last_frame_id)
        predicted = None
        status = TrackStatus.LOST

        if self.tracked is not None:
            predicted = shifted(
                self.tracked,
                self.velocity[0] * frame_delta,
                self.velocity[1] * frame_delta,
            )
            predicted = clipped(predicted, width, height)

        if self.tracked is None or predicted is None:
            selected = self._highest_confidence(raw.detections)
            if selected is not None:
                self.tracked = selected
                self.velocity = (0.0, 0.0)
                self.missed_frames = 0
                status = TrackStatus.MEASURED
            else:
                self._clear()
        else:
            matched = self._match(predicted, raw.detections, frame_delta)
            if matched is not None:
                old_x, old_y = self.tracked.center
                new_x, new_y = matched.center
                observed_velocity = (
                    (new_x - old_x) / frame_delta,
                    (new_y - old_y) / frame_delta,
                )
                self.velocity = (
                    self.velocity[0] * (1.0 - self.velocity_alpha)
                    + observed_velocity[0] * self.velocity_alpha,
                    self.velocity[1] * (1.0 - self.velocity_alpha)
                    + observed_velocity[1] * self.velocity_alpha,
                )
                self.tracked = blended(predicted, matched, self.smooth_alpha)
                self.missed_frames = 0
                status = TrackStatus.MEASURED
            else:
                self.missed_frames += 1
                if self.missed_frames <= self.hold_frames:
                    self.tracked = Detection(
                        x1=predicted.x1,
                        y1=predicted.y1,
                        x2=predicted.x2,
                        y2=predicted.y2,
                        confidence=max(0.0, self.tracked.confidence * 0.85),
                        class_id=0,
                    )
                    status = TrackStatus.PREDICTED
                else:
                    self._clear()

        self.last_frame_id = packet.frame_id
        detections = (self.tracked,) if self.tracked is not None else ()
        completed = time.monotonic()
        return DetectionResult(
            frame_id=packet.frame_id,
            captured_monotonic=packet.captured_monotonic,
            completed_monotonic=completed,
            inference_ms=(completed - started) * 1000.0,
            detections=detections,
            source_image=packet.image,
            status=status,
            preprocess_ms=raw.preprocess_ms,
            gpu_ms=raw.gpu_ms,
            postprocess_ms=raw.postprocess_ms,
            missed_frames=self.missed_frames,
        )

    def _match(
        self,
        predicted: Detection,
        candidates: Sequence[Detection],
        frame_delta: int,
    ) -> Optional[Detection]:
        px, py = predicted.center
        speed = math.hypot(*self.velocity)
        gate = self.match_distance + min(160.0, speed * frame_delta * 1.5)
        ranked = []
        for candidate in candidates:
            width_ratio = candidate.width / max(1e-6, predicted.width)
            height_ratio = candidate.height / max(1e-6, predicted.height)
            if not (0.4 <= width_ratio <= 2.5 and 0.4 <= height_ratio <= 2.5):
                continue
            cx, cy = candidate.center
            distance = math.hypot(cx - px, cy - py)
            overlap = detection_iou(predicted, candidate)
            if distance <= gate or overlap >= 0.05:
                ranked.append((distance, -overlap, -candidate.confidence, candidate))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[:3])
        return ranked[0][3]

    @staticmethod
    def _highest_confidence(
        candidates: Sequence[Detection],
    ) -> Optional[Detection]:
        return max(candidates, key=lambda item: item.confidence) if candidates else None

    def _clear(self) -> None:
        self.tracked = None
        self.velocity = (0.0, 0.0)
        self.missed_frames = 0
