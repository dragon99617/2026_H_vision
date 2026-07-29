from __future__ import annotations

import math
import statistics
import threading
from collections import deque

from .types import DetectionResult, TrackStatus


class TrackingStats:
    """Bounded, thread-safe tracking counters for hardware acceptance runs."""

    def __init__(self, maximum_centers: int = 36000) -> None:
        self._lock = threading.Lock()
        self._counts = {status: 0 for status in TrackStatus}
        self._centers = deque(maxlen=maximum_centers)
        self._prediction_streak = 0
        self._maximum_prediction_streak = 0

    def record(self, result: DetectionResult) -> None:
        with self._lock:
            self._counts[result.status] += 1
            if result.status == TrackStatus.PREDICTED:
                self._prediction_streak += 1
                self._maximum_prediction_streak = max(
                    self._maximum_prediction_streak,
                    self._prediction_streak,
                )
            else:
                self._prediction_streak = 0
            if result.detection is not None:
                self._centers.append(result.detection.center)

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            centers = list(self._centers)
            maximum_prediction_streak = self._maximum_prediction_streak
        total = sum(counts.values())
        measured = counts[TrackStatus.MEASURED]
        predicted = counts[TrackStatus.PREDICTED]
        lost = counts[TrackStatus.LOST]
        jitter = 0.0
        if centers:
            median_x = statistics.median(point[0] for point in centers)
            median_y = statistics.median(point[1] for point in centers)
            radii = sorted(
                math.hypot(point[0] - median_x, point[1] - median_y)
                for point in centers
            )
            jitter = float(radii[round((len(radii) - 1) * 0.95)])
        return {
            "tracking_total_frames": total,
            "measured_frames": measured,
            "predicted_frames": predicted,
            "lost_frames": lost,
            "measured_detection_coverage": measured / float(total) if total else 0.0,
            "effective_output_rate": (
                (measured + predicted) / float(total) if total else 0.0
            ),
            "max_predicted_streak_frames": maximum_prediction_streak,
            "center_samples": len(centers),
            "center_jitter_p95_radius_px": jitter,
        }
