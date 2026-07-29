from __future__ import annotations

import unittest

from ball_runtime.runtime_metrics import TrackingStats
from ball_runtime.types import Detection, DetectionResult, TrackStatus


def result(status: TrackStatus, x: float = 100.0, y: float = 50.0):
    detections = ()
    if status != TrackStatus.LOST:
        detections = (Detection(x - 5, y - 5, x + 5, y + 5, 0.9),)
    return DetectionResult(
        frame_id=1,
        captured_monotonic=1.0,
        completed_monotonic=1.01,
        inference_ms=10.0,
        detections=detections,
        status=status,
    )


class TrackingStatsTests(unittest.TestCase):
    def test_rates_prediction_streak_and_jitter(self) -> None:
        stats = TrackingStats(maximum_centers=4)
        for item in (
            result(TrackStatus.MEASURED, 100, 50),
            result(TrackStatus.PREDICTED, 101, 50),
            result(TrackStatus.PREDICTED, 102, 50),
            result(TrackStatus.LOST),
        ):
            stats.record(item)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot["tracking_total_frames"], 4)
        self.assertEqual(snapshot["measured_frames"], 1)
        self.assertEqual(snapshot["predicted_frames"], 2)
        self.assertEqual(snapshot["lost_frames"], 1)
        self.assertAlmostEqual(snapshot["measured_detection_coverage"], 0.25)
        self.assertAlmostEqual(snapshot["effective_output_rate"], 0.75)
        self.assertEqual(snapshot["max_predicted_streak_frames"], 2)
        self.assertEqual(snapshot["center_samples"], 3)
        self.assertAlmostEqual(snapshot["center_jitter_p95_radius_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
