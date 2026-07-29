from __future__ import annotations

import unittest

from ball_runtime.rgb_tube_position_detector import RgbTubePositionDetector
from ball_runtime.tube_geometry import TubeGeometryConfig, segment_white_tube
from ball_runtime.types import (
    Detection,
    DetectionResult,
    FramePacket,
    TrackStatus,
)
from tests.test_tube_geometry import synthetic_flat_scene


class FakeBallDetector:
    input_shape = (1, 3, 416, 640)
    output_shape = (1, 300, 6)
    fast_path = "fake"

    def __init__(self, detection, status=TrackStatus.MEASURED) -> None:
        self.detection = detection
        self.status = status
        self.reset_count = 0

    def detect(self, packet):
        return DetectionResult(
            frame_id=packet.frame_id,
            captured_monotonic=packet.captured_monotonic,
            completed_monotonic=packet.captured_monotonic + 0.001,
            inference_ms=1.0,
            detections=(self.detection,) if self.detection else (),
            source_image=packet.image,
            status=self.status,
        )

    def reset(self):
        self.reset_count += 1

    def close(self):
        pass


class RgbTubePositionDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image, _ = synthetic_flat_scene()
        self.config = TubeGeometryConfig()
        self.contour = segment_white_tube(self.image, self.config)

    def detection_at_fraction(self, fraction: float) -> Detection:
        negative = self.contour.endpoint_negative_px
        positive = self.contour.endpoint_positive_px
        x = negative[0] + (positive[0] - negative[0]) * fraction
        y = negative[1] + (positive[1] - negative[1]) * fraction
        return Detection(x - 10, y - 10, x + 10, y + 10, 0.9)

    def packet(self):
        return FramePacket(1, 1.0, self.image)

    def test_center_ball_is_zero_without_depth_or_calibration(self) -> None:
        ball = FakeBallDetector(self.detection_at_fraction(0.5))
        detector = RgbTubePositionDetector(ball, self.config)
        result = detector.detect(self.packet())
        self.assertEqual(result.status, TrackStatus.MEASURED)
        self.assertAlmostEqual(result.position_cm, 0.0, places=4)
        self.assertEqual(result.position_source, "rgb-contour")
        self.assertIsNone(result.tube_pose)
        self.assertTrue(result.has_valid_position)

    def test_quarter_axis_maps_to_minus_six_point_two_five(self) -> None:
        ball = FakeBallDetector(self.detection_at_fraction(0.25))
        result = RgbTubePositionDetector(ball, self.config).detect(
            self.packet()
        )
        self.assertAlmostEqual(result.position_cm, -6.25, delta=0.1)

    def test_ball_outside_contour_is_rejected(self) -> None:
        ball = FakeBallDetector(Detection(0, 0, 20, 20, 0.9))
        result = RgbTubePositionDetector(ball, self.config).detect(
            self.packet()
        )
        self.assertEqual(result.status, TrackStatus.LOST)
        self.assertIsNone(result.position_cm)
        self.assertEqual(ball.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
