from __future__ import annotations

import unittest

from ball_runtime.latest import LatestValue
from ball_runtime.tube_geometry import TubeGeometryConfig, segment_white_tube
from ball_runtime.tube_position_detector import TubePositionDetector
from ball_runtime.types import (
    Detection,
    DetectionResult,
    FramePacket,
    TrackStatus,
    TubePose,
    TubeState,
)
from tests.test_tube_geometry import calibration, synthetic_flat_scene


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


class TubePositionDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image, _ = synthetic_flat_scene()
        self.config = TubeGeometryConfig()
        self.contour = segment_white_tube(self.image, self.config)
        self.pose = TubePose(
            valid=True,
            captured_monotonic=1.0,
            center_m=(0.0, 0.0, 0.6),
            direction=(-1.0, 0.0, 0.0),
            endpoint_negative_m=(0.125, 0.0, 0.6),
            endpoint_positive_m=(-0.125, 0.0, 0.6),
            endpoint_negative_px=(848.33, 400.0),
            endpoint_positive_px=(431.67, 400.0),
            fitted_length_m=0.25,
            confidence=0.95,
        )
        self.states = LatestValue()
        self.states.publish(
            TubeState(1, 1.01, self.contour, self.pose, 10.0)
        )

    def packet(self, timestamp=1.05):
        return FramePacket(
            1,
            timestamp,
            self.image,
            calibration=calibration(),
            depth_captured_monotonic=1.0,
        )

    def test_external_pose_produces_signed_position(self) -> None:
        ball = FakeBallDetector(Detection(630, 390, 650, 410, 0.9))
        detector = TubePositionDetector(ball, self.config, self.states)
        result = detector.detect(self.packet())
        self.assertEqual(result.status, TrackStatus.MEASURED)
        self.assertAlmostEqual(result.position_cm, 0.0, places=4)
        self.assertEqual(result.geometry_ms, 10.0)
        self.assertTrue(result.has_valid_position)

    def test_predicted_ball_retains_predicted_v2_status(self) -> None:
        ball = FakeBallDetector(
            Detection(700, 390, 720, 410, 0.7),
            TrackStatus.PREDICTED,
        )
        detector = TubePositionDetector(ball, self.config, self.states)
        result = detector.detect(self.packet())
        self.assertEqual(result.status, TrackStatus.PREDICTED)
        self.assertIsNotNone(result.position_cm)

    def test_pose_older_than_100ms_prohibits_position(self) -> None:
        ball = FakeBallDetector(Detection(630, 390, 650, 410, 0.9))
        detector = TubePositionDetector(ball, self.config, self.states)
        result = detector.detect(self.packet(1.101))
        self.assertEqual(result.status, TrackStatus.LOST)
        self.assertIsNone(result.position_cm)
        self.assertEqual(result.tube_pose.reason, "tube pose expired")

    def test_ball_outside_tube_is_rejected_and_tracker_reset(self) -> None:
        ball = FakeBallDetector(Detection(0, 0, 20, 20, 0.9))
        detector = TubePositionDetector(ball, self.config, self.states)
        result = detector.detect(self.packet())
        self.assertEqual(result.status, TrackStatus.LOST)
        self.assertIsNone(result.detection)
        self.assertEqual(ball.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
