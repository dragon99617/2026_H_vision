from __future__ import annotations

import time
import unittest

import numpy as np

from ball_runtime.single_ball_tracker import SingleBallTrackingDetector
from ball_runtime.types import (
    Detection,
    DetectionResult,
    FramePacket,
    TrackStatus,
)


def ball(center_x: float, center_y: float, confidence: float) -> Detection:
    return Detection(
        center_x - 10.0,
        center_y - 10.0,
        center_x + 10.0,
        center_y + 10.0,
        confidence,
    )


class FakeDetector:
    input_shape = (1, 3, 480, 768)
    output_shape = (1, 300, 6)
    fast_path = "fake"

    def __init__(self, outputs) -> None:
        self.outputs = iter(outputs)
        self.closed = False

    def detect(self, packet: FramePacket) -> DetectionResult:
        detections = tuple(next(self.outputs))
        return DetectionResult(
            frame_id=packet.frame_id,
            captured_monotonic=packet.captured_monotonic,
            completed_monotonic=time.monotonic(),
            inference_ms=1.0,
            detections=detections,
            source_image=packet.image,
            status=TrackStatus.MEASURED if detections else TrackStatus.LOST,
            preprocess_ms=0.2,
            gpu_ms=0.7,
            postprocess_ms=0.1,
        )

    def close(self) -> None:
        self.closed = True


class SingleBallTrackerTests(unittest.TestCase):
    def packet(self, frame_id: int) -> FramePacket:
        image = np.full((120, 400, 3), 114, dtype=np.uint8)
        return FramePacket(frame_id, time.monotonic(), image)

    def test_acquires_highest_confidence(self) -> None:
        tracker = SingleBallTrackingDetector(
            base_detector=FakeDetector(
                [[ball(100, 60, 0.60), ball(300, 60, 0.95)]]
            )
        )
        result = tracker.detect(self.packet(1))
        self.assertEqual(result.status, TrackStatus.MEASURED)
        self.assertAlmostEqual(result.detection.center[0], 300.0)

    def test_keeps_identity_when_other_detection_is_more_confident(self) -> None:
        tracker = SingleBallTrackingDetector(
            base_detector=FakeDetector(
                [
                    [ball(100, 60, 0.90), ball(300, 60, 0.80)],
                    [ball(110, 60, 0.40), ball(300, 60, 0.99)],
                ]
            )
        )
        tracker.detect(self.packet(1))
        result = tracker.detect(self.packet(2))
        self.assertLess(result.detection.center[0], 130.0)
        self.assertAlmostEqual(result.detection.confidence, 0.40)

    def test_predicts_two_frames_then_reports_lost(self) -> None:
        tracker = SingleBallTrackingDetector(
            base_detector=FakeDetector(
                [
                    [ball(100, 60, 0.9)],
                    [ball(110, 60, 0.9)],
                    [],
                    [],
                    [],
                ]
            ),
            hold_frames=2,
        )
        measured = tracker.detect(self.packet(1))
        moved = tracker.detect(self.packet(2))
        predicted1 = tracker.detect(self.packet(3))
        predicted2 = tracker.detect(self.packet(4))
        lost = tracker.detect(self.packet(5))
        self.assertEqual(measured.status, TrackStatus.MEASURED)
        self.assertEqual(moved.status, TrackStatus.MEASURED)
        self.assertEqual(predicted1.status, TrackStatus.PREDICTED)
        self.assertEqual(predicted2.status, TrackStatus.PREDICTED)
        self.assertGreater(
            predicted2.detection.center[0],
            predicted1.detection.center[0],
        )
        self.assertEqual(lost.status, TrackStatus.LOST)
        self.assertIsNone(lost.detection)

    def test_rejects_implausible_size_jump(self) -> None:
        huge = Detection(0, 0, 300, 119, 0.99)
        tracker = SingleBallTrackingDetector(
            base_detector=FakeDetector([[ball(100, 60, 0.9)], [huge]])
        )
        tracker.detect(self.packet(1))
        result = tracker.detect(self.packet(2))
        self.assertEqual(result.status, TrackStatus.PREDICTED)

    def test_close_releases_base_detector(self) -> None:
        fake = FakeDetector([])
        tracker = SingleBallTrackingDetector(base_detector=fake)
        tracker.close()
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
