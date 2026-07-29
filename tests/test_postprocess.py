from __future__ import annotations

import unittest

import numpy as np

from ball_runtime.tensorrt_detector import parse_end2end_output, preprocess


class EndToEndPostprocessTests(unittest.TestCase):
    def test_768x480_letterbox_mapping(self) -> None:
        output = np.zeros((1, 300, 6), dtype=np.float32)
        # 1280x800 -> 768x480 exactly at scale 0.6.
        output[0, 0] = [60, 60, 180, 120, 0.90, 0]
        detections = parse_end2end_output(
            output,
            frame_width=1280,
            frame_height=800,
            scale=0.6,
            pad_x=0.0,
            pad_y=0.0,
            confidence=0.25,
        )
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].x1, 100.0, places=4)
        self.assertAlmostEqual(detections[0].y1, 100.0, places=4)
        self.assertAlmostEqual(detections[0].x2, 300.0, places=4)
        self.assertAlmostEqual(detections[0].y2, 200.0, places=4)

    def test_640x416_letterbox_mapping(self) -> None:
        output = np.zeros((1, 300, 6), dtype=np.float32)
        # 1280x800 -> 640x400 with 8 pixels vertical padding.
        output[0, 0] = [50, 58, 150, 108, 0.90, 0]
        detections = parse_end2end_output(
            output, 1280, 800, 0.5, 0.0, 8.0, 0.25
        )
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].center[0], 200.0)
        self.assertAlmostEqual(detections[0].center[1], 150.0)

    def test_preprocess_returns_requested_static_shape(self) -> None:
        image = np.zeros((800, 1280, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess(
            image,
            input_width=640,
            input_height=416,
            dtype=np.float32,
        )
        self.assertEqual(blob.shape, (1, 3, 416, 640))
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual((pad_x, pad_y), (0.0, 8.0))

    def test_rejects_invalid_and_duplicate_center_rows(self) -> None:
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [10, 10, 30, 30, 0.9, 9]
        output[0, 1] = [10.1, 10.2, 30.1, 29.9, 0.8, 0]
        output[0, 2] = [30, 30, 10, 10, 0.99, 0]
        detections = parse_end2end_output(
            output, 100, 100, 1.0, 0.0, 0.0, 0.25
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 0)


if __name__ == "__main__":
    unittest.main()
