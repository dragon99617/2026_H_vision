from __future__ import annotations

import unittest

import numpy as np

from ball_runtime.tube_geometry import TubeGeometryConfig, segment_white_tube
from ball_runtime.types import (
    Detection,
    DetectionResult,
    TrackStatus,
    TubePose,
)
from ball_runtime.visualize import draw_debug
from tests.test_tube_geometry import synthetic_flat_scene


class DebugVisualizationTests(unittest.TestCase):
    def _draw(self, result):
        return draw_debug(
            result.source_image,
            result,
            1,
            60.0,
            30.0,
            30.0,
            59.0,
            0,
            25.0,
            "off",
            0.0,
            (1, 3, 416, 640),
            "test",
            "OrbbecSDK",
            "20W",
            "tube-v2",
        )

    def test_invalid_depth_still_draws_clearly_marked_2d_ruler(self) -> None:
        image, _ = synthetic_flat_scene()
        contour = segment_white_tube(image, TubeGeometryConfig())
        result = DetectionResult(
            frame_id=1,
            captured_monotonic=1.0,
            completed_monotonic=1.01,
            inference_ms=10.0,
            detections=(Detection(630, 390, 650, 410, 0.9),),
            source_image=image,
            status=TrackStatus.LOST,
            ball_status=TrackStatus.MEASURED,
            tube_contour=contour,
            tube_pose=TubePose(
                valid=False,
                captured_monotonic=1.0,
                reason="too few valid depth samples",
            ),
        )
        canvas = self._draw(result)
        orange_axis_pixels = np.all(
            canvas == np.array((0, 165, 255), dtype=np.uint8), axis=2
        ).sum()
        green_ball_pixels = np.all(
            canvas == np.array((0, 220, 0), dtype=np.uint8), axis=2
        ).sum()
        self.assertGreater(orange_axis_pixels, 20)
        self.assertGreater(green_ball_pixels, 20)

    def test_rgb_contour_mode_draws_valid_green_centimetre_ruler(self) -> None:
        image, _ = synthetic_flat_scene()
        contour = segment_white_tube(image, TubeGeometryConfig())
        result = DetectionResult(
            frame_id=1,
            captured_monotonic=1.0,
            completed_monotonic=1.01,
            inference_ms=10.0,
            detections=(Detection(630, 390, 650, 410, 0.9),),
            source_image=image,
            status=TrackStatus.MEASURED,
            ball_status=TrackStatus.MEASURED,
            tube_contour=contour,
            position_cm=0.0,
            position_source="rgb-contour",
        )
        canvas = self._draw(result)
        green_axis_pixels = np.all(
            canvas == np.array((0, 220, 0), dtype=np.uint8), axis=2
        ).sum()
        self.assertGreater(green_axis_pixels, 20)


if __name__ == "__main__":
    unittest.main()
