from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from ball_runtime.tube_geometry import (
    TubeGeometryConfig,
    TubePoseEstimator,
    closest_axis_position_cm,
    contour_axis_position_cm,
    point_in_tube,
    project_points,
    segment_tube,
    segment_white_tube,
)
from ball_runtime.types import (
    CameraIntrinsics,
    RgbdCalibration,
    TubePose,
)


def calibration() -> RgbdCalibration:
    color = CameraIntrinsics(
        1280, 800, 1000.0, 1000.0, 640.0, 400.0, (0.0,) * 8
    )
    depth = CameraIntrinsics(
        640, 400, 500.0, 500.0, 320.0, 200.0, (0.0,) * 8
    )
    return RgbdCalibration(
        color=color,
        depth=depth,
        depth_to_color_rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        depth_to_color_translation_m=(0.0, 0.0, 0.0),
    )


def synthetic_flat_scene(tube_color=(245, 245, 245)):
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    cv2.rectangle(image, (430, 330), (850, 470), tube_color, -1)
    depth = np.zeros((400, 640), dtype=np.uint16)
    cv2.rectangle(depth, (215, 165), (425, 235), 600, -1)
    return image, depth


class TubeGeometryTests(unittest.TestCase):
    def test_auto_mode_detects_dark_green_tube(self) -> None:
        image, _ = synthetic_flat_scene((30, 75, 40))
        contour = segment_tube(image, TubeGeometryConfig())
        self.assertTrue(contour.valid, contour.reason)
        self.assertGreaterEqual(contour.projected_length_px, 400.0)
        self.assertTrue(point_in_tube(contour, (640.0, 400.0)))

    def test_dark_green_mode_keeps_specular_highlight(self) -> None:
        image, _ = synthetic_flat_scene((25, 65, 35))
        cv2.rectangle(image, (600, 335), (680, 465), (235, 235, 235), -1)
        contour = segment_tube(
            image,
            TubeGeometryConfig(color_mode="dark-green"),
        )
        self.assertTrue(contour.valid, contour.reason)
        self.assertGreaterEqual(contour.projected_length_px, 400.0)

    def test_dark_green_mode_rejects_white_tube(self) -> None:
        image, _ = synthetic_flat_scene()
        contour = segment_tube(
            image,
            TubeGeometryConfig(color_mode="dark-green"),
        )
        self.assertFalse(contour.valid)

    def test_auto_mode_prefers_green_tube_over_white_distractor(self) -> None:
        image = np.zeros((800, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (430, 420), (850, 560), (30, 75, 40), -1)
        cv2.rectangle(image, (300, 120), (980, 220), (245, 245, 245), -1)
        contour = segment_tube(image, TubeGeometryConfig())
        self.assertTrue(contour.valid, contour.reason)
        self.assertGreater(contour.center_px[1], 350.0)

    def test_auto_mode_ignores_undersized_green_distractor(self) -> None:
        image = np.zeros((800, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (490, 480), (790, 550), (30, 75, 40), -1)
        cv2.rectangle(image, (430, 160), (850, 300), (245, 245, 245), -1)
        contour = segment_tube(image, TubeGeometryConfig())
        self.assertTrue(contour.valid, contour.reason)
        self.assertLess(contour.center_px[1], 350.0)

    def test_segment_and_fit_flat_tube(self) -> None:
        config = TubeGeometryConfig()
        image, depth = synthetic_flat_scene()
        contour = segment_white_tube(image, config)
        self.assertTrue(contour.valid, contour.reason)
        self.assertGreaterEqual(contour.projected_length_px, 400.0)
        self.assertTrue(point_in_tube(contour, (640.0, 400.0)))
        self.assertFalse(point_in_tube(contour, (10.0, 10.0)))

        estimator = TubePoseEstimator(config)
        pose = estimator.update(
            depth, 0.001, 1, 10.0, calibration(), contour
        )
        self.assertTrue(pose.valid, pose.reason)
        self.assertGreaterEqual(pose.valid_bin_ratio, 0.60)
        self.assertLessEqual(pose.rms_m, 0.004)
        self.assertAlmostEqual(pose.fitted_length_m, 0.25, places=6)
        self.assertLess(
            pose.endpoint_positive_px[0], pose.endpoint_negative_px[0]
        )

    def test_positive_end_defaults_left_and_can_be_overridden(self) -> None:
        image, _ = synthetic_flat_scene()
        default_contour = segment_white_tube(image, TubeGeometryConfig())
        self.assertLess(
            default_contour.endpoint_positive_px[0],
            default_contour.endpoint_negative_px[0],
        )

        legacy_contour = segment_white_tube(
            image,
            TubeGeometryConfig(positive_end="image-right"),
        )
        self.assertGreater(
            legacy_contour.endpoint_positive_px[0],
            legacy_contour.endpoint_negative_px[0],
        )

    def test_signed_position_for_multiple_pitch_angles(self) -> None:
        calib = calibration()
        for pitch_degrees in (-20.0, 0.0, 20.0):
            pitch = math.radians(pitch_degrees)
            direction = np.array(
                (math.cos(pitch), 0.0, math.sin(pitch)), dtype=np.float64
            )
            center = np.array((0.0, 0.0, 0.70), dtype=np.float64)
            pose = TubePose(
                valid=True,
                captured_monotonic=1.0,
                center_m=tuple(center),
                direction=tuple(direction),
            )
            for expected_cm in (-12.5, -6.25, 0.0, 6.25, 12.5):
                point = center + direction * (expected_cm / 100.0)
                pixel = project_points(point.reshape(1, 3), calib.color)[0]
                actual_cm, _ = closest_axis_position_cm(
                    tuple(pixel), pose, calib.color
                )
                self.assertAlmostEqual(actual_cm, expected_cm, places=5)

    def test_position_is_clamped_to_physical_ends(self) -> None:
        calib = calibration()
        pose = TubePose(
            valid=True,
            captured_monotonic=1.0,
            center_m=(0.0, 0.0, 0.7),
            direction=(1.0, 0.0, 0.0),
        )
        for x_m, expected in ((-0.5, -12.5), (0.5, 12.5)):
            pixel = project_points(
                np.array(((x_m, 0.0, 0.7),)), calib.color
            )[0]
            actual, _ = closest_axis_position_cm(
                tuple(pixel), pose, calib.color
            )
            self.assertEqual(actual, expected)

    def test_rgb_contour_projection_maps_ends_center_and_clamps(self) -> None:
        image, _ = synthetic_flat_scene()
        contour = segment_white_tube(image, TubeGeometryConfig())
        negative = np.asarray(contour.endpoint_negative_px)
        positive = np.asarray(contour.endpoint_positive_px)
        for fraction, expected in (
            (-0.2, -12.5),
            (0.0, -12.5),
            (0.25, -6.25),
            (0.5, 0.0),
            (0.75, 6.25),
            (1.0, 12.5),
            (1.2, 12.5),
        ):
            point = negative + (positive - negative) * fraction
            actual, projected = contour_axis_position_cm(
                tuple(point), contour, 25.0
            )
            self.assertAlmostEqual(actual, expected, places=5)
            self.assertEqual(len(projected), 2)

    def test_pose_expires_after_configured_age(self) -> None:
        estimator = TubePoseEstimator(
            TubeGeometryConfig(pose_max_age_ms=100.0)
        )
        estimator.last_pose = TubePose(
            valid=True,
            captured_monotonic=2.0,
            center_m=(0.0, 0.0, 0.6),
            direction=(1.0, 0.0, 0.0),
        )
        self.assertIsNotNone(estimator.current_pose(2.099))
        self.assertIsNone(estimator.current_pose(2.101))

    def test_failed_fresh_depth_invalidates_previous_pose(self) -> None:
        config = TubeGeometryConfig()
        image, depth = synthetic_flat_scene()
        contour = segment_white_tube(image, config)
        estimator = TubePoseEstimator(config)
        self.assertTrue(
            estimator.update(
                depth, 0.001, 1, 1.0, calibration(), contour
            ).valid
        )
        failed = estimator.update(
            np.zeros_like(depth), 0.001, 2, 1.03, calibration(), contour
        )
        self.assertFalse(failed.valid)
        self.assertIsNone(estimator.current_pose(1.04))

    def test_near_vertical_tube_without_direction_history_is_invalid(self) -> None:
        image = np.zeros((800, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (570, 190), (710, 610), (245, 245, 245), -1)
        depth = np.zeros((400, 640), dtype=np.uint16)
        cv2.rectangle(depth, (285, 95), (355, 305), 600, -1)
        config = TubeGeometryConfig()
        contour = segment_white_tube(image, config)
        self.assertTrue(contour.valid)
        pose = TubePoseEstimator(config).update(
            depth, 0.001, 1, 1.0, calibration(), contour
        )
        self.assertFalse(pose.valid)
        self.assertIn("sign deadband", pose.reason)

    def test_near_vertical_tube_keeps_existing_direction_history(self) -> None:
        config = TubeGeometryConfig()
        horizontal_image, horizontal_depth = synthetic_flat_scene()
        estimator = TubePoseEstimator(config)
        horizontal = estimator.update(
            horizontal_depth,
            0.001,
            1,
            1.0,
            calibration(),
            segment_white_tube(horizontal_image, config),
        )
        self.assertTrue(horizontal.valid)

        vertical_image = np.zeros((800, 1280, 3), dtype=np.uint8)
        cv2.rectangle(
            vertical_image, (570, 190), (710, 610), (245, 245, 245), -1
        )
        vertical_depth = np.zeros((400, 640), dtype=np.uint16)
        cv2.rectangle(vertical_depth, (285, 95), (355, 305), 600, -1)
        vertical = estimator.update(
            vertical_depth,
            0.001,
            2,
            1.03,
            calibration(),
            segment_white_tube(vertical_image, config),
        )
        self.assertTrue(vertical.valid, vertical.reason)
        self.assertLess(
            abs(
                vertical.endpoint_positive_px[0]
                - vertical.endpoint_negative_px[0]
            ),
            20.0,
        )


if __name__ == "__main__":
    unittest.main()
