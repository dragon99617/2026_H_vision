from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import (
    CameraIntrinsics,
    Detection,
    RgbdCalibration,
    TubeContour,
    TubePose,
)


@dataclass(frozen=True)
class TubeGeometryConfig:
    tube_length_cm: float = 25.0
    min_projected_length_px: float = 400.0
    min_valid_bin_ratio: float = 0.60
    max_rms_m: float = 0.004
    min_observed_length_m: float = 0.22
    max_observed_length_m: float = 0.28
    pose_max_age_ms: float = 100.0
    sign_deadband_px: float = 20.0
    color_mode: str = "auto"
    white_min_gray: int = 105
    white_max_saturation: int = 150
    green_hue_min: int = 35
    green_hue_max: int = 95
    green_min_saturation: int = 45
    green_min_value: int = 25
    green_min_excess: int = 3
    mask_expand_px: int = 18
    longitudinal_bins: int = 40
    depth_stride: int = 4
    min_depth_m: float = 0.12
    max_depth_m: float = 2.5

    def __post_init__(self) -> None:
        if self.tube_length_cm <= 0:
            raise ValueError("tube_length_cm must be positive")
        if not 0.0 < self.min_valid_bin_ratio <= 1.0:
            raise ValueError("min_valid_bin_ratio must be in (0, 1]")
        if self.longitudinal_bins < 8:
            raise ValueError("longitudinal_bins must be at least 8")
        if self.depth_stride < 1:
            raise ValueError("depth_stride must be positive")
        if self.color_mode not in ("auto", "white", "dark-green"):
            raise ValueError("color_mode must be auto, white, or dark-green")
        if not 0 <= self.white_min_gray <= 255:
            raise ValueError("white_min_gray must be in 0..255")
        if not 0 <= self.white_max_saturation <= 255:
            raise ValueError("white_max_saturation must be in 0..255")
        if not 0 <= self.green_hue_min <= self.green_hue_max <= 179:
            raise ValueError("green hue range must be ordered inside 0..179")
        if not 0 <= self.green_min_saturation <= 255:
            raise ValueError("green_min_saturation must be in 0..255")
        if not 0 <= self.green_min_value <= 255:
            raise ValueError("green_min_value must be in 0..255")
        if not 0 <= self.green_min_excess <= 255:
            raise ValueError("green_min_excess must be in 0..255")


def _invalid_contour(reason: str) -> TubeContour:
    return TubeContour(valid=False, reason=reason)


def segment_tube(
    image: np.ndarray,
    config: TubeGeometryConfig,
    excluded_box: Optional[Detection] = None,
) -> TubeContour:
    """Extract one elongated white or dark-green half-pipe on a dark field."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return _invalid_contour("invalid color image")
    height, width = image.shape[:2]
    analysis_scale = min(1.0, 320.0 / max(1.0, float(width)))
    if analysis_scale < 1.0:
        working = cv2.resize(
            image,
            (
                max(1, int(round(width * analysis_scale))),
                max(1, int(round(height * analysis_scale))),
            ),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        working = image
    working_height, working_width = working.shape[:2]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    exclusion = None
    if excluded_box is not None:
        margin = max(2, int(config.mask_expand_px * analysis_scale * 0.5))
        x1 = max(0, int(math.floor(excluded_box.x1 * analysis_scale)) - margin)
        y1 = max(0, int(math.floor(excluded_box.y1 * analysis_scale)) - margin)
        x2 = min(
            working_width,
            int(math.ceil(excluded_box.x2 * analysis_scale)) + margin,
        )
        y2 = min(
            working_height,
            int(math.ceil(excluded_box.y2 * analysis_scale)) + margin,
        )
        exclusion = (x1, y1, x2, y2)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    green_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )
    minimum_area = max(95.0, working_width * working_height * 0.006)

    def exclude_ball(mask: np.ndarray) -> np.ndarray:
        if exclusion is not None:
            x1, y1, x2, y2 = exclusion
            mask[y1:y2, x1:x2] = 0
        return mask

    def white_pixels(adaptive: bool) -> np.ndarray:
        threshold = config.white_min_gray
        if adaptive:
            otsu_threshold, _ = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            threshold = max(
                config.white_min_gray,
                min(220, int(otsu_threshold)),
            )
        bright = cv2.compare(gray, threshold, cv2.CMP_GE)
        low_saturation = cv2.compare(
            hsv[:, :, 1],
            config.white_max_saturation,
            cv2.CMP_LE,
        )
        return cv2.bitwise_and(bright, low_saturation)

    def find_candidates(raw_mask: np.ndarray, close_kernel: np.ndarray):
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum_area or len(contour) < 5:
                continue
            points = contour.reshape(-1, 2).astype(np.float64)
            centered = points - points.mean(axis=0)
            covariance = centered.T.dot(centered) / max(1, len(points) - 1)
            values, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, int(np.argmax(values))]
            longitudinal = points.dot(axis)
            lateral_axis = np.array((-axis[1], axis[0]))
            lateral = points.dot(lateral_axis)
            length = float(longitudinal.max() - longitudinal.min())
            tube_width = float(lateral.max() - lateral.min())
            elongation = length / max(1.0, tube_width)
            if (
                length
                < config.min_projected_length_px * analysis_scale * 0.65
                or elongation < 2.2
            ):
                continue
            score = length * math.sqrt(area) * min(8.0, elongation)
            candidates.append((score, contour, length))
        return candidates

    working_contour = None
    selected_color = ""
    if config.color_mode in ("auto", "dark-green"):
        green_hsv = cv2.inRange(
            hsv,
            (
                config.green_hue_min,
                config.green_min_saturation,
                config.green_min_value,
            ),
            (config.green_hue_max, 255, 255),
        )
        blue, green_channel, red = cv2.split(working)
        green_excess = cv2.subtract(
            green_channel,
            cv2.max(blue, red),
        )
        dominant_green = cv2.compare(
            green_excess,
            config.green_min_excess,
            cv2.CMP_GE,
        )
        green_core = cv2.bitwise_and(green_hsv, dominant_green)
        # Preserve white specular highlights only when they touch green paint.
        highlight_neighborhood = cv2.dilate(
            green_core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        green_mask = cv2.bitwise_or(
            green_core,
            cv2.bitwise_and(
                white_pixels(adaptive=False),
                highlight_neighborhood,
            ),
        )
        candidates = find_candidates(
            exclude_ball(green_mask),
            green_close_kernel,
        )
        if config.color_mode == "auto":
            full_length_gate = (
                config.min_projected_length_px * analysis_scale
            )
            candidates = [
                candidate
                for candidate in candidates
                if candidate[2] >= full_length_gate
            ]
        if candidates:
            working_contour = max(candidates, key=lambda item: item[0])[1]
            selected_color = "dark-green"
    if working_contour is None and config.color_mode in ("auto", "white"):
        candidates = find_candidates(
            exclude_ball(white_pixels(adaptive=True)),
            white_close_kernel,
        )
        if candidates:
            working_contour = max(candidates, key=lambda item: item[0])[1]
            selected_color = "white"
    if working_contour is None:
        return _invalid_contour(
            "no elongated %s tube contour" % config.color_mode
        )

    working_hull = cv2.convexHull(working_contour)
    hull = np.rint(
        working_hull.astype(np.float64) / analysis_scale
    ).astype(np.int32)
    points = hull.reshape(-1, 2).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T.dot(centered) / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0.0:
        axis = -axis
    lateral_axis = np.array((-axis[1], axis[0]))
    longitudinal = centered.dot(axis)
    lateral = centered.dot(lateral_axis)
    lo = float(longitudinal.min())
    hi = float(longitudinal.max())
    tube_width = float(lateral.max() - lateral.min())
    length = hi - lo
    negative = center + axis * lo
    positive = center + axis * hi

    tube_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(tube_mask, hull, 255)
    length_score = min(1.0, length / max(1.0, config.min_projected_length_px))
    elongation_score = min(1.0, length / max(1.0, 4.0 * tube_width))
    confidence = 0.55 * length_score + 0.45 * elongation_score
    if selected_color == "dark-green":
        confidence = min(1.0, confidence + 0.03)
    valid = length >= config.min_projected_length_px
    return TubeContour(
        valid=valid,
        mask=tube_mask,
        contour=hull,
        endpoint_negative_px=(float(negative[0]), float(negative[1])),
        endpoint_positive_px=(float(positive[0]), float(positive[1])),
        center_px=(float(center[0]), float(center[1])),
        axis_px=(float(axis[0]), float(axis[1])),
        projected_length_px=length,
        width_px=tube_width,
        confidence=float(confidence),
        reason="" if valid else "projected tube length below gate",
    )


def segment_white_tube(
    image: np.ndarray,
    config: TubeGeometryConfig,
    excluded_box: Optional[Detection] = None,
) -> TubeContour:
    """Backward-compatible alias; color selection follows ``config.color_mode``."""
    return segment_tube(image, config, excluded_box)


def point_in_tube(contour: TubeContour, point: Tuple[float, float]) -> bool:
    if not contour.valid or contour.contour is None:
        return False
    distance = cv2.pointPolygonTest(
        contour.contour,
        (float(point[0]), float(point[1])),
        True,
    )
    return distance >= -18.0


def _distortion_coefficients(intrinsics: CameraIntrinsics) -> np.ndarray:
    coefficients = np.zeros(8, dtype=np.float64)
    values = tuple(float(item) for item in intrinsics.distortion)
    coefficients[: min(len(values), len(coefficients))] = values[: len(coefficients)]
    return coefficients


def _camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(
        (
            (intrinsics.fx, 0.0, intrinsics.cx),
            (0.0, intrinsics.fy, intrinsics.cy),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def pixels_to_rays(
    pixels: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        pixels,
        _camera_matrix(intrinsics),
        _distortion_coefficients(intrinsics),
    ).reshape(-1, 2)
    rays = np.column_stack(
        (undistorted[:, 0], undistorted[:, 1], np.ones(len(undistorted)))
    )
    norms = np.linalg.norm(rays, axis=1)
    return rays / np.maximum(norms[:, None], 1e-12)


def project_points(
    points_m: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    z = np.maximum(points[:, 2], 1e-12)
    x = points[:, 0] / z
    y = points[:, 1] / z
    coefficients = _distortion_coefficients(intrinsics)
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numerator / np.maximum(denominator, 1e-12)
    distorted_x = (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    )
    distorted_y = (
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    )
    return np.column_stack(
        (
            distorted_x * intrinsics.fx + intrinsics.cx,
            distorted_y * intrinsics.fy + intrinsics.cy,
        )
    )


def deproject_depth_pixels(
    pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    rays = pixels_to_rays(pixels, intrinsics)
    z_scale = np.asarray(depth_m, dtype=np.float64) / np.maximum(rays[:, 2], 1e-9)
    return rays * z_scale[:, None]


def transform_depth_to_color(
    points_depth_m: np.ndarray, calibration: RgbdCalibration
) -> np.ndarray:
    rotation = np.asarray(
        calibration.depth_to_color_rotation, dtype=np.float64
    ).reshape(3, 3)
    translation = np.asarray(
        calibration.depth_to_color_translation_m, dtype=np.float64
    ).reshape(1, 3)
    return np.asarray(points_depth_m, dtype=np.float64).dot(rotation.T) + translation


def _fit_robust_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 6:
        raise ValueError("not enough points for line fit")
    # Deterministic RANSAC seed: longitudinal bins arrive in axis order, so
    # pairs separated by at least one third of the tube give useful hypotheses.
    best_inliers = np.ones(len(points), dtype=bool)
    best_count = 0
    left_candidates = np.linspace(
        0, max(0, len(points) // 4), num=4, dtype=np.int32
    )
    right_candidates = np.linspace(
        min(len(points) - 1, len(points) * 3 // 4),
        len(points) - 1,
        num=4,
        dtype=np.int32,
    )
    left_grid = np.repeat(left_candidates, len(right_candidates))
    right_grid = np.tile(right_candidates, len(left_candidates))
    origins = points[left_grid]
    directions = points[right_grid] - origins
    norms = np.sqrt(np.sum(directions * directions, axis=1))
    usable_hypotheses = norms >= 1e-8
    if usable_hypotheses.any():
        origins = origins[usable_hypotheses]
        directions = (
            directions[usable_hypotheses]
            / norms[usable_hypotheses, None]
        )
        offsets = points[None, :, :] - origins[:, None, :]
        projections = np.einsum("hpi,hi->hp", offsets, directions)
        residual_vectors = (
            offsets - projections[:, :, None] * directions[:, None, :]
        )
        residuals = np.sqrt(
            np.sum(residual_vectors * residual_vectors, axis=2)
        )
        hypotheses = residuals <= 0.006
        counts = hypotheses.sum(axis=1)
        best_index = int(np.argmax(counts))
        best_count = int(counts[best_index])
        best_inliers = hypotheses[best_index]
    inliers = (
        best_inliers if best_count >= max(6, len(points) // 2) else np.ones(len(points), dtype=bool)
    )
    for _ in range(4):
        sample = points[inliers]
        center = np.median(sample, axis=0)
        covariance = np.cov((sample - center).T)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        direction /= max(1e-12, np.linalg.norm(direction))
        offsets = points - center
        residuals = np.linalg.norm(
            offsets - offsets.dot(direction)[:, None] * direction[None, :],
            axis=1,
        )
        median = float(np.median(residuals[inliers]))
        robust_sigma = 1.4826 * float(
            np.median(np.abs(residuals[inliers] - median))
        )
        threshold = max(0.003, median + 3.0 * max(0.0005, robust_sigma))
        updated = residuals <= threshold
        if updated.sum() < 6 or np.array_equal(updated, inliers):
            break
        inliers = updated
    sample = points[inliers]
    center = sample.mean(axis=0)
    covariance = np.cov((sample - center).T)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    direction /= max(1e-12, np.linalg.norm(direction))
    offsets = sample - center
    residuals = np.linalg.norm(
        offsets - offsets.dot(direction)[:, None] * direction[None, :],
        axis=1,
    )
    rms = float(math.sqrt(float(np.mean(residuals * residuals))))
    return center, direction, rms, inliers


def _invalid_pose(timestamp: float, reason: str, **values) -> TubePose:
    return TubePose(
        valid=False,
        captured_monotonic=timestamp,
        reason=reason,
        **values,
    )


class TubePoseEstimator:
    """Fit the trough center axis from depth points selected by the RGB mask."""

    def __init__(self, config: TubeGeometryConfig) -> None:
        self.config = config
        self.last_pose: Optional[TubePose] = None
        self.last_depth_frame_id = 0
        self._positive_direction: Optional[np.ndarray] = None
        self._depth_ray_key = None
        self._depth_rays = None
        self._depth_xx = None
        self._depth_yy = None

    def _sampled_depth_rays(
        self, shape: Tuple[int, int], intrinsics: CameraIntrinsics
    ):
        key = (
            tuple(shape),
            self.config.depth_stride,
            intrinsics,
        )
        if key != self._depth_ray_key:
            stride = self.config.depth_stride
            yy, xx = np.mgrid[0 : shape[0] : stride, 0 : shape[1] : stride]
            pixels = np.column_stack(
                (xx.reshape(-1), yy.reshape(-1))
            ).astype(np.float64)
            rays = pixels_to_rays(pixels, intrinsics)
            # Store rays with z=1 so multiplication by the sensor depth value
            # directly yields metric XYZ.
            rays = rays / np.maximum(rays[:, 2:3], 1e-12)
            self._depth_ray_key = key
            self._depth_rays = rays
            self._depth_xx = xx
            self._depth_yy = yy
        return self._depth_rays, self._depth_xx, self._depth_yy

    def current_pose(self, now: Optional[float] = None) -> Optional[TubePose]:
        if self.last_pose is None or not self.last_pose.valid:
            return None
        moment = time.monotonic() if now is None else float(now)
        age_ms = (moment - self.last_pose.captured_monotonic) * 1000.0
        if age_ms > self.config.pose_max_age_ms:
            return None
        return self.last_pose

    def update(
        self,
        depth: np.ndarray,
        depth_scale_m: float,
        depth_frame_id: int,
        depth_timestamp: float,
        calibration: RgbdCalibration,
        contour: TubeContour,
        excluded_box: Optional[Detection] = None,
    ) -> TubePose:
        if depth_frame_id == self.last_depth_frame_id:
            pose = self.current_pose()
            return pose if pose is not None else _invalid_pose(
                depth_timestamp, "tube pose expired"
            )
        self.last_depth_frame_id = int(depth_frame_id)
        pose = self._fit(
            depth,
            depth_scale_m,
            depth_timestamp,
            calibration,
            contour,
            excluded_box,
        )
        self.last_pose = pose
        return pose

    def _fit(
        self,
        depth: np.ndarray,
        depth_scale_m: float,
        timestamp: float,
        calibration: RgbdCalibration,
        contour: TubeContour,
        excluded_box: Optional[Detection],
    ) -> TubePose:
        if not contour.valid or contour.mask is None or contour.axis_px is None:
            return _invalid_pose(timestamp, "invalid RGB tube contour")
        if depth is None or depth.ndim != 2:
            return _invalid_pose(timestamp, "invalid depth frame")
        stride = self.config.depth_stride
        rays, xx, yy = self._sampled_depth_rays(
            depth.shape, calibration.depth
        )
        raw = depth[::stride, ::stride].reshape(-1).astype(np.float64)
        depth_m = raw * float(depth_scale_m)
        valid = (
            (raw > 0)
            & (depth_m >= self.config.min_depth_m)
            & (depth_m <= self.config.max_depth_m)
        )
        valid_count = int(valid.sum())
        if valid_count < self.config.longitudinal_bins * 4:
            return _invalid_pose(
                timestamp,
                "too few valid depth samples (%d/%d sampled, %.1f%%)"
                % (
                    valid_count,
                    len(raw),
                    100.0 * valid_count / max(1, len(raw)),
                ),
            )
        points_depth = rays[valid] * depth_m[valid, None]
        points_color = transform_depth_to_color(points_depth, calibration)
        in_front = points_color[:, 2] > 1e-5
        points_color = points_color[in_front]
        if len(points_color) < self.config.longitudinal_bins * 4:
            return _invalid_pose(
                timestamp,
                "too few projected depth samples (%d)"
                % len(points_color),
            )
        color_pixels = project_points(points_color, calibration.color)
        ix = np.rint(color_pixels[:, 0]).astype(np.int32)
        iy = np.rint(color_pixels[:, 1]).astype(np.int32)
        inside = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < contour.mask.shape[1])
            & (iy < contour.mask.shape[0])
        )
        if excluded_box is not None:
            margin = 4.0
            inside &= ~(
                (color_pixels[:, 0] >= excluded_box.x1 - margin)
                & (color_pixels[:, 0] <= excluded_box.x2 + margin)
                & (color_pixels[:, 1] >= excluded_box.y1 - margin)
                & (color_pixels[:, 1] <= excluded_box.y2 + margin)
            )
        selected_indices = np.flatnonzero(inside)
        if len(selected_indices):
            mask_inside = (
                contour.mask[iy[selected_indices], ix[selected_indices]] > 0
            )
            selected_indices = selected_indices[mask_inside]
        if len(selected_indices) < self.config.longitudinal_bins * 4:
            return _invalid_pose(
                timestamp,
                "insufficient depth inside tube mask (%d samples)"
                % len(selected_indices),
            )
        points = points_color[selected_indices]
        pixels = color_pixels[selected_indices]

        center_px = np.asarray(contour.center_px, dtype=np.float64)
        axis_px = np.asarray(contour.axis_px, dtype=np.float64)
        lateral_px = np.array((-axis_px[1], axis_px[0]))
        longitudinal = (pixels - center_px).dot(axis_px)
        lateral = (pixels - center_px).dot(lateral_px)
        half_length = max(1.0, contour.projected_length_px * 0.5)
        bin_count = self.config.longitudinal_bins
        bin_ids = np.floor(
            (longitudinal + half_length)
            * bin_count
            / (2.0 * half_length)
        ).astype(np.int32)
        usable = (bin_ids >= 0) & (bin_ids < bin_count)
        bin_ids = bin_ids[usable]
        side_ids = (lateral[usable] >= 0.0).astype(np.int32)
        grouped = bin_ids * 2 + side_ids
        grouped_points = points[usable]
        counts = np.bincount(grouped, minlength=bin_count * 2).reshape(
            bin_count, 2
        )
        sums = np.empty((bin_count, 2, 3), dtype=np.float64)
        for coordinate in range(3):
            sums[:, :, coordinate] = np.bincount(
                grouped,
                weights=grouped_points[:, coordinate],
                minlength=bin_count * 2,
            ).reshape(bin_count, 2)
        means = sums / np.maximum(counts[:, :, None], 1)
        both_sides = (counts[:, 0] >= 2) & (counts[:, 1] >= 2)
        left_edge_points = means[both_sides, 0, :]
        right_edge_points = means[both_sides, 1, :]
        midpoint_array = (left_edge_points + right_edge_points) * 0.5
        valid_bins = int(both_sides.sum())
        valid_ratio = valid_bins / float(self.config.longitudinal_bins)
        if valid_ratio < self.config.min_valid_bin_ratio:
            return _invalid_pose(
                timestamp,
                "valid depth bin ratio below gate",
                valid_bin_ratio=valid_ratio,
            )
        try:
            left_center, left_direction, left_rms, _ = _fit_robust_line(
                left_edge_points
            )
            right_center, right_direction, right_rms, _ = _fit_robust_line(
                right_edge_points
            )
        except ValueError as exc:
            return _invalid_pose(
                timestamp, str(exc), valid_bin_ratio=valid_ratio
            )
        if float(left_direction.dot(right_direction)) < 0.0:
            right_direction = -right_direction
        direction = left_direction + right_direction
        direction /= max(1e-12, np.linalg.norm(direction))
        center = (left_center + right_center) * 0.5
        rms = max(left_rms, right_rms)
        along = (midpoint_array - center).dot(direction)
        observed_length = float(
            np.percentile(along, 98) - np.percentile(along, 2)
        )
        if rms > self.config.max_rms_m:
            return _invalid_pose(
                timestamp,
                "3D tube fit RMS above gate",
                observed_length_m=observed_length,
                rms_m=rms,
                valid_bin_ratio=valid_ratio,
            )
        if not (
            self.config.min_observed_length_m
            <= observed_length
            <= self.config.max_observed_length_m
        ):
            return _invalid_pose(
                timestamp,
                "observed tube length outside gate",
                observed_length_m=observed_length,
                rms_m=rms,
                valid_bin_ratio=valid_ratio,
            )
        low = float(np.percentile(along, 2))
        high = float(np.percentile(along, 98))
        center = center + direction * ((low + high) * 0.5)

        half_m = self.config.tube_length_cm / 200.0
        endpoint_a = center - direction * half_m
        endpoint_b = center + direction * half_m
        projected = project_points(
            np.vstack((endpoint_a, endpoint_b)), calibration.color
        )
        dx = float(projected[1, 0] - projected[0, 0])
        if abs(dx) >= self.config.sign_deadband_px:
            if dx < 0.0:
                direction = -direction
            self._positive_direction = direction.copy()
        elif self._positive_direction is not None:
            if float(direction.dot(self._positive_direction)) < 0.0:
                direction = -direction
        else:
            return _invalid_pose(
                timestamp,
                "positive end ambiguous inside sign deadband",
                observed_length_m=observed_length,
                rms_m=rms,
                valid_bin_ratio=valid_ratio,
            )
        endpoint_negative = center - direction * half_m
        endpoint_positive = center + direction * half_m
        projected = project_points(
            np.vstack((endpoint_negative, endpoint_positive)),
            calibration.color,
        )
        horizontal_norm = math.hypot(float(direction[0]), float(direction[1]))
        pitch = math.degrees(
            math.atan2(float(direction[2]), max(1e-12, horizontal_norm))
        )
        rms_score = max(0.0, 1.0 - rms / self.config.max_rms_m)
        length_score = max(
            0.0, 1.0 - abs(observed_length - 0.25) / 0.03
        )
        confidence = (
            0.30 * contour.confidence
            + 0.30 * min(1.0, valid_ratio)
            + 0.25 * rms_score
            + 0.15 * length_score
        )
        return TubePose(
            valid=True,
            captured_monotonic=timestamp,
            center_m=tuple(float(value) for value in center),
            direction=tuple(float(value) for value in direction),
            endpoint_negative_m=tuple(float(value) for value in endpoint_negative),
            endpoint_positive_m=tuple(float(value) for value in endpoint_positive),
            endpoint_negative_px=tuple(float(value) for value in projected[0]),
            endpoint_positive_px=tuple(float(value) for value in projected[1]),
            observed_length_m=observed_length,
            fitted_length_m=self.config.tube_length_cm / 100.0,
            rms_m=rms,
            valid_bin_ratio=valid_ratio,
            confidence=float(min(1.0, max(0.0, confidence))),
            pitch_degrees=pitch,
        )


def closest_axis_position_cm(
    pixel: Tuple[float, float],
    pose: TubePose,
    color_intrinsics: CameraIntrinsics,
    tube_length_cm: float = 25.0,
) -> Tuple[float, Tuple[float, float]]:
    """Return signed axis position and the closest point projected to RGB."""
    if not pose.valid or pose.center_m is None or pose.direction is None:
        raise ValueError("valid tube pose is required")
    ray = pixels_to_rays(np.asarray((pixel,), dtype=np.float64), color_intrinsics)[0]
    center = np.asarray(pose.center_m, dtype=np.float64)
    direction = np.asarray(pose.direction, dtype=np.float64)
    direction /= max(1e-12, np.linalg.norm(direction))
    # Solve t*ray - s*axis = center in least-squares form. The camera origin is 0.
    solution, _, _, _ = np.linalg.lstsq(
        np.column_stack((ray, -direction)), center, rcond=None
    )
    position_m = float(solution[1])
    half_m = float(tube_length_cm) / 200.0
    position_m = min(half_m, max(-half_m, position_m))
    projected_point = center + direction * position_m
    projected_pixel = project_points(
        projected_point.reshape(1, 3), color_intrinsics
    )[0]
    return (
        position_m * 100.0,
        (float(projected_pixel[0]), float(projected_pixel[1])),
    )


def contour_axis_position_cm(
    pixel: Tuple[float, float],
    contour: TubeContour,
    tube_length_cm: float = 25.0,
) -> Tuple[float, Tuple[float, float]]:
    """Project an RGB ball centre onto the detected 2D tube axis."""
    if (
        not contour.valid
        or contour.endpoint_negative_px is None
        or contour.endpoint_positive_px is None
    ):
        raise ValueError("valid RGB tube contour is required")
    negative = np.asarray(contour.endpoint_negative_px, dtype=np.float64)
    positive = np.asarray(contour.endpoint_positive_px, dtype=np.float64)
    point = np.asarray(pixel, dtype=np.float64)
    axis = positive - negative
    squared_length = float(axis.dot(axis))
    if squared_length < 1.0:
        raise ValueError("RGB tube contour axis is too short")
    fraction = float((point - negative).dot(axis) / squared_length)
    fraction = min(1.0, max(0.0, fraction))
    projected = negative + axis * fraction
    position_cm = (fraction - 0.5) * float(tube_length_cm)
    return (
        position_cm,
        (float(projected[0]), float(projected[1])),
    )
