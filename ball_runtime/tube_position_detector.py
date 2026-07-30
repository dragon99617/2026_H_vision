from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

from .tube_geometry import (
    TubeGeometryConfig,
    TubePoseEstimator,
    closest_axis_position_cm,
    point_in_tube,
    segment_tube,
)
from .types import (
    DetectionResult,
    FramePacket,
    TrackStatus,
    TubeContour,
    TubePose,
)


class TubePositionDetector:
    """Add live 3D trough pose and signed centimetres to the ball tracker."""

    def __init__(
        self,
        ball_detector,
        config: TubeGeometryConfig,
        tube_states=None,
    ) -> None:
        self.ball_detector = ball_detector
        self.config = config
        self.pose_estimator = TubePoseEstimator(config)
        self.pose_updates = 0
        self.valid_pose_updates = 0
        self.last_pose_reason = "no depth pose"
        self.last_contour = None
        self.tube_states = tube_states

    @property
    def input_shape(self):
        return self.ball_detector.input_shape

    @property
    def output_shape(self):
        return self.ball_detector.output_shape

    @property
    def fast_path(self):
        return "%s+tube-rgbd" % self.ball_detector.fast_path

    def close(self) -> None:
        self.ball_detector.close()

    def detect(self, packet: FramePacket) -> DetectionResult:
        raw = self.ball_detector.detect(packet)
        geometry_started = time.monotonic()
        detection = raw.detection

        pose: Optional[TubePose] = None
        external_geometry_ms = 0.0
        if self.tube_states is not None:
            _, state = self.tube_states.get()
            if state is None:
                contour = TubeContour(
                    valid=False, reason="waiting for first tube pose"
                )
            else:
                contour = state.contour
                pose = state.pose
                external_geometry_ms = state.geometry_ms
                self.last_pose_reason = state.pose.reason
                if (
                    pose is not None
                    and (
                        packet.captured_monotonic
                        - pose.captured_monotonic
                    )
                    * 1000.0
                    > self.config.pose_max_age_ms
                ):
                    pose = replace(
                        pose,
                        valid=False,
                        reason="tube pose expired",
                    )
                    self.last_pose_reason = "tube pose expired"
            is_new_depth = False
        else:
            contour = None
            is_new_depth = (
                packet.depth is not None
                and packet.calibration is not None
                and packet.depth_frame_id > 0
                and packet.depth_frame_id
                != self.pose_estimator.last_depth_frame_id
            )
        if self.tube_states is None and is_new_depth:
            contour = segment_tube(packet.image, self.config)
            self.last_contour = contour
            self.pose_updates += 1
            pose = self.pose_estimator.update(
                packet.depth,
                packet.depth_scale_m,
                packet.depth_frame_id,
                packet.depth_captured_monotonic,
                packet.calibration,
                contour,
                detection,
            )
            self.last_pose_reason = pose.reason
            if pose.valid:
                self.valid_pose_updates += 1
        elif self.tube_states is None:
            contour = self.last_contour
            pose = self.pose_estimator.current_pose(packet.captured_monotonic)
            if pose is None:
                self.last_pose_reason = "tube pose unavailable or older than %.0f ms" % (
                    self.config.pose_max_age_ms
                )

        if contour is None:
            contour = segment_tube(packet.image, self.config)
            self.last_contour = contour
        if detection is not None and not point_in_tube(contour, detection.center):
            self.ball_detector.reset()
            detection = None
            raw = replace(
                raw,
                detections=(),
                status=TrackStatus.LOST,
                missed_frames=0,
            )

        ball_status = raw.status
        output_status = TrackStatus.LOST
        position_cm = None
        projected = None
        tube_confidence = (
            pose.confidence
            if pose is not None and pose.valid and contour.valid
            else 0.0
        )
        if (
            detection is not None
            and contour.valid
            and pose is not None
            and pose.valid
            and packet.calibration is not None
        ):
            age_ms = (
                packet.captured_monotonic - pose.captured_monotonic
            ) * 1000.0
            if age_ms <= self.config.pose_max_age_ms:
                position_cm, projected = closest_axis_position_cm(
                    detection.center,
                    pose,
                    packet.calibration.color,
                    self.config.tube_length_cm,
                )
                output_status = ball_status

        completed = time.monotonic()
        depth_timestamp = (
            pose.captured_monotonic
            if pose is not None
            else packet.depth_captured_monotonic
        )
        depth_age_ms = (
            max(0.0, (packet.captured_monotonic - depth_timestamp) * 1000.0)
            if depth_timestamp > 0.0
            else float("inf")
        )
        return DetectionResult(
            frame_id=raw.frame_id,
            captured_monotonic=raw.captured_monotonic,
            completed_monotonic=completed,
            inference_ms=raw.inference_ms + (completed - geometry_started) * 1000.0,
            detections=raw.detections,
            source_image=raw.source_image,
            status=output_status,
            preprocess_ms=raw.preprocess_ms,
            gpu_ms=raw.gpu_ms,
            postprocess_ms=raw.postprocess_ms,
            missed_frames=raw.missed_frames,
            ball_status=ball_status,
            tube_contour=contour,
            tube_pose=pose,
            position_cm=position_cm,
            position_projected_px=projected,
            tube_confidence=tube_confidence,
            geometry_ms=(
                external_geometry_ms
                if self.tube_states is not None
                else (completed - geometry_started) * 1000.0
            ),
            depth_age_ms=depth_age_ms,
            position_source="rgbd",
        )
