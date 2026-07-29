from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from .latest import LatestValue, RollingRate
from .tube_geometry import (
    TubeGeometryConfig,
    TubePoseEstimator,
    segment_white_tube,
)
from .types import (
    DetectionResult,
    FramePacket,
    TubeContour,
    TubePose,
    TubeState,
)


class TubePoseWorker:
    """Process only fresh depth frames; stale queued frames are never retained."""

    def __init__(
        self,
        frames: LatestValue[FramePacket],
        ball_results: LatestValue[DetectionResult],
        output: LatestValue[TubeState],
        stop_event: threading.Event,
        config: TubeGeometryConfig,
    ) -> None:
        self.frames = frames
        self.ball_results = ball_results
        self.output = output
        self.stop_event = stop_event
        self.estimator = TubePoseEstimator(config)
        self.config = config
        self.rate = RollingRate(3.0)
        self.processed_count = 0
        self.valid_count = 0
        self.skipped_depth_frames = 0
        self.last_geometry_ms = 0.0
        self.last_error: Optional[str] = None
        self.thread = threading.Thread(
            target=self._run, name="tube-pose", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout=timeout)

    def pose_fps(self) -> float:
        return self.rate.value(time.monotonic())

    def _run(self) -> None:
        sequence = 0
        previous_depth_id = 0
        while not self.stop_event.is_set():
            next_sequence, packet = self.frames.wait_after(
                sequence, timeout=0.5
            )
            if packet is None or next_sequence <= sequence:
                continue
            sequence = next_sequence
            if (
                packet.depth is None
                or packet.calibration is None
                or packet.depth_frame_id <= 0
                or packet.depth_frame_id == previous_depth_id
            ):
                continue
            if previous_depth_id:
                self.skipped_depth_frames += max(
                    0, packet.depth_frame_id - previous_depth_id - 1
                )
            previous_depth_id = packet.depth_frame_id
            _, ball_result = self.ball_results.get()
            excluded = (
                ball_result.detection if ball_result is not None else None
            )
            started = time.monotonic()
            try:
                contour = segment_white_tube(
                    packet.image, self.config
                )
                pose = self.estimator.update(
                    packet.depth,
                    packet.depth_scale_m,
                    packet.depth_frame_id,
                    packet.depth_captured_monotonic,
                    packet.calibration,
                    contour,
                    excluded,
                )
            except Exception as exc:
                self.last_error = str(exc)
                print(
                    "tube pose error: %s" % exc,
                    file=sys.stderr,
                    flush=True,
                )
                contour = TubeContour(
                    valid=False,
                    reason="tube pose worker exception",
                )
                pose = TubePose(
                    valid=False,
                    captured_monotonic=packet.depth_captured_monotonic,
                    reason=str(exc),
                )
            completed = time.monotonic()
            self.last_geometry_ms = (completed - started) * 1000.0
            self.processed_count += 1
            if pose.valid:
                self.valid_count += 1
            self.rate.tick(completed)
            self.output.publish(
                TubeState(
                    depth_frame_id=packet.depth_frame_id,
                    processed_monotonic=completed,
                    contour=contour,
                    pose=pose,
                    geometry_ms=self.last_geometry_ms,
                )
            )
