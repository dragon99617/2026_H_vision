from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from .latest import LatestValue, RollingRate
from .rgb_tube_position_detector import smooth_rgb_tube_contour
from .tube_geometry import TubeGeometryConfig, segment_tube
from .types import FramePacket, RgbTubeState, TubeContour


class RgbTubeWorker:
    """Fit the latest RGB tube contour without blocking ball inference."""

    def __init__(
        self,
        frames: LatestValue[FramePacket],
        output: LatestValue[RgbTubeState],
        stop_event: threading.Event,
        config: TubeGeometryConfig,
        smoothing_alpha: float = 0.35,
    ) -> None:
        self.frames = frames
        self.output = output
        self.stop_event = stop_event
        self.config = config
        self.smoothing_alpha = float(smoothing_alpha)
        self.rate = RollingRate(2.0)
        self.processed_count = 0
        self.valid_count = 0
        self.skipped_frames = 0
        self.last_geometry_ms = 0.0
        self.last_error: Optional[str] = None
        self.thread = threading.Thread(
            target=self._run,
            name="rgb-tube-contour",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout=timeout)

    def contour_fps(self) -> float:
        return self.rate.value(time.monotonic())

    def _run(self) -> None:
        sequence = 0
        previous_frame_id = 0
        previous_contour = None
        while not self.stop_event.is_set():
            next_sequence, packet = self.frames.wait_after(
                sequence,
                timeout=0.5,
            )
            if packet is None or next_sequence <= sequence:
                continue
            sequence = next_sequence
            if previous_frame_id:
                self.skipped_frames += max(
                    0, packet.frame_id - previous_frame_id - 1
                )
            previous_frame_id = packet.frame_id
            started = time.monotonic()
            try:
                contour = segment_tube(packet.image, self.config)
                if contour.valid:
                    contour = smooth_rgb_tube_contour(
                        previous_contour,
                        contour,
                        self.smoothing_alpha,
                    )
                    previous_contour = contour
                else:
                    previous_contour = None
            except Exception as exc:
                self.last_error = str(exc)
                print(
                    "RGB tube contour error: %s" % exc,
                    file=sys.stderr,
                    flush=True,
                )
                contour = TubeContour(
                    valid=False,
                    reason="RGB tube worker exception: %s" % exc,
                )
                previous_contour = None
            completed = time.monotonic()
            self.last_geometry_ms = (completed - started) * 1000.0
            self.processed_count += 1
            if contour.valid:
                self.valid_count += 1
            self.rate.tick(completed)
            self.output.publish(
                RgbTubeState(
                    frame_id=packet.frame_id,
                    captured_monotonic=packet.captured_monotonic,
                    processed_monotonic=completed,
                    contour=contour,
                    geometry_ms=self.last_geometry_ms,
                )
            )
