from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Callable, Optional

from .latest import LatestValue, RollingRate
from .runtime_metrics import TrackingStats
from .types import DetectionResult, FramePacket


class InferenceWorker:
    def __init__(
        self,
        detector,
        frames: LatestValue[FramePacket],
        results: LatestValue[DetectionResult],
        stop_event: threading.Event,
        on_result: Optional[Callable[[DetectionResult], None]] = None,
    ) -> None:
        self.detector = detector
        self.frames = frames
        self.results = results
        self.stop_event = stop_event
        self.on_result = on_result
        self.rate = RollingRate(3.0)
        self.result_count = 0
        self.skipped_frames = 0
        self.last_inference_ms = 0.0
        self.last_latency_ms = 0.0
        self.latencies_ms = deque(maxlen=600)
        self.tracking_stats = TrackingStats()
        self.last_error: Optional[str] = None
        self.thread = threading.Thread(target=self._run, name="tensorrt-inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout=timeout)

    def inference_fps(self) -> float:
        return self.rate.value(time.monotonic())

    def latency_p95_ms(self) -> float:
        values = sorted(self.latencies_ms)
        if not values:
            return 0.0
        return float(values[round((len(values) - 1) * 0.95)])

    def _run(self) -> None:
        sequence = 0
        previous_frame_id = 0
        while not self.stop_event.is_set():
            next_sequence, packet = self.frames.wait_after(sequence, timeout=0.5)
            if packet is None or next_sequence <= sequence:
                continue
            sequence = next_sequence
            if previous_frame_id:
                self.skipped_frames += max(0, packet.frame_id - previous_frame_id - 1)
            previous_frame_id = packet.frame_id
            try:
                result = self.detector.detect(packet)
            except Exception as exc:
                self.last_error = str(exc)
                print("fatal inference error: %s" % exc, file=sys.stderr, flush=True)
                self.stop_event.set()
                return
            self.result_count += 1
            self.last_inference_ms = result.inference_ms
            self.last_latency_ms = result.latency_ms
            self.latencies_ms.append(result.latency_ms)
            self.tracking_stats.record(result)
            self.rate.tick(result.completed_monotonic)
            self.results.publish(result)
            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception as exc:
                    print("result callback error: %s" % exc, file=sys.stderr, flush=True)
