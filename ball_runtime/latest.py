from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


class LatestValue(Generic[T]):
    """A one-slot, overwrite-on-publish exchange."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: Optional[T] = None
        self._sequence = 0

    def publish(self, value: T) -> int:
        with self._condition:
            self._value = value
            self._sequence += 1
            self._condition.notify_all()
            return self._sequence

    def get(self):
        with self._condition:
            return self._sequence, self._value

    def wait_after(self, sequence: int, timeout: float = 0.5):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence,
                timeout=max(0.0, timeout),
            )
            return self._sequence, self._value

    @property
    def depth(self) -> int:
        with self._condition:
            return 0 if self._value is None else 1


class RollingRate:
    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = max(0.1, float(window_seconds))
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, timestamp: float) -> None:
        with self._lock:
            self._timestamps.append(timestamp)
            self._trim(timestamp)

    def value(self, now: float) -> float:
        with self._lock:
            self._trim(now)
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while len(self._timestamps) > 1 and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
