from __future__ import annotations

import errno
import glob
import os
import select
import sys
import termios
import threading
import time
from pathlib import Path
from typing import Optional

from .latest import RollingRate


def discover_serial_device() -> Optional[str]:
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def configure_serial(fd: int, baud: int) -> None:
    speed_name = "B%d" % baud
    if not hasattr(termios, speed_name):
        raise RuntimeError("unsupported baud rate: %d" % baud)
    speed = getattr(termios, speed_name)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = speed
    attrs[5] = speed
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


class SerialWriter:
    """Non-blocking serial writer with one pending packet and reconnect support."""

    def __init__(
        self,
        device: str = "auto",
        baud: int = 921600,
        reconnect_seconds: float = 1.0,
    ) -> None:
        self.requested_device = device
        self.baud = baud
        self.reconnect_seconds = reconnect_seconds
        self._condition = threading.Condition()
        self._pending: Optional[bytes] = None
        self._pending_sequence = 0
        self._stopping = False
        self.connected_device: Optional[str] = None
        self.sent_packets = 0
        self.sent_bytes = 0
        self.replaced_packets = 0
        self.last_write_ms = 0.0
        self.rate = RollingRate(3.0)
        self.last_error: Optional[str] = None
        self.thread = threading.Thread(target=self._run, name="serial-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, packet: bytes) -> None:
        with self._condition:
            if self._pending is not None:
                self.replaced_packets += 1
            self._pending = packet
            self._pending_sequence += 1
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self.thread.join(timeout=3.0)

    def _resolve(self) -> Optional[str]:
        if self.requested_device == "auto":
            return discover_serial_device()
        return self.requested_device if Path(self.requested_device).exists() else None

    def _open(self, device: str) -> int:
        fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            configure_serial(fd, self.baud)
        except Exception:
            os.close(fd)
            raise
        return fd

    def _next_packet(self, last_sequence: int):
        with self._condition:
            self._condition.wait_for(
                lambda: self._stopping or self._pending_sequence > last_sequence,
                timeout=0.5,
            )
            if self._stopping:
                return last_sequence, None
            if self._pending_sequence <= last_sequence:
                return last_sequence, None
            packet = self._pending
            sequence = self._pending_sequence
            self._pending = None
            return sequence, packet

    def _run(self) -> None:
        fd: Optional[int] = None
        last_sequence = 0
        while True:
            with self._condition:
                if self._stopping:
                    break
            if fd is None:
                device = self._resolve()
                if device is None:
                    self._sleep_or_stop()
                    continue
                try:
                    fd = self._open(device)
                    self.connected_device = device
                    self.last_error = None
                    print(
                        "serial connected: %s @ %d" % (device, self.baud),
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    self._sleep_or_stop()
                    continue

            sequence, packet = self._next_packet(last_sequence)
            if packet is None:
                continue
            last_sequence = sequence
            try:
                write_started = time.monotonic()
                self._write_all(fd, packet)
                completed_at = time.monotonic()
                self.last_write_ms = (completed_at - write_started) * 1000.0
                self.sent_packets += 1
                self.sent_bytes += len(packet)
                self.rate.tick(completed_at)
            except Exception as exc:
                self.last_error = str(exc)
                print("serial disconnected: %s" % exc, file=sys.stderr, flush=True)
                os.close(fd)
                fd = None
                self.connected_device = None

        if fd is not None:
            os.close(fd)
        self.connected_device = None

    def write_fps(self) -> float:
        return self.rate.value(time.monotonic())

    def _sleep_or_stop(self) -> None:
        with self._condition:
            self._condition.wait(timeout=self.reconnect_seconds)

    @staticmethod
    def _write_all(fd: int, packet: bytes) -> None:
        view = memoryview(packet)
        offset = 0
        deadline = time.monotonic() + 0.5
        while offset < len(view):
            if time.monotonic() >= deadline:
                raise TimeoutError("serial write timed out")
            _, writable, _ = select.select([], [fd], [], 0.05)
            if not writable:
                continue
            try:
                written = os.write(fd, view[offset:])
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                raise
            if written <= 0:
                raise OSError("serial write returned zero")
            offset += written
