from __future__ import annotations

import socket
import time
from typing import Tuple

from .latest import RollingRate


def parse_udp_endpoint(value: str) -> Tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("UDP endpoint must be HOST:PORT")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError("UDP port must be in 1..65535")
    return host, port


class DatagramWriter:
    """Non-blocking, best-effort localhost output for the C++ control process."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = parse_udp_endpoint(endpoint)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.sent_packets = 0
        self.sent_bytes = 0
        self.dropped_packets = 0
        self.rate = RollingRate(3.0)

    def submit(self, packet: bytes) -> None:
        try:
            written = self.socket.sendto(packet, self.endpoint)
            if written != len(packet):
                self.dropped_packets += 1
                return
            self.sent_packets += 1
            self.sent_bytes += written
            self.rate.tick(time.monotonic())
        except (BlockingIOError, InterruptedError, OSError):
            self.dropped_packets += 1

    def write_fps(self) -> float:
        return self.rate.value(time.monotonic())

    def close(self) -> None:
        self.socket.close()
