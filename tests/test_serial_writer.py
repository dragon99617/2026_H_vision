from __future__ import annotations

import os
import pty
import select
import tempfile
import time
import unittest
from pathlib import Path

from ball_runtime.serial_protocol import decode_packet, encode_packet
from ball_runtime.serial_writer import SerialWriter
from ball_runtime.types import TrackStatus


class SerialWriterTests(unittest.TestCase):
    def test_writes_packet_to_pseudo_terminal(self) -> None:
        master_fd, slave_fd = pty.openpty()
        slave_path = os.ttyname(slave_fd)
        writer = SerialWriter(slave_path, baud=921600, reconnect_seconds=0.02)
        packet = encode_packet(55, TrackStatus.MEASURED, 100, 200, 900)
        try:
            writer.start()
            deadline = time.monotonic() + 2.0
            while writer.connected_device is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(writer.connected_device, slave_path)
            writer.submit(packet)
            received = bytearray()
            while len(received) < len(packet) and time.monotonic() < deadline:
                readable, _, _ = select.select([master_fd], [], [], 0.1)
                if readable:
                    received.extend(os.read(master_fd, 4096))
            self.assertEqual(bytes(received), packet)
            self.assertEqual(
                decode_packet(bytes(received)),
                (55, TrackStatus.MEASURED, 100, 200, 900),
            )
        finally:
            writer.stop()
            os.close(master_fd)
            os.close(slave_fd)

    def test_retries_until_explicit_device_appears(self) -> None:
        master_fd, slave_fd = pty.openpty()
        slave_path = os.ttyname(slave_fd)
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "ball-serial"
            writer = SerialWriter(
                str(requested),
                baud=921600,
                reconnect_seconds=0.01,
            )
            try:
                writer.start()
                time.sleep(0.05)
                self.assertIsNone(writer.connected_device)
                requested.symlink_to(slave_path)
                deadline = time.monotonic() + 2.0
                while (
                    writer.connected_device is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertEqual(writer.connected_device, str(requested))
                packet = encode_packet(
                    77,
                    TrackStatus.PREDICTED,
                    12,
                    34,
                    500,
                )
                writer.submit(packet)
                readable, _, _ = select.select([master_fd], [], [], 1.0)
                self.assertTrue(readable)
                self.assertEqual(os.read(master_fd, len(packet)), packet)
            finally:
                writer.stop()
        os.close(master_fd)
        os.close(slave_fd)


if __name__ == "__main__":
    unittest.main()
