from __future__ import annotations

import struct
from typing import Tuple

from .types import DetectionResult, TrackStatus

MAGIC = b"\xA5\x5A"
PIXEL_VERSION = 1
TUBE_VERSION = 2
VERSION = PIXEL_VERSION
PAYLOAD_LENGTH = 11
PACKET_LENGTH = 18


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_packet(
    frame_id: int,
    status: int,
    x: int = 0,
    y: int = 0,
    confidence: int = 0,
) -> bytes:
    if int(status) not in set(int(item) for item in TrackStatus):
        raise ValueError("invalid tracking status")
    for name, value in (("x", x), ("y", y), ("confidence", confidence)):
        if not 0 <= int(value) <= 0xFFFF:
            raise ValueError("%s is outside uint16 range" % name)
    if int(status) == int(TrackStatus.LOST):
        x = y = confidence = 0
    payload = struct.pack(
        "<IBHHH",
        int(frame_id) & 0xFFFFFFFF,
        int(status),
        int(x),
        int(y),
        int(confidence),
    )
    body = struct.pack("<BH", PIXEL_VERSION, PAYLOAD_LENGTH) + payload
    return MAGIC + body + struct.pack("<H", crc16_ccitt_false(body))


def encode_tube_packet(
    frame_id: int,
    status: int,
    position_x100_cm: int = 0,
    ball_confidence: int = 0,
    tube_confidence: int = 0,
) -> bytes:
    if int(status) not in set(int(item) for item in TrackStatus):
        raise ValueError("invalid tracking status")
    if not -0x8000 <= int(position_x100_cm) <= 0x7FFF:
        raise ValueError("position_x100_cm is outside int16 range")
    for name, value in (
        ("ball_confidence", ball_confidence),
        ("tube_confidence", tube_confidence),
    ):
        if not 0 <= int(value) <= 1000:
            raise ValueError("%s must be in 0..1000" % name)
    if int(status) == int(TrackStatus.LOST):
        position_x100_cm = ball_confidence = tube_confidence = 0
    payload = struct.pack(
        "<IBhHH",
        int(frame_id) & 0xFFFFFFFF,
        int(status),
        int(position_x100_cm),
        int(ball_confidence),
        int(tube_confidence),
    )
    body = struct.pack("<BH", TUBE_VERSION, PAYLOAD_LENGTH) + payload
    return MAGIC + body + struct.pack("<H", crc16_ccitt_false(body))


def encode_result(
    result: DetectionResult,
    width: int = 1280,
    height: int = 800,
    protocol: str = "pixel-v1",
) -> bytes:
    if protocol == "tube-v2":
        detection = result.detection
        if (
            detection is None
            or not result.has_valid_position
            or result.position_cm is None
        ):
            return encode_tube_packet(result.frame_id, TrackStatus.LOST)
        position = min(
            0x7FFF,
            max(-0x8000, int(round(result.position_cm * 100.0))),
        )
        ball_confidence = min(
            1000, max(0, int(round(detection.confidence * 1000.0)))
        )
        tube_confidence = min(
            1000, max(0, int(round(result.tube_confidence * 1000.0)))
        )
        return encode_tube_packet(
            result.frame_id,
            result.status,
            position,
            ball_confidence,
            tube_confidence,
        )
    if protocol != "pixel-v1":
        raise ValueError("protocol must be pixel-v1 or tube-v2")
    detection = result.detection
    ball_status = result.effective_ball_status
    if detection is None or ball_status == TrackStatus.LOST:
        return encode_packet(result.frame_id, TrackStatus.LOST)
    center_x, center_y = detection.center
    x = min(width - 1, max(0, int(round(center_x))))
    y = min(height - 1, max(0, int(round(center_y))))
    confidence = min(1000, max(0, int(round(detection.confidence * 1000.0))))
    return encode_packet(result.frame_id, ball_status, x, y, confidence)


def decode_packet(packet: bytes) -> Tuple[int, TrackStatus, int, int, int]:
    if len(packet) != PACKET_LENGTH:
        raise ValueError("packet length must be %d" % PACKET_LENGTH)
    if packet[:2] != MAGIC:
        raise ValueError("invalid magic")
    version, payload_length = struct.unpack_from("<BH", packet, 2)
    if version not in (PIXEL_VERSION, TUBE_VERSION):
        raise ValueError("unsupported protocol version")
    if payload_length != PAYLOAD_LENGTH:
        raise ValueError("invalid payload length")
    body = packet[2:-2]
    expected_crc = struct.unpack_from("<H", packet, PACKET_LENGTH - 2)[0]
    if crc16_ccitt_false(body) != expected_crc:
        raise ValueError("CRC mismatch")
    if version == PIXEL_VERSION:
        frame_id, raw_status, value_a, value_b, value_c = struct.unpack_from(
            "<IBHHH", packet, 5
        )
    else:
        frame_id, raw_status, value_a, value_b, value_c = struct.unpack_from(
            "<IBhHH", packet, 5
        )
    try:
        status = TrackStatus(raw_status)
    except ValueError as exc:
        raise ValueError("invalid tracking status") from exc
    if status == TrackStatus.LOST and (value_a or value_b or value_c):
        raise ValueError("lost packet must contain zero values")
    return frame_id, status, value_a, value_b, value_c
