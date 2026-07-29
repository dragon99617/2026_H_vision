from __future__ import annotations

import unittest

from ball_runtime.serial_protocol import (
    PACKET_LENGTH,
    crc16_ccitt_false,
    decode_packet,
    encode_packet,
    encode_result,
    encode_tube_packet,
)
from ball_runtime.types import Detection, DetectionResult, TrackStatus, TubePose


class SerialProtocolTests(unittest.TestCase):
    def test_crc_ccitt_false_standard_vector(self) -> None:
        self.assertEqual(crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_fixed_measured_packet(self) -> None:
        packet = encode_packet(0x12345678, TrackStatus.MEASURED, 640, 400, 876)
        self.assertEqual(len(packet), PACKET_LENGTH)
        self.assertEqual(
            decode_packet(packet),
            (0x12345678, TrackStatus.MEASURED, 640, 400, 876),
        )

    def test_lost_packet_forces_values_to_zero(self) -> None:
        packet = encode_packet(42, TrackStatus.LOST, 10, 20, 500)
        self.assertEqual(
            decode_packet(packet),
            (42, TrackStatus.LOST, 0, 0, 0),
        )

    def test_result_rounds_clamps_and_scales_confidence(self) -> None:
        result = DetectionResult(
            frame_id=9,
            captured_monotonic=1.0,
            completed_monotonic=1.1,
            inference_ms=5.0,
            detections=(Detection(1250, 770, 1350, 850, 0.8764),),
            status=TrackStatus.PREDICTED,
        )
        self.assertEqual(
            decode_packet(encode_result(result)),
            (9, TrackStatus.PREDICTED, 1279, 799, 876),
        )

    def test_bad_crc_is_rejected(self) -> None:
        packet = bytearray(
            encode_packet(1, TrackStatus.MEASURED, 2, 3, 400)
        )
        packet[7] ^= 0x01
        with self.assertRaisesRegex(ValueError, "CRC"):
            decode_packet(bytes(packet))

    def test_invalid_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "length"):
            decode_packet(b"\xA5\x5A")

    def test_v2_signed_centimetres_packet(self) -> None:
        packet = encode_tube_packet(
            0xABCDEF01,
            TrackStatus.MEASURED,
            -625,
            876,
            943,
        )
        self.assertEqual(len(packet), PACKET_LENGTH)
        self.assertEqual(packet[2], 2)
        self.assertEqual(
            decode_packet(packet),
            (0xABCDEF01, TrackStatus.MEASURED, -625, 876, 943),
        )

    def test_v2_invalid_result_is_all_zero(self) -> None:
        result = DetectionResult(
            frame_id=11,
            captured_monotonic=1.0,
            completed_monotonic=1.1,
            inference_ms=5.0,
            detections=(Detection(10, 20, 30, 40, 0.9),),
            status=TrackStatus.MEASURED,
        )
        self.assertEqual(
            decode_packet(encode_result(result, protocol="tube-v2")),
            (11, TrackStatus.LOST, 0, 0, 0),
        )

    def test_v2_result_scales_position_and_both_confidences(self) -> None:
        pose = TubePose(
            valid=True,
            captured_monotonic=1.0,
            center_m=(0.0, 0.0, 0.5),
            direction=(1.0, 0.0, 0.0),
        )
        result = DetectionResult(
            frame_id=12,
            captured_monotonic=1.0,
            completed_monotonic=1.1,
            inference_ms=5.0,
            detections=(Detection(10, 20, 30, 40, 0.8764),),
            status=TrackStatus.PREDICTED,
            tube_pose=pose,
            tube_confidence=0.9434,
            position_cm=-6.254,
        )
        self.assertEqual(
            decode_packet(encode_result(result, protocol="tube-v2")),
            (12, TrackStatus.PREDICTED, -625, 876, 943),
        )

    def test_v2_accepts_valid_rgb_contour_position_without_depth_pose(self) -> None:
        result = DetectionResult(
            frame_id=13,
            captured_monotonic=1.0,
            completed_monotonic=1.1,
            inference_ms=5.0,
            detections=(Detection(10, 20, 30, 40, 0.91),),
            status=TrackStatus.MEASURED,
            tube_confidence=0.82,
            position_cm=3.125,
            position_source="rgb-contour",
        )
        self.assertTrue(result.has_valid_position)
        self.assertEqual(
            decode_packet(encode_result(result, protocol="tube-v2")),
            (13, TrackStatus.MEASURED, 312, 910, 820),
        )


if __name__ == "__main__":
    unittest.main()
