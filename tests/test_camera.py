from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np

from ball_runtime.camera import (
    CameraWorker,
    discover_orbbec_rgb_device,
    orbbec_gstreamer_pipeline,
    resolve_device,
)
from ball_runtime.latest import LatestValue
from ball_runtime.orbbec_sdk import HardwareMjpegDecoder, OrbbecSdkCapture


class CameraPipelineTests(unittest.TestCase):
    def test_nvjpegdec_allows_extra_time_for_jetson_cold_start(self) -> None:
        self.assertEqual(
            HardwareMjpegDecoder.FIRST_SAMPLE_TIMEOUT_NS,
            1_000_000_000,
        )
        self.assertEqual(
            HardwareMjpegDecoder.STEADY_SAMPLE_TIMEOUT_NS,
            250_000_000,
        )

    def test_manual_exposure_is_clamped_to_device_step(self) -> None:
        self.assertEqual(
            OrbbecSdkCapture._clamp_to_range(73, 10, 100, 5),
            75,
        )
        self.assertEqual(
            OrbbecSdkCapture._clamp_to_range(1000, 10, 100, 5),
            100,
        )

    def test_336l_pipeline_is_latest_frame_60fps_hardware_decode(self) -> None:
        pipeline = orbbec_gstreamer_pipeline("/dev/video6", 1280, 800, 60)
        self.assertIn("image/jpeg,width=1280,height=800,framerate=60/1", pipeline)
        self.assertIn("nvjpegdec", pipeline)
        self.assertIn("max-buffers=1 drop=true sync=false", pipeline)

    def test_discovery_selects_mjpeg_orbbec_node_not_fixed_number(self) -> None:
        descriptors = {
            "/dev/video0": "orbbec gemini 336l depth camera",
            "/dev/video4": "orbbec gemini 336l rgb camera",
        }
        formats = {
            "/dev/video0": {"NV12", "Y8  "},
            "/dev/video4": {"YUYV", "MJPG"},
        }
        with mock.patch(
            "ball_runtime.camera.glob.glob",
            return_value=["/dev/video0", "/dev/video4"],
        ), mock.patch(
            "ball_runtime.camera._node_descriptor",
            side_effect=lambda path: descriptors[path],
        ), mock.patch(
            "ball_runtime.camera._v4l2_formats",
            side_effect=lambda path: formats[path],
        ):
            self.assertEqual(discover_orbbec_rgb_device(), "/dev/video4")

    def test_discovery_reports_connected_camera_with_missing_rgb_node(self) -> None:
        with mock.patch(
            "ball_runtime.camera.glob.glob",
            return_value=["/dev/video0", "/dev/video2"],
        ), mock.patch(
            "ball_runtime.camera._node_descriptor",
            return_value="orbbec gemini 336l depth camera",
        ), mock.patch(
            "ball_runtime.camera._v4l2_formats",
            side_effect=({"NV12"}, {"YV12"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "MJPEG RGB UVC"):
                discover_orbbec_rgb_device()

    def test_explicit_depth_node_is_rejected_before_pipeline_open(self) -> None:
        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "ball_runtime.camera._v4l2_formats",
            return_value={"NV12"},
        ):
            with self.assertRaisesRegex(RuntimeError, "not an MJPEG"):
                resolve_device("/dev/video0")

    def test_camera_worker_reconnects_after_read_failures(self) -> None:
        stop = threading.Event()

        class FailedCapture:
            def read(self):
                return False, None

            def release(self):
                pass

        class RecoveredCapture:
            def read(self):
                stop.set()
                return True, np.zeros((800, 1280, 3), dtype=np.uint8)

            def release(self):
                pass

        output = LatestValue()
        worker = CameraWorker(output, stop, reconnect_seconds=0.0)
        with mock.patch(
            "ball_runtime.camera.resolve_device",
            return_value="/dev/video6",
        ), mock.patch(
            "ball_runtime.camera.open_orbbec_capture",
            side_effect=(FailedCapture(), RecoveredCapture()),
        ):
            worker._run()
        _, packet = output.get()
        self.assertIsNotNone(packet)
        self.assertEqual(packet.frame_id, 1)
        self.assertEqual(worker.reconnects, 1)


if __name__ == "__main__":
    unittest.main()
