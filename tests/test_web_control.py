from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from ball_runtime.latest import LatestValue
from ball_runtime.types import (
    Detection,
    DetectionResult,
    TrackStatus,
    TubeContour,
)
from ball_runtime.web_control import (
    FrameRecorder,
    FrameJpegEncoder,
    LatestJpeg,
    NxControlClient,
    WebControlServer,
    load_recording_manifest,
    live_page,
)


class WebControlTests(unittest.TestCase):
    @staticmethod
    def _result(frame_id, image):
        return DetectionResult(
            frame_id=frame_id,
            captured_monotonic=time.monotonic(),
            completed_monotonic=time.monotonic(),
            inference_ms=1.0,
            detections=(Detection(8, 6, 24, 18, 0.9),),
            source_image=image,
            status=TrackStatus.MEASURED,
            tube_contour=TubeContour(
                valid=True,
                endpoint_negative_px=(4.0, 12.0),
                endpoint_positive_px=(28.0, 12.0),
            ),
            position_cm=0.0,
            position_projected_px=(16.0, 12.0),
            position_source="rgb-contour",
        )

    def test_production_services_use_rgb_contour_runtime(self) -> None:
        project = Path(__file__).resolve().parent.parent
        service_files = (
            project / "deploy/systemd/ball-vision-web.service.in",
            project / "nx_control/deploy/ball-vision.service",
        )
        for service_file in service_files:
            service = service_file.read_text(encoding="utf-8")
            self.assertIn("run_rgb.py", service, str(service_file))
            self.assertIn("--position-mode rgb-contour", service, str(service_file))
            self.assertNotIn("/run.py", service, str(service_file))
        installed_service = service_files[0].read_text(encoding="utf-8")
        self.assertIn("--web-record", installed_service)
        self.assertIn("--web-record-dir ${WEB_RECORD_DIR}", installed_service)

    def test_page_contains_preview_and_task_controls(self) -> None:
        page = live_page(37)
        self.assertIn("/latest.jpg", page)
        self.assertIn("/api/vision/task", page)
        self.assertIn('data-task="3"', page)
        self.assertIn('data-task="6"', page)
        self.assertIn("setTimeout(nextImage,37)", page)
        self.assertIn('href="/records"', page)
        self.assertIn("/api/recording/stop", page)

    def test_recorder_writes_manifest_and_keeps_frame_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ball-record-test-") as temporary:
            recorder = FrameRecorder(
                temporary,
                enabled=True,
                prefix="ball",
                test_name="test run",
                fps=20,
            )
            session_id = recorder.status()["session_id"]
            self.assertIsNotNone(session_id)
            recorder.write(b"jpeg-one", 32, 24)
            recorder.write(b"jpeg-two", 32, 24)
            with mock.patch.object(
                recorder,
                "_make_mp4_locked",
                side_effect=FileNotFoundError("gst-launch-1.0"),
            ):
                self.assertEqual(recorder.stop(), session_id)

            manifest = load_recording_manifest(temporary, str(session_id))
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["width"], 32)
            self.assertEqual(manifest["height"], 24)
            self.assertIsNone(manifest["mp4"])
            self.assertIn("MP4 conversion failed", manifest["error"])
            self.assertIsNone(load_recording_manifest(temporary, "../escape"))

    def test_invalid_task_and_target_are_rejected_locally(self) -> None:
        client = NxControlClient("/tmp/unused-ball-control.sock")
        invalid_task = client.set_task("7")
        self.assertFalse(invalid_task["ok"])
        self.assertEqual(invalid_task["error_code"], "INVALID_TASK")

        invalid_target = client.set_task("6", 10.1)
        self.assertFalse(invalid_target["ok"])
        self.assertEqual(invalid_target["error_code"], "TARGET_OUT_OF_RANGE")

        inactive_target = client.set_target(2.0)
        self.assertFalse(inactive_target["ok"])
        self.assertEqual(inactive_target["error_code"], "TASK6_NOT_ACTIVE")

    def test_encoder_reuses_detection_result_and_draws_preview_overlay(self) -> None:
        results = LatestValue()
        latest = LatestJpeg()
        stop_event = threading.Event()
        encoder = FrameJpegEncoder(
            results,
            latest,
            stop_event,
            quality=70,
            interval_ms=20,
            preview_width=16,
        )
        encoder.start()
        try:
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:, :, 1] = 180
            results.publish(self._result(42, image))
            jpeg, sequence, frame_id, error = latest.wait(2.0)
            self.assertEqual(error, "")
            self.assertIsNotNone(jpeg)
            self.assertEqual(jpeg[:2], b"\xff\xd8")
            self.assertEqual(jpeg[-2:], b"\xff\xd9")
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (12, 16))
            self.assertGreater(sequence, 0)
            self.assertEqual(frame_id, 42)
            results.publish(self._result(43, image))
            time.sleep(0.1)
            self.assertEqual(latest.status()["frame_sequence"], sequence)
        finally:
            stop_event.set()
            encoder.join(timeout=2.0)
        self.assertFalse(encoder.is_alive())

    def test_http_recording_history_and_mp4_range_playback(self) -> None:
        results = LatestValue()
        stop_event = threading.Event()
        with tempfile.TemporaryDirectory(prefix="ball-web-record-test-") as temporary:
            server = WebControlServer(
                results,
                stop_event,
                "127.0.0.1",
                0,
                os.path.join(temporary, "unused-control.sock"),
                interval_ms=10,
                preview_width=16,
                record=True,
                record_dir=temporary,
                record_prefix="ball",
                record_name="http",
                record_fps=20,
            )

            def make_test_mp4(session_dir, session_id, _frame_count):
                output = session_dir / (session_id + ".mp4")
                output.write_bytes(b"0123456789")
                return output

            server.recorder._make_mp4_locked = make_test_mp4
            server.start()
            host, port = server.address[:2]
            base = "http://%s:%d" % (host, port)
            try:
                image = np.full((24, 32, 3), 127, dtype=np.uint8)
                results.publish(self._result(7, image))
                deadline = time.monotonic() + 2.0
                recording = {}
                while time.monotonic() < deadline:
                    with urllib.request.urlopen(
                        base + "/api/recording/status", timeout=2.0
                    ) as response:
                        recording = json.loads(response.read().decode("utf-8"))
                    if recording["frame_count"] >= 1:
                        break
                    time.sleep(0.02)
                self.assertGreaterEqual(recording["frame_count"], 1)
                session_id = recording["session_id"]

                request = urllib.request.Request(
                    base + "/api/recording/stop", data=b"", method="POST"
                )
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    stopped = json.loads(response.read().decode("utf-8"))
                self.assertEqual(stopped["stopped_session"], session_id)

                with urllib.request.urlopen(base + "/records", timeout=2.0) as response:
                    records = response.read().decode("utf-8")
                self.assertIn(str(session_id), records)
                self.assertIn("/video?id=", records)

                video_request = urllib.request.Request(
                    base + "/record_video.mp4?id=" + str(session_id),
                    headers={"Range": "bytes=2-5"},
                )
                with urllib.request.urlopen(video_request, timeout=2.0) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
                    self.assertEqual(response.read(), b"2345")

                start_request = urllib.request.Request(
                    base + "/api/recording/start", data=b"", method="POST"
                )
                with urllib.request.urlopen(start_request, timeout=2.0) as response:
                    started = json.loads(response.read().decode("utf-8"))
                self.assertTrue(started["recording"])
                self.assertNotEqual(started["session_id"], session_id)
            finally:
                stop_event.set()
                server.stop()

    def test_http_image_and_task_request_share_the_runtime(self) -> None:
        results = LatestValue()
        stop_event = threading.Event()
        with tempfile.TemporaryDirectory(prefix="ball-web-test-") as temporary:
            control_path = os.path.join(temporary, "control.sock")
            fake_controller = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            fake_controller.bind(control_path)
            fake_controller.settimeout(2.0)
            controller_error = []

            def serve_controller() -> None:
                try:
                    for _ in range(2):
                        raw, peer = fake_controller.recvfrom(4096)
                        parts = raw.decode("utf-8").split()
                        request_id = parts[1]
                        task = parts[2] if parts[0] == "TASK" else "idle"
                        target = float(parts[3]) if len(parts) == 4 else None
                        response = {
                            "ok": True,
                            "request_id": request_id,
                            "requested_task": task,
                            "active_task": task,
                            "control_mode": "hold_target" if task == "6" else "idle",
                            "target_cm": target,
                            "running": task != "idle",
                            "applied": True,
                            "controller_online": True,
                            "controller_state": "hold_target" if task == "6" else "idle",
                            "ball_position_cm": 0.0,
                            "vision_ok": True,
                            "dmmc_ok": True,
                            "safety_latched": False,
                            "error_code": "",
                            "last_error": "",
                            "message": "ok",
                        }
                        fake_controller.sendto(json.dumps(response).encode("utf-8"), peer)
                except Exception as exc:  # pragma: no cover - surfaced below
                    controller_error.append(exc)

            controller_thread = threading.Thread(target=serve_controller, daemon=True)
            controller_thread.start()
            server = WebControlServer(
                results,
                stop_event,
                "127.0.0.1",
                0,
                control_path,
                jpeg_quality=70,
                interval_ms=25,
            )
            server.start()
            host, port = server.address[:2]
            base = "http://%s:%d" % (host, port)
            try:
                image = np.full((24, 32, 3), 127, dtype=np.uint8)
                results.publish(self._result(99, image))
                with urllib.request.urlopen(base + "/latest.jpg", timeout=2.0) as response:
                    jpeg = response.read()
                    self.assertEqual(response.headers["X-Camera-Frame-Id"], "99")
                self.assertEqual(jpeg[:2], b"\xff\xd8")

                with urllib.request.urlopen(base + "/api/vision/status", timeout=2.0) as response:
                    status = json.loads(response.read().decode("utf-8"))
                self.assertEqual(status["active_task"], "idle")

                request = urllib.request.Request(
                    base + "/api/vision/task",
                    data=json.dumps({"task": "6", "target_cm": -7.3}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    task = json.loads(response.read().decode("utf-8"))
                self.assertEqual(task["active_task"], "6")
                self.assertAlmostEqual(task["target_cm"], -7.3)
            finally:
                stop_event.set()
                server.stop()
                fake_controller.close()
                controller_thread.join(timeout=2.0)
            self.assertEqual(controller_error, [])


if __name__ == "__main__":
    unittest.main()
