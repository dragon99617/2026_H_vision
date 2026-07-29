from __future__ import annotations

import fcntl
import glob
import os
import struct
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

import cv2

from .latest import LatestValue, RollingRate
from .types import FramePacket

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
VIDIOC_ENUM_FMT = 0xC0405602
MJPEG_FORMATS = {"MJPG", "JPEG"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _node_descriptor(raw_path: str) -> str:
    name = Path(raw_path).name
    base = Path("/sys/class/video4linux") / name
    return " ".join(
        (
            _read_text(base / "name"),
            _read_text(base / "device/interface"),
        )
    ).lower()


def _v4l2_formats(raw_path: str) -> Set[str]:
    """Return capture pixel formats without starting or changing the stream."""
    formats: Set[str] = set()
    fd = None
    try:
        fd = os.open(raw_path, os.O_RDONLY | os.O_NONBLOCK)
        for index in range(64):
            descriptor = bytearray(64)
            struct.pack_into(
                "=II",
                descriptor,
                0,
                index,
                V4L2_BUF_TYPE_VIDEO_CAPTURE,
            )
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, descriptor, True)
            except OSError:
                break
            value = struct.unpack_from("=I", descriptor, 44)[0]
            formats.add(struct.pack("<I", value).decode("latin1"))
    except OSError:
        return set()
    finally:
        if fd is not None:
            os.close(fd)
    return formats


def _format_summary(nodes: List[tuple]) -> str:
    return ", ".join(
        "%s:%s" % (path, "/".join(sorted(formats)) or "no-capture-formats")
        for path, formats in nodes
    )


def discover_orbbec_rgb_device() -> str:
    candidates = []
    orbbec_nodes = []
    for raw_path in sorted(glob.glob("/dev/video*")):
        name = Path(raw_path).name
        base = Path("/sys/class/video4linux") / name
        descriptor = _node_descriptor(raw_path)
        if "orbbec" not in descriptor:
            continue
        formats = _v4l2_formats(raw_path)
        orbbec_nodes.append((raw_path, formats))
        if not formats.intersection(MJPEG_FORMATS):
            continue
        index_text = _read_text(base / "index")
        index = int(index_text) if index_text.isdigit() else 999
        rgb_rank = 0 if "rgb" in descriptor else 1
        candidates.append((rgb_rank, index, raw_path))

    if candidates:
        candidates.sort()
        return candidates[0][2]
    if orbbec_nodes:
        raise RuntimeError(
            "Gemini 336L is connected, but its MJPEG RGB UVC capture node is "
            "not registered; found %s. Reconnect the USB 3 cable or reboot, "
            "then verify that an additional RGB /dev/video node appears."
            % _format_summary(orbbec_nodes)
        )
    raise RuntimeError(
        "No Orbbec V4L2 device found. Connect the Gemini 336L or pass --device."
    )


def resolve_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        if not Path(requested).exists():
            raise RuntimeError("Camera device does not exist: %s" % requested)
        formats = _v4l2_formats(requested)
        if not formats.intersection(MJPEG_FORMATS):
            raise RuntimeError(
                "Camera %s is not an MJPEG capture node (formats: %s)"
                % (requested, "/".join(sorted(formats)) or "none")
            )
        return requested
    return discover_orbbec_rgb_device()


def orbbec_gstreamer_pipeline(
    device: str,
    width: int = 1280,
    height: int = 800,
    fps: int = 60,
) -> str:
    return (
        "v4l2src device=%s io-mode=mmap do-timestamp=true ! "
        "image/jpeg,width=%d,height=%d,framerate=%d/1 ! "
        "jpegparse ! nvjpegdec ! "
        "video/x-raw(memory:NVMM),format=Y42B ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    ) % (device, width, height, fps)


def open_orbbec_capture(
    device: str,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
    pipeline = orbbec_gstreamer_pipeline(device, width, height, fps)
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "Failed to open Orbbec pipeline for %s at %dx%d@%d"
            % (device, width, height, fps)
        )
    return capture


class CameraWorker:
    def __init__(
        self,
        output: LatestValue[FramePacket],
        stop_event: threading.Event,
        device: str = "auto",
        width: int = 1280,
        height: int = 800,
        fps: int = 60,
        backend: str = "v4l2",
        depth_width: int = 640,
        depth_height: int = 400,
        depth_fps: int = 30,
        enable_depth: bool = True,
        color_auto_exposure: bool = False,
        color_exposure: Optional[int] = None,
        color_exposure_scale: float = 0.30,
        color_gain: Optional[int] = None,
        reconnect_seconds: float = 1.0,
    ) -> None:
        self.output = output
        self.stop_event = stop_event
        self.requested_device = device
        self.width = width
        self.height = height
        self.fps = fps
        if backend not in ("sdk", "v4l2"):
            raise ValueError("camera backend must be sdk or v4l2")
        self.backend = backend
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.enable_depth = bool(enable_depth)
        self.color_auto_exposure = bool(color_auto_exposure)
        self.color_exposure = color_exposure
        self.color_exposure_scale = float(color_exposure_scale)
        self.color_gain = color_gain
        self.color_controls = {}
        self.reconnect_seconds = reconnect_seconds
        self.rate = RollingRate(2.0)
        self.depth_rate = RollingRate(3.0)
        self.frame_count = 0
        self.depth_frame_count = 0
        self.last_depth_frame_id = 0
        self.read_failures = 0
        self.reconnects = 0
        self.device: Optional[str] = None
        self.last_error: Optional[str] = None
        self.thread = threading.Thread(target=self._run, name="orbbec-capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 3.0) -> None:
        self.thread.join(timeout=timeout)

    def capture_fps(self) -> float:
        return self.rate.value(time.monotonic())

    def depth_fps_value(self) -> float:
        return self.depth_rate.value(time.monotonic())

    def color_controls_text(self) -> str:
        if not self.color_controls:
            return "exposure unavailable"
        return "AE=%s exp=%s gain=%s" % (
            "on" if self.color_controls.get("auto_exposure") else "off",
            self.color_controls.get("exposure_current", "?"),
            self.color_controls.get("gain_current", "?"),
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            capture = None
            try:
                if self.backend == "sdk":
                    self._run_sdk()
                    continue
                self.device = resolve_device(self.requested_device)
                capture = open_orbbec_capture(
                    self.device, self.width, self.height, self.fps
                )
                self.last_error = None
                print(
                    "camera opened: %s %dx%d@%d MJPG -> nvjpegdec"
                    % (self.device, self.width, self.height, self.fps),
                    file=sys.stderr,
                    flush=True,
                )
                consecutive_failures = 0
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    now = time.monotonic()
                    if not ok or frame is None:
                        self.read_failures += 1
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            raise RuntimeError("five consecutive camera read failures")
                        continue
                    consecutive_failures = 0
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        raise RuntimeError(
                            "camera returned %dx%d, expected %dx%d"
                            % (
                                frame.shape[1],
                                frame.shape[0],
                                self.width,
                                self.height,
                            )
                        )
                    self.frame_count += 1
                    self.rate.tick(now)
                    self.output.publish(
                        FramePacket(
                            frame_id=self.frame_count,
                            captured_monotonic=now,
                            image=frame,
                            camera_backend="v4l2-nvjpegdec",
                        )
                    )
            except Exception as exc:
                self.last_error = str(exc)
                self.reconnects += 1
                print(
                    "camera unavailable, retrying: %s" % exc,
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if capture is not None:
                    capture.release()
            self.stop_event.wait(self.reconnect_seconds)

    def _run_sdk(self) -> None:
        from .orbbec_sdk import OrbbecSdkCapture

        capture = None
        try:
            capture = OrbbecSdkCapture(
                color_width=self.width,
                color_height=self.height,
                color_fps=self.fps,
                depth_width=self.depth_width,
                depth_height=self.depth_height,
                depth_fps=self.depth_fps,
                enable_depth=self.enable_depth,
                color_auto_exposure=self.color_auto_exposure,
                color_exposure=self.color_exposure,
                color_exposure_scale=self.color_exposure_scale,
                color_gain=self.color_gain,
            )
            self.color_controls = dict(capture.color_controls)
            self.device = "OrbbecSDK"
            self.last_error = None
            print(
                "camera opened: OrbbecSDK RGB %dx%d@%d MJPG -> %s, "
                "depth %s, %s"
                % (
                    self.width,
                    self.height,
                    self.fps,
                    capture.decoder_backend,
                    (
                        "%dx%d@%d Y16"
                        % (
                            self.depth_width,
                            self.depth_height,
                            self.depth_fps,
                        )
                        if self.enable_depth
                        else "off"
                    ),
                    self.color_controls_text(),
                ),
                file=sys.stderr,
                flush=True,
            )
            consecutive_timeouts = 0
            while not self.stop_event.is_set():
                try:
                    (
                        frame,
                        depth,
                        depth_frame_id,
                        depth_timestamp,
                        depth_scale_m,
                        calibration,
                    ) = capture.read()
                except TimeoutError:
                    consecutive_timeouts += 1
                    self.read_failures += 1
                    if consecutive_timeouts >= 8:
                        raise RuntimeError("eight consecutive Orbbec SDK timeouts")
                    continue
                consecutive_timeouts = 0
                now = time.monotonic()
                if frame.shape[:2] != (self.height, self.width):
                    raise RuntimeError(
                        "SDK color returned %dx%d, expected %dx%d"
                        % (
                            frame.shape[1],
                            frame.shape[0],
                            self.width,
                            self.height,
                        )
                    )
                if (
                    depth_frame_id > 0
                    and depth_frame_id != self.last_depth_frame_id
                ):
                    self.last_depth_frame_id = depth_frame_id
                    self.depth_frame_count += 1
                    self.depth_rate.tick(now)
                self.frame_count += 1
                self.rate.tick(now)
                self.output.publish(
                    FramePacket(
                        frame_id=self.frame_count,
                        captured_monotonic=now,
                        image=frame,
                        depth=depth,
                        depth_frame_id=depth_frame_id,
                        depth_captured_monotonic=depth_timestamp,
                        depth_scale_m=depth_scale_m,
                        calibration=calibration,
                        camera_backend="orbbec-sdk-%s"
                        % capture.decoder_backend,
                    )
                )
        finally:
            if capture is not None:
                capture.close()
