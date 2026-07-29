from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import CameraIntrinsics, RgbdCalibration

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BRIDGE = PROJECT_DIR / "native/liborbbec_bridge.so"
DEFAULT_SDK_DIR = PROJECT_DIR / "third_party/orbbec_sdk"


class _BridgeCalibration(ctypes.Structure):
    _fields_ = [
        ("color_width", ctypes.c_int),
        ("color_height", ctypes.c_int),
        ("color_fx", ctypes.c_float),
        ("color_fy", ctypes.c_float),
        ("color_cx", ctypes.c_float),
        ("color_cy", ctypes.c_float),
        ("color_distortion", ctypes.c_float * 8),
        ("depth_width", ctypes.c_int),
        ("depth_height", ctypes.c_int),
        ("depth_fx", ctypes.c_float),
        ("depth_fy", ctypes.c_float),
        ("depth_cx", ctypes.c_float),
        ("depth_cy", ctypes.c_float),
        ("depth_distortion", ctypes.c_float * 8),
        ("depth_to_color_rotation", ctypes.c_float * 9),
        ("depth_to_color_translation_m", ctypes.c_float * 3),
    ]


class _BridgeFrameInfo(ctypes.Structure):
    _fields_ = [
        ("color_index", ctypes.c_uint64),
        ("depth_index", ctypes.c_uint64),
        ("color_timestamp_us", ctypes.c_uint64),
        ("depth_timestamp_us", ctypes.c_uint64),
        ("color_size", ctypes.c_uint32),
        ("depth_size", ctypes.c_uint32),
        ("depth_scale_m", ctypes.c_float),
    ]


class _BridgeDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("serial", ctypes.c_char * 128),
        ("firmware", ctypes.c_char * 64),
        ("connection", ctypes.c_char * 32),
        ("vid", ctypes.c_int),
        ("pid", ctypes.c_int),
    ]


class _BridgeColorControls(ctypes.Structure):
    _fields_ = [
        ("auto_exposure_supported", ctypes.c_int),
        ("exposure_supported", ctypes.c_int),
        ("gain_supported", ctypes.c_int),
        ("auto_exposure", ctypes.c_int),
        ("exposure_current", ctypes.c_int),
        ("exposure_min", ctypes.c_int),
        ("exposure_max", ctypes.c_int),
        ("exposure_step", ctypes.c_int),
        ("exposure_default", ctypes.c_int),
        ("gain_current", ctypes.c_int),
        ("gain_min", ctypes.c_int),
        ("gain_max", ctypes.c_int),
        ("gain_step", ctypes.c_int),
        ("gain_default", ctypes.c_int),
    ]


def _decode_c_string(value) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")


def _camera_intrinsics(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        distortion=tuple(float(value) for value in distortion),
    )


class HardwareMjpegDecoder:
    """Decode SDK MJPEG frames with Jetson nvjpegdec, with an explicit fallback."""

    FIRST_SAMPLE_TIMEOUT_NS = 1_000_000_000
    STEADY_SAMPLE_TIMEOUT_NS = 250_000_000

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._decoded_frames = 0
        self.backend = "opencv-imdecode"
        self.pipeline = None
        self.source = None
        self.sink = None
        self._gst = None
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            description = (
                "appsrc name=source is-live=true block=true do-timestamp=true "
                "format=time "
                "caps=image/jpeg,width=%d,height=%d,framerate=%d/1 ! "
                "jpegparse ! nvjpegdec ! "
                "video/x-raw(memory:NVMM),format=Y42B ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink max-buffers=1 drop=true sync=false"
            ) % (self.width, self.height, self.fps)
            pipeline = Gst.parse_launch(description)
            source = pipeline.get_by_name("source")
            sink = pipeline.get_by_name("sink")
            if source is None or sink is None:
                raise RuntimeError("GStreamer appsrc/appsink creation failed")
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer nvjpegdec pipeline failed to start")
            self._gst = Gst
            self.pipeline = pipeline
            self.source = source
            self.sink = sink
            self.backend = "nvjpegdec"
        except Exception:
            self.close()

    def decode(self, jpeg: memoryview) -> np.ndarray:
        if self.pipeline is None:
            data = np.frombuffer(jpeg, dtype=np.uint8)
            decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("OpenCV failed to decode Orbbec MJPEG frame")
            return decoded
        Gst = self._gst
        buffer = Gst.Buffer.new_allocate(None, len(jpeg), None)
        buffer.fill(0, bytes(jpeg))
        flow = self.source.emit("push-buffer", buffer)
        if flow != Gst.FlowReturn.OK:
            raise RuntimeError("nvjpegdec appsrc rejected MJPEG frame")
        timeout_ns = (
            self.FIRST_SAMPLE_TIMEOUT_NS
            if self._decoded_frames == 0
            else self.STEADY_SAMPLE_TIMEOUT_NS
        )
        sample = self.sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            bus = self.pipeline.get_bus()
            message = bus.pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if message is not None and message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                detail = ": %s" % debug if debug else ""
                raise RuntimeError(
                    "nvjpegdec GStreamer error: %s%s" % (error, detail)
                )
            if message is not None and message.type == Gst.MessageType.EOS:
                raise RuntimeError("nvjpegdec reached unexpected end of stream")
            raise RuntimeError(
                "timed out after %.0f ms waiting for nvjpegdec output"
                % (timeout_ns / 1_000_000.0)
            )
        output = sample.get_buffer()
        ok, mapping = output.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map nvjpegdec output")
        try:
            expected = self.width * self.height * 3
            pixels = np.frombuffer(mapping.data, dtype=np.uint8)
            if pixels.size < expected:
                raise RuntimeError(
                    "nvjpegdec returned %d bytes, expected %d"
                    % (pixels.size, expected)
                )
            decoded = (
                pixels[:expected].reshape(self.height, self.width, 3).copy()
            )
            self._decoded_frames += 1
            return decoded
        finally:
            output.unmap(mapping)

    def close(self) -> None:
        if self.pipeline is not None and self._gst is not None:
            try:
                self.source.emit("end-of-stream")
                self.pipeline.set_state(self._gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.source = None
        self.sink = None


class OrbbecSdkCapture:
    def __init__(
        self,
        color_width: int = 1280,
        color_height: int = 800,
        color_fps: int = 60,
        depth_width: int = 640,
        depth_height: int = 400,
        depth_fps: int = 30,
        enable_depth: bool = True,
        color_auto_exposure: bool = False,
        color_exposure: Optional[int] = None,
        color_exposure_scale: float = 0.30,
        color_gain: Optional[int] = None,
        bridge_path: Path = DEFAULT_BRIDGE,
    ) -> None:
        if not bridge_path.is_file():
            raise RuntimeError(
                "Orbbec bridge is missing: %s; run tools/setup_orbbec_sdk.sh"
                % bridge_path
            )
        sdk_dir = DEFAULT_SDK_DIR
        config_path = sdk_dir / "lib/OrbbecSDKConfig.xml"
        if config_path.is_file():
            os.environ.setdefault("OB_CONFIG_FILE_PATH", str(config_path))
        self.library = ctypes.CDLL(str(bridge_path))
        self._configure_api()
        self.enable_depth = bool(enable_depth)
        self.handle = self.library.ob_bridge_create(
            color_width,
            color_height,
            color_fps,
            depth_width if self.enable_depth else 0,
            depth_height if self.enable_depth else 0,
            depth_fps if self.enable_depth else 0,
        )
        if not self.handle:
            raise RuntimeError(self._last_error())
        calibration = _BridgeCalibration()
        if not self.library.ob_bridge_get_calibration(
            self.handle, ctypes.byref(calibration)
        ):
            self.close()
            raise RuntimeError(self._last_error())
        self.calibration = RgbdCalibration(
            color=_camera_intrinsics(
                calibration.color_width,
                calibration.color_height,
                calibration.color_fx,
                calibration.color_fy,
                calibration.color_cx,
                calibration.color_cy,
                calibration.color_distortion,
            ),
            depth=_camera_intrinsics(
                calibration.depth_width,
                calibration.depth_height,
                calibration.depth_fx,
                calibration.depth_fy,
                calibration.depth_cx,
                calibration.depth_cy,
                calibration.depth_distortion,
            ),
            depth_to_color_rotation=tuple(
                float(value) for value in calibration.depth_to_color_rotation
            ),
            depth_to_color_translation_m=tuple(
                float(value)
                for value in calibration.depth_to_color_translation_m
            ),
        )
        raw_device = _BridgeDeviceInfo()
        if not self.library.ob_bridge_get_device_info(
            self.handle, ctypes.byref(raw_device)
        ):
            self.close()
            raise RuntimeError(self._last_error())
        self.device_info = {
            "name": _decode_c_string(raw_device.name),
            "serial": _decode_c_string(raw_device.serial),
            "firmware": _decode_c_string(raw_device.firmware),
            "connection": _decode_c_string(raw_device.connection),
            "vid": int(raw_device.vid),
            "pid": int(raw_device.pid),
        }
        self.color_controls = self._apply_color_controls(
            auto_exposure=bool(color_auto_exposure),
            exposure=color_exposure,
            exposure_scale=float(color_exposure_scale),
            gain=color_gain,
        )
        if (
            self.calibration.color.width != color_width
            or self.calibration.color.height != color_height
            or (
                self.enable_depth
                and (
                    self.calibration.depth.width != depth_width
                    or self.calibration.depth.height != depth_height
                )
            )
        ):
            self.close()
            raise RuntimeError("Orbbec SDK returned unexpected stream dimensions")
        self.jpeg_buffer = np.empty(4 * 1024 * 1024, dtype=np.uint8)
        self.depth_buffer = np.empty(
            (
                depth_height if self.enable_depth else 1,
                depth_width if self.enable_depth else 1,
            ),
            dtype=np.uint16,
        )
        self.decoder = HardwareMjpegDecoder(
            color_width, color_height, color_fps
        )
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_id = 0
        self.latest_depth_timestamp = 0.0
        self.latest_depth_scale_m = 0.001
        self.decoder_backend = self.decoder.backend

    def _configure_api(self) -> None:
        self.library.ob_bridge_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.ob_bridge_create.restype = ctypes.c_void_p
        self.library.ob_bridge_destroy.argtypes = [ctypes.c_void_p]
        self.library.ob_bridge_get_calibration.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_BridgeCalibration),
        ]
        self.library.ob_bridge_get_calibration.restype = ctypes.c_int
        self.library.ob_bridge_get_device_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_BridgeDeviceInfo),
        ]
        self.library.ob_bridge_get_device_info.restype = ctypes.c_int
        self.library.ob_bridge_get_color_controls.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_BridgeColorControls),
        ]
        self.library.ob_bridge_get_color_controls.restype = ctypes.c_int
        self.library.ob_bridge_set_color_controls.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.ob_bridge_set_color_controls.restype = ctypes.c_int
        self.library.ob_bridge_wait.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_uint32,
            ctypes.POINTER(_BridgeFrameInfo),
            ctypes.c_uint32,
        ]
        self.library.ob_bridge_wait.restype = ctypes.c_int
        self.library.ob_bridge_last_error.argtypes = []
        self.library.ob_bridge_last_error.restype = ctypes.c_char_p

    @staticmethod
    def _clamp_to_range(value: int, minimum: int, maximum: int, step: int) -> int:
        clamped = min(int(maximum), max(int(minimum), int(value)))
        step = max(1, int(step))
        return int(minimum) + int(
            round((clamped - int(minimum)) / float(step))
        ) * step

    def _read_color_controls(self) -> dict:
        raw = _BridgeColorControls()
        if not self.library.ob_bridge_get_color_controls(
            self.handle, ctypes.byref(raw)
        ):
            raise RuntimeError(self._last_error())
        return {
            name: int(getattr(raw, name))
            for name, _ctype in raw._fields_
        }

    def _apply_color_controls(
        self,
        auto_exposure: bool,
        exposure: Optional[int],
        exposure_scale: float,
        gain: Optional[int],
    ) -> dict:
        if exposure_scale <= 0.0:
            raise ValueError("color_exposure_scale must be positive")
        controls = self._read_color_controls()
        auto_value = -1
        exposure_value = -1
        gain_value = -1
        if controls["auto_exposure_supported"]:
            auto_value = 1 if auto_exposure else 0
        if not auto_exposure and controls["exposure_supported"]:
            requested = (
                int(exposure)
                if exposure is not None
                else int(
                    round(
                        controls["exposure_default"]
                        * float(exposure_scale)
                    )
                )
            )
            exposure_value = self._clamp_to_range(
                requested,
                controls["exposure_min"],
                controls["exposure_max"],
                controls["exposure_step"],
            )
        if not auto_exposure and controls["gain_supported"]:
            requested_gain = (
                int(gain)
                if gain is not None
                else int(controls["gain_default"])
            )
            gain_value = self._clamp_to_range(
                requested_gain,
                controls["gain_min"],
                controls["gain_max"],
                controls["gain_step"],
            )
        if not self.library.ob_bridge_set_color_controls(
            self.handle,
            auto_value,
            exposure_value,
            gain_value,
        ):
            raise RuntimeError(self._last_error())
        return self._read_color_controls()

    def _last_error(self) -> str:
        raw = self.library.ob_bridge_last_error()
        return raw.decode("utf-8", "replace") if raw else "unknown Orbbec SDK error"

    def read(
        self, timeout_ms: int = 250
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        int,
        float,
        float,
        RgbdCalibration,
    ]:
        info = _BridgeFrameInfo()
        received = self.library.ob_bridge_wait(
            self.handle,
            self.jpeg_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.jpeg_buffer.nbytes,
            self.depth_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            self.depth_buffer.nbytes,
            ctypes.byref(info),
            int(timeout_ms),
        )
        if received < 0:
            raise RuntimeError(self._last_error())
        if received == 0:
            raise TimeoutError("Orbbec SDK frame timeout")
        received_monotonic = time.monotonic()
        jpeg = memoryview(self.jpeg_buffer)[: info.color_size]
        color = self.decoder.decode(jpeg)
        if info.depth_size:
            expected = self.calibration.depth.width * self.calibration.depth.height * 2
            if info.depth_size != expected:
                raise RuntimeError(
                    "Orbbec depth payload is %d bytes, expected %d"
                    % (info.depth_size, expected)
                )
            self.latest_depth = self.depth_buffer.copy()
            self.latest_depth_id = int(info.depth_index)
            hardware_age_us = max(
                0, int(info.color_timestamp_us) - int(info.depth_timestamp_us)
            )
            self.latest_depth_timestamp = (
                received_monotonic - hardware_age_us / 1_000_000.0
            )
            self.latest_depth_scale_m = float(info.depth_scale_m)
        return (
            color,
            self.latest_depth,
            self.latest_depth_id,
            self.latest_depth_timestamp,
            self.latest_depth_scale_m,
            self.calibration,
        )

    def close(self) -> None:
        decoder = getattr(self, "decoder", None)
        if decoder is not None:
            decoder.close()
        handle = getattr(self, "handle", None)
        if handle:
            self.library.ob_bridge_destroy(handle)
            self.handle = None
