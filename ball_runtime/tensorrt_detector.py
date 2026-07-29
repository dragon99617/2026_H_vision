from __future__ import annotations

import ctypes
import ctypes.util
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

if "bool" not in np.__dict__:
    np.bool = bool

import tensorrt as trt

from .native_cuda import NativeCudaLibrary
from .types import Detection, DetectionResult, FramePacket, TrackStatus


class CudaRuntime:
    def __init__(self) -> None:
        library_name = ctypes.util.find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(library_name)
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMallocHost.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMallocHost.restype = ctypes.c_int
        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cudaFreeHost.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemsetAsync.restype = ctypes.c_int
        self._configure_graph_api()

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)),
            "cudaMalloc",
        )
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaFree(pointer), "cudaFree")

    def malloc_host(self, shape, dtype):
        dtype = np.dtype(dtype)
        count = int(np.prod(shape))
        size = count * dtype.itemsize
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.cudaMallocHost(ctypes.byref(pointer), ctypes.c_size_t(size)),
            "cudaMallocHost",
        )
        buffer_type = ctypes.c_byte * size
        buffer = buffer_type.from_address(pointer.value)
        array = np.frombuffer(buffer, dtype=dtype, count=count).reshape(shape)
        return pointer, buffer, array

    def free_host(self, pointer: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaFreeHost(pointer), "cudaFreeHost")

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def stream_destroy(self, stream: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def stream_synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def memcpy_htod_async(
        self, destination: ctypes.c_void_p, source: np.ndarray, stream: ctypes.c_void_p
    ) -> None:
        self._copy(
            destination,
            source.ctypes.data_as(ctypes.c_void_p),
            source.nbytes,
            1,
            stream,
        )

    def memcpy_dtoh_async(
        self, destination: np.ndarray, source: ctypes.c_void_p, stream: ctypes.c_void_p
    ) -> None:
        self._copy(
            destination.ctypes.data_as(ctypes.c_void_p),
            source,
            destination.nbytes,
            2,
            stream,
        )

    def memset_async(
        self,
        destination: ctypes.c_void_p,
        value: int,
        size: int,
        stream: ctypes.c_void_p,
    ) -> None:
        self._check(
            self.lib.cudaMemsetAsync(
                destination,
                ctypes.c_int(value),
                ctypes.c_size_t(size),
                stream,
            ),
            "cudaMemsetAsync",
        )

    def _copy(self, destination, source, size: int, kind: int, stream) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                destination,
                source,
                ctypes.c_size_t(size),
                ctypes.c_int(kind),
                stream,
            ),
            "cudaMemcpyAsync",
        )

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError("%s failed with CUDA status %d" % (operation, status))

    def _configure_graph_api(self) -> None:
        self.graph_available = all(
            hasattr(self.lib, name)
            for name in (
                "cudaStreamBeginCapture",
                "cudaStreamEndCapture",
                "cudaGraphInstantiate",
                "cudaGraphLaunch",
                "cudaGraphExecDestroy",
                "cudaGraphDestroy",
            )
        )
        if not self.graph_available:
            return
        self.lib.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.cudaStreamBeginCapture.restype = ctypes.c_int
        self.lib.cudaStreamEndCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cudaStreamEndCapture.restype = ctypes.c_int
        self.lib.cudaGraphInstantiate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.cudaGraphInstantiate.restype = ctypes.c_int
        self.lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaGraphLaunch.restype = ctypes.c_int
        self.lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphExecDestroy.restype = ctypes.c_int
        self.lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphDestroy.restype = ctypes.c_int

    def graph_begin_capture(self, stream) -> None:
        self._check(
            self.lib.cudaStreamBeginCapture(stream, ctypes.c_int(0)),
            "cudaStreamBeginCapture",
        )

    def graph_end_capture(self, stream):
        graph = ctypes.c_void_p()
        self._check(
            self.lib.cudaStreamEndCapture(stream, ctypes.byref(graph)),
            "cudaStreamEndCapture",
        )
        executable = ctypes.c_void_p()
        error_node = ctypes.c_void_p()
        self._check(
            self.lib.cudaGraphInstantiate(
                ctypes.byref(executable),
                graph,
                ctypes.byref(error_node),
                None,
                ctypes.c_size_t(0),
            ),
            "cudaGraphInstantiate",
        )
        return graph, executable

    def graph_launch(self, executable, stream) -> None:
        self._check(self.lib.cudaGraphLaunch(executable, stream), "cudaGraphLaunch")

    def graph_destroy(self, graph, executable) -> None:
        if executable:
            self.lib.cudaGraphExecDestroy(executable)
        if graph:
            self.lib.cudaGraphDestroy(graph)


class TensorRTDetector:
    def __init__(self, engine_path: str, confidence: float = 0.25) -> None:
        path = Path(engine_path)
        if not path.is_file():
            raise FileNotFoundError(
                "TensorRT engine not found: %s. Run tools/build_engine.sh first." % path
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = float(confidence)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine: %s" % path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_index = self._find_input()
        self.output_indices = [
            index
            for index in range(self.engine.num_bindings)
            if not self.engine.binding_is_input(index)
        ]
        if len(self.output_indices) != 1:
            raise RuntimeError(
                "Expected one output binding, found %d" % len(self.output_indices)
            )
        self.output_index = self.output_indices[0]
        self._validate_bindings()
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        self.cuda = CudaRuntime()
        self.stream = self.cuda.stream_create()
        self.bindings = [0] * self.engine.num_bindings
        self.host_buffers = {}
        self.host_pointers = {}
        self.host_backing = {}
        self.device_buffers = {}
        for index in range(self.engine.num_bindings):
            shape = tuple(int(value) for value in self.engine.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            host_pointer, host_backing, host = self.cuda.malloc_host(shape, dtype)
            device = self.cuda.malloc(host.nbytes)
            self.host_buffers[index] = host
            self.host_pointers[index] = host_pointer
            self.host_backing[index] = host_backing
            self.device_buffers[index] = device
            self.bindings[index] = int(device.value)
        self.native_cuda: Optional[NativeCudaLibrary] = None
        self.image_host_pointer = None
        self.image_host_backing = None
        self.image_host = None
        self.image_device = None
        self.image_shape = None
        self.graph = None
        self.graph_executable = None
        self.native_runs = 0
        self.fast_path = "cpu-pinned"
        self.last_preprocess_ms = 0.0
        self.last_gpu_ms = 0.0
        if self.host_buffers[self.input_index].dtype == np.float32:
            try:
                self.native_cuda = NativeCudaLibrary()
                self.fast_path = "cuda-preprocess"
            except Exception:
                self.native_cuda = None
        self.closed = False

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return tuple(int(v) for v in self.engine.get_binding_shape(self.input_index))

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return tuple(int(v) for v in self.engine.get_binding_shape(self.output_index))

    def detect(self, packet: FramePacket) -> DetectionResult:
        started = time.monotonic()
        frame_height, frame_width = packet.image.shape[:2]
        scale = min(
            self.input_width / frame_width,
            self.input_height / frame_height,
        )
        resized_width = int(round(frame_width * scale))
        resized_height = int(round(frame_height * scale))
        pad_x = float((self.input_width - resized_width) // 2)
        pad_y = float((self.input_height - resized_height) // 2)

        preprocess_started = time.monotonic()
        if self.native_cuda is not None:
            self._prepare_native_image(packet.image)
            np.copyto(self.image_host, packet.image)
            self.last_preprocess_ms = (
                time.monotonic() - preprocess_started
            ) * 1000.0
            gpu_started = time.monotonic()
            if self.graph_executable is not None:
                self.cuda.graph_launch(self.graph_executable, self.stream)
            else:
                self._enqueue_native_sequence(frame_width, frame_height)
            self.cuda.stream_synchronize(self.stream)
            self.last_gpu_ms = (time.monotonic() - gpu_started) * 1000.0
            self.native_runs += 1
            if (
                self.native_runs == 1
                and self.cuda.graph_available
                and self.graph_executable is None
            ):
                self._try_build_graph(frame_width, frame_height)
        else:
            blob, _, _, _ = preprocess(
                packet.image,
                input_width=self.input_width,
                input_height=self.input_height,
                dtype=self.host_buffers[self.input_index].dtype,
            )
            np.copyto(self.host_buffers[self.input_index], blob)
            self.last_preprocess_ms = (
                time.monotonic() - preprocess_started
            ) * 1000.0
            gpu_started = time.monotonic()
            self._enqueue_cpu_sequence()
            self.cuda.stream_synchronize(self.stream)
            self.last_gpu_ms = (time.monotonic() - gpu_started) * 1000.0
        postprocess_started = time.monotonic()
        detections = parse_end2end_output(
            self.host_buffers[self.output_index],
            frame_width=packet.image.shape[1],
            frame_height=packet.image.shape[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence=self.confidence,
        )
        completed = time.monotonic()
        postprocess_ms = (completed - postprocess_started) * 1000.0
        return DetectionResult(
            frame_id=packet.frame_id,
            captured_monotonic=packet.captured_monotonic,
            completed_monotonic=completed,
            inference_ms=(completed - started) * 1000.0,
            detections=tuple(detections),
            source_image=packet.image,
            status=TrackStatus.MEASURED if detections else TrackStatus.LOST,
            preprocess_ms=self.last_preprocess_ms,
            gpu_ms=self.last_gpu_ms,
            postprocess_ms=postprocess_ms,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.graph is not None or self.graph_executable is not None:
            self.cuda.graph_destroy(self.graph, self.graph_executable)
            self.graph = None
            self.graph_executable = None
        if self.image_device is not None:
            self.cuda.free(self.image_device)
            self.image_device = None
        if self.image_host_pointer is not None:
            self.cuda.free_host(self.image_host_pointer)
            self.image_host_pointer = None
        for pointer in self.device_buffers.values():
            self.cuda.free(pointer)
        self.device_buffers.clear()
        for pointer in self.host_pointers.values():
            self.cuda.free_host(pointer)
        self.host_pointers.clear()
        self.host_backing.clear()
        self.host_buffers.clear()
        if self.stream:
            self.cuda.stream_destroy(self.stream)
            self.stream = None

    def _prepare_native_image(self, image: np.ndarray) -> None:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CUDA preprocessing requires uint8 HxWx3 BGR input")
        shape = tuple(int(value) for value in image.shape)
        if self.image_shape == shape:
            return
        if self.image_shape is not None:
            raise RuntimeError(
                "CUDA graph input shape changed from %s to %s" % (self.image_shape, shape)
            )
        pointer, backing, host = self.cuda.malloc_host(shape, np.uint8)
        self.image_host_pointer = pointer
        self.image_host_backing = backing
        self.image_host = host
        self.image_device = self.cuda.malloc(host.nbytes)
        self.image_shape = shape

    def _enqueue_native_sequence(self, frame_width: int, frame_height: int) -> None:
        self.cuda.memcpy_htod_async(self.image_device, self.image_host, self.stream)
        self.native_cuda.launch_preprocess(
            self.image_device,
            frame_width,
            frame_height,
            frame_width * 3,
            self.device_buffers[self.input_index],
            self.input_width,
            self.input_height,
            self.stream,
        )
        self._enqueue_engine_and_output()

    def _enqueue_cpu_sequence(self) -> None:
        self.cuda.memcpy_htod_async(
            self.device_buffers[self.input_index],
            self.host_buffers[self.input_index],
            self.stream,
        )
        self._enqueue_engine_and_output()

    def _enqueue_engine_and_output(self) -> None:
        ok = self.context.execute_async_v2(self.bindings, int(self.stream.value))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")
        self.cuda.memcpy_dtoh_async(
            self.host_buffers[self.output_index],
            self.device_buffers[self.output_index],
            self.stream,
        )

    def _try_build_graph(self, frame_width: int, frame_height: int) -> None:
        try:
            self.cuda.graph_begin_capture(self.stream)
            self._enqueue_native_sequence(frame_width, frame_height)
            self.graph, self.graph_executable = self.cuda.graph_end_capture(self.stream)
            self.fast_path = "cuda-preprocess+graph"
        except Exception:
            # A failed stream capture invalidates that stream.  A fresh stream
            # keeps the reliable non-graph CUDA path available.
            try:
                self.cuda.stream_destroy(self.stream)
            except Exception:
                pass
            self.stream = self.cuda.stream_create()
            self.fast_path = "cuda-preprocess"
            self.graph = None
            self.graph_executable = None

    def _find_input(self) -> int:
        inputs = [
            index
            for index in range(self.engine.num_bindings)
            if self.engine.binding_is_input(index)
        ]
        if len(inputs) != 1:
            raise RuntimeError("Expected one input binding, found %d" % len(inputs))
        return inputs[0]

    def _validate_bindings(self) -> None:
        input_shape = tuple(
            int(value) for value in self.engine.get_binding_shape(self.input_index)
        )
        output_shape = tuple(
            int(value) for value in self.engine.get_binding_shape(self.output_index)
        )
        if (
            len(input_shape) != 4
            or input_shape[0] != 1
            or input_shape[1] != 3
            or input_shape[2] <= 0
            or input_shape[3] <= 0
        ):
            raise RuntimeError(
                "Unexpected engine input shape %s; expected static 1x3xHxW"
                % (input_shape,)
            )
        if (
            len(output_shape) != 3
            or output_shape[0] != 1
            or output_shape[1] <= 0
            or output_shape[2] != 6
        ):
            raise RuntimeError(
                "Unexpected end-to-end output shape %s; expected 1xN x6"
                % (output_shape,)
            )


def preprocess(
    frame: np.ndarray,
    input_width: int,
    input_height: int,
    dtype,
):
    frame_height, frame_width = frame.shape[:2]
    scale = min(input_width / frame_width, input_height / frame_height)
    resized_width = int(round(frame_width * scale))
    resized_height = int(round(frame_height * scale))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    canvas[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(dtype, copy=False)
    chw *= np.array(1.0 / 255.0, dtype=dtype)
    blob = np.ascontiguousarray(chw[np.newaxis, :, :, :])
    return blob, scale, float(pad_x), float(pad_y)


def parse_end2end_output(
    output: np.ndarray,
    frame_width: int,
    frame_height: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    confidence: float,
) -> List[Detection]:
    rows = np.asarray(output).reshape(-1, 6)
    valid_scores = np.isfinite(rows[:, 4]) & (rows[:, 4] >= confidence)
    selected = rows[valid_scores]
    if selected.size == 0:
        return []
    boxes = selected[:, :4].astype(np.float32, copy=True)
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_x) / scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_y) / scale
    finite = np.all(np.isfinite(boxes), axis=1)
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, frame_width - 1.0)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, frame_height - 1.0)
    valid_boxes = finite & (boxes[:, 2] > boxes[:, 0]) & (
        boxes[:, 3] > boxes[:, 1]
    )
    boxes = boxes[valid_boxes]
    scores = selected[valid_boxes, 4]
    if boxes.size == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    boxes = boxes[order]
    scores = scores[order]
    detections = [
        Detection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            confidence=float(score),
            # The engine is validated as a one-class ball model; the class
            # column is intentionally ignored for TensorRT 8.4 compatibility.
            class_id=0,
        )
        for box, score in zip(boxes, scores)
    ]
    unique: List[Detection] = []
    occupied_centers = set()
    for detection in detections:
        center_x, center_y = detection.center
        center_key = (int(round(center_x)), int(round(center_y)))
        if center_key in occupied_centers:
            continue
        occupied_centers.add(center_key)
        unique.append(detection)
    return unique
