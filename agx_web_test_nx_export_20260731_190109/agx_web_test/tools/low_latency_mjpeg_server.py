#!/usr/bin/env python3
import argparse
import fcntl
import os
import select
import signal
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler

import cv2

VIDIOC_QUERYCAP = 0x80685600
VIDIOC_S_FMT = 0xC0CC5605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0
V4L2_PIX_FMT_MJPEG = int.from_bytes(b"MJPG", "little")


class SharedFrame:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.error = None

    def update(self, jpeg):
        with self.condition:
            self.jpeg = bytes(jpeg)
            self.sequence += 1
            self.condition.notify_all()

    def set_error(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()


class V4L2MjpegCapture:
    def __init__(self, device, width, height, fps, buffers):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.buffers_count = buffers
        self.fd = None
        self.maps = []

    def open(self):
        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        fcntl.ioctl(self.fd, VIDIOC_QUERYCAP, bytearray(104))

        fmt = bytearray(208)
        struct.pack_into("I", fmt, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("IIIII", fmt, 8, self.width, self.height, V4L2_PIX_FMT_MJPEG, V4L2_FIELD_ANY, 0)
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)

        req = bytearray(20)
        struct.pack_into("IIII", req, 0, self.buffers_count, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, 0)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
        count = struct.unpack_from("I", req, 0)[0]
        if count == 0:
            raise RuntimeError("V4L2 returned zero mmap buffers")

        import mmap

        for index in range(count):
            buf = self._empty_buffer()
            struct.pack_into("I", buf, 0, index)
            struct.pack_into("I", buf, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            struct.pack_into("I", buf, 8, V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)
            length = struct.unpack_from("I", buf, 32)[0]
            offset = struct.unpack_from("I", buf, 44)[0]
            mapped = mmap.mmap(self.fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
            self.maps.append(mapped)
            fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)

        buf_type = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, buf_type)

    def read_frame(self):
        select.select([self.fd], [], [], 1.0)
        buf = self._empty_buffer()
        struct.pack_into("I", buf, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", buf, 8, V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        index = struct.unpack_from("I", buf, 0)[0]
        used = struct.unpack_from("I", buf, 24)[0]
        frame = self.maps[index][:used]
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
        return frame

    def close(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE))
            except OSError:
                pass
        for mapped in self.maps:
            mapped.close()
        self.maps.clear()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    @staticmethod
    def _empty_buffer():
        return bytearray(88)


def opencv_capture_loop(args, shared, stop_event):
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open camera {args.device}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    while not stop_event.is_set():
        ok, frame = capture.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            shared.update(jpeg.tobytes())
    capture.release()


def direct_v4l2_capture_loop(args, shared, stop_event):
    capture = V4L2MjpegCapture(args.device, args.width, args.height, args.fps, args.buffers)
    capture.open()
    while not stop_event.is_set():
        try:
            shared.update(capture.read_frame())
        except BlockingIOError:
            time.sleep(0.001)
    capture.close()


def capture_loop(args, shared, stop_event):
    try:
        try:
            print("Trying direct V4L2 MJPEG mode...")
            direct_v4l2_capture_loop(args, shared, stop_event)
        except Exception as direct_error:
            if args.backend == "direct":
                raise
            print(f"Direct V4L2 mode failed: {direct_error}")
            print("Falling back to OpenCV low-buffer mode.")
            opencv_capture_loop(args, shared, stop_event)
    except Exception as exc:
        shared.set_error(str(exc))


def make_handler(shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("%s - %s" % (self.client_address[0], fmt % args))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(
                    b"<!doctype html><html><head><meta name='viewport' "
                    b"content='width=device-width,initial-scale=1'>"
                    b"<title>AGX Low Latency Preview</title>"
                    b"<style>html,body{margin:0;background:#000;height:100%;overflow:hidden;}"
                    b"img{width:100vw;height:100vh;object-fit:contain;}</style>"
                    b"</head><body><img src='/stream.mjpg'></body></html>"
                )
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_sequence = -1
            while True:
                with shared.condition:
                    shared.condition.wait_for(
                        lambda: shared.sequence != last_sequence or shared.error is not None,
                        timeout=1.0,
                    )
                    if shared.error is not None:
                        raise RuntimeError(shared.error)
                    jpeg = shared.jpeg
                    last_sequence = shared.sequence

                if jpeg is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Cache-Control: no-store\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video6")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--backend", choices=["auto", "direct"], default="auto")
    args = parser.parse_args()

    shared = SharedFrame()
    stop_event = threading.Event()
    worker = threading.Thread(target=capture_loop, args=(args, shared, stop_event), daemon=True)
    worker.start()

    class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ThreadingServer((args.host, args.port), make_handler(shared))

    def stop(_signum, _frame):
        stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"Serving low-latency MJPEG preview on http://{args.host}:{args.port}/")
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()
