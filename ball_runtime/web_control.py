from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import urlparse

import cv2

from .latest import LatestValue
from .types import DetectionResult
from .visualize import draw_web_preview


class LatestJpeg:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._sequence = 0
        self._frame_id = 0
        self._encoded_monotonic = 0.0
        self._last_consumer_monotonic = 0.0
        self._error = ""

    def update(self, jpeg: bytes, frame_id: int) -> None:
        with self._condition:
            self._jpeg = bytes(jpeg)
            self._sequence += 1
            self._frame_id = int(frame_id)
            self._encoded_monotonic = time.monotonic()
            self._error = ""
            self._condition.notify_all()

    def set_error(self, error: str) -> None:
        with self._condition:
            self._error = str(error)
            self._condition.notify_all()

    def wait(self, timeout: float = 2.0):
        with self._condition:
            if self._jpeg is None:
                self._condition.wait_for(
                    lambda: self._jpeg is not None or bool(self._error),
                    timeout=max(0.0, timeout),
                )
            return self._jpeg, self._sequence, self._frame_id, self._error

    def mark_consumer(self) -> None:
        with self._condition:
            self._last_consumer_monotonic = time.monotonic()

    def encoding_needed(self, idle_after_s: float = 2.0) -> bool:
        with self._condition:
            return self._jpeg is None or (
                self._last_consumer_monotonic > 0.0
                and time.monotonic() - self._last_consumer_monotonic
                <= max(0.1, float(idle_after_s))
            )

    def status(self) -> Dict[str, object]:
        with self._condition:
            age_ms = (
                max(0.0, (time.monotonic() - self._encoded_monotonic) * 1000.0)
                if self._encoded_monotonic > 0.0
                else None
            )
            return {
                "frame_available": self._jpeg is not None,
                "frame_sequence": self._sequence,
                "frame_id": self._frame_id,
                "frame_age_ms": age_ms,
                "frame_error": self._error,
            }


class FrameJpegEncoder(threading.Thread):
    def __init__(
        self,
        results: LatestValue[DetectionResult],
        latest: LatestJpeg,
        stop_event: threading.Event,
        quality: int,
        interval_ms: int = 50,
        preview_width: int = 640,
    ) -> None:
        super().__init__(name="web-jpeg-encoder", daemon=True)
        self.results = results
        self.latest = latest
        self.stop_event = stop_event
        self.quality = max(1, min(100, int(quality)))
        self.minimum_interval_s = max(0.0, int(interval_ms) / 1000.0)
        self.preview_width = max(0, int(preview_width))

    def run(self) -> None:
        sequence = 0
        next_encode = 0.0
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        while not self.stop_event.is_set():
            next_sequence, result = self.results.wait_after(sequence, timeout=0.5)
            if next_sequence == sequence or result is None:
                continue
            sequence = next_sequence
            if not self.latest.encoding_needed():
                continue
            delay = next_encode - time.monotonic()
            if delay > 0.0 and self.stop_event.wait(delay):
                break
            newest_sequence, newest_result = self.results.get()
            if newest_result is not None and newest_sequence >= sequence:
                sequence = newest_sequence
                result = newest_result
            try:
                if result.source_image is None:
                    raise RuntimeError("detection result has no source image")
                image = draw_web_preview(result.source_image, result)
                if (
                    self.preview_width > 0
                    and image.shape[1] > self.preview_width
                ):
                    scale = self.preview_width / float(image.shape[1])
                    image = cv2.resize(
                        image,
                        (
                            self.preview_width,
                            max(1, int(round(image.shape[0] * scale))),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(".jpg", image, params)
                if not ok:
                    raise RuntimeError("OpenCV failed to encode the preview frame")
                self.latest.update(encoded.tobytes(), result.frame_id)
                next_encode = time.monotonic() + self.minimum_interval_s
            except Exception as exc:
                self.latest.set_error(str(exc))


class NxControlClient:
    TASKS = {"3", "4", "5", "6"}

    def __init__(
        self,
        socket_path: str,
        timeout_s: float = 0.5,
        target_limit_cm: float = 10.0,
    ) -> None:
        self._lock = threading.Lock()
        self.socket_path = str(socket_path)
        self.timeout_s = max(0.05, float(timeout_s))
        self.target_limit_cm = abs(float(target_limit_cm))
        self.request_id = 0
        self.last_state: Dict[str, object] = {
            "ok": False,
            "request_id": "0",
            "requested_task": "idle",
            "active_task": "idle",
            "control_mode": "idle",
            "target_cm": None,
            "running": False,
            "applied": False,
            "controller_online": False,
            "controller_state": "offline",
            "ball_position_cm": 0.0,
            "vision_ok": False,
            "dmmc_ok": False,
            "safety_latched": False,
            "error_code": "CONTROLLER_OFFLINE",
            "last_error": "",
            "message": "等待连接 NX 控制程序",
        }

    def status(self):
        return self._request("STATUS", str(self.last_state.get("active_task", "idle")))

    def set_task(self, task, target_cm=None):
        requested_task = self._normalize_task(task)
        if requested_task not in self.TASKS:
            return self._local_error(
                requested_task,
                "INVALID_TASK",
                "未知任务，只能选择 3 / 4 / 5 / 6",
            )
        command = "TASK {request_id} " + requested_task
        if requested_task == "6":
            try:
                target = float(target_cm)
            except (TypeError, ValueError):
                return self._local_error(
                    requested_task,
                    "TARGET_REQUIRED",
                    "任务6必须设置目标位置",
                )
            if not math.isfinite(target) or abs(target) > self.target_limit_cm:
                return self._local_error(
                    requested_task,
                    "TARGET_OUT_OF_RANGE",
                    "任务6目标必须在 -%.1f 至 +%.1f cm"
                    % (self.target_limit_cm, self.target_limit_cm),
                )
            command += " %.6f" % target
        return self._request(command, requested_task)

    def start(self):
        return self._request("START", str(self.last_state.get("active_task", "idle")))

    def stop(self):
        return self._request("STOP", str(self.last_state.get("active_task", "idle")))

    def reset(self):
        return self._request("RESET", "5")

    def set_target(self, target_cm):
        if self.last_state.get("active_task") != "6":
            return self._local_error(
                str(self.last_state.get("active_task", "idle")),
                "TASK6_NOT_ACTIVE",
                "请先选择任务6",
            )
        return self.set_task("6", target_cm)

    def _request(self, command_template: str, requested_task: str):
        with self._lock:
            self.request_id += 1
            request_id = str(self.request_id)
            if "{request_id}" in command_template:
                command = command_template.format(request_id=request_id)
            else:
                command = "%s %s" % (command_template, request_id)
            client_path = "/tmp/ball_web_%d_%d_%s.sock" % (
                os.getpid(),
                threading.get_ident(),
                request_id,
            )
            client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                try:
                    os.unlink(client_path)
                except FileNotFoundError:
                    pass
                client.bind(client_path)
                client.settimeout(self.timeout_s)
                client.sendto(command.encode("utf-8"), self.socket_path)
                raw = client.recv(8192)
                state = json.loads(raw.decode("utf-8"))
                if not isinstance(state, dict) or state.get("request_id") != request_id:
                    raise ValueError("controller returned a mismatched response")
                state["controller_online"] = True
                self.last_state = state
                return dict(state)
            except (
                FileNotFoundError,
                ConnectionRefusedError,
                socket.timeout,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                state = dict(self.last_state)
                state.update(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "requested_task": requested_task,
                        "applied": False,
                        "controller_online": False,
                        "controller_state": "offline",
                        "error_code": "CONTROLLER_OFFLINE",
                        "message": "NX 控制程序未响应：%s" % exc,
                    }
                )
                return state
            finally:
                client.close()
                try:
                    os.unlink(client_path)
                except FileNotFoundError:
                    pass

    def _local_error(self, requested_task: str, error_code: str, message: str):
        with self._lock:
            self.request_id += 1
            state = dict(self.last_state)
            state.update(
                {
                    "ok": False,
                    "request_id": str(self.request_id),
                    "requested_task": requested_task,
                    "applied": False,
                    "error_code": error_code,
                    "message": message,
                }
            )
            return state

    @staticmethod
    def _normalize_task(task) -> str:
        if task is None:
            return ""
        value = str(task).strip().lower()
        return value[4:] if value.startswith("task") else value


def live_page(interval_ms: int) -> str:
    interval = max(10, int(interval_ms))
    return """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ball Car Control</title>
<style>
html,body{margin:0;background:#000;height:100%%;overflow:hidden;color:#fff;font:14px sans-serif}
#view{width:100vw;height:100vh;object-fit:contain;display:block}
.panel{position:fixed;left:8px;top:8px;max-width:calc(100vw - 32px);padding:8px;
background:rgba(0,0,0,.68);border-radius:7px}
.row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:3px 0}
button,input{font:inherit} button{color:#fff;background:#333;border:1px solid #777;
border-radius:4px;padding:5px 9px} button.active{background:#0878d1;border-color:#52b9ff}
button:disabled{opacity:.5} input[type=range]{width:130px}.ok{color:#82ffab}
.warn{color:#ffd36b}.error{color:#ff8181}
</style></head><body>
<img id="view" alt="camera preview">
<div class="panel">
 <div class="row">
  <button class="task" data-task="3">任务3</button><button class="task" data-task="4">任务4</button>
  <button class="task" data-task="5">任务5</button><button class="task" data-task="6">任务6</button>
  <span>目标</span><input id="target" type="range" min="-10" max="10" step="0.1" value="0">
  <span id="targetLabel">0.0cm</span>
 </div>
 <div class="row"><button id="start">开始</button><button id="stop">停止</button>
  <button id="reset">复位</button><span id="frameStatus"></span></div>
 <div id="controlStatus" class="row warn">正在连接控制程序…</div>
</div>
<script>
const image=document.getElementById('view'), controlStatus=document.getElementById('controlStatus');
const frameStatus=document.getElementById('frameStatus'), target=document.getElementById('target');
const targetLabel=document.getElementById('targetLabel');
const buttons=Array.from(document.querySelectorAll('.task,#start,#stop,#reset'));
let imageRunning=true, activeTask='idle';
function nextImage(){if(imageRunning) image.src='/latest.jpg?t='+Date.now()}
image.onload=()=>setTimeout(nextImage,%d); image.onerror=()=>setTimeout(nextImage,250);
target.oninput=()=>targetLabel.textContent=Number(target.value).toFixed(1)+'cm';
target.onchange=()=>{if(activeTask==='6') post('/api/vision/target',{target_cm:Number(target.value)})};
async function post(url,body){
 buttons.forEach(b=>b.disabled=true); controlStatus.className='row warn'; controlStatus.textContent='正在发送指令…';
 try{const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
 render(await response.json())}catch(error){controlStatus.className='row error';controlStatus.textContent='指令失败：'+error}
 finally{buttons.forEach(b=>b.disabled=false)}
}
function render(state){
 activeTask=String(state.active_task||'idle'); document.querySelectorAll('.task').forEach(b=>b.classList.toggle('active',b.dataset.task===activeTask));
 if(activeTask==='6'&&typeof state.target_cm==='number'){target.value=state.target_cm;targetLabel.textContent=state.target_cm.toFixed(1)+'cm'}
 const health=state.controller_online?'vision '+(state.vision_ok?'OK':'LOST')+' | DMMC '+(state.dmmc_ok?'OK':'LOST'):'NX OFFLINE';
 const missingInput=!state.vision_ok||!state.dmmc_ok;
 controlStatus.className='row '+(state.safety_latched||!state.ok||(state.running&&missingInput)?'error':(state.controller_online&&!missingInput?'ok':'warn'));
 controlStatus.textContent='task '+activeTask+' | '+(state.running?'RUNNING':'STOPPED')+' | '+health+' | '+(state.message||'');
}
async function refresh(){try{const response=await fetch('/api/vision/status?t='+Date.now());render(await response.json())}catch(error){controlStatus.className='row error';controlStatus.textContent='状态连接失败：'+error}}
async function refreshFrame(){try{const response=await fetch('/healthz?t='+Date.now()),state=await response.json();frameStatus.textContent=state.frame_available?'frame '+state.frame_id:'等待相机'}catch(error){frameStatus.textContent='图传离线'}}
document.querySelectorAll('.task').forEach(b=>b.onclick=()=>post('/api/vision/task',b.dataset.task==='6'?{task:'6',target_cm:Number(target.value)}:{task:b.dataset.task}));
document.getElementById('start').onclick=()=>post('/api/vision/start'); document.getElementById('stop').onclick=()=>post('/api/vision/stop');
document.getElementById('reset').onclick=()=>post('/api/vision/reset');
setInterval(refresh,1000);setInterval(refreshFrame,1000);window.addEventListener('beforeunload',()=>imageRunning=false);
refresh();refreshFrame();nextImage();
</script></body></html>""" % interval


def make_handler(latest: LatestJpeg, control: NxControlClient, interval_ms: int):
    page = live_page(interval_ms).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_bytes(200, "text/html; charset=utf-8", page)
                return
            if path == "/latest.jpg":
                latest.mark_consumer()
                jpeg, sequence, frame_id, error = latest.wait(2.0)
                if jpeg is None:
                    self.send_error(503, error or "No camera frame available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("X-Frame-Seq", str(sequence))
                self.send_header("X-Camera-Frame-Id", str(frame_id))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if path == "/healthz":
                status = latest.status()
                status["ok"] = True
                self._send_json(status)
                return
            if path in ("/api/vision/status", "/api/task"):
                self._send_json(control.status())
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path in ("/api/vision/task", "/api/task"):
                payload = self._read_json()
                self._send_json(control.set_task(payload.get("task"), payload.get("target_cm")))
                return
            if path == "/api/vision/start":
                self._send_json(control.start())
                return
            if path == "/api/vision/stop":
                self._send_json(control.stop())
                return
            if path == "/api/vision/reset":
                self._send_json(control.reset())
                return
            if path == "/api/vision/target":
                payload = self._read_json()
                self._send_json(control.set_target(payload.get("target_cm")))
                return
            self.send_error(404)

        def _read_json(self) -> Dict[str, object]:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                return {}
            if length <= 0 or length > 4096:
                return {}
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                return value if isinstance(value, dict) else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}

        def _send_json(self, value: Dict[str, object]) -> None:
            self._send_bytes(
                200,
                "application/json; charset=utf-8",
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
            )

        def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


class WebControlServer:
    def __init__(
        self,
        results: LatestValue[DetectionResult],
        stop_event: threading.Event,
        host: str,
        port: int,
        control_socket: str,
        control_timeout_ms: int = 500,
        jpeg_quality: int = 70,
        interval_ms: int = 50,
        preview_width: int = 640,
    ) -> None:
        self.stop_event = stop_event
        self.latest = LatestJpeg()
        self.control = NxControlClient(
            control_socket,
            timeout_s=max(1, int(control_timeout_ms)) / 1000.0,
        )
        self.encoder = FrameJpegEncoder(
            results,
            self.latest,
            stop_event,
            jpeg_quality,
            interval_ms=interval_ms,
            preview_width=preview_width,
        )
        self.server = ThreadingHTTPServer(
            (str(host), int(port)),
            make_handler(self.latest, self.control, interval_ms),
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="web-control-http",
            daemon=True,
        )
        self._started = False

    @property
    def address(self):
        return self.server.server_address

    def start(self) -> None:
        self.encoder.start()
        self.thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            self.server.server_close()
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.encoder.is_alive():
            self.encoder.join(timeout=2.0)
        self._started = False
