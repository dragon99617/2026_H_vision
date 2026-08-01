from __future__ import annotations

import html
import json
import math
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, quote, urlparse

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


class FrameRecorder:
    def __init__(
        self,
        record_dir,
        enabled: bool = False,
        prefix: str = "ball_web",
        test_name: str = "run",
        fps: float = 20.0,
        mp4_bitrate_kbps: int = 1200,
    ) -> None:
        self.record_dir = Path(record_dir)
        self.enabled = bool(enabled)
        self.prefix = self._safe_name(prefix, "ball_web")
        self.test_name = self._safe_name(test_name, "run")
        self.fps = max(1.0, float(fps))
        self.mp4_bitrate_kbps = max(1, int(mp4_bitrate_kbps))
        self._lock = threading.Lock()
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._last_session_id: Optional[str] = None
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._frame_count = 0
        self._width = 0
        self._height = 0
        self._mp4_name: Optional[str] = None
        self._error = ""
        if self.enabled:
            self.start()

    @staticmethod
    def _safe_name(value: str, fallback: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(value)
        ).strip("_")
        return safe or fallback

    def start(self) -> Optional[str]:
        with self._lock:
            if not self.enabled:
                return None
            if self._session_dir is not None:
                return self._session_id
            self.record_dir.mkdir(parents=True, exist_ok=True)
            stem = "%s_%s_%s" % (
                self.prefix,
                self.test_name,
                time.strftime("%Y%m%d_%H%M%S"),
            )
            session_id = stem
            suffix = 1
            while (self.record_dir / session_id).exists():
                session_id = "%s_%02d" % (stem, suffix)
                suffix += 1
            session_dir = self.record_dir / session_id
            session_dir.mkdir(parents=False, exist_ok=False)
            self._session_id = session_id
            self._session_dir = session_dir
            self._last_session_id = session_id
            self._started_at = time.time()
            self._stopped_at = None
            self._frame_count = 0
            self._width = 0
            self._height = 0
            self._mp4_name = None
            self._error = ""
            self._write_manifest_locked()
            return session_id

    def stop(self) -> Optional[str]:
        with self._lock:
            if self._session_dir is None or self._session_id is None:
                return None
            session_id = self._session_id
            session_dir = self._session_dir
            frame_count = self._frame_count
            self._stopped_at = time.time()
            self._write_manifest_locked()
            if frame_count > 0:
                try:
                    mp4_path = self._make_mp4_locked(
                        session_dir, session_id, frame_count
                    )
                    self._mp4_name = mp4_path.name
                except (OSError, subprocess.SubprocessError) as exc:
                    self._error = "MP4 conversion failed: %s" % exc
                self._write_manifest_locked()
            self._session_id = None
            self._session_dir = None
            return session_id

    def write(self, jpeg: bytes, width: int, height: int) -> None:
        with self._lock:
            if self._session_dir is None:
                return
            try:
                if self._frame_count == 0:
                    self._width = int(width)
                    self._height = int(height)
                frame_path = self._session_dir / ("%06d.jpg" % self._frame_count)
                frame_path.write_bytes(jpeg)
                self._frame_count += 1
                if self._frame_count == 1 or self._frame_count % 30 == 0:
                    self._write_manifest_locked()
            except OSError as exc:
                self._error = "Recording failed: %s" % exc

    def is_recording(self) -> bool:
        with self._lock:
            return self._session_dir is not None

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "recording": self._session_dir is not None,
                "session_id": self._session_id,
                "last_session_id": self._last_session_id,
                "frame_count": self._frame_count,
                "mp4": self._mp4_name,
                "error": self._error,
            }

    def _manifest_locked(self) -> Dict[str, object]:
        return {
            "id": self._session_id or self._last_session_id,
            "width": self._width,
            "height": self._height,
            "fps": self.fps,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "frame_count": self._frame_count,
            "mp4": self._mp4_name,
            "error": self._error,
        }

    def _write_manifest_locked(self) -> None:
        if self._session_dir is None:
            return
        (self._session_dir / "manifest.json").write_text(
            json.dumps(self._manifest_locked(), indent=2),
            encoding="utf-8",
        )

    def _make_mp4_locked(
        self, session_dir: Path, session_id: str, frame_count: int
    ) -> Path:
        output = session_dir / (session_id + ".mp4")
        fps = max(1, int(round(self.fps)))
        command = [
            "gst-launch-1.0",
            "-q",
            "-e",
            "multifilesrc",
            "location=%s/%%06d.jpg" % session_dir,
            "index=0",
            "num-buffers=%d" % frame_count,
            "caps=image/jpeg,framerate=%d/1" % fps,
            "!",
            "jpegdec",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "x264enc",
            "speed-preset=ultrafast",
            "tune=zerolatency",
            "bitrate=%d" % self.mp4_bitrate_kbps,
            "key-int-max=%d" % fps,
            "byte-stream=false",
            "!",
            "video/x-h264,profile=baseline",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "mp4mux",
            "faststart=true",
            "!",
            "filesink",
            "location=%s" % output,
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return output


def _session_dir(record_dir, session_id: str) -> Optional[Path]:
    if not session_id or Path(session_id).name != session_id or session_id in (".", ".."):
        return None
    path = Path(record_dir) / session_id
    return path if path.is_dir() else None


def load_recording_manifest(record_dir, session_id: str) -> Optional[Dict[str, object]]:
    session_dir = _session_dir(record_dir, session_id)
    if session_dir is None:
        return None
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    manifest["frame_count"] = sum(1 for _ in session_dir.glob("*.jpg"))
    mp4_name = manifest.get("mp4")
    if mp4_name and Path(str(mp4_name)).name != str(mp4_name):
        manifest["mp4"] = None
    return manifest


class FrameJpegEncoder(threading.Thread):
    def __init__(
        self,
        results: LatestValue[DetectionResult],
        latest: LatestJpeg,
        stop_event: threading.Event,
        quality: int,
        interval_ms: int = 50,
        preview_width: int = 640,
        recorder: Optional[FrameRecorder] = None,
    ) -> None:
        super().__init__(name="web-jpeg-encoder", daemon=True)
        self.results = results
        self.latest = latest
        self.stop_event = stop_event
        self.quality = max(1, min(100, int(quality)))
        self.minimum_interval_s = max(0.0, int(interval_ms) / 1000.0)
        self.preview_width = max(0, int(preview_width))
        self.recorder = recorder

    def run(self) -> None:
        sequence = 0
        next_encode = 0.0
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        while not self.stop_event.is_set():
            next_sequence, result = self.results.wait_after(sequence, timeout=0.5)
            if next_sequence == sequence or result is None:
                continue
            sequence = next_sequence
            if not self.latest.encoding_needed() and not (
                self.recorder is not None and self.recorder.is_recording()
            ):
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
                jpeg = encoded.tobytes()
                self.latest.update(jpeg, result.frame_id)
                if self.recorder is not None:
                    self.recorder.write(jpeg, image.shape[1], image.shape[0])
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
.panel{position:fixed;left:8px;top:8px;max-width:calc(100vw - 32px);padding:8px;background:rgba(0,0,0,.68);border-radius:7px}
.row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:3px 0}
button,input{font:inherit}button{color:#fff;background:#333;border:1px solid #777;border-radius:4px;padding:5px 9px}
button.active{background:#0878d1;border-color:#52b9ff}button:disabled{opacity:.5}input[type=range]{width:130px}
.ok{color:#82ffab}.warn{color:#ffd36b}.error{color:#ff8181}a{color:#9ad7ff}
</style></head><body>
<img id="view" alt="camera preview">
<div class="panel">
 <div class="row">
  <button class="task" data-task="3">任务3</button><button class="task" data-task="4">任务4</button>
  <button class="task" data-task="5">任务5</button><button class="task" data-task="6">任务6</button>
  <span>目标</span><input id="target" type="range" min="-10" max="10" step="0.1" value="0"><span id="targetLabel">0.0cm</span>
 </div>
 <div class="row"><button id="start">开始</button><button id="stop">停止</button><button id="reset">复位</button><span id="frameStatus"></span></div>
 <div class="row"><a href="/records">历史回放</a><button id="startRecording">开始录制</button><button id="stopRecording">停止录制</button><span id="recordStatus"></span></div>
 <div id="controlStatus" class="row warn">正在连接控制程序…</div>
</div>
<script>
const image=document.getElementById('view'),controlStatus=document.getElementById('controlStatus');
const frameStatus=document.getElementById('frameStatus'),target=document.getElementById('target');
const targetLabel=document.getElementById('targetLabel'),recordStatus=document.getElementById('recordStatus');
const buttons=Array.from(document.querySelectorAll('.task,#start,#stop,#reset'));
let imageRunning=true,activeTask='idle';
function nextImage(){if(imageRunning)image.src='/latest.jpg?t='+Date.now()}
image.onload=()=>setTimeout(nextImage,%d);image.onerror=()=>setTimeout(nextImage,250);
target.oninput=()=>targetLabel.textContent=Number(target.value).toFixed(1)+'cm';
target.onchange=()=>{if(activeTask==='6')post('/api/vision/target',{target_cm:Number(target.value)})};
async function post(url,body){
 buttons.forEach(b=>b.disabled=true);controlStatus.className='row warn';controlStatus.textContent='正在发送指令…';
 try{const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});render(await response.json())}
 catch(error){controlStatus.className='row error';controlStatus.textContent='指令失败：'+error}
 finally{buttons.forEach(b=>b.disabled=false)}
}
function render(state){
 activeTask=String(state.active_task||'idle');document.querySelectorAll('.task').forEach(b=>b.classList.toggle('active',b.dataset.task===activeTask));
 if(activeTask==='6'&&typeof state.target_cm==='number'){target.value=state.target_cm;targetLabel.textContent=state.target_cm.toFixed(1)+'cm'}
 const health=state.controller_online?'vision '+(state.vision_ok?'OK':'LOST')+' | DMMC '+(state.dmmc_ok?'OK':'LOST'):'NX OFFLINE';
 const missingInput=!state.vision_ok||!state.dmmc_ok;
 controlStatus.className='row '+(state.safety_latched||!state.ok||(state.running&&missingInput)?'error':(state.controller_online&&!missingInput?'ok':'warn'));
 controlStatus.textContent='task '+activeTask+' | '+(state.running?'RUNNING':'STOPPED')+' | '+health+' | '+(state.message||'');
}
async function refresh(){try{const response=await fetch('/api/vision/status?t='+Date.now());render(await response.json())}catch(error){controlStatus.className='row error';controlStatus.textContent='状态连接失败：'+error}}
async function refreshFrame(){try{const response=await fetch('/healthz?t='+Date.now()),state=await response.json();frameStatus.textContent=state.frame_available?'frame '+state.frame_id:'等待相机'}catch(error){frameStatus.textContent='图传离线'}}
async function refreshRecording(){
 try{const response=await fetch('/api/recording/status?t='+Date.now()),state=await response.json();
  recordStatus.className=state.error?'error':(state.recording?'ok':'warn');
  recordStatus.textContent=state.recording?'REC '+state.frame_count:(state.enabled?'录制已停止':'录制未启用');
  document.getElementById('startRecording').disabled=!state.enabled||state.recording;
  document.getElementById('stopRecording').disabled=!state.recording
 }catch(error){recordStatus.className='error';recordStatus.textContent='录制状态失败'}
}
async function recordingAction(action){
 const button=document.getElementById(action+'Recording');button.disabled=true;recordStatus.className='warn';
 recordStatus.textContent=action==='stop'?'正在生成 MP4…':'正在创建录像…';
 const url=action==='stop'?'/api/recording/stop':'/api/recording/start';
 try{await fetch(url,{method:'POST'});await refreshRecording()}
 catch(error){recordStatus.className='error';recordStatus.textContent='录制操作失败：'+error}
}
document.querySelectorAll('.task').forEach(b=>b.onclick=()=>post('/api/vision/task',b.dataset.task==='6'?{task:'6',target_cm:Number(target.value)}:{task:b.dataset.task}));
document.getElementById('start').onclick=()=>post('/api/vision/start');document.getElementById('stop').onclick=()=>post('/api/vision/stop');
document.getElementById('reset').onclick=()=>post('/api/vision/reset');
document.getElementById('startRecording').onclick=()=>recordingAction('start');document.getElementById('stopRecording').onclick=()=>recordingAction('stop');
setInterval(refresh,1000);setInterval(refreshFrame,1000);setInterval(refreshRecording,1000);
window.addEventListener('beforeunload',()=>imageRunning=false);refresh();refreshFrame();refreshRecording();nextImage();
</script></body></html>""" % interval


def recordings_page(record_dir) -> str:
    root = Path(record_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for session_dir in sorted(root.iterdir(), key=lambda value: value.name, reverse=True):
        if not session_dir.is_dir():
            continue
        manifest = load_recording_manifest(root, session_dir.name)
        if manifest is None or int(manifest.get("frame_count", 0)) <= 0:
            continue
        session_id = session_dir.name
        encoded_id = quote(session_id, safe="")
        has_mp4 = bool(manifest.get("mp4"))
        href = "/video?id=%s" % encoded_id if has_mp4 else "/play?id=%s" % encoded_id
        label = "MP4" if has_mp4 else "JPEG frames"
        rows.append(
            "<li><a href='%s'>%s</a> (%d frames, %s)</li>"
            % (href, html.escape(session_id), int(manifest.get("frame_count", 0)), label)
        )
    body = "\n".join(rows) if rows else "<li>暂无历史录像</li>"
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>历史回放</title><style>body{font:18px sans-serif;background:#111;color:#eee}a{color:#8fd3ff}</style>
</head><body><h2>历史回放</h2><p><a href="/">返回实时画面</a></p><ul>%s</ul></body></html>""" % body


def video_page(session_id: str) -> str:
    encoded_id = quote(session_id, safe="")
    safe_id = html.escape(session_id)
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title>
<style>html,body{margin:0;background:#000;color:#fff;font:16px sans-serif}video{width:100vw;height:calc(100vh - 52px);background:#000;display:block}
.bar{height:52px;padding:12px;background:#151515;box-sizing:border-box}a{color:#8fd3ff;margin-right:12px}</style></head>
<body><video controls playsinline preload="metadata" src="/record_video.mp4?id=%s"></video>
<div class="bar"><a href="/records">历史列表</a><a href="/">实时画面</a><span>%s</span></div></body></html>""" % (safe_id, encoded_id, safe_id)


def frame_play_page(session_id: str, frame_count: int, fps: float) -> str:
    default_fps = min(20.0, max(1.0, float(fps)))
    interval = max(1, int(round(1000.0 / default_fps)))
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title>
<style>html,body{margin:0;background:#000;color:#fff;font:14px sans-serif;height:100%%;overflow:hidden}img{width:100vw;height:calc(100vh - 88px);object-fit:contain;display:block}
.controls{height:88px;padding:8px;background:#151515;box-sizing:border-box}input[type=range]{width:100%%}button,a{font:16px sans-serif;margin-right:8px;color:#fff}a{color:#8fd3ff}</style></head>
<body><img id="frame"><div class="controls"><input id="slider" type="range" min="0" max="%d" value="0"><div>
<button id="play">播放</button><button id="pause">暂停</button><button id="fps15">15fps</button><button id="fps20">20fps</button><button id="fps30">30fps</button>
<a href="/records">历史列表</a><a href="/">实时画面</a><span id="label"></span></div></div><script>
const session=%s,count=%d,frame=document.getElementById('frame'),slider=document.getElementById('slider'),label=document.getElementById('label');
let interval=%d,idx=0,playing=false,loading=false;
function show(i){idx=Math.max(0,Math.min(count-1,i));slider.value=idx;label.textContent=(idx+1)+' / '+count;loading=true;frame.src='/record_frame?id='+encodeURIComponent(session)+'&i='+idx+'&t='+Date.now()}
function pause(){playing=false}function play(){if(idx>=count-1)show(0);playing=true;if(!loading)scheduleNext()}
function scheduleNext(){if(!playing)return;if(idx>=count-1){pause();return}setTimeout(()=>{if(playing)show(idx+1)},interval)}
function setFps(fps){interval=Math.max(1,Math.round(1000/fps));label.textContent=(idx+1)+' / '+count+' @ '+fps+'fps'}
slider.oninput=()=>{pause();show(Number(slider.value))};frame.onload=()=>{loading=false;if(playing)scheduleNext()};frame.onerror=()=>{loading=false;if(playing)setTimeout(scheduleNext,200)};
document.getElementById('play').onclick=play;document.getElementById('pause').onclick=pause;document.getElementById('fps15').onclick=()=>setFps(15);
document.getElementById('fps20').onclick=()=>setFps(20);document.getElementById('fps30').onclick=()=>setFps(30);show(0);
</script></body></html>""" % (html.escape(session_id), frame_count - 1, json.dumps(session_id), frame_count, interval)


def make_handler(
    latest: LatestJpeg,
    control: NxControlClient,
    interval_ms: int,
    recorder: Optional[FrameRecorder] = None,
):
    if recorder is None:
        recorder = FrameRecorder("records", enabled=False)
    page = live_page(interval_ms).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
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
            if path in ("/api/status", "/api/recording/status"):
                self._send_json(recorder.status())
                return
            if path == "/api/stop_recording":
                self._stop_recording()
                return
            if path == "/records":
                self._send_bytes(
                    200,
                    "text/html; charset=utf-8",
                    recordings_page(recorder.record_dir).encode("utf-8"),
                )
                return
            if path == "/video":
                self._send_video_page(query.get("id", [""])[0])
                return
            if path == "/record_video.mp4":
                self._send_record_video(query.get("id", [""])[0])
                return
            if path == "/play":
                self._send_frame_play_page(query.get("id", [""])[0])
                return
            if path == "/record_frame":
                try:
                    index = int(query.get("i", ["0"])[0])
                except ValueError:
                    self.send_error(400, "Invalid frame index")
                    return
                self._send_record_frame(query.get("id", [""])[0], index)
                return
            if path == "/api/session":
                manifest = load_recording_manifest(
                    recorder.record_dir, query.get("id", [""])[0]
                )
                if manifest is None:
                    self.send_error(404, "No such recording")
                else:
                    self._send_json(manifest)
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
            if path == "/api/recording/start":
                session_id = recorder.start()
                response = recorder.status()
                response.update({"ok": session_id is not None, "started_session": session_id})
                self._send_json(response)
                return
            if path in ("/api/recording/stop", "/api/stop_recording"):
                self._stop_recording()
                return
            self.send_error(404)

        def _stop_recording(self) -> None:
            session_id = recorder.stop()
            response = recorder.status()
            response.update({"ok": True, "stopped_session": session_id})
            self._send_json(response)

        def _send_video_page(self, session_id: str) -> None:
            manifest = load_recording_manifest(recorder.record_dir, session_id)
            if manifest is None or not manifest.get("mp4"):
                self.send_error(404, "No MP4 for this recording")
                return
            self._send_bytes(
                200,
                "text/html; charset=utf-8",
                video_page(session_id).encode("utf-8"),
            )

        def _send_record_video(self, session_id: str) -> None:
            manifest = load_recording_manifest(recorder.record_dir, session_id)
            session_dir = _session_dir(recorder.record_dir, session_id)
            if manifest is None or session_dir is None or not manifest.get("mp4"):
                self.send_error(404, "No MP4 for this recording")
                return
            mp4_path = session_dir / str(manifest["mp4"])
            if not mp4_path.is_file():
                self.send_error(404, "MP4 file missing")
                return
            size = mp4_path.stat().st_size
            start = 0
            end = size - 1
            partial = False
            range_header = self.headers.get("Range", "")
            if range_header:
                try:
                    if not range_header.startswith("bytes=") or "," in range_header:
                        raise ValueError
                    start_text, separator, end_text = range_header[6:].partition("-")
                    if not separator or size <= 0:
                        raise ValueError
                    if start_text:
                        start = int(start_text)
                        end = int(end_text) if end_text else size - 1
                    else:
                        suffix_length = int(end_text)
                        if suffix_length <= 0:
                            raise ValueError
                        start = max(0, size - suffix_length)
                        end = size - 1
                    end = min(end, size - 1)
                    if start < 0 or start > end:
                        raise ValueError
                    partial = True
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with mp4_path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _send_frame_play_page(self, session_id: str) -> None:
            manifest = load_recording_manifest(recorder.record_dir, session_id)
            if manifest is None or int(manifest.get("frame_count", 0)) <= 0:
                self.send_error(404, "No such recording")
                return
            self._send_bytes(
                200,
                "text/html; charset=utf-8",
                frame_play_page(
                    session_id,
                    int(manifest["frame_count"]),
                    float(manifest.get("fps", 20.0)),
                ).encode("utf-8"),
            )

        def _send_record_frame(self, session_id: str, index: int) -> None:
            session_dir = _session_dir(recorder.record_dir, session_id)
            if session_dir is None or index < 0:
                self.send_error(404, "No such frame")
                return
            frame_path = session_dir / ("%06d.jpg" % index)
            if not frame_path.is_file():
                self.send_error(404, "No such frame")
                return
            self._send_bytes(200, "image/jpeg", frame_path.read_bytes())

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
        record: bool = False,
        record_dir="records",
        record_prefix: str = "ball_web",
        record_name: str = "run",
        record_fps: float = 20.0,
        mp4_bitrate_kbps: int = 1200,
    ) -> None:
        self.stop_event = stop_event
        self.latest = LatestJpeg()
        self.control = NxControlClient(
            control_socket,
            timeout_s=max(1, int(control_timeout_ms)) / 1000.0,
        )
        self.recorder = FrameRecorder(
            record_dir,
            enabled=record,
            prefix=record_prefix,
            test_name=record_name,
            fps=record_fps,
            mp4_bitrate_kbps=mp4_bitrate_kbps,
        )
        self.encoder = FrameJpegEncoder(
            results,
            self.latest,
            stop_event,
            jpeg_quality,
            interval_ms=interval_ms,
            preview_width=preview_width,
            recorder=self.recorder,
        )
        self.server = ThreadingHTTPServer(
            (str(host), int(port)),
            make_handler(self.latest, self.control, interval_ms, self.recorder),
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
            self.recorder.stop()
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.encoder.is_alive():
            self.encoder.join(timeout=2.0)
        self.recorder.stop()
        self._started = False
