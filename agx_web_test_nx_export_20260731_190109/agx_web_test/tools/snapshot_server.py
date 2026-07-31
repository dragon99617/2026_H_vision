#!/usr/bin/env python3
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer, ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import cv2


class LatestFrame:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.error = None

    def update(self, jpeg):
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.condition.notify_all()

    def set_error(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def wait_for_frame(self, timeout=2.0):
        with self.condition:
            if self.jpeg is None and self.error is None:
                self.condition.wait(timeout)
            if self.error is not None:
                raise RuntimeError(self.error)
            return self.jpeg, self.sequence


class FrameRecorder:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.session_id = None
        self.session_dir = None
        self.started_at = None
        self.stopped_at = None
        self.frame_count = 0
        self.mp4_path = None
        if args.record:
            self.start()

    def start(self):
        with self.lock:
            if self.session_dir is not None:
                return self.session_id
            root = Path(self.args.record_dir)
            root.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.args.test_name)
            self.session_id = f"{self.args.record_prefix}_{safe_name}_{timestamp}"
            self.session_dir = root / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.started_at = time.time()
            self.stopped_at = None
            self.frame_count = 0
            self.mp4_path = None
            self._write_manifest_locked()
            print(f"Recording frames to {self.session_dir}")
            return self.session_id

    def stop(self):
        with self.lock:
            if self.session_dir is None:
                return None
            self.stopped_at = time.time()
            session = self.session_id
            self._write_manifest_locked()
            session_dir = self.session_dir
            frame_count = self.frame_count
            record_fps = self.args.record_fps
            print(f"Recording stopped: {session_dir} ({frame_count} frames)")
            if frame_count > 0:
                try:
                    self.mp4_path = self._make_mp4_locked(session_dir, session, frame_count, record_fps)
                    self._write_manifest_locked()
                    print(f"MP4 ready: {self.mp4_path}")
                except Exception as exc:
                    print(f"MP4 conversion failed: {exc}")
            self.session_id = None
            self.session_dir = None
            return session

    def write(self, jpeg):
        with self.lock:
            if self.session_dir is None:
                return
            frame_path = self.session_dir / f"{self.frame_count:06d}.jpg"
            frame_path.write_bytes(jpeg)
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self._write_manifest_locked()

    def status(self):
        with self.lock:
            return {
                "recording": self.session_dir is not None,
                "session_id": self.session_id,
                "frame_count": self.frame_count,
            }

    def _write_manifest_locked(self):
        if self.session_dir is None:
            return
        manifest = {
            "id": self.session_id,
            "width": self.args.width,
            "height": self.args.height,
            "fps": self.args.record_fps,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "frame_count": self.frame_count,
            "mp4": self.mp4_path.name if self.mp4_path else None,
        }
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _make_mp4_locked(self, session_dir, session_id, frame_count, record_fps):
        output = session_dir / f"{session_id}.mp4"
        fps = max(1, int(round(float(record_fps))))
        bitrate_kbps = int(self.args.mp4_bitrate_kbps)
        cmd = [
            "gst-launch-1.0", "-q", "-e",
            "multifilesrc",
            f"location={session_dir}/%06d.jpg",
            "index=0",
            f"num-buffers={frame_count}",
            f"caps=image/jpeg,framerate={fps}/1",
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
            f"bitrate={bitrate_kbps}",
            f"key-int-max={fps}",
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
            f"location={output}",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return output


class NxControlClient:
    TASK_INFO = {
        "3": ("swing_test", None, "任务3：静止摆动"),
        "4": ("hold_center", 0.0, "任务4：中心保持"),
        "5": ("hold_center", 0.0, "任务5：中心保持"),
        "6": ("hold_target", 0.0, "任务6：指定位置"),
    }

    def __init__(self, socket_path, target_limit_cm=10.0, timeout_s=0.5):
        self.lock = threading.Lock()
        self.socket_path = str(socket_path)
        self.target_limit_cm = abs(float(target_limit_cm))
        self.timeout_s = max(0.05, float(timeout_s))
        self.request_id = 0
        self.last_state = {
            "ok": False,
            "request_id": "0",
            "requested_task": "5",
            "active_task": "5",
            "control_mode": "hold_center",
            "target_cm": 0.0,
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
        return self._request("STATUS", requested_task=self.last_state["active_task"])

    def set_task(self, task, target_cm=None):
        requested_task = self._normalize_task(task)
        if requested_task not in self.TASK_INFO:
            return self._local_error(
                requested_task, "INVALID_TASK", f"未知任务 {task}，只能选择 3 / 4 / 5 / 6"
            )
        command = f"TASK {{request_id}} {requested_task}"
        if requested_task == "6":
            try:
                target = float(target_cm)
            except (TypeError, ValueError):
                return self._local_error(
                    requested_task, "TARGET_REQUIRED", "任务6必须设置目标位置"
                )
            if not math.isfinite(target) or abs(target) > self.target_limit_cm:
                return self._local_error(
                    requested_task,
                    "TARGET_OUT_OF_RANGE",
                    f"任务6目标必须在 -{self.target_limit_cm:g} 至 +{self.target_limit_cm:g} cm",
                )
            command += f" {target:.6f}"
        return self._request(command, requested_task=requested_task)

    def start(self):
        return self._request("START", requested_task=self.last_state["active_task"])

    def stop(self):
        return self._request("STOP", requested_task=self.last_state["active_task"])

    def reset(self):
        return self._request("RESET", requested_task="5")

    def set_target(self, target_cm):
        if self.last_state.get("active_task") != "6":
            return self._local_error(
                self.last_state.get("active_task", ""),
                "TASK6_NOT_ACTIVE",
                "请先选择任务6",
            )
        return self.set_task("6", target_cm)

    def _request(self, command_template, requested_task):
        with self.lock:
            self.request_id += 1
            request_id = str(self.request_id)
            command = command_template.format(request_id=request_id)
            if "{request_id}" not in command_template:
                command = f"{command_template} {request_id}"
            client_path = (
                f"/tmp/agx_web_task_{os.getpid()}_{threading.get_ident()}_{request_id}.sock"
            )
            client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                if os.path.exists(client_path):
                    os.unlink(client_path)
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
            except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError, ValueError, json.JSONDecodeError) as exc:
                state = dict(self.last_state)
                state.update({
                    "ok": False,
                    "request_id": request_id,
                    "requested_task": str(requested_task),
                    "applied": False,
                    "controller_online": False,
                    "controller_state": "offline",
                    "error_code": "CONTROLLER_OFFLINE",
                    "message": f"NX 控制程序未响应：{exc}",
                })
                return state
            finally:
                client.close()
                try:
                    os.unlink(client_path)
                except FileNotFoundError:
                    pass

    def _local_error(self, requested_task, error_code, message):
        with self.lock:
            self.request_id += 1
            state = dict(self.last_state)
            state.update({
                "ok": False,
                "request_id": str(self.request_id),
                "requested_task": str(requested_task),
                "applied": False,
                "error_code": error_code,
                "message": message,
            })
            return state

    def _normalize_task(self, task):
        if task is None:
            return ""
        text = str(task).strip().lower()
        if text.startswith("task"):
            text = text[4:]
        return text


def load_manifest(record_dir, session_id):
    session_dir = Path(record_dir) / session_id
    manifest_path = session_dir / "manifest.json"
    if not session_dir.is_dir() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = len(list(session_dir.glob("*.jpg")))
    return manifest


def capture_loop(args, latest, recorder, stop_event):
    try:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open camera {args.device}")

        params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.003)
                continue
            ok, jpeg = cv2.imencode(".jpg", frame, params)
            if ok:
                jpeg_bytes = jpeg.tobytes()
                latest.update(jpeg_bytes)
                recorder.write(jpeg_bytes)
        cap.release()
    except Exception as exc:
        latest.set_error(str(exc))


def make_handler(latest, recorder, vision_state, args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/latest.jpg":
                self._send_latest()
                return
            if path == "/api/status":
                self._send_json(recorder.status())
                return
            if path in ("/api/vision/status", "/api/task"):
                self._send_json(vision_state.status())
                return
            if path == "/api/stop_recording":
                session = recorder.stop()
                self._send_json({"ok": True, "stopped_session": session})
                return
            if path == "/records":
                self._send_records_page()
                return
            if path == "/video":
                self._send_video_page(query.get("id", [""])[0])
                return
            if path == "/record_video.mp4":
                self._send_record_video(query.get("id", [""])[0])
                return
            if path == "/play":
                self._send_play_page(query.get("id", [""])[0])
                return
            if path == "/record_frame":
                session_id = query.get("id", [""])[0]
                index = int(query.get("i", ["0"])[0])
                self._send_record_frame(session_id, index)
                return
            if path == "/api/session":
                session_id = query.get("id", [""])[0]
                manifest = load_manifest(args.record_dir, session_id)
                if manifest is None:
                    self.send_error(404, "No such recording")
                    return
                self._send_json(manifest)
                return
            if path in ("/", "/index.html"):
                self._send_live_page()
                return

            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/api/vision/task", "/api/task"):
                payload = self._read_json()
                self._send_json(vision_state.set_task(payload.get("task"), payload.get("target_cm")))
                return
            if path == "/api/vision/start":
                self._send_json(vision_state.start())
                return
            if path == "/api/vision/stop":
                self._send_json(vision_state.stop())
                return
            if path == "/api/vision/reset":
                self._send_json(vision_state.reset())
                return
            if path == "/api/vision/target":
                payload = self._read_json()
                self._send_json(vision_state.set_target(payload.get("target_cm")))
                return

            self.send_error(404)

        def _send_latest(self):
            jpeg, seq = latest.wait_for_frame()
            if jpeg is None:
                self.send_error(503, "No frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Frame-Seq", str(seq))
            self.end_headers()
            self.wfile.write(jpeg)

        def _send_live_page(self):
            interval = max(1, int(args.interval_ms))
            html = f"""<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AGX Live Preview</title>
  <style>
    html,body{{margin:0;background:#000;height:100%;overflow:hidden;}}
    img{{width:100vw;height:100vh;object-fit:contain;display:block;}}
    .bar{{position:fixed;left:8px;top:8px;background:rgba(0,0,0,.58);color:#fff;
      font:14px sans-serif;padding:6px 8px;border-radius:6px;max-width:calc(100vw - 32px);}}
    .row{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:2px 0;}}
    a,button,input{{font:inherit;}}
    a{{color:#9ad7ff;margin-right:4px;}}
    button{{color:#fff;background:#333;border:1px solid #777;border-radius:4px;padding:4px 8px;}}
    button.active{{background:#0b75d1;border-color:#4bb2ff;}}
    input[type=range]{{width:120px;}}
    .msg{{color:#ddd;}}
    .msg.ok{{color:#8dffad;}}
    .msg.warn{{color:#ffd36b;}}
    .msg.error{{color:#ff7d7d;}}
    button:disabled{{opacity:.55;}}
  </style>
</head>
<body>
  <img id="view" alt="camera">
  <div class="bar">
    <div class="row">
      <button class="task" data-task="3">任务3</button>
      <button class="task" data-task="4">任务4</button>
      <button class="task" data-task="5">任务5</button>
      <button class="task" data-task="6">任务6</button>
      <span>目标</span>
      <input id="target" type="range" min="-10" max="10" step="0.1" value="0">
      <span id="targetLabel">0.0cm</span>
    </div>
    <div class="row">
      <button id="startTask">开始</button>
      <button id="stopTask">停止</button>
      <button id="resetTask">复位</button>
      <a href="/records">recordings</a>
      <button id="stopRec">stop recording</button>
      <span id="status"></span>
    </div>
    <div class="row msg" id="visionStatus"></div>
  </div>
  <script>
    const img = document.getElementById('view');
    const status = document.getElementById('status');
    const visionStatus = document.getElementById('visionStatus');
    const target = document.getElementById('target');
    const targetLabel = document.getElementById('targetLabel');
    const controlButtons = Array.from(document.querySelectorAll('.task, #startTask, #stopTask, #resetTask'));
    const intervalMs = {interval};
    let running = true;
    function next() {{
      if (!running) return;
      img.src = '/latest.jpg?t=' + Date.now();
    }}
    img.onload = () => setTimeout(next, intervalMs);
    img.onerror = () => setTimeout(next, 200);
    document.getElementById('stopRec').onclick = async () => {{
      status.textContent = 'converting mp4... wait';
      await fetch('/api/stop_recording');
      await updateStatus();
    }};
    target.oninput = () => {{
      targetLabel.textContent = Number(target.value).toFixed(1) + 'cm';
    }};
    async function postJson(url, body) {{
      controlButtons.forEach(button => button.disabled = true);
      visionStatus.className = 'row msg warn';
      visionStatus.textContent = '正在发送指令…';
      try {{
        const response = await fetch(url, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(body || {{}})
        }});
        const state = await response.json();
        renderVision(state);
        return state;
      }} catch (error) {{
        visionStatus.className = 'row msg error';
        visionStatus.textContent = '网页指令发送失败：' + error;
      }} finally {{
        controlButtons.forEach(button => button.disabled = false);
      }}
    }}
    async function setTask(task) {{
      const body = {{task}};
      if (task === '6') body.target_cm = Number(target.value);
      await postJson('/api/vision/task', body);
    }}
    async function updateVisionStatus() {{
      const response = await fetch('/api/vision/status?t=' + Date.now());
      const state = await response.json();
      renderVision(state);
    }}
    function renderVision(state) {{
      document.querySelectorAll('.task').forEach(button => {{
        button.classList.toggle('active', button.dataset.task === state.active_task);
      }});
      if (state.active_task === '6' && typeof state.target_cm === 'number') {{
        target.value = state.target_cm;
        targetLabel.textContent = state.target_cm.toFixed(1) + 'cm';
      }}
      const runText = state.running ? 'RUNNING' : 'STOPPED';
      const onlineText = state.controller_online ? 'NX ONLINE' : 'NX OFFLINE';
      const targetText = typeof state.target_cm === 'number'
        ? ` | target ${{state.target_cm.toFixed(1)}}cm`
        : '';
      const healthText = state.controller_online
        ? ` | vision ${{state.vision_ok ? 'OK' : 'LOST'}} | DMMC ${{state.dmmc_ok ? 'OK' : 'LOST'}}`
        : '';
      const safeText = state.safety_latched ? ` | SAFE: ${{state.last_error || 'latched'}}` : '';
      visionStatus.className = state.safety_latched || !state.ok
        ? 'row msg error'
        : (state.controller_online ? 'row msg ok' : 'row msg warn');
      visionStatus.textContent =
        `${{onlineText}} | task ${{state.active_task}} | ${{state.control_mode}} | ${{runText}}${{targetText}}${{healthText}}${{safeText}} | ${{state.message}}`;
    }}
    document.querySelectorAll('.task').forEach(button => {{
      button.onclick = () => setTask(button.dataset.task);
    }});
    document.getElementById('startTask').onclick = () => postJson('/api/vision/start');
    document.getElementById('stopTask').onclick = () => postJson('/api/vision/stop');
    document.getElementById('resetTask').onclick = () => postJson('/api/vision/reset');
    async function updateStatus() {{
      const response = await fetch('/api/status?t=' + Date.now());
      const state = await response.json();
      status.textContent = state.recording ? `rec ${{state.frame_count}}` : 'recording stopped';
    }}
    setInterval(updateStatus, 1000);
    setInterval(updateVisionStatus, 1000);
    window.addEventListener('beforeunload', () => running = false);
    updateStatus();
    updateVisionStatus();
    next();
  </script>
</body>
</html>"""
            self._send_html(html)

        def _send_records_page(self):
            root = Path(args.record_dir)
            root.mkdir(parents=True, exist_ok=True)
            rows = []
            for session_dir in sorted(root.iterdir(), reverse=True):
                if not session_dir.is_dir():
                    continue
                manifest = load_manifest(args.record_dir, session_dir.name)
                if manifest is None or manifest["frame_count"] == 0:
                    continue
                if manifest.get("mp4"):
                    href = f"/video?id={session_dir.name}"
                    label = "video"
                else:
                    href = f"/play?id={session_dir.name}"
                    label = "frames"
                rows.append(
                    f"<li><a href='{href}'>{session_dir.name}</a> "
                    f"({manifest['frame_count']} frames, {label})</li>"
                )
            body = "\n".join(rows) if rows else "<li>No recordings yet</li>"
            html = f"""<!doctype html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recordings</title>
<style>body{{font:18px sans-serif;background:#111;color:#eee;}}a{{color:#8fd3ff;}}</style>
</head><body><h2>Recordings</h2><p><a href="/">Live preview</a></p><ul>{body}</ul></body></html>"""
            self._send_html(html)

        def _send_video_page(self, session_id):
            manifest = load_manifest(args.record_dir, session_id)
            if manifest is None or not manifest.get("mp4"):
                self.send_error(404, "No MP4 for this recording yet")
                return
            html = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{session_id}</title>
<style>
html,body{{margin:0;background:#000;color:#fff;font:16px sans-serif;}}
video{{width:100vw;height:calc(100vh - 48px);background:#000;display:block;}}
.bar{{height:48px;padding:10px;background:#151515;box-sizing:border-box;}}
a{{color:#8fd3ff;margin-right:12px;}}
</style></head>
<body>
<video controls playsinline preload="metadata" src="/record_video.mp4?id={session_id}"></video>
<div class="bar"><a href="/records">records</a><a href="/">live</a>
<span>{session_id}</span></div>
</body></html>"""
            self._send_html(html)

        def _send_record_video(self, session_id):
            manifest = load_manifest(args.record_dir, session_id)
            if manifest is None or not manifest.get("mp4"):
                self.send_error(404, "No MP4 for this recording")
                return
            mp4_path = Path(args.record_dir) / session_id / manifest["mp4"]
            if not mp4_path.is_file():
                self.send_error(404, "MP4 file missing")
                return
            size = mp4_path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                start_s, _, end_s = range_header[6:].partition("-")
                start = int(start_s or "0")
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                if start > end:
                    self.send_error(416)
                    return
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                with mp4_path.open("rb") as fh:
                    fh.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = fh.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with mp4_path.open("rb") as fh:
                while True:
                    chunk = fh.read(256 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def _send_play_page(self, session_id):
            manifest = load_manifest(args.record_dir, session_id)
            if manifest is None or manifest["frame_count"] == 0:
                self.send_error(404, "No such recording")
                return
            frame_count = manifest["frame_count"]
            source_fps = max(1.0, float(manifest.get("fps", 30)))
            default_play_fps = min(20.0, source_fps)
            interval = max(1, int(1000.0 / default_play_fps))
            html = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{session_id}</title>
<style>
html,body{{margin:0;background:#000;color:#fff;font:14px sans-serif;height:100%;overflow:hidden;}}
img{{width:100vw;height:calc(100vh - 76px);object-fit:contain;display:block;}}
.controls{{height:96px;padding:8px;background:#151515;box-sizing:border-box;}}
input[type=range]{{width:100%;}}
button,a{{font:16px sans-serif;margin-right:8px;color:#fff;}}
</style></head>
<body>
<img id="frame">
<div class="controls">
  <input id="slider" type="range" min="0" max="{frame_count - 1}" value="0">
  <div>
    <button id="play">play</button>
    <button id="pause">pause</button>
    <button id="fps15">15fps</button>
    <button id="fps20">20fps</button>
    <button id="fps30">30fps</button>
    <a href="/records">records</a>
    <a href="/">live</a>
    <span id="label"></span>
  </div>
</div>
<script>
const session = {json.dumps(session_id)};
const count = {frame_count};
let interval = {interval};
const frame = document.getElementById('frame');
const slider = document.getElementById('slider');
const label = document.getElementById('label');
let idx = 0;
let playing = false;
let loading = false;
function show(i) {{
  idx = Math.max(0, Math.min(count - 1, i));
  slider.value = idx;
  label.textContent = `${{idx + 1}} / ${{count}}`;
  loading = true;
  frame.src = `/record_frame?id=${{encodeURIComponent(session)}}&i=${{idx}}&t=${{Date.now()}}`;
}}
function pause() {{
  playing = false;
}}
function play() {{
  if (idx >= count - 1) show(0);
  playing = true;
  if (!loading) scheduleNext();
}}
function scheduleNext() {{
  if (!playing) return;
  if (idx >= count - 1) {{ pause(); return; }}
  setTimeout(() => {{
    if (!playing) return;
    show(idx + 1);
  }}, interval);
}}
function setFps(fps) {{
  interval = Math.max(1, Math.round(1000 / fps));
  label.textContent = `${{idx + 1}} / ${{count}} @ ${{fps}}fps`;
}}
slider.oninput = () => {{ pause(); show(Number(slider.value)); }};
frame.onload = () => {{
  loading = false;
  if (playing) scheduleNext();
}};
frame.onerror = () => {{
  loading = false;
  if (playing) setTimeout(scheduleNext, 200);
}};
document.getElementById('play').onclick = play;
document.getElementById('pause').onclick = pause;
document.getElementById('fps15').onclick = () => setFps(15);
document.getElementById('fps20').onclick = () => setFps(20);
document.getElementById('fps30').onclick = () => setFps(30);
show(0);
</script></body></html>"""
            self._send_html(html)

        def _send_record_frame(self, session_id, index):
            frame_path = Path(args.record_dir) / session_id / f"{index:06d}.jpg"
            if not frame_path.is_file():
                self.send_error(404, "No such frame")
                return
            jpeg = frame_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        def _send_json(self, value):
            data = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html):
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(min(length, 4096))
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video6")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=65)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--interval-ms", type=int, default=50)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-dir", default="records")
    parser.add_argument("--record-prefix", default="agx_web")
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--mp4-bitrate-kbps", type=int, default=1200)
    parser.add_argument("--test-name", default="run")
    parser.add_argument("--target-limit-cm", type=float, default=10.0)
    parser.add_argument("--control-socket", default="/tmp/ball_nx_control.sock")
    parser.add_argument("--control-timeout-ms", type=int, default=500)
    args = parser.parse_args()

    latest = LatestFrame()
    recorder = FrameRecorder(args)
    vision_state = NxControlClient(
        args.control_socket,
        args.target_limit_cm,
        max(1, args.control_timeout_ms) / 1000.0,
    )
    stop_event = threading.Event()
    worker = threading.Thread(target=capture_loop, args=(args, latest, recorder, stop_event), daemon=True)
    worker.start()

    class Server(ThreadingMixIn, TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server((args.host, args.port), make_handler(latest, recorder, vision_state, args))
    shutdown_started = threading.Event()

    def stop(_signum, _frame):
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"Serving latest-frame preview on http://{args.host}:{args.port}/")
    print(f"NX runtime task socket: {args.control_socket}")
    try:
        server.serve_forever()
    finally:
        recorder.stop()
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
