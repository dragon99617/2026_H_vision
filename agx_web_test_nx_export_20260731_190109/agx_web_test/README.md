# AGX Camera Web Test

Use this folder when the Orbbec Gemini 336L is connected directly to the AGX and
you want to view the camera from a phone browser.

This folder can also be copied to the Orin NX later. The rule is simple: the
machine connected to the camera runs this web service, and the phone opens that
machine's IP address.

> This is a standalone camera test tool. Do not autostart it together with the
> production `run_rgb.py` service because both processes would open the same
> Orbbec. The installed-car service uses `run_rgb.py --web-port 8080`, which
> shares the frame already owned by the inference runtime.

## Start

1. Start the AGX hotspot:

   ```bash
   ./scripts/start_hotspot.sh
   ```

2. Connect the phone to:

   ```text
   Wi-Fi: AGX-336L
   Password: agx336l888
   ```

3. On the NX, start `ball_nx_control` first. The web page and controller must
   use the same local command socket:

   ```bash
   cd /home/orin/2026ti/ball_yolo26s_nx/nx_control
   ./build/ball_nx_control \
     --config config/nx-control.conf \
     --dmmc /dev/ttyACM0 \
     --task 5 \
     --command-socket /tmp/ball_nx_control.sock \
     --log logs/live.csv
   ```

4. In another terminal, start the web preview:

   ```bash
   WIDTH=424 HEIGHT=240 FPS=60 JPEG_QUALITY=55 SNAPSHOT_INTERVAL_MS=17 RECORD_FPS=60 MP4_BITRATE_KBPS=1200 TEST_NAME=req5_run01 ./scripts/run_web_snapshot.sh
   ```

5. Open this on the phone:

   ```text
   http://192.168.88.1:8080/
   ```

If the AGX is connected to a phone hotspot instead of running its own hotspot,
open the AGX wlan IP instead, for example:

```text
http://192.168.43.9:8080/
```

## Vision Task Buttons

The live page has buttons for tasks 3, 4, 5 and 6:

```text
任务3  任务4  任务5  任务6  开始  停止  复位
```

The buttons now control the running `ball_nx_control` process through the local
Unix datagram socket `/tmp/ball_nx_control.sock`. The C++ controller applies a
command on a 50 Hz control-cycle boundary, then returns its actual task and
safety state to the page. The web service does not kill or restart the control
process.

The browser uses these HTTP endpoints:

```text
GET  /api/vision/status
POST /api/vision/task
POST /api/vision/start
POST /api/vision/stop
POST /api/vision/reset
POST /api/vision/target
```

Example task 6 request:

```json
{"task": "6", "target_cm": -7.3}
```

Example response:

```json
{
  "ok": true,
  "request_id": "1",
  "requested_task": "6",
  "active_task": "6",
  "control_mode": "hold_target",
  "target_cm": -7.3,
  "running": true,
  "applied": true,
  "safety_latched": false,
  "message": "任务6已启动，目标位置 -7.3 cm"
}
```

Suggested mapping:

```text
task 3 -> control_mode swing_test
task 4 -> control_mode hold_center, target_cm 0.0
task 5 -> control_mode hold_center, target_cm 0.0
task 6 -> control_mode hold_target, target_cm from the web slider
```

Task 4 and task 5 intentionally use the same center-hold control logic, but the
response preserves which button was selected. Task 6 rejects a missing target
or a target outside `[-10, +10] cm`.

Button behavior:

- selecting task 3/4/5/6 switches and immediately starts that task;
- `开始` restarts the selected task;
- `停止` stops the selected task and commands zero beam angle. Task 3 reports
  `standby_hold` on the wire so the mechanism stays enabled at 0 degrees; the
  other tasks report `idle`;
- `复位` selects task 5 and leaves it stopped until `开始` is pressed;
- the status line reports NX online/offline, vision and DMMC health, active
  task, target, SAFE latch, and the controller's message.

The socket path and web wait timeout can be overridden when starting the web
server:

```bash
CONTROL_SOCKET=/tmp/ball_nx_control.sock CONTROL_TIMEOUT_MS=500 \
  ./scripts/run_web_snapshot.sh
```

## Stop

For the most convenient playback flow, do not press `Ctrl-C` immediately.
On the phone live page, tap `stop recording`. Wait until the AGX terminal prints
`MP4 ready`. The web server keeps running, so you can open `/records` and replay
with the browser's native video progress bar.

Press `Ctrl-C` only when you want to shut down the web server.

To stop the hotspot:

```bash
./scripts/stop_hotspot.sh
```

## If It Is Too Laggy

For phone hotspots, the steadiest browser mode is usually latest-frame polling:

```bash
WIDTH=424 HEIGHT=240 FPS=60 JPEG_QUALITY=55 SNAPSHOT_INTERVAL_MS=17 RECORD_FPS=60 MP4_BITRATE_KBPS=1200 ./scripts/run_web_snapshot.sh
```

This drops old frames instead of letting the browser build up a delayed MJPEG
queue.

Use the low-latency script first:

```bash
./scripts/run_web_preview_low_latency.sh
```

It tries to send the camera's MJPEG frames directly to the browser. If that
direct V4L2 path is not compatible with the current camera node, it falls back
to an OpenCV low-buffer path automatically.

Try a lower mode:

```bash
WIDTH=1280 HEIGHT=720 FPS=30 ./scripts/run_web_preview_low_latency.sh
```

If the low-latency script has compatibility trouble, fall back to the OpenCV
script:

```bash
./scripts/run_web_preview.sh
```

## Camera Node

The default camera node is `/dev/video6`. If the AGX lists the Gemini 336L on a
different node, edit `config.env`:

```text
CAMERA_DEVICE=/dev/video2
```

## Recordings

`run_web_snapshot.sh` records by default into `records/`.

List recordings:

```bash
./scripts/list_recordings.sh
```

Open web playback on the phone. Converted recordings open as normal MP4 videos
with play/pause and a draggable progress bar:

```text
http://<AGX_IP>:8080/records
```

Disable recording for a quick preview:

```bash
RECORD=0 ./scripts/run_web_snapshot.sh
```

## Export To NX

Create a clean package without local recordings:

```bash
cd /home/orin/2026ti/agx_web_test
chmod +x ./scripts/export_for_nx.sh
./scripts/export_for_nx.sh
```

The script prints the package path, like:

```text
/home/orin/2026ti/agx_web_test_nx_export_20260731_120000.tar.gz
```

Copy that file to the NX. For example, replace `NX_IP` with the NX address:

```bash
scp /home/orin/2026ti/agx_web_test_nx_export_*.tar.gz orin@NX_IP:/home/orin/2026ti/
```

On the NX:

```bash
cd /home/orin/2026ti
tar -xzf agx_web_test_nx_export_*.tar.gz
cd agx_web_test
WIDTH=424 HEIGHT=240 FPS=60 JPEG_QUALITY=55 SNAPSHOT_INTERVAL_MS=17 ./scripts/run_web_snapshot.sh
```
