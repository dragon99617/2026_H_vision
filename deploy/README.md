# Ball Car Power-On Deployment

This deployment makes the Jetson create its own Wi-Fi hotspot and start the
controller, Orbbec RGB inference, shared preview and task HTTP API after every
boot. Only `run_rgb.py` opens the camera; the web preview reuses its latest
RGB frame. Production control uses the validated `rgb-contour` coordinate
pipeline and does not depend on the unreliable tube depth fit.

## Safety and scope

- Power-on state is `idle`. Selecting task 3/4/5/6 starts that task.
- The web page controls the ball/tube tasks. It does not drive chassis wheels.
- Keep an independent hardware emergency stop. Wi-Fi and the web stop button
  are not safety-rated.
- The HTTP API has no separate login. Use a new WPA2 hotspot password and do
  not bridge the control port to an untrusted network. The installed service
  binds only to the hotspot address, not to Ethernet or phone-tethering links.

## One-time installation

Prefer connecting the DMMC and Orbbec first. If DMMC is present, the installer
automatically selects its stable `/dev/serial/by-id/...` name. Without hardware
it installs a temporary `/dev/ttyACM0` fallback; re-run the installer with the
stable name before acceptance. Build and test the controller, then install:

```bash
cd /home/d/2026ti/2026H/ball_yolo26s_nx/nx_control
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
(cd build && ctest --output-on-failure)

cd /home/d/2026ti/2026H/ball_yolo26s_nx
export HOTSPOT_PASSWORD='replace-with-a-new-12-character-password'
export DMMC_DEVICE='/dev/serial/by-id/replace-with-the-real-device'
./deploy/install_on_car.sh
sudo reboot
```

The installer does not switch the active Wi-Fi immediately. Set
`ACTIVATE_HOTSPOT_NOW=1` only when it is acceptable to disconnect the current
SSH/Wi-Fi session. `START_NOW=1` starts the services without waiting for reboot.
Use `DRY_RUN=1` to render and validate the configuration without changing the
system. Installation disables the legacy `ball-vision.service` so only the
combined `ball-vision-web.service` can own the Orbbec.

## Normal use

1. Power on the car and wait for the Jetson to boot.
2. Connect the phone to `AGX-336L` with the password chosen during install.
3. Open `http://192.168.88.1:8080/`.
4. Select task 3/4/5/6. Task selection starts immediately.

The same API can be called directly:

```bash
curl http://192.168.88.1:8080/api/vision/status
curl -X POST -H 'Content-Type: application/json' \
  -d '{"task":"3"}' \
  http://192.168.88.1:8080/api/vision/task
curl -X POST http://192.168.88.1:8080/api/vision/stop
```

## Acceptance and diagnostics

```bash
./deploy/car_health_check.sh
journalctl -u ball-nx-control -u ball-vision-web -b --no-pager
```

Before competition, perform at least ten cold power-cycle tests and verify:

- hotspot and page appear without SSH or keyboard intervention;
- controller starts in `idle`;
- image, vision health and DMMC health are visible;
- each task starts once, stop returns to zero angle, and reset selects task 5;
- unplug/replug recovery works for both DMMC and Orbbec;
- the physical emergency stop still works when Wi-Fi is disconnected.
