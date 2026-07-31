#!/usr/bin/env bash
set -uo pipefail

ENV_FILE="${ENV_FILE:-/etc/ball-car/ball-car.env}"
HOTSPOT_NAME="${HOTSPOT_NAME:-agx-336l-hotspot}"
HOTSPOT_ADDRESS="${HOTSPOT_ADDRESS:-192.168.88.1}"
failures=0

ok() { printf 'OK    %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

if [ -r "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
else
    bad "missing $ENV_FILE"
fi
WEB_PORT="${WEB_PORT:-8080}"
WEB_HOST="${WEB_HOST:-$HOTSPOT_ADDRESS}"

for service in ball-nx-control.service ball-vision-web.service; do
    if systemctl is-active --quiet "$service"; then
        ok "$service active"
    else
        bad "$service is not active"
        systemctl --no-pager --full status "$service" 2>/dev/null | tail -n 8
    fi
done

active_connections="$(nmcli -t -f NAME connection show --active 2>/dev/null)"
if grep -Fxq "$HOTSPOT_NAME" <<<"$active_connections"; then
    ok "hotspot $HOTSPOT_NAME active"
else
    bad "hotspot $HOTSPOT_NAME is not active"
fi

if [ -S /run/ball-nx/control.sock ]; then
    ok "runtime control socket ready"
else
    bad "missing /run/ball-nx/control.sock"
fi

if [ -n "${DMMC_DEVICE:-}" ] && [ -e "$DMMC_DEVICE" ]; then
    ok "DMMC present: $DMMC_DEVICE"
else
    bad "DMMC device missing: ${DMMC_DEVICE:-not configured}"
fi

if lsusb 2>/dev/null | grep -Eiq '2bc5|Orbbec'; then
    ok "Orbbec USB device present"
else
    bad "Orbbec USB device not found"
fi

health_url="http://${WEB_HOST}:${WEB_PORT}/healthz"
status_url="http://${WEB_HOST}:${WEB_PORT}/api/vision/status"
if health_json="$(curl --max-time 2 -fsS "$health_url")"; then
    echo "$health_json"
    if grep -q '"frame_available": true' <<<"$health_json"; then
        ok "preview frame is available"
    else
        bad "preview endpoint responds but has no camera frame"
    fi
else
    bad "preview endpoint failed: $health_url"
fi
if control_json="$(curl --max-time 2 -fsS "$status_url")"; then
    echo "$control_json"
    if grep -q '"controller_online": true' <<<"$control_json"; then
        ok "controller API is online"
    else
        bad "controller API responds but controller is offline"
    fi
else
    bad "controller API failed: $status_url"
fi

if [ "$failures" -eq 0 ]; then
    echo "READY: connect the phone and open http://${WEB_HOST}:${WEB_PORT}/"
    exit 0
fi
echo "NOT READY: $failures check(s) failed"
exit 1
