#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_ROOT="${APP_ROOT:-$REPO_ROOT}"
CAR_USER="${CAR_USER:-${SUDO_USER:-$USER}}"
WIFI_IFACE="${WIFI_IFACE:-wlan0}"
HOTSPOT_NAME="${HOTSPOT_NAME:-agx-336l-hotspot}"
HOTSPOT_SSID="${HOTSPOT_SSID:-AGX-336L}"
HOTSPOT_PASSWORD="${HOTSPOT_PASSWORD:-}"
HOTSPOT_ADDRESS="${HOTSPOT_ADDRESS:-192.168.88.1/24}"
HOTSPOT_CHANNEL="${HOTSPOT_CHANNEL:-149}"
WEB_PORT="${WEB_PORT:-8080}"
WEB_JPEG_QUALITY="${WEB_JPEG_QUALITY:-70}"
WEB_INTERVAL_MS="${WEB_INTERVAL_MS:-50}"
WEB_PREVIEW_WIDTH="${WEB_PREVIEW_WIDTH:-640}"
START_NOW="${START_NOW:-0}"
ACTIVATE_HOTSPOT_NOW="${ACTIVATE_HOTSPOT_NOW:-0}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [ "$DRY_RUN" = "1" ]; then
    ROOT=()
elif [ "$EUID" -eq 0 ]; then
    ROOT=()
else
    ROOT=(sudo)
    sudo -v
fi

id "$CAR_USER" >/dev/null 2>&1 || fail "Linux user does not exist: $CAR_USER"
[ -x "$APP_ROOT/nx_control/build/ball_nx_control" ] || \
    fail "missing controller binary; build nx_control first"
[ -f "$APP_ROOT/run.py" ] || fail "run.py not found below APP_ROOT=$APP_ROOT"
[ -f "$APP_ROOT/models/default.json" ] || fail "models/default.json is missing"
[ -f "$APP_ROOT/native/liborbbec_bridge.so" ] || \
    fail "Orbbec bridge is missing; run tools/setup_orbbec_sdk.sh"

case "$APP_ROOT" in
    *' '*|*'|'*|*$'\n'*) fail "APP_ROOT must not contain spaces, pipes or newlines" ;;
esac
case "$CAR_USER" in
    *[!a-zA-Z0-9_-]*) fail "CAR_USER contains unsupported characters" ;;
esac

if [ -z "${DMMC_DEVICE:-}" ]; then
    candidates=()
    for candidate in /dev/serial/by-id/*; do
        [ -L "$candidate" ] && candidates+=("$candidate")
    done
    if [ "${#candidates[@]}" -eq 1 ]; then
        DMMC_DEVICE="${candidates[0]}"
    else
        DMMC_DEVICE="/dev/ttyACM0"
        echo "WARNING: DMMC is not connected; temporarily using /dev/ttyACM0." >&2
        echo "Re-run this installer with DMMC_DEVICE=/dev/serial/by-id/... before acceptance." >&2
    fi
fi
case "$DMMC_DEVICE" in
    /dev/serial/by-id/*|/dev/ttyACM*|/dev/ttyUSB*) ;;
    *) fail "DMMC_DEVICE must be a serial device below /dev" ;;
esac
if [ ! -e "$DMMC_DEVICE" ]; then
    echo "WARNING: DMMC device is currently absent: $DMMC_DEVICE" >&2
fi

[ "${#HOTSPOT_PASSWORD}" -ge 12 ] || \
    fail "set HOTSPOT_PASSWORD to a new password of at least 12 characters"
[[ "$WEB_PORT" =~ ^[0-9]+$ ]] && [ "$WEB_PORT" -ge 1 ] && [ "$WEB_PORT" -le 65535 ] || \
    fail "WEB_PORT must be in 1..65535"
[[ "$WEB_JPEG_QUALITY" =~ ^[0-9]+$ ]] && [ "$WEB_JPEG_QUALITY" -ge 1 ] && [ "$WEB_JPEG_QUALITY" -le 100 ] || \
    fail "WEB_JPEG_QUALITY must be in 1..100"
[[ "$WEB_INTERVAL_MS" =~ ^[0-9]+$ ]] && [ "$WEB_INTERVAL_MS" -ge 10 ] && [ "$WEB_INTERVAL_MS" -le 1000 ] || \
    fail "WEB_INTERVAL_MS must be in 10..1000"
[[ "$WEB_PREVIEW_WIDTH" =~ ^[0-9]+$ ]] && [ "$WEB_PREVIEW_WIDTH" -ge 160 ] && [ "$WEB_PREVIEW_WIDTH" -le 1920 ] || \
    fail "WEB_PREVIEW_WIDTH must be in 160..1920"

nm_devices="$(nmcli -t -f DEVICE,TYPE device status)"
grep -q "^${WIFI_IFACE}:wifi$" <<<"$nm_devices" || \
    fail "Wi-Fi interface not managed by NetworkManager: $WIFI_IFACE"
nm_wifi="$(nmcli -f WIFI-PROPERTIES.AP device show "$WIFI_IFACE")"
grep -q 'yes' <<<"$nm_wifi" || \
    fail "Wi-Fi interface does not support AP mode: $WIFI_IFACE"

tmp_dir="$(mktemp -d /tmp/ball-car-install.XXXXXX)"
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

render_unit() {
    local source="$1"
    local output="$2"
    sed \
        -e "s|@APP_ROOT@|$APP_ROOT|g" \
        -e "s|@CAR_USER@|$CAR_USER|g" \
        "$source" >"$output"
}

render_unit "$SCRIPT_DIR/systemd/ball-nx-control.service.in" \
    "$tmp_dir/ball-nx-control.service"
render_unit "$SCRIPT_DIR/systemd/ball-vision-web.service.in" \
    "$tmp_dir/ball-vision-web.service"

printf '%s\n' \
    "DMMC_DEVICE=$DMMC_DEVICE" \
    "WEB_HOST=${HOTSPOT_ADDRESS%/*}" \
    "WEB_PORT=$WEB_PORT" \
    "WEB_JPEG_QUALITY=$WEB_JPEG_QUALITY" \
    "WEB_INTERVAL_MS=$WEB_INTERVAL_MS" \
    "WEB_PREVIEW_WIDTH=$WEB_PREVIEW_WIDTH" \
    >"$tmp_dir/ball-car.env"

if [ "$DRY_RUN" = "1" ]; then
    grep -R '@APP_ROOT@\|@CAR_USER@' "$tmp_dir" && \
        fail "unresolved systemd template placeholder"
    echo "DRY RUN: rendered deployment successfully"
    sed -n '1,220p' "$tmp_dir/ball-nx-control.service"
    sed -n '1,220p' "$tmp_dir/ball-vision-web.service"
    sed -n '1,80p' "$tmp_dir/ball-car.env"
    exit 0
fi

"${ROOT[@]}" install -d -m 0755 /etc/ball-car
"${ROOT[@]}" install -m 0644 "$tmp_dir/ball-car.env" /etc/ball-car/ball-car.env
"${ROOT[@]}" install -m 0644 "$tmp_dir/ball-nx-control.service" \
    /etc/systemd/system/ball-nx-control.service
"${ROOT[@]}" install -m 0644 "$tmp_dir/ball-vision-web.service" \
    /etc/systemd/system/ball-vision-web.service

udev_rules="$APP_ROOT/third_party/orbbec_sdk/shared/99-obsensor-libusb.rules"
if [ -f "$udev_rules" ]; then
    "${ROOT[@]}" install -m 0644 "$udev_rules" \
        /etc/udev/rules.d/99-obsensor-libusb.rules
    "${ROOT[@]}" udevadm control --reload-rules
fi

"${ROOT[@]}" usermod -a -G dialout,video "$CAR_USER"

if "${ROOT[@]}" nmcli connection show "$HOTSPOT_NAME" >/dev/null 2>&1; then
    :
else
    "${ROOT[@]}" nmcli connection add type wifi ifname "$WIFI_IFACE" \
        con-name "$HOTSPOT_NAME" ssid "$HOTSPOT_SSID"
fi
"${ROOT[@]}" nmcli connection modify "$HOTSPOT_NAME" \
    connection.interface-name "$WIFI_IFACE" \
    connection.autoconnect yes \
    connection.autoconnect-priority 100 \
    802-11-wireless.ssid "$HOTSPOT_SSID" \
    802-11-wireless.mode ap \
    802-11-wireless.band a \
    802-11-wireless.channel "$HOTSPOT_CHANNEL" \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$HOTSPOT_PASSWORD" \
    ipv4.method shared \
    ipv4.addresses "$HOTSPOT_ADDRESS" \
    ipv6.method ignore

"${ROOT[@]}" systemctl daemon-reload
"${ROOT[@]}" systemctl disable --now ball-vision.service >/dev/null 2>&1 || true
"${ROOT[@]}" systemctl enable ball-nx-control.service ball-vision-web.service

if command -v ufw >/dev/null 2>&1; then
    ufw_status="$("${ROOT[@]}" ufw status)"
    if grep -q '^Status: active' <<<"$ufw_status"; then
        "${ROOT[@]}" ufw allow in on "$WIFI_IFACE" to any port "$WEB_PORT" proto tcp
    fi
fi

if [ "$START_NOW" = "1" ]; then
    "${ROOT[@]}" systemctl restart ball-nx-control.service ball-vision-web.service
fi
if [ "$ACTIVATE_HOTSPOT_NOW" = "1" ]; then
    echo "Activating the AGX hotspot now; the current Wi-Fi connection will disconnect."
    "${ROOT[@]}" nmcli connection up "$HOTSPOT_NAME"
fi

echo
echo "Ball car deployment installed."
echo "  user:       $CAR_USER"
echo "  app:        $APP_ROOT"
echo "  DMMC:       $DMMC_DEVICE"
echo "  hotspot:    $HOTSPOT_SSID"
echo "  control UI: http://${HOTSPOT_ADDRESS%/*}:$WEB_PORT/"
echo
echo "Reboot to apply group membership and hotspot autoconnect: sudo reboot"
echo "After reboot run: $SCRIPT_DIR/car_health_check.sh"
