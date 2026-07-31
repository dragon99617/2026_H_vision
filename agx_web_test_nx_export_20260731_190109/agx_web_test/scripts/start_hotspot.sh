#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config.env"

CON_NAME="${CON_NAME:-agx-336l-hotspot}"
IFACE="${IFACE:-wlan0}"
SSID="${SSID:-$HOTSPOT_SSID}"
PASSWORD="${PASSWORD:-$HOTSPOT_PASSWORD}"
ADDRESS="${ADDRESS:-$HOTSPOT_ADDRESS}"
CHANNEL="${CHANNEL:-$HOTSPOT_CHANNEL}"

if [ "${#PASSWORD}" -lt 8 ]; then
    echo "PASSWORD must be at least 8 characters for WPA/WPA2."
    exit 2
fi

echo "Starting AGX hotspot for phone preview:"
echo "  interface: $IFACE"
echo "  ssid:      $SSID"
echo "  password:  $PASSWORD"
echo "  address:   $ADDRESS"
echo "  channel:   $CHANNEL"
echo
echo "This will disconnect $IFACE from its current Wi-Fi network."

if nmcli connection show "$CON_NAME" >/dev/null 2>&1; then
    sudo nmcli connection modify "$CON_NAME" \
        connection.interface-name "$IFACE" \
        802-11-wireless.ssid "$SSID" \
        802-11-wireless.mode ap \
        802-11-wireless.band a \
        802-11-wireless.channel "$CHANNEL" \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.psk "$PASSWORD" \
        ipv4.method shared \
        ipv4.addresses "$ADDRESS" \
        ipv6.method ignore \
        connection.autoconnect no
else
    sudo nmcli connection add type wifi ifname "$IFACE" con-name "$CON_NAME" ssid "$SSID"
    sudo nmcli connection modify "$CON_NAME" \
        802-11-wireless.mode ap \
        802-11-wireless.band a \
        802-11-wireless.channel "$CHANNEL" \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.psk "$PASSWORD" \
        ipv4.method shared \
        ipv4.addresses "$ADDRESS" \
        ipv6.method ignore \
        connection.autoconnect no
fi

sudo nmcli connection up "$CON_NAME"

echo
echo "Hotspot is up."
echo "Connect phone to '$SSID'."
echo "Preview URL: http://${ADDRESS%/*}:8080/"
