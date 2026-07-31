#!/usr/bin/env bash
set -euo pipefail

CON_NAME="${CON_NAME:-agx-336l-hotspot}"

sudo nmcli connection down "$CON_NAME"
echo "Hotspot stopped: $CON_NAME"
