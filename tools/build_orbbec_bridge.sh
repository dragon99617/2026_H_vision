#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_dir="${ORBBEC_SDK_DIR:-$project_dir/third_party/orbbec_sdk}"

if [[ ! -f "$sdk_dir/include/libobsensor/ObSensor.hpp" ||
      ! -f "$sdk_dir/lib/libOrbbecSDK.so" ]]; then
    echo "Orbbec SDK not found at $sdk_dir" >&2
    echo "Run: bash tools/setup_orbbec_sdk.sh" >&2
    exit 1
fi

g++ -std=c++14 -O3 -fPIC -shared -pthread \
  "$project_dir/native/orbbec_bridge.cpp" \
  -I"$sdk_dir/include" \
  -L"$sdk_dir/lib" -lOrbbecSDK \
  -Wl,-rpath,'$ORIGIN/../third_party/orbbec_sdk/lib' \
  -o "$project_dir/native/liborbbec_bridge.so"

echo "built $project_dir/native/liborbbec_bridge.so"
