#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_version="2.8.6"
deb_name="OrbbecSDK_v${sdk_version}_arm64.deb"
download_dir="$project_dir/.cache/orbbec"
extract_dir="$download_dir/extracted"
target_dir="$project_dir/third_party/orbbec_sdk"
url="https://github.com/orbbec/OrbbecSDK_v2/releases/download/v${sdk_version}/${deb_name}"

mkdir -p "$download_dir" "$project_dir/third_party"
if [[ ! -f "$download_dir/$deb_name" ]]; then
    curl -fL --retry 3 "$url" -o "$download_dir/$deb_name"
fi
rm -rf "$extract_dir"
mkdir -p "$extract_dir"
dpkg-deb -x "$download_dir/$deb_name" "$extract_dir"
rm -rf "$target_dir"
mv "$extract_dir/opt/OrbbecSDK_v${sdk_version}" "$target_dir"

bash "$project_dir/tools/build_orbbec_bridge.sh"
echo "Orbbec SDK $sdk_version installed locally at $target_dir"
