#!/usr/bin/env bash
# Install the approved Flutter SDK outside the Git working tree.
set -euo pipefail

readonly FLUTTER_VERSION="3.44.7"
readonly FLUTTER_CHANNEL="stable"
readonly HOST_OS="macos"
readonly HOST_ARCH="arm64"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cache-directory>" >&2
  exit 64
fi

cache_dir=$(python3 -c 'import os, sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' "$1")
repo_root=$(git rev-parse --show-toplevel)
repo_root=$(cd "$repo_root" && pwd -P)
if [[ "$cache_dir" == "$repo_root" || "$cache_dir" == "$repo_root/"* ]]; then
  echo "Flutter cache directory must be outside the repository: $cache_dir" >&2
  exit 64
fi

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) ;;
  *)
    echo "Unsupported Flutter host: $(uname -s)/$(uname -m); expected Darwin/arm64" >&2
    exit 65
    ;;
esac

readonly default_metadata_url="https://storage.googleapis.com/flutter_infra_release/releases/releases_macos.json"
metadata_url=${FLUTTER_RELEASES_URL:-$default_metadata_url}
manifest_path=${FLUTTER_RELEASE_MANIFEST:-"$repo_root/apps/mobile/flutter_release_manifest.json"}

if [[ ! -f "$manifest_path" ]]; then
  echo "Flutter release manifest is missing: $manifest_path" >&2
  exit 66
fi

mkdir -p "$cache_dir"
target_dir="$cache_dir/flutter-${FLUTTER_VERSION}-${HOST_OS}-${HOST_ARCH}"
binary_path="$target_dir/flutter/bin/flutter"
lock_dir="$cache_dir/.flutter-bootstrap.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "another Flutter bootstrap is already running (or a stale lock is present): $lock_dir" >&2
  exit 75
fi
work_dir=""
cleanup() {
  if [[ -n "$work_dir" ]]; then
    rm -rf "$work_dir"
  fi
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT
work_dir=$(mktemp -d "$cache_dir/.flutter-bootstrap.XXXXXX")

metadata_file="$work_dir/releases.json"
curl --fail --location --silent --show-error --retry 3 --retry-delay 1 "$metadata_url" --output "$metadata_file"

selection=$(python3 - "$metadata_file" "$manifest_path" "$metadata_url" <<'PY'
import json
import re
import sys
from urllib.parse import urljoin

metadata_path, manifest_path, metadata_url = sys.argv[1:]
with open(metadata_path, encoding="utf-8") as source:
    metadata = json.load(source)
with open(manifest_path, encoding="utf-8") as source:
    manifest = json.load(source)

expected = {
    "schema_version": 1,
    "flutter_version": "3.44.7",
    "channel": "stable",
    "host_os": "macos",
    "host_arch": "arm64",
    "archive": "stable/macos/flutter_macos_arm64_3.44.7-stable.zip",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"release manifest {key} is not the approved value")
if manifest.get("official_release_json_url") != metadata_url:
    raise SystemExit("official metadata URL does not match manifest")

records = [
    release
    for release in metadata.get("releases", [])
    if release.get("version") == manifest["flutter_version"]
    and release.get("channel") == manifest["channel"]
    and release.get("archive") == manifest["archive"]
]
if len(records) != 1:
    raise SystemExit("official metadata archive does not match manifest")

base_url = metadata.get("base_url")
if not isinstance(base_url, str):
    raise SystemExit("official metadata is missing base_url")
official_url = urljoin(f"{base_url.rstrip('/')}/", manifest["archive"])
if official_url != manifest.get("official_release_url"):
    raise SystemExit("official metadata archive URL does not match manifest")

release_hash = records[0].get("hash")
if not isinstance(release_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", release_hash):
    raise SystemExit("official metadata is missing a valid release hash")
if release_hash.lower() != manifest.get("official_release_hash", "").lower():
    raise SystemExit("official metadata release hash does not match manifest")
official_sha256 = records[0].get("sha256")
if not isinstance(official_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", official_sha256):
    raise SystemExit("official metadata is missing a valid SHA-256 hash")
if official_sha256.lower() != manifest.get("official_archive_sha256", "").lower():
    raise SystemExit("official metadata SHA-256 does not match manifest")

print(manifest["archive"])
print(manifest["official_release_url"])
print(official_sha256.lower())
PY
)
archive_name=$(printf '%s\n' "$selection" | sed -n '1p')
archive_url=$(printf '%s\n' "$selection" | sed -n '2p')
expected_sha256=$(printf '%s\n' "$selection" | sed -n '3p')

downloads_dir="$cache_dir/downloads"
mkdir -p "$downloads_dir"
archive_file="$downloads_dir/${archive_name##*/}"
archive_part="$archive_file.part"
if [[ -f "$archive_file" ]]; then
  archive_to_verify="$archive_file"
else
  curl --continue-at - --fail --location --silent --show-error --retry 3 --retry-delay 1 "$archive_url" --output "$archive_part"
  archive_to_verify="$archive_part"
fi

if command -v sha256sum >/dev/null 2>&1; then
  actual_sha256=$(sha256sum "$archive_to_verify" | awk '{print $1}')
else
  actual_sha256=$(shasum -a 256 "$archive_to_verify" | awk '{print $1}')
fi
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  rm -f "$archive_part"
  echo "official SHA-256 verification failed for $archive_name" >&2
  exit 66
fi

if [[ "$archive_to_verify" == "$archive_part" ]]; then
  mv "$archive_part" "$archive_file"
fi

staging_dir="$work_dir/extracted"
mkdir "$staging_dir"
unzip -q "$archive_file" -d "$staging_dir"
if [[ ! -x "$staging_dir/flutter/bin/flutter" ]]; then
  echo "Flutter archive did not contain flutter/bin/flutter" >&2
  exit 66
fi
replacement_dir="$cache_dir/.flutter-sdk-replacement.$$"
previous_dir="$cache_dir/.flutter-sdk-previous.$$"
mv "$staging_dir" "$replacement_dir"
if [[ -e "$target_dir" ]]; then
  mv "$target_dir" "$previous_dir"
fi
mv "$replacement_dir" "$target_dir"
rm -rf "$previous_dir"
printf '%s\n' "$binary_path"
