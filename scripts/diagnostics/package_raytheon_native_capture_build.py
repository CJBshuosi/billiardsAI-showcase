#!/usr/bin/env python3
"""Build the immutable Raytheon native-capture compilation probe packet."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_packet(project_root: Path, output_root: Path) -> Path:
    project = project_root.resolve()
    destination = output_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    packet_name = "raytheon-native-capture-build-20260906-v1"
    with tempfile.TemporaryDirectory(prefix="raytheon-native-capture-build-") as directory:
        staging = Path(directory) / packet_name
        staging.mkdir()
        sources = {
            project / "scripts/diagnostics/raytheon_native_capture_build/run.ps1": staging
            / "run.ps1",
            project / "scripts/diagnostics/raytheon_native_capture_build/README.txt": staging
            / "README.txt",
            project / "native/windows/mainline1_uvc_capture/CMakeLists.txt": staging
            / "native/windows/mainline1_uvc_capture/CMakeLists.txt",
            project / "native/windows/mainline1_uvc_capture/main.cpp": staging
            / "native/windows/mainline1_uvc_capture/main.cpp",
            project / "native/windows/mainline1_uvc_capture/protocol.md": staging
            / "native/windows/mainline1_uvc_capture/protocol.md",
        }
        for source, target in sources.items():
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"packet source is missing or unsafe: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        inventory = {
            path.relative_to(staging).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "1.0",
            "kind": "raytheon_native_capture_build_packet",
            "packet_name": packet_name,
            "baseline_commit": "9a144955377b289ce47de71769d1305dd8585de4",
            "production_ready": False,
            "mutates_service_or_production_data": False,
            "inventory": inventory,
        }
        (staging / "packet-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        archive = destination / f"{packet_name}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    bundle.write(path, f"{packet_name}/{path.relative_to(staging).as_posix()}")
    return archive


def main() -> int:
    project = Path(__file__).resolve().parents[2]
    archive = build_packet(project, project / "output/field-packages")
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
