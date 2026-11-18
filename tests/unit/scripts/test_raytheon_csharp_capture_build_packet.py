from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.diagnostics.package_raytheon_csharp_capture_build import build_packet


def test_packet_builds_without_external_windows_toolchain(tmp_path: Path) -> None:
    archive = build_packet(Path.cwd(), tmp_path)

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest_name = next(name for name in names if name.endswith("packet-manifest.json"))
        root = manifest_name.removesuffix("packet-manifest.json")
        manifest = json.loads(bundle.read(manifest_name))

        assert manifest["kind"] == "raytheon_csharp_capture_build_packet"
        assert manifest["baseline_commit"].startswith("9a14495")
        assert manifest["production_ready"] is False
        assert manifest["mutates_service_or_production_data"] is False
        assert root + "run.ps1" in names
        assert root + "native/windows/mainline1_uvc_capture_csharp/main.cs" in names
        assert root + "native/windows/mainline1_uvc_capture/protocol.md" in names
        assert archive.name == "raytheon-csharp-capture-build-20260906-v1.zip"
