from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.deployment.package_raytheon_installed_usb_upgrade import (
    PACKET_NAME,
    RELEASE_VERSION,
    build_packet,
)

ROOT = Path(__file__).resolve().parents[2]


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    calibration = root / "calibrations" / "rer-u3-test"
    calibration.mkdir(parents=True)
    (root / "capture-setup.json").write_text('{"schema_version":"1.0"}')
    (root / "calibrations" / "registry.json").write_text('{"schema_version":"1.0"}')
    (calibration / "calibration.json").write_text('{"schema_version":"1.0"}')
    return root


def _table_code(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "permanent-table-code.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\npermanent-table-code")
    state = tmp_path / "permanent-table-code.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "code_kind": "permanent",
                "environment_id": "cloud1-public-example",
                "device_id": "raytheon-01",
                "table_code": "do-not-package-this-secret-code",
                "image_path": str(image),
            }
        )
    )
    return image, state


def _packet(tmp_path: Path) -> Path:
    image, state = _table_code(tmp_path)
    return build_packet(
        ROOT,
        tmp_path / "out",
        candidate_root=_candidate(tmp_path),
        table_code_image=image,
        table_code_state=state,
    )


def test_commercial_pilot_has_one_original_service_entrypoint(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    assert PACKET_NAME == "billiardsAI-raytheon-commercial-pilot-20260907"
    assert RELEASE_VERSION == "2026.09.07-commercial-pilot.23"
    assert packet.name == f"{PACKET_NAME}.zip"

    with zipfile.ZipFile(packet) as outer:
        prefix = f"{PACKET_NAME}/"
        outer_files = [name for name in outer.namelist() if not name.endswith("/")]
        manifest_bytes = outer.read(prefix + "installed-usb-upgrade-manifest.json")
        manifest = json.loads(manifest_bytes)
        release_payload = outer.read(prefix + manifest["release"]["path"])
        install = outer.read(prefix + "install.ps1").decode("utf-8-sig")
    release_path = tmp_path / "release.zip"
    release_path.write_bytes(release_payload)
    with zipfile.ZipFile(release_path) as release:
        release_files = set(release.namelist())
        launcher = release.read("scripts/algorithm/windows/serve-field-route-shell.ps1").decode(
            "utf-8-sig"
        )

    assert outer_files.count(prefix + "install.ps1") == 1
    assert not any("installed-usb-upgrade-20260907-v" in name for name in outer_files)
    assert manifest["service"]["port"] == 8787
    assert manifest["service"]["task_name"] == "billiardsAI-field-service"
    assert manifest["join"]["code_kind"] == "permanent"
    assert manifest["join"]["device_id"] == "raytheon-01"
    assert prefix + manifest["join"]["image_path"] in outer_files
    assert "do-not-package-this-secret-code" not in manifest_bytes.decode()
    assert "8765" not in install + launcher
    assert "serve_field_route_shell.py" not in release_files
    assert "CaptureProfilePath" in launcher
    assert "serve_field_route_shell.py" not in launcher
    forbidden = (
        "scripts/diagnostics/",
        "package_raytheon_",
        "raytheon_eight_ball_field",
        "run.ps1",
    )
    assert not any(any(token in name for token in forbidden) for name in release_files)


def test_installed_graph_performs_no_network_package_install(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    with zipfile.ZipFile(packet) as outer:
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(outer.read(prefix + "installed-usb-upgrade-manifest.json"))
        texts = [outer.read(prefix + "install.ps1").decode("utf-8-sig")]
        release_path = tmp_path / "release.zip"
        release_path.write_bytes(outer.read(prefix + manifest["release"]["path"]))
    with zipfile.ZipFile(release_path) as release:
        texts.extend(
            release.read(name).decode("utf-8-sig")
            for name in release.namelist()
            if name.endswith((".ps1", ".py"))
        )
    installed_graph = "\n".join(texts).lower()
    for token in (
        "pip install",
        "npm install",
        "winget install",
        "choco install",
        "invoke-webrequest -uri http",
    ):
        assert token not in installed_graph
