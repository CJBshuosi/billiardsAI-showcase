from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.deployment.package_raytheon_installed_usb_upgrade import (
    PACKET_NAME,
    build_packet,
)

ROOT = Path(__file__).resolve().parents[3]


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    calibration = root / "calibrations" / "rer-u3-test"
    native = root / "native" / "windows" / "mainline1_uvc_capture_csharp"
    calibration.mkdir(parents=True)
    native.mkdir(parents=True)
    (root / "capture-setup.json").write_text('{"schema_version":"1.0"}')
    (root / "calibrations" / "registry.json").write_text('{"schema_version":"1.0"}')
    (calibration / "calibration.json").write_text('{"schema_version":"1.0"}')
    (native / "main.cs").write_text("untrusted runtime compiler input")
    (root / "join-qr.png").write_bytes(b"expired-sidecar-qr")
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


def _packet(tmp_path: Path, *, candidate_archive: Path | None = None) -> Path:
    image, state = _table_code(tmp_path)
    source = (
        {"candidate_archive": candidate_archive}
        if candidate_archive is not None
        else {"candidate_root": _candidate(tmp_path)}
    )
    return build_packet(
        ROOT,
        tmp_path / "output",
        table_code_image=image,
        table_code_state=state,
        **source,
    )


def test_packet_rejects_a_development_preview_table_code(tmp_path: Path) -> None:
    image, state = _table_code(tmp_path)
    metadata = json.loads(state.read_text())
    metadata["code_kind"] = "development_preview"
    state.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="permanent Raytheon table code"):
        build_packet(
            ROOT,
            tmp_path / "output",
            candidate_root=_candidate(tmp_path),
            table_code_image=image,
            table_code_state=state,
        )


def test_field_test_packet_can_carry_an_explicitly_labelled_development_preview(
    tmp_path: Path,
) -> None:
    image, state = _table_code(tmp_path)
    metadata = json.loads(state.read_text())
    metadata["code_kind"] = "development_preview"
    state.write_text(json.dumps(metadata))

    packet = build_packet(
        ROOT,
        tmp_path / "output",
        candidate_root=_candidate(tmp_path),
        table_code_image=image,
        table_code_state=state,
        allow_development_preview=True,
    )

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(archive.read(prefix + "installed-usb-upgrade-manifest.json"))
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")
    assert manifest["join"]["code_kind"] == "development_preview"
    assert manifest["join"]["launch_gate_status"] == "FAIL"
    assert manifest["join"]["field_test_only"] is True
    assert '"development_preview"' in install
    assert "field_test_only" in install


def test_packet_upgrades_original_service_without_sidecar_and_with_permanent_qr(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        names = set(archive.namelist())
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(archive.read(prefix + "installed-usb-upgrade-manifest.json"))
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")
        release_name = prefix + manifest["release"]["path"]
        release_bytes = archive.read(release_name)

    assert prefix + "install.ps1" in names
    assert not any(name.endswith("/run.ps1") for name in names)
    assert not any("join-qr" in name for name in names)
    assert prefix + "join/table-code.png" in names
    assert packet.name == "billiardsAI-raytheon-commercial-pilot-20260907.zip"
    assert manifest["kind"] == "commercial_pilot_field_service"
    assert manifest["service"] == {
        "task_name": "billiardsAI-field-service",
        "kiosk_task_name": "billiardsAI-field-kiosk",
        "port": 8787,
        "runtime_root": r"%LOCALAPPDATA%\billiardsAI\field-runtime",
    }
    assert manifest["capture"]["device_fingerprint"] == (
        "865a64805fad161c4f09aeb35214fa23b061be0d8a08c9e6fc99f39c86481a93"
    )
    assert manifest["capture"]["approved_native_helper_sha256"] == (
        "760839bde41fc5f16c4d73e14c24c35e67dfdb5aefdd7e141112134c6aa7ba7c"
    )
    assert "native_source" not in manifest["capture"]
    assert not any("mainline1_uvc_capture_csharp" in name for name in names)
    assert manifest["cloud"] == {
        "development_environment_id": "cloud1-public-example",
        "production_writes": False,
    }
    assert manifest["version"] == "2026.09.07-commercial-pilot.23"
    assert manifest["release"]["path"].endswith("commercial-pilot.23.zip")
    assert manifest["state_policy"] == {
        "on_version_upgrade": "archive_and_start_new_session",
        "before_archive": "retire_cloud_session_fail_closed",
        "archive_root": r"%LOCALAPPDATA%\billiardsAI\field-runtime\backups",
    }
    assert manifest["join"]["code_kind"] == "permanent"
    assert manifest["join"]["environment_id"] == "cloud1-public-example"
    assert manifest["join"]["device_id"] == "raytheon-01"
    assert manifest["join"]["image_path"] == "join/table-code.png"
    assert "table_code" not in manifest["join"]
    assert manifest["assets"]["identity"]["gate_status"] == "PASS"
    assert manifest["assets"]["identity"]["model_generation"] == "v4"
    assert manifest["assets"]["identity"]["approval_basis"] == (
        "commercial_candidate_owner_approved"
    )
    assert manifest["assets"]["identity"]["benchmark_gate_status"] == "FAIL"
    assert "8765" not in install
    assert "8787" in install
    assert "current.json" in install
    assert "CaptureProfile" in install
    assert hashlib.sha256(release_bytes).hexdigest() == manifest["release"]["sha256"]


def test_packet_inventory_covers_every_file_and_matches_bytes(tmp_path: Path) -> None:
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(archive.read(prefix + "installed-usb-upgrade-manifest.json"))
        actual = {
            name.removeprefix(prefix)
            for name in archive.namelist()
            if not name.endswith("/") and name != prefix + "installed-usb-upgrade-manifest.json"
        }
        assert set(manifest["files"]) == actual
        for name, expected in manifest["files"].items():
            payload = archive.read(prefix + name)
            assert len(payload) == expected["bytes"]
            assert hashlib.sha256(payload).hexdigest() == expected["sha256"]


def test_packet_can_promote_capture_from_the_last_installed_bundle(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    previous = tmp_path / "previous.zip"
    with zipfile.ZipFile(previous, "w") as archive:
        for path in candidate.rglob("*"):
            if path.is_file():
                archive.write(path, f"previous/capture/{path.relative_to(candidate).as_posix()}")

    packet = _packet(tmp_path, candidate_archive=previous)

    with zipfile.ZipFile(packet) as archive:
        assert f"{PACKET_NAME}/capture/capture-setup.json" in archive.namelist()


def test_release_contains_the_original_launcher_and_no_sidecar_entrypoint(tmp_path: Path) -> None:
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as outer:
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(outer.read(prefix + "installed-usb-upgrade-manifest.json"))
        release_path = tmp_path / "release.zip"
        release_path.write_bytes(outer.read(prefix + manifest["release"]["path"]))
    with zipfile.ZipFile(release_path) as release:
        names = set(release.namelist())
        launcher = release.read("scripts/algorithm/windows/serve-field-route-shell.ps1").decode(
            "utf-8"
        )
        live_source = release.read("src/billiards_ai/lab/field_assistance/live_source.py").decode(
            "utf-8"
        )
        realtime_runtime = release.read(
            "src/billiards_ai/lab/mainline1_product_realtime_runtime.py"
        ).decode("utf-8")
        screen_page = release.read("src/billiards_ai/lab/field_assistance/pages.py").decode("utf-8")
        screen_server = release.read("src/billiards_ai/lab/field_assistance/server.py").decode(
            "utf-8"
        )
        cloud_runtime = release.read(
            "src/billiards_ai/lab/field_assistance/cloud_runtime.py"
        ).decode("utf-8")
        field_service = release.read("scripts/algorithm/serve_mainline1_usb_field.py").decode(
            "utf-8"
        )
        postgame_worker = release.read("scripts/algorithm/run_mainline1_shot_worker.py").decode(
            "utf-8"
        )

    assert "release-manifest.json" in names
    assert "scripts/algorithm/windows/install-field-route-service.ps1" in names
    assert "scripts/algorithm/retire_existing_field_session.py" in names
    assert "scripts/algorithm/serve_mainline1_usb_field.py" in names
    assert "capture_profile" in launcher
    assert "8787" in launcher
    assert "8765" not in launcher
    assert not any("diagnostics/raytheon_eight_ball_field" in name for name in names)
    assert "return camera_frame.mjpeg" in live_source
    assert "live schematic encoding failed" not in live_source
    fatal_lag_branch = (
        'raise CaptureContractError("capture-to-tracker queue lag exceeded 2 frames")'
    )
    assert fatal_lag_branch not in realtime_runtime
    assert "evidence_recording_unavailable" not in realtime_runtime
    assert "self._ring.append(frame)" in realtime_runtime
    assert "def mjpeg_snapshot" in live_source
    assert 'route == "/media/live.mjpg"' in screen_server
    assert "multipart/x-mixed-replace" in screen_server
    assert "'/media/live.mjpg?epoch='" in screen_page
    assert "setInterval(sync,200)" in screen_page
    assert "media_frame_id" in screen_page
    assert "route-overlay" in screen_page
    assert "_BackgroundPostgameAnalysisWorker" in field_service
    assert "list_ended_sessions_requiring_postgame" in cloud_runtime
    assert "run_mainline1_shot_worker.py" in field_service
    gate_call = postgame_worker.index("if not _wait_for_initial_resource_gate(")
    model_load = postgame_worker.index("        import onnxruntime", gate_call)
    assert gate_call < model_load


@pytest.mark.parametrize(
    "missing_module",
    [
        None,
        "mainline1_runtime_support",
        "mainline1_identity_backends",
        "mainline1_identity_preprocessing",
        "field_assistance/http_transport",
    ],
)
def test_release_entrypoint_imports_from_only_the_shipped_files(
    tmp_path: Path,
    missing_module: str | None,
) -> None:
    """Catches an incomplete allowlist that imports from the checkout in tests."""
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as outer:
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(outer.read(prefix + "installed-usb-upgrade-manifest.json"))
        release_zip = tmp_path / "release.zip"
        release_zip.write_bytes(outer.read(prefix + manifest["release"]["path"]))
    release_root = tmp_path / "release"
    with zipfile.ZipFile(release_zip) as release:
        release.extractall(release_root)
    if missing_module is not None:
        required = release_root / "src/billiards_ai/lab" / (missing_module + ".py")
        assert required.is_file(), "required runtime module was not shipped"
        required.unlink()

    probe = """
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / "src")]
import scripts.algorithm.serve_mainline1_usb_field as entrypoint
from billiards_ai.edge.route_assistance import local_media_cleanup, local_media_lock
from billiards_ai.lab.field_assistance import cloud_runtime
from billiards_ai.lab.field_assistance import http_transport, server
from billiards_ai.lab import mainline1_runtime_support as runtime_support
from billiards_ai.lab import mainline1_identity_backends as backends
from billiards_ai.lab import mainline1_identity_preprocessing as preprocessing
from billiards_ai.lab import mainline1_runtime as runtime
from billiards_ai.lab.training import identity_crop_export, identity_dinov2
from billiards_ai.lab.training import identity_group_backend, identity_visual_evidence
import scripts.algorithm.run_mainline1_product_realtime as legacy
import scripts.algorithm.run_mainline1_shot_worker as worker

module_path = Path(entrypoint.__file__).resolve()
if not module_path.is_relative_to(root):
    raise RuntimeError(f"entrypoint escaped shipped release: {module_path}")
if not Path(cloud_runtime.__file__).resolve().is_relative_to(root):
    raise RuntimeError("cloud runtime escaped shipped release")
if not Path(local_media_lock.__file__).resolve().is_relative_to(root):
    raise RuntimeError("session media lock escaped shipped release")
if not Path(local_media_cleanup.__file__).resolve().is_relative_to(root):
    raise RuntimeError("local media cleanup escaped shipped release")
assert entrypoint.prepare_cloud_runtime is cloud_runtime.prepare_cloud_runtime
assert entrypoint._BackgroundFieldCommandWorker is cloud_runtime._BackgroundFieldCommandWorker
assert entrypoint.realtime is runtime_support
assert server._ClosingRequestReader is http_transport._ClosingRequestReader
assert server._RANGE_PATTERN is http_transport._RANGE_PATTERN
for name in ("_send_bytes", "_send_json", "_error", "_stream_media",
             "_stream_media_response", "_send_media_headers", "_range_error"):
    assert getattr(server._FieldAssistanceHandler, name) is getattr(
        http_transport.FieldHTTPResponseMixin, name)
for name in ("_detector", "_session", "_capture", "_preflight_capture", "_select_table_corners"):
    assert getattr(legacy, name) is getattr(runtime_support, name)
assert identity_dinov2.Dinov2IdentityFeatureBackend is backends.Dinov2IdentityFeatureBackend
assert worker.Dinov2IdentityFeatureBackend is backends.Dinov2IdentityFeatureBackend
assert identity_visual_evidence.OpenCvIdentityVisualEvidenceBackend is (
    backends.OpenCvIdentityVisualEvidenceBackend)
assert identity_crop_export.IdentityCropExportError is preprocessing.IdentityCropExportError
assert identity_crop_export._square_crop is runtime._square_crop is (
    preprocessing.square_identity_crop)
assert identity_group_backend._normalise is runtime._normalise is (
    preprocessing.normalise_identity_features)
for name, module in tuple(sys.modules.items()):
    if name.startswith(("billiards_ai", "scripts.")) and getattr(module, "__file__", None):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("module escaped shipped release: " + name)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(release_root)],
        cwd=release_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    if missing_module is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, "incomplete release unexpectedly imported"
        assert missing_module.replace("/", ".") in result.stderr


def test_release_pins_usb_task_profile_and_accepts_the_current_state_schema(
    tmp_path: Path,
) -> None:
    """Catches shipping a replay task or pre-v12 store in the USB upgrade."""
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as outer:
        prefix = f"{PACKET_NAME}/"
        manifest = json.loads(outer.read(prefix + "installed-usb-upgrade-manifest.json"))
        release_path = tmp_path / "release.zip"
        release_path.write_bytes(outer.read(prefix + manifest["release"]["path"]))
    with zipfile.ZipFile(release_path) as release:
        launcher = release.read("scripts/algorithm/windows/serve-field-route-shell.ps1").decode(
            "utf-8"
        )
        installer = release.read(
            "scripts/algorithm/windows/install-field-route-service.ps1"
        ).decode("utf-8")
        persistence = release.read("src/billiards_ai/edge/route_assistance/persistence.py").decode(
            "utf-8"
        )

    assert "CaptureProfilePath" in launcher
    assert "capture_profile_path" in installer
    assert "RequireUsbCapture" in installer
    assert "$null -ne $CurrentMediaFrameId" in installer
    assert "[long]$CurrentMediaFrameId -gt [long]$FirstMediaFrameId" in installer
    assert '$HealthPayload.frame.phase -notin @("degraded", "recovering")' in installer
    assert "_SCHEMA_VERSION = 30" in persistence
    with zipfile.ZipFile(release_path) as release:
        assert "src/billiards_ai/edge/route_assistance/managed_sessions.py" in release.namelist()


def test_installer_requires_advancing_frames_recording_and_unchanged_qr(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")
        installer = archive.read(prefix + "install-field-route-service.ps1").decode("utf-8-sig")

    assert "Test-NativeCameraPreflight" not in install
    assert "Test-RecorderPreflight" in install
    assert "media_frame_id" in installer
    assert "FirstMediaFrameId" in installer
    assert "recording_health" in installer
    assert "table-code.png" in installer
    assert "ExpectedJoinQrSha256" in install
    assert "JoinQrSha256After" in install
    assert '"-JoinQrAsset", $InstalledJoinQr' in install
    assert "Export-FieldTaskDefinitions" in installer
    assert "Restore-FieldTaskDefinitions" in installer


def test_packaged_installer_receipt_uses_a_defined_camera_contract(
    tmp_path: Path,
) -> None:
    """Catches a removed camera preflight variable surviving in the PASS receipt."""
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")

    contract_definition = install.index("$CameraContract = [ordered]@{")
    receipt_definition = install.index("$Receipt = [ordered]@{")
    contract_receipt = install.index("camera_contract = $CameraContract")

    assert contract_definition < receipt_definition < contract_receipt
    assert "$CameraPreflight" not in install


def test_installer_keeps_the_operator_informed_during_long_running_steps(
    tmp_path: Path,
) -> None:
    """Catches a field installer that looks frozen while preflights are running."""
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")

    stages = (
        "bundle_verified",
        "approved_native_capture_started",
        "approved_native_capture_ready",
        "camera_contract_started",
        "camera_contract_ready",
        "recorder_preflight_started",
        "recorder_preflight_ready",
        "service_preflight_started",
        "service_preflight_ready",
        "service_install_started",
        "service_install_ready",
        "advancing_frames_started",
        "advancing_frames_ready",
        "permanent_qr_verified",
    )
    for stage in stages:
        assert f'Write-InstallProgress -Stage "{stage}"' in install
    assert "Resolve-ApprovedNativeCaptureHelper" in install
    assert "Compile-NativeCaptureHelper" not in install
    assert "csc.exe" not in install
    assert "$PreflightOutput = & powershell.exe" not in install
    assert "$InstallOutput = & powershell.exe" not in install
    assert "function Invoke-InstallerProcess" in install
    assert 'Start-Process -FilePath "powershell.exe"' in install
    assert "function Write-ServiceHealthProgress" in install
    assert "[switch]$ReportServiceHealth" in install
    assert "current_version" in install
    assert "task_state" in install
    assert "source_kind" in install
    assert "media_frame_id" in install
    assert "frame_phase" in install
    assert "display_phase" in install
    assert "recording_status" in install
    assert "video_status" in install
    assert "analysis_status" in install
    assert "route_status" in install
    assert "degraded_reason" in install
    assert (
        "Invoke-InstallerProcess -Arguments $InstallArguments "
        '-Description "service installation and health verification" '
        "-ReportServiceHealth"
    ) in install
    assert "& powershell.exe @InstallArguments" not in install


def test_outer_installer_rejects_a_rolled_back_release_before_pass(
    tmp_path: Path,
) -> None:
    """Catches reporting .4/.5 PASS while current.json has rolled back to .9."""
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")

    child_exit_check = install.index("if ($InstallExitCode -ne 0)")
    installed_config_read = install.index(
        "$InstalledConfiguration = Get-Content -LiteralPath $CurrentPointer",
        child_exit_check,
    )
    version_check = install.index(
        "[string]$InstalledConfiguration.version -ne [string]$Manifest.version",
        installed_config_read,
    )
    release_path_check = install.index(
        "Installed service release mismatch",
        version_check,
    )
    service_ready = install.index(
        'Write-InstallProgress -Stage "service_install_ready"',
        release_path_check,
    )
    advancing_frames = install.index(
        'Write-InstallProgress -Stage "advancing_frames_started"',
        service_ready,
    )

    assert child_exit_check < installed_config_read < version_check
    assert version_check < release_path_check < service_ready < advancing_frames


def test_outer_installer_stops_the_inner_process_when_wait_is_interrupted(
    tmp_path: Path,
) -> None:
    """Catches Ctrl+C leaving the child installer alive after the outer exits."""
    packet = _packet(tmp_path)

    with zipfile.ZipFile(packet) as archive:
        prefix = f"{PACKET_NAME}/"
        install = archive.read(prefix + "install.ps1").decode("utf-8-sig")

    dispose = install.index("$Process.Dispose()")
    stop_live_child = install.rfind(
        "if ($null -ne $Process -and -not $Process.HasExited)",
        0,
        dispose,
    )
    stop_process = install.index("Stop-Process -Id $Process.Id -Force", stop_live_child)

    assert stop_live_child < stop_process < dispose
