import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
FOLDER = ROOT / "scripts/diagnostics/raytheon_table"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(FOLDER))
    path = FOLDER / "motion_check.py"
    assert path.exists(), "Motion diagnostic not implemented"
    spec = importlib.util.spec_from_file_location("motion_check", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_clip(folder, controls, *, mutation=None):
    folder.mkdir()
    # One-second synthetic video: timestamp evidence is hand-defined, not from evaluator.
    stamps = [100 + i / 60 for i in range(61)]
    image = np.zeros((1080, 1920, 3), np.uint8)
    writer = cv2.VideoWriter(
        str(folder / "camera.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 60, (1920, 1080)
    )
    assert writer.isOpened()
    count = 60 if mutation == "truncated" else 61
    for i in range(count):
        image[:] = i
        writer.write(image)
    writer.release()
    cv2.imwrite(str(folder / "baseline.png"), image)
    for filename, header, rows in [
        (
            "frames.csv",
            ["saved_index", "capture_index", "host_perf_counter_seconds"],
            [[i, i, f"{s:.9f}"] for i, s in enumerate(stamps)],
        ),
        (
            "capture-times.csv",
            ["capture_index", "host_perf_counter_seconds", "queued_for_video"],
            [[i, f"{s:.9f}", "True"] for i, s in enumerate(stamps)],
        ),
    ]:
        with (folder / filename).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    report = dict(
        status="SAMPLE_COLLECTED_REVIEW_REQUIRED",
        mode_matches_request=True,
        decoded_geometry=[1920, 1080],
        sample_completed=True,
        record_requested_seconds=1,
        cadence_without_recording=dict(within_5_percent=True),
        cadence_with_recording=dict(frames=61, measured_fps=60, within_5_percent=True),
        controls_after_mode=controls,
        controls_after_capture=controls,
        observed_after_capture=dict(
            frame_width=1920, frame_height=1080, fps=60, fourcc_text="MJPG"
        ),
        archive=dict(saved_frames=61, queue_drops=0, writer_error=None),
    )
    if mutation == "drops":
        report["archive"]["queue_drops"] = 1
    if mutation == "controls":
        report["controls_after_capture"] = {**controls, "focus": 999}
    if mutation == "mode":
        report["observed_after_capture"]["fourcc_text"] = "YUY2"
    if mutation == "nan_time":
        p = folder / "frames.csv"
        p.write_text(p.read_text().replace("100.000000000", "nan"))
    if mutation == "wrong_id":
        p = folder / "frames.csv"
        p.write_text(p.read_text().replace("0,0,100.000000000", "0,1,100.000000000"))
    (folder / "report.json").write_text(json.dumps(report))


def test_clip_decodes_and_timestamps_match_without_claiming_tracking(module, tmp_path):
    folder = tmp_path / "clip"
    make_clip(folder, module.table.EXPECTED_CONTROLS)
    result = module.verify_clip(folder)
    assert result["decoded_frames"] == 61
    assert result["measured_recording_fps"] == pytest.approx(60)
    assert result["tracking_evaluated"] is False
    assert result["status"] == "MEDIA_INTEGRITY_VERIFIED_REVIEW_REQUIRED"
    assert (
        result["video_sha256"] == hashlib.sha256((folder / "camera.avi").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation", ["truncated", "drops", "controls", "mode", "nan_time", "wrong_id"]
)
def test_bad_media_cannot_be_declared_complete(module, tmp_path, mutation):
    folder = tmp_path / "clip"
    make_clip(folder, module.table.EXPECTED_CONTROLS, mutation=mutation)
    with pytest.raises(ValueError):
        module.verify_clip(folder)


def build_packet(target, diagnostic_packet_assets):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_table.py"),
            "--output",
            str(target),
            "--lens-candidate",
            str(diagnostic_packet_assets.lens),
            "--frozen-candidate",
            str(diagnostic_packet_assets.frozen),
            "--motion-check",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_clean_package_import_and_one_line_powershell(tmp_path, diagnostic_packet_assets):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    target = downloads / "billiardsAI-rer-motion-check-20260905-v1.zip"
    build_packet(target, diagnostic_packet_assets=diagnostic_packet_assets)
    root = tmp_path / "packet"
    with zipfile.ZipFile(target) as z:
        m = json.loads(z.read("packet-manifest.json"))
        assert m["packet_version"] == "rer-motion-check-20260905-v1"
        for name, info in m["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == info["sha256"]
        z.extractall(root)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/diagnostics/raytheon_table/motion_check.py"),
            "--help",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    fake = tmp_path / "python with spaces"
    fake.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    doc = ROOT / "docs/operations/rer-motion-check-20260905.md"
    assert doc.exists(), "Motion instructions missing"
    command = doc.read_text().split("```powershell\n")[1].split("\n```")[0]
    assert "\n" not in command
    command = command.replace(
        r"C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe", str(fake)
    )
    prefix = (
        f"$env:USERPROFILE='{tmp_path}'; function powershell.exe {{ & '{pwsh}' @args; "
        "$global:LASTEXITCODE=$LASTEXITCODE }; "
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", prefix + command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("[")))
    assert args[0].endswith("motion_check.py")
    assert args[1] == "--output"


def test_packaged_audit_entry_decodes_real_video_and_refuses_overwrite(
    module, tmp_path, diagnostic_packet_assets
):
    target = tmp_path / "packet.zip"
    build_packet(target, diagnostic_packet_assets=diagnostic_packet_assets)
    root = tmp_path / "packet"
    with zipfile.ZipFile(target) as z:
        z.extractall(root)
    folder = tmp_path / "clip"
    make_clip(folder, module.table.EXPECTED_CONTROLS)
    dest = folder / "media-review.json"
    command = [
        sys.executable,
        str(root / "scripts/diagnostics/raytheon_table/motion_check.py"),
        "--audit-clip",
        str(folder),
        "--audit-output",
        str(dest),
    ]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(dest.read_text())["decoded_frames"] == 61
    before = dest.read_bytes()
    again = subprocess.run(
        command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=20
    )
    assert again.returncode != 0
    assert dest.read_bytes() == before


@pytest.mark.parametrize(
    "outcome", ["save", "cancel", "second_timeout", "audit_fail", "wrong_device"]
)
def test_motion_session_preserves_partial_evidence_and_no_activation(
    module, tmp_path, monkeypatch, outcome, diagnostic_packet_assets
):
    target = tmp_path / "packet.zip"
    build_packet(target, diagnostic_packet_assets=diagnostic_packet_assets)
    root = tmp_path / "packet"
    with zipfile.ZipFile(target) as z:
        z.extractall(root)
    monkeypatch.setattr(
        module, "__file__", str(root / "scripts/diagnostics/raytheon_table/motion_check.py")
    )
    before = (root / "frozen-plane-candidate.json").read_bytes()

    def prompt(*_):
        if outcome == "cancel":
            raise KeyboardInterrupt
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    fingerprint = json.loads((root / "lens-candidate.json").read_text())["device_fingerprint"]
    monkeypatch.setattr(
        module,
        "enumerate_usb_devices",
        lambda **_: [
            types.SimpleNamespace(
                durable_fingerprint="wrong" if outcome == "wrong_device" else fingerprint,
                model_dump=lambda **_: {},
            )
        ],
    )
    spec = importlib.util.spec_from_file_location(
        "camera_check", ROOT / "scripts/diagnostics/raytheon_camera/camera_check.py"
    )
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)

    def capture(folder, **kwargs):
        assert kwargs["sample_seconds"] == 10 and kwargs["record_seconds"] == 20
        assert (
            kwargs["require_mode"] is True
            and kwargs["expected_controls"] == module.table.EXPECTED_CONTROLS
        )
        if outcome == "second_timeout" and folder.name == "fast_cushion":
            return dict(status="TIMEOUT")
        folder.mkdir()
        (folder / "camera.avi").write_bytes(b"owned synthetic capture boundary")
        return dict(status="COMPLETED")

    def bounded(cmd, *_, **kwargs):
        assert kwargs["timeout"] == 45
        if outcome == "audit_fail":
            return dict(status="ERROR")
        dest = Path(cmd[cmd.index("--audit-output") + 1])
        dest.write_text(
            json.dumps(
                dict(status="MEDIA_INTEGRITY_VERIFIED_REVIEW_REQUIRED", tracking_evaluated=False)
            )
        )
        return dict(status="COMPLETED")

    monkeypatch.setattr(camera, "run_camera_child", capture)
    monkeypatch.setattr(camera, "run_bounded", bounded)
    monkeypatch.setitem(sys.modules, "camera_check", camera)
    output = tmp_path / "result"
    assert module.main(["--output", str(output)]) == (0 if outcome == "save" else 2)
    assert (root / "frozen-plane-candidate.json").read_bytes() == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["tracking_evaluated"] is False
    assert summary["active_calibration_written"] is False
    assert summary["production_ready"] is False
    assert summary["acceptance"] == "NOT_EVALUATED"
    if outcome == "save":
        assert len(summary["clips"]) == 3
    if outcome == "second_timeout":
        assert (output / "slow/media-review.json").exists()
    with zipfile.ZipFile(tmp_path / "SEND_BACK-result.zip") as z:
        assert z.testzip() is None
        for name, info in summary["inventory"].items():
            assert hashlib.sha256(z.read("result/" + name)).hexdigest() == info["sha256"]
