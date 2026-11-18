"""Behavioral checks for the isolated camera diagnostic packet; no field writers."""

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts/diagnostics/raytheon_camera/camera_check.py"


@pytest.fixture
def packet():
    assert SCRIPT.is_file(), "Standalone camera diagnostic is not implemented"
    spec = importlib.util.spec_from_file_location("camera_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cadence_uses_delivered_intervals_and_rejects_backwards_time(packet):
    result = packet.cadence([1.0, 1.02, 1.04, 1.06], 60)
    assert result["measured_fps"] == pytest.approx(50)
    assert result["within_5_percent"] is False
    assert result["maximum_gap_ms"] == pytest.approx(20)
    result = packet.cadence([1.0, 1.02, 1.01], 60)
    assert result["monotonic"] is False
    assert result["within_5_percent"] is False
    assert packet.cadence([], 60)["measured_fps"] is None


def test_equal_timestamps_are_not_clock_reversals_or_a_cadence_pass(packet):
    result = packet.cadence([1.0, 1.0, 1.05, 1.1], 30)
    assert result["monotonic"] is True
    assert result["timestamp_reversals"] == 0
    assert result["equal_timestamps"] == 1
    assert result["measured_fps"] == pytest.approx(30)
    assert result["within_5_percent"] is False
    assert packet.cadence([1.0, 1.0], 60)["measured_fps"] is None
    assert packet.cadence([1.0, float("nan")], 60)["measured_fps"] is None


def test_hung_camera_worker_is_killed_and_failure_retained(packet, tmp_path):
    started = time.monotonic()
    result = packet.run_bounded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "worker.log",
        timeout=0.2,
    )
    assert time.monotonic() - started < 5
    assert result["status"] == "TIMEOUT"
    assert result["returncode"] is not None


def test_field_audit_never_migrates_or_exports_player_payloads(packet, tmp_path):
    db_path = tmp_path / "field.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            PRAGMA user_version=3;
            CREATE TABLE field_sessions(session_id TEXT, status TEXT, created_at_utc TEXT);
            INSERT INTO field_sessions VALUES('session-a','open','2026-09-05');
            CREATE TABLE field_shots(payload_json TEXT);
            INSERT INTO field_shots VALUES('PRIVATE_OPENID_AND_MEDIA_URL');
        """)
    before = db_path.read_bytes()
    result = packet.inspect_state_db(db_path)
    assert result["schema_version"] == 3
    assert result["counts"]["field_shots"] == 1
    assert result["recent_sessions"][0]["session_id"] == "session-a"
    assert "PRIVATE_OPENID" not in json.dumps(result)
    assert db_path.read_bytes() == before
    absent = tmp_path / "missing.sqlite3"
    assert packet.inspect_state_db(absent)["status"] == "MISSING"
    assert not absent.exists()


def test_result_zip_hashes_exact_files_and_refuses_overwrite(packet, tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "clip.avi").write_bytes(b"video-evidence")
    target = packet.pack_result(result_dir, {"status": "PARTIAL"})
    with zipfile.ZipFile(target) as archive:
        summary = json.loads(archive.read("run/summary.json"))
        assert summary["status"] == "PARTIAL"
        assert summary["acceptance"] == "NOT_EVALUATED"
        assert (
            summary["inventory"]["clip.avi"]["sha256"]
            == hashlib.sha256(b"video-evidence").hexdigest()
        )
        assert archive.read("run/clip.avi") == b"video-evidence"
    with pytest.raises(FileExistsError):
        packet.pack_result(result_dir, {})


def test_result_zip_rejects_symlink_to_unrelated_data(packet, tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    secret = tmp_path / "private.txt"
    secret.write_text("unrelated private data")
    (result_dir / "leak.txt").symlink_to(secret)
    with pytest.raises(ValueError, match="symlink"):
        packet.pack_result(result_dir, {})


def test_process_audit_retains_input_paths_but_drops_tokens_and_urls(packet):
    command = (
        'python.exe "C:\\release\\serve_field_route_shell.py" '
        '--artifact "C:\\data with spaces\\replay" --state-db C:\\runtime\\state.sqlite3 '
        "--control-token TOP_SECRET --sync-base-url https://private/?secret=HMAC"
    )
    result = packet.safe_command(command)
    assert result["safe_arguments"] == {
        "artifact": r"C:\data with spaces\replay",
        "state-db": r"C:\runtime\state.sqlite3",
    }
    assert "TOP_SECRET" not in json.dumps(result)
    assert "HMAC" not in json.dumps(result)
    assert "https://" not in json.dumps(result)
    assert result["command_sha256"] == hashlib.sha256(command.encode()).hexdigest()


def test_inventory_does_not_mistake_its_own_query_for_a_live_capture(packet, tmp_path, monkeypatch):
    inventory = {
        "inventory_pid": 111,
        "processes": [
            {
                "Name": "powershell.exe",
                "ProcessId": 111,
                "CommandLine": "query contains run_mainline1_usb_live.py",
            },
            {
                "Name": "python.exe",
                "ProcessId": 222,
                "CommandLine": "python serve_field_route_shell.py --control-token SECRET",
            },
        ],
        "tasks": [],
        "errors": [],
    }

    def run(command, **kwargs):
        if command[0] == "powershell.exe":
            return SimpleNamespace(returncode=0, stdout=json.dumps(inventory))
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(packet.subprocess, "run", run)
    audit = packet.collect_windows_audit(tmp_path, tmp_path)
    assert audit["camera_apps_running"] == []
    assert [p["ProcessId"] for p in audit["processes"]] == [222]
    assert "SECRET" not in json.dumps(audit)
    inventory["processes"].append(
        {
            "Name": "python.exe",
            "ProcessId": 333,
            "CommandLine": "python run_mainline1_usb_live.py",
        }
    )
    audit = packet.collect_windows_audit(tmp_path, tmp_path)
    assert audit["camera_apps_running"] == [{"name": "python.exe", "pid": 333}]


class SyntheticCapture:
    """Only the hardware boundary is fake; encoding, reports and frame CSV are real."""

    def __init__(self, cv, *, geometry=(64, 48), fail_after=None):
        import numpy as np

        self.cv = cv
        self.geometry = geometry
        self.frame = np.zeros((geometry[1], geometry[0], 3), dtype=np.uint8)
        self.frame[:, :, 1] = 90
        self.count = 0
        self.fail_after = fail_after
        self.released = False
        self.settings = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        assert prop in {
            self.cv.CAP_PROP_FOURCC,
            self.cv.CAP_PROP_FRAME_WIDTH,
            self.cv.CAP_PROP_FRAME_HEIGHT,
            self.cv.CAP_PROP_FPS,
        }, "Diagnostic must never write exposure, focus or other image controls"
        self.settings[prop] = value
        return True

    def get(self, prop):
        return self.settings.get(prop, 0)

    def read(self):
        time.sleep(0.002)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            return False, None
        return True, self.frame.copy()

    def release(self):
        self.released = True


def test_fps_reinitialization_cannot_erase_final_mjpg_request(packet, tmp_path):
    import cv2

    class ResetFormatCapture(SyntheticCapture):
        def set(self, prop, value):
            result = super().set(prop, value)
            if prop == cv2.CAP_PROP_FPS:
                self.settings[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
            return result

    capture = ResetFormatCapture(cv2)
    report = packet.capture_run(
        cv2,
        tmp_path / "format",
        index=0,
        width=64,
        height=48,
        seconds=0,
        sample_seconds=0.01,
        capture_factory=lambda _: capture,
    )
    assert report["observed"]["fourcc_text"] == "MJPG"
    assert report["mode_matches_request"] is True
    assert report["negotiation_steps"][-1]["property"] == "FOURCC"
    assert report["observed_after_capture"]["fourcc_text"] == "MJPG"


def test_msmf_is_explicit_and_capture_uses_high_resolution_clock(packet, tmp_path, monkeypatch):
    import csv

    import cv2

    class MsmfCv:
        def VideoCapture(self, index, backend):
            assert index == 2 and backend == cv2.CAP_MSMF
            return SyntheticCapture(cv2)

        def __getattr__(self, name):
            return getattr(cv2, name)

    def coarse_clock_must_not_be_used():
        raise AssertionError("Capture must not use the Windows coarse monotonic clock")

    monkeypatch.setattr(packet.time, "monotonic", coarse_clock_must_not_be_used)
    output = tmp_path / "msmf"
    report = packet.capture_run(
        MsmfCv(),
        output,
        index=2,
        width=64,
        height=48,
        backend="MSMF",
        seconds=0,
        sample_seconds=0.02,
    )
    assert report["status"] == "READBACK_COLLECTED_REVIEW_REQUIRED"
    assert report["backend_requested"] == "MSMF"
    rows = list(csv.DictReader((output / "cadence-times.csv").open()))
    assert len(rows) == report["cadence_without_recording"]["frames"]
    assert int(rows[-1]["host_perf_counter_ns"]) > int(rows[0]["host_perf_counter_ns"])
    assert report["clock_info"]["sample_clock"] == "perf_counter_ns"


def test_capture_preserves_geometry_and_delivered_frame_timing(packet, tmp_path):
    import cv2

    capture = SyntheticCapture(cv2)
    output = tmp_path / "capture"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0.05,
        sample_seconds=0.03,
        capture_factory=lambda _: capture,
    )
    assert report["status"] == "SAMPLE_COLLECTED_REVIEW_REQUIRED"
    assert report["trusted_pnp_capture_binding"] is False
    assert report["control_support_and_lock"] == "NOT_VERIFIED"
    assert capture.released
    assert cv2.imread(str(output / "baseline.png")).shape == (48, 64, 3)
    video = cv2.VideoCapture(str(output / "camera.avi"))
    try:
        assert video.isOpened()
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == report["archive"]["saved_frames"]
        ok, frame = video.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        video.release()
    import csv

    rows = list(csv.DictReader((output / "frames.csv").open()))
    assert len(rows) == report["archive"]["saved_frames"]
    assert float(rows[-1]["host_perf_counter_seconds"]) > float(
        rows[0]["host_perf_counter_seconds"]
    )


def test_capture_failure_keeps_evidence_and_releases_device(packet, tmp_path):
    import cv2

    capture = SyntheticCapture(cv2, fail_after=17)
    output = tmp_path / "failed"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0.01,
        sample_seconds=1,
        capture_factory=lambda _: capture,
    )
    assert report["status"] == "ERROR"
    assert report["reason"] == "FRAME_READ_FAILED_IN_CADENCE_SAMPLE"
    assert capture.released
    assert (output / "baseline.png").is_file()
    assert json.loads((output / "report.json").read_text())["status"] == "ERROR"


def test_wrong_geometry_never_becomes_a_successful_sample(packet, tmp_path):
    import cv2

    capture = SyntheticCapture(cv2, geometry=(32, 24))
    report = packet.capture_run(
        cv2,
        tmp_path / "wrong",
        index=0,
        width=64,
        height=48,
        capture_factory=lambda _: capture,
    )
    assert report["status"] == "ERROR"
    assert report["reason"] == "DECODED_GEOMETRY_MISMATCH"
    assert capture.released


def test_readback_only_does_not_create_an_extra_video(packet, tmp_path):
    import cv2

    capture = SyntheticCapture(cv2)
    output = tmp_path / "reopen"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0,
        sample_seconds=0.02,
        capture_factory=lambda _: capture,
    )
    assert report["status"] == "READBACK_COLLECTED_REVIEW_REQUIRED"
    assert not (output / "camera.avi").exists()
    assert (output / "baseline.png").is_file()


def test_slow_writer_is_reported_without_blocking_capture(packet, tmp_path):
    import cv2

    class SlowWriter:
        def __init__(self, *args):
            self.writer = cv2.VideoWriter(*args)

        def isOpened(self):
            return self.writer.isOpened()

        def write(self, image):
            time.sleep(0.025)
            self.writer.write(image)

        def release(self):
            self.writer.release()

    class SlowCv:
        VideoWriter = SlowWriter

        def __getattr__(self, name):
            return getattr(cv2, name)

    report = packet.capture_run(
        SlowCv(),
        tmp_path / "slow",
        index=0,
        width=64,
        height=48,
        seconds=0.2,
        sample_seconds=0.02,
        capture_factory=lambda _: SyntheticCapture(cv2),
    )
    assert report["archive"]["queue_drops"] > 0
    assert (
        report["archive"]["saved_frames"] + report["archive"]["queue_drops"]
        == (report["cadence_with_recording"]["frames"])
    )


def test_packaging_contains_only_allowlisted_tools_and_exact_hashes(tmp_path):
    import subprocess

    builder = SCRIPT.parent.parent / "package_raytheon_camera.py"
    assert builder.is_file(), "Packet builder not implemented"
    output = tmp_path / "camera.zip"
    subprocess.run([sys.executable, str(builder), "--output", str(output)], check=True)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "run.ps1",
            "camera_check.py",
            "README.txt",
            "packet-manifest.json",
        }
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["baseline_commit"] == "9a144955377b289ce47de71769d1305dd8585de4"
        for name in ("run.ps1", "camera_check.py", "README.txt"):
            assert (
                manifest["files"][name]["sha256"] == hashlib.sha256(archive.read(name)).hexdigest()
            )
        assert manifest["baseline_source_sha256"]["src/billiards_ai/lab/mainline1_frame_source.py"]
    result = subprocess.run(
        [sys.executable, str(builder), "--output", str(output)], capture_output=True
    )
    assert result.returncode != 0


def test_video_size_limit_stops_sample_without_deleting_evidence(packet, tmp_path):
    import cv2

    capture = SyntheticCapture(cv2)
    output = tmp_path / "size-limit"
    result = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0.03,
        sample_seconds=0.01,
        maximum_video_bytes=1,
        capture_factory=lambda _: capture,
    )
    assert result["status"] == "ERROR"
    assert result["reason"] == "VIDEO_SIZE_LIMIT_EXCEEDED"
    assert (output / "camera.avi").exists()
    assert capture.released


def test_backend_comparison_requires_separate_preview_choices_and_no_video(
    packet, tmp_path, monkeypatch
):
    import cv2
    import numpy as np

    calls = []

    def run(output, *, index, backend, scan=False, **kwargs):
        calls.append((backend, index, scan, kwargs))
        chosen = 0 if backend == "MSMF" else 1
        if scan:
            if index != chosen:
                return {"status": "ERROR"}
            output.mkdir(parents=True)
            cv2.imwrite(str(output / "baseline.png"), np.zeros((48, 64, 3), dtype=np.uint8))
        else:
            assert index == chosen, "Indices must not be reused across backend namespaces"
            assert kwargs["record_seconds"] == 0, "Retest must isolate acquisition from encoding"
        return {"status": "COMPLETED", "report": {"status": "READBACK_COLLECTED_REVIEW_REQUIRED"}}

    answers = iter(["0", "1"])
    monkeypatch.setattr(packet, "run_camera_child", run)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(packet.os, "startfile", lambda _: None, raising=False)
    summary = {"runs": []}
    packet.collect_comparison(tmp_path, cv2, summary)
    assert summary["status"] == "COLLECTED_REVIEW_REQUIRED"
    assert summary["selection"]["MSMF"]["runtime_index"] == 0
    assert summary["selection"]["DSHOW"]["runtime_index"] == 1
    assert len([c for c in calls if not c[2]]) == 6
    assert {r["backend"] for r in summary["runs"]} == {"MSMF", "DSHOW"}


def test_comparison_timeout_stops_before_another_camera_open(packet, tmp_path, monkeypatch):
    import cv2

    calls = []

    def timeout(output, **kwargs):
        calls.append(kwargs)
        return {"status": "TIMEOUT"}

    monkeypatch.setattr(packet, "run_camera_child", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        packet.collect_comparison(tmp_path, cv2, {"runs": []})
    assert len(calls) == 1


def test_worker_command_preserves_backend_and_sample_duration(packet, tmp_path, monkeypatch):
    def bounded(command, log_path, **kwargs):
        assert command[command.index("--backend") + 1] == "MSMF"
        assert command[command.index("--sample-seconds") + 1] == "15"
        assert command[command.index("--record-seconds") + 1] == "0"
        return {"status": "ERROR", "returncode": 2}

    monkeypatch.setattr(packet, "run_bounded", bounded)
    result = packet.run_camera_child(
        tmp_path / "worker",
        index=1,
        backend="MSMF",
        record_seconds=0,
        sample_seconds=15,
    )
    assert result["status"] == "ERROR"
    assert json.loads((tmp_path / "worker.process.json").read_text())["returncode"] == 2


@pytest.mark.parametrize("output_args", [[], ["-OutputParent", ""]])
def test_powershell_launcher_archives_missing_python_in_default_output(tmp_path, output_args):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell unavailable; launcher runtime check requires PowerShell")
    archive_path = tmp_path / "packet.zip"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT.parent.parent / "package_raytheon_camera.py"),
            "--output",
            str(archive_path),
        ],
        check=True,
        capture_output=True,
    )
    destination = tmp_path / "unpacked with spaces"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    result = subprocess.run(
        [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(destination / "run.ps1"),
            "-ProjectRoot",
            str(tmp_path / "absent-project"),
            "-RuntimeRoot",
            str(tmp_path / "absent-runtime"),
            "-PythonPath",
            str(tmp_path / "absent-python"),
            *output_args,
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    outputs = list(destination.glob("SEND_BACK-*-bootstrap.zip"))
    assert len(outputs) == 1, result.stdout + result.stderr
    with zipfile.ZipFile(outputs[0]) as archive:
        error_name = next(n for n in archive.namelist() if n.endswith("bootstrap-error.json"))
        error = json.loads(archive.read(error_name).decode("utf-8-sig"))
    assert error["status"] == "BOOTSTRAP_FAILED"
    assert "Python not found" in error["error"]
    assert error["acceptance"] == "NOT_EVALUATED"


def test_powershell_parses_inventory_and_operator_one_liner(packet, monkeypatch):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell unavailable")
    actual_run = subprocess.run
    commands = []

    def inventory_run(command, **kwargs):
        if command[0] == "powershell.exe":
            commands.append(command[-1])
            return SimpleNamespace(returncode=0, stdout='{"processes":[],"tasks":[]}')
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(packet.subprocess, "run", inventory_run)
    packet.collect_windows_audit("not-an-existing-project", "not-an-existing-runtime")
    commands.extend(
        line
        for line in (SCRIPT.parent / "README.txt").read_text().splitlines()
        if line.startswith("& {")
    )
    assert len(commands) == 2
    for command in commands:
        result = actual_run(
            [
                pwsh,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$Source=[Console]::In.ReadToEnd(); $Tokens=$null; $ParseErrors=$null; "
                "[System.Management.Automation.Language.Parser]::ParseInput"
                "($Source,[ref]$Tokens,[ref]$ParseErrors) | Out-Null; "
                "if ($ParseErrors.Count -gt 0) { $ParseErrors | ConvertTo-Json; exit 1 }",
            ],
            input=command,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_strict_sample_rejects_silent_yuy2_fallback_before_recording(packet, tmp_path):
    import cv2

    class WrongFormat(SyntheticCapture):
        def get(self, prop):
            if prop == cv2.CAP_PROP_FOURCC:
                return cv2.VideoWriter_fourcc(*"YUY2")
            return super().get(prop)

    output = tmp_path / "wrong-format"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0.01,
        sample_seconds=0.01,
        require_mode=True,
        capture_factory=lambda _: WrongFormat(cv2),
    )
    assert report["reason"] == "REQUESTED_MODE_NOT_CONFIRMED"
    assert not (output / "camera.avi").exists()


def test_failed_long_sample_keeps_successful_frame_timestamps(packet, tmp_path):
    import csv

    import cv2

    output = tmp_path / "partial-cadence"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0,
        sample_seconds=1800,
        capture_factory=lambda _: SyntheticCapture(cv2, fail_after=18),
    )
    assert report["status"] == "ERROR"
    assert report["cadence_without_recording"]["frames"] == 3
    with (output / "cadence-times.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert report["sample_completed"] is False


def test_stalled_long_worker_does_not_wait_thirty_minutes(packet, tmp_path):
    result = packet.run_bounded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "stalled.log",
        timeout=1900,
        heartbeat_path=tmp_path / "progress.json",
        stall_timeout=0.2,
    )
    assert result["status"] == "TIMEOUT"
    assert result["reason"] == "WORKER_PROGRESS_STALLED"
    assert result["elapsed_seconds"] < 5


def test_long_worker_budget_and_cli_support_full_thirty_minutes(packet, tmp_path, monkeypatch):
    def bounded(command, log_path, **kwargs):
        assert command[command.index("--sample-seconds") + 1] == "1800"
        assert "--require-mode" in command
        assert kwargs["timeout"] >= 1850
        assert kwargs["stall_timeout"] == 30
        # Execute the actual CLI: missing device fails quickly, not argument parsing.
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert "invalid choice" not in result.stderr
        assert (tmp_path / "long" / "report.json").is_file()
        return {"status": "ERROR", "returncode": result.returncode}

    monkeypatch.setattr(packet, "run_bounded", bounded)
    packet.run_camera_child(
        tmp_path / "long",
        index=9999,
        sample_seconds=1800,
        record_seconds=0,
        require_mode=True,
    )


@pytest.mark.parametrize("failed_stage", [None, "motion-1", "stability-30min"])
def test_motion_workflow_is_dshow_only_and_stops_after_worker_failure(
    packet, tmp_path, monkeypatch, failed_stage
):
    import cv2

    monkeypatch.setattr(packet, "select_camera", lambda *args: 2)
    monkeypatch.setattr("builtins.input", lambda _: "")
    calls = []

    def run(output, **kwargs):
        calls.append((output.name, kwargs))
        assert kwargs["backend"] == "DSHOW" and kwargs["index"] == 2
        assert kwargs["require_mode"] is True
        if output.name == failed_stage:
            return {"status": "TIMEOUT"}
        return {
            "status": "COMPLETED",
            "report": {
                "controls_after_capture": {"exposure": -7, "autofocus": 2, "auto_wb": 0},
                "controls_changed_during_run": False,
            },
        }

    monkeypatch.setattr(packet, "run_camera_child", run)
    summary = {"runs": []}
    if failed_stage:
        with pytest.raises(RuntimeError):
            packet.collect_motion_stability(tmp_path, cv2, summary)
        assert calls[-1][0] == failed_stage
    else:
        packet.collect_motion_stability(tmp_path, cv2, summary)
        assert [name for name, _ in calls] == [
            "preflight",
            "motion-1",
            "motion-2",
            "motion-3",
            "stability-30min",
            "reopen",
        ]
        assert calls[4][1]["sample_seconds"] == 1800
        assert calls[4][1]["record_seconds"] == 0
        assert all(calls[i][1]["record_seconds"] == 10 for i in (1, 2, 3))
        assert summary["status"] == "COLLECTED_REVIEW_REQUIRED"


def test_motion_preflight_blocks_unlocked_focus(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "select_camera", lambda *args: 0)
    calls = []

    def run(output, **kwargs):
        calls.append(output)
        return {
            "status": "COMPLETED",
            "report": {
                "controls_after_capture": {"exposure": -7, "autofocus": 1, "auto_wb": 0},
            },
        }

    monkeypatch.setattr(packet, "run_camera_child", run)
    with pytest.raises(RuntimeError, match="CONTROL"):
        packet.collect_motion_stability(tmp_path, None, {"runs": []})
    assert len(calls) == 1


def test_reopened_camera_cannot_silently_change_manual_focus(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "select_camera", lambda *args: 0)
    monkeypatch.setattr("builtins.input", lambda _: "")
    calls = []

    def run(output, **kwargs):
        calls.append(output.name)
        return {
            "status": "COMPLETED",
            "report": {
                "controls_after_capture": {
                    "exposure": -7,
                    "autofocus": 2,
                    "auto_wb": 0,
                    "focus": 100 if output.name == "preflight" else 300,
                },
                "controls_changed_during_run": False,
            },
        }

    monkeypatch.setattr(packet, "run_camera_child", run)
    summary = {"runs": []}
    with pytest.raises(RuntimeError, match="CONTROL"):
        packet.collect_motion_stability(tmp_path, None, summary)
    assert calls == ["preflight", "motion-1"]
    assert summary["runs"][-1]["controls_changed_since_preflight"] is True


def test_thirty_minute_sample_keeps_one_handle_and_periodic_checkpoints(
    packet, tmp_path, monkeypatch
):
    import csv

    import cv2

    clock_ns = [0]
    opened = []

    class VirtualCapture(SyntheticCapture):
        def read(self):
            clock_ns[0] += 1_000_000_000
            return True, self.frame.copy()

    capture = VirtualCapture(cv2)

    def factory(index):
        opened.append(index)
        return capture

    monkeypatch.setattr(packet.time, "perf_counter_ns", lambda: clock_ns[0])
    output = tmp_path / "thirty-minutes"
    report = packet.capture_run(
        cv2,
        output,
        index=2,
        width=64,
        height=48,
        seconds=0,
        sample_seconds=1800,
        capture_factory=factory,
        require_mode=True,
    )
    assert opened == [2] and capture.released
    assert report["sample_completed"] is True
    assert report["cadence_without_recording"]["frames"] == 1800
    with (output / "cadence-times.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1800
    assert (
        int(rows[-1]["host_perf_counter_ns"]) - int(rows[0]["host_perf_counter_ns"])
        == 1799_000_000_000
    )
    progress = json.loads((output / "progress.json").read_text())
    assert progress["frames"] == 1800
    assert not (output / "camera.avi").exists()


def test_motion_uses_real_control_serialization_contract(packet, tmp_path, monkeypatch):
    import cv2

    class ManualCapture(SyntheticCapture):
        def get(self, prop):
            return {
                cv2.CAP_PROP_EXPOSURE: -7,
                cv2.CAP_PROP_AUTOFOCUS: 2,
                cv2.CAP_PROP_AUTO_WB: 0,
            }.get(prop, super().get(prop))

    monkeypatch.setattr(packet, "select_camera", lambda *args: 0)
    monkeypatch.setattr("builtins.input", lambda _: "")
    controls = packet.read_controls(cv2, ManualCapture(cv2))
    monkeypatch.setattr(
        packet,
        "run_camera_child",
        lambda *args, **kwargs: {
            "status": "COMPLETED",
            "report": {
                "controls_after_capture": controls,
                "controls_changed_during_run": False,
            },
        },
    )
    summary = {"runs": []}
    packet.collect_motion_stability(tmp_path, cv2, summary)
    assert len(summary["runs"]) == 6
    assert summary["status"] == "COLLECTED_REVIEW_REQUIRED"


def test_killing_blocked_capture_retains_last_flushed_checkpoint(packet, tmp_path):
    import csv

    output = tmp_path / "blocked"
    program = """
import runpy, sys, time
import cv2
p = runpy.run_path(sys.argv[1])
t = runpy.run_path(sys.argv[2])
clock = [0]
time.perf_counter_ns = lambda: clock[0]
class FrozenCapture(t["SyntheticCapture"]):
    def read(self):
        self.count += 1
        if self.count == 39:
            time.sleep(30)
        clock[0] += 1_000_000_000
        return True, self.frame.copy()
p["capture_run"](cv2, sys.argv[3], index=0, width=64, height=48,
                 seconds=0, sample_seconds=1800,
                 capture_factory=lambda _: FrozenCapture(cv2))
"""
    result = packet.run_bounded(
        [sys.executable, "-c", program, str(SCRIPT), str(Path(__file__)), str(output)],
        tmp_path / "blocked.log",
        timeout=1900,
        heartbeat_path=output / "progress.json",
        stall_timeout=2,
    )
    assert result["reason"] == "WORKER_PROGRESS_STALLED"
    with (output / "cadence-times.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 20
    assert json.loads((output / "progress.json").read_text())["frames"] == 20


def test_expected_exposure_blocks_wrong_readback_before_video(packet, tmp_path):
    import cv2

    capture = SyntheticCapture(cv2)
    capture.settings[cv2.CAP_PROP_EXPOSURE] = -7
    output = tmp_path / "wrong-exposure"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0.01,
        expected_controls={"exposure": -8},
        capture_factory=lambda _: capture,
    )
    assert report["reason"] == "EXPECTED_CONTROLS_MISMATCH_BEFORE_CAPTURE"
    assert not (output / "camera.avi").exists()
    assert capture.released


def test_recording_progress_is_refreshed_and_duration_is_not_hardcoded(packet, tmp_path):
    import cv2

    output = tmp_path / "record-progress"
    report = packet.capture_run(
        cv2,
        output,
        index=0,
        width=64,
        height=48,
        seconds=0.08,
        sample_seconds=0.01,
        capture_factory=lambda _: SyntheticCapture(cv2),
    )
    marker = json.loads((output / "recording-started.json").read_text())
    assert marker["duration_seconds"] == 0.08
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "recording_finished"
    assert progress["frames"] == report["cadence_with_recording"]["frames"]


@pytest.mark.parametrize("problem", [None, "busy-second", "wrong-second", "timeout-first"])
def test_exposure_comparison_rechecks_apps_and_guards_controls(
    packet, tmp_path, monkeypatch, problem
):
    import cv2

    audits = []
    selected = []
    calls = []
    monkeypatch.setattr("builtins.input", lambda _: "")

    def audit(*args):
        audits.append(1)
        return {
            "errors": [],
            "camera_apps_running": [{"name": "AMCap.exe"}]
            if problem == "busy-second" and len(audits) == 2
            else [],
        }

    def select(root, cv, summary, backend):
        assert backend == "DSHOW"
        selected.append(root.name)
        return 0

    def worker(output, **kwargs):
        calls.append((output, kwargs))
        assert kwargs["backend"] == "DSHOW"
        assert kwargs["sample_seconds"] <= 10
        expected = kwargs["expected_controls"]
        exposure = -7 if output.parent.name == "exposure-7" else -8
        assert expected["exposure"] == exposure
        assert expected["autofocus"] == 2 and expected["auto_wb"] == 0
        if exposure == -8:
            assert expected["focus"] == 100 and expected["gain"] == 140
        if problem == "wrong-second" and exposure == -8:
            return {"status": "ERROR"}
        if problem == "timeout-first" and output.name == "shot":
            return {"status": "TIMEOUT"}
        return {
            "status": "COMPLETED",
            "report": {
                "controls_after_capture": {
                    "exposure": exposure,
                    "autofocus": 2,
                    "auto_wb": 0,
                    "focus": 100,
                    "gain": 140,
                }
            },
        }

    monkeypatch.setattr(packet, "collect_windows_audit", audit)
    monkeypatch.setattr(packet, "select_camera", select)
    monkeypatch.setattr(packet, "run_camera_child", worker)
    summary = {"runs": []}
    if problem:
        with pytest.raises(RuntimeError):
            packet.collect_exposure_comparison(tmp_path, cv2, summary)
        expected_calls = {"busy-second": 2, "wrong-second": 3, "timeout-first": 2}
        assert len(calls) == expected_calls[problem]
    else:
        packet.collect_exposure_comparison(tmp_path, cv2, summary)
        assert len(audits) == 2
        assert selected == ["exposure-7", "exposure-8"]
        assert [k["record_seconds"] for _, k in calls] == [0, 30, 0, 30]
        assert [k["sample_seconds"] for _, k in calls] == [2, 10, 2, 10]
        assert summary["status"] == "COLLECTED_REVIEW_REQUIRED"


def test_extended_recording_cli_preserves_duration_and_expected_controls(
    packet, tmp_path, monkeypatch
):
    def bounded(command, log_path, **kwargs):
        assert command[command.index("--record-seconds") + 1] == "30"
        assert json.loads(command[command.index("--expected-controls") + 1]) == {"exposure": -8}
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert "invalid choice" not in result.stderr
        assert (tmp_path / "cli" / "report.json").is_file()
        return {"status": "ERROR", "returncode": result.returncode}

    monkeypatch.setattr(packet, "run_bounded", bounded)
    packet.run_camera_child(
        tmp_path / "cli",
        index=9999,
        record_seconds=30,
        sample_seconds=10,
        expected_controls={"exposure": -8},
    )


def test_thirty_second_recording_keeps_heartbeat_alive(packet, tmp_path, monkeypatch):
    import cv2

    clock = [0]
    checkpoints = []
    actual_write = packet.write_json

    class VirtualCapture(SyntheticCapture):
        def read(self):
            clock[0] += 1_000_000_000
            return True, self.frame.copy()

    def write(path, payload):
        actual_write(path, payload)
        if Path(path).name == "progress.json" and payload.get("phase") == "recording":
            checkpoints.append(payload["frames"])

    monkeypatch.setattr(packet, "write_json", write)
    monkeypatch.setattr(packet.time, "perf_counter_ns", lambda: clock[0])
    result = packet.capture_run(
        cv2,
        tmp_path / "thirty-second-video",
        index=0,
        width=64,
        height=48,
        seconds=30,
        sample_seconds=10,
        capture_factory=lambda _: VirtualCapture(cv2),
    )
    assert checkpoints == [0, 5, 10, 15, 20, 25, 30]
    assert result["status"] == "SAMPLE_COLLECTED_REVIEW_REQUIRED"
    assert result["cadence_with_recording"]["frames"] == 30


def test_exposure_changed_during_capture_is_not_collected_as_success(packet, tmp_path):
    import cv2

    class DriftingCapture(SyntheticCapture):
        def get(self, prop):
            if prop == cv2.CAP_PROP_EXPOSURE:
                return -7 if self.count < 16 else -8
            return super().get(prop)

    result = packet.capture_run(
        cv2,
        tmp_path / "drift",
        index=0,
        width=64,
        height=48,
        seconds=0,
        sample_seconds=0.02,
        expected_controls={"exposure": -7},
        capture_factory=lambda _: DriftingCapture(cv2),
    )
    assert result["status"] == "ERROR"
    assert result["reason"] == "EXPECTED_CONTROLS_MISMATCH_AFTER_CAPTURE"
