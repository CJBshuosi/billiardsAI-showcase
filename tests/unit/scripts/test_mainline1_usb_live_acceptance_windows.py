from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "algorithm" / "windows" / "test-mainline1-usb-live-acceptance.ps1"


def _evidence(root: Path) -> None:
    root.mkdir(parents=True)
    payloads = {
        "probe.json": {
            "status": "ready",
            "requested_width": 1920,
            "requested_height": 1080,
            "requested_fps": 60,
            "observed_width": 1920,
            "observed_height": 1080,
            "observed_fps": 60,
            "measured_fps": 60,
            "timestamp_reversals": 0,
        },
        "cadence.json": {
            "status": "ready",
            "duration_seconds": 1800,
            "observed_width": 1920,
            "observed_height": 1080,
            "observed_fps": 60,
            "measured_fps": 60,
            "timestamp_reversals": 0,
        },
        "calibration.json": {
            "gate_status": "READY",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "distortion": {"model": "OPENCV", "verified": True},
            "maximum_reference_error_mm": 20,
        },
        "shots.json": {
            "attempted": 10,
            "published": 8,
            "abstained": 2,
            "incomplete_authoritative_count": 0,
            "cross_shot_contamination_count": 0,
        },
        "disconnect.json": {
            "idle_disconnect_abstained": True,
            "in_shot_disconnect_abstained": True,
            "reconnect_ready": True,
            "synthesized_shot_end": False,
        },
        "sessions.json": {
            "session_ids": ["session-one", "session-two"],
            "second_session_joinable": True,
            "cross_session_leak_count": 0,
        },
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_acceptance_script_is_read_only_and_checks_every_hardware_gate() -> None:
    source = SCRIPT.read_text(encoding="ascii")

    for name in (
        "probe.json",
        "cadence.json",
        "calibration.json",
        "shots.json",
        "disconnect.json",
        "sessions.json",
    ):
        assert name in source
    for gate in (
        "CAPTURE_MODE_NOT_1080P60",
        "CAPTURE_CADENCE_NOT_60FPS",
        "TIMESTAMP_REVERSAL",
        "DISTORTION_NOT_VERIFIED",
        "REFERENCE_ERROR_EXCEEDS_20MM",
        "CALIBRATION_STALE",
        "INCOMPLETE_SHOT_PUBLISHED",
        "DISCONNECT_NOT_FAIL_CLOSED",
        "TWO_SESSION_ROLLOVER_FAILED",
    ):
        assert gate in source
    assert "Start-Service" not in source
    assert "Register-ScheduledTask" not in source
    assert "register_calibration" not in source


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is unavailable")
def test_valid_fixture_passes_and_incomplete_shot_fails(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _evidence(evidence)
    output = tmp_path / "accepted"
    command = [
        "pwsh",
        "-NoProfile",
        "-File",
        str(SCRIPT),
        "-EvidenceRoot",
        str(evidence),
        "-OutputRoot",
        str(output),
    ]

    passed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert passed.returncode == 0, passed.stderr
    summary = json.loads((output / "acceptance.json").read_text(encoding="utf-8-sig"))
    assert summary["status"] == "PASS"

    shots_path = evidence / "shots.json"
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    shots["incomplete_authoritative_count"] = 1
    shots_path.write_text(json.dumps(shots), encoding="utf-8")
    failed = subprocess.run(
        [*command[:-1], str(tmp_path / "rejected")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode == 2
    assert "INCOMPLETE_SHOT_PUBLISHED" in failed.stdout
