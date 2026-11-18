"""Real CLI processes, synthetic video, temporary local records; never profile credit."""

from __future__ import annotations

import hashlib
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from urllib.request import Request, urlopen

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.mainline1_shot_local_video import run_mainline1_shot_local_video
from tests.integration.lab.field_assistance.test_shell_cli import REPOSITORY_ROOT, _environment
from tests.unit.lab.test_mainline1_shot_local_video import (
    RecordingBackend,
    RecordingClassifier,
    _calibration,
    _config,
    _synthetic_video,
)


@pytest.fixture
def cli_source(tmp_path):
    video = _synthetic_video(tmp_path / "synthetic.avi")
    source = run_mainline1_shot_local_video(
        video=video,
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "source",
    )
    return video, source.path


def replay_cli(tmp_path, video, source):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/run_field_route_replay.py",
            "--video",
            str(video),
            "--mainline1-artifact",
            str(source),
            "--context-group",
            "solid",
            "--stable-duration-ms",
            "100",
            "--output-root",
            str(tmp_path / "replay"),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["shot_count"] == 1
    return result["output"]


@contextmanager
def shell(artifact, db_path, source_args):
    process = subprocess.Popen(
        [
            sys.executable,
            "scripts/algorithm/serve_field_route_shell.py",
            "--artifact",
            str(artifact),
            "--state-db",
            str(db_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--control-token",
            "synthetic-test-token",
            *source_args,
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        lines = queue.Queue()
        threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
        line = lines.get(timeout=15)
        assert line, process.stderr.read()
        ready = json.loads(line)
        assert ready["status"] == "READY"
        yield ready
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def control(ready, payload):
    url = ready["screen_url"].removesuffix("/screen") + "/api/control"
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Control-Token": "synthetic-test-token",
        },
    )
    with urlopen(request, timeout=5) as response:
        assert response.status == 200


def set_test_participants(ready):
    for role in ("red", "blue"):
        control(
            ready,
            {
                "command": "emergency_set_participant",
                "role": role,
                "user_id": f"synthetic-cli-{role}",
                "reason": "isolated synthetic CLI test",
            },
        )
    control(
        ready,
        {
            "command": "emergency_set_breaker",
            "role": "red",
            "reason": "isolated synthetic CLI test",
        },
    )


def consume_arrived_timeline(ready):
    # Leave the five-frame source's observed boundaries in the past, avoiding a
    # millisecond-tie fixture. Facts still come ONLY from the verified timeline.
    time.sleep(0.65)
    url = ready["screen_url"].removesuffix("/screen") + "/api/screen"
    with urlopen(url, timeout=5) as response:
        assert response.status == 200


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "omitted",
        "partial",
        "video_only",
        "empty_video",
        "empty_source",
        "missing",
        "tampered",
        "missing_calibration",
        "missing_source_id",
        "foreign_video",
        "wrong_source",
        "window_only",
    ],
)
def test_normal_cli_source_to_frozen_ownership_and_pending(tmp_path, cli_source, mode):
    video, source = cli_source
    if mode == "window_only":
        path = source / "world-events.json"
        rows = json.loads(path.read_text())
        rows["events"] = [row for row in rows["events"] if row["event_type"] != "shot_start"]
        path.write_text(json.dumps(rows))
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("artifact_id_algorithm", None)
        manifest["inventory"]["world-events.json"] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest))
    artifact = replay_cli(tmp_path, video, source)
    if mode == "tampered":
        (source / "world-events.json").write_text("{}")
    if mode in {"missing_calibration", "missing_source_id"}:
        path = source / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["calibration" if mode == "missing_calibration" else "artifact_id"]
        path.write_text(json.dumps(manifest))
    if mode == "foreign_video":
        video = tmp_path / "foreign.avi"
        video.write_bytes(b"foreign-video")
    if mode == "wrong_source":
        other = run_mainline1_shot_local_video(
            video=video,
            calibration=_calibration(),
            vision_backend=RecordingBackend(),
            identity_classifier=RecordingClassifier(),
            config=replace(_config(), fragment=replace(_config().fragment, ambiguity_mm=4)),
            output_root=tmp_path / "other-source",
        )
        source = other.path
    if mode == "missing":
        source = tmp_path / "absent-source"
    args = ["--video", str(video), "--mainline1-artifact", str(source)]
    if mode == "omitted":
        args = []
    elif mode == "partial":
        args = ["--mainline1-artifact", str(source)]
    elif mode == "video_only":
        args = ["--video", str(video)]
    elif mode == "empty_video":
        args[1] = ""
    elif mode == "empty_source":
        args[3] = ""
    db_path = tmp_path / "test-only.sqlite3"
    with shell(artifact, db_path, args) as ready:
        set_test_participants(ready)
        control(ready, {"command": "start"})
        consume_arrived_timeline(ready)
        store = SQLiteFieldRecordStore(db_path)
        receipt = store.get_shot_ownership(ready["session_id"], "shot-0001")
        assert receipt.evidence_status == ("bound_at_start" if mode == "valid" else "unknown")
        assert receipt.actor_user_id == ("synthetic-cli-red" if mode == "valid" else None)
        assert receipt.source_kind == "development_replay"
        expected_reason = {
            "valid": "SOURCE_START_VERIFIED",
            "omitted": "SOURCE_INPUTS_MISSING",
            "partial": "SOURCE_INPUTS_INCOMPLETE",
            "video_only": "SOURCE_INPUTS_INCOMPLETE",
            "empty_video": "SOURCE_CONTENT_INVALID",
            "empty_source": "SOURCE_CONTENT_INVALID",
            "missing": "SOURCE_CONTENT_INVALID",
            "tampered": "SOURCE_CONTENT_INVALID",
            "missing_calibration": "SOURCE_CONTENT_INVALID",
            "missing_source_id": "SOURCE_CONTENT_INVALID",
            "foreign_video": "SOURCE_CONTENT_INVALID",
            "wrong_source": "SOURCE_LINK_MISMATCH",
            "window_only": "SOURCE_START_UNVERIFIED",
        }
        assert receipt.evidence_reason == expected_reason[mode]
        pending = store.get_turn_confirmation(ready["session_id"], "shot-0001")
        assert pending.status == "pending" and pending.action == "abstain"
        assert pending.resolved_command_id is None
        assert store.load_session(ready["session_id"]).match.revision == 3


def test_cli_restart_never_binds_unprocessed_old_start_to_recovered_roster(tmp_path, cli_source):
    video, source = cli_source
    artifact = replay_cli(tmp_path, video, source)
    args = ["--video", str(video), "--mainline1-artifact", str(source)]
    db_path = tmp_path / "restart.sqlite3"
    with shell(artifact, db_path, args) as ready:
        set_test_participants(ready)
        control(ready, {"command": "start"})
        time.sleep(0.35)  # No GET/command consumed the arrived start before process exit.
        session_id = ready["session_id"]
    with shell(artifact, db_path, args) as ready:
        assert ready["session_id"] == session_id
        control(ready, {"command": "start"})
        consume_arrived_timeline(ready)
        store = SQLiteFieldRecordStore(db_path)
        receipt = store.get_shot_ownership(session_id, "shot-0001")
        assert (
            receipt.actor_user_id is None and receipt.evidence_reason == "TEMPORAL_ORDER_UNPROVEN"
        )
        assert store.get_turn_confirmation(session_id, "shot-0001").status == "pending"


@pytest.mark.parametrize("valid_source", [True, False])
def test_optional_source_cannot_downgrade_mandatory_replay_integrity_failure(
    tmp_path, cli_source, valid_source
):
    from pathlib import Path

    video, source = cli_source
    artifact = replay_cli(tmp_path, video, source)
    (Path(artifact) / "presentation.mp4").write_bytes(b"tampered-replay")
    if not valid_source:
        source = tmp_path / "missing-source"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/serve_field_route_shell.py",
            "--artifact",
            artifact,
            "--state-db",
            str(tmp_path / "forbidden.sqlite3"),
            "--video",
            str(video),
            "--mainline1-artifact",
            str(source),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2 and "READY" not in result.stdout
    assert "field route shell failed" in result.stderr
    assert not (tmp_path / "forbidden.sqlite3").exists()


@pytest.mark.skipif(
    shutil.which("pwsh") is None,
    reason="PowerShell unavailable; wrapper execution not validated on this host",
)
@pytest.mark.parametrize("with_source", [False, True])
def test_windows_wrapper_legacy_optional_sources_cannot_bypass_required_usb_profile(
    tmp_path, with_source
):
    project = tmp_path / "project"
    runner = project / "scripts" / "algorithm" / "serve_mainline1_usb_field.py"
    runner.parent.mkdir(parents=True)
    (project / "src").mkdir()
    runner.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    artifact = tmp_path / "existing-replay"
    artifact.mkdir()
    video = tmp_path / "missing original video.avi"
    source = tmp_path / "missing source artifact"
    args = ["-Video", str(video), "-Mainline1Artifact", str(source)] if with_source else []
    completed = subprocess.run(
        [
            shutil.which("pwsh"),
            "-NoProfile",
            "-File",
            str(REPOSITORY_ROOT / "scripts/algorithm/windows/serve-field-route-shell.ps1"),
            "-ProjectRoot",
            str(project),
            "-Artifact",
            str(artifact),
            "-PythonPath",
            sys.executable,
            "-StateDb",
            str(tmp_path / "state" / "field.sqlite3"),
            *args,
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    # The commercial wrapper requires the installed USB capture contract.
    # Legacy replay/source flags cannot bypass it or resolve absent inputs.
    assert completed.returncode != 0
    assert "requires the installed RER U3 capture" in completed.stderr
    assert "profile" in completed.stderr
    assert "missing original video" not in completed.stderr
    assert "missing source artifact" not in completed.stderr
    assert not any(line.startswith("[") for line in completed.stdout.splitlines())
