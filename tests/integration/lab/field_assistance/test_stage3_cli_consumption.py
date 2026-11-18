"""Real replay/shell CLI → SQLite → Stage 3, synthetic source/users only.

Not phone approval, recorded-video ownership, Windows or cloud acceptance.
"""

import hashlib
import json
import sqlite3
import subprocess
import sys

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.integration.lab.field_assistance.test_shell_cli import REPOSITORY_ROOT, _environment
from tests.integration.lab.field_assistance.test_turn_source_cli import (
    cli_source,  # noqa: F401
    consume_arrived_timeline,
    control,
    replay_cli,
    set_test_participants,
    shell,
)
from tests.unit.edge.route_assistance.test_memory_gate import attempt
from tests.unit.lab.field_assistance.test_turn_sources import source_pair  # noqa: F401


def test_normal_cli_accepts_verified_source_for_stage3_consumption(source_pair, tmp_path):  # noqa: F811
    inputs, _, _ = source_pair
    replay = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/run_field_route_replay.py",
            "--video",
            str(inputs.video_path),
            "--mainline1-artifact",
            str(inputs.artifact_path),
            "--context-group",
            "solid",
            "--stable-duration-ms",
            "100",
            "--output-root",
            str(tmp_path / "cli-replay"),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert replay.returncode == 0, replay.stderr
    artifact = json.loads(replay.stdout)["output"]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/serve_field_route_shell.py",
            "--artifact",
            artifact,
            "--video",
            str(inputs.video_path),
            "--mainline1-artifact",
            str(inputs.artifact_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--state-db",
            str(tmp_path / "only-test.sqlite3"),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "READY"


def fact_snapshot(store, session_id):
    # Open sessions must remain resumable; final reports require ended sessions.
    # Compare persisted shot facts without weakening that production contract.
    return store.list_shots(session_id)


@pytest.mark.parametrize("mode", ["valid", "omitted", "invalid", "window_only"])
def test_normal_cli_receipt_is_consumed_without_personal_credit_or_fact_loss(
    tmp_path,
    cli_source,  # noqa: F811
    mode,
):
    video, source = cli_source
    if mode == "window_only":
        events_path = source / "world-events.json"
        payload = json.loads(events_path.read_text())
        payload["events"] = [row for row in payload["events"] if row["event_type"] != "shot_start"]
        events_path.write_text(json.dumps(payload))
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("artifact_id_algorithm", None)
        manifest["inventory"]["world-events.json"] = {
            "bytes": events_path.stat().st_size,
            "sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest))
    artifact = replay_cli(tmp_path, video, source)
    args = ["--video", str(video), "--mainline1-artifact", str(source)]
    if mode == "omitted":
        args = []
    elif mode == "invalid":
        args[-1] = str(tmp_path / "absent-original-source")
    db_path = tmp_path / "synthetic-stage3.sqlite3"
    with shell(artifact, db_path, args) as ready:
        set_test_participants(ready)
        control(ready, {"command": "start"})
        consume_arrived_timeline(ready)
        store = SQLiteFieldRecordStore(db_path)
        session_id = ready["session_id"]
        receipt = store.get_shot_ownership(session_id, "shot-0001")
        assert receipt.evidence_status == ("bound_at_start" if mode == "valid" else "unknown")
        assert receipt.actor_user_id == ("synthetic-cli-red" if mode == "valid" else None)
        assert receipt.source_start_event_id == (
            "shot-0001-event-shot-start" if mode == "valid" else None
        )
        before = fact_snapshot(store, session_id)
        assert len(before) == 1 and before[0].settled_ms is not None
        # attempt invokes the real field service entry with a valid approved
        # fixture and asserts no growth batch or inbox receipt was persisted.
        assert attempt(store, session_id) == (
            "DEVELOPMENT_REPLAY_EXCLUDED" if mode == "valid" else "OWNERSHIP_UNKNOWN"
        )
        assert fact_snapshot(store, session_id) == before
        assert store.get_shot_ownership(session_id, "shot-0001") == receipt
        assert store.get_turn_confirmation(session_id, "shot-0001").status == "pending"


@pytest.mark.parametrize("legacy_bound", [False, True])
def test_cli_restart_with_source_never_upgrades_unknown_or_legacy_ownership(
    tmp_path,
    cli_source,  # noqa: F811
    legacy_bound,
):
    video, source = cli_source
    artifact = replay_cli(tmp_path, video, source)
    valid_args = ["--video", str(video), "--mainline1-artifact", str(source)]
    db_path = tmp_path / "synthetic-restart.sqlite3"
    with shell(artifact, db_path, valid_args if legacy_bound else []) as ready:
        set_test_participants(ready)
        control(ready, {"command": "start"})
        consume_arrived_timeline(ready)
        session_id = ready["session_id"]
    if legacy_bound:
        # Reconstruct M1 persisted JSON only inside this disposable fixture.
        with sqlite3.connect(db_path) as db:
            payload = json.loads(
                db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
            )
            del payload["source_start_event_id"]
            del payload["evidence_reason"]
            db.execute("UPDATE field_shot_ownership SET payload_json=?", (json.dumps(payload),))
    store = SQLiteFieldRecordStore(db_path)
    old = store.get_shot_ownership(session_id, "shot-0001")
    with sqlite3.connect(db_path) as db:
        original_json = db.execute(
            "SELECT payload_json FROM field_shot_ownership WHERE session_id=? AND shot_id=?",
            (session_id, "shot-0001"),
        ).fetchone()[0]
    facts = fact_snapshot(store, session_id)
    with shell(artifact, db_path, valid_args) as ready:
        assert ready["session_id"] == session_id
        control(ready, {"command": "start"})
        consume_arrived_timeline(ready)
        reopened = SQLiteFieldRecordStore(db_path)
        assert reopened.get_shot_ownership(session_id, "shot-0001") == old
        assert old.source_start_event_id is None
        assert attempt(reopened, session_id) == (
            "OWNERSHIP_SOURCE_UNVERIFIED" if legacy_bound else "OWNERSHIP_UNKNOWN"
        )
        assert fact_snapshot(reopened, session_id) == facts
        # Missing M1 keys must stay absent on disk, not be silently filled with
        # null/defaults that deserialize into the same dataclass values.
        with sqlite3.connect(db_path) as db:
            assert (
                db.execute(
                    "SELECT payload_json FROM field_shot_ownership "
                    "WHERE session_id=? AND shot_id=?",
                    (session_id, "shot-0001"),
                ).fetchone()[0]
                == original_json
            )
