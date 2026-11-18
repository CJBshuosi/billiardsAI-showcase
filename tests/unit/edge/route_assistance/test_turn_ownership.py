import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import FieldShotRecord, ShotMediaRange
from tests.fixtures.persistence.legacy_field_schema import (
    assert_legacy_schema,
    create_legacy_database,
    legacy_rows,
)


def make_store(tmp_path):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.db",
        session_id_factory=lambda: "session-1",
        now_utc=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )
    store.open_or_resume_session(artifact_id="replay-1")
    return store


def match(revision=1, role=WristbandRole.RED):
    return MatchControlSnapshot(
        red_user_id="user-red",
        blue_user_id="user-blue",
        breaker=WristbandRole.RED,
        active_role=role,
        revision=revision,
    )


def shot(sequence=1, role=WristbandRole.RED):
    return FieldShotRecord(
        session_id="session-1",
        shot_id=f"shot-{sequence:04d}",
        sequence=sequence,
        actor_role=role,
        started_ms=100 * sequence,
        settled_ms=None,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=ShotMediaRange("replay-1", "presentation.mp4", "development_replay", 0, 10000),
    )


def test_start_freezes_user_role_revision_and_separate_source_service_times(tmp_path):
    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    start = shot()
    store.record_shot_start(
        start,
        source_mainline1_artifact_id="a" * 64,
        source_start_event_id="shot-0001-event-shot-start",
    )
    receipt = store.get_shot_ownership("session-1", "shot-0001")
    assert receipt.actor_user_id == "user-red"
    assert receipt.actor_role is WristbandRole.RED
    assert receipt.match_revision == 1
    assert receipt.source_started_ms == 100
    assert receipt.recorded_at_utc == "2026-08-31T00:00:00Z"
    assert receipt.evidence_status == "bound_at_start"
    assert receipt.source_start_event_id == "shot-0001-event-shot-start"
    assert receipt.source_kind == "development_replay"
    assert receipt.revision == 1 and receipt.supersedes_revision is None
    store.save_match("session-1", match(2, WristbandRole.BLUE), expected_revision=1)
    store.record_shot_start(start, source_mainline1_artifact_id="a" * 64)
    store.record_shot_settle(replace(start, settled_ms=200))
    store.record_shot_start(start, source_mainline1_artifact_id="a" * 64)
    restarted = SQLiteFieldRecordStore(store.path)
    assert restarted.get_shot_ownership("session-1", "shot-0001") == receipt


@pytest.mark.parametrize("unknown_cause", ["missing_user", "unverified_source", "ambiguous_time"])
def test_unknown_owner_cannot_be_filled_from_later_roster(tmp_path, unknown_cause):
    store = make_store(tmp_path)
    if unknown_cause != "missing_user":
        store.save_match("session-1", match(), expected_revision=0)
    source = None if unknown_cause == "unverified_source" else "a" * 64
    store.record_shot_start(
        shot(),
        source_mainline1_artifact_id=source,
        ownership_ambiguous=unknown_cause == "ambiguous_time",
    )
    receipt = store.get_shot_ownership("session-1", "shot-0001")
    assert receipt.evidence_status == "unknown"
    assert receipt.actor_user_id is None and receipt.actor_role is None
    revision = store.load_session("session-1").match.revision
    store.save_match("session-1", match(revision + 1), expected_revision=revision)
    store.record_shot_start(shot(), source_mainline1_artifact_id=source)
    assert store.get_shot_ownership("session-1", "shot-0001") == receipt


def test_changed_source_and_settle_actor_cannot_rewrite_start(tmp_path):
    store = make_store(tmp_path)
    store.record_shot_start(shot(), source_mainline1_artifact_id="a" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        store.record_shot_start(shot(), source_mainline1_artifact_id="b" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        store.record_shot_settle(replace(shot(), actor_role=WristbandRole.BLUE, settled_ms=200))


def test_receipt_insert_failure_rolls_back_shot_start(tmp_path):
    store = make_store(tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON field_shot_ownership "
            "BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END"
        )
    with pytest.raises(ValueError):
        store.record_shot_start(shot(), source_mainline1_artifact_id="a" * 64)
    assert store.list_shots("session-1") == ()


def test_v4_upgrade_does_not_backfill_legacy_shots_or_rewrite_payload(tmp_path):
    store = make_store(tmp_path)
    store.record_shot_start(shot())
    session = store.load_session("session-1")
    legacy_path = tmp_path / "field-v4.db"
    before = create_legacy_database(store.path, legacy_path, 4)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 4)
    upgraded = SQLiteFieldRecordStore(legacy_path)
    assert upgraded.load_session("session-1") == session
    assert upgraded.get_shot_ownership("session-1", "shot-0001") is None
    upgraded.record_shot_start(shot())
    assert upgraded.get_shot_ownership("session-1", "shot-0001") is None
    with sqlite3.connect(legacy_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert legacy_rows(db, ownership=False) == before
        assert db.execute("SELECT * FROM field_shot_ownership").fetchall() == []


def test_failed_v4_upgrade_rolls_back_ddl_and_keeps_legacy_rows(tmp_path, monkeypatch):
    from billiards_ai.edge.route_assistance import persistence

    store = make_store(tmp_path)
    store.record_shot_start(shot())
    legacy_path = tmp_path / "field-v4.db"
    before = create_legacy_database(store.path, legacy_path, 4)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 4)
    monkeypatch.setattr(
        persistence,
        "_OWNERSHIP_SCHEMA",
        persistence._OWNERSHIP_SCHEMA + "CREATE TABLE broken syntax;",
    )
    with pytest.raises(ValueError):
        SQLiteFieldRecordStore(legacy_path)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 4)
        assert (
            db.execute(
                "SELECT name FROM sqlite_master WHERE name='field_shot_ownership'"
            ).fetchall()
            == []
        )
        assert legacy_rows(db, ownership=False) == before


def test_foreign_media_artifact_cannot_bind_ownership(tmp_path):
    store = make_store(tmp_path)
    foreign = replace(shot(), media=replace(shot().media, artifact_id="other-replay"))
    with pytest.raises(ValueError, match="artifact"):
        store.record_shot_start(foreign, source_mainline1_artifact_id="a" * 64)
    assert store.list_shots("session-1") == ()


def test_source_id_without_verified_event_stays_unknown(tmp_path):
    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    store.record_shot_start(shot(), source_mainline1_artifact_id="a" * 64)
    receipt = store.get_shot_ownership("session-1", "shot-0001")
    assert receipt.evidence_status == "unknown" and receipt.actor_user_id is None
    assert receipt.source_start_event_id is None
    with pytest.raises(ValueError, match="conflicting"):
        store.record_shot_start(
            shot(),
            source_mainline1_artifact_id="a" * 64,
            source_start_event_id="shot-0001-event-shot-start",
        )
    assert store.get_shot_ownership("session-1", "shot-0001") == receipt


def test_legacy_ownership_read_keeps_absent_source_evidence_absent(tmp_path):
    import json

    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    store.record_shot_start(
        shot(),
        source_mainline1_artifact_id="a" * 64,
        source_start_event_id="shot-0001-event-shot-start",
    )
    with sqlite3.connect(store.path) as db:
        payload = json.loads(
            db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
        )
        del payload["source_start_event_id"]
        del payload["evidence_reason"]
        raw = json.dumps(payload)
        db.execute("UPDATE field_shot_ownership SET payload_json=?", (raw,))
    legacy = store.get_shot_ownership("session-1", "shot-0001")
    assert legacy.evidence_status == "bound_at_start"  # historical claim preserved, not upgraded
    assert legacy.source_start_event_id is None and legacy.evidence_reason is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0] == raw
