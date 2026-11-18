from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore


def test_corrupt_sqlite_header_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"this is not a sqlite database")

    with pytest.raises(ValueError, match="database is invalid"):
        SQLiteFieldRecordStore(
            path,
            now_utc=lambda: datetime(2026, 8, 26, tzinfo=UTC),
            session_id_factory=lambda: "session-1",
        )


def test_version_one_database_migrates_without_losing_sessions(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE field_sessions (
              session_id TEXT PRIMARY KEY,
              artifact_id TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at_utc TEXT NOT NULL,
              ended_at_utc TEXT
            );
            CREATE TABLE field_match_state (
              session_id TEXT PRIMARY KEY,
              payload_json TEXT NOT NULL,
              revision INTEGER NOT NULL
            );
            CREATE TABLE field_shots (
              session_id TEXT NOT NULL,
              shot_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              PRIMARY KEY(session_id, shot_id)
            );
            INSERT INTO field_sessions VALUES(
              'session-1','artifact-1','open','2026-08-26T00:00:00Z',NULL
            );
            INSERT INTO field_match_state VALUES(
              'session-1',
              '{"active_role":null,"blue_group":null,"blue_user_id":null,"breaker":null,"red_group":null,"red_user_id":null,"revision":0}',
              0
            );
            PRAGMA user_version=1;
            """
        )

    reopened = SQLiteFieldRecordStore(path)
    assert reopened.load_session("session-1").artifact_id == "artifact-1"


def test_version_two_database_migrates_and_recovers_command_ledger(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    fixture = Path(__file__).parents[3] / "fixtures/synthetic/field-store-v2.sql"
    with sqlite3.connect(path) as connection:
        connection.executescript(fixture.read_text())
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        } == {
            "field_sessions",
            "field_match_state",
            "field_shots",
            "field_sync_jobs",
            "field_sync_media",
        }

    reopened = SQLiteFieldRecordStore(path)
    result = reopened.apply_field_command(
        FieldCommandEnvelope(
            command_id="cmd-1",
            session_id="session-1",
            actor_user_id="user-alice",
            command_type="claim_role",
            payload={"role": "red"},
            expected_revision=0,
            server_sequence=1,
        ),
        reduce_field_command,
    )

    assert result.outcome == "applied"
    assert reopened.load_session("session-1").match.red_user_id == "user-alice"


def test_failed_v2_upgrade_leaves_a_restartable_v2_database(tmp_path: Path, monkeypatch) -> None:
    from billiards_ai.edge.route_assistance import persistence

    path = tmp_path / "field.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE field_sessions (
              session_id TEXT PRIMARY KEY,
              artifact_id TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at_utc TEXT NOT NULL,
              ended_at_utc TEXT
            );
            CREATE TABLE field_match_state (
              session_id TEXT PRIMARY KEY,
              payload_json TEXT NOT NULL,
              revision INTEGER NOT NULL
            );
            CREATE TABLE field_shots (
              session_id TEXT NOT NULL,
              shot_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              PRIMARY KEY(session_id, shot_id)
            );
            PRAGMA user_version=1;
            """
        )

    command_schema = persistence._COMMAND_LEDGER_SCHEMA
    monkeypatch.setattr(persistence, "_COMMAND_LEDGER_SCHEMA", command_schema + "INVALID SQL;")
    with pytest.raises(ValueError):
        SQLiteFieldRecordStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='field_sync_jobs'"
        ).fetchone() == (1,)
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='field_applied_commands'"
            ).fetchone()
            is None
        )

    monkeypatch.setattr(persistence, "_COMMAND_LEDGER_SCHEMA", command_schema)
    assert SQLiteFieldRecordStore(path)


def test_recovery_state_survives_restart_and_is_not_filtered_by_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    ids = iter(("session-a", "session-b"))
    store = SQLiteFieldRecordStore(path, session_id_factory=lambda: next(ids))
    first = store.open_or_resume_session(artifact_id="artifact-a")
    store.bind_artifact_path("artifact-a", tmp_path / "artifact-a")
    store.end_session(first.session_id)
    second = store.open_or_resume_session(artifact_id="artifact-b")
    store.bind_artifact_path("artifact-b", tmp_path / "artifact-b")
    store.end_session(second.session_id)
    store.enqueue_cloud_close_intent(first.session_id)

    reopened = SQLiteFieldRecordStore(path)

    assert [session.session_id for session in reopened.list_ended_sessions_without_sync()] == [
        "session-a",
        "session-b",
    ]
    assert reopened.artifact_path_for_session("session-a") == (tmp_path / "artifact-a").resolve()
    assert reopened.list_pending_cloud_close_session_ids() == ("session-a",)
    reopened.complete_cloud_close_intent("session-a")
    assert SQLiteFieldRecordStore(path).list_pending_cloud_close_session_ids() == ()


def test_version_ten_finalization_backfills_local_postgame_outbox(tmp_path: Path) -> None:
    """Catches a restart after v10 finalization losing the local-only handoff."""
    path = tmp_path / "v10.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE field_sessions(session_id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL,
              status TEXT NOT NULL, created_at_utc TEXT NOT NULL, ended_at_utc TEXT);
            CREATE TABLE field_postgame_finalizations(session_id TEXT PRIMARY KEY,
              manifest_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL);
            CREATE TABLE field_postgame_manifests(session_id TEXT PRIMARY KEY,
              ended_at_utc TEXT NOT NULL, deadline_at_utc TEXT NOT NULL,
              expected_jobs_json TEXT NOT NULL);
            INSERT INTO field_sessions VALUES(
              'session-1','artifact-1','ended','2026-01-01T00:00:00Z','2026-01-01T00:01:00Z');
            INSERT INTO field_postgame_finalizations VALUES(
              'session-1','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','{}');
            INSERT INTO field_postgame_manifests VALUES(
              'session-1','2026-01-01T00:01:00Z','2026-01-01T00:03:00Z','[]');
            PRAGMA user_version=10;
            """
        )
    SQLiteFieldRecordStore(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT status FROM local_postgame_outbox").fetchone()
        assert row[0] == "pending"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 30
