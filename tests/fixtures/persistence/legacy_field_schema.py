"""Build isolated historical databases without running historical product Python."""

import sqlite3
from pathlib import Path

_BASE_COLUMNS = {
    "field_sessions": "session_id artifact_id status created_at_utc ended_at_utc",
    "field_match_state": "session_id payload_json revision",
    "field_shots": "session_id shot_id sequence payload_json",
    "field_sync_jobs": (
        "sync_job_id session_id manifest_hash manifest_json status retry_count "
        "next_attempt_at_utc last_error_code created_at_utc updated_at_utc"
    ),
    "field_sync_media": (
        "sync_job_id clip_id shot_id local_path content_type content_sha256 byte_size status"
    ),
    "field_applied_commands": (
        "command_id session_id server_sequence actor_user_id command_type payload_digest "
        "outcome result_code resulting_revision completed_at_utc"
    ),
    "field_admin_audit": (
        "event_id session_id command_type reason payload_json resulting_revision created_at_utc"
    ),
}
_OWNERSHIP_COLUMNS = "session_id shot_id payload_json"
_V7_COLUMNS = {
    "field_turn_confirmations": (
        "session_id shot_id evidence_digest match_revision source_mainline1_artifact_id "
        "status stale_reason resolved_command_id created_at_utc"
    ),
    "field_session_artifacts": "session_id artifact_path",
    "field_cloud_close_intents": "session_id status updated_at_utc",
}


def assert_legacy_schema(db: sqlite3.Connection, version: int) -> None:
    """Check the independent historical table and column boundary before migration."""
    assert version in (4, 5, 7)
    expected = dict(_BASE_COLUMNS)
    if version >= 5:
        expected["field_shot_ownership"] = _OWNERSHIP_COLUMNS
    if version == 7:
        expected.update(_V7_COLUMNS)
    assert db.execute("PRAGMA user_version").fetchone()[0] == version
    assert {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    } == set(expected)
    for table, columns in expected.items():
        assert tuple(row[1] for row in db.execute(f"PRAGMA table_info({table})")) == tuple(
            columns.split()
        )


def legacy_rows(db: sqlite3.Connection, *, ownership: bool) -> dict[str, list[tuple]]:
    """Read only the original session, match, start and optional ownership columns."""
    columns = {
        table: _BASE_COLUMNS[table]
        for table in ("field_sessions", "field_match_state", "field_shots")
    }
    if ownership:
        columns["field_shot_ownership"] = _OWNERSHIP_COLUMNS
    return {
        table: db.execute(f"SELECT {','.join(names.split())} FROM {table} ORDER BY 1, 2").fetchall()
        for table, names in columns.items()
    }


def create_legacy_database(source: Path, target: Path, version: int) -> dict[str, list[tuple]]:
    """Copy explicit legacy rows unchanged into a separate, genuine historical schema."""
    assert version in (4, 5, 7)
    assert not target.exists()
    with sqlite3.connect(source) as db:
        before = legacy_rows(db, ownership=version >= 5)
    with sqlite3.connect(target) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(Path(__file__).with_name(f"field-schema-v{version}.sql").read_text())
        assert_legacy_schema(db, version)
        for table, rows in before.items():
            columns = (
                _OWNERSHIP_COLUMNS if table == "field_shot_ownership" else _BASE_COLUMNS[table]
            )
            names = columns.split()
            db.executemany(
                f"INSERT INTO {table} ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                rows,
            )
        assert legacy_rows(db, ownership=version >= 5) == before
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    return before
