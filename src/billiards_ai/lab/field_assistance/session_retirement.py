"""Fail-closed retirement of an installed field session."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import FieldSessionStatus


def _cloud_session_ids_needing_retirement(state_db: Path) -> tuple[str, ...]:
    with sqlite3.connect(state_db.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT sessions.session_id FROM field_sessions AS sessions "
            "JOIN field_cloud_close_intents AS intents USING(session_id) "
            "WHERE intents.status IN ('open','pending') "
            "ORDER BY intents.updated_at_utc, sessions.session_id"
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def retire_existing_session(
    *,
    state_db: Path,
    gateway,
    expected_table_id: str,
) -> dict[str, object]:
    """Close cloud first, then durably end exactly one installed open session."""

    resolved = state_db.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("installed field state database is missing or unsafe")
    store = SQLiteFieldRecordStore(resolved)
    session_ids = _cloud_session_ids_needing_retirement(resolved)
    if not session_ids:
        return {
            "schema_version": "1.0",
            "status": "NOT_NEEDED",
            "session_id": None,
            "table_id": expected_table_id,
        }
    if len(session_ids) != 1:
        raise ValueError("installed field state must contain exactly one open session")
    session_id = session_ids[0]
    lease = gateway.close_field_session(session_id)
    if (
        lease.session_id != session_id
        or lease.table_id != expected_table_id
        or lease.status != "ended"
    ):
        raise OSError("cloud returned an invalid installed-session close receipt")
    if store.load_session(session_id).status is FieldSessionStatus.OPEN:
        store.end_session(session_id)
    if session_id in store.list_pending_cloud_close_session_ids():
        store.complete_cloud_close_intent(session_id)
    return {
        "schema_version": "1.0",
        "status": "RETIRED",
        "session_id": session_id,
        "table_id": expected_table_id,
    }
