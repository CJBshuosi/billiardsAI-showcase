"""Concurrent old-schema openers retain one metadata-only selection migration."""

import sqlite3
import threading

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore


def test_schema26_opener_accepts_another_opener_already_at_schema29(tmp_path, monkeypatch):
    path = tmp_path / "field.sqlite3"
    store = SQLiteFieldRecordStore(path)
    session = store.open_or_resume_session(artifact_id="old-legacy")
    with sqlite3.connect(path) as db:
        for table in (
            "field_original_clip_sources",
            "field_recording_cleanup",
            "field_recording_ends",
            "field_recording_evidence",
            "field_session_recording_policy",
        ):
            db.execute("DROP TABLE " + table)
        db.execute("PRAGMA user_version=26")
    entered, release = threading.Event(), threading.Event()
    original = SQLiteFieldRecordStore._migrate_recording27
    failures = []

    def wait_for_other_opener(db):
        if threading.current_thread().name == "first-schema-opener":
            entered.set()
            assert release.wait(5)
        original(db)

    monkeypatch.setattr(
        SQLiteFieldRecordStore, "_migrate_recording27", staticmethod(wait_for_other_opener)
    )

    def open_store():
        try:
            SQLiteFieldRecordStore(path)
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=open_store, name="first-schema-opener")
    first.start()
    assert entered.wait(5)
    try:
        SQLiteFieldRecordStore(path)
    finally:
        release.set()
        first.join(5)
    assert not first.is_alive()
    assert failures == []
    reopened = SQLiteFieldRecordStore(path)
    assert reopened.get_session_policy(session.session_id) == "legacy"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert db.execute("SELECT COUNT(*) FROM field_recording_cleanup").fetchone()[0] == 0
