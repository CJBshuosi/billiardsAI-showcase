"""SQL-scoped metadata scheduling; returned candidates confer no erasure authority."""
# ruff: noqa: F811

from __future__ import annotations

import importlib
import sqlite3
from dataclasses import replace
from datetime import timedelta

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    SESSION,
    accept,
    scene,  # noqa: F401
)


def scheduler(store, scope=SCOPE, **kwargs):
    return importlib.import_module(
        "billiards_ai.edge.route_assistance.local_media_schedule"
    ).LocalMediaCleanupScheduler(store, scope, **kwargs)


def inject(
    store, kind, session, *, status="pending", due="1970-01-01T00:00:00Z", lease=None, scope=SCOPE
):
    """Independent synthetic rows include intentionally missing executor context."""
    with sqlite3.connect(store.path) as db:
        if session != SESSION and scope is not None:
            cols = [r[1] for r in db.execute("PRAGMA table_info(field_managed_open_requests)")]
            original = db.execute("SELECT * FROM field_managed_open_requests LIMIT 1").fetchone()
            row = dict(zip(cols, original, strict=True))
            row.update(
                environment_id=scope.environment_id,
                device_id=scope.device_id,
                table_id=scope.table_id,
                session_id=session,
                birth_id=session,
                open_request_id=session,
                artifact_id=session,
                lifecycle="retired",
                remote_status="ended",
                retired_at_utc=due,
                sealed_close_json="{}",
                reservation_order=db.execute(
                    "SELECT COUNT(*) FROM field_managed_open_requests"
                ).fetchone()[0]
                + 2,
            )
            db.execute(
                "INSERT OR IGNORE INTO field_managed_open_requests VALUES("
                + ",".join("?" for _ in cols)
                + ")",
                tuple(row.values()),
            )
        if kind == "owner":
            db.execute(
                "INSERT INTO field_owner_media_deletions "
                "(clip_id,deletion_job_id,session_id,shot_id,task_json,task_sha256,local_path,"
                "selection_json,status,attempts,next_attempt_at_utc,created_at_utc,"
                "lease_until_utc) "
                "VALUES(?,?,?,?,?,?,?,?,?,0,?,?,?)",
                (
                    session,
                    session,
                    session,
                    session,
                    "{}",
                    "a" * 64,
                    "/synthetic",
                    "{}",
                    status,
                    due,
                    due,
                    lease,
                ),
            )
        else:
            db.execute(
                "INSERT INTO field_local_media_discards "
                "(session_id,shot_id,postgame_manifest_sha256,job_id,artifact_id,clip_id,"
                "local_path,content_sha256,byte_size,content_type,status,attempts,"
                "next_attempt_at_utc,lease_until_utc) VALUES(?,?,?,?,?,?,?,?,1,?,?,0,?,?)",
                (
                    session,
                    session,
                    "a" * 64,
                    session,
                    "b" * 64,
                    session,
                    "/synthetic",
                    "c" * 64,
                    "video/mp4",
                    status,
                    due,
                    lease,
                ),
            )


def test_one_candidate_advances_before_resolver_and_restart_does_not_restart_head(scene):
    accept(scene)
    for i in range(35):
        inject(scene, "owner", f"managed-{i:03d}")
    first = scheduler(scene).next_candidate(NOW)
    assert (first.kind, first.session_id) == ("owner", "managed-000")
    restarted = SQLiteFieldRecordStore(scene.path)
    second = scheduler(restarted).next_candidate(NOW)
    assert second.session_id == "managed-001"
    # Simulate missing/bad resource resolution for the first 33 sessions.
    later = [scheduler(scene).next_candidate(NOW).session_id for _ in range(33)]
    assert later[-1] == "managed-034"
    assert scheduler(scene).next_candidate(NOW).session_id == SESSION


def test_owner_and_discard_alternate_persistently_despite_bad_contexts(scene):
    accept(scene)
    inject(scene, "discard", "managed-discard")
    for i in range(20):
        inject(scene, "owner", f"managed-{i:03d}")
    picks = [scheduler(SQLiteFieldRecordStore(scene.path)).next_candidate(NOW) for _ in range(6)]
    assert [p.kind for p in picks] == ["owner", "discard"] * 3
    assert [p.session_id for p in picks[::2]] == [f"managed-{i:03d}" for i in range(3)]


def test_due_and_lease_filters_happen_in_sql_and_leave_originals_unchanged(scene):
    accept(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(
            "UPDATE field_owner_media_deletions SET status='leased',lease_until_utc=?",
            ("2026-11-02T00:00:00Z",),
        )
    inject(scene, "owner", "managed-a-future", due="2026-11-02T00:00:00Z")
    inject(scene, "owner", "managed-b-leased", status="leased", lease="2026-11-02T00:00:00Z")
    inject(scene, "owner", "managed-c-blocked", status="blocked")
    inject(scene, "owner", "managed-d-completed", status="completed")
    inject(scene, "owner", "managed-z-due", status="retry")
    with sqlite3.connect(scene.path) as db:
        before = db.execute("SELECT * FROM field_owner_media_deletions").fetchall()
    s = scheduler(scene)
    assert s.next_candidate(NOW).session_id == "managed-z-due"
    assert s.next_candidate(NOW).session_id == "managed-z-due"
    assert s.next_candidate(NOW + timedelta(days=1)).session_id == SESSION
    assert s.next_candidate(NOW + timedelta(days=1)).session_id == "managed-a-future"
    with sqlite3.connect(scene.path) as db:
        assert db.execute("SELECT * FROM field_owner_media_deletions").fetchall() == before


def test_scope_filters_both_kinds_and_uses_no_unbounded_discovery(scene, monkeypatch):
    accept(scene)
    inject(scene, "owner", "managed-a-foreign", scope=replace(SCOPE, table_id="foreign"))
    inject(scene, "discard", "managed-b-foreign", scope=replace(SCOPE, table_id="foreign"))
    inject(scene, "discard", "managed-own")
    inject(scene, "discard", "ordinary", scope=None)

    def unbounded(*args, **kwargs):
        raise AssertionError("must select in SQL")

    monkeypatch.setattr(scene, "list_pending_owner_delete_session_ids", unbounded)
    monkeypatch.setattr(scene, "list_pending_local_discard_session_ids", unbounded)
    s = scheduler(scene)
    assert s.next_candidate(NOW).session_id == SESSION
    assert s.next_candidate(NOW).session_id == "managed-own"
    foreign = scheduler(scene, replace(SCOPE, table_id="foreign"))
    assert foreign.next_candidate(NOW).session_id == "managed-a-foreign"
    assert foreign.next_candidate(NOW).session_id == "managed-b-foreign"


def test_no_scope_does_not_adopt_owner_and_preserves_discard_exclusion(scene):
    accept(scene)
    inject(scene, "discard", "managed-own")
    inject(scene, "discard", "ordinary", scope=None)
    assert scheduler(scene, None).next_candidate(NOW).session_id == "managed-own"
    pick = scheduler(scene, None, exclude_managed=True).next_candidate(NOW)
    assert (pick.kind, pick.session_id) == ("discard", "ordinary")
    # A retained binding marks managed data even without the managed- prefix.
    inject(scene, "discard", "aaa-retained")
    assert scheduler(scene, None, exclude_managed=True).next_candidate(NOW).session_id == "ordinary"


def test_fixed_upper_defers_new_arrivals_until_wrap_without_resetting_cursor(scene):
    accept(scene)
    inject(scene, "owner", "managed-a")
    s = scheduler(scene)
    assert s.next_candidate(NOW).session_id == SESSION
    inject(scene, "owner", "managed-z-arrived-later")
    assert s.next_candidate(NOW).session_id == "managed-a"
    # Exhaust finite sweep, wrap and only then admit the new upper.
    assert s.next_candidate(NOW).session_id == SESSION
    assert s.next_candidate(NOW).session_id == "managed-a"
    assert s.next_candidate(NOW).session_id == "managed-z-arrived-later"
