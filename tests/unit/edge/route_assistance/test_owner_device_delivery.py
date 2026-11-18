"""Temporary SQLite durability and exact completion wire; no media or network I/O."""
# ruff: noqa: F811

from __future__ import annotations

import importlib
import json
import sqlite3
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    accept,
    canonical,
    journal,
    scene,  # noqa: F401
)

TABLE = "field_owner_completion_outbox"
STATE = "field_local_media_schedule_state"


def module():
    return importlib.import_module("billiards_ai.edge.route_assistance.owner_device_delivery")


def queue(store, scope=SCOPE):
    return module().OwnerDeletionReceiptOutbox(store, scope)


def finish(store):
    accepted = accept(store)
    claim = store.claim_due_owner_delete("worker", NOW)
    assert store.complete_owner_delete(claim, progress_json=journal(claim.clip_id, state="deleted"))
    return store.get_owner_delete(accepted.clip_id)


def originals(store):
    with sqlite3.connect(store.path) as db:
        return db.execute("SELECT * FROM field_owner_media_deletions").fetchall()


def ack(item, **changes):
    return (
        dict(
            contract="owner_delete_device_ack_v1",
            deletion_job_id=item.deletion_job_id,
            clip_id=item.clip_id,
            task_sha256=item.task_sha256,
            cloud_erase_receipt_sha256=item.cloud_erase_receipt_sha256,
            local_receipt_sha256=item.local_receipt_sha256,
            received_at_utc="2026-11-01T00:00:00.000Z",
        )
        | changes
    )


def test_schema24_upgrade_is_real_ddl_and_preserves_original_rows(scene):
    accept(scene)
    before = originals(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(f"DROP TABLE IF EXISTS {TABLE}")
        db.execute(f"DROP TABLE IF EXISTS {STATE}")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=24")
    SQLiteFieldRecordStore(scene.path)
    assert originals(scene) == before
    with sqlite3.connect(scene.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert db.execute(f"SELECT * FROM {TABLE}").fetchall() == []
        assert db.execute(f"SELECT * FROM {STATE}").fetchall() == []


def test_completion_queue_failure_rolls_back_terminal_metadata(scene):
    accepted = accept(scene)
    claim = scene.claim_due_owner_delete("worker", NOW)
    assert scene.update_owner_delete_progress(claim, journal(claim.clip_id))
    before = originals(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(
            f"CREATE TRIGGER reject_queue BEFORE INSERT ON {TABLE} "
            "BEGIN SELECT RAISE(ABORT,'synthetic queue failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        scene.complete_owner_delete(claim, progress_json=journal(accepted.clip_id, state="deleted"))
    assert originals(scene) == before
    with sqlite3.connect(scene.path) as db:
        db.execute("DROP TRIGGER reject_queue")
    assert scene.complete_owner_delete(
        claim, progress_json=journal(accepted.clip_id, state="deleted")
    )
    assert queue(scene).claim_due(NOW) is not None


def test_exact_completion_bytes_and_ack_never_change_original_task_journal_receipt(scene):
    completed = finish(scene)
    before = originals(scene)
    q = queue(scene)
    item = q.claim_due(NOW)
    receipt = json.loads(completed.receipt_json)
    body = canonical(
        dict(
            contract="owner_delete_device_completion_v1",
            task_sha256=completed.task_sha256,
            local_receipt=receipt,
            local_receipt_sha256=completed.receipt_sha256,
        )
    )
    assert item.completion_json == body
    assert item.completion_sha256 == sha256(body.encode("ascii")).hexdigest()
    assert item.local_receipt_json == completed.receipt_json
    assert item.local_receipt_sha256 == completed.receipt_sha256
    assert len(json.loads(completed.task.canonical_json())) == 19
    assert len(receipt) == 7
    assert "path" not in body and "objects" not in body
    assert q.acknowledge(item, ack(item))
    assert originals(scene) == before
    assert q.claim_due(NOW + timedelta(days=2)) is None
    assert q.reconcile().queued == 0
    assert originals(scene) == before
    # Exact ACK replay is idempotent; server's original time cannot be replaced.
    assert q.acknowledge(item, ack(item))
    assert not q.acknowledge(item, ack(item, received_at_utc="2026-11-02T00:00:00.000Z"))


def test_restart_stale_tokens_due_and_independent_attempts(scene):
    completed = finish(scene)
    first = queue(scene).claim_due(NOW, lease_seconds=60)
    restarted = SQLiteFieldRecordStore(scene.path, now_utc=lambda: NOW)
    q = queue(restarted)
    assert q.claim_due(NOW + timedelta(seconds=59)) is None
    second = q.claim_due(NOW + timedelta(seconds=60))
    assert second.claim_token != first.claim_token and second.attempts == 2
    assert not q.acknowledge(first, ack(first))
    assert not q.retry(first, "transport_unavailable", NOW)
    assert not q.block(first, "invalid_ack")
    assert q.retry(second, "transport_unavailable", NOW + timedelta(seconds=120))
    assert q.claim_due(NOW + timedelta(seconds=119)) is None
    third = q.claim_due(NOW + timedelta(seconds=120))
    assert third.attempts == 3
    assert q.block(third, "invalid_ack")
    assert q.claim_due(NOW + timedelta(days=1)) is None
    assert scene.get_owner_delete(completed.clip_id) == completed


@pytest.mark.parametrize(
    "change",
    [
        {"extra": True},
        {"clip_id": "other"},
        {"task_sha256": "f" * 64},
        {"local_receipt_sha256": "f" * 64},
        {"cloud_erase_receipt_sha256": "f" * 64},
        {"contract": "other"},
        {"received_at_utc": "2026-11-01T00:00:00Z"},
        {"received_at_utc": "2026-02-30T00:00:00.000Z"},
    ],
)
def test_ack_requires_exact_schema_and_all_identity_fields(scene, change):
    finish(scene)
    q = queue(scene)
    item = q.claim_due(NOW)
    with pytest.raises(ValueError, match="ack"):
        q.acknowledge(item, ack(item, **change))
    assert q.acknowledge(item, ack(item))


def test_foreign_scope_and_durable_send_receive_turns(scene):
    finish(scene)
    q = queue(scene)
    foreign = queue(scene, replace(SCOPE, table_id="other"))
    assert foreign.claim_due(NOW) is None
    assert foreign.reconcile().scanned == 0
    assert q.next_turn() == "receive"
    assert queue(SQLiteFieldRecordStore(scene.path)).next_turn() == "send"
    assert q.next_turn() == "receive"
    assert foreign.next_turn() == "receive"


def test_reconcile_validates_legacy_completion_and_never_pending_or_blocked(scene):
    accepted = accept(scene)
    q = queue(scene)
    assert q.reconcile().scanned == 0
    claim = scene.claim_due_owner_delete("w", NOW)
    assert scene.block_owner_delete(claim, "UNSAFE")
    assert q.reconcile().queued == 0
    # Construct the historical completed row via the original completion boundary.
    with sqlite3.connect(scene.path) as db:
        db.execute("UPDATE field_owner_media_deletions SET status='pending'")
    claim = scene.claim_due_owner_delete("w", NOW)
    assert scene.complete_owner_delete(
        claim, progress_json=journal(accepted.clip_id, state="deleted")
    )
    before = originals(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(f"DELETE FROM {TABLE}")
    result = queue(SQLiteFieldRecordStore(scene.path)).reconcile()
    assert (result.scanned, result.queued, result.diagnostics) == (1, 1, ())
    assert originals(scene) == before


@pytest.mark.parametrize(
    "column,value",
    [
        ("task_sha256", "f" * 64),
        ("progress_json", "{}"),
        ("receipt_sha256", "f" * 64),
        ("selection_json", "{}"),
    ],
)
def test_reconcile_diagnoses_corrupt_originals_without_rewriting_them(scene, column, value):
    finish(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(f"DELETE FROM {TABLE}")
        db.execute(f"UPDATE field_owner_media_deletions SET {column}=?", (value,))
    before = originals(scene)
    result = queue(scene).reconcile()
    assert (result.scanned, result.queued) == (1, 0)
    assert result.diagnostics == ("invalid_completion",)
    assert originals(scene) == before
    assert queue(scene).claim_due(NOW) is None


def test_reconcile_persisted_cursor_advances_past_more_than_16_bad_rows_and_fixed_upper(scene):
    finish(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(f"DELETE FROM {TABLE}")
        row = db.execute("SELECT * FROM field_owner_media_deletions").fetchone()
        cols = [r[1] for r in db.execute("PRAGMA table_info(field_owner_media_deletions)")]
        db.execute("DELETE FROM field_owner_media_deletions")
        for index in range(33):
            bad = dict(zip(cols, row, strict=True))
            bad.update(
                clip_id=f"bad-{index}",
                deletion_job_id=f"bad-job-{index}",
                shot_id=f"bad-shot-{index}",
                task_json="{}",
            )
            db.execute(
                "INSERT INTO field_owner_media_deletions VALUES("
                + ",".join("?" for _ in cols)
                + ")",
                tuple(bad.values()),
            )
        db.execute(
            "INSERT INTO field_owner_media_deletions VALUES(" + ",".join("?" for _ in cols) + ")",
            row,
        )
    q = queue(scene)
    first = q.reconcile(limit=100)
    assert (first.scanned, first.queued) == (16, 0)
    # A new arrival must not extend the current sweep's finite upper boundary.
    with sqlite3.connect(scene.path) as db:
        bad = dict(zip(cols, row, strict=True))
        bad.update(clip_id="later", deletion_job_id="later", shot_id="later", task_json="{}")
        db.execute(
            "INSERT INTO field_owner_media_deletions VALUES(" + ",".join("?" for _ in cols) + ")",
            tuple(bad.values()),
        )
    assert queue(SQLiteFieldRecordStore(scene.path)).reconcile().scanned == 16
    third = q.reconcile()
    assert (third.scanned, third.queued) == (2, 1)
    assert q.claim_due(NOW) is not None


@pytest.mark.parametrize("value", [-1, True, 9007199254740992, 1.5])
def test_completion_codec_rejects_non_js_safe_reclaimed_bytes(scene, value):
    completed = finish(scene)
    receipt = json.loads(completed.receipt_json) | {"reclaimed_bytes": value}
    with pytest.raises(ValueError):
        module().completion_json(completed.task, receipt)


def test_completion_codec_allows_more_than_one_clip_copy(scene):
    completed = finish(scene)
    receipt = json.loads(completed.receipt_json) | {"reclaimed_bytes": 9007199254740991}
    body = json.loads(module().completion_json(completed.task, receipt))
    assert body["local_receipt"]["reclaimed_bytes"] == 9007199254740991


def test_outbox_errors_are_fixed_codes_without_paths(scene):
    finish(scene)
    q = queue(scene)
    item = q.claim_due(NOW)
    with pytest.raises(ValueError, match="error code"):
        q.block(item, "/private/raw/error")
    with pytest.raises(ValueError, match="error code"):
        q.retry(item, "arbitrary exception", NOW)


def test_schema25_ddl_failure_rolls_back_tables_and_version(scene):
    with sqlite3.connect(scene.path) as db:
        db.execute(f"DROP TABLE {TABLE}")
        db.execute(f"DROP TABLE {STATE}")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=24")

    class FaultConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE TABLE field_local_media_schedule_state" in sql:
                raise sqlite3.OperationalError("synthetic DDL failure")
            return super().execute(sql, parameters)

    class FaultStore(SQLiteFieldRecordStore):
        def _connect(self):
            db = sqlite3.connect(self.path, factory=FaultConnection)
            db.row_factory = sqlite3.Row
            return db

    with pytest.raises(ValueError, match="database is invalid"):
        FaultStore(scene.path)
    with sqlite3.connect(scene.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 24
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
        assert TABLE not in names and STATE not in names


def test_corrupt_outbox_is_quarantined_and_does_not_emit_changed_bytes(scene):
    finish(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(f"UPDATE {TABLE} SET completion_json='{{}}'")
    q = queue(scene)
    assert q.claim_due(NOW) is None
    with sqlite3.connect(scene.path) as db:
        row = db.execute(f"SELECT status,last_error_code FROM {TABLE}").fetchone()
        assert row == ("blocked", "invalid_completion")


@pytest.mark.parametrize("lease", [True, 0, -1, 0.5, float("inf")])
def test_local_lease_requires_positive_whole_seconds(scene, lease):
    finish(scene)
    with pytest.raises(ValueError, match="lease"):
        queue(scene).claim_due(NOW, lease_seconds=lease)
