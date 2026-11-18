"""Permanent upload evidence, using temporary synthetic selected media only."""
# Imported pytest fixtures are injected by their public names.
# ruff: noqa: F811

import importlib
import sqlite3
import uuid
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import media  # noqa: F401
from tests.unit.edge.route_assistance.test_managed_session_persistence import snapshot
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    owner_module,
    payload,
)


def module():
    return importlib.import_module("billiards_ai.edge.route_assistance.saved_upload_requests")


@pytest.fixture
def scene(media):
    store = media[0]
    item = store.claim_due_saved_media(NOW, lease_seconds=1, managed_scope=SCOPE)
    assert item.shot_id == "shot-0002"
    return store, item


def attempt(sequence=1):
    return dict(
        contract="managed_upload_attempt_v1", operation_id=str(uuid.uuid4()), sequence=sequence
    )


def ledger(scene):
    return module().SavedUploadRequestLedger(scene[0], SCOPE)


def test_permanent_pre_authorize_identity_and_immutable_success(scene):
    store, item = scene
    before = snapshot(store.path)
    with ledger(scene).use(item) as session:
        assert session.current() is None
        row = session.load_or_create("cloud-job-1")
        assert row.state == "requested" and row.upload_attempt is None
        assert str(uuid.UUID(row.request_id)) == row.request_id
        assert session.load_or_create("cloud-job-1") == row
        with pytest.raises(ValueError):
            session.load_or_create("cloud-job-2")
        received = session.record_received(row.request_id, attempt())
        with pytest.raises(TypeError):
            received.upload_attempt["sequence"] = 2
        with pytest.raises(FrozenInstanceError):
            received.state = "started"
        assert session.record_received(row.request_id, dict(received.upload_attempt)) == received
        started = session.record_started(row.request_id)
        assert started.state == "started"
        success = session.record_succeeded(row.request_id, 200)
        assert success.state == "succeeded" and success.terminal_sha256
        store._now_utc = lambda: NOW + timedelta(microseconds=1)
        assert session.record_succeeded(row.request_id, 200) == success
        assert session.current() == success
        for call in (
            lambda: session.record_unknown(row.request_id),
            lambda: session.retire(row.request_id),
            lambda: session.record_succeeded(row.request_id, True),
            lambda: session.record_succeeded(row.request_id, 201),
        ):
            with pytest.raises(ValueError):
                call()
    with pytest.raises(ValueError):
        session.load_or_create("cloud-job-1")
    with ledger(scene).use(item) as session:
        assert session.current() == success
    after = snapshot(store.path)
    changed_tables = {"field_saved_upload_requests", "field_upload_receipt_outbox"}
    assert {k: v for k, v in before.items() if k not in changed_tables} == {
        k: v for k, v in after.items() if k not in changed_tables
    }


def test_retirement_never_reuses_uuid_or_accepts_late_response_and_cap64(scene):
    with ledger(scene).use(scene[1]) as session:
        ids = []
        for _ in range(64):
            row = session.load_or_create("cloud-job-1")
            ids.append(row.request_id)
            retired = session.retire(row.request_id, attempt())
            assert retired.state == "retired"
            assert session.current() is None
            with pytest.raises(ValueError):
                session.record_received(row.request_id, dict(retired.upload_attempt))
            with pytest.raises(ValueError):
                session.record_started(row.request_id)
        assert len(set(ids)) == 64
        with pytest.raises(module().SavedUploadRequestError, match="request_limit"):
            session.load_or_create("cloud-job-1")
    with sqlite3.connect(scene[0].path) as db:
        assert db.execute("SELECT count(*) FROM field_saved_upload_requests").fetchone()[0] == 64


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"sequence": 1},
        dict(contract="managed_upload_attempt_v1", operation_id=str(uuid.uuid4()), sequence=True),
        dict(contract="managed_upload_attempt_v1", operation_id=str(uuid.uuid4()), sequence=65),
        dict(contract="managed_upload_attempt_v1", operation_id="invalid", sequence=1),
    ],
)
def test_strict_attempt_and_transition_rejection(scene, bad):
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        for call in (
            lambda: session.record_received(row.request_id, bad),
            lambda: session.record_started(row.request_id),
            lambda: session.record_succeeded(row.request_id, 200),
            lambda: session.record_unknown(row.request_id),
        ):
            with pytest.raises(ValueError):
                call()
        received = session.record_received(row.request_id, attempt())
        with pytest.raises(ValueError):
            session.record_received(row.request_id, attempt())
        assert session.current() == received


def test_crashed_started_is_unknown_and_cannot_retire_or_allocate(scene):
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        session.record_started(row.request_id)
    with ledger(scene).use(scene[1]) as session:
        unknown = session.current()
        assert unknown.state == "unknown"
        assert session.record_unknown(row.request_id) == unknown
        for call in (
            lambda: session.retire(row.request_id),
            lambda: session.record_started(row.request_id),
            lambda: session.record_succeeded(row.request_id, 200),
        ):
            with pytest.raises(ValueError):
                call()
        assert session.load_or_create("cloud-job-1") == unknown


def test_expired_claim_and_owner_fence_deny_admission_but_preserve_terminal(scene):
    store, item = scene
    with ledger(scene).use(item) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        session.record_started(row.request_id)
        replacement = store.claim_due_saved_media(NOW + timedelta(seconds=2), managed_scope=SCOPE)
        assert replacement.claim_token != item.claim_token
        store.accept_owner_delete_task(
            owner_module().OwnerDeleteDeviceTask.from_payload(payload(store, shot="shot-0002")),
            managed_scope=SCOPE,
        )
        success = session.record_succeeded(row.request_id, 200)
        assert success.state == "succeeded"
        with pytest.raises(ValueError):
            session.load_or_create("cloud-job-1")
        assert session.current() == success


def test_expired_lease_without_reclaim_cannot_start(scene):
    store, item = scene
    with ledger(scene).use(item) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        store._now_utc = lambda: NOW + timedelta(seconds=2)
        with pytest.raises(ValueError):
            session.record_started(row.request_id)
        assert session.current().state == "received"


@pytest.mark.parametrize(
    "change", ["scope", "item", "resource", "epoch", "selection", "descriptor", "manifest"]
)
def test_original_authority_mismatch_fails_closed(scene, change):
    store, item = scene
    chosen_scope = replace(SCOPE, device_id="foreign") if change == "scope" else SCOPE
    if change == "item":
        item = replace(item, byte_size=item.byte_size + 1)
    with sqlite3.connect(store.path) as db:
        if change == "resource":
            db.execute("DELETE FROM field_managed_resource_specs")
        elif change == "epoch":
            db.execute("DELETE FROM field_managed_capture_epochs WHERE source_epoch_id='epoch-2'")
        elif change == "selection":
            db.execute("UPDATE field_clip_save_requests SET receipt_json=receipt_json || ' '")
        elif change == "manifest":
            db.execute("UPDATE field_postgame_manifests SET expected_jobs_json='[]'")
        elif change == "descriptor":
            db.execute("UPDATE field_postgame_clip_availability SET byte_size=byte_size+1")
    with pytest.raises(ValueError):
        with module().SavedUploadRequestLedger(store, chosen_scope).use(item) as session:
            session.load_or_create("cloud-job-1")


def test_fresh_and_schema22_migration_identical_without_backfill(scene, tmp_path):
    store, _ = scene
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=22")
    before = snapshot(store.path)
    SQLiteFieldRecordStore(store.path)
    after = snapshot(store.path)
    assert after.pop("field_recording_ends") == []
    assert after.pop("field_recording_evidence") == []
    assert after.pop("field_session_recording_policy") == []
    assert after.pop("field_rule_fact_preimages") == []
    assert after.pop("field_owner_completion_outbox") == []
    assert after.pop("field_local_media_schedule_state") == []
    assert after.pop("field_upload_receipt_outbox") == []
    assert after.pop("field_saved_upload_requests") == []
    assert after == before
    fresh = SQLiteFieldRecordStore(tmp_path / "fresh.db")
    with sqlite3.connect(store.path) as migrated, sqlite3.connect(fresh.path) as new:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == 30
        assert (
            migrated.execute(
                "SELECT sql FROM sqlite_master WHERE name='field_saved_upload_requests'"
            ).fetchone()
            == new.execute(
                "SELECT sql FROM sqlite_master WHERE name='field_saved_upload_requests'"
            ).fetchone()
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("state", "bogus"),
        ("request_id", "not-uuid"),
        ("binding_sha256", "0" * 64),
        ("upload_attempt_json", '{"sequence":true}'),
        ("terminal_json", '{"status_code":200}'),
        ("created_at_utc", "not-time"),
    ],
)
def test_corrupt_ledger_fails_closed_on_every_read(scene, column, value):
    with ledger(scene).use(scene[1]) as session:
        session.load_or_create("cloud-job-1")
        with sqlite3.connect(scene[0].path) as db:
            db.execute("PRAGMA ignore_check_constraints=ON")
            db.execute(f"UPDATE field_saved_upload_requests SET {column}=?", (value,))
        with pytest.raises(ValueError):
            session.current()
        with pytest.raises(ValueError):
            session.load_or_create("cloud-job-1")


def test_public_session_constructor_does_not_grant_unlocked_authority(scene):
    session = module().SavedUploadRequestSession(ledger(scene), scene[1])
    with pytest.raises(ValueError):
        session.load_or_create("cloud-job-1")


def test_require_current_claim_is_held_atomic_admission_and_released_session_refuses(scene):
    store, item = scene
    with ledger(scene).use(item) as session:
        assert session.require_current_claim() is None
        store._now_utc = lambda: NOW + timedelta(seconds=2)
        with pytest.raises(module().SavedUploadRequestError, match="stale_claim"):
            session.require_current_claim()
    for call in (
        session.current,
        session.require_current_claim,
        lambda: session.record_unknown(str(uuid.uuid4())),
        lambda: session.retire(str(uuid.uuid4())),
    ):
        with pytest.raises(module().SavedUploadRequestError, match="session_released"):
            call()


def test_valid_fractional_terminal_timestamps_compare_as_instants(scene):
    store, item = scene
    with ledger(scene).use(item) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        store._now_utc = lambda: NOW + timedelta(microseconds=1)
        session.record_started(row.request_id)
        store._now_utc = lambda: NOW + timedelta(microseconds=2)
        assert session.record_succeeded(row.request_id, 200).state == "succeeded"


@pytest.mark.parametrize("point", ["create", "received", "started", "claim_check"])
def test_accepted_owner_deletion_fences_all_new_admission(scene, point):
    store, item = scene
    with ledger(scene).use(item) as session:
        if point in {"received", "started"}:
            row = session.load_or_create("cloud-job-1")
        if point == "started":
            session.record_received(row.request_id, attempt())
        value = owner_module().OwnerDeleteDeviceTask.from_payload(payload(store, shot="shot-0002"))
        store.accept_owner_delete_task(value, managed_scope=SCOPE)
        before = snapshot(store.path)
        with pytest.raises(ValueError):
            if point == "create":
                session.load_or_create("cloud-job-1")
            elif point == "received":
                session.record_received(row.request_id, attempt())
            elif point == "started":
                session.record_started(row.request_id)
            else:
                session.require_current_claim()
        assert snapshot(store.path) == before


def test_unknown_is_durable_after_transport_exception_and_claim_loss(scene):
    store, item = scene
    with ledger(scene).use(item) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        session.record_started(row.request_id)
        store._now_utc = lambda: NOW + timedelta(seconds=2)
        unknown = session.record_unknown(row.request_id)
        assert unknown.state == "unknown"
        assert session.current() == unknown
        with pytest.raises(ValueError):
            session.record_succeeded(row.request_id, 200)


@pytest.mark.parametrize(
    "column,value",
    [
        ("terminal_sha256", "0" * 64),
        ("terminal_json", "{}"),
        ("upload_attempt_json", "null"),
        ("started_at_utc", None),
    ],
)
def test_corrupt_terminal_history_never_replays_as_success(scene, column, value):
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        session.record_started(row.request_id)
        session.record_succeeded(row.request_id, 200)
        with sqlite3.connect(scene[0].path) as db:
            db.execute(f"UPDATE field_saved_upload_requests SET {column}=?", (value,))
        with pytest.raises(ValueError):
            session.current()
        with pytest.raises(ValueError):
            session.record_succeeded(row.request_id, 200)


@pytest.mark.parametrize(
    "column,value",
    [
        ("request_id", "33333333-3333-4333-8333-333333333333"),
        ("cloud_job_id", "cloud-job-tampered"),
        ("created_at_utc", "2026-11-01T00:00:00.000001Z"),
    ],
)
def test_validly_shaped_identity_edits_fail_binding_validation(scene, column, value):
    with ledger(scene).use(scene[1]) as session:
        session.load_or_create("cloud-job-1")
        with sqlite3.connect(scene[0].path) as db:
            db.execute(f"UPDATE field_saved_upload_requests SET {column}=?", (value,))
        with pytest.raises(ValueError):
            session.current()


@pytest.mark.parametrize("fault", ["index", "commit"])
def test_schema23_fault_leaves_exact_schema22_bytes_and_can_retry(scene, fault):
    store, _ = scene
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=22")
        schema = db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
    before = snapshot(store.path)

    class FaultConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if fault == "index" and "CREATE UNIQUE INDEX one_active_saved_upload_request" in sql:
                raise sqlite3.OperationalError("synthetic schema23 index fault")
            return super().execute(sql, parameters)

        def commit(self):
            if fault == "commit":
                raise sqlite3.OperationalError("synthetic schema23 commit fault")
            return super().commit()

    class FaultStore(SQLiteFieldRecordStore):
        def _connect(self):
            db = sqlite3.connect(self.path, factory=FaultConnection)
            db.row_factory = sqlite3.Row
            return db

    with pytest.raises(ValueError):
        FaultStore(store.path)
    assert snapshot(store.path) == before
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 22
        assert (
            db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() == schema
        )
    SQLiteFieldRecordStore(store.path)
    after = snapshot(store.path)
    assert after.pop("field_recording_ends") == []
    assert after.pop("field_recording_evidence") == []
    assert after.pop("field_session_recording_policy") == []
    assert after.pop("field_rule_fact_preimages") == []
    assert after.pop("field_owner_completion_outbox") == []
    assert after.pop("field_local_media_schedule_state") == []
    assert after.pop("field_upload_receipt_outbox") == []
    assert after.pop("field_saved_upload_requests") == []
    assert after == before


def test_multiple_active_corruption_is_not_hidden_by_current(scene):
    with ledger(scene).use(scene[1]) as session:
        original = session.load_or_create("cloud-job-1")
        session.retire(original.request_id)
        session.load_or_create("cloud-job-1")
        with sqlite3.connect(scene[0].path) as db:
            db.execute("DROP INDEX one_active_saved_upload_request")
            db.execute("UPDATE field_saved_upload_requests SET state='requested'")
        with pytest.raises(ValueError):
            session.current()


def test_busy_database_is_fixed_retryable_error_without_row_mutation(scene, monkeypatch):
    store, item = scene
    original = store._connect

    def short_timeout():
        db = original()
        db.execute("PRAGMA busy_timeout=1")
        return db

    monkeypatch.setattr(store, "_connect", short_timeout)
    with ledger(scene).use(item) as session:
        row = session.load_or_create("cloud-job-1")
        with sqlite3.connect(store.path) as db:
            db.execute("BEGIN IMMEDIATE")
            with pytest.raises(module().SavedUploadRequestError) as failure:
                session.record_received(row.request_id, attempt())
            assert failure.value.code == "database_busy" and failure.value.retryable
        assert session.current() == row
