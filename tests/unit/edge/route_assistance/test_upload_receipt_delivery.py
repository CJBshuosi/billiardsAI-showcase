"""Receipt transport metadata uses only real temporary SQLite and synthetic files."""
# Imported pytest fixtures are injected by their public names.
# ruff: noqa: F811

import hashlib
import importlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionBinding
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotFact,
    LiveShotMediaReference,
    ShotMediaRange,
)
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import (
    media,  # noqa: F401
    resource_spec,
)
from tests.unit.edge.route_assistance.test_managed_session_persistence import snapshot
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    BIRTH,
    NOW,
    SCOPE,
    SESSION,
    owner_module,
    payload,
)
from tests.unit.edge.route_assistance.test_saved_upload_requests import (
    attempt,
    ledger,
    scene,  # noqa: F401
)


@pytest.fixture
def metadata_store(tmp_path):
    """Two genuine selected clips with retained resources/epochs, no file dependency."""
    resource = resource_spec(tmp_path, tmp_path / "artifacts", tmp_path / "jobs")
    store = SQLiteFieldRecordStore(tmp_path / "field.db", now_utc=lambda: NOW)
    request = store.reserve_managed_open(
        SCOPE,
        artifact_id=resource.artifact_id,
        rule_profile_id=None,
        resource_spec=resource,
    )
    store.record_managed_binding(
        SCOPE,
        request.open_request_id,
        ManagedSessionBinding(
            "1.0",
            SESSION,
            BIRTH,
            "managed_session_birth_v1",
            "table-1",
            "device-1",
            "open",
        ),
    )
    session = store.materialize_managed_session(SCOPE, request.open_request_id)
    store.save_match(
        SESSION,
        replace(
            session.match,
            red_user_id="owner-red",
            blue_user_id="owner-blue",
            active_role=WristbandRole.RED,
            revision=1,
        ),
        expected_revision=0,
    )
    for sequence, role in ((1, WristbandRole.RED), (2, WristbandRole.BLUE)):
        if sequence == 2:
            store.save_match(
                SESSION,
                replace(store.load_session(SESSION).match, active_role=role, revision=2),
                expected_revision=1,
            )
        epoch = f"epoch-{sequence}"
        store.register_managed_capture_epoch(SCOPE, request.open_request_id, epoch)
        shot = FieldShotRecord(
            SESSION,
            f"shot-{sequence:04d}",
            sequence,
            role,
            sequence * 1000,
            None,
            None,
            None,
            None,
            ShotMediaRange(
                resource.artifact_id,
                f"usb:{epoch}",
                "usb_live",
                sequence * 1000,
                sequence * 1000 + 1,
                epoch,
            ),
        )
        store.record_shot_start(shot, source_start_event_id=f"start-{sequence}")
        store.record_shot_settle(
            replace(
                shot,
                settled_ms=sequence * 1000 + 500,
                media=replace(shot.media, end_ms=sequence * 1000 + 500),
                live_job_id=f"job-{sequence}",
                live_evidence_digest="d" * 64,
            ),
            source_epoch_id=epoch,
            live_fact=LiveShotFact.create(
                session_id=SESSION,
                shot_id=shot.shot_id,
                sequence=sequence,
                actor_role=role,
                actor_user_id="owner-red" if sequence == 1 else "owner-blue",
                start_frame_id=sequence * 100,
                settle_frame_id=sequence * 100 + 10,
                mode="observation",
                observed_outcome="unknown",
                rule_revision=sequence,
                recording_reference=f"session-recording:{epoch}:100-110",
            ),
        )
        store.attach_live_shot_media(
            LiveShotMediaReference(
                SESSION,
                shot.shot_id,
                epoch,
                sequence,
                f"job-{sequence}",
                str(tmp_path / f"raw-{sequence}"),
                "d" * 64,
            )
        )
    store.end_session(SESSION)
    manifest = store.get_postgame_manifest(SESSION)
    for sequence in (1, 2):
        store.record_postgame_clip_available(
            session_id=SESSION,
            shot_id=f"shot-{sequence:04d}",
            job_id=f"job-{sequence}",
            artifact_id=str(sequence) * 64,
            local_path=tmp_path / f"clip-{sequence}.mp4",
            content_sha256="c" * 64,
            byte_size=24,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        store.save_postgame_clip(
            request_id=f"save-{sequence}",
            session_id=SESSION,
            shot_id=f"shot-{sequence:04d}",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
    return store


def delivery():
    return importlib.import_module("billiards_ai.edge.route_assistance.upload_receipt_delivery")


def outbox(scene, scope=SCOPE):
    return delivery().UploadReceiptOutbox(scene[0], scope)


def rows(store):
    with sqlite3.connect(store.path) as db:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='field_upload_receipt_outbox'"
        ).fetchone():
            return []
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM field_upload_receipt_outbox")]


def terminal(scene, *, success=False):
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        if not success:
            return session.retire(row.request_id)
        session.record_received(row.request_id, attempt())
        session.record_started(row.request_id)
        scene[0]._now_utc = lambda: NOW + timedelta(microseconds=123456)
        return session.record_succeeded(row.request_id, 200)


def ack(item):
    return dict(
        contract="managed_upload_receipt_ack_v1",
        upload_request=dict(contract="managed_upload_request_v1", request_id=item.request_id),
        receipt_sha256=item.receipt_sha256,
        received_at_utc="2026-11-01T00:00:00.001Z",
        erasable=False,
    )


@pytest.mark.parametrize("success", [False, True])
def test_terminal_transition_atomically_enqueues_exact_wire_and_preserves_originals(scene, success):
    before = snapshot(scene[0].path)
    record = terminal(scene, success=success)
    queued = rows(scene[0])
    assert len(queued) == 1, "terminal transition must enqueue in its own transaction"
    value = json.loads(queued[0]["receipt_json"])
    evidence = dict(kind="retired_before_start", local_binding_sha256=record.binding_sha256)
    if success:
        evidence = dict(
            kind="http200",
            local_binding_sha256=record.binding_sha256,
            started_at_utc="2026-11-01T00:00:00.000000Z",
            received_at_utc="2026-11-01T00:00:00.123456Z",
            status_code=200,
        )
    assert value == dict(
        contract="managed_upload_receipt_v1",
        upload_request=dict(contract="managed_upload_request_v1", request_id=record.request_id),
        upload_attempt=None if record.upload_attempt is None else dict(record.upload_attempt),
        evidence=evidence,
    )
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert queued[0]["receipt_json"] == raw
    assert queued[0]["receipt_sha256"] == hashlib.sha256(raw.encode("ascii")).hexdigest()
    with ledger(scene).use(scene[1]) as session:
        assert session.get(record.request_id) == record
        assert (
            session.record_succeeded(record.request_id, 200)
            if success
            else session.retire(record.request_id)
        ) == record
    assert rows(scene[0]) == queued
    after = snapshot(scene[0].path)
    for name in before.keys() - {"field_saved_upload_requests", "field_upload_receipt_outbox"}:
        assert after[name] == before[name]
    with sqlite3.connect(scene[0].path) as db:
        assert db.execute(
            "SELECT terminal_json,terminal_sha256 FROM field_saved_upload_requests"
        ).fetchone() == (record.terminal_json, record.terminal_sha256)


@pytest.mark.parametrize("success", [False, True])
def test_queue_insertion_failure_rolls_back_terminal_transition(scene, success):
    with sqlite3.connect(scene[0].path) as db:
        db.execute(
            "CREATE TRIGGER receipt_fault BEFORE INSERT ON field_upload_receipt_outbox "
            "BEGIN SELECT RAISE(ABORT,'synthetic receipt fault'); END"
        )
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        if success:
            session.record_received(row.request_id, attempt())
            session.record_started(row.request_id)
        before = snapshot(scene[0].path)
        with pytest.raises(ValueError):
            if success:
                session.record_succeeded(row.request_id, 200)
            else:
                session.retire(row.request_id)
        assert snapshot(scene[0].path) == before


def test_started_unknown_requested_and_received_never_enqueue(scene):
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        assert rows(scene[0]) == []
        session.record_received(row.request_id, attempt())
        assert rows(scene[0]) == []
        session.record_started(row.request_id)
        assert rows(scene[0]) == []
        session.record_unknown(row.request_id)
    assert outbox(scene).reconcile().scanned == 0
    assert outbox(scene).claim_due(NOW) is None


def test_retired_attempt_and_evidence_lookup_never_grant_admission(scene):
    with ledger(scene).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        received = session.record_received(row.request_id, attempt())
        retired = session.retire(row.request_id)
        assert retired.upload_attempt == received.upload_attempt
        assert session.get(row.request_id) == retired
        assert session.get("11111111-1111-4111-8111-111111111111") is None
    with pytest.raises(ValueError, match="session_released"):
        session.get(row.request_id)
    item = outbox(scene).claim_due(NOW)
    assert json.loads(item.receipt_json)["upload_attempt"] == dict(received.upload_attempt)


@pytest.mark.parametrize("saved_status", ["acknowledged", "blocked", "owner_deleted"])
def test_receipt_lease_survives_saved_settlement_deletion_and_missing_files(scene, saved_status):
    record = terminal(scene, success=True)
    store, saved = scene
    if saved_status == "owner_deleted":
        store.accept_owner_delete_task(
            owner_module().OwnerDeleteDeviceTask.from_payload(payload(store, shot="shot-0002")),
            managed_scope=SCOPE,
        )
    else:
        with sqlite3.connect(store.path) as db:
            db.execute(
                "UPDATE field_saved_media_outbox SET status=?,claim_token=NULL,"
                "lease_until_utc=NULL WHERE clip_id=?",
                (saved_status, saved.clip_id),
            )
    saved.local_path.unlink()
    box = outbox(scene)
    item = box.claim_due(NOW)
    assert item.request_id == record.request_id and item.clip_id == saved.clip_id
    assert not hasattr(item, "local_path")
    with pytest.raises(FrozenInstanceError):
        item.claim_token = "changed"
    assert outbox(scene, replace(SCOPE, device_id="foreign")).claim_due(NOW) is None
    assert box.acknowledge(item, ack(item)) is True
    assert rows(store)[0]["status"] == "acknowledged"
    assert box.claim_due(NOW + timedelta(days=1)) is None


def test_independent_expiring_lease_retries_same_bytes_and_stale_ack_cannot_settle(scene):
    terminal(scene)
    box = outbox(scene)
    first = box.claim_due(NOW, lease_seconds=1)
    assert first.attempts == 1
    assert box.claim_due(NOW) is None
    second = box.claim_due(NOW + timedelta(seconds=2))
    assert second.attempts == 2 and second.claim_token != first.claim_token
    assert second.receipt_json == first.receipt_json
    assert box.acknowledge(first, ack(first)) is False
    assert box.retry(first, "receipt_unavailable", "2026-11-01T00:00:03.000000Z") is False
    assert box.block(first, "receipt_conflict") is False
    assert box.retry(second, "receipt_unavailable", "2026-11-01T00:00:03.000000Z") is True
    assert box.claim_due(NOW + timedelta(seconds=2)) is None
    third = box.claim_due(NOW + timedelta(seconds=3))
    assert third.receipt_sha256 == first.receipt_sha256
    assert box.block(third, "receipt_conflict") is True
    assert box.claim_due(NOW + timedelta(days=1)) is None
    assert rows(scene[0])[0]["last_error_code"] == "receipt_conflict"


@pytest.mark.parametrize("change", ["extra", "request", "digest", "erasable", "timestamp"])
def test_ack_shape_identity_digest_and_false_are_strict(scene, change):
    terminal(scene)
    box = outbox(scene)
    item = box.claim_due(NOW)
    value = ack(item)
    if change == "extra":
        value["extra"] = None
    elif change == "request":
        value["upload_request"]["request_id"] = "11111111-1111-4111-8111-111111111111"
    elif change == "digest":
        value["receipt_sha256"] = "0" * 64
    elif change == "erasable":
        value["erasable"] = 0
    else:
        value["received_at_utc"] = "2026-11-01T00:00:00.000001Z"
    before = rows(scene[0])
    with pytest.raises(ValueError):
        box.acknowledge(item, value)
    assert rows(scene[0]) == before
    with pytest.raises(ValueError):
        box.block(item, "raw provider error https://secret")


@pytest.mark.parametrize("corrupt_id", [False, True])
def test_reconcile_recovers_only_valid_original_metadata_and_wraps_bounded_cursor(
    metadata_store,
    corrupt_id,
):
    # Bad early clip must not starve later valid receipts. Use distinct clips so
    # whole-history validation cannot correctly invalidate the later clip as well.
    store = metadata_store
    first_saved = store.claim_due_saved_media(NOW, managed_scope=SCOPE)
    scene = (store, first_saved)
    first = terminal(scene)
    second_saved = store.claim_due_saved_media(NOW, managed_scope=SCOPE)
    assert second_saved.clip_id != first_saved.clip_id
    second = terminal((store, second_saved))
    ordered = sorted([first.request_id, second.request_id])
    bad_id, good_id = ordered
    with sqlite3.connect(store.path) as db:
        db.execute("DELETE FROM field_upload_receipt_outbox")
        db.execute(
            "UPDATE field_saved_upload_requests SET binding_sha256=? WHERE request_id=?",
            ("0" * 64, bad_id),
        )
        if corrupt_id:
            db.execute(
                "UPDATE field_saved_upload_requests SET request_id=? WHERE request_id=?",
                ("000-invalid-request-id", bad_id),
            )
            bad_id = "000-invalid-request-id"
    box = outbox(scene)
    bad = box.reconcile(limit=1)
    assert (bad.scanned, bad.queued, bad.next_request_id) == (1, 0, bad_id)
    assert bad.diagnostics == ("invalid_state",)
    good = box.reconcile(after_request_id=bad.next_request_id, limit=1)
    assert (good.scanned, good.queued, good.next_request_id) == (1, 1, good_id)
    wrapped = box.reconcile(after_request_id=good.next_request_id, limit=1)
    assert wrapped.scanned == 1 and wrapped.next_request_id == bad_id
    assert wrapped.diagnostics == ("invalid_state",)
    assert [r["request_id"] for r in rows(store)] == [good_id]
    with pytest.raises(ValueError):
        box.reconcile(limit=17)


@pytest.mark.parametrize("change", ["epoch", "spec", "save", "terminal", "body", "history"])
def test_claim_revalidates_original_binding_history_and_immutable_body(scene, change):
    terminal(scene, success=True)
    with sqlite3.connect(scene[0].path) as db:
        if change == "epoch":
            db.execute("DELETE FROM field_managed_capture_epochs WHERE source_epoch_id='epoch-2'")
        elif change == "spec":
            db.execute("DELETE FROM field_managed_resource_specs")
        elif change == "save":
            db.execute("UPDATE field_clip_save_requests SET receipt_json=receipt_json || ' '")
        elif change == "terminal":
            db.execute("UPDATE field_saved_upload_requests SET terminal_sha256=?", ("0" * 64,))
        elif change == "body":
            db.execute("UPDATE field_upload_receipt_outbox SET receipt_json='{}'")
        else:
            db.execute("UPDATE field_saved_upload_requests SET binding_sha256=?", ("0" * 64,))
    assert outbox(scene).claim_due(NOW) is None
    assert rows(scene[0])[0]["status"] == "blocked"
    assert rows(scene[0])[0]["last_error_code"] == "receipt_invalid"


@pytest.mark.parametrize("fault", ["index", "commit"])
def test_schema24_ddl_only_migration_preserves_all_original_rows_and_failure_rolls_back(
    scene,
    tmp_path,
    fault,
):
    terminal(scene, success=True)
    store = scene[0]
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=23")
        schema = db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
    before = snapshot(store.path)

    class FaultConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if fault == "index" and "CREATE INDEX due_upload_receipts" in sql:
                raise sqlite3.OperationalError("synthetic schema24 fault")
            return super().execute(sql, parameters)

        def commit(self):
            if fault == "commit":
                raise sqlite3.OperationalError("synthetic schema24 commit fault")
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
        assert db.execute("PRAGMA user_version").fetchone()[0] == 23
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
    assert after == before
    fresh = SQLiteFieldRecordStore(tmp_path / "fresh.db")
    with sqlite3.connect(store.path) as migrated, sqlite3.connect(fresh.path) as new:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == 30
        assert (
            migrated.execute(
                "SELECT sql FROM sqlite_master WHERE name='field_upload_receipt_outbox'"
            ).fetchone()
            == new.execute(
                "SELECT sql FROM sqlite_master WHERE name='field_upload_receipt_outbox'"
            ).fetchone()
        )
    assert outbox(scene).reconcile().queued == 1
    after.pop("field_saved_upload_requests")
    current = snapshot(store.path)
    for name in after:
        assert after[name] == current[name]


def test_projection_preserves_two_ordered_microseconds_within_same_millisecond(scene):
    store = scene[0]
    with ledger(scene).use(scene[1]) as session:
        request = session.load_or_create("cloud-job-1")
        session.record_received(request.request_id, attempt())
        store._now_utc = lambda: NOW + timedelta(microseconds=1)
        session.record_started(request.request_id)
        store._now_utc = lambda: NOW + timedelta(microseconds=2)
        original = session.record_succeeded(request.request_id, 200)
    item = outbox(scene).claim_due(NOW)
    evidence = json.loads(item.receipt_json)["evidence"]
    assert evidence["started_at_utc"] == "2026-11-01T00:00:00.000001Z"
    assert evidence["received_at_utc"] == "2026-11-01T00:00:00.000002Z"
    assert hashlib.sha256(original.terminal_json.encode()).hexdigest() == original.terminal_sha256


def test_recovery_batches_at_most16_and_retained_cap64_history_is_revalidated(scene):
    store = scene[0]
    with ledger(scene).use(scene[1]) as session:
        for _ in range(64):
            row = session.load_or_create("cloud-job-1")
            session.retire(row.request_id)
        with pytest.raises(ValueError, match="request_limit"):
            session.load_or_create("cloud-job-1")
    with sqlite3.connect(store.path) as db:
        db.execute("DELETE FROM field_upload_receipt_outbox")
    original = snapshot(store.path)["field_saved_upload_requests"]
    box = outbox(scene)
    cursor = None
    for _ in range(4):
        batch = box.reconcile(after_request_id=cursor)
        assert batch.scanned == batch.queued == 16
        assert batch.diagnostics == ()
        assert cursor is None or batch.next_request_id > cursor
        cursor = batch.next_request_id
    assert box.reconcile(after_request_id=cursor).scanned == 0
    assert snapshot(store.path)["field_saved_upload_requests"] == original
    assert len(rows(store)) == 64
    # Synthetic corruption exceeding the cap must invalidate sending, even if
    # the individual queued row's original binding still matches.
    with sqlite3.connect(store.path) as db:
        db.execute(
            "INSERT INTO field_saved_upload_requests SELECT ?,clip_id,cloud_job_id,"
            "binding_json,binding_sha256,state,upload_attempt_json,created_at_utc,"
            "started_at_utc,terminal_json,terminal_sha256 "
            "FROM field_saved_upload_requests LIMIT 1",
            (str(uuid.uuid4()),),
        )
    assert box.claim_due(NOW) is None


def test_changed_existing_queue_is_rejected_even_on_repeated_terminal_call(scene):
    record = terminal(scene)
    with sqlite3.connect(scene[0].path) as db:
        db.execute("UPDATE field_upload_receipt_outbox SET receipt_sha256=?", ("0" * 64,))
    before = snapshot(scene[0].path)
    with ledger(scene).use(scene[1]) as session:
        with pytest.raises(ValueError):
            session.retire(record.request_id)
    assert snapshot(scene[0].path) == before


def test_claim_contention_is_retryable_and_never_blocks_valid_evidence(scene, monkeypatch):
    terminal(scene)
    original = scene[0]._connect

    def short_timeout():
        db = original()
        db.execute("PRAGMA busy_timeout=1")
        return db

    monkeypatch.setattr(scene[0], "_connect", short_timeout)
    before = rows(scene[0])
    with sqlite3.connect(scene[0].path) as db:
        db.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError) as failure:
            outbox(scene).claim_due(NOW)
        assert failure.value.code == "database_busy"
        assert failure.value.retryable is True
    assert rows(scene[0]) == before


@pytest.mark.parametrize("operation", ["claim", "reconcile"])
def test_no_database_write_transaction_is_held_while_acquiring_canonical_pin(
    scene,
    monkeypatch,
    operation,
):
    terminal(scene)
    if operation == "reconcile":
        with sqlite3.connect(scene[0].path) as db:
            db.execute("DELETE FROM field_upload_receipt_outbox")
    ledger_module = importlib.import_module(
        "billiards_ai.edge.route_assistance.saved_upload_requests"
    )
    original = ledger_module.session_media_lock

    @contextmanager
    def checked_pin(*args, **kwargs):
        # Another connection can acquire the writer before the canonical pin;
        # holding a writer while waiting here would invert the native lock order.
        with sqlite3.connect(scene[0].path, timeout=0.01) as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        with original(*args, **kwargs):
            yield

    monkeypatch.setattr(ledger_module, "session_media_lock", checked_pin)
    if operation == "claim":
        assert outbox(scene).claim_due(NOW) is not None
    else:
        assert outbox(scene).reconcile().queued == 1
