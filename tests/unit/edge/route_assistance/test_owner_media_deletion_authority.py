"""Local trusted owner authority: synthetic SQLite, no transport or file erasure."""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import MAX_SAVED_CLIP_BYTES
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotFact,
    LiveShotMediaReference,
    ShotMediaRange,
)
from tests.unit.edge.route_assistance.test_managed_session_persistence import snapshot

NOW = datetime(2026, 11, 1, tzinfo=UTC)
SCOPE = ManagedSessionScope("dev", "device-1", "table-1")
BIRTH = "22222222-2222-4222-8222-222222222222"
SESSION = f"managed-{BIRTH}"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def owner_module():
    return importlib.import_module("billiards_ai.edge.route_assistance.owner_media_deletion")


@pytest.fixture
def scene(tmp_path):
    store = SQLiteFieldRecordStore(tmp_path / "field.db", now_utc=lambda: NOW)
    request = store.reserve_managed_open(SCOPE, artifact_id="artifact-1", rule_profile_id=None)
    binding = ManagedSessionBinding(
        "1.0", SESSION, BIRTH, "managed_session_birth_v1", "table-1", "device-1", "open"
    )
    store.record_managed_binding(SCOPE, request.open_request_id, binding)
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
        shot = FieldShotRecord(
            SESSION,
            f"shot-{sequence:04d}",
            sequence,
            role,
            1000 * sequence,
            None,
            None,
            None,
            None,
            ShotMediaRange(
                "artifact-1",
                f"usb:epoch-{sequence}",
                "usb_live",
                1000 * sequence,
                1000 * sequence + 1,
                f"epoch-{sequence}",
            ),
        )
        store.record_shot_start(shot, source_start_event_id=f"start-{sequence}")
        store.record_shot_settle(
            replace(
                shot,
                settled_ms=1000 * sequence + 500,
                media=replace(shot.media, end_ms=1000 * sequence + 500),
                live_job_id=f"job-{sequence}",
                live_evidence_digest="d" * 64,
            ),
            source_epoch_id=f"epoch-{sequence}",
            live_fact=LiveShotFact.create(
                session_id=SESSION,
                shot_id=shot.shot_id,
                sequence=sequence,
                actor_role=role,
                actor_user_id="owner-red" if sequence == 1 else "owner-blue",
                start_frame_id=100 * sequence,
                settle_frame_id=100 * sequence + 10,
                mode="observation",
                observed_outcome="unknown",
                rule_revision=sequence,
                recording_reference=f"session-recording:epoch-{sequence}:100-110",
            ),
        )
        store.attach_live_shot_media(
            LiveShotMediaReference(
                SESSION,
                shot.shot_id,
                f"epoch-{sequence}",
                sequence,
                f"job-{sequence}",
                str(tmp_path / f"raw-{sequence}"),
                "d" * 64,
            )
        )
    store.end_session(SESSION)
    manifest = store.get_postgame_manifest(SESSION)
    for sequence in (1, 2):
        # First selection intentionally predates availability (lost publish ACK case).
        if sequence == 1:
            store.save_postgame_clip(
                request_id="save-1",
                session_id=SESSION,
                shot_id="shot-0001",
                postgame_manifest_sha256=manifest.manifest_sha256,
            )
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
        if sequence == 2:
            store.save_postgame_clip(
                request_id="save-2",
                session_id=SESSION,
                shot_id="shot-0002",
                postgame_manifest_sha256=manifest.manifest_sha256,
            )
    return store


def payload(store, shot="shot-0001"):
    item = next(item for item in store.list_saved_media_outbox() if item.shot_id == shot)
    return dict(
        contract="owner_delete_device_v1",
        deletion_job_id="owner-delete-" + hashlib.sha256(item.clip_id.encode()).hexdigest()[:32],
        environment_id=SCOPE.environment_id,
        device_id=SCOPE.device_id,
        table_id=SCOPE.table_id,
        session_id=item.session_id,
        shot_id=item.shot_id,
        source_job_id=item.job_id,
        artifact_id=item.artifact_id,
        clip_id=item.clip_id,
        owner_user_id=item.owner_user_id,
        actor_role=item.actor_role.value,
        content_sha256=item.content_sha256,
        byte_size=item.byte_size,
        ownership_receipt_sha256=item.ownership_receipt_sha256,
        save_request_id=item.save_request_id,
        save_receipt_sha256=item.save_receipt_sha256,
        postgame_manifest_sha256=item.postgame_manifest_sha256,
        cloud_erase_receipt_sha256="e" * 64,
    )


def task(store, **changes):
    return owner_module().OwnerDeleteDeviceTask.from_payload(payload(store) | changes)


def accept(store):
    return store.accept_owner_delete_task(task(store), managed_scope=SCOPE)


def test_strict_task_canonical_identity(scene):
    value = task(scene)
    assert value.actor_role is WristbandRole.RED
    assert value.canonical_json() == canonical(payload(scene))
    assert value.task_sha256 == hashlib.sha256(canonical(payload(scene)).encode()).hexdigest()


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": True},
        {"byte_size": True},
        {"byte_size": "24"},
        {"byte_size": 0},
        {"byte_size": MAX_SAVED_CLIP_BYTES + 1},
        {"actor_role": "RED"},
        {"owner_user_id": "x\x7fy"},
        {"owner_user_id": "界" * 86},
        {"cloud_erase_receipt_sha256": "A" * 64},
        {"deletion_job_id": "owner-delete-bad"},
        {"session_id": "managed-invalid"},
        {"device_id": "../device"},
        {"shot_id": 4},
        {"save_request_id": "bad/request"},
        {"clip_id": "other"},
    ],
)
def test_task_rejects_malformed_and_coerced_fields(scene, changes):
    with pytest.raises(ValueError):
        task(scene, **changes)


@pytest.mark.parametrize("outbox_status", ["pending", "leased", "acknowledged"])
def test_accept_is_permanent_metadata_only_and_replay_preserves_original_bytes(
    scene, outbox_status
):
    with sqlite3.connect(scene.path) as db:
        db.execute(
            "UPDATE field_saved_media_outbox SET "
            "status=?,claim_token='old-token',lease_until_utc='2099-01-01T00:00:00Z' WHERE "
            "save_request_id='save-1'",
            (outbox_status,),
        )
    before = snapshot(scene.path)
    accepted = accept(scene)
    assert accepted.status == "pending" and accepted.attempts == 0
    assert accepted.local_path.is_absolute() and not accepted.local_path.exists()
    assert accepted.task_sha256 == accepted.task.task_sha256
    assert accepted.job_id == "job-1" and accepted.content_type == "video/mp4"
    assert scene.owner_delete_for_shot(SESSION, "shot-0001") == accepted
    assert scene.owner_delete_for_shot(SESSION, "shot-0002") is None
    assert scene.get_owner_delete("unknown") is None
    after = snapshot(scene.path)
    assert after["field_local_media_discards"] == []
    for name in before.keys() - {"field_owner_media_deletions", "field_saved_media_outbox"}:
        assert before[name] == after[name]
    outbox = next(item for item in scene.list_saved_media_outbox() if item.shot_id == "shot-0001")
    assert outbox.status == "blocked" and outbox.last_error_code == "OWNER_CLIP_DELETE_REQUESTED"
    assert outbox.claim_token is None and outbox.lease_until_utc is None
    restarted = SQLiteFieldRecordStore(scene.path, now_utc=lambda: NOW)
    assert restarted.accept_owner_delete_task(accepted.task, managed_scope=SCOPE) == accepted
    assert snapshot(scene.path) == after
    assert len(scene.list_saved_media_outbox()) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_user_id", "owner-blue"),
        ("actor_role", "blue"),
        ("content_sha256", "f" * 64),
        ("byte_size", 25),
        ("ownership_receipt_sha256", "f" * 64),
        ("save_request_id", "different"),
        ("save_receipt_sha256", "f" * 64),
        ("postgame_manifest_sha256", "f" * 64),
        ("source_job_id", "other"),
        ("environment_id", "other"),
        ("device_id", "other"),
        ("table_id", "other"),
    ],
)
def test_admission_compares_every_authority_identity_without_writes(scene, field, value):
    before = snapshot(scene.path)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(task(scene, **{field: value}), managed_scope=SCOPE)
    assert snapshot(scene.path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_managed_open_requests",
        "DELETE FROM field_clip_save_requests WHERE request_id='save-1'",
        "DELETE FROM field_saved_media_outbox WHERE save_request_id='save-1'",
        "UPDATE field_shot_ownership SET payload_json='{}' WHERE shot_id='shot-0001'",
        "UPDATE field_live_shot_media SET payload_json='{}' WHERE shot_id='shot-0001'",
        "UPDATE field_clip_save_requests SET receipt_json='{}' WHERE request_id='save-1'",
        "UPDATE field_postgame_clip_availability SET byte_size=25 WHERE shot_id='shot-0001'",
        "UPDATE field_sessions SET status='open'",
    ],
)
def test_corrupt_or_unmanaged_authority_rejected_atomically(scene, damage):
    original = task(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(damage)
    before = snapshot(scene.path)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(original, managed_scope=SCOPE)
    assert snapshot(scene.path) == before


def test_replay_rejects_changed_cloud_receipt_and_resolved_selection(scene):
    original = accept(scene)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(
            replace(original.task, cloud_erase_receipt_sha256="f" * 64), managed_scope=SCOPE
        )
    with sqlite3.connect(scene.path) as db:
        db.execute(
            "UPDATE field_postgame_clip_availability SET local_path='/different.mp4' WHERE "
            "shot_id='shot-0001'"
        )
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(original.task, managed_scope=SCOPE)


def test_fences_refuse_old_save_availability_and_stale_upload_mutations(scene):
    upload = scene.claim_due_saved_media(NOW)
    # Clip IDs sort independently of shot; fence the claimed selection.
    original = owner_module().OwnerDeleteDeviceTask.from_payload(payload(scene, upload.shot_id))
    scene.accept_owner_delete_task(original, managed_scope=SCOPE)
    assert not scene.is_saved_media_claim_current(upload.clip_id, upload.claim_token)
    operations = [
        lambda: scene.save_postgame_clip(
            request_id=upload.save_request_id,
            session_id=SESSION,
            shot_id=upload.shot_id,
            postgame_manifest_sha256=upload.postgame_manifest_sha256,
        ),
        lambda: scene.save_postgame_clip(
            request_id="new-save",
            session_id=SESSION,
            shot_id=upload.shot_id,
            postgame_manifest_sha256=upload.postgame_manifest_sha256,
        ),
        lambda: scene.record_postgame_clip_available(
            session_id=SESSION,
            shot_id=upload.shot_id,
            job_id=upload.job_id,
            artifact_id=upload.artifact_id,
            local_path=upload.local_path,
            content_sha256=upload.content_sha256,
            byte_size=upload.byte_size,
            postgame_manifest_sha256=upload.postgame_manifest_sha256,
        ),
        lambda: scene.acknowledge_saved_media(upload.clip_id, upload.claim_token),
        lambda: scene.retry_saved_media(upload.clip_id, upload.claim_token, error_code="RETRY"),
        lambda: scene.block_saved_media(upload.clip_id, upload.claim_token, error_code="OTHER"),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    other = scene.claim_due_saved_media(NOW)
    assert other.shot_id != upload.shot_id
    assert scene.is_saved_media_claim_current(other.clip_id, other.claim_token)
    assert not scene.is_saved_media_claim_current(other.clip_id, "wrong")
    assert scene.claim_due_saved_media(NOW) is None


def journal(clip_id, *, state="planned", count=1):
    return canonical(
        dict(
            version=1,
            clip_id=clip_id,
            objects=[
                dict(
                    path=f"/synthetic/{index}",
                    root="artifacts",
                    staged_path=f"/synthetic/staged/{index}",
                    sha256="c" * 64,
                    size=24,
                    identity=[1, index + 1, 24, 1],
                    state=state,
                    reclaimed_bytes=24 if state == "deleted" else 0,
                )
                for index in range(count)
            ],
        )
    )


def test_claim_rotation_due_filters_checkpoint_digest_and_terminal_receipt(scene):
    accepted = accept(scene)
    assert scene.list_pending_owner_delete_session_ids(managed_scope=SCOPE) == (SESSION,)
    assert (
        scene.list_pending_owner_delete_session_ids(
            managed_scope=replace(SCOPE, table_id="foreign")
        )
        == ()
    )
    assert scene.claim_due_owner_delete("worker", NOW, session_ids=()) is None
    assert scene.claim_due_owner_delete("worker", NOW, session_ids=("managed-other",)) is None
    first = scene.claim_due_owner_delete("worker", NOW, session_ids=(SESSION,))
    assert first.status == "leased" and first.attempts == 1 and first.claim_token
    assert scene.claim_due_owner_delete("worker", NOW) is None
    progress = journal(first.clip_id)
    assert not scene.update_owner_delete_progress(replace(first, task_sha256="f" * 64), progress)
    assert scene.update_owner_delete_progress(first, progress)
    second = scene.claim_due_owner_delete("worker2", NOW + timedelta(seconds=31))
    assert second.claim_token != first.claim_token and second.attempts == 2
    assert not scene.update_owner_delete_progress(first, progress)
    assert not scene.complete_owner_delete(
        first, progress_json=journal(first.clip_id, state="deleted")
    )
    assert scene.retry_owner_delete(second, "BUSY", NOW + timedelta(minutes=2))
    assert scene.get_owner_delete(first.clip_id).status == "retry"
    assert scene.claim_due_owner_delete("worker3", NOW + timedelta(minutes=1)) is None
    third = scene.claim_due_owner_delete("worker3", NOW + timedelta(minutes=2))
    terminal = journal(third.clip_id, state="deleted")
    assert scene.complete_owner_delete(third, progress_json=terminal)
    completed = scene.get_owner_delete(third.clip_id)
    expected = dict(
        contract="local_owner_erase_receipt_v1",
        deletion_job_id=accepted.task.deletion_job_id,
        clip_id=third.clip_id,
        task_sha256=third.task_sha256,
        cloud_erase_receipt_sha256="e" * 64,
        journal_sha256=hashlib.sha256(terminal.encode()).hexdigest(),
        reclaimed_bytes=24,
    )
    assert completed.status == "completed" and completed.receipt_json == canonical(expected)
    assert completed.receipt_sha256 == hashlib.sha256(canonical(expected).encode()).hexdigest()
    assert not scene.complete_owner_delete(third, progress_json=terminal)
    assert scene.claim_due_owner_delete("worker", NOW + timedelta(days=1)) is None
    assert scene.list_pending_owner_delete_session_ids(managed_scope=SCOPE) == ()
    restarted = SQLiteFieldRecordStore(scene.path)
    assert restarted.accept_owner_delete_task(accepted.task, managed_scope=SCOPE) == completed


@pytest.mark.parametrize("lease", [True, False, 0, -1, 1.5, float("nan"), "30"])
def test_owner_lease_requires_nonboolean_positive_integer(scene, lease):
    accept(scene)
    with pytest.raises(ValueError):
        scene.claim_due_owner_delete("worker", NOW, lease_seconds=lease)


@pytest.mark.parametrize(
    "invalid",
    ["[]", "{}", "null", "{bad", "wrong-clip", "too-many", "too-large", "nonterminal", "bool-size"],
)
def test_owner_journal_bounds_and_terminal_completion(scene, invalid):
    accept(scene)
    claim = scene.claim_due_owner_delete("worker", NOW)
    progress = journal(claim.clip_id, state="deleted")
    if invalid in ("[]", "{}", "null", "{bad"):
        progress = invalid
    elif invalid == "wrong-clip":
        progress = journal("foreign", state="deleted")
    elif invalid == "too-many":
        progress = journal(claim.clip_id, state="deleted", count=65)
    elif invalid == "too-large":
        progress = canonical(json.loads(progress) | {"padding": "x" * 65536})
    elif invalid == "nonterminal":
        progress = journal(claim.clip_id)
    elif invalid == "bool-size":
        decoded = json.loads(progress)
        decoded["objects"][0]["size"] = True
        progress = canonical(decoded)
    with pytest.raises(ValueError):
        scene.complete_owner_delete(claim, progress_json=progress)
    assert scene.get_owner_delete(claim.clip_id).status == "leased"


def test_blocked_owner_is_never_automatically_retried(scene):
    accept(scene)
    claim = scene.claim_due_owner_delete("worker", NOW)
    assert scene.block_owner_delete(claim, "UNSAFE", progress_json=journal(claim.clip_id))
    assert scene.get_owner_delete(claim.clip_id).status == "blocked"
    assert scene.claim_due_owner_delete("worker", NOW + timedelta(days=1)) is None
    assert not scene.retry_owner_delete(claim, "STALE", NOW)


def test_schema21_metadata_upgrade_preserves_all_existing_rows(scene):
    with sqlite3.connect(scene.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE IF EXISTS field_saved_upload_requests")
        db.execute("DROP TABLE IF EXISTS field_owner_media_deletions")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=21")
    before = snapshot(scene.path)
    SQLiteFieldRecordStore(scene.path)
    after = snapshot(scene.path)
    assert {name: after[name] for name in before} == before
    assert after["field_owner_media_deletions"] == []
    with sqlite3.connect(scene.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30


@pytest.mark.parametrize(
    "field",
    [
        "contract",
        "deletion_job_id",
        "environment_id",
        "device_id",
        "table_id",
        "session_id",
        "shot_id",
        "source_job_id",
        "artifact_id",
        "clip_id",
        "owner_user_id",
        "actor_role",
        "content_sha256",
        "byte_size",
        "ownership_receipt_sha256",
        "save_request_id",
        "save_receipt_sha256",
        "postgame_manifest_sha256",
        "cloud_erase_receipt_sha256",
    ],
)
@pytest.mark.parametrize("kind", ["missing", "null", "boolean"])
def test_every_task_field_is_required_without_coercion(scene, field, kind):
    value = payload(scene)
    if kind == "missing":
        value.pop(field)
    else:
        value[field] = None if kind == "null" else True
    with pytest.raises(ValueError):
        owner_module().OwnerDeleteDeviceTask.from_payload(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "managed-33333333-3333-4333-8333-333333333333"),
        ("shot_id", "shot-9999"),
        ("artifact_id", "f" * 64),
    ],
)
def test_valid_but_changed_clip_lineage_is_rejected(scene, field, value):
    from billiards_ai.edge.route_assistance.postgame import session_scoped_clip_id

    changed = payload(scene) | {field: value}
    changed["clip_id"] = session_scoped_clip_id(
        changed["session_id"], changed["shot_id"], changed["source_job_id"], changed["artifact_id"]
    )
    changed["deletion_job_id"] = (
        "owner-delete-" + hashlib.sha256(changed["clip_id"].encode()).hexdigest()[:32]
    )
    before = snapshot(scene.path)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(
            owner_module().OwnerDeleteDeviceTask.from_payload(changed), managed_scope=SCOPE
        )
    assert snapshot(scene.path) == before


def test_admission_does_no_filesystem_work_and_refuses_foreign_configured_scope(scene, monkeypatch):
    from pathlib import Path

    original = task(scene)
    before = snapshot(scene.path)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(
            original, managed_scope=replace(SCOPE, environment_id="foreign")
        )
    assert snapshot(scene.path) == before

    def forbidden(*args, **kwargs):
        pytest.fail("metadata admission touched filesystem media")

    for name in ("stat", "open", "unlink", "rename", "read_bytes"):
        monkeypatch.setattr(Path, name, forbidden)
    assert scene.accept_owner_delete_task(original, managed_scope=SCOPE).status == "pending"


@pytest.mark.parametrize(
    "damage",
    [
        "UPDATE field_clip_save_requests SET "
        "receipt_json=json_remove(receipt_json,'$.owner_user_id','$.actor_role',"
        "'$.ownership_receipt_sha256') WHERE request_id='save-1'",
        "UPDATE field_saved_media_outbox SET status='blocked',last_error_code='UNSAFE' "
        "WHERE save_request_id='save-1'",
        "UPDATE field_saved_media_outbox SET save_request_id='save-2' WHERE "
        "save_request_id='save-1'",
        "UPDATE field_postgame_clip_availability SET content_type='text/plain' WHERE "
        "shot_id='shot-0001'",
        "UPDATE field_postgame_clip_availability SET clip_id='different' WHERE shot_id='shot-0001'",
        "UPDATE field_shots SET "
        "payload_json=json_set(payload_json,'$.live_job_id','different') WHERE shot_id='shot-0001'",
        "UPDATE field_managed_open_requests SET phase='bound'",
    ],
)
def test_additional_corrupted_selection_or_binding_refuses(scene, damage):
    original = task(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(damage)
    before = snapshot(scene.path)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(original, managed_scope=SCOPE)
    assert snapshot(scene.path) == before


def test_new_claims_cannot_revive_a_tombstoned_outbox_and_deadline_is_checked(scene):
    accepted = accept(scene)
    with sqlite3.connect(scene.path) as db:
        db.execute(
            "UPDATE field_saved_media_outbox SET status='pending' WHERE clip_id=?",
            (accepted.clip_id,),
        )
    other = scene.claim_due_saved_media(NOW)
    assert other.clip_id != accepted.clip_id
    assert scene.claim_due_saved_media(NOW) is None
    expired = SQLiteFieldRecordStore(scene.path, now_utc=lambda: NOW + timedelta(minutes=2))
    assert not expired.is_saved_media_claim_current(other.clip_id, other.claim_token)
    for operation in (
        lambda: scene.acknowledge_saved_media(accepted.clip_id, "token"),
        lambda: scene.retry_saved_media(accepted.clip_id, "token", error_code="BAD"),
        lambda: scene.block_saved_media(accepted.clip_id, "token", error_code="BAD"),
    ):
        with pytest.raises(ValueError, match="OWNER_CLIP_DELETE_REQUESTED"):
            operation()


@pytest.mark.parametrize(
    "mutation",
    ["nan", "duplicate", "object", "bad-identity", "bool-reclaimed", "oversized", "empty"],
)
def test_checkpoint_refuses_malformed_bounded_journals(scene, mutation):
    accept(scene)
    claim = scene.claim_due_owner_delete("worker", NOW)
    value = json.loads(journal(claim.clip_id))
    if mutation == "nan":
        value["untrusted"] = float("nan")
    elif mutation == "duplicate":
        value["objects"] *= 2
    elif mutation == "object":
        value["objects"] = [None]
    elif mutation == "bad-identity":
        value["objects"][0]["identity"] = [1, 0, 24, 1]
    elif mutation == "bool-reclaimed":
        value["objects"][0]["reclaimed_bytes"] = False
    elif mutation == "oversized":
        value["objects"][0]["size"] = 8 * 1024**3 + 1
    else:
        value["objects"] = []
    before = scene.get_owner_delete(claim.clip_id)
    with pytest.raises(ValueError):
        scene.update_owner_delete_progress(claim, canonical(value))
    assert scene.get_owner_delete(claim.clip_id) == before


def test_absent_journal_is_terminal_without_reclaimed_bytes(scene):
    accept(scene)
    claim = scene.claim_due_owner_delete("worker", NOW)
    assert scene.complete_owner_delete(claim, progress_json=journal(claim.clip_id, state="absent"))
    assert json.loads(scene.get_owner_delete(claim.clip_id).receipt_json)["reclaimed_bytes"] == 0


def test_owner_tombstone_leaves_facts_only_delivery_authority_intact(scene):
    accept(scene)
    rows = [scene.claim_due_live_shot_fact(NOW), scene.claim_due_live_shot_fact(NOW)]
    assert {row.fact.actor_user_id for row in rows} == {"owner-red", "owner-blue"}


def test_unicode_task_uses_ascii_escape_canonical_bytes(scene):
    value = task(scene, owner_user_id="用户")
    assert '"owner_user_id":"\\u7528\\u6237"' in value.canonical_json()


def test_concurrent_identical_deliveries_create_one_permanent_authority(scene):
    from concurrent.futures import ThreadPoolExecutor

    original = task(scene)
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(
            pool.map(
                lambda _: scene.accept_owner_delete_task(original, managed_scope=SCOPE), range(8)
            )
        )
    assert len(set(states)) == 1
    with sqlite3.connect(scene.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_owner_media_deletions").fetchone()[0] == 1


def test_schema21_migration_failure_rolls_back_metadata_and_version(scene):
    with sqlite3.connect(scene.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE field_owner_media_deletions")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=21")
    before = snapshot(scene.path)

    class FaultConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE INDEX due_owner_media_deletions" in sql:
                raise sqlite3.OperationalError("synthetic index failure")
            return super().execute(sql, parameters)

    class FaultStore(SQLiteFieldRecordStore):
        def _connect(self):
            db = sqlite3.connect(self.path, factory=FaultConnection)
            db.row_factory = sqlite3.Row
            return db

    with pytest.raises(ValueError):
        FaultStore(scene.path)
    assert snapshot(scene.path) == before
    with sqlite3.connect(scene.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 21


@pytest.mark.parametrize(
    "column,value", [("media_path", "/changed/raw"), ("media_sha256", "f" * 64)]
)
def test_exact_replay_revalidates_original_live_media_bytes(scene, column, value):
    original = accept(scene)
    with sqlite3.connect(scene.path) as db:
        row = json.loads(
            db.execute(
                "SELECT payload_json FROM field_live_shot_media WHERE shot_id='shot-0001'"
            ).fetchone()[0]
        )
        row[column] = value
        db.execute(
            "UPDATE field_live_shot_media SET payload_json=? WHERE shot_id='shot-0001'",
            (canonical(row),),
        )
    before = snapshot(scene.path)
    with pytest.raises(ValueError):
        scene.accept_owner_delete_task(original.task, managed_scope=SCOPE)
    assert snapshot(scene.path) == before
