"""Controlled local task delivery and synthetic media; no cloud delivery proof."""

import json
import sqlite3
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_owner_delete_once
from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance import cloud_runtime
from billiards_ai.lab.field_assistance.managed_resource_recovery import (
    ManagedRuntimeResourceResolver,
)
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import media as media
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    SESSION,
    payload,
)
from tests.unit.lab.field_assistance import test_managed_resource_recovery as resource_fixtures
from tests.unit.lab.field_assistance.test_cloud_runtime import _args
from tests.unit.lab.test_mainline1_shot_jobs import _populated_ring
from tests.unit.lab.test_mainline1_shot_worker import _worker

scene = resource_fixtures.scene
resource_assets_fixture = resource_fixtures.resource_assets_fixture


def owner_worker(media, **kwargs):
    store, db, root, jobs, *_ = media
    return cloud_runtime._BackgroundLocalMediaCleanupWorker(
        store=store, jobs_db=db, artifact_root=root, jobs_root=jobs, **kwargs
    )


def original_resolver(media, diagnostics):
    spec = media[0].get_managed_resource_spec_for_session(SCOPE, SESSION).spec
    Path(spec.paths["ring_root"]).mkdir(exist_ok=True)
    return ManagedRuntimeResourceResolver(media[0], SCOPE, diagnostics)


@pytest.mark.parametrize("entrypoint", ["usb", "blackbox"])
def test_capture_entrypoints_start_shared_owner_cleanup_with_original_roots(
    media, tmp_path, entrypoint
):
    # Catches factory wiring that runs owner work against the current profile.
    if entrypoint == "usb":
        from tests.unit.scripts.test_serve_mainline1_usb_field import load_script
    else:
        from tests.unit.scripts.test_serve_mainline1_blackbox_field import load_script

    store, _, root, _, overlay, raw, accepted, opponent = media
    original_resolver(media, {})
    done = threading.Event()
    workers = []

    def owner_process(*args, **kwargs):
        result = process_local_owner_delete_once(*args, now_utc=NOW, **kwargs)
        if result.outcome == "completed":
            done.set()
        return result

    def cleanup(**kwargs):
        worker = cloud_runtime._BackgroundLocalMediaCleanupWorker(
            **kwargs, owner_process_once=owner_process, poll_seconds=0.001
        )
        workers.append(worker)
        return worker

    def forbidden(**kwargs):
        pytest.fail("offline owner cleanup requested cloud credentials")

    def idle(**kwargs):
        return SimpleNamespace(start=lambda: None, stop=lambda: None)

    factories = replace(
        cloud_runtime.FieldCloudFactories(),
        credentials=forbidden,
        transport=forbidden,
        postgame_worker=idle,
        local_cleanup_worker=cleanup,
    )
    args = _args(
        tmp_path / "current-profile",
        sync_base_url=None,
        device_id=None,
        credential_target=None,
        cloudbase_environment_id=None,
        expected_development_environment_id=None,
    )
    state = SimpleNamespace(
        record_store=store,
        resource_session_ids=(),
        managed_session_scope=SCOPE,
        managed_resource_spec_record=store.get_managed_resource_spec_for_session(SCOPE, SESSION),
        recovery_diagnostics={},
    )
    runtime = load_script().prepare_cloud_runtime(args, factories=factories)
    try:
        runtime.start(
            SimpleNamespace(field_state=state), postgame_command_factory=lambda sid: [sid]
        )
        assert done.wait(3), store.get_owner_delete(accepted.clip_id)
    finally:
        runtime.stop()
    assert not workers[0]._thread.is_alive()
    assert store.get_owner_delete(accepted.clip_id).status == "completed"
    assert not overlay.exists() and not (raw / "segment-1.mjpeg").exists()
    assert opponent[0].read_bytes() == b"synthetic-overlay-2"
    assert not args.artifact_root.exists()
    assert not args.jobs_root.exists()
    assert (root / "postgame" / SESSION).is_dir()


@pytest.mark.parametrize("failure", ["no_resolver", "missing_original_root"])
def test_unavailable_original_context_diagnoses_without_claim_or_current_fallback(media, failure):
    diagnostics = {}
    resolver = original_resolver(media, diagnostics)
    if failure == "missing_original_root":
        spec = media[0].get_managed_resource_spec_for_session(SCOPE, SESSION).spec
        Path(spec.paths["ring_root"]).rmdir()
    worker = owner_worker(
        media,
        managed_scope=SCOPE,
        recovery_diagnostics=diagnostics,
        resource_resolver=None if failure == "no_resolver" else resolver,
    )
    before = media[0].get_owner_delete(media[6].clip_id)
    assert not worker.run_once()
    assert media[0].get_owner_delete(media[6].clip_id) == before
    assert diagnostics == {SESSION: "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE"}
    assert media[4].read_bytes() == b"synthetic-overlay-1"


@pytest.mark.parametrize("scope", [None, ManagedSessionScope("dev", "other-device", "table-1")])
def test_unmanaged_or_foreign_scope_never_resolves_or_claims_owner_work(media, scope):
    class ForbiddenResolver:
        def resolve(self, *args, **kwargs):
            pytest.fail("foreign or unmanaged owner task reached resource resolution")

    before = media[0].get_owner_delete(media[6].clip_id)
    worker = owner_worker(media, managed_scope=scope, resource_resolver=ForbiddenResolver())
    assert not worker.run_once()
    assert media[0].get_owner_delete(media[6].clip_id) == before
    assert media[4].exists()


def test_owner_attempt_stop_cancels_and_retains_thread_ownership_until_join(media):
    entered, release = threading.Event(), threading.Event()
    resolver = original_resolver(media, {})

    def process(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        assert kwargs["cancelled"]()
        return process_local_owner_delete_once(*args, now_utc=NOW, **kwargs)

    worker = owner_worker(
        media,
        managed_scope=SCOPE,
        resource_resolver=resolver,
        owner_process_once=process,
        join_timeout_seconds=0.001,
    )
    worker.start()
    try:
        assert entered.wait(3)
        with pytest.raises(TimeoutError, match="cleanup"):
            worker.stop()
        assert worker._thread.is_alive()
    finally:
        release.set()
        worker._thread.join(3)
        worker.stop()
    assert not worker._thread.is_alive()
    assert media[0].get_owner_delete(media[6].clip_id).status == "pending"
    assert media[4].exists()


def test_owner_dispatch_passes_only_resolved_session_and_cleanup_purpose(media):
    calls = []
    original = original_resolver(media, {})

    class Resolver:
        def resolve(self, sid, *, purpose):
            calls.append((sid, purpose))
            return original.resolve(sid, purpose=purpose)

    def process(store, db, root, jobs, **kwargs):
        assert (db, root, jobs) == media[1:4]
        assert kwargs["session_ids"] == (SESSION,)
        return process_local_owner_delete_once(store, db, root, jobs, now_utc=NOW, **kwargs)

    worker = owner_worker(
        media,
        managed_scope=SCOPE,
        resource_resolver=Resolver(),
        owner_process_once=process,
    )
    assert worker.run_once()
    assert calls == [(SESSION, "cleanup")]
    assert not worker.run_once()
    assert media[0].get_owner_delete(media[6].clip_id).status == "completed"


@pytest.mark.parametrize("scene", ["owner-composition"], indirect=True)
def test_local_owner_erasure_composes_restart_partial_finalization_and_lost_acks(
    scene, tmp_path, monkeypatch
):
    # Removing lifecycle dispatch or substituting current roots must leave red
    # bytes behind; generation/finalization and upload fences stay actual.
    store, sid = scene.store, scene.sid
    manifest = store.get_postgame_manifest(sid)
    red_receipt = store.save_postgame_clip(
        request_id="save-red",
        session_id=sid,
        shot_id="shot-1",
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    blue_receipt = store.save_postgame_clip(
        request_id="save-blue",
        session_id=sid,
        shot_id="shot-2",
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    with store._connect() as db:
        original_saves = db.execute(
            "SELECT request_id,receipt_json FROM field_clip_save_requests ORDER BY request_id"
        ).fetchall()
    worker = _worker(store, scene.jobs, scene.args.artifact_root / "postgame" / sid)
    worker._deps.vision_backend.package_identity = (
        worker._deps.vision_backend.package_identity.model_copy(
            update={"package_id": scene.spec.model_package_id}
        )
    )
    worker._deps = replace(
        worker._deps,
        capture_setup_id=scene.spec.capture_setup_id,
        calibration_artifact_id=scene.spec.calibration_artifact_id,
        config_sha256=scene.spec.config_sha256,
        model_package_id=scene.spec.model_package_id,
        identity_training_package_id=scene.spec.identity_training_package_id,
        identity_training_package_sha256=scene.spec.identity_training_package_sha256,
    )
    first = worker.process_one("red-worker", sid, NOW)
    assert first is not None and first.job_id == "job-1"
    assert not store.is_postgame_finalized(sid)
    publish_calls = []

    class LostPublishAck:
        def run_saved_media_once(self, item, **kwargs):
            publish_calls.append((item.shot_id, item.owner_user_id))
            # A controlled local port records publication but loses its reply.
            raise TimeoutError("synthetic publish ACK lost")

    sync = cloud_runtime._BackgroundSavedMediaSyncWorker(
        store=store,
        client=LostPublishAck(),
        trusted_postgame_root=tmp_path / "current-postgame",
        wake_event=threading.Event(),
        managed_scope=SCOPE,
        now_utc=lambda: NOW,
    )
    assert sync.run_once()
    red_outbox = store.list_saved_media_outbox()[0]
    assert publish_calls == [("shot-1", "user-a")]
    assert red_outbox.status == "pending"
    assert red_outbox.last_error_code == "SAVED_MEDIA_WORKER_ERROR"
    value = OwnerDeleteDeviceTask.from_payload(payload(store, "shot-1"))
    accepted = store.accept_owner_delete_task(value, managed_scope=SCOPE)
    facts = [store.get_live_shot_fact(sid, f"shot-{n}") for n in (1, 2, 3)]
    assert [fact.actor_user_id for fact in facts] == ["user-a", "user-b", "user-b"]
    owners = [store.get_shot_ownership(sid, f"shot-{n}") for n in (1, 2, 3)]
    blue_raw = scene.jobs.snapshot("job-2").job.media_root
    blue_bytes = {p: p.read_bytes() for p in blue_raw.iterdir()}
    diagnostic = {}
    resolver = ManagedRuntimeResourceResolver(store, SCOPE, diagnostic)

    class PowerLoss(BaseException):
        pass

    def fault(point, obj):
        if point == "after_detach":
            raise PowerLoss()

    cleanup = cloud_runtime._BackgroundLocalMediaCleanupWorker(
        store=store,
        jobs_db=tmp_path / "current-jobs" / "wrong.db",
        artifact_root=tmp_path / "current-artifacts",
        jobs_root=tmp_path / "current-jobs",
        managed_scope=SCOPE,
        resource_resolver=resolver,
        owner_process_once=lambda *a, **kw: process_local_owner_delete_once(
            *a, now_utc=NOW, lease_seconds=1, fault_hook=fault, **kw
        ),
    )
    with pytest.raises(PowerLoss):
        cleanup.run_once()
    cleanup.stop()
    interrupted = store.get_owner_delete(accepted.clip_id)
    assert interrupted.status == "leased" and interrupted.receipt_json is None
    assert list(scene.args.artifact_root.rglob("*.detached"))
    restored = SQLiteFieldRecordStore(store.path, now_utc=lambda: NOW + timedelta(seconds=2))
    # A changed current calibration/profile and missing model cannot redirect or
    # prevent cleanup of already validated original copies.
    scene.args.calibration("b")
    model = scene.args.detector_training_run / "checkpoint_best.pth"
    model_bytes = model.read_bytes()
    model.unlink()
    resumed = cloud_runtime._BackgroundLocalMediaCleanupWorker(
        store=restored,
        jobs_db=tmp_path / "current-jobs" / "wrong.db",
        artifact_root=tmp_path / "current-artifacts",
        jobs_root=tmp_path / "current-jobs",
        managed_scope=SCOPE,
        resource_resolver=ManagedRuntimeResourceResolver(restored, SCOPE, diagnostic),
        owner_process_once=lambda *a, **kw: process_local_owner_delete_once(
            *a, now_utc=NOW + timedelta(seconds=2), **kw
        ),
    )
    assert resumed.run_once()
    completed = restored.get_owner_delete(accepted.clip_id)
    assert completed.status == "completed" and completed.receipt_json
    assert completed.claim_token != interrupted.claim_token
    assert not (first.artifact_path / "overlay.mp4").exists()
    red_raw = scene.jobs.snapshot("job-1").job.media_root
    assert not list(red_raw.glob("*.mjpgseg")) and not list(red_raw.glob("*.mjpeg"))
    assert (red_raw / "interval.json").is_file()
    # Simulated lost completion ACK: exact trusted redelivery returns the same
    # immutable receipt and cannot reclaim a fresh attempt.
    assert restored.accept_owner_delete_task(value, managed_scope=SCOPE) == completed
    assert restored.get_owner_delete(accepted.clip_id).receipt_json == completed.receipt_json
    assert not resumed.run_once()
    assert {p: p.read_bytes() for p in blue_raw.iterdir()} == blue_bytes
    assert [restored.get_live_shot_fact(sid, f"shot-{n}") for n in (1, 2, 3)] == facts
    assert [restored.get_shot_ownership(sid, f"shot-{n}") for n in (1, 2, 3)] == owners
    # Keep reviewing to exercise late analysis after red's physical erasure.
    # S1 permits completion independently of this still-pending analysis.
    assert restored.active_postgame_review(scene.spec.artifact_id) == manifest
    model.write_bytes(model_bytes)
    second = worker.process_one("blue-worker", sid, NOW + timedelta(seconds=3))
    assert second is not None and second.job_id == "job-2"
    blue_overlay = (second.artifact_path / "overlay.mp4").read_bytes()
    assert worker.process_one("unsaved-worker", sid, NOW + timedelta(seconds=3)) is not None
    assert restored.is_postgame_finalized(sid)
    restored.complete_postgame_review(scene.spec.artifact_id, sid, manifest.manifest_sha256)
    assert not restored.is_postgame_discarded(sid, "shot-1")
    assert not restored.is_postgame_discarded(sid, "shot-2")
    assert restored.is_postgame_discarded(sid, "shot-3")
    assert red_receipt.status.value == "SAVE_PENDING"
    with pytest.raises(ValueError, match="OWNER_CLIP_DELETE_REQUESTED"):
        restored.save_postgame_clip(
            request_id="save-red",
            session_id=sid,
            shot_id="shot-1",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
    with restored._connect() as db:
        assert (
            db.execute(
                "SELECT request_id,receipt_json FROM field_clip_save_requests ORDER BY request_id"
            ).fetchall()
            == original_saves
        )
    assert (
        restored.save_postgame_clip(
            request_id="save-blue",
            session_id=sid,
            shot_id="shot-2",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == blue_receipt
    )

    # Blue can publish normally after red acceptance. Red cannot be reuploaded.
    class BlueClient:
        def run_saved_media_once(self, item, **kwargs):
            publish_calls.append((item.shot_id, item.owner_user_id))
            assert item.shot_id == "shot-2"
            return SimpleNamespace(outcome="complete")

    sync._store, sync._client = restored, BlueClient()
    assert sync.run_once()
    assert not sync.run_once()
    assert publish_calls == [("shot-1", "user-a"), ("shot-2", "user-b")]
    with sqlite3.connect(scene.args.jobs_root / "analysis-jobs.sqlite3") as db:
        db.execute("UPDATE shot_analysis_jobs SET status='pending' WHERE job_id='job-1'")
    import billiards_ai.lab.mainline1_shot_worker as shot_module

    monkeypatch.setattr(
        shot_module,
        "load_materialized_interval",
        lambda job: pytest.fail("owner-deleted source was reopened for generation"),
    )
    assert worker.process_one("never-regenerate", sid, NOW + timedelta(seconds=4)) is None
    # A later capture root is independent of old cleanup and old session pins.
    next_root = scene.args.ring_root / "sessions" / "next-game" / "epoch-1"
    next_ring = _populated_ring(next_root)
    try:
        next_shot = next_ring.freeze_shot()
        captured = {p: p.read_bytes() for p in next_root.iterdir()}
        with session_media_lock(scene.args.artifact_root, "next-game", exclusive=True):
            assert resumed.run_once()  # unsaved shot 3, using retained roots
            assert not resumed.run_once()
        assert {p: p.read_bytes() for p in next_root.iterdir()} == captured
        assert len(list(next_shot.frames())) == 8
        assert next_ring.health.state.value == "running"
    finally:
        next_ring.close()
    assert (second.artifact_path / "overlay.mp4").read_bytes() == blue_overlay
    assert {p: p.read_bytes() for p in blue_raw.iterdir()} == blue_bytes
    assert [restored.get_live_shot_fact(sid, f"shot-{n}") for n in (1, 2, 3)] == facts
    with restored._connect() as db:
        finalized = json.loads(
            db.execute(
                "SELECT manifest_json FROM field_postgame_finalizations WHERE session_id=?", (sid,)
            ).fetchone()[0]
        )
    assert [row["shot_id"] for row in finalized["clips"]] == ["shot-1", "shot-2", "shot-3"]
    assert not (tmp_path / "current-artifacts").exists()
    assert not (tmp_path / "current-jobs").exists()
