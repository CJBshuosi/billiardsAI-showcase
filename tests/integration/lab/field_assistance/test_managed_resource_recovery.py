"""Real original-context child/job/artifact/collector, synthetic model execution."""

import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.vision.contracts import VisionPackageIdentity
from tests.unit.lab.field_assistance import test_managed_resource_recovery as fixtures
from tests.unit.lab.field_assistance.test_managed_startup import SCOPE
from tests.unit.lab.test_mainline1_shot_worker import _FakeClassifier, _FakeDetector
from tests.unit.scripts.test_run_mainline1_shot_worker import _module

recovery = fixtures.recovery
scene = fixtures.scene
resource_assets_fixture = fixtures.resource_assets_fixture


def child_argv(ctx):
    argv = [
        "--state-db",
        str(ctx.store.path),
        "--jobs-db",
        str(ctx.args.jobs_root / "analysis-jobs.sqlite3"),
        "--artifact-root",
        str(ctx.args.artifact_root / "postgame" / ctx.sid),
        "--session-id",
        ctx.sid,
        "--worker-id",
        "reconstructed-worker",
        "--fps",
        "60",
        "--batch-size",
        str(ctx.spec.batch_size),
        "--device",
        ctx.spec.device,
        "--max-empty-polls",
        "1",
        "--poll-seconds",
        "0.001",
    ]
    for name in (
        "capture_setup",
        "calibration_root",
        "config",
        "detector_training_run",
        "identity_training_run",
        "dinov2_assets",
    ):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(ctx.args, name))])
    for name, value in dict(
        environment_id=SCOPE.environment_id,
        device_id=SCOPE.device_id,
        table_id=SCOPE.table_id,
        open_request_id=ctx.request.open_request_id,
        resource_spec_sha256=ctx.spec.spec_sha256,
    ).items():
        argv.extend(["--managed-" + name.replace("_", "-"), value])
    return argv


def run_child(ctx, monkeypatch):
    module = _module()
    detector = _FakeDetector()
    detector.package_identity = VisionPackageIdentity(
        package_id=ctx.spec.model_package_id, manifest_sha256=ctx.spec.model_manifest_sha256
    )
    # Heavy model SDK/inference construction only: retained context, parsing,
    # exact calibration, worker, queue, artifacts, availability all stay actual.
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(from_checkpoint=lambda _: None))
    monkeypatch.setattr(module, "load_development_vision_backend", lambda *a, **kw: detector)
    monkeypatch.setattr(
        module, "load_development_group_classifier", lambda *a, **kw: _FakeClassifier()
    )
    monkeypatch.setattr(module, "Dinov2IdentityFeatureBackend", lambda **kw: object())
    ctx.args.calibration("b")
    args = child_argv(ctx)
    records, resolved = module._managed_context(module._parser().parse_args(args))
    assert resolved.spec_record.spec.calibration_artifact_id == "a" * 64
    assert records.load_session(ctx.sid).status.value == "ended"
    assert module.main(args) == 0
    jobs = SQLiteShotAnalysisJobStore(ctx.args.jobs_root / "analysis-jobs.sqlite3")
    with sqlite3.connect(ctx.args.jobs_root / "analysis-jobs.sqlite3") as connection:
        assert all(
            jobs.status(j.job_id).value == "available"
            for j in [
                jobs.snapshot(jid).job
                for jid in ctx.store.get_postgame_manifest(ctx.sid).expected_jobs
            ]
        ), connection.execute("SELECT failure_reason FROM shot_analysis_jobs").fetchall()
    return module


def test_actual_child_reconstructs_original_multiple_epochs_and_exact_calibration(
    scene, monkeypatch
):
    db = scene.args.jobs_root / "analysis-jobs.sqlite3"
    with sqlite3.connect(db) as connection:
        before = connection.execute(
            "SELECT job_id,payload_json,payload_sha256 FROM shot_analysis_jobs ORDER BY job_id"
        ).fetchall()
    run_child(scene, monkeypatch)
    with sqlite3.connect(db) as connection:
        assert (
            connection.execute(
                "SELECT job_id,payload_json,payload_sha256 FROM shot_analysis_jobs ORDER BY job_id"
            ).fetchall()
            == before
        )
    assert len(scene.store.list_postgame_clips(scene.sid)) == 2


@pytest.mark.parametrize("unavailable", [None, "foreign-scope", "missing-db"])
def test_screen_historical_media_uses_only_validated_original_queue(scene, tmp_path, unavailable):
    from dataclasses import replace

    from billiards_ai.lab.field_assistance.managed_resource_recovery import (
        ManagedRuntimeResourceUnavailable,
    )
    from billiards_ai.lab.field_assistance.server import _ServerState

    state = object.__new__(_ServerState)
    state.record_store = scene.store
    state.managed_session_scope = SCOPE
    state.session_id = "current-different-scene"
    current = SQLiteShotAnalysisJobStore(tmp_path / "current-jobs.sqlite3")
    state.media_job_store = current
    original = scene.args.jobs_root / "analysis-jobs.sqlite3"
    if unavailable == "foreign-scope":
        state.managed_session_scope = replace(SCOPE, table_id="foreign-table")
    elif unavailable == "missing-db":
        original.rename(original.with_suffix(".retained"))
    if unavailable:
        with pytest.raises(ManagedRuntimeResourceUnavailable):
            state._media_jobs_for_session(scene.sid)
        if unavailable == "missing-db":
            assert not original.exists()  # No replacement/discovery database is created.
    else:
        resolved = state._media_jobs_for_session(scene.sid)
        assert resolved._database == original
        assert len(resolved.list_session_jobs(scene.sid)) == 2
    assert state.media_job_store is current
    assert current.list_session_work_items(scene.sid) == ()


@pytest.mark.parametrize("crash_point", ["after_detach", "after_unlink"])
def test_actual_original_cleanup_recovers_journal_without_old_models(
    scene, monkeypatch, crash_point
):
    from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once

    run_child(scene, monkeypatch)
    manifest = scene.store.get_postgame_manifest(scene.sid)
    scene.store.complete_postgame_review(
        scene.spec.artifact_id, scene.sid, manifest.manifest_sha256
    )
    # Verify no asset dependency is introduced by cleanup; analysis notice stays.
    (scene.args.detector_training_run / "checkpoint_best.pth").unlink()
    diagnostics = {}
    resolver = recovery().ManagedRuntimeResourceResolver(scene.store, SCOPE, diagnostics)
    with pytest.raises(recovery().ManagedRuntimeResourceUnavailable):
        resolver.resolve(scene.sid, purpose="analysis")

    class Crash(BaseException):
        pass

    def fault(point, obj):
        if point == crash_point:
            raise Crash()

    now = datetime.now(UTC)
    context = resolver.resolve(scene.sid, purpose="cleanup")
    with pytest.raises(Crash):
        process_local_discard_once(
            scene.store,
            context.jobs_db,
            context.artifact_root,
            context.jobs_root,
            now_utc=now,
            fault_hook=fault,
            session_ids=(scene.sid,),
        )
    with sqlite3.connect(scene.store.path) as connection:
        journal = connection.execute(
            "SELECT progress_json FROM field_local_media_discards WHERE status='leased'"
        ).fetchone()[0]
    assert json.loads(journal)["objects"]
    # Reconstruct store/resolver/collector after partial detach/unlink and lease expiry.
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

    restored = SQLiteFieldRecordStore(scene.store.path)
    resolver = recovery().ManagedRuntimeResourceResolver(restored, SCOPE, diagnostics)
    worker = _BackgroundLocalMediaCleanupWorker(
        store=restored,
        jobs_db=scene.args.jobs_root / "wrong.sqlite",
        artifact_root=scene.args.artifact_root / "wrong",
        jobs_root=scene.args.jobs_root / "wrong",
        managed_scope=SCOPE,
        resource_resolver=resolver,
        process_once=lambda *a, **kw: process_local_discard_once(
            *a, now_utc=now + timedelta(seconds=40), **kw
        ),
    )
    assert worker.run_once()
    assert worker.run_once()
    assert not worker.run_once()
    assert diagnostics[scene.sid] == "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE"
    with sqlite3.connect(scene.store.path) as connection:
        rows = connection.execute(
            "SELECT status,progress_json FROM field_local_media_discards"
        ).fetchall()
    assert all(status == "completed" for status, _ in rows)
    assert not list((scene.args.artifact_root / "postgame").rglob("overlay.mp4"))
    assert all(
        (job.media_root / "interval.json").exists()
        for job in scene.jobs.list_session_jobs(scene.sid)
    )


def test_actual_child_ignores_unrelated_corrupt_due_job(scene, monkeypatch):
    from dataclasses import replace

    job = scene.jobs.list_session_jobs(scene.sid)[0]
    scene.jobs.enqueue(replace(job, job_id="foreign-job", session_id="foreign-session"))
    with sqlite3.connect(scene.args.jobs_root / "analysis-jobs.sqlite3") as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json='untrusted' WHERE job_id='foreign-job'"
        )
        # Put the foreign row first in the queue's existing attempt_count ordering.
        connection.execute(
            "UPDATE shot_analysis_jobs SET attempt_count=1 WHERE job_id!='foreign-job'"
        )
    run_child(scene, monkeypatch)
    assert scene.jobs.status("foreign-job").value == "pending"


@pytest.mark.parametrize(
    "changed",
    [
        "config",
        "jobs-db",
        "capture-setup",
        "calibration-root",
        "detector-training-run",
        "identity-training-run",
        "dinov2-assets",
        "artifact-root",
        "device",
        "batch-size",
        "fps",
        "managed-resource-spec-sha256",
        "managed-open-request-id",
    ],
)
def test_actual_child_wrong_context_never_reads_config_or_claims(scene, monkeypatch, changed):
    module = _module()
    args = child_argv(scene)
    index = args.index("--" + changed)
    args[index + 1] = {
        "device": "cuda",
        "batch-size": str(scene.spec.batch_size + 1),
        "fps": "59",
    }.get(changed) or (
        "0" * 64
        if changed == "managed-resource-spec-sha256"
        else "99999999-9999-4999-8999-999999999999"
        if changed == "managed-open-request-id"
        else str(scene.args.jobs_root / "missing")
    )
    fixtures.guard_child_pre_admission(monkeypatch, module)
    assert module.main(args) == 2
    with sqlite3.connect(scene.args.jobs_root / "analysis-jobs.sqlite3") as connection:
        assert connection.execute(
            "SELECT DISTINCT attempt_count FROM shot_analysis_jobs"
        ).fetchall() == [(0,)]
    assert not (scene.args.jobs_root / "missing").exists()


def test_late_created_discard_uses_original_roots_and_preserves_saved_clip_without_models(
    scene, monkeypatch
):
    run_child(scene, monkeypatch)
    resolver = recovery().ManagedRuntimeResourceResolver(scene.store, SCOPE, {})
    worker = _BackgroundLocalMediaCleanupWorker(
        store=scene.store,
        jobs_db=scene.args.jobs_root / "unused.sqlite",
        artifact_root=scene.args.artifact_root / "unused",
        jobs_root=scene.args.jobs_root / "unused",
        resource_resolver=resolver,
        managed_scope=SCOPE,
    )
    assert not worker.run_once()
    manifest = scene.store.get_postgame_manifest(scene.sid)
    receipt = scene.store.save_postgame_clip(
        request_id="saved-00000001",
        session_id=scene.sid,
        shot_id="shot-1",
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    clips = {clip.shot_id: clip for clip in scene.store.list_postgame_clips(scene.sid)}
    from pathlib import Path

    saved = Path(clips["shot-1"].local_path)
    saved_bytes = saved.read_bytes()
    scene.store.complete_postgame_review(
        scene.spec.artifact_id, scene.sid, manifest.manifest_sha256
    )
    (scene.args.detector_training_run / "checkpoint_best.pth").unlink()
    assert worker.run_once()
    assert not worker.run_once()
    assert saved.read_bytes() == saved_bytes
    assert (
        scene.store.save_postgame_clip(
            request_id="saved-00000001",
            session_id=scene.sid,
            shot_id="shot-1",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == receipt
    )
    assert not Path(clips["shot-2"].local_path).exists()
    assert not (scene.args.jobs_root / "unused.sqlite").exists()


@pytest.mark.parametrize(
    "sync",
    [
        ["--sync-base-url", "https://dev.invalid"],
        [
            "--development-analysis-sync",
            "--sync-base-url",
            "https://foreign.invalid",
            "--device-id",
            "device-1",
            "--credential-target",
            "billiardsAI/development/device-1",
            "--development-cloudbase-environment-id",
            "dev",
        ],
        [
            "--development-analysis-sync",
            "--sync-base-url",
            "https://dev.invalid",
            "--device-id",
            "device-1",
            "--credential-target",
            "foreign-target",
            "--development-cloudbase-environment-id",
            "dev",
        ],
    ],
)
def test_managed_child_refuses_unvalidated_ephemeral_sync_before_config(scene, monkeypatch, sync):
    module = _module()
    fixtures.guard_child_pre_admission(monkeypatch, module)
    db = scene.args.jobs_root / "analysis-jobs.sqlite3"
    before = db.read_bytes()
    assert module.main(child_argv(scene) + sync) == 2
    assert db.read_bytes() == before
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT DISTINCT attempt_count FROM shot_analysis_jobs"
        ).fetchall() == [(0,)]
