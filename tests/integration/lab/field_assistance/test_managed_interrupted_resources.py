"""Interrupted source diagnostics coexist with real recovery and accepted owner cleanup."""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_owner_delete_once
from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker
from billiards_ai.lab.mainline1_session_analysis import SQLiteSessionAnalysisOutbox
from tests.integration.lab.field_assistance.test_managed_resource_recovery import run_child
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import NOW, payload
from tests.unit.lab.field_assistance import test_managed_resource_recovery as fixtures
from tests.unit.lab.field_assistance.test_managed_startup import SCOPE

scene = fixtures.scene
resource_assets_fixture = fixtures.resource_assets_fixture


def immutable_rows(scene):
    with sqlite3.connect(scene.store.path) as db:
        shots = db.execute("SELECT payload_json FROM field_shots ORDER BY sequence").fetchall()
    with sqlite3.connect(scene.args.jobs_root / "analysis-jobs.sqlite3") as db:
        jobs = db.execute(
            "SELECT job_id,payload_json,payload_sha256 FROM shot_analysis_jobs ORDER BY job_id"
        ).fetchall()
    return shots, jobs


@pytest.mark.parametrize("scene", ["interrupted"], indirect=True)
def test_actual_source_interruption_keeps_real_jobs_recoverable_without_finalization(
    scene, monkeypatch
):
    manifest = scene.store.get_postgame_manifest(scene.sid)
    assert manifest.expected_jobs == ("job-1", "job-2")
    before = immutable_rows(scene)
    for purpose in ("analysis", "cleanup"):
        assert fixtures.resolve(scene, purpose).spec_record.spec == scene.spec
    run_child(scene, monkeypatch)
    assert immutable_rows(scene) == before
    assert scene.store.get_postgame_manifest(scene.sid) == manifest
    assert not scene.store.is_postgame_finalized(scene.sid)
    clips = scene.store.list_postgame_clips(scene.sid)
    assert {clip.shot_id for clip in clips} == {"shot-1", "shot-2"}
    assert all(Path(clip.local_path).read_bytes() for clip in clips)
    projection = (
        SQLiteSessionAnalysisOutbox(scene.args.jobs_root / "analysis-jobs.sqlite3")
        .entries(scene.sid)[-1]
        .payload
    )
    assert projection["status"] == "FAILED_RETRYING"
    assert projection["total_shots"] == 3 and projection["retrying_shots"] == 1
    assert projection["frozen_shot_ids"] == ["shot-1", "shot-2", "shot-3"]
    assert projection["published_clips"] == []
    with sqlite3.connect(scene.args.jobs_root / "analysis-jobs.sqlite3") as db:
        assert db.execute(
            "SELECT job_id,status,attempt_count FROM shot_analysis_jobs ORDER BY job_id"
        ).fetchall() == [("job-1", "available", 1), ("job-2", "available", 1)]


@pytest.mark.parametrize("scene", ["owner-interrupted"], indirect=True)
def test_accepted_owner_cleanup_uses_original_resources_despite_capture_loss_and_missing_models(
    scene, monkeypatch, tmp_path
):
    store, sid = scene.store, scene.sid
    manifest = store.get_postgame_manifest(sid)
    receipts = [
        store.save_postgame_clip(
            request_id=f"save-{n}",
            session_id=sid,
            shot_id=f"shot-{n}",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        for n in (1, 2)
    ]
    owners = [store.get_shot_ownership(sid, f"shot-{n}") for n in (1, 2, 3)]
    facts = [store.get_live_shot_fact(sid, f"shot-{n}") for n in (1, 2, 3)]
    assert [owner.actor_user_id for owner in owners] == ["user-a", "user-b", "user-b"]
    assert [fact.actor_user_id for fact in facts] == ["user-a", "user-b", "user-b"]
    run_child(scene, monkeypatch)
    clips = {clip.shot_id: clip for clip in store.list_postgame_clips(sid)}
    blue_overlay = Path(clips["shot-2"].local_path)
    blue_raw = scene.jobs.snapshot("job-2").job.media_root
    retained = {p: p.read_bytes() for p in [blue_overlay, *blue_raw.iterdir()]}
    task = OwnerDeleteDeviceTask.from_payload(payload(store, "shot-1"))
    assert task.owner_user_id == "user-a"
    accepted = store.accept_owner_delete_task(task, managed_scope=SCOPE)
    assert accepted.status == "pending"
    with sqlite3.connect(store.path) as db:
        saves = db.execute(
            "SELECT request_id,receipt_json FROM field_clip_save_requests ORDER BY request_id"
        ).fetchall()
    (scene.args.detector_training_run / "checkpoint_best.pth").unlink()
    restored = SQLiteFieldRecordStore(store.path, now_utc=lambda: NOW + timedelta(seconds=2))
    diagnostics = {}
    resolver = fixtures.recovery().ManagedRuntimeResourceResolver(restored, SCOPE, diagnostics)
    with pytest.raises(fixtures.recovery().ManagedRuntimeResourceUnavailable):
        resolver.resolve(sid, purpose="analysis")
    cleanup = _BackgroundLocalMediaCleanupWorker(
        store=restored,
        jobs_db=tmp_path / "wrong" / "jobs.sqlite",
        artifact_root=tmp_path / "wrong" / "artifacts",
        jobs_root=tmp_path / "wrong",
        managed_scope=SCOPE,
        resource_resolver=resolver,
        owner_process_once=lambda *a, **kw: process_local_owner_delete_once(
            *a, now_utc=NOW + timedelta(seconds=2), **kw
        ),
    )
    assert cleanup.run_once()
    completed = restored.get_owner_delete(accepted.clip_id)
    assert completed.status == "completed" and completed.receipt_json
    assert restored.accept_owner_delete_task(task, managed_scope=SCOPE) == completed
    assert not cleanup.run_once()
    assert not Path(clips["shot-1"].local_path).exists()
    red_raw = scene.jobs.snapshot("job-1").job.media_root
    assert not list(red_raw.glob("*.mjpgseg"))
    assert (red_raw / "interval.json").exists()
    assert {p: p.read_bytes() for p in retained} == retained
    assert [restored.get_shot_ownership(sid, f"shot-{n}") for n in (1, 2, 3)] == owners
    assert [restored.get_live_shot_fact(sid, f"shot-{n}") for n in (1, 2, 3)] == facts
    assert not restored.is_postgame_finalized(sid)
    assert diagnostics[sid] == "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE"
    assert not (tmp_path / "wrong").exists()
    with sqlite3.connect(store.path) as db:
        assert (
            db.execute(
                "SELECT request_id,receipt_json FROM field_clip_save_requests ORDER BY request_id"
            ).fetchall()
            == saves
        )
    assert (
        restored.save_postgame_clip(
            request_id="save-2",
            session_id=sid,
            shot_id="shot-2",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == receipts[1]
    )


@pytest.mark.parametrize("scene", ["interrupted"], indirect=True)
@pytest.mark.parametrize(
    "mutation",
    [
        "diagnostic_epoch_missing",
        "diagnostic_epoch_foreign",
        "diagnostic_spec",
        "diagnostic_wrong_shot",
        "unknown_reason",
        "real_job_missing",
        "duplicate_real_job",
        "real_job_hash",
        "real_job_epoch",
    ],
)
def test_interruption_never_exempts_missing_foreign_or_corrupt_original_work(scene, mutation):
    import json

    db_path = scene.args.jobs_root / "analysis-jobs.sqlite3"
    with sqlite3.connect(scene.store.path) as db:
        if mutation == "diagnostic_epoch_missing":
            db.execute(
                "DELETE FROM field_managed_capture_epochs WHERE source_epoch_id='epoch-interrupted'"
            )
        elif mutation == "diagnostic_epoch_foreign":
            db.execute(
                "UPDATE field_managed_capture_epochs SET session_id=? "
                "WHERE source_epoch_id='epoch-interrupted'",
                ("managed-99999999-9999-4999-8999-999999999999",),
            )
        elif mutation == "diagnostic_spec":
            db.execute(
                "UPDATE field_managed_capture_epochs SET spec_sha256=? "
                "WHERE source_epoch_id='epoch-interrupted'",
                ("0" * 64,),
            )
        elif mutation in {"diagnostic_wrong_shot", "unknown_reason", "duplicate_real_job"}:
            sid = "shot-2" if mutation == "duplicate_real_job" else "shot-3"
            shot = json.loads(
                db.execute(
                    "SELECT payload_json FROM field_shots WHERE shot_id=?", (sid,)
                ).fetchone()[0]
            )
            shot["live_job_id"] = {
                "diagnostic_wrong_shot": (
                    "abstained:capture_interrupted:epoch-interrupted:foreign-shot"
                ),
                "unknown_reason": "abstained:unknown:epoch-interrupted:shot-3",
                "duplicate_real_job": "job-1",
            }[mutation]
            db.execute(
                "UPDATE field_shots SET payload_json=? WHERE shot_id=?", (json.dumps(shot), sid)
            )
    with sqlite3.connect(db_path) as db:
        if mutation == "real_job_missing":
            db.execute("DELETE FROM shot_analysis_jobs WHERE job_id='job-2'")
        elif mutation == "real_job_hash":
            db.execute(
                "UPDATE shot_analysis_jobs SET payload_sha256=? WHERE job_id='job-2'", ("0" * 64,)
            )
        elif mutation == "real_job_epoch":
            import hashlib

            raw = db.execute(
                "SELECT payload_json FROM shot_analysis_jobs WHERE job_id='job-2'"
            ).fetchone()[0]
            job = json.loads(raw)
            job["source_epoch_id"] = "epoch-interrupted"
            raw = json.dumps(job, sort_keys=True, separators=(",", ":"))
            db.execute(
                "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? "
                "WHERE job_id='job-2'",
                (raw, hashlib.sha256(raw.encode()).hexdigest()),
            )
    before = immutable_rows(scene)
    manifest = scene.store.get_postgame_manifest(scene.sid)
    media = {p: p.read_bytes() for p in (scene.args.jobs_root / "media").rglob("*") if p.is_file()}
    for purpose in ("analysis", "cleanup"):
        with pytest.raises(fixtures.recovery().ManagedRuntimeResourceUnavailable):
            fixtures.resolve(scene, purpose)
    assert immutable_rows(scene) == before
    assert scene.store.get_postgame_manifest(scene.sid) == manifest
    assert {p: p.read_bytes() for p in media} == media
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT DISTINCT attempt_count FROM shot_analysis_jobs").fetchall() == [
            (0,)
        ]
