"""Physical owner erasure: real temporary copied media, SQLite and session pins."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

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
from billiards_ai.edge.route_assistance.runtime_resources import ManagedRuntimeResourceSpec
from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob
from tests.unit.edge.route_assistance.test_managed_session_persistence import snapshot
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    BIRTH,
    NOW,
    SCOPE,
    SESSION,
    task,
)
from tests.unit.edge.route_assistance.test_runtime_resources import payload as resource_payload


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def resource_spec(tmp_path, root, jobs):
    value = json.dumps(resource_payload(), ensure_ascii=False).replace(
        "/original/台球", str(tmp_path)
    )
    value = json.loads(value)
    value["paths"]["artifact_root"] = str(root)
    value["paths"]["jobs_root"] = str(jobs)
    value["artifact_id"] = "usb-field-" + sha(str(root).encode())
    value["rule_profile_id"] = None

    return ManagedRuntimeResourceSpec.from_payload(value)


def copied_media(root, jobs, sequence, resource):
    artifact = str(sequence) * 64
    output = root / "postgame" / SESSION / artifact
    output.mkdir(parents=True)
    overlay = output / "overlay.mp4"
    overlay.write_bytes(b"synthetic-overlay-" + str(sequence).encode())
    raw = jobs / "media" / SESSION / f"job-{sequence}"
    raw.mkdir(parents=True)
    segments = {"segment-1.mjpeg": b"raw-one", "segment-2.mjpgseg": b"raw-two"}
    for name, data in segments.items():
        (raw / name).write_bytes(data)
    identity = dict(
        schema_version="1.0",
        source_epoch_id=f"epoch-{sequence}",
        width=640,
        height=480,
        start_frame_id=1,
        end_frame_id=2,
        frame_count=2,
    )
    identity["segments"] = [{"name": name, "sha256": sha(data)} for name, data in segments.items()]
    digest = sha(canonical(identity).encode())
    interval = {key: value for key, value in identity.items() if key != "segments"}
    interval.update(media_sha256=digest, segment_sha256={n: sha(d) for n, d in segments.items()})
    (raw / "interval.json").write_text(canonical(interval))
    job = ShotAnalysisJob(
        f"job-{sequence}",
        SESSION,
        f"shot-{sequence:04d}",
        f"epoch-{sequence}",
        1,
        2,
        raw,
        digest,
        resource.capture_setup_id,
        resource.calibration_artifact_id,
        resource.config_sha256,
        resource.model_package_id,
        resource.identity_training_package_id,
        resource.identity_training_package_sha256,
    )
    encoded = canonical(job.identity())
    with sqlite3.connect(jobs / "analysis-jobs.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS shot_analysis_jobs("
            "job_id, payload_json, payload_sha256, status, artifact_id)"
        )
        db.execute(
            "INSERT INTO shot_analysis_jobs VALUES(?,?,?,?,?)",
            (job.job_id, encoded, sha(encoded.encode()), "available", artifact),
        )
    manifest = dict(job.identity())
    manifest.update(
        kind="mainline1_shot_local_video",
        artifact_role="usb_postgame_shot",
        source_kind="usb_live",
        live_shot_id=job.shot_id,
        artifact_id=artifact,
        inventory={
            "overlay.mp4": {"bytes": overlay.stat().st_size, "sha256": sha(overlay.read_bytes())},
            "summary.json": {"bytes": 2, "sha256": sha(b"{}")},
        },
    )
    (output / "manifest.json").write_text(canonical(manifest))
    (output / "summary.json").write_text("{}")
    return job, overlay, raw


@pytest.fixture
def media(tmp_path):
    root, jobs = tmp_path / "台球-artifacts", tmp_path / "jobs"
    resource = resource_spec(tmp_path, root, jobs)
    store = SQLiteFieldRecordStore(tmp_path / "field.db", now_utc=lambda: NOW)
    request = store.reserve_managed_open(
        SCOPE,
        artifact_id=resource.artifact_id,
        rule_profile_id=resource.rule_profile_id,
        resource_spec=resource,
    )
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
    store.bind_artifact_path(resource.artifact_id, root)
    objects = {}
    for sequence, role in ((1, WristbandRole.RED), (2, WristbandRole.BLUE)):
        if sequence == 2:
            store.save_match(
                SESSION,
                replace(store.load_session(SESSION).match, active_role=role, revision=2),
                expected_revision=1,
            )
        job, overlay, raw = copied_media(root, jobs, sequence, resource)
        objects[sequence] = (overlay, raw)
        store.register_managed_capture_epoch(SCOPE, request.open_request_id, job.source_epoch_id)
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
                resource.artifact_id,
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
                str(raw),
                job.media_sha256,
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
            local_path=objects[sequence][0],
            content_sha256=sha(objects[sequence][0].read_bytes()),
            byte_size=objects[sequence][0].stat().st_size,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        if sequence == 2:
            store.save_postgame_clip(
                request_id="save-2",
                session_id=SESSION,
                shot_id="shot-0002",
                postgame_manifest_sha256=manifest.manifest_sha256,
            )
    value = task(store)
    accepted = store.accept_owner_delete_task(value, managed_scope=SCOPE)
    return store, jobs / "analysis-jobs.sqlite3", root, jobs, *objects[1], accepted, objects[2]


def run(media, **kwargs):
    from billiards_ai.edge.route_assistance.local_media_cleanup import (
        process_local_owner_delete_once,
    )

    store, db, root, jobs, *_ = media
    kwargs.setdefault("now_utc", NOW)
    return process_local_owner_delete_once(store, db, root, jobs, **kwargs)


def test_selected_owner_unlinks_exact_media_and_preserves_metadata_and_opponent(media):
    store, _, root, jobs, overlay, raw, accepted, opponent = media
    before = snapshot(store.path)
    metadata = {
        p: p.read_bytes()
        for p in (
            overlay.parent / "manifest.json",
            overlay.parent / "summary.json",
            raw / "interval.json",
        )
    }
    opponent_files = {
        p: p.read_bytes()
        for directory in (opponent[0].parent, opponent[1])
        for p in directory.iterdir()
    }
    result = run(media)
    assert result.outcome == "completed", result.error_code
    assert result.reclaimed_bytes == len(b"synthetic-overlay-1raw-oneraw-two")
    assert not overlay.exists()
    assert not (raw / "segment-1.mjpeg").exists()
    assert not (raw / "segment-2.mjpgseg").exists()
    assert all(p.read_bytes() == data for p, data in (metadata | opponent_files).items())
    after = snapshot(store.path)
    mutable = {"field_owner_media_deletions", "field_owner_completion_outbox"}
    assert {k: v for k, v in before.items() if k not in mutable} == {
        k: v for k, v in after.items() if k not in mutable
    }
    assert before["field_owner_completion_outbox"] == []
    assert len(after["field_owner_completion_outbox"]) == 1
    state = store.get_owner_delete(accepted.clip_id)
    progress, receipt = json.loads(state.progress_json), json.loads(state.receipt_json)
    assert state.status == "completed"
    assert state.task == accepted.task and state.task_sha256 == accepted.task_sha256
    assert state.receipt_sha256 == sha(state.receipt_json.encode())
    assert receipt == dict(
        contract="local_owner_erase_receipt_v1",
        deletion_job_id=accepted.task.deletion_job_id,
        clip_id=accepted.clip_id,
        task_sha256=accepted.task_sha256,
        cloud_erase_receipt_sha256="e" * 64,
        journal_sha256=sha(canonical(progress).encode()),
        reclaimed_bytes=result.reclaimed_bytes,
    )
    assert len(progress["objects"]) == 3
    for index, obj in enumerate(progress["objects"]):
        staging_root = root if index == 0 else jobs
        expected = staging_root / ".local-owner-media-cleanup" / sha(accepted.clip_id.encode())
        assert Path(obj["staged_path"]) == expected / f"{index:03d}.detached"
        assert obj["state"] == "deleted" and not Path(obj["staged_path"]).exists()
    assert "台球" not in state.progress_json and "\\u53f0" in state.progress_json
    assert not (root / ".local-media-cleanup").exists()
    assert not after["field_local_media_discards"]
    assert (store.path.parent / ".local-owner-media-cleanup-control").is_dir()
    assert not (root / ".local-owner-media-cleanup-control").exists()


@pytest.mark.parametrize(
    "unsafe",
    [
        "missing_overlay",
        "missing_segment",
        "unknown_raw",
        "unknown_output",
        "corrupt_overlay",
        "corrupt_segment",
        "hardlink_overlay",
        "hardlink_segment",
        "symlink_overlay",
        "symlink_raw",
        "shared_recording",
        "shared_payload",
        "job_digest",
        "job_status",
        "job_artifact",
        "output_artifact",
        "root_binding",
        "resource_missing",
        "resource_root",
        "resource_jobs",
        "foreign_db",
        "foreign_jobs",
        "foreign_artifact_root",
        "output_metadata_link",
    ],
)
def test_unsafe_selected_inventory_never_detaches_any_media(media, unsafe):
    store, db, root, jobs, overlay, raw, accepted, _ = media
    if unsafe.startswith("missing_"):
        (overlay if unsafe == "missing_overlay" else raw / "segment-1.mjpeg").unlink()
    elif unsafe.startswith("unknown_"):
        ((raw if unsafe == "unknown_raw" else overlay.parent) / "unknown.mp4").write_bytes(b"x")
    elif unsafe.startswith("corrupt_"):
        (overlay if unsafe == "corrupt_overlay" else raw / "segment-1.mjpeg").write_bytes(
            b"changed"
        )
    elif unsafe.startswith("hardlink_"):
        path = overlay if unsafe == "hardlink_overlay" else raw / "segment-1.mjpeg"
        os.link(path, root.parent / "linked")
    elif unsafe == "symlink_overlay":
        overlay.unlink()
        overlay.symlink_to(media[7][0])
    elif unsafe == "symlink_raw":
        target = raw.with_name("raw-real")
        raw.rename(target)
        raw.symlink_to(target, target_is_directory=True)
    elif unsafe == "output_metadata_link":
        path = overlay.parent / "summary.json"
        path.unlink()
        path.symlink_to(overlay)
    elif unsafe == "shared_recording":
        path = raw / "interval.json"
        value = json.loads(path.read_text()) | {"storage_kind": "session_segment_reference"}
        path.write_text(canonical(value))
    elif unsafe in {"shared_payload", "job_digest", "job_status", "job_artifact"}:
        with sqlite3.connect(db) as connection:
            if unsafe == "shared_payload":
                payload = json.loads(
                    connection.execute(
                        "SELECT payload_json FROM shot_analysis_jobs WHERE job_id='job-1'"
                    ).fetchone()[0]
                )
                payload["media_root"] = str(jobs / "shared-recording")
                encoded = canonical(payload)
                connection.execute(
                    "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? "
                    "WHERE job_id='job-1'",
                    (encoded, sha(encoded.encode())),
                )
            else:
                column, value = {
                    "job_digest": ("payload_sha256", "0" * 64),
                    "job_status": ("status", "pending"),
                    "job_artifact": ("artifact_id", "f" * 64),
                }[unsafe]
                connection.execute(
                    f"UPDATE shot_analysis_jobs SET {column}=? WHERE job_id='job-1'", (value,)
                )
    elif unsafe == "output_artifact":
        path = overlay.parent / "manifest.json"
        path.write_text(canonical(json.loads(path.read_text()) | {"artifact_id": "f" * 64}))
    elif unsafe == "root_binding":
        with sqlite3.connect(store.path) as connection:
            connection.execute("UPDATE field_session_artifacts SET artifact_path=?", (str(jobs),))
    elif unsafe.startswith("resource_"):
        with sqlite3.connect(store.path) as connection:
            if unsafe == "resource_missing":
                connection.execute("DELETE FROM field_managed_resource_specs")
            else:
                value = json.loads(
                    connection.execute(
                        "SELECT spec_json FROM field_managed_resource_specs"
                    ).fetchone()[0]
                )
                key = "artifact_root" if unsafe == "resource_root" else "jobs_root"
                value["paths"][key] = str(root.parent / "foreign")
                if key == "artifact_root":
                    value["artifact_id"] = "usb-field-" + sha(value["paths"][key].encode())
                spec = ManagedRuntimeResourceSpec.from_payload(value)
                connection.execute(
                    "UPDATE field_managed_resource_specs SET spec_json=?,spec_sha256=?",
                    (spec.canonical_json, spec.spec_sha256),
                )
    elif unsafe.startswith("foreign_"):
        values = list(media)
        index = {"foreign_db": 1, "foreign_jobs": 3, "foreign_artifact_root": 2}[unsafe]
        foreign = root.parent / unsafe
        if index == 1:
            foreign.write_bytes(db.read_bytes())
        else:
            foreign.mkdir()
        values[index] = foreign
        media = tuple(values)
    result = run(media)
    assert result.outcome == "blocked", result
    assert store.get_owner_delete(accepted.clip_id).status == "blocked"
    assert result.reclaimed_bytes == 0
    assert not list(root.rglob("*.detached"))
    assert not list(jobs.rglob("*.detached"))
    assert (raw / "segment-2.mjpgseg").is_file()
    if unsafe != "missing_overlay":
        assert overlay.exists()


def test_owner_scope_filter_can_leave_task_unclaimed(media):
    assert run(media, session_ids=()).outcome == "idle"
    assert media[0].get_owner_delete(media[6].clip_id).status == "pending"
    assert media[4].is_file()


class PowerLoss(BaseException):
    pass


@pytest.mark.parametrize("point", ["after_detach", "after_unlink"])
def test_restart_renews_local_token_and_recovers_exact_journal(media, point):
    store, _, _, _, overlay, raw, accepted, _ = media

    def crash(observed, obj):
        if observed == point:
            raise PowerLoss()

    with pytest.raises(PowerLoss):
        run(media, fault_hook=crash, lease_seconds=1)
    crashed = store.get_owner_delete(accepted.clip_id)
    progress = json.loads(crashed.progress_json)
    assert progress["objects"][0]["state"] == (
        "planned" if point == "after_detach" else "unlinking"
    )
    assert not overlay.exists()
    staged = Path(progress["objects"][0]["staged_path"])
    assert staged.exists() == (point == "after_detach")
    restarted = SQLiteFieldRecordStore(store.path, now_utc=lambda: NOW)
    resumed_media = (restarted, *media[1:])
    result = run(resumed_media, now_utc=NOW + timedelta(seconds=2))
    assert result.outcome == "completed", result
    final = restarted.get_owner_delete(accepted.clip_id)
    assert final.task_sha256 == crashed.task_sha256
    assert final.attempts == 2
    assert not staged.exists() and not overlay.exists()
    assert not list(raw.glob("*.mjpeg")) and not list(raw.glob("*.mjpgseg"))
    journal = json.loads(final.progress_json)
    if point == "after_unlink":
        assert journal["objects"][0]["state"] == "absent"
        assert journal["objects"][0]["reclaimed_bytes"] == 0
        assert result.reclaimed_bytes == len(b"raw-oneraw-two")
    else:
        assert result.reclaimed_bytes == len(b"synthetic-overlay-1raw-oneraw-two")
    assert not store.update_owner_delete_progress(crashed, crashed.progress_json)


def test_completed_delivery_replays_original_receipt_without_filesystem_mutation(
    media, monkeypatch
):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    assert run(media).outcome == "completed"
    store, _, _, _, _, _, accepted, _ = media
    completed = store.get_owner_delete(accepted.clip_id)
    replay = store.accept_owner_delete_task(accepted.task, managed_scope=SCOPE)
    assert replay == completed

    def forbidden(*args, **kwargs):
        pytest.fail("completed task must not enter media ports")

    monkeypatch.setattr(cleanup, "_port", forbidden)
    assert run(media).outcome == "idle"
    assert store.get_owner_delete(accepted.clip_id) == completed


def test_real_reader_pin_prevents_unlink_and_receipt_then_retry_succeeds(media):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    with session_media_lock(media[2], SESSION, exclusive=False):
        assert run(media).outcome == "retry"
        state = media[0].get_owner_delete(media[6].clip_id)
        assert state.status == "retry" and state.receipt_json is None
        assert state.progress_json == "{}"
        assert media[4].is_file()
    assert run(media, now_utc=NOW + timedelta(seconds=3)).outcome == "completed"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_planned",
        "missing_detached",
        "replacement_stage",
        "duplicate_source",
        "unknown_stage",
        "reappeared",
    ],
)
def test_recovery_blocks_unproven_absence_or_replacement(media, failure):
    def crash(point, obj):
        if point == ("after_unlink" if failure == "reappeared" else "after_detach"):
            raise PowerLoss()

    with pytest.raises(PowerLoss):
        run(media, fault_hook=crash, lease_seconds=1)
    state = media[0].get_owner_delete(media[6].clip_id)
    first = json.loads(state.progress_json)["objects"][0]
    staged = Path(first["staged_path"])
    if failure == "missing_planned":
        staged.unlink()
    elif failure == "missing_detached":
        value = json.loads(state.progress_json)
        value["objects"][0]["state"] = "detached"
        assert media[0].update_owner_delete_progress(state, canonical(value))
        staged.unlink()
    elif failure == "replacement_stage":
        staged.write_bytes(b"replacement")
    elif failure == "duplicate_source":
        media[4].write_bytes(b"replacement")
    elif failure == "unknown_stage":
        staged.with_name("unknown").write_bytes(b"unknown")
    else:
        # A terminal deletion checkpoint is durable before a later crash.
        value = json.loads(state.progress_json)
        value["objects"][0]["state"] = "deleted"
        value["objects"][0]["reclaimed_bytes"] = first["size"]
        assert media[0].update_owner_delete_progress(state, canonical(value))
        media[4].write_bytes(b"reappeared")
    result = run(media, now_utc=NOW + timedelta(seconds=2))
    assert result.outcome == "blocked", result
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert (media[5] / "segment-2.mjpgseg").is_file()
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None


def test_lost_checkpoint_fence_stops_before_first_detach(media, monkeypatch):
    original = media[0].update_owner_delete_progress

    def rotate_before_inventory_checkpoint(claim, encoded):
        if json.loads(encoded):
            replacement = media[0].claim_due_owner_delete(
                "replacement", NOW + timedelta(seconds=31)
            )
            assert replacement.claim_token != claim.claim_token
        return original(claim, encoded)

    monkeypatch.setattr(
        media[0], "update_owner_delete_progress", rotate_before_inventory_checkpoint
    )
    assert run(media).outcome == "stale"
    assert media[4].is_file()
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None


def test_lost_fence_after_final_verify_preserves_staged_object(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    original = cleanup._verify
    rotated = []

    def rotate(descriptor, obj, cancelled):
        original(descriptor, obj, cancelled)
        if obj["state"] == "unlinking" and not rotated:
            rotated.append(
                media[0].claim_due_owner_delete("replacement", NOW + timedelta(seconds=31))
            )

    monkeypatch.setattr(cleanup, "_verify", rotate)
    assert run(media).outcome == "stale"
    staged = next(media[2].rglob("*.detached"))
    assert staged.read_bytes() == b"synthetic-overlay-1"
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None


def test_expired_live_executor_cannot_be_taken_over_by_second_collector(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    original = cleanup._PosixFS.unlink
    attempts = []

    def compete(self, directory, path, descriptor):
        if not attempts:
            old_token = media[0].get_owner_delete(media[6].clip_id).claim_token
            attempts.append(run(media, now_utc=NOW + timedelta(hours=1)))
            assert media[0].get_owner_delete(media[6].clip_id).claim_token == old_token
            with pytest.raises(TimeoutError):
                with session_media_lock(media[2], SESSION, exclusive=False):
                    pytest.fail("reader entered during owner erasure")
        return original(self, directory, path, descriptor)

    monkeypatch.setattr(cleanup._PosixFS, "unlink", compete)
    assert run(media, lease_seconds=1).outcome == "completed"
    assert attempts[0].outcome == "retry"
    assert attempts[0].error_code == "collector_control_unavailable"
    assert media[0].get_owner_delete(media[6].clip_id).attempts == 1


@pytest.mark.parametrize("unsafe", ["unknown_stage", "staged_collision", "cross_volume"])
def test_all_staging_is_preflighted_before_first_detach(media, monkeypatch, unsafe):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    jobs_stage = media[3] / ".local-owner-media-cleanup" / sha(media[6].clip_id.encode())
    jobs_stage.mkdir(parents=True, mode=0o700)
    jobs_stage.parent.chmod(0o700)
    if unsafe == "unknown_stage":
        (jobs_stage / "unknown").write_bytes(b"foreign")
    elif unsafe == "staged_collision":
        (jobs_stage / "001.detached").write_bytes(b"foreign")
    else:
        original = cleanup._PosixFS.device

        def wrong_device(self, directory):
            value = original(self, directory)
            return value + 1 if os.fstat(directory).st_ino == jobs_stage.stat().st_ino else value

        monkeypatch.setattr(cleanup._PosixFS, "device", wrong_device)
    result = run(media)
    assert result.outcome == "blocked", result
    assert media[4].is_file()
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert result.reclaimed_bytes == 0


def test_reappeared_source_after_unlink_prevents_completion_receipt(media):
    def reappear(point, obj):
        if point == "after_unlink":
            Path(obj["path"]).write_bytes(b"reappeared")

    result = run(media, fault_hook=reappear)
    assert result.outcome == "blocked"
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None
    assert media[4].read_bytes() == b"reappeared"


@pytest.mark.parametrize("point", ["after_detach", "after_unlink"])
def test_io_retry_preserves_durable_journal_and_resumes(media, point):
    import errno

    def interrupted(observed, obj):
        if observed == point:
            raise OSError(errno.EIO, "synthetic device interruption")

    result = run(media, fault_hook=interrupted)
    state = media[0].get_owner_delete(media[6].clip_id)
    assert result.outcome == "retry" and result.error_code == "media_busy"
    assert state.status == "retry" and state.receipt_json is None
    assert json.loads(state.progress_json)["objects"][0]["state"] == (
        "planned" if point == "after_detach" else "unlinking"
    )
    assert run(media, now_utc=NOW + timedelta(seconds=3)).outcome == "completed"


def test_cancellation_during_detach_retries_without_unlink(media):
    stopped = []

    def stop(point, obj):
        if point == "after_detach":
            stopped.append(True)

    result = run(media, fault_hook=stop, cancelled=lambda: bool(stopped))
    assert result.outcome == "retry"
    assert next(media[2].rglob("*.detached")).read_bytes() == b"synthetic-overlay-1"
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None


def test_cancelled_before_claim_never_acquires_authority(media):
    assert run(media, cancelled=lambda: True).outcome == "retry"
    assert media[0].get_owner_delete(media[6].clip_id).status == "pending"
    assert media[4].is_file()


@pytest.mark.parametrize("limit", ["objects", "metadata", "media", "journal"])
def test_owner_bounded_preinventory_blocks_without_mutation(media, monkeypatch, limit):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    if limit == "objects":
        interval = media[5] / "interval.json"
        value = json.loads(interval.read_text())
        value["segment_sha256"] = {f"segment-{i}.mjpeg": "a" * 64 for i in range(64)}
        interval.write_text(canonical(value))
    elif limit == "metadata":
        (media[5] / "interval.json").write_text(" " * (1024 * 1024 + 1))
    elif limit == "media":
        monkeypatch.setattr(cleanup, "_MAX_MEDIA_BYTES", 1)
    else:
        inventory = cleanup._inventory

        def oversized(*args, **kwargs):
            value = inventory(*args, **kwargs)
            value["overflow"] = "x" * 65536
            return value

        monkeypatch.setattr(cleanup, "_inventory", oversized)
    result = run(media)
    assert result.outcome == "blocked", result
    assert media[4].is_file()
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None


def test_reparse_attribute_test_port_blocks_before_any_detach(media, monkeypatch):
    from types import SimpleNamespace

    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    original = os.fstat
    selected = (media[5] / "segment-1.mjpeg").stat().st_ino

    def reparse(descriptor):
        info = original(descriptor)
        if info.st_ino == selected:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_nlink=info.st_nlink,
                st_file_attributes=0x400,
                st_ino=info.st_ino,
            )
        return info

    monkeypatch.setattr(cleanup.os, "fstat", reparse)
    assert run(media).outcome == "blocked"
    assert media[4].is_file()
    assert (media[5] / "segment-1.mjpeg").is_file()


@pytest.mark.parametrize("platform", ["nt", "unsupported"])
def test_owner_port_unavailable_fails_closed_without_posix_fallback(media, monkeypatch, platform):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    def unavailable():
        raise AttributeError("synthetic native API unavailable")

    monkeypatch.setattr(cleanup, "_windows_api", unavailable)
    assert run(media, platform=platform).outcome == "blocked"
    assert media[4].is_file()
    assert (media[5] / "segment-1.mjpeg").is_file()


def test_cleanup_does_not_remove_directories_or_unrelated_metadata(media):
    root, jobs = media[2], media[3]
    before = {p for base in (root, jobs) for p in base.rglob("*") if p.is_dir()}
    sentinel = jobs / "kept.json"
    sentinel.write_bytes(b"unrelated-metadata")
    assert run(media).outcome == "completed"
    assert all(p.is_dir() for p in before)
    assert sentinel.read_bytes() == b"unrelated-metadata"


def test_archive_path_is_rejected_before_any_filesystem_lookup(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    def forbidden(*args, **kwargs):
        pytest.fail("archive path must be rejected before filesystem lookup")

    monkeypatch.setattr(cleanup, "_assert_directory_chain", forbidden)
    values = (media[0], media[1], media[2] / ".project-archive", *media[3:])
    assert run(values).outcome == "blocked"
    assert media[4].is_file()


def test_completion_fence_loss_keeps_terminal_journal_for_next_local_claim(media, monkeypatch):
    store = media[0]
    complete = store.complete_owner_delete

    def rotate_before_complete(claim, **kwargs):
        replacement = store.claim_due_owner_delete("replacement", NOW + timedelta(seconds=31))
        assert replacement.claim_token != claim.claim_token
        return complete(claim, **kwargs)

    monkeypatch.setattr(store, "complete_owner_delete", rotate_before_complete)
    assert run(media).outcome == "stale"
    state = store.get_owner_delete(media[6].clip_id)
    assert state.receipt_json is None
    assert all(obj["state"] == "deleted" for obj in json.loads(state.progress_json)["objects"])
    assert not media[4].exists()
    monkeypatch.setattr(store, "complete_owner_delete", complete)
    assert run(media, now_utc=NOW + timedelta(seconds=62)).outcome == "completed"


@pytest.mark.parametrize(
    "authority", ["availability", "outbox", "save", "ownership", "shot", "media"]
)
def test_selected_authority_is_revalidated_inside_pin_before_media_reads(media, authority):
    store = media[0]
    with sqlite3.connect(store.path) as connection:
        if authority == "availability":
            connection.execute(
                "UPDATE field_postgame_clip_availability SET content_sha256=? "
                "WHERE shot_id='shot-0001'",
                ("0" * 64,),
            )
        elif authority == "outbox":
            connection.execute(
                "UPDATE field_saved_media_outbox SET last_error_code='OTHER' WHERE clip_id=?",
                (media[6].clip_id,),
            )
        else:
            table, column, where = {
                "save": ("field_clip_save_requests", "receipt_json", "request_id='save-1'"),
                "ownership": ("field_shot_ownership", "payload_json", "shot_id='shot-0001'"),
                "shot": ("field_shots", "payload_json", "shot_id='shot-0001'"),
                "media": ("field_live_shot_media", "payload_json", "shot_id='shot-0001'"),
            }[authority]
            connection.execute(f"UPDATE {table} SET {column}={column} || ' ' WHERE {where}")
    result = run(media)
    assert result.outcome == "blocked", result
    assert media[4].is_file()
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert store.get_owner_delete(media[6].clip_id).receipt_json is None


@pytest.mark.parametrize(
    "dependency",
    [
        "capture_setup_id",
        "calibration_artifact_id",
        "config_sha256",
        "model_package_id",
        "identity_training_package_id",
        "identity_training_package_sha256",
    ],
)
def test_available_job_dependency_must_match_original_resource_spec(media, dependency):
    with sqlite3.connect(media[1]) as connection:
        value = json.loads(
            connection.execute(
                "SELECT payload_json FROM shot_analysis_jobs WHERE job_id='job-1'"
            ).fetchone()[0]
        )
        value[dependency] = "f" * 64
        encoded = canonical(value)
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? WHERE job_id='job-1'",
            (encoded, sha(encoded.encode())),
        )
    path = media[4].parent / "manifest.json"
    path.write_text(canonical(json.loads(path.read_text()) | {dependency: value[dependency]}))
    assert run(media).outcome == "blocked"
    assert media[4].is_file()
    assert (media[5] / "segment-1.mjpeg").is_file()


@pytest.mark.parametrize("lineage", ["missing_epoch", "epoch_spec", "media_digest", "source_epoch"])
def test_copied_job_must_match_original_selected_live_reference_and_epoch(media, lineage):
    if lineage in {"missing_epoch", "epoch_spec"}:
        with sqlite3.connect(media[0].path) as connection:
            if lineage == "missing_epoch":
                connection.execute(
                    "DELETE FROM field_managed_capture_epochs WHERE source_epoch_id='epoch-1'"
                )
            else:
                connection.execute(
                    "UPDATE field_managed_capture_epochs SET spec_sha256=? "
                    "WHERE source_epoch_id='epoch-1'",
                    ("0" * 64,),
                )
    else:
        interval_path = media[5] / "interval.json"
        interval = json.loads(interval_path.read_text())
        if lineage == "media_digest":
            (media[5] / "segment-1.mjpeg").write_bytes(b"different-raw")
            interval["segment_sha256"]["segment-1.mjpeg"] = sha(b"different-raw")
        else:
            interval["source_epoch_id"] = "different-epoch"
        identity = {
            key: interval[key]
            for key in (
                "schema_version",
                "source_epoch_id",
                "width",
                "height",
                "start_frame_id",
                "end_frame_id",
                "frame_count",
            )
        }
        identity["segments"] = [
            {"name": n, "sha256": d} for n, d in interval["segment_sha256"].items()
        ]
        # Match the manifest's insertion order exactly to the real source producer.
        identity["segments"].sort(key=lambda obj: obj["name"])
        interval["media_sha256"] = sha(canonical(identity).encode())
        interval_path.write_text(canonical(interval))
        with sqlite3.connect(media[1]) as connection:
            value = json.loads(
                connection.execute(
                    "SELECT payload_json FROM shot_analysis_jobs WHERE job_id='job-1'"
                ).fetchone()[0]
            )
            value.update(
                source_epoch_id=interval["source_epoch_id"], media_sha256=interval["media_sha256"]
            )
            encoded = canonical(value)
            connection.execute(
                "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? "
                "WHERE job_id='job-1'",
                (encoded, sha(encoded.encode())),
            )
        manifest_path = media[4].parent / "manifest.json"
        manifest_path.write_text(canonical(json.loads(manifest_path.read_text()) | value))
    result = run(media)
    assert result.outcome == "blocked", result
    assert media[4].is_file()
    assert (media[5] / "segment-2.mjpgseg").is_file()


def short_busy_timeout(store, monkeypatch):
    connect = store._connect

    def short_connect():
        connection = connect()
        connection.execute("PRAGMA busy_timeout=1")
        return connection

    monkeypatch.setattr(store, "_connect", short_connect)


def test_busy_authority_database_before_claim_returns_retry_without_mutation(media, monkeypatch):
    store = media[0]
    short_busy_timeout(store, monkeypatch)
    with sqlite3.connect(store.path) as lock:
        lock.execute("BEGIN IMMEDIATE")
        result = run(media)
    assert result.outcome == "retry"
    assert media[4].is_file()
    assert store.get_owner_delete(media[6].clip_id).status == "pending"


def test_wrapped_sqlite_busy_during_authority_recheck_retries(media, monkeypatch):
    store = media[0]
    short_busy_timeout(store, monkeypatch)
    accept = store.accept_owner_delete_task

    def temporarily_busy(*args, **kwargs):
        with sqlite3.connect(store.path) as lock:
            lock.execute("BEGIN IMMEDIATE")
            return accept(*args, **kwargs)

    monkeypatch.setattr(store, "accept_owner_delete_task", temporarily_busy)
    result = run(media)
    assert result.outcome == "retry"
    assert store.get_owner_delete(media[6].clip_id).status == "retry"
    assert media[4].is_file()


def test_busy_terminal_checkpoint_retains_lease_and_journal_for_recovery(media, monkeypatch):
    store = media[0]
    short_busy_timeout(store, monkeypatch)
    lock = sqlite3.connect(store.path)

    def busy(point, obj):
        if point == "after_detach":
            lock.execute("BEGIN IMMEDIATE")

    try:
        result = run(media, fault_hook=busy, lease_seconds=1)
        assert result.outcome == "retry"
        assert next(media[2].rglob("*.detached")).read_bytes() == b"synthetic-overlay-1"
    finally:
        lock.rollback()
        lock.close()
    state = store.get_owner_delete(media[6].clip_id)
    assert state.status == "leased" and state.receipt_json is None
    assert json.loads(state.progress_json)["objects"][0]["state"] == "planned"
    assert run(media, now_utc=NOW + timedelta(seconds=2)).outcome == "completed"


@pytest.mark.parametrize("change", ["discard_namespace", "source_path", "digest", "inode"])
def test_recovery_journal_cannot_expand_original_owner_inventory(media, change):
    def crash(point, obj):
        if point == "after_detach":
            raise PowerLoss()

    with pytest.raises(PowerLoss):
        run(media, fault_hook=crash, lease_seconds=1)
    state = media[0].get_owner_delete(media[6].clip_id)
    journal = json.loads(state.progress_json)
    obj = journal["objects"][0]
    original_staged = Path(obj["staged_path"])
    if change == "discard_namespace":
        obj["staged_path"] = obj["staged_path"].replace(
            ".local-owner-media-cleanup/", ".local-media-cleanup/"
        )
    elif change == "source_path":
        obj["path"] = str(media[7][0])
    elif change == "digest":
        obj["sha256"] = "0" * 64
    else:
        obj["identity"][1] += 1
    assert media[0].update_owner_delete_progress(state, canonical(journal))
    result = run(media, now_utc=NOW + timedelta(seconds=2))
    assert result.outcome == "blocked"
    assert original_staged.read_bytes() == b"synthetic-overlay-1"
    assert media[7][0].read_bytes() == b"synthetic-overlay-2"
    assert (media[5] / "segment-1.mjpeg").is_file()
    assert media[0].get_owner_delete(media[6].clip_id).receipt_json is None


@pytest.mark.parametrize("reopen_store", [False, True])
def test_different_root_collector_cannot_take_over_expired_live_unlink(
    media, monkeypatch, reopen_store
):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    store, db, root, jobs, _, raw, accepted, _ = media
    foreign_root = root.parent / "foreign-artifacts"
    foreign_root.mkdir()
    sentinel = foreign_root / "unrelated.mp4"
    sentinel.write_bytes(b"foreign-owner-media")
    if reopen_store:
        alias = store.path.with_name("field-alias.db")
        alias.symlink_to(store.path)
        competitor = SQLiteFieldRecordStore(alias)
        assert competitor.path == store.path
    else:
        competitor = store
    unlink = cleanup._PosixFS.unlink
    attempts, before_and_after = [], []

    def compete(self, directory, path, descriptor):
        if not attempts:
            before = store.get_owner_delete(accepted.clip_id)
            staged_bytes = path.read_bytes()
            source_bytes = (raw / "segment-1.mjpeg").read_bytes()
            attempts.append(
                cleanup.process_local_owner_delete_once(
                    competitor, db, foreign_root, jobs, now_utc=NOW + timedelta(hours=1)
                )
            )
            before_and_after.append((before, store.get_owner_delete(accepted.clip_id)))
            assert path.read_bytes() == staged_bytes
            assert (raw / "segment-1.mjpeg").read_bytes() == source_bytes
            assert sentinel.read_bytes() == b"foreign-owner-media"
        return unlink(self, directory, path, descriptor)

    monkeypatch.setattr(cleanup._PosixFS, "unlink", compete)
    original = run(media, lease_seconds=1)
    assert original.outcome == "completed", (original, attempts, before_and_after)
    assert attempts[0].outcome == "retry"
    assert attempts[0].error_code == "collector_control_unavailable"
    before, after = before_and_after[0]
    assert before == after
    assert after.claim_token == before.claim_token and after.attempts == 1
    completed = store.get_owner_delete(accepted.clip_id)
    assert completed.status == "completed" and completed.attempts == 1
    assert completed.receipt_json is not None
    assert not media[4].exists()
    assert not (raw / "segment-1.mjpeg").exists()
    assert not (raw / "segment-2.mjpgseg").exists()
    assert list(foreign_root.iterdir()) == [sentinel]


def test_store_control_lock_is_independent_for_distinct_databases_in_same_parent(
    media, monkeypatch
):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    independent = SQLiteFieldRecordStore(media[0].path.with_name("independent.db"))
    unlink = cleanup._PosixFS.unlink
    attempts = []

    def inspect_other_store(self, directory, path, descriptor):
        if not attempts:
            attempts.append(
                cleanup.process_local_owner_delete_once(
                    independent, media[1], media[2], media[3], now_utc=NOW
                )
            )
        return unlink(self, directory, path, descriptor)

    monkeypatch.setattr(cleanup._PosixFS, "unlink", inspect_other_store)
    assert run(media).outcome == "completed"
    assert attempts[0].outcome == "idle"
    control = media[0].path.parent / ".local-owner-media-cleanup-control"
    assert len(list(control.rglob("*.lock"))) == 2
    assert not (media[2] / ".local-owner-media-cleanup-control").exists()


def test_outbox_storage_failure_recovers_existing_physical_journal_without_second_unlink(
    media, monkeypatch
):
    from billiards_ai.edge.route_assistance import owner_device_delivery as delivery

    store, _, _, _, overlay, raw, accepted, _ = media
    enqueue = delivery.enqueue_owner_completion_in

    def storage_full(*args, **kwargs):
        error = sqlite3.OperationalError("synthetic queue full")
        error.sqlite_errorcode = sqlite3.SQLITE_FULL
        raise error

    monkeypatch.setattr(delivery, "enqueue_owner_completion_in", storage_full)
    first = run(media)
    assert first.outcome == "retry", first
    state = store.get_owner_delete(accepted.clip_id)
    assert state.status == "retry" and state.receipt_json is None
    assert not overlay.exists() and not (raw / "segment-1.mjpeg").exists()
    progress = json.loads(state.progress_json)
    assert all(obj["state"] == "deleted" for obj in progress["objects"])
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT * FROM field_owner_completion_outbox").fetchall() == []

    def no_second_unlink(*args, **kwargs):
        raise AssertionError("terminal original journal must recover without second unlink")

    monkeypatch.setattr(delivery, "enqueue_owner_completion_in", enqueue)
    monkeypatch.setattr(os, "unlink", no_second_unlink)
    recovered = run(media, now_utc=NOW + timedelta(seconds=3))
    assert recovered.outcome == "completed", recovered
    completed = store.get_owner_delete(accepted.clip_id)
    assert completed.progress_json == state.progress_json
    item = delivery.OwnerDeletionReceiptOutbox(store, SCOPE).claim_due(NOW + timedelta(seconds=3))
    assert item.local_receipt_json == completed.receipt_json
    assert item.local_receipt_sha256 == completed.receipt_sha256
