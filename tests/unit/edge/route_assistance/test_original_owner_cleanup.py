"""Owner deletion erases selected original bytes without requiring analysis success."""

import errno
from dataclasses import replace
from datetime import timedelta

import pytest

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_owner_delete_once
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionBinding
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import resource_spec
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    BIRTH,
    NOW,
    SCOPE,
    SESSION,
    task,
)
from tests.unit.lab.test_mainline1_shot_worker import _record_ended_usb_session


def _scene(tmp_path, *, second_shot=False):
    root, jobs_root = tmp_path / "artifacts", tmp_path / "jobs"
    root.mkdir()
    resource = resource_spec(tmp_path, root, jobs_root)
    records = SQLiteFieldRecordStore(tmp_path / "field.sqlite3", now_utc=lambda: NOW)
    request = records.reserve_managed_open(
        SCOPE,
        artifact_id=resource.artifact_id,
        rule_profile_id=None,
        resource_spec=resource,
    )
    records.record_managed_binding(
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
    session = records.materialize_managed_session(SCOPE, request.open_request_id)
    records.save_match(
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
    records.bind_artifact_path(resource.artifact_id, root)
    records.register_managed_capture_epoch(SCOPE, request.open_request_id, "epoch-1")
    jobs_db = jobs_root / "analysis-jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(jobs_db)
    source = _record_ended_usb_session(
        records,
        jobs,
        tmp_path,
        session_id=SESSION,
        shot_id="shot-0001",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id=resource.artifact_id,
        media_root=jobs_root / "media" / SESSION / "job-1",
        resource=resource,
        end_session=not second_shot,
    )
    if second_shot:
        _record_ended_usb_session(
            records,
            jobs,
            tmp_path / "second",
            session_id=SESSION,
            shot_id="shot-0002",
            job_id="job-2",
            epoch="epoch-1",
            artifact_id=resource.artifact_id,
            media_root=jobs_root / "media" / SESSION / "job-2",
            resource=resource,
            sequence=2,
        )
    manifest = records.get_postgame_manifest(SESSION)
    records.save_postgame_clip(
        request_id="save-1",
        session_id=SESSION,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    encoder = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=root / "postgame" / SESSION,
    )
    output = encoder.process_one("original", SESSION, NOW)
    assert output is not None
    return records, jobs, source, root, jobs_root, jobs_db, output


def test_owner_erases_selected_original_and_raw_while_analysis_is_pending(tmp_path):
    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)
    selected = records.list_saved_media_outbox()[0]
    assert selected.owner_user_id == "owner-red"
    accepted = records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    assert records.owner_delete_for_shot(SESSION, source.shot_id) is not None
    assert records.claim_due_saved_media(NOW, managed_scope=SCOPE) is None
    result = process_local_owner_delete_once(
        records,
        jobs_db,
        root,
        jobs_root,
        now_utc=NOW,
        session_ids=(SESSION,),
    )
    assert result.outcome == "completed", result.error_code
    assert not selected.local_path.exists()
    assert not tuple(source.media_root.glob("*.mjpgseg"))
    assert not tuple(source.media_root.glob("*.mjpeg"))
    assert (output.artifact_path / "manifest.json").is_file()
    assert records.get_owner_delete(accepted.clip_id).status == "completed"


def test_owner_delete_also_erases_late_canonical_sibling(tmp_path):
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)
    canonical = _worker(
        records,
        jobs,
        root / "postgame" / SESSION,
        resource=source,
    ).process_one("semantic-worker", SESSION, NOW)
    assert canonical is not None
    overlay = canonical.artifact_path / "overlay.mp4"
    assert overlay.is_file()
    selected = records.list_saved_media_outbox()[0]
    assert selected.local_path == output.artifact_path / "original.mp4"
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    result = process_local_owner_delete_once(
        records,
        jobs_db,
        root,
        jobs_root,
        now_utc=NOW,
        session_ids=(SESSION,),
    )
    assert result.outcome == "completed", result.error_code
    assert not selected.local_path.exists()
    assert not overlay.exists()


def test_owner_fence_prevents_new_analysis_claims_and_retries(tmp_path):
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)
    before = jobs.snapshot(source.job_id)
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    semantic = _worker(records, jobs, root / "postgame" / SESSION, resource=source)
    assert semantic.process_one("after-owner-fence", SESSION, NOW) is None
    assert jobs.snapshot(source.job_id) == before


def test_owner_delete_erases_canonical_published_before_lost_completion(tmp_path, monkeypatch):
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)
    monkeypatch.setattr(jobs, "complete_claim", lambda *args: False)
    semantic = _worker(records, jobs, root / "postgame" / SESSION, resource=source)
    assert semantic.process_one("interrupted-after-publish", SESSION, NOW) is None
    overlays = tuple((root / "postgame" / SESSION).glob("*/overlay.mp4"))
    assert len(overlays) == 1
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    result = process_local_owner_delete_once(
        records,
        jobs_db,
        root,
        jobs_root,
        now_utc=NOW,
        session_ids=(SESSION,),
    )
    assert result.outcome == "completed", result.error_code
    assert not overlays[0].exists()


def test_deleted_original_does_not_block_other_shot_analysis_finalization(tmp_path):
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path, second_shot=True)
    semantic = _worker(records, jobs, root / "postgame" / SESSION, resource=source)
    assert semantic.process_one("first-analysis", SESSION, NOW) is not None
    assert not records.is_postgame_finalized(SESSION)
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    erased = process_local_owner_delete_once(
        records,
        jobs_db,
        root,
        jobs_root,
        now_utc=NOW,
        session_ids=(SESSION,),
    )
    assert erased.outcome == "completed", erased.error_code
    assert not output.artifact_path.joinpath("original.mp4").exists()
    second = semantic.process_one("second-analysis", SESSION, NOW)
    assert second is not None
    assert second.artifact_path.joinpath("overlay.mp4").is_file()
    assert records.is_postgame_finalized(SESSION)


@pytest.mark.parametrize("video_bytes", [b"", b"partial synthetic encoder output"])
def test_owner_erases_partial_canonical_and_recovers_unlink_checkpoint(
    tmp_path, monkeypatch, video_bytes
):
    from billiards_ai.lab import mainline1_shot_local_video as publisher
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    class ProcessStopped(BaseException):
        pass

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)

    def stop_encoder(**kwargs):
        kwargs["output"].write_bytes(video_bytes)
        raise ProcessStopped()

    monkeypatch.setattr(publisher, "render_canonical_overlay", stop_encoder)
    with pytest.raises(ProcessStopped):
        _worker(records, jobs, root / "postgame" / SESSION, resource=source).process_one(
            "partial-canonical",
            SESSION,
            NOW,
        )
    (overlay,) = (root / "postgame" / SESSION).glob(".pending-*/overlay.mp4")
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)

    def interrupted(point, obj):
        if point == "after_unlink" and obj["path"] == str(overlay):
            raise OSError(errno.EIO, "interrupted checkpoint")

    first = process_local_owner_delete_once(
        records,
        jobs_db,
        root,
        jobs_root,
        now_utc=NOW,
        session_ids=(SESSION,),
        fault_hook=interrupted,
    )
    assert first.outcome == "retry", first.error_code
    second = process_local_owner_delete_once(
        records,
        jobs_db,
        root,
        jobs_root,
        now_utc=NOW + timedelta(seconds=60),
        session_ids=(SESSION,),
    )
    assert second.outcome == "completed", second.error_code
    assert not overlay.exists() and not output.artifact_path.joinpath("original.mp4").exists()


@pytest.mark.parametrize("purpose_aware", [True, False])
def test_supervisor_does_not_launch_a_child_for_fenced_recording_work(tmp_path, purpose_aware):
    import sys

    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundPostgameAnalysisWorker

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    marker = tmp_path / "unexpected-worker-start.txt"

    def command(session_id, *, purpose):
        return [
            sys.executable,
            "-B",
            "-c",
            "from pathlib import Path; Path("
            + repr(str(marker))
            + ").write_text("
            + repr(purpose)
            + ")",
        ]

    factory = (
        command if purpose_aware else lambda session_id: command(session_id, purpose="analysis")
    )
    worker = _BackgroundPostgameAnalysisWorker(store=records, command_factory=factory)
    try:
        worker.run_once()
        if worker._active_process is not None:
            worker._active_process.wait(timeout=5)
        assert not marker.exists()
    finally:
        worker.stop()
