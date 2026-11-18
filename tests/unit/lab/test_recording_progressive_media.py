"""Capture-bound encoding remains authoritative across finalization and cleanup."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    project_recording_playback,
    validate_recording_capture_media_authority,
)
from billiards_ai.lab.mainline1_session_recording import ContinuousSessionRecorder
from billiards_ai.lab.mainline1_shot_jobs import ShotMediaEncodeJobV1, SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _ordinary_scene


def _scene(root):
    records, session, artifacts = _ordinary_scene(
        root,
        production_layout=True,
        capture_stopped=True,
    )
    jobs = SQLiteShotAnalysisJobStore(root / "jobs/analysis-jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=artifacts / "postgame" / session,
    )
    return records, session, artifacts, jobs, worker


def _finalize(records, session):
    capture = records.get_recording_capture_end(session)
    raw = ContinuousSessionRecorder(
        root=Path(capture["raw_manifest_path"]).parent,
        session_id=session,
        expected_fps=60,
        read_only=True,
    )
    records.prepare_recording_end(
        session,
        "epoch",
        capture["cutoff"],
        raw.finalized_manifest_path,
        raw.finalized_segment_receipts,
        background=True,
    )


def test_capture_task_and_playback_survive_final_inventory_without_duplicate_jobs(tmp_path):
    records, session, _, jobs, worker = _scene(tmp_path)
    manifest = records.get_postgame_manifest(session)
    assert records.get_recording_end_manifest(session) is None
    (original,) = worker.schedule(session)
    assert original.source.kind == "recording_capture"
    output = worker.process_one("encoder", session, NOW)
    assert output is not None
    (view,) = project_recording_playback(records, jobs, manifest)
    assert view.playback_clip is not None
    reopened = SQLiteShotAnalysisJobStore(tmp_path / "jobs/analysis-jobs.sqlite3")
    assert reopened.snapshot(original.job_id).job == original
    _finalize(records, session)
    assert worker.schedule(session) == (original,)
    (final,) = project_recording_playback(records, jobs, records.get_postgame_manifest(session))
    assert final.playback_clip.clip_id == view.playback_clip.clip_id


@pytest.mark.parametrize("mutation", ["capture", "epoch", "range", "evidence", "reference"])
def test_capture_media_authority_rejects_foreign_or_changed_per_shot_lineage(tmp_path, mutation):
    records, session, _, _, worker = _scene(tmp_path)
    (job,) = worker.schedule(session)
    source = job.source
    epoch, end, media_root = job.source_epoch_id, job.end_frame_id, job.media_root
    if mutation == "capture":
        source = replace(source, capture_end_sha256="b" * 64)
    elif mutation == "epoch":
        epoch = "foreign-epoch"
    elif mutation == "range":
        end = 6
    elif mutation == "evidence":
        source = replace(source, shot_evidence_sha256="b" * 64)
    else:
        media_root = tmp_path / "foreign-reference"
        source = replace(source, reference_path=media_root / "interval.json")
    changed = ShotMediaEncodeJobV1.create(
        session_id=session,
        shot_id=job.shot_id,
        source_epoch_id=epoch,
        start_frame_id=job.start_frame_id,
        end_frame_id=end,
        media_root=media_root,
        media_sha256=job.media_sha256,
        source=source,
    )
    with pytest.raises(RecordingPlaybackError, match="authority|evidence|root"):
        validate_recording_capture_media_authority(records, changed)


def test_final_detection_cannot_complete_selection_while_encoding_is_pending(tmp_path):
    records, session, _, jobs, worker = _scene(tmp_path)
    worker.schedule(session)
    _finalize(records, session)
    manifest = records.get_postgame_manifest(session)
    with pytest.raises(ValueError, match="processing|ready|encoding"):
        records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    assert worker.process_one("encoder", session, NOW) is not None
    records.complete_postgame_review(
        "recording",
        session,
        manifest.manifest_sha256,
        media_jobs=jobs,
    )
    assert records.active_postgame_review("recording") is None


def test_capture_bound_playback_is_erased_only_after_final_processing_and_selection(tmp_path):
    records, session, artifacts, jobs, worker = _scene(tmp_path)
    output = worker.process_one("encoder", session, NOW)
    assert output is not None
    _finalize(records, session)
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review(
        "recording",
        session,
        manifest.manifest_sha256,
        media_jobs=jobs,
    )
    result = process_recording_discard_once(
        records,
        tmp_path / "jobs/analysis-jobs.sqlite3",
        artifacts,
        tmp_path / "jobs",
        ring_root=tmp_path / "ring",
        session_ids=(session,),
        now_utc=datetime.now(UTC),
    )
    assert result.outcome == "completed", result
    assert not (output.artifact_path / "original.mp4").exists()
    assert not tuple((tmp_path / "ring").rglob("*.mjpgseg"))
