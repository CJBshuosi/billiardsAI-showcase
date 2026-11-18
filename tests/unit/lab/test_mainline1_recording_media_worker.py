from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import RecordedMediaV1, RecordingBoundaryV1
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    project_recording_playback,
)
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
    SessionRecordingEvidenceStore,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJobStatus,
    ShotMediaEncodeJobSnapshot,
    SQLiteShotAnalysisJobStore,
    materialize_recorded_interval_reference,
)
from billiards_ai.lab.mainline1_shot_worker import (
    Mainline1RecordingMediaWorker,
    MediaEncodingResourceGate,
)
from tests.unit.edge.route_assistance.test_recording_evidence import _store
from tests.unit.lab.test_mainline1_shot_worker import _record_ended_usb_session, _worker

NOW = datetime(2026, 11, 9, 12, tzinfo=UTC)


def _jpeg(frame_id: int) -> bytes:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    cv2.circle(image, (6 + frame_id * 2, 12), 3, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _ordinary_scene(
    tmp_path: Path,
    *,
    production_layout=False,
    recording_frames=6,
    segment_seconds=1.0,
    capture_stopped=False,
    live=False,
):
    records, session_id = _store(tmp_path)
    original_root = (tmp_path / "artifact").resolve()
    original_root.mkdir()
    records.bind_artifact_path("recording", original_root)
    boundary = RecordingBoundaryV1("shot-1", "epoch", 1, 2, 16, 33, True)
    records.record_recording_start(session_id, "shot-1", "epoch", 1, 16)
    records.record_recording_boundary(session_id, boundary)
    recorder = ContinuousSessionRecorder(
        root=(tmp_path / "ring" / "sessions" / session_id / "epoch" / "session-recording")
        if production_layout
        else tmp_path / "raw",
        session_id=session_id,
        expected_fps=60,
        config=SessionRecordingConfig(
            reserve_bytes=1,
            minimum_write_mbps=0,
            segment_duration_seconds=segment_seconds,
        ),
    )
    evidence_store = SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id=session_id,
        pre_roll_frames=1,
        post_roll_frames=3,
        maximum_shot_frames=60,
    )
    for frame_id in range(recording_frames):
        recorder.append(
            CaptureFrame(
                "epoch",
                frame_id,
                frame_id * 16_666_667,
                32,
                24,
                _jpeg(frame_id),
            )
        )
    interval = evidence_store.freeze_window(
        shot_id="shot-1",
        source_epoch_id="epoch",
        start_frame_id=1,
        end_frame_id=5,
    )
    reference_root = (
        tmp_path / "jobs" / "media" / session_id / "recording-ref-1"
        if production_layout
        else tmp_path / "reference"
    )
    materialize_recorded_interval_reference(interval, reference_root, recording_identity=True)
    reference = reference_root / "interval.json"
    records.record_recording_media(
        session_id,
        RecordedMediaV1(
            boundary,
            "RECORDED",
            "recording-ref-1",
            str(reference),
            hashlib.sha256(reference.read_bytes()).hexdigest(),
            interval.media_sha256,
            interval.start_frame_id,
            interval.end_frame_id,
        ),
    )
    recorder.close()
    if live:
        return records, session_id, original_root
    if capture_stopped:
        records.record_recording_capture_end(
            session_id,
            "epoch",
            {
                "frame_id": recording_frames - 1,
                "captured_at_ns": (recording_frames - 1) * 16_666_667,
            },
            recorder.finalized_manifest_path,
            first_frame_id=0,
            width=32,
            height=24,
            last_frame_sha256=hashlib.sha256(_jpeg(recording_frames - 1)).hexdigest(),
        )
        records.end_session(session_id)
        return records, session_id, original_root
    records.prepare_recording_end(
        session_id,
        "epoch",
        {
            "frame_id": recording_frames - 1,
            "captured_at_ns": (recording_frames - 1) * 16_666_667,
        },
        recorder.finalized_manifest_path,
        recorder.finalized_segment_receipts,
    )
    records.end_session(session_id)
    return records, session_id, original_root


def _legacy_scene(tmp_path: Path):
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "legacy-session"
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    source = _record_ended_usb_session(
        records,
        jobs,
        tmp_path,
        session_id="legacy-session",
        shot_id="shot-1",
        job_id="analysis-job-1",
        epoch="epoch-1",
        artifact_id="legacy-artifact",
    )
    original_root = (tmp_path / "artifact").resolve()
    original_root.mkdir()
    records.bind_artifact_path("legacy-artifact", original_root)
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / source.session_id,
    )
    return records, jobs, source, original_root, worker


def _legacy_two_shot_scene(tmp_path: Path, *, production_layout=False):
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "legacy-session"
    )
    jobs = SQLiteShotAnalysisJobStore(
        tmp_path / "jobs" / "analysis-jobs.sqlite3"
        if production_layout
        else tmp_path / "jobs.sqlite3"
    )
    first = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "first",
        session_id="legacy-session",
        shot_id="shot-1",
        job_id="analysis-job-1",
        epoch="epoch-1",
        artifact_id="legacy-artifact",
        sequence=1,
        end_session=False,
        media_root=tmp_path / "jobs/media/legacy-session/analysis-job-1"
        if production_layout
        else None,
    )
    second = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "second",
        session_id="legacy-session",
        shot_id="shot-2",
        job_id="analysis-job-2",
        epoch="epoch-1",
        artifact_id="legacy-artifact",
        sequence=2,
        media_root=tmp_path / "jobs/media/legacy-session/analysis-job-2"
        if production_layout
        else None,
    )
    original_root = (tmp_path / "artifact").resolve()
    original_root.mkdir()
    records.bind_artifact_path("legacy-artifact", original_root)
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / first.session_id,
    )
    return records, jobs, (first, second), original_root, worker


def _run_canonical_worker(records, jobs, source, original_root):
    worker = _worker(records, jobs, original_root / "postgame" / source.session_id)
    result = worker.process_one("canonical-worker", source.session_id, NOW)
    assert result is not None and result.job_id == source.job_id
    return records.list_postgame_clips(source.session_id)[0]


def test_ordinary_media_worker_publishes_without_analysis_or_canonical_availability(
    tmp_path: Path,
) -> None:
    records, session_id, original_root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / session_id,
    )
    frozen = records.get_recording_end_manifest(session_id)

    scheduled = worker.schedule(session_id)
    result = worker.process_one("media-worker", session_id, NOW)

    assert len(scheduled) == 1
    assert result is not None
    snapshot = jobs.snapshot(scheduled[0].job_id)
    assert isinstance(snapshot, ShotMediaEncodeJobSnapshot)
    assert snapshot.status is ShotAnalysisJobStatus.AVAILABLE
    assert snapshot.artifact_id == result.artifact_id
    assert jobs.list_session_jobs(session_id) == ()
    assert records.list_postgame_clips(session_id) == ()
    assert not records.is_postgame_finalized(session_id)
    assert records.get_recording_end_manifest(session_id) == frozen


def test_legacy_pending_analysis_coexists_with_available_raw_media(tmp_path: Path) -> None:
    records, jobs, source, _, worker = _legacy_scene(tmp_path)

    result = worker.process_one("media-worker", source.session_id, NOW)

    assert result is not None
    assert jobs.snapshot(source.job_id).status is ShotAnalysisJobStatus.PENDING
    media = jobs.list_session_media_jobs(source.session_id)
    assert len(media) == 1
    assert jobs.snapshot(media[0].job_id).status is ShotAnalysisJobStatus.AVAILABLE
    projection = project_recording_playback(
        records, jobs, records.get_postgame_manifest(source.session_id)
    )
    assert projection[0].playback_clip is not None
    assert projection[0].playback_clip.job_id == media[0].job_id
    assert projection[0].canonical_clip.clip_id is None
    assert records.list_postgame_clips(source.session_id)[0].clip_id is None
    assert not records.is_postgame_finalized(source.session_id)


def test_legacy_canonical_first_skips_media_task(tmp_path: Path) -> None:
    records, jobs, source, original_root, worker = _legacy_scene(tmp_path)
    canonical = _run_canonical_worker(records, jobs, source, original_root)

    assert worker.schedule(source.session_id) == ()
    assert jobs.list_session_media_jobs(source.session_id) == ()
    projection = project_recording_playback(
        records, jobs, records.get_postgame_manifest(source.session_id)
    )
    assert projection[0].encoding_status == "CANONICAL_AVAILABLE"
    assert projection[0].playback_clip.clip_id == canonical.clip_id
    assert records.list_postgame_clips(source.session_id)[0].clip_id == canonical.clip_id


def test_scheduled_media_identity_stays_stable_when_canonical_arrives_later(
    tmp_path: Path,
) -> None:
    records, jobs, source, original_root, worker = _legacy_scene(tmp_path)
    scheduled = worker.schedule(source.session_id)
    canonical = _run_canonical_worker(records, jobs, source, original_root)

    result = worker.process_one("media-worker", source.session_id, NOW)

    assert result is not None and result.job_id == scheduled[0].job_id
    assert jobs.snapshot(scheduled[0].job_id).status is ShotAnalysisJobStatus.AVAILABLE
    after = records.list_postgame_clips(source.session_id)[0]
    assert (after.clip_id, after.job_id) == (canonical.clip_id, source.job_id)
    projection = project_recording_playback(
        records, jobs, records.get_postgame_manifest(source.session_id)
    )
    assert projection[0].playback_clip.job_id == scheduled[0].job_id
    assert projection[0].playback_clip.clip_id != canonical.clip_id
    assert projection[0].canonical_clip.clip_id == canonical.clip_id


def test_legacy_failed_analysis_does_not_block_media(tmp_path: Path) -> None:
    records, jobs, source, _, worker = _legacy_scene(tmp_path)
    claim = jobs.claim_for_session(
        "analysis-worker",
        source.session_id,
        NOW,
        expected_job_ids=(source.job_id,),
    )
    assert claim is not None
    jobs.fail_claim(
        claim,
        "semantic dependency failed",
        now_utc=NOW,
        retry_delay=timedelta(hours=1),
    )

    result = worker.process_one("media-worker", source.session_id, NOW)

    assert result is not None
    assert jobs.snapshot(source.job_id).status is ShotAnalysisJobStatus.FAILED_RETRYING
    media = jobs.list_session_media_jobs(source.session_id)
    assert jobs.snapshot(media[0].job_id).status is ShotAnalysisJobStatus.AVAILABLE
    projection = project_recording_playback(
        records, jobs, records.get_postgame_manifest(source.session_id)
    )
    assert projection[0].playback_clip is not None
    assert projection[0].encoding_status == "AVAILABLE"


def test_partial_analysis_projection_keeps_both_independent_raw_clips_playable(
    tmp_path: Path,
) -> None:
    from billiards_ai.lab.field_assistance import server as field_server

    records, jobs, sources, original_root, worker = _legacy_two_shot_scene(tmp_path)
    scheduled = worker.schedule("legacy-session")
    assert len(scheduled) == 2
    assert worker.process_one("media-worker", "legacy-session", NOW) is not None
    assert worker.process_one("media-worker", "legacy-session", NOW) is not None
    _run_canonical_worker(records, jobs, sources[0], original_root)
    state = field_server._ServerState(
        artifact_path=original_root,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=records,
        session_id="legacy-session",
        artifact_id="legacy-artifact",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        media_job_store=jobs,
    )

    payload = state.postgame_review_payload()

    assert payload["analysis_status"] == "PARTIAL"
    assert payload["available_count"] == 1
    assert payload["playback"]["available_count"] == 2
    assert all(clip["media_url"] for clip in payload["clips"])
    assert jobs.snapshot(sources[1].job_id).status is ShotAnalysisJobStatus.PENDING
    assert records.list_postgame_clips("legacy-session")[1].clip_id is None


def test_corrupt_playback_is_a_failed_row_without_hiding_healthy_siblings(tmp_path):
    records, jobs, sources, _, worker = _legacy_two_shot_scene(tmp_path)
    first = worker.process_one("media-1", "legacy-session", NOW)
    second = worker.process_one("media-2", "legacy-session", NOW)
    assert first is not None and second is not None
    before = jobs.snapshot(first.job_id)
    with (first.artifact_path / "original.mp4").open("ab") as stream:
        stream.write(b"tampered")
    rows = project_recording_playback(
        records, jobs, records.get_postgame_manifest("legacy-session")
    )
    assert [row.shot_id for row in rows] == [source.shot_id for source in sources]
    assert rows[0].encoding_status == "FAILED"
    assert rows[0].terminal_failure
    assert rows[0].error_code == "MEDIA_ARTIFACT_INVALID"
    assert rows[0].playback_clip is None
    assert rows[1].encoding_status == "AVAILABLE"
    assert rows[1].playback_clip is not None
    assert jobs.snapshot(first.job_id) == before  # A read-only failure projection.


@pytest.mark.parametrize("order", ["raw-first", "scheduled-first"])
def test_pending_save_keeps_first_available_selected_media(tmp_path, order):
    records, jobs, source, original_root, media_worker = _legacy_scene(tmp_path)
    manifest = records.get_postgame_manifest(source.session_id)
    receipt = records.save_postgame_clip(
        request_id="save-original-canonical",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert receipt.status == "SAVE_PENDING"
    media_job = media_worker.schedule(source.session_id)[0]
    if order == "raw-first":
        assert media_worker.process_one("media", source.session_id, NOW) is not None
        before = project_recording_playback(records, jobs, manifest)[0].playback_clip
        assert before is not None and before.job_id == media_job.job_id
        selected_before_canonical = records.list_saved_media_outbox()
        assert len(selected_before_canonical) == 1
        assert selected_before_canonical[0].local_path == before.local_path
        assert records.list_postgame_clips(source.session_id)[0].save_status == "SAVED"
    canonical = _run_canonical_worker(records, jobs, source, original_root)
    if order == "scheduled-first":
        assert media_worker.process_one("media", source.session_id, NOW) is not None
    row = project_recording_playback(records, jobs, manifest)[0]
    assert row.playback_clip.job_id == media_job.job_id
    assert row.playback_clip.clip_id != canonical.clip_id
    if order == "raw-first":
        assert row.playback_clip.clip_id == before.clip_id
    assert row.canonical_clip.save_status == "SAVED"
    outbox = records.list_saved_media_outbox()
    assert len(outbox) == 1
    assert outbox[0].job_id == source.job_id
    assert outbox[0].artifact_id == canonical.artifact_id
    assert outbox[0].clip_id == canonical.clip_id
    assert outbox[0].local_path == canonical.local_path
    assert outbox[0].content_sha256 == hashlib.sha256(canonical.local_path.read_bytes()).hexdigest()
    if order == "raw-first":
        assert outbox == selected_before_canonical
    assert (
        _worker(records, jobs, original_root / "postgame" / source.session_id).process_one(
            "analysis-restart", source.session_id, NOW + timedelta(seconds=1)
        )
        is None
    )
    assert (
        media_worker.process_one("media-restart", source.session_id, NOW + timedelta(seconds=1))
        is None
    )
    assert records.list_saved_media_outbox() == outbox
    with sqlite3.connect(records.path) as connection:
        finalization = json.loads(
            connection.execute(
                "SELECT manifest_json FROM field_postgame_finalizations WHERE session_id=?",
                (source.session_id,),
            ).fetchone()[0]
        )
    assert {clip["job_id"] for clip in finalization["clips"]} == {source.job_id}


def test_rule_abstention_receipt_and_match_remain_unchanged_while_media_plays(
    tmp_path: Path,
) -> None:
    from billiards_ai.edge.route_assistance.eight_ball_rules import (
        PROFILE_ID,
        RuleOutcomeKind,
        decide_eight_ball,
    )
    from tests.unit.edge.route_assistance.test_rule_persistence import (
        _facts,
        _match,
    )

    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-rules"
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    session = records.open_or_resume_session(
        artifact_id="legacy-artifact", rule_profile_id=PROFILE_ID
    )
    records.save_match(session.session_id, _match(), expected_revision=0)
    source = _record_ended_usb_session(
        records,
        jobs,
        tmp_path,
        session_id="session-rules",
        shot_id="shot-0001",
        job_id="analysis-job-1",
        epoch="epoch-rules",
        artifact_id="legacy-artifact",
        end_session=False,
    )
    facts = replace(_facts(), abstention_reasons=("first_contact_abstained",))
    decision = decide_eight_ball(_match(), facts)
    assert decision.kind is RuleOutcomeKind.ABSTAIN
    receipt = records.apply_automatic_rule_decision(facts, decision, elapsed_ms=15.0)
    records.end_session(source.session_id)
    original_root = (tmp_path / "artifact").resolve()
    original_root.mkdir()
    records.bind_artifact_path("legacy-artifact", original_root)
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / source.session_id,
    )
    manifest = records.get_postgame_manifest(source.session_id)
    records.save_postgame_clip(
        request_id="save-abstained-original",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    result = worker.process_one("media-worker", source.session_id, NOW)

    assert result is not None
    assert receipt.status == "abstained"
    assert records.get_rule_decision(source.session_id, source.shot_id).status == "abstained"
    assert records.load_session(source.session_id).match == _match()
    assert records.list_command_ledger(source.session_id) == ()
    projection = project_recording_playback(
        records, jobs, records.get_postgame_manifest(source.session_id)
    )
    assert projection[0].playback_clip is not None
    saved = records.list_saved_media_outbox()
    assert len(saved) == 1 and saved[0].owner_user_id == "user-red"
    assert records.get_rule_decision(source.session_id, source.shot_id) == receipt


def test_legacy_scheduler_rejects_cross_session_source_payload(tmp_path: Path) -> None:
    _, jobs, source, _, worker = _legacy_scene(tmp_path)
    payload = source.identity()
    payload["session_id"] = "foreign-session"
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? WHERE job_id=?",
            (raw, hashlib.sha256(raw.encode()).hexdigest(), source.job_id),
        )

    with pytest.raises(ValueError, match="lineage"):
        worker.schedule(source.session_id)
    assert jobs.list_session_media_jobs(source.session_id) == ()


def test_live_priority_pause_releases_claim_without_consuming_retry(
    tmp_path: Path, monkeypatch
) -> None:
    records, session_id, original_root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")

    class Gate(MediaEncodingResourceGate):
        allowed = True

        def may_run_media_encoding(self, session_id):
            return self.allowed

    gate = Gate()
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / session_id,
        resource_gate=gate,
    )
    scheduled = worker.schedule(session_id)
    import billiards_ai.lab.mainline1_shot_worker as worker_module

    def pause_during_publish(job, root, *, authority_check, continue_check):
        gate.allowed = False
        continue_check()

    monkeypatch.setattr(worker_module, "publish_recording_playback", pause_during_publish)

    assert worker.process_one("media-worker", session_id, NOW) is None
    snapshot = jobs.snapshot(scheduled[0].job_id)
    assert isinstance(snapshot, ShotMediaEncodeJobSnapshot)
    assert snapshot.status is ShotAnalysisJobStatus.PENDING
    assert snapshot.attempt_count == 0
    assert not snapshot.terminal_failure


def test_slow_active_encoder_renews_lease_before_another_worker_can_reclaim(
    tmp_path: Path, monkeypatch
) -> None:
    records, session_id, original_root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    monotonic_value = [0.0]
    utc_value = [NOW]
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / session_id,
        monotonic=lambda: monotonic_value[0],
        lease_now_utc=lambda: utc_value[0],
    )
    scheduled = worker.schedule(session_id)
    import billiards_ai.lab.mainline1_shot_worker as worker_module

    def slow_publish(job, root, *, authority_check, continue_check):
        monotonic_value[0] = 31.0
        utc_value[0] = NOW.replace(second=31)
        continue_check()
        assert (
            jobs.claim_media_for_session(
                "competing-worker",
                session_id,
                NOW.replace(minute=2, second=1),
                expected_job_ids=(job.job_id,),
            )
            is None
        )
        path = root / "simulated-artifact"
        path.mkdir(parents=True)
        return SimpleNamespace(artifact_id="simulated-artifact", path=path)

    monkeypatch.setattr(worker_module, "publish_recording_playback", slow_publish)

    result = worker.process_one("slow-worker", session_id, NOW)
    assert result is not None
    snapshot = jobs.snapshot(scheduled[0].job_id)
    assert isinstance(snapshot, ShotMediaEncodeJobSnapshot)
    assert snapshot.status is ShotAnalysisJobStatus.AVAILABLE
    assert snapshot.attempt_count == 1


def test_long_encoding_failure_backoff_starts_at_failure_time(tmp_path: Path, monkeypatch) -> None:
    records, session_id, original_root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    failed_at = NOW.replace(minute=5)
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / session_id,
        lease_now_utc=lambda: failed_at,
    )
    scheduled = worker.schedule(session_id)
    import billiards_ai.lab.mainline1_shot_worker as worker_module

    def fail_after_long_encode(*args, **kwargs):
        raise RecordingPlaybackError("MEDIA_H264_ENCODER_UNAVAILABLE", "codec unavailable")

    monkeypatch.setattr(worker_module, "publish_recording_playback", fail_after_long_encode)
    assert worker.process_one("media-worker", session_id, NOW) is None
    assert (
        jobs.claim_media_for_session(
            "early-retry",
            session_id,
            failed_at.replace(second=29),
            expected_job_ids=(scheduled[0].job_id,),
        )
        is None
    )
    assert (
        jobs.claim_media_for_session(
            "due-retry",
            session_id,
            failed_at.replace(second=31),
            expected_job_ids=(scheduled[0].job_id,),
        )
        is not None
    )


def test_live_priority_change_before_claim_is_a_clean_idle_return(tmp_path: Path) -> None:
    records, session_id, original_root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")

    class Gate(MediaEncodingResourceGate):
        calls = 0

        def may_run_media_encoding(self, session_id):
            self.calls += 1
            return self.calls == 1

    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=original_root / "postgame" / session_id,
        resource_gate=Gate(),
    )

    assert worker.process_one("media-worker", session_id, NOW) is None
    media = jobs.list_session_media_jobs(session_id)
    assert len(media) == 1
    snapshot = jobs.snapshot(media[0].job_id)
    assert isinstance(snapshot, ShotMediaEncodeJobSnapshot)
    assert snapshot.status is ShotAnalysisJobStatus.PENDING
    assert snapshot.attempt_count == 0


def test_same_shot_discard_terminals_scheduled_media_and_blocks_projection(
    tmp_path: Path, monkeypatch
) -> None:
    records, jobs, source, _, worker = _legacy_scene(tmp_path)
    scheduled = worker.schedule(source.session_id)
    monkeypatch.setattr(
        records,
        "_is_postgame_discarded_in",
        lambda connection, session_id, shot_id: (
            (session_id, shot_id) == (source.session_id, source.shot_id)
        ),
    )

    assert worker.process_one("media-worker", source.session_id, NOW) is None
    snapshot = jobs.snapshot(scheduled[0].job_id)
    assert snapshot.terminal_failure is True
    assert snapshot.error_code == "MEDIA_SOURCE_REVOKED"
    assert worker.process_one("media-worker", source.session_id, NOW + timedelta(days=1)) is None
    projection = project_recording_playback(
        records, jobs, records.get_postgame_manifest(source.session_id)
    )
    assert projection[0].encoding_status == "DELETED"
    assert projection[0].playback_clip is None


def test_committed_ordinary_interval_encodes_while_session_open_without_end_receipt(tmp_path):
    records, session, root = _ordinary_scene(tmp_path, live=True)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=root / "postgame" / session
    )
    scheduled = worker.schedule(session)
    assert len(scheduled) == 1
    assert scheduled[0].source.kind == "recording_interval"
    assert records.get_recording_capture_end(session) is None
    assert records.get_postgame_manifest(session) is None
    result = worker.process_one("live-media", session, NOW)
    assert result is not None
    assert jobs.snapshot(result.job_id).status is ShotAnalysisJobStatus.AVAILABLE
    assert records.load_session(session).status.value == "open"
    with pytest.raises(ValueError, match="ended session"):
        records.list_postgame_clips(session)
    # End does not create a second source identity for already generated pixels.
    raw = tmp_path / "raw/session-recording.json"
    manifest = json.loads(raw.read_text())
    records.prepare_recording_end(
        session,
        "epoch",
        {"frame_id": 5, "captured_at_ns": 5 * 16_666_667},
        raw,
        [
            dict(
                name=row["name"],
                sha256=row["sha256"],
                **ContinuousSessionRecorder.segment_file_identity(raw.parent / row["name"]),
            )
            for row in manifest["segments"]
        ],
    )
    records.end_session(session)
    assert [job.job_id for job in worker.schedule(session)] == [result.job_id]
    assert worker.process_one("after-end", session, NOW) is None


@pytest.mark.parametrize("action", ["resume", "discard", "stop"])
def test_recording_pressure_keeps_one_claim_and_observes_cross_connection_fence(
    tmp_path, monkeypatch, action
):
    records, session, root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    entered = threading.Event()

    class Gate(MediaEncodingResourceGate):
        cooperative_pause = True
        allowed = True
        stopped = False

        def may_run_media_encoding(self, session_id):
            return self.allowed

        def cancellation_requested(self):
            return self.stopped

    gate = Gate()
    worker = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=root / "postgame" / session, resource_gate=gate
    )
    scheduled = worker.schedule(session)
    import billiards_ai.lab.mainline1_shot_worker as module

    original = module.publish_recording_playback
    calls, writers, written = [], [], []

    class PressureWriter:
        def __init__(self, *args):
            self.writer = cv2.VideoWriter(*args)
            writers.append(self)

        def isOpened(self):
            return self.writer.isOpened()

        def write(self, image):
            self.writer.write(image)
            written.append(1)
            if len(written) == 1:
                gate.allowed = False
                entered.set()

        def release(self):
            self.writer.release()

    def paused_publish(job, artifact_root, *, authority_check, continue_check):
        calls.append(job.job_id)
        return original(
            job,
            artifact_root,
            authority_check=authority_check,
            continue_check=continue_check,
            writer_factory=PressureWriter,
        )

    monkeypatch.setattr(module, "publish_recording_playback", paused_publish)
    results, errors = [], []

    def run():
        try:
            results.append(worker.process_one("same-claim", session, NOW))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(2)
        assert jobs.snapshot(scheduled[0].job_id).status is ShotAnalysisJobStatus.RUNNING
        if action == "resume":
            gate.allowed = True
        elif action == "stop":
            gate.stopped = True
        else:
            # Independent connection commits a negative revocation fault-control.
            # No positive product event, ownership or media result is injected.
            with sqlite3.connect(records.path) as other:
                other.execute(
                    "INSERT INTO field_local_media_discards "
                    "(session_id,shot_id,postgame_manifest_sha256,job_id,artifact_id,clip_id,"
                    "local_path,content_sha256,byte_size,content_type,status,attempts,"
                    "next_attempt_at_utc) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,'pending',0,?)",
                    (
                        session,
                        "shot-1",
                        "f" * 64,
                        scheduled[0].job_id,
                        "revoked",
                        "revoked",
                        "revoked",
                        "f" * 64,
                        1,
                        "video/mp4",
                        NOW.isoformat(),
                    ),
                )
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert calls == [scheduled[0].job_id] and len(writers) == 1
        snapshot = jobs.snapshot(scheduled[0].job_id)
        if action == "resume":
            assert results[0] is not None and snapshot.status is ShotAnalysisJobStatus.AVAILABLE
            assert snapshot.attempt_count == 1
            assert len(written) == scheduled[0].end_frame_id - scheduled[0].start_frame_id + 1
        elif action == "discard":
            assert results == [None] and snapshot.terminal_failure
            assert snapshot.error_code == "MEDIA_SOURCE_REVOKED"
        else:
            assert results == [None] and snapshot.status is ShotAnalysisJobStatus.PENDING
            assert snapshot.attempt_count == 0
    finally:
        gate.stopped = True
        gate.allowed = True
        thread.join(3)


def test_delete_committed_after_last_continue_check_cannot_complete_media_claim(
    tmp_path, monkeypatch
):
    records, session, root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=root / "postgame" / session
    )
    job = worker.schedule(session)[0]
    renew = jobs.renew_media_claim
    fenced = []

    def commit_fence(claim, now):
        result = renew(claim, now)
        if not fenced:
            with sqlite3.connect(records.path) as other:
                other.execute(
                    "INSERT INTO field_local_media_discards "
                    "(session_id,shot_id,postgame_manifest_sha256,job_id,artifact_id,clip_id,"
                    "local_path,content_sha256,byte_size,content_type,status,attempts,"
                    "next_attempt_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,'pending',0,?)",
                    (
                        session,
                        "shot-1",
                        "f" * 64,
                        job.job_id,
                        "revoked",
                        "revoked",
                        "revoked",
                        "f" * 64,
                        1,
                        "video/mp4",
                        NOW.isoformat(),
                    ),
                )
            fenced.append(True)
        return result

    monkeypatch.setattr(jobs, "renew_media_claim", commit_fence)
    assert worker.process_one("encoder", session, NOW) is None
    assert fenced == [True]
    snapshot = jobs.snapshot(job.job_id)
    assert snapshot.status is not ShotAnalysisJobStatus.AVAILABLE
    assert snapshot.terminal_failure and snapshot.error_code == "MEDIA_SOURCE_REVOKED"


@pytest.mark.parametrize("operation", ["payload", "clip"])
def test_media_hashing_does_not_hold_match_lock(tmp_path, monkeypatch, operation):
    from billiards_ai.lab.field_assistance import server as field_server

    records, jobs, _, root, worker = _legacy_two_shot_scene(tmp_path)
    worker.schedule("legacy-session")
    assert worker.process_one("media", "legacy-session", NOW) is not None
    manifest = records.get_postgame_manifest("legacy-session")
    clip = project_recording_playback(records, jobs, manifest)[0].playback_clip
    state = field_server._ServerState(
        artifact_path=root,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=records,
        session_id="legacy-session",
        artifact_id="legacy-artifact",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        media_job_store=jobs,
    )
    original = field_server.project_recording_playback

    def projection(*a, **kw):
        acquired = []

        def attempt():
            locked = state.match_lock.acquire(timeout=0.3)
            acquired.append(locked)
            if locked:
                state.match_lock.release()

        t = threading.Thread(target=attempt)
        t.start()
        t.join(1)
        assert acquired == [True], "media verification held match_lock"
        return original(*a, **kw)

    monkeypatch.setattr(field_server, "project_recording_playback", projection)
    if operation == "payload":
        assert state.postgame_review_payload()["playback"]["available_count"] == 1
    else:
        assert state.review_clip(clip.clip_id, manifest.manifest_sha256) == clip


def test_review_projection_busy_lease_is_a_controlled_conflict(tmp_path):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
    from billiards_ai.lab.field_assistance import server as field_server

    records, jobs, _, root, _ = _legacy_two_shot_scene(tmp_path)
    state = field_server._ServerState(
        artifact_path=root,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=records,
        session_id="legacy-session",
        artifact_id="legacy-artifact",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        media_job_store=jobs,
    )
    with session_media_lock(root, "legacy-session", exclusive=True, timeout_seconds=0):
        with pytest.raises(ValueError, match="busy"):
            state.postgame_review_payload()


def test_expired_final_claim_cannot_publish_even_without_a_replacement(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from datetime import timedelta

    records, session, root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    clock = [NOW]
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=root / "postgame" / session,
        lease_now_utc=lambda: clock[0],
    )
    job = worker.schedule(session)[0]
    guard = records.media_publication_guard

    @contextmanager
    def pause_after_renew(job):
        with guard(job) as allowed:
            clock[0] += timedelta(seconds=121)
            yield allowed

    monkeypatch.setattr(records, "media_publication_guard", pause_after_renew)
    assert worker.process_one("paused", session, NOW) is None
    assert jobs.snapshot(job.job_id).status is ShotAnalysisJobStatus.RUNNING


def test_target_clip_lookup_never_hashes_healthy_sibling_media(tmp_path, monkeypatch):
    import billiards_ai.lab.mainline1_recording_media as media_module
    from billiards_ai.lab.field_assistance import server as field_server

    records, jobs, _, root, worker = _legacy_two_shot_scene(tmp_path)
    assert worker.process_one("first", "legacy-session", NOW) is not None
    assert worker.process_one("second", "legacy-session", NOW) is not None
    manifest = records.get_postgame_manifest("legacy-session")
    clips = project_recording_playback(records, jobs, manifest)
    target = clips[0].playback_clip
    sibling = clips[1].playback_clip.local_path
    state = field_server._ServerState(
        artifact_path=root,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=records,
        session_id="legacy-session",
        artifact_id="legacy-artifact",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        media_job_store=jobs,
    )
    original = media_module._hash_media_file

    def hash_target(path, **kwargs):
        assert path != sibling, "single-clip GET hashed unrelated sibling"
        return original(path, **kwargs)

    monkeypatch.setattr(media_module, "_hash_media_file", hash_target)
    assert state.review_clip(target.clip_id, manifest.manifest_sha256) == target
