"""Explicit selection completion fences ordinary original media independently of encoding."""

import threading

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.unit.edge.route_assistance.test_recording_end import _prepared
from tests.unit.lab.test_mainline1_recording_media_worker import _ordinary_scene


def _state(records, session, root, jobs, *, artifact_id="recording"):
    from billiards_ai.edge.route_assistance.control import MatchControlSnapshot
    from billiards_ai.lab.field_assistance.server import _ServerState

    return _ServerState(
        artifact_path=root,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=records,
        session_id=session,
        artifact_id=artifact_id,
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        media_job_store=jobs,
    )


def test_empty_recording_selection_completes_and_exact_retry_survives_restart(tmp_path):
    records, session, _ = _prepared(tmp_path, empty=True)
    records.end_session(session)
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    assert records.active_postgame_review("recording") is None
    reopened = SQLiteFieldRecordStore(records.path)
    reopened.complete_postgame_review("recording", session, manifest.manifest_sha256)
    assert reopened.active_postgame_review("recording") is None


def test_complete_unencoded_unknown_recording_revokes_read_and_late_publication(tmp_path):
    records, session, _ = _ordinary_scene(tmp_path)
    manifest = records.get_postgame_manifest(session)
    original_owner = records.get_shot_ownership(session, "shot-1")
    assert original_owner.evidence_status == "unknown"
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    assert records.active_postgame_review("recording") is None
    assert records.is_postgame_discarded(session, "shot-1")
    reopened = SQLiteFieldRecordStore(records.path)
    assert reopened.is_postgame_discarded(session, "shot-1")
    assert reopened.get_shot_ownership(session, "shot-1") == original_owner
    assert reopened.list_saved_media_outbox() == ()


@pytest.mark.parametrize("change", ["artifact", "manifest"])
def test_invalid_recording_completion_does_not_revoke_media(tmp_path, change):
    records, session, _ = _ordinary_scene(tmp_path)
    manifest = records.get_postgame_manifest(session)
    with pytest.raises(ValueError):
        records.complete_postgame_review(
            "foreign" if change == "artifact" else "recording",
            session,
            "f" * 64 if change == "manifest" else manifest.manifest_sha256,
        )
    assert records.active_postgame_review("recording").session_id == session
    assert not records.is_postgame_discarded(session, "shot-1")


def test_unknown_save_reports_ownership_and_offers_explicit_completion(tmp_path):
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore

    records, session, root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    state = _state(records, session, root, jobs)
    review = state.postgame_review_payload(kiosk=True)
    assert review.get("kiosk_nonce") == state._kiosk_nonce
    assert review["can_complete"] is True
    assert review["clips"][0]["can_save"] is False
    assert review["clips"][0]["ownership_unavailable_reason"] == "SHOT_OWNERSHIP_UNAVAILABLE"
    with pytest.raises(ValueError, match="ownership"):
        records.save_postgame_clip(
            request_id="save-unknown",
            session_id=session,
            shot_id="shot-1",
            postgame_manifest_sha256=review["postgame_manifest_sha256"],
        )
    assert records.list_saved_media_outbox() == ()


def test_completed_selection_terminals_existing_encoder_and_does_not_reschedule(tmp_path):
    from datetime import UTC, datetime

    from billiards_ai.lab.mainline1_recording_media import project_recording_playback
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker

    records, session, root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=root / "postgame" / session,
    )
    scheduled = worker.schedule(session)
    assert len(scheduled) == 1
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    assert worker.process_one("late-encoder", session, datetime.now(UTC)) is None
    snapshot = jobs.snapshot(scheduled[0].job_id)
    assert snapshot.terminal_failure is True
    assert snapshot.error_code == "MEDIA_SOURCE_REVOKED"
    assert project_recording_playback(records, jobs, manifest)[0].playback_clip is None
    assert not list(root.rglob("*.mp4"))


def test_completion_fences_new_reads_while_existing_reader_still_holds_pin(tmp_path):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore

    records, session, root = _ordinary_scene(tmp_path)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    state = _state(records, session, root, jobs)
    review = state.postgame_review_payload(kiosk=True)
    with session_media_lock(root, session, exclusive=False):
        result = state.mutate_postgame_review(
            {"session_id": session, "postgame_manifest_sha256": review["postgame_manifest_sha256"]},
            state._kiosk_nonce,
            complete=True,
        )
        assert result["review_session_id"] is None
        assert records.is_postgame_discarded(session, "shot-1")
        assert (tmp_path / "raw" / "session-recording.json").is_file()
