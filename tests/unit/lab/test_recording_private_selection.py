"""Trusted start ownership, rather than successful analysis, gates selected bytes."""

import hashlib
import threading
from datetime import UTC, datetime, timedelta

import pytest

from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJobStatus
from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _legacy_scene


def test_trusted_pending_selection_saves_original_when_analysis_fails(tmp_path):
    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    manifest = records.get_postgame_manifest(source.session_id)
    ownership = records.get_shot_ownership(source.session_id, source.shot_id)
    assert ownership.evidence_status == "bound_at_start"
    assert ownership.actor_user_id == "user-a"
    receipt = records.save_postgame_clip(
        request_id="save-trusted-original",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert receipt.status == "SAVE_PENDING"
    claim = jobs.claim_for_session(
        "analysis",
        source.session_id,
        NOW,
        expected_job_ids=(source.job_id,),
    )
    assert claim is not None
    jobs.fail_claim(
        claim, "semantic dependency unavailable", now_utc=NOW, retry_delay=timedelta(hours=1)
    )
    artifact = encoder.process_one("original-media", source.session_id, NOW)
    assert artifact is not None
    outbox = records.list_saved_media_outbox()
    assert len(outbox) == 1
    saved = outbox[0]
    assert saved.owner_user_id == "user-a"
    assert saved.ownership_receipt == ownership
    assert saved.job_id == source.job_id
    assert saved.local_path == artifact.artifact_path / "original.mp4"
    assert saved.content_sha256 == hashlib.sha256(saved.local_path.read_bytes()).hexdigest()
    assert jobs.snapshot(source.job_id).status is ShotAnalysisJobStatus.FAILED_RETRYING
    assert not records.is_postgame_finalized(source.session_id)
    assert (
        records.save_postgame_clip(
            request_id="save-trusted-original",
            session_id=source.session_id,
            shot_id=source.shot_id,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == receipt
    )
    assert (
        encoder.process_one("restart-media", source.session_id, NOW + timedelta(seconds=1)) is None
    )
    assert records.list_saved_media_outbox() == outbox


def test_selected_original_signed_upload_recovers_lost_ack_without_reupload(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.edge.route_assistance.sync import FieldSyncClient
    from scripts.algorithm.serve_mainline1_usb_field import _BackgroundSavedMediaSyncWorker
    from tests.integration.lab.field_assistance.test_postgame_screen_saved_clips import (
        _CloudContractTransport,
        _Credentials,
    )

    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    manifest = records.get_postgame_manifest(source.session_id)
    receipt = records.save_postgame_clip(
        request_id="lost-ack-original",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert encoder.process_one("original-only", source.session_id, NOW) is not None
    (selected,) = records.list_saved_media_outbox()
    media = selected.local_path.read_bytes()
    peer = _CloudContractTransport(shot_count=1)
    # This transport models an already admitted facts-only session; it checks
    # signatures and upload bytes but is not live cloud acceptance evidence.
    peer.session_published = True
    started = datetime.now(UTC)
    for attempt in range(2):
        clock = started + timedelta(minutes=attempt + 1)
        records = SQLiteFieldRecordStore(records.path, now_utc=lambda clock=clock: clock)
        sync = _BackgroundSavedMediaSyncWorker(
            store=records,
            client=FieldSyncClient(credentials=_Credentials(), transport=peer),
            trusted_postgame_root=root / "postgame",
            wake_event=threading.Event(),
            now_utc=lambda clock=clock: clock,
        )
        assert sync.run_once()
    assert (peer.create_count, peer.upload_count, peer.complete_count) == (2, 1, 1)
    assert peer.uploaded_bytes == media
    assert peer.saved_descriptor["artifact_id"] == selected.artifact_id
    assert peer.saved_descriptor["job_id"] == source.job_id
    assert peer.saved_descriptor["owner_user_id"] == "user-a"
    assert records.get_saved_media_outbox(selected.clip_id).status == "acknowledged"
    assert (
        records.save_postgame_clip(
            request_id=receipt.request_id,
            session_id=source.session_id,
            shot_id=source.shot_id,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == receipt
    )
    assert jobs.snapshot(source.job_id).status is ShotAnalysisJobStatus.PENDING
    assert not records.is_postgame_finalized(source.session_id)


def test_selected_original_availability_does_not_upgrade_failed_analysis(tmp_path):
    from tests.unit.edge.route_assistance.test_recording_selection import _state

    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    manifest = records.get_postgame_manifest(source.session_id)
    records.save_postgame_clip(
        request_id="honest-status",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    claim = jobs.claim_for_session(
        "analysis",
        source.session_id,
        NOW,
        expected_job_ids=(source.job_id,),
    )
    jobs.fail_claim(claim, "semantic failure", now_utc=NOW, retry_delay=timedelta(hours=1))
    assert encoder.process_one("original-media", source.session_id, NOW) is not None
    state = _state(records, source.session_id, root, jobs, artifact_id="legacy-artifact")
    review = state.postgame_review_payload(kiosk=True)
    assert review["available_count"] == 0
    assert review["analysis_status"] == "FAILED_RETRYING"
    assert review["playback"]["available_count"] == 1
    assert review["clips"][0]["save_status"] == "SAVED"


def test_kiosk_selection_after_original_ready_converges_without_another_encode(tmp_path):
    from tests.unit.edge.route_assistance.test_recording_selection import _state

    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    artifact = encoder.process_one("original-media", source.session_id, NOW)
    assert artifact is not None
    assert records.list_saved_media_outbox() == ()
    state = _state(records, source.session_id, root, jobs, artifact_id="legacy-artifact")
    review = state.postgame_review_payload(kiosk=True)
    payload = dict(
        request_id="late-screen-selection",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    receipt = state.mutate_postgame_review(payload, review["kiosk_nonce"], complete=False)
    saved = records.list_saved_media_outbox()
    assert len(saved) == 1
    assert saved[0].owner_user_id == "user-a"
    assert saved[0].local_path == artifact.artifact_path / "original.mp4"
    assert state.mutate_postgame_review(payload, review["kiosk_nonce"], complete=False) == receipt
    assert records.list_saved_media_outbox() == saved


@pytest.mark.parametrize("change", ["missing", "job_digest", "artifact_manifest"])
def test_original_saved_upload_fails_closed_when_typed_provenance_changes(
    tmp_path, change, monkeypatch
):
    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    monkeypatch.setattr(records, "_now_utc", lambda: NOW)
    manifest = records.get_postgame_manifest(source.session_id)
    records.save_postgame_clip(
        request_id="source-pinned-save",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert encoder.process_one("original-media", source.session_id, NOW) is not None
    saved = records.list_saved_media_outbox()
    assert len(saved) == 1
    with records._connect() as db:
        if change == "missing":
            db.execute("DELETE FROM field_original_clip_sources")
        elif change == "job_digest":
            db.execute("UPDATE field_original_clip_sources SET media_job_sha256=?", ("f" * 64,))
        else:
            db.execute(
                "UPDATE field_original_clip_sources SET artifact_manifest_sha256=?", ("f" * 64,)
            )
    with pytest.raises(ValueError, match="original|source|provenance"):
        records.get_saved_media_outbox(saved[0].clip_id)
    assert records.claim_due_saved_media(NOW + timedelta(days=1)) is None
    with records._connect() as db:
        row = db.execute("SELECT status,last_error_code FROM field_saved_media_outbox").fetchone()
        assert row["status"] == "blocked"
        assert row["last_error_code"] == "SYNC_PERSISTED_DESCRIPTOR_INVALID"


def test_late_canonical_analysis_does_not_replace_original_selection(tmp_path):
    from tests.unit.lab.test_mainline1_recording_media_worker import _run_canonical_worker

    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    manifest = records.get_postgame_manifest(source.session_id)
    first = records.save_postgame_clip(
        request_id="first-original",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert encoder.process_one("original-media", source.session_id, NOW) is not None
    saved = records.list_saved_media_outbox()
    assert len(saved) == 1
    canonical = _run_canonical_worker(records, jobs, source, root)
    assert canonical is not None
    assert records.is_postgame_finalized(source.session_id)
    assert records.list_saved_media_outbox() == saved
    assert (
        records.save_postgame_clip(
            request_id="first-original",
            session_id=source.session_id,
            shot_id=source.shot_id,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == first
    )
    second = records.save_postgame_clip(
        request_id="second-click",
        session_id=source.session_id,
        shot_id=source.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert second.status == "SAVED"
    assert second.clip_id == saved[0].clip_id
    assert records.list_saved_media_outbox() == saved


def test_saving_one_original_does_not_read_unrelated_shot_media(tmp_path, monkeypatch):
    from billiards_ai.lab import mainline1_recording_media as media
    from tests.unit.lab.test_mainline1_recording_media_worker import _legacy_two_shot_scene

    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path)
    assert encoder.process_one("first", "legacy-session", NOW) is not None
    assert encoder.process_one("second", "legacy-session", NOW) is not None
    manifest = records.get_postgame_manifest("legacy-session")
    records.save_postgame_clip(
        request_id="select-one",
        session_id="legacy-session",
        shot_id=sources[0].shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    verify = media.verify_recording_playback_artifact

    def selected_only(path, job, **kwargs):
        if job.shot_id != sources[0].shot_id:
            raise AssertionError("saving one shot read unrelated original media")
        return verify(path, job, **kwargs)

    monkeypatch.setattr(media, "verify_recording_playback_artifact", selected_only)
    media.publish_selected_original_clips(records, jobs, manifest)
    assert len(records.list_saved_media_outbox()) == 1
