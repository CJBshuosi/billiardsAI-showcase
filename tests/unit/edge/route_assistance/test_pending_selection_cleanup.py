"""Finishing review cancels unselected work without waiting for semantic success."""

import pytest

from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once
from tests.unit.edge.route_assistance.test_recording_selection import _state
from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _legacy_two_shot_scene


def test_complete_pending_review_preserves_selected_original_and_erases_unselected_source(tmp_path):
    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path, production_layout=True)
    first, second = sources
    preserved = {
        path: path.read_bytes()
        for path in first.media_root.iterdir()
        if path.suffix in {".mjpeg", ".mjpgseg"}
    }
    discarded = tuple(
        path for path in second.media_root.iterdir() if path.suffix in {".mjpeg", ".mjpgseg"}
    )
    assert preserved and discarded
    manifest = records.get_postgame_manifest("legacy-session")
    original = records.save_postgame_clip(
        request_id="keep-first",
        session_id="legacy-session",
        shot_id=first.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert original.status == "SAVE_PENDING"
    state = _state(records, "legacy-session", root, jobs, artifact_id="legacy-artifact")
    review = state.postgame_review_payload(kiosk=True)
    assert review["can_complete"] is True
    with session_media_lock(root, "legacy-session", exclusive=False):
        state.mutate_postgame_review(
            dict(session_id="legacy-session", postgame_manifest_sha256=manifest.manifest_sha256),
            review["kiosk_nonce"],
            complete=True,
        )
    assert not records.is_postgame_discarded("legacy-session", first.shot_id)
    assert records.is_postgame_discarded("legacy-session", second.shot_id)
    records = SQLiteFieldRecordStore(records.path)
    records.complete_postgame_review("legacy-artifact", "legacy-session", manifest.manifest_sha256)
    assert encoder.process_one("finish-selected-only", "legacy-session", NOW) is not None
    selected = records.list_saved_media_outbox()
    assert len(selected) == 1 and selected[0].shot_id == first.shot_id
    before = selected[0].local_path.read_bytes()
    result = process_recording_discard_once(
        records,
        tmp_path / "jobs/analysis-jobs.sqlite3",
        root,
        tmp_path / "jobs",
        ring_root=None,
        session_ids=("legacy-session",),
    )
    assert result.outcome == "completed", result.error_code
    assert selected[0].local_path.read_bytes() == before
    assert all(path.read_bytes() == original for path, original in preserved.items())
    assert all(not path.exists() for path in discarded)
    assert encoder.process_one("no-unsaved-retry", "legacy-session", NOW) is None
    assert records.list_saved_media_outbox() == selected


@pytest.mark.parametrize("kind", ["original", "canonical"])
def test_finish_erases_published_media_when_job_completion_was_lost(tmp_path, monkeypatch, kind):
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path, production_layout=True)
    monkeypatch.setattr(jobs, "complete_claim", lambda *args, **kwargs: False)
    worker = (
        encoder if kind == "original" else _worker(records, jobs, root / "postgame/legacy-session")
    )
    assert worker.process_one("interrupted-after-publish", "legacy-session", NOW) is None
    media = tuple(
        (root / "postgame/legacy-session").glob(
            "*/" + ("original.mp4" if kind == "original" else "overlay.mp4")
        )
    )
    assert len(media) == 1 and media[0].stat().st_size > 0
    manifest = records.get_postgame_manifest("legacy-session")
    records.complete_postgame_review("legacy-artifact", "legacy-session", manifest.manifest_sha256)
    result = process_recording_discard_once(
        records,
        tmp_path / "jobs/analysis-jobs.sqlite3",
        root,
        tmp_path / "jobs",
        ring_root=None,
        session_ids=("legacy-session",),
    )
    assert result.outcome == "completed", result.error_code
    assert not media[0].exists()


@pytest.mark.parametrize("video_bytes", [b"", b"interrupted synthetic encoder output"])
def test_finish_erases_interrupted_canonical_encoder_output(tmp_path, monkeypatch, video_bytes):
    from billiards_ai.lab import mainline1_shot_local_video as publisher
    from tests.unit.lab.test_mainline1_shot_worker import _worker

    class ProcessStopped(BaseException):
        pass

    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path, production_layout=True)

    def stop_encoder(**kwargs):
        kwargs["output"].write_bytes(video_bytes)
        raise ProcessStopped()

    monkeypatch.setattr(publisher, "render_canonical_overlay", stop_encoder)
    with pytest.raises(ProcessStopped):
        _worker(records, jobs, root / "postgame/legacy-session").process_one(
            "interrupted-encoder",
            "legacy-session",
            NOW,
        )
    overlays = tuple((root / "postgame/legacy-session").glob(".pending-*/overlay.mp4"))
    assert len(overlays) == 1
    manifest = records.get_postgame_manifest("legacy-session")
    records.complete_postgame_review("legacy-artifact", "legacy-session", manifest.manifest_sha256)
    result = process_recording_discard_once(
        records,
        tmp_path / "jobs/analysis-jobs.sqlite3",
        root,
        tmp_path / "jobs",
        ring_root=None,
        session_ids=("legacy-session",),
    )
    assert result.outcome == "completed", result.error_code
    assert not overlays[0].exists()
