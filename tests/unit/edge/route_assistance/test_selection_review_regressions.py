"""Independent S1 findings exercised through real synthetic encoders and journals."""

from datetime import UTC, datetime, timedelta

import pytest

from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _legacy_two_shot_scene
from tests.unit.lab.test_mainline1_shot_worker import _record_ended_usb_session, _worker


def _finish(records):
    manifest = records.get_postgame_manifest("legacy-session")
    records.complete_postgame_review("legacy-artifact", "legacy-session", manifest.manifest_sha256)
    return manifest


def _drain(records, root, temporary):
    now = datetime.now(UTC) + timedelta(seconds=1)
    for attempt in range(32):
        result = process_recording_discard_once(
            records,
            temporary / "jobs/analysis-jobs.sqlite3",
            root,
            temporary / "jobs",
            ring_root=None,
            session_ids=("legacy-session",),
            now_utc=now + timedelta(seconds=attempt),
        )
        assert result.outcome in {"pending", "completed"}, result.error_code
        if result.outcome == "completed":
            return result
    pytest.fail("bounded recording cleanup did not finish")


def test_review_ir01_scheduled_before_first_encode_cleans_without_output_directory(tmp_path):
    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path, production_layout=True)
    assert len(encoder.schedule("legacy-session")) == 2
    assert not (root / "postgame/legacy-session").exists()
    raw = [path for source in sources for path in source.media_root.glob("*.mjpgseg")]
    assert len(raw) == 2
    _finish(records)
    _drain(records, root, tmp_path)
    assert all(not path.exists() for path in raw)
    assert records.list_saved_media_outbox() == ()


@pytest.mark.parametrize("flow", ["partial", "finalized"])
def test_review_ir02_canonical_retry_erases_prior_interrupted_video(tmp_path, monkeypatch, flow):
    from billiards_ai.lab import mainline1_shot_local_video as publisher

    class ProcessStopped(BaseException):
        pass

    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path, production_layout=True)
    semantic = _worker(records, jobs, root / "postgame/legacy-session")
    held = jobs.claim_for_session(
        "delay-second",
        "legacy-session",
        NOW,
        expected_job_ids=(sources[1].job_id,),
    )
    jobs.fail_claim(
        held, "synthetic delayed dependency", now_utc=NOW, retry_delay=timedelta(hours=2)
    )
    render = publisher.render_canonical_overlay

    def interrupted(**kwargs):
        kwargs["output"].write_bytes(b"interrupted synthetic canonical encoding")
        raise ProcessStopped()

    monkeypatch.setattr(publisher, "render_canonical_overlay", interrupted)
    with pytest.raises(ProcessStopped):
        semantic.process_one("interrupted", "legacy-session", NOW)
    (orphan,) = (root / "postgame/legacy-session").glob(".pending-shot-*/overlay.mp4")
    monkeypatch.setattr(publisher, "render_canonical_overlay", render)
    restarted = semantic.process_one("restarted", "legacy-session", NOW + timedelta(hours=1))
    assert restarted is not None and restarted.job_id == sources[0].job_id
    if flow == "finalized":
        assert (
            semantic.process_one("second", "legacy-session", NOW + timedelta(hours=3)) is not None
        )
        assert records.is_postgame_finalized("legacy-session")
    _finish(records)
    if flow == "partial":
        _drain(records, root, tmp_path)
    else:
        outcomes = [
            process_local_discard_once(
                records,
                tmp_path / "jobs/analysis-jobs.sqlite3",
                root,
                tmp_path / "jobs",
                now_utc=datetime.now(UTC) + timedelta(seconds=1),
                session_ids=("legacy-session",),
            ).outcome
            for _ in range(3)
        ]
        assert outcomes == ["completed", "completed", "idle"]
    assert not orphan.exists()
    assert not tuple(root.rglob("*.mp4"))


def test_review_ir03_later_selected_canonical_publishes_after_earlier_discard(tmp_path):
    records, jobs, sources, root, encoder = _legacy_two_shot_scene(tmp_path, production_layout=True)
    manifest = records.get_postgame_manifest("legacy-session")
    receipt = records.save_postgame_clip(
        request_id="keep-second-canonical",
        session_id="legacy-session",
        shot_id=sources[1].shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    _finish(records)
    result = _worker(records, jobs, root / "postgame/legacy-session").process_one(
        "selected-analysis",
        "legacy-session",
        NOW,
    )
    assert result is not None and result.job_id == sources[1].job_id
    assert result.artifact_path.joinpath("overlay.mp4").is_file()
    selected = records.list_saved_media_outbox()
    assert len(selected) == 1 and selected[0].shot_id == sources[1].shot_id
    assert selected[0].owner_user_id == "user-a"
    assert (
        records.save_postgame_clip(
            request_id=receipt.request_id,
            session_id="legacy-session",
            shot_id=sources[1].shot_id,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == receipt
    )
    assert not records.is_postgame_finalized("legacy-session")
    _drain(records, root, tmp_path)
    assert selected[0].local_path.is_file()
    assert not tuple(sources[0].media_root.glob("*.mjpgseg"))


def test_review_ir04_sixty_five_recorded_shots_are_reclaimed_in_bounded_batches(tmp_path):
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "legacy-session"
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs/analysis-jobs.sqlite3")
    for index in range(65):
        _record_ended_usb_session(
            records,
            jobs,
            tmp_path,
            session_id="legacy-session",
            shot_id=f"shot-{index:03}",
            job_id=f"job-{index:03}",
            epoch="epoch-1",
            artifact_id="legacy-artifact",
            sequence=index + 1,
            end_session=index == 64,
            media_root=tmp_path / "jobs/media/legacy-session" / f"job-{index:03}",
        )
    root = tmp_path / "artifacts"
    root.mkdir()
    records.bind_artifact_path("legacy-artifact", root)
    encoder = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=root / "postgame/legacy-session"
    )
    for _index in range(65):
        assert encoder.process_one("original", "legacy-session", NOW) is not None
    assert len(tuple(root.rglob("original.mp4"))) == 65
    _finish(records)
    _drain(records, root, tmp_path)
    assert not tuple(root.rglob("*.mp4"))
    assert not tuple((tmp_path / "jobs/media").rglob("*.mjpgseg"))
