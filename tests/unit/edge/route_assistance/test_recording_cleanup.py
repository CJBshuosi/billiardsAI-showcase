"""Unselected ordinary bytes are actually erased through the background executor."""

import errno
import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _ordinary_scene


@pytest.mark.parametrize("born_live", [False, True])
def test_completed_ordinary_review_erases_raw_and_playback_in_original_context(tmp_path, born_live):
    records, session, root = _ordinary_scene(tmp_path, production_layout=True, live=born_live)
    jobs_root = tmp_path / "jobs"
    jobs = SQLiteShotAnalysisJobStore(jobs_root / "analysis-jobs.sqlite3")
    encoder = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=root / "postgame" / session,
    )
    output = encoder.process_one("encoder", session, NOW)
    assert output is not None
    output_root = root / "postgame" / session / output.artifact_id
    media = output_root / "original.mp4"
    assert media.is_file()
    raw = tuple((tmp_path / "ring").rglob("*.mjpgseg"))
    assert raw
    metadata = output_root / "manifest.json"
    unrelated = root / "do-not-touch.mp4"
    unrelated.write_bytes(b"unrelated preserved file")
    if born_live:
        from billiards_ai.lab.mainline1_session_recording import ContinuousSessionRecorder

        raw_manifest = (
            tmp_path / "ring/sessions" / session / "epoch/session-recording/session-recording.json"
        )
        raw_inventory = json.loads(raw_manifest.read_text())
        records.prepare_recording_end(
            session,
            "epoch",
            {"frame_id": 5, "captured_at_ns": 5 * 16_666_667},
            raw_manifest,
            [
                dict(
                    name=row["name"],
                    sha256=row["sha256"],
                    **ContinuousSessionRecorder.segment_file_identity(
                        raw_manifest.parent / row["name"]
                    ),
                )
                for row in raw_inventory["segments"]
            ],
        )
        records.end_session(session)
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    context = SimpleNamespace(
        spec_record=None,
        jobs_db=jobs_root / "analysis-jobs.sqlite3",
        artifact_root=root,
        jobs_root=jobs_root,
        ring_root=tmp_path / "ring",
    )
    worker = _BackgroundLocalMediaCleanupWorker(
        store=records,
        jobs_db=jobs_root / "analysis-jobs.sqlite3",
        artifact_root=root,
        jobs_root=jobs_root,
        resource_resolver=SimpleNamespace(resolve=lambda sid, purpose: context),
    )
    assert worker.run_once() is True
    assert not media.exists()
    assert all(not path.exists() for path in raw)
    assert metadata.is_file()
    assert unrelated.read_bytes() == b"unrelated preserved file"
    assert records.list_saved_media_outbox() == ()


def test_recovery_rejects_forged_inventory_even_with_recomputed_checksum(tmp_path, monkeypatch):
    from billiards_ai.edge.route_assistance import recording_cleanup

    records, session, root = _ordinary_scene(tmp_path, production_layout=True)
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    stop = threading.Event()
    original_targets = recording_cleanup._targets

    def freeze_then_cancel(*args, **kwargs):
        result = original_targets(*args, **kwargs)
        stop.set()
        return result

    now = datetime.now(UTC)
    arguments = dict(
        store=records,
        jobs_db=tmp_path / "jobs/analysis-jobs.sqlite3",
        artifact_root=root,
        jobs_root=tmp_path / "jobs",
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )
    with monkeypatch.context() as controlled:
        controlled.setattr(recording_cleanup, "_targets", freeze_then_cancel)
        first = recording_cleanup.process_recording_discard_once(
            **arguments,
            now_utc=now,
            cancelled=stop.is_set,
        )
    assert first.outcome == "retry"
    unrelated = root / "another-saved-clip.mp4"
    payload = b"private sibling bytes must survive forged cleanup metadata"
    unrelated.write_bytes(payload)
    with records._connect() as db:
        row = db.execute(
            "SELECT * FROM field_recording_cleanup WHERE session_id=?", (session,)
        ).fetchone()
        assert row["progress_json"] == "{}"
        targets = json.loads(row["inventory_json"])
        targets[0] = [
            "artifacts",
            str(unrelated),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        ]
        forged = json.dumps(targets, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        db.execute(
            "UPDATE field_recording_cleanup SET inventory_json=?,inventory_sha256=? "
            "WHERE session_id=?",
            (forged, hashlib.sha256(forged.encode()).hexdigest(), session),
        )
    result = recording_cleanup.process_recording_discard_once(
        **arguments,
        now_utc=now + timedelta(seconds=2),
    )
    assert result.outcome == "blocked"
    assert unrelated.read_bytes() == payload
    assert tuple((tmp_path / "ring").rglob("*.mjpgseg"))


@pytest.mark.parametrize("fault_point", ["after_detach", "after_unlink"])
def test_recording_erase_recovers_original_journal_after_process_interruption(
    tmp_path, fault_point
):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once

    records, session, root = _ordinary_scene(tmp_path, production_layout=True)
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    now = datetime.now(UTC)
    arguments = dict(
        jobs_db=tmp_path / "jobs/analysis-jobs.sqlite3",
        artifact_root=root,
        jobs_root=tmp_path / "jobs",
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )

    def crash(point, obj):
        if point == fault_point:
            raise OSError(errno.EIO, "controlled storage interruption")

    first = process_recording_discard_once(records, **arguments, now_utc=now, fault_hook=crash)
    assert first.outcome == "retry"
    second = process_recording_discard_once(
        SQLiteFieldRecordStore(records.path),
        **arguments,
        now_utc=now + timedelta(seconds=2),
    )
    assert second.outcome == "completed"
    assert not tuple((tmp_path / "ring").rglob("*.mjpgseg"))
    assert not tuple((tmp_path / "ring").rglob("*.detached"))
    assert records.get_recording_end_manifest(session) is not None


def test_recording_erase_yields_after_sixteen_objects_and_resumes_after_restart(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once

    records, session, root = _ordinary_scene(
        tmp_path,
        production_layout=True,
        recording_frames=40,
        segment_seconds=1 / 60,
    )
    raw = tuple((tmp_path / "ring").rglob("*.mjpgseg"))
    assert len(raw) == 40
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    arguments = dict(
        jobs_db=tmp_path / "jobs/analysis-jobs.sqlite3",
        artifact_root=root,
        jobs_root=tmp_path / "jobs",
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )
    first = process_recording_discard_once(records, **arguments)
    assert first.outcome == "pending"
    assert sum(not path.exists() for path in raw) == 16
    reopened = SQLiteFieldRecordStore(records.path)
    second = process_recording_discard_once(reopened, **arguments)
    assert second.outcome == "pending"
    assert sum(not path.exists() for path in raw) == 32
    third = process_recording_discard_once(reopened, **arguments)
    assert third.outcome == "completed"
    assert all(not path.exists() for path in raw)


def test_recording_cleanup_recovers_unlinked_partial_encoder_output(tmp_path):
    from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once

    records, session, root = _ordinary_scene(tmp_path, production_layout=True)
    jobs_root = tmp_path / "jobs"
    jobs_db = jobs_root / "analysis-jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(jobs_db)
    encoder = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=root / "postgame" / session,
    )
    job = encoder.schedule(session)[0]
    pending = (
        root / "postgame" / session / (f".pending-recording-playback-{job.job_id}-" + "a" * 32)
    )
    pending.mkdir(parents=True)
    temporary = pending / "original.mp4"
    temporary.write_bytes(b"owned partial encoder output")
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    now = datetime.now(UTC)

    def crash(point, obj):
        if point == "after_unlink" and obj["path"] == str(temporary):
            raise OSError(errno.EIO, "interrupted unlink checkpoint")

    arguments = dict(
        store=records,
        jobs_db=jobs_db,
        artifact_root=root,
        jobs_root=jobs_root,
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )
    first = process_recording_discard_once(**arguments, now_utc=now, fault_hook=crash)
    assert first.outcome == "retry"
    second = process_recording_discard_once(**arguments, now_utc=now + timedelta(seconds=2))
    assert second.outcome == "completed", second.error_code
    assert not temporary.exists()
    assert not tuple((tmp_path / "ring").rglob("*.mjpgseg"))


@pytest.mark.parametrize("location", ["source", "staged"])
def test_completed_batch_reappearance_blocks_before_next_batch(tmp_path, location):
    from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once

    records, session, root = _ordinary_scene(
        tmp_path,
        production_layout=True,
        recording_frames=40,
        segment_seconds=1 / 60,
    )
    raw = {path: path.read_bytes() for path in (tmp_path / "ring").rglob("*.mjpgseg")}
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    arguments = dict(
        store=records,
        jobs_db=tmp_path / "jobs/analysis-jobs.sqlite3",
        artifact_root=root,
        jobs_root=tmp_path / "jobs",
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )
    assert process_recording_discard_once(**arguments).outcome == "pending"
    removed = next(path for path in raw if not path.exists())
    if location == "source":
        replacement = removed
    else:
        stage = next((tmp_path / "ring/.recording-media-cleanup").iterdir())
        replacement = stage / "000.detached"
    replacement.write_bytes(raw[removed])
    remaining = {path: path.read_bytes() for path in raw if path.exists()}
    result = process_recording_discard_once(**arguments)
    assert result.outcome == "blocked"
    assert replacement.read_bytes() == raw[removed]
    assert all(path.read_bytes() == content for path, content in remaining.items())


@pytest.mark.parametrize(
    "progress",
    [
        [],
        None,
        {"version": 1, "cursor": 0, "reclaimed_bytes": -1, "batch": None},
        {"version": 1, "cursor": 0, "reclaimed_bytes": 0, "batch": []},
    ],
)
def test_malformed_selection_journal_blocks_before_erasure(tmp_path, progress):
    from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once

    records, session, root = _ordinary_scene(tmp_path, production_layout=True)
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    raw = {path: path.read_bytes() for path in (tmp_path / "ring").rglob("*.mjpgseg")}
    with records._connect() as db:
        db.execute("UPDATE field_recording_cleanup SET progress_json=?", (json.dumps(progress),))
    result = process_recording_discard_once(
        records,
        tmp_path / "jobs/analysis-jobs.sqlite3",
        root,
        tmp_path / "jobs",
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )
    assert result.outcome == "blocked"
    assert all(path.read_bytes() == data for path, data in raw.items())


def test_corrupted_typed_playback_metadata_produces_durable_blocked_result(tmp_path):
    from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once

    records, session, root = _ordinary_scene(tmp_path, production_layout=True)
    jobs_root = tmp_path / "jobs"
    jobs_db = jobs_root / "analysis-jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(jobs_db)
    encoder = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=root / "postgame" / session,
    )
    output = encoder.process_one("original", session, NOW)
    metadata = output.artifact_path / "manifest.json"
    value = json.loads(metadata.read_bytes())
    value["output"]["byte_size"] += 1
    metadata.write_text(json.dumps(value))
    manifest = records.get_postgame_manifest(session)
    records.complete_postgame_review("recording", session, manifest.manifest_sha256)
    result = process_recording_discard_once(
        records,
        jobs_db,
        root,
        jobs_root,
        ring_root=tmp_path / "ring",
        session_ids=(session,),
    )
    assert result.outcome == "blocked"
    assert (output.artifact_path / "original.mp4").is_file()
    assert tuple((tmp_path / "ring").rglob("*.mjpgseg"))
    with records._connect() as db:
        assert db.execute("SELECT status FROM field_recording_cleanup").fetchone()[0] == "blocked"
