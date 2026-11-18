from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing, RealtimeRingError
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    RecordingEndMediaSourceV1,
    ShotAnalysisClaim,
    ShotAnalysisJob,
    ShotAnalysisJobStatus,
    ShotMediaEncodeClaim,
    ShotMediaEncodeJobSnapshot,
    ShotMediaEncodeJobV1,
    SQLiteShotAnalysisJobStore,
    materialize_frozen_interval,
    quarantine_frozen_interval,
)

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)


def _jpeg(value: int) -> bytes:
    image = np.full((12, 16, 3), value % 251, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _frame(frame_id: int) -> CaptureFrame:
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=round(frame_id * 1_000_000_000 / 60),
        width=16,
        height=12,
        mjpeg=_jpeg(frame_id),
    )


def _populated_ring(root: Path) -> MjpegSegmentRing:
    ring = MjpegSegmentRing(
        root,
        expected_fps=4,
        pre_roll_frames=2,
        post_roll_frames=2,
        maximum_shot_frames=12,
        maximum_bytes=1024 * 1024,
    )
    for frame_id in range(8):
        ring.append(_frame(frame_id))
        if frame_id == 2:
            ring.begin_shot(frame_id)
        if frame_id == 5:
            ring.settle_shot(frame_id)
    return ring


@pytest.fixture
def shot_job(tmp_path: Path) -> ShotAnalysisJob:
    return ShotAnalysisJob(
        job_id="job-0001",
        session_id="session-0001",
        shot_id="shot-0001",
        source_epoch_id="epoch-1",
        start_frame_id=10,
        end_frame_id=20,
        media_root=tmp_path / "jobs" / "shot-0001",
        media_sha256="a" * 64,
        capture_setup_id="capture-setup-1",
        calibration_artifact_id="calibration-1",
        config_sha256="b" * 64,
        model_package_id="rfdetr-seg-small-1",
    )


@pytest.fixture
def media_job(tmp_path: Path) -> ShotMediaEncodeJobV1:
    media_root = tmp_path / "jobs" / "shot-media-0001"
    return ShotMediaEncodeJobV1.create(
        session_id="session-0001",
        shot_id="shot-media-0001",
        source_epoch_id="epoch-1",
        start_frame_id=10,
        end_frame_id=20,
        media_root=media_root,
        media_sha256="c" * 64,
        source=RecordingEndMediaSourceV1(
            kind="recording_end",
            recording_end_sha256="d" * 64,
            shot_evidence_sha256="e" * 64,
            reference_path=media_root / "interval.json",
            reference_sha256="f" * 64,
        ),
        max_attempts=2,
    )


def test_materialized_interval_survives_ring_release(tmp_path: Path) -> None:
    ring = _populated_ring(tmp_path / "ring")
    frozen = ring.freeze_shot()

    copied = materialize_frozen_interval(frozen, tmp_path / "jobs" / "shot-0001")
    ring.release_shot()

    assert [frame.capture_frame_id for frame in copied.frames()] == list(
        range(copied.start_frame_id, copied.end_frame_id + 1)
    )
    assert copied.media_sha256 == frozen.media_sha256


def test_materialization_rejects_hash_mutation_without_activating_destination(
    tmp_path: Path,
) -> None:
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    first_segment = next(iter(frozen.segment_sha256))
    (frozen.root / first_segment).write_bytes(b"mutated")
    destination = tmp_path / "jobs" / "shot-0001"

    with pytest.raises(RealtimeRingError, match="hash"):
        materialize_frozen_interval(frozen, destination)

    assert not destination.exists()
    assert not list(destination.parent.glob(".pending-shot-0001-*"))


def test_materialization_rejects_truncated_segment_even_when_hash_is_declared(
    tmp_path: Path,
) -> None:
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    first_segment = next(iter(frozen.segment_sha256))
    path = frozen.root / first_segment
    path.write_bytes(path.read_bytes()[:-1])
    declared = dict(frozen.segment_sha256)
    declared[first_segment] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = replace(frozen, segment_sha256=declared)
    destination = tmp_path / "jobs" / "shot-0001"

    with pytest.raises(RealtimeRingError, match="truncated"):
        materialize_frozen_interval(frozen, destination)

    assert not destination.exists()


def test_quarantine_preserves_mutated_raw_segments_and_diagnostic_manifest(
    tmp_path: Path,
) -> None:
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    first_segment = next(iter(frozen.segment_sha256))
    (frozen.root / first_segment).write_bytes(b"mutated evidence")

    record = quarantine_frozen_interval(
        frozen,
        tmp_path / "quarantine" / "shot-0001",
        "declared segment hash changed",
    )
    manifest = json.loads((record.root / "quarantine.json").read_text(encoding="utf-8"))

    assert (record.root / first_segment).read_bytes() == b"mutated evidence"
    assert manifest["reason"] == "declared segment hash changed"
    assert (
        manifest["segments"][first_segment]["declared_sha256"]
        != manifest["segments"][first_segment]["observed_sha256"]
    )
    assert len(record.manifest_sha256) == 64


@pytest.mark.parametrize(
    "mutation",
    ["omit_manifest_entry", "change_declared_hash", "delete_media"],
)
def test_existing_quarantine_requires_every_exact_declared_segment(
    tmp_path: Path,
    mutation: str,
) -> None:
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    destination = tmp_path / "quarantine" / "shot-0001"
    reason = "normal materialization failed"
    quarantine_frozen_interval(frozen, destination, reason)
    first_segment = next(iter(frozen.segment_sha256))
    manifest_path = destination / "quarantine.json"

    if mutation == "delete_media":
        (destination / first_segment).unlink()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "omit_manifest_entry":
            del manifest["segments"][first_segment]
        else:
            manifest["segments"][first_segment]["declared_sha256"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RealtimeRingError, match="quarantine|segment"):
        quarantine_frozen_interval(frozen, destination, reason)


def test_enqueue_reuses_same_content_key_and_rejects_changed_content(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")

    assert store.enqueue(shot_job).created is True
    assert store.enqueue(shot_job).created is False
    with pytest.raises(ValueError, match="content key conflict"):
        store.enqueue(replace(shot_job, media_sha256="f" * 64))


def test_pending_job_survives_store_restart(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    database = tmp_path / "state.sqlite3"
    SQLiteShotAnalysisJobStore(database).enqueue(shot_job)

    claimed = SQLiteShotAnalysisJobStore(database).claim("worker-a", NOW)

    assert claimed == shot_job


def test_running_job_is_exclusive_until_lease_is_older_than_120_seconds(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)

    assert store.claim("worker-a", NOW) == shot_job
    assert store.claim("worker-b", NOW + timedelta(seconds=120)) is None
    assert store.claim("worker-b", NOW + timedelta(seconds=121)) == shot_job


def test_failed_job_is_immediately_retryable(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)
    claim = store.claim_for_session(
        "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert claim is not None
    store.fail_claim(claim, "detector unavailable", now_utc=NOW, retry_delay=timedelta())

    assert store.status(shot_job.job_id) is ShotAnalysisJobStatus.FAILED_RETRYING
    assert store.claim("worker-b", NOW + timedelta(seconds=1)) == shot_job


def test_failed_job_does_not_starve_later_pending_job(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    later_job = replace(
        shot_job,
        job_id="job-0002",
        shot_id="shot-0002",
        media_root=tmp_path / "jobs" / "shot-0002",
    )
    store.enqueue(shot_job)
    store.enqueue(later_job)
    claim = store.claim_for_session(
        "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert claim is not None
    store.fail_claim(claim, "detector unavailable", now_utc=NOW, retry_delay=timedelta())

    assert store.claim("worker-a", NOW + timedelta(seconds=1)) == later_job


def test_complete_is_idempotent_and_never_replaces_artifact(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)
    claim = store.claim_for_session(
        "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert claim is not None

    assert store.complete_claim(claim, "artifact-1") is True
    assert store.complete_claim(claim, "artifact-1") is False
    assert store.status(shot_job.job_id) is ShotAnalysisJobStatus.AVAILABLE


def test_expired_claim_token_cannot_complete_or_fail_a_reclaimed_job(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    """Catches removal of the lease-generation fence on delayed workers."""
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)

    first = store.claim_for_session(
        "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert isinstance(first, ShotAnalysisClaim)
    second = store.claim_for_session(
        "worker-b",
        shot_job.session_id,
        NOW + timedelta(seconds=121),
        expected_job_ids=(shot_job.job_id,),
    )
    assert isinstance(second, ShotAnalysisClaim)

    assert store.complete_claim(first, "artifact-stale") is False
    assert store.fail_claim(first, "stale worker", now_utc=NOW) is False
    assert store.complete_claim(second, "artifact-current") is True


def test_legacy_completion_methods_cannot_bypass_the_claim_fence(
    tmp_path: Path, shot_job: ShotAnalysisJob
) -> None:
    """Catches restoring a job-ID-only write path beside fenced worker writes."""
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)
    assert (
        store.claim_for_session(
            "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
        )
        is not None
    )

    with pytest.raises(ValueError, match="lease claim token"):
        store.complete(shot_job.job_id, "artifact-1")
    with pytest.raises(ValueError, match="lease claim token"):
        store.fail_retrying(shot_job.job_id, "failed")


def test_failed_claim_waits_for_its_persisted_backoff_before_reclaim(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    """Catches a failed job hot-looping immediately back into a worker."""
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)
    claim = store.claim_for_session(
        "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert isinstance(claim, ShotAnalysisClaim)

    assert store.fail_claim(claim, "detector unavailable", now_utc=NOW) is True
    assert (
        store.claim_for_session(
            "worker-b",
            shot_job.session_id,
            NOW + timedelta(seconds=1),
            expected_job_ids=(shot_job.job_id,),
        )
        is None
    )
    assert (
        store.claim_for_session(
            "worker-b",
            shot_job.session_id,
            NOW + timedelta(seconds=31),
            expected_job_ids=(shot_job.job_id,),
        )
        is not None
    )


def test_claim_rejects_database_payload_mutation(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = SQLiteShotAnalysisJobStore(database)
    store.enqueue(shot_job)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json = ? WHERE job_id = ?",
            ('{"mutated":true}', shot_job.job_id),
        )

    with pytest.raises(ValueError, match="payload hash mismatch"):
        store.claim("worker-a", NOW)

    assert store.status(shot_job.job_id) is ShotAnalysisJobStatus.PENDING


@pytest.mark.parametrize("transition", ["complete", "fail_retrying"])
def test_state_transition_rejects_database_payload_mutation(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
    transition: str,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = SQLiteShotAnalysisJobStore(database)
    store.enqueue(shot_job)
    claim = store.claim_for_session(
        "worker-a", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert claim is not None
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json = ? WHERE job_id = ?",
            ('{"mutated":true}', shot_job.job_id),
        )

    with pytest.raises(ValueError, match="payload hash mismatch"):
        if transition == "complete":
            store.complete_claim(claim, "artifact-1")
        else:
            store.fail_claim(claim, "detector unavailable", now_utc=NOW)

    assert store.status(shot_job.job_id) is ShotAnalysisJobStatus.RUNNING


def test_scoped_claim_never_decodes_unrelated_corrupt_due_job(tmp_path, shot_job):
    db = tmp_path / "jobs.sqlite"
    store = SQLiteShotAnalysisJobStore(db)
    foreign = replace(shot_job, job_id="foreign", session_id="foreign-session")
    store.enqueue(foreign)
    store.enqueue(shot_job)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json='corrupt' WHERE job_id='foreign'"
        )
    claim = store.claim_for_session(
        "worker", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert claim.job == shot_job
    assert store.status("foreign") is ShotAnalysisJobStatus.PENDING


def test_scoped_claim_preserves_global_order_across_expected_id_chunks(tmp_path, shot_job):
    store = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite")
    first = replace(shot_job, job_id="z-first")
    second = replace(shot_job, job_id="a-second")
    store.enqueue(first)
    store.enqueue(second)
    expected = tuple(f"missing-{i:04d}" for i in range(1100)) + (second.job_id, first.job_id)
    claim = store.claim_for_session("worker", shot_job.session_id, NOW, expected_job_ids=expected)
    assert claim.job == first
    assert (
        store.claim_for_session("other", shot_job.session_id, NOW, expected_job_ids=expected).job
        == second
    )


def test_media_job_identity_is_deterministic_and_strict(media_job: ShotMediaEncodeJobV1) -> None:
    rebuilt = ShotMediaEncodeJobV1.create(
        session_id=media_job.session_id,
        shot_id=media_job.shot_id,
        source_epoch_id=media_job.source_epoch_id,
        start_frame_id=media_job.start_frame_id,
        end_frame_id=media_job.end_frame_id,
        media_root=media_job.media_root,
        media_sha256=media_job.media_sha256,
        source=media_job.source,
        max_attempts=media_job.max_attempts,
    )
    changed = ShotMediaEncodeJobV1.create(
        session_id=media_job.session_id,
        shot_id=media_job.shot_id,
        source_epoch_id=media_job.source_epoch_id,
        start_frame_id=media_job.start_frame_id,
        end_frame_id=media_job.end_frame_id + 1,
        media_root=media_job.media_root,
        media_sha256=media_job.media_sha256,
        source=media_job.source,
        max_attempts=media_job.max_attempts,
    )
    changed_policy = ShotMediaEncodeJobV1.create(
        session_id=media_job.session_id,
        shot_id=media_job.shot_id,
        source_epoch_id=media_job.source_epoch_id,
        start_frame_id=media_job.start_frame_id,
        end_frame_id=media_job.end_frame_id,
        media_root=media_job.media_root,
        media_sha256=media_job.media_sha256,
        source=media_job.source,
        max_attempts=media_job.max_attempts + 1,
    )

    assert rebuilt == media_job
    assert rebuilt.job_id.startswith("media-encode-")
    assert changed.job_id != rebuilt.job_id
    assert changed_policy.job_id == rebuilt.job_id
    with pytest.raises(ValueError, match="job ID"):
        replace(media_job, job_id="media-encode-" + "0" * 64)
    with pytest.raises(ValueError, match="reference path"):
        replace(
            media_job,
            source=replace(media_job.source, reference_path=Path("foreign/interval.json")),
        )
    store = SQLiteShotAnalysisJobStore(media_job.media_root.parent / "identity.sqlite3")
    store.enqueue_media(media_job)
    with pytest.raises(ValueError, match="content key conflict"):
        store.enqueue_media(changed_policy)


def test_legacy_media_source_identity_is_part_of_the_job_key(tmp_path: Path) -> None:
    values = dict(
        session_id="session-legacy",
        shot_id="shot-legacy",
        source_epoch_id="epoch-legacy",
        start_frame_id=1,
        end_frame_id=2,
        media_root=tmp_path / "legacy",
        media_sha256="1" * 64,
    )
    first = ShotMediaEncodeJobV1.create(
        **values,
        source=LegacyFrozenJobMediaSourceV1(
            kind="legacy_frozen_job",
            postgame_manifest_sha256="2" * 64,
            source_job_id="analysis-1",
            source_job_payload_sha256="3" * 64,
        ),
    )
    second = ShotMediaEncodeJobV1.create(
        **values,
        source=replace(first.source, source_job_payload_sha256="4" * 64),
    )

    assert first.job_id != second.job_id


def test_analysis_and_media_claims_are_type_isolated(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
    media_job: ShotMediaEncodeJobV1,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue_media(media_job)
    store.enqueue(shot_job)

    assert store.claim("analysis-worker", NOW) == shot_job
    assert (
        store.claim_for_session(
            "analysis-scoped",
            media_job.session_id,
            NOW,
            expected_job_ids=(media_job.job_id,),
        )
        is None
    )
    media_claim = store.claim_media_for_session("media-worker", media_job.session_id, NOW)
    assert isinstance(media_claim, ShotMediaEncodeClaim)
    assert media_claim.job == media_job
    assert store.complete_claim(media_claim, "raw-artifact-1") is True
    assert store.status(media_job.job_id) is ShotAnalysisJobStatus.AVAILABLE


def test_media_terminal_failure_is_fenced_and_survives_restart(
    tmp_path: Path,
    media_job: ShotMediaEncodeJobV1,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = SQLiteShotAnalysisJobStore(database)
    store.enqueue_media(media_job)
    first = store.claim_media_for_session("worker-a", media_job.session_id, NOW)
    assert isinstance(first, ShotMediaEncodeClaim)
    assert store.fail_media_claim(
        first,
        "codec unavailable",
        error_code="H264_ENCODER_UNAVAILABLE",
        now_utc=NOW,
        retry_delay=timedelta(),
    )
    second = store.claim_media_for_session(
        "worker-b", media_job.session_id, NOW + timedelta(seconds=1)
    )
    assert isinstance(second, ShotMediaEncodeClaim)
    assert store.complete_claim(first, "stale-artifact") is False
    assert store.fail_media_claim(
        second,
        "codec unavailable",
        error_code="H264_ENCODER_UNAVAILABLE",
        now_utc=NOW + timedelta(seconds=1),
    )

    snapshot = store.snapshot(media_job.job_id)
    assert isinstance(snapshot, ShotMediaEncodeJobSnapshot)
    assert snapshot.status is ShotAnalysisJobStatus.FAILED_RETRYING
    assert snapshot.terminal_failure is True
    assert snapshot.error_code == "H264_ENCODER_UNAVAILABLE"
    assert snapshot.attempt_count == 2
    restarted = SQLiteShotAnalysisJobStore(database)
    assert (
        restarted.claim_media_for_session("worker-c", media_job.session_id, NOW + timedelta(days=1))
        is None
    )


def test_media_repeated_lease_expiry_reaches_terminal_limit(
    tmp_path: Path,
    media_job: ShotMediaEncodeJobV1,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue_media(media_job)
    first = store.claim_media_for_session("worker-a", media_job.session_id, NOW)
    assert isinstance(first, ShotMediaEncodeClaim)
    second = store.claim_media_for_session(
        "worker-b", media_job.session_id, NOW + timedelta(seconds=121)
    )
    assert isinstance(second, ShotMediaEncodeClaim)

    assert (
        store.claim_media_for_session(
            "worker-c", media_job.session_id, NOW + timedelta(seconds=242)
        )
        is None
    )
    snapshot = store.snapshot(media_job.job_id)
    assert isinstance(snapshot, ShotMediaEncodeJobSnapshot)
    assert snapshot.terminal_failure is True
    assert snapshot.error_code == "MEDIA_LEASE_EXHAUSTED"
    assert snapshot.attempt_count == media_job.max_attempts


def test_media_lists_and_payload_digest_do_not_change_legacy_analysis_projection(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
    media_job: ShotMediaEncodeJobV1,
) -> None:
    store = SQLiteShotAnalysisJobStore(tmp_path / "state.sqlite3")
    store.enqueue(shot_job)
    store.enqueue_media(media_job)

    assert store.list_session_jobs(shot_job.session_id) == (shot_job,)
    assert store.list_session_media_jobs(shot_job.session_id) == (media_job,)
    assert store.list_session_work_items(shot_job.session_id) == (shot_job, media_job)
    assert len(store.payload_sha256(media_job.job_id)) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("start_frame_id", True),
        lambda payload: payload.__setitem__("media_root", 7),
        lambda payload: payload["source"].__setitem__("reference_sha256", 9),
    ],
)
def test_media_payload_rejects_wrong_json_types(
    tmp_path: Path,
    media_job: ShotMediaEncodeJobV1,
    mutation,
) -> None:
    database = tmp_path / "state.sqlite3"
    store = SQLiteShotAnalysisJobStore(database)
    store.enqueue_media(media_job)
    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            "SELECT payload_json FROM shot_analysis_jobs WHERE job_id=?", (media_job.job_id,)
        ).fetchone()[0]
        payload = json.loads(raw)
        mutation(payload)
        changed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? WHERE job_id=?",
            (changed, hashlib.sha256(changed.encode()).hexdigest(), media_job.job_id),
        )

    with pytest.raises(ValueError, match="payload"):
        store.snapshot(media_job.job_id)


def test_typed_lists_ignore_corruption_in_the_other_job_kind(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
    media_job: ShotMediaEncodeJobV1,
) -> None:
    first_db = tmp_path / "first.sqlite3"
    first = SQLiteShotAnalysisJobStore(first_db)
    first.enqueue(shot_job)
    first.enqueue_media(media_job)
    with sqlite3.connect(first_db) as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json='corrupt' WHERE job_id=?",
            (media_job.job_id,),
        )
    assert first.list_session_jobs(shot_job.session_id) == (shot_job,)

    second_db = tmp_path / "second.sqlite3"
    second = SQLiteShotAnalysisJobStore(second_db)
    second.enqueue(shot_job)
    second.enqueue_media(media_job)
    with sqlite3.connect(second_db) as connection:
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json='corrupt' WHERE job_id=?",
            (shot_job.job_id,),
        )
    assert second.list_session_media_jobs(media_job.session_id) == (media_job,)


def test_old_schema_rows_migrate_without_rewriting_analysis_payload(
    tmp_path: Path,
    shot_job: ShotAnalysisJob,
) -> None:
    source_db = tmp_path / "source.sqlite3"
    SQLiteShotAnalysisJobStore(source_db).enqueue(shot_job)
    with sqlite3.connect(source_db) as connection:
        payload_json, payload_sha256 = connection.execute(
            "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
            (shot_job.job_id,),
        ).fetchone()

    legacy_db = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy_db) as connection:
        connection.execute(
            "CREATE TABLE shot_analysis_jobs (job_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL,"
            "payload_sha256 TEXT NOT NULL,status TEXT NOT NULL CHECK (status IN "
            "('pending','running','available','failed_retrying')),claimed_by TEXT,"
            "claimed_at_utc TEXT,artifact_id TEXT,failure_reason TEXT,"
            "attempt_count INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute(
            "INSERT INTO shot_analysis_jobs(job_id,payload_json,payload_sha256,status) "
            "VALUES(?,?,?,'pending')",
            (shot_job.job_id, payload_json, payload_sha256),
        )

    migrated = SQLiteShotAnalysisJobStore(legacy_db)
    with sqlite3.connect(legacy_db) as connection:
        assert connection.execute(
            "SELECT payload_json,payload_sha256,job_kind FROM shot_analysis_jobs WHERE job_id=?",
            (shot_job.job_id,),
        ).fetchone() == (payload_json, payload_sha256, None)
    claim = migrated.claim_for_session(
        "worker", shot_job.session_id, NOW, expected_job_ids=(shot_job.job_id,)
    )
    assert isinstance(claim, ShotAnalysisClaim)
    assert migrated.complete_claim(claim, "artifact-legacy") is True


def test_recovery_queue_open_never_creates_or_migrates_missing_schema(tmp_path):
    import sqlite3

    missing = tmp_path / "missing.sqlite3"
    with pytest.raises((ValueError, sqlite3.Error)):
        SQLiteShotAnalysisJobStore.open_existing(missing)
    assert not missing.exists()
    empty = tmp_path / "empty.sqlite3"
    with sqlite3.connect(empty) as db:
        db.execute("CREATE TABLE unrelated(value TEXT)")
    before = empty.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        SQLiteShotAnalysisJobStore.open_existing(empty)
    assert empty.read_bytes() == before


def test_recovery_queue_connection_never_recreates_a_removed_original(tmp_path):
    path = tmp_path / "original.sqlite3"
    SQLiteShotAnalysisJobStore(path)
    recovered = SQLiteShotAnalysisJobStore.open_existing(path)
    assert recovered.list_session_work_items("original-session") == ()
    path.rename(tmp_path / "retained.sqlite3")
    with pytest.raises(ValueError, match="unavailable"):
        recovered.list_session_work_items("original-session")
    assert not path.exists()
