from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import session_scoped_clip_id
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotMediaReference,
    ShotMediaRange,
)
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing
from billiards_ai.lab.mainline1_session_analysis import (
    DevelopmentSessionAnalysisPublisher,
    SQLiteSessionAnalysisOutbox,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJob,
    ShotAnalysisJobStatus,
    SQLiteShotAnalysisJobStore,
    materialize_frozen_interval,
)
from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
from billiards_ai.lab.mainline1_shot_worker import (
    Mainline1ShotWorker,
    Mainline1ShotWorkerDependencies,
    load_materialized_interval,
)
from billiards_ai.lab.shot_local.cue import CueTrackingConfig
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.fragments import FragmentTrackingConfig
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.shared.calibration.homography import calibrate
from billiards_ai.shared.models import SessionAnalysisProjectionV1
from tests.unit.lab.test_mainline1_shot_jobs import _populated_ring

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)
MODEL_PACKAGE_ID = "a" * 64
IDENTITY_PACKAGE_SHA256 = "b" * 64
CONFIG_SHA256 = "c" * 64


def _detection(detection_id: str, x: int, y: int = 60) -> BallInstanceObservation:
    return BallInstanceObservation(
        detection_id=detection_id,
        bbox=PixelBox(left=x - 4, top=y - 4, right=x + 4, bottom=y + 4),
        center=PixelPoint(x=x, y=y),
        confidence=0.95,
        class_label="ball",
    )


class _FakeDetector:
    package_identity = VisionPackageIdentity(
        package_id=MODEL_PACKAGE_ID,
        manifest_sha256="d" * 64,
    )

    def infer(self, frame) -> VisionFrameObservation:
        cue_x = (50, 50, 60, 70, 70, 70, 70, 70)[frame.frame_id]
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            package=self.package_identity,
            threshold=0.25,
            detections=(
                _detection(f"cue-{frame.frame_id}", cue_x),
                _detection(f"target-{frame.frame_id}", 100),
            ),
            runtime_ref="fake-detector/1",
        )


class _FakeClassifier:
    def classify(
        self,
        *,
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
    ) -> Mapping[str, Mapping[str, Any]]:
        return {
            detection.detection_id: {
                "group": "cue" if detection.detection_id.startswith("cue-") else "solid",
                "confidence": 0.95,
            }
            for detection in detections
        }


def _jpeg(frame_id: int) -> bytes:
    image = np.full((120, 160, 3), frame_id, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _materialized_interval(root: Path, *, epoch: str, job_id: str, media_root: Path | None = None):
    ring = MjpegSegmentRing(
        root / "ring",
        expected_fps=60,
        pre_roll_frames=2,
        post_roll_frames=2,
        maximum_shot_frames=12,
        maximum_bytes=8 * 1024 * 1024,
    )
    timestamps_ms = (0, 40, 80, 120, 160, 240, 280, 320)
    for frame_id, timestamp_ms in enumerate(timestamps_ms):
        ring.append(
            CaptureFrame(
                source_epoch_id=epoch,
                capture_frame_id=frame_id,
                captured_at_ns=timestamp_ms * 1_000_000,
                width=160,
                height=120,
                mjpeg=_jpeg(frame_id),
            )
        )
        if frame_id == 2:
            ring.begin_shot(frame_id)
        if frame_id == 5:
            ring.settle_shot(frame_id)
    frozen = ring.freeze_shot()
    return materialize_frozen_interval(frozen, media_root or root / "media" / job_id)


def _worker_config() -> Mainline1ShotLocalConfig:
    return Mainline1ShotLocalConfig(
        window=ShotWindowConfig(
            motion_threshold_mm=8,
            motion_confirm_frames=2,
            still_duration_ms=100,
        ),
        cue=CueTrackingConfig(ambiguity_margin=0.08, reacquire_confidence_min=0.55),
        fragment=FragmentTrackingConfig(
            motion_threshold_mm=5,
            continuation_distance_mm=40,
            ambiguity_mm=3,
        ),
        event=ShotLocalEventConfig(
            first_contact_distance_mm=35,
            first_contact_ambiguity_mm=3,
            cue_motion_min_mm=5,
            contact_time_tolerance_ms=120,
            cushion_distance_mm=10,
            pocket_distance_mm=15,
            pot_missing_frames=3,
            after_match_distance_mm=20,
            table_length_mm=160,
            table_width_mm=120,
        ),
    )


def _record_ended_usb_session(
    records: SQLiteFieldRecordStore,
    jobs: SQLiteShotAnalysisJobStore,
    root: Path,
    *,
    session_id: str,
    shot_id: str,
    job_id: str,
    epoch: str,
    artifact_id: str,
    sequence: int = 1,
    end_session: bool = True,
    media_root: Path | None = None,
    resource=None,
) -> ShotAnalysisJob:
    materialized = _materialized_interval(root, epoch=epoch, job_id=job_id, media_root=media_root)
    from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole

    session = (
        records.load_session(session_id)
        if records.has_managed_session_identity(session_id)
        else records.open_or_resume_session(artifact_id=artifact_id)
    )
    if session.match.revision == 0:
        records.save_match(
            session_id,
            MatchControlSnapshot(
                red_user_id="user-a",
                blue_user_id="user-b",
                active_role=WristbandRole.RED,
                revision=1,
            ),
            expected_revision=0,
        )
    media = ShotMediaRange(
        artifact_id=artifact_id,
        source_locator=f"usb:{epoch}",
        source_kind="usb_live",
        start_ms=0,
        end_ms=320,
        source_epoch_id=epoch,
    )
    started = FieldShotRecord(
        session_id=session_id,
        shot_id=shot_id,
        sequence=sequence,
        actor_role=WristbandRole.RED,
        started_ms=80,
        settled_ms=None,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=media,
    )
    records.record_shot_start(started, source_start_event_id=f"start-{shot_id}")
    settled = replace(
        started,
        settled_ms=240,
        live_evidence_digest="e" * 64,
        live_job_id=job_id,
    )
    records.record_shot_settle(settled, source_epoch_id=epoch)
    records.attach_live_shot_media(
        LiveShotMediaReference(
            session_id=session_id,
            shot_id=shot_id,
            source_epoch_id=epoch,
            match_revision=1,
            job_id=job_id,
            media_path=str(materialized.root),
            media_sha256=materialized.media_sha256,
        )
    )
    job = ShotAnalysisJob(
        job_id=job_id,
        session_id=session_id,
        shot_id=shot_id,
        source_epoch_id=epoch,
        start_frame_id=materialized.start_frame_id,
        end_frame_id=materialized.end_frame_id,
        media_root=materialized.root,
        media_sha256=materialized.media_sha256,
        capture_setup_id=resource.capture_setup_id if resource else "capture-1",
        calibration_artifact_id=resource.calibration_artifact_id if resource else "calibration-1",
        config_sha256=resource.config_sha256 if resource else CONFIG_SHA256,
        model_package_id=resource.model_package_id if resource else MODEL_PACKAGE_ID,
        identity_training_package_id=resource.identity_training_package_id
        if resource
        else "identity-1",
        identity_training_package_sha256=resource.identity_training_package_sha256
        if resource
        else IDENTITY_PACKAGE_SHA256,
    )
    jobs.enqueue(job)
    if end_session:
        records.end_session(session_id)
    return job


def _worker(
    records: SQLiteFieldRecordStore,
    jobs: SQLiteShotAnalysisJobStore,
    artifact_root: Path,
    *,
    analysis_publisher: DevelopmentSessionAnalysisPublisher | None = None,
    resource=None,
) -> Mainline1ShotWorker:
    # Production binds this capture root before starting the analysis worker.
    for session in records.list_ended_sessions_requiring_postgame():
        root = (
            artifact_root.parent.parent
            if artifact_root.parent.name == "postgame"
            else artifact_root.parent
        )
        root.mkdir(parents=True, exist_ok=True)
        records.bind_artifact_path(session.artifact_id, root)
    detector = _FakeDetector()
    if resource is not None:
        detector.package_identity = detector.package_identity.model_copy(
            update={"package_id": resource.model_package_id},
        )
    return Mainline1ShotWorker(
        records=records,
        jobs=jobs,
        artifact_root=artifact_root,
        session_analysis_publisher=analysis_publisher,
        dependencies=Mainline1ShotWorkerDependencies(
            calibration=calibrate(
                "fake-camera",
                ((0, 120), (160, 120), (160, 0), (0, 0)),
                table_length_mm=160,
                table_width_mm=120,
            ),
            config=_worker_config(),
            vision_backend=detector,
            identity_classifier=_FakeClassifier(),
            fps=60,
            capture_setup_id=resource.capture_setup_id if resource else "capture-1",
            calibration_artifact_id=resource.calibration_artifact_id
            if resource
            else "calibration-1",
            config_sha256=resource.config_sha256 if resource else CONFIG_SHA256,
            model_package_id=resource.model_package_id if resource else MODEL_PACKAGE_ID,
            identity_training_package_id=resource.identity_training_package_id
            if resource
            else "identity-1",
            identity_training_package_sha256=resource.identity_training_package_sha256
            if resource
            else IDENTITY_PACKAGE_SHA256,
        ),
    )


def test_real_worker_does_not_publish_available_without_durable_media_sync(
    tmp_path: Path,
) -> None:
    class Gateway:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def publish_session_analysis(self, payload):
            self.payloads.append(json.loads(json.dumps(payload)))
            return SessionAnalysisProjectionV1.model_validate(
                {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "frozen_shots_digest",
                        "frozen_shot_ids",
                        "published_shots",
                        "published_clips",
                    }
                }
            )

    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    job = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )
    gateway = Gateway()
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=gateway,
    )
    worker = _worker(
        records,
        jobs,
        tmp_path / "artifacts",
        analysis_publisher=publisher,
    )

    assert worker.process_one("worker-1", job.session_id, NOW) is not None
    assert worker.process_one("worker-1", job.session_id, NOW + timedelta(seconds=1)) is None

    assert [(row["revision"], row["status"]) for row in gateway.payloads] == [
        (1, "PROCESSING"),
    ]


def _job(root: Path, media_sha256: str) -> ShotAnalysisJob:
    return ShotAnalysisJob(
        job_id="job-1",
        session_id="session-1",
        shot_id="shot-1",
        source_epoch_id="epoch-1",
        start_frame_id=0,
        end_frame_id=7,
        media_root=root,
        media_sha256=media_sha256,
        capture_setup_id="capture-1",
        calibration_artifact_id="calibration-1",
        config_sha256="a" * 64,
        model_package_id="model-1",
    )


def test_loader_rehashes_and_replays_the_exact_materialized_interval(tmp_path: Path) -> None:
    """Catches accepting an interval manifest or path without checking its bytes."""
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    root = tmp_path / "jobs" / "job-1"
    materialized = materialize_frozen_interval(frozen, root)

    loaded = load_materialized_interval(_job(root, materialized.media_sha256))

    assert [frame.capture_frame_id for frame in loaded.frames()] == list(range(8))


def test_loader_rejects_a_mutated_segment_before_analysis_can_start(tmp_path: Path) -> None:
    """Catches a post-end worker trusting immutable-media declarations blindly."""
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    root = tmp_path / "jobs" / "job-1"
    materialized = materialize_frozen_interval(frozen, root)
    (root / next(iter(materialized.segment_sha256))).write_bytes(b"changed")

    with pytest.raises(Exception, match="hash|interval"):
        load_materialized_interval(_job(root, materialized.media_sha256))


@pytest.mark.parametrize("mutation", ["schema", "media", "traversal", "frame_count"])
def test_loader_rejects_tampered_manifest_contract(tmp_path: Path, mutation: str) -> None:
    """Catches accepting a worker-controlled interval declaration before decode."""
    frozen = _populated_ring(tmp_path / "ring").freeze_shot()
    root = tmp_path / "jobs" / "job-1"
    materialized = materialize_frozen_interval(frozen, root)
    manifest_path = root / "interval.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "schema":
        manifest["schema_version"] = "2.0"
    elif mutation == "media":
        manifest["media_sha256"] = "f" * 64
    elif mutation == "traversal":
        digest = next(iter(manifest["segment_sha256"].values()))
        manifest["segment_sha256"] = {"../outside.mjpg": digest}
    else:
        manifest["frame_count"] = 99
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest|frame count|unsafe|identity"):
        load_materialized_interval(_job(root, materialized.media_sha256))


def test_process_one_runs_the_real_postgame_pipeline_and_finalizes_session(
    tmp_path: Path,
) -> None:
    """Proves the worker calls every real stage around only a fake detector."""
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    job = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )

    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    worker = _worker(records, jobs, tmp_path / "artifacts")
    with session_media_lock(tmp_path, job.session_id, exclusive=True):
        assert worker.process_one("worker-1", job.session_id, NOW) is None
        assert jobs.status(job.job_id) is ShotAnalysisJobStatus.PENDING
        assert not (tmp_path / "artifacts").exists()
    result = worker.process_one("worker-1", job.session_id, NOW)

    assert result is not None
    assert result.job_id == job.job_id
    assert jobs.status(job.job_id) is ShotAnalysisJobStatus.AVAILABLE
    manifest = json.loads((result.artifact_path / "manifest.json").read_text())
    assert manifest["job_id"] == job.job_id
    assert manifest["session_id"] == job.session_id
    assert manifest["live_shot_id"] == job.shot_id
    assert json.loads((result.artifact_path / "summary.json").read_text())["shot_count"] == 1
    assert records.is_postgame_finalized(job.session_id)
    with sqlite3.connect(records.path) as connection:
        finalization = connection.execute(
            "SELECT manifest_json FROM field_postgame_finalizations WHERE session_id=?",
            (job.session_id,),
        ).fetchone()
        outbox = connection.execute(
            "SELECT manifest_json,status FROM local_postgame_outbox WHERE session_id=?",
            (job.session_id,),
        ).fetchone()
    assert finalization is not None
    assert outbox == (finalization[0], "pending")
    local_manifest = json.loads(finalization[0])
    assert local_manifest["report"]["session_id"] == job.session_id
    assert local_manifest["clips"][0]["job_id"] == job.job_id
    assert local_manifest["clips"][0]["shot_id"] == job.shot_id
    review_clip = records.list_postgame_clips(job.session_id)[0]
    overlay = (result.artifact_path / "overlay.mp4").resolve()
    assert review_clip.shot_id == job.shot_id
    assert review_clip.artifact_id == result.artifact_id
    assert review_clip.local_path == overlay
    assert review_clip.content_sha256 == hashlib.sha256(overlay.read_bytes()).hexdigest()
    assert review_clip.byte_size == overlay.stat().st_size
    assert review_clip.content_type == "video/mp4"
    assert review_clip.save_status == "NOT_SAVED"
    assert records.list_saved_media_outbox() == ()
    records.complete_postgame_review(
        "field-artifact-1",
        job.session_id,
        records.get_postgame_manifest(job.session_id).manifest_sha256,
    )
    overlay.unlink()  # Simulate the separately owned collector, after finalization.
    assert (
        _worker(records, jobs, tmp_path / "artifacts").process_one(
            "restarted", job.session_id, NOW + timedelta(seconds=2)
        )
        is None
    )
    assert records.is_postgame_finalized(job.session_id)
    assert not overlay.exists()


def test_worker_completion_converges_a_save_before_clip_is_ready(tmp_path: Path) -> None:
    """Catches verified worker output never meeting an earlier kiosk save intent."""
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    job = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )
    manifest = records.get_postgame_manifest(job.session_id)
    assert manifest is not None
    receipt = records.save_postgame_clip(
        request_id="save-before-ready",
        session_id=job.session_id,
        shot_id=job.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert receipt.status == "SAVE_PENDING"
    assert records.list_saved_media_outbox() == ()

    result = _worker(records, jobs, tmp_path / "artifacts").process_one(
        "worker-1", job.session_id, NOW
    )

    assert result is not None
    outbox = records.list_saved_media_outbox()
    assert len(outbox) == 1
    assert outbox[0].session_id == job.session_id
    assert outbox[0].shot_id == job.shot_id
    assert outbox[0].artifact_id == result.artifact_id
    assert records.list_postgame_clips(job.session_id)[0].save_status == "SAVED"


def test_worker_availability_then_kiosk_save_enqueues_once_across_replay(
    tmp_path: Path,
) -> None:
    """Catches replaying finalization or save creating duplicate saved-media work."""
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    job = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )
    worker = _worker(records, jobs, tmp_path / "artifacts")
    result = worker.process_one("worker-1", job.session_id, NOW)
    assert result is not None
    assert worker.process_one("worker-1", job.session_id, NOW + timedelta(seconds=1)) is None
    manifest = records.get_postgame_manifest(job.session_id)
    assert manifest is not None

    first = records.save_postgame_clip(
        request_id="save-after-ready",
        session_id=job.session_id,
        shot_id=job.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    replay = records.save_postgame_clip(
        request_id="save-after-ready",
        session_id=job.session_id,
        shot_id=job.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )

    assert first == replay
    assert first.status == "SAVED"
    assert len(records.list_saved_media_outbox()) == 1


@pytest.mark.parametrize("blocker", ["pending", "failed", "interrupted"])
def test_available_shot_is_reviewable_before_other_jobs_finish(
    tmp_path: Path,
    blocker: str,
) -> None:
    """Catches all-session finalization gating per-shot review availability."""
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    first = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
        sequence=1,
        end_session=False,
    )
    if blocker == "interrupted":
        interrupted_epoch = "epoch-interrupted"
        interrupted = FieldShotRecord(
            session_id="session-1",
            shot_id="shot-2",
            sequence=2,
            actor_role=records.load_session("session-1").match.active_role,
            started_ms=80,
            settled_ms=None,
            recommendation_id=None,
            power_band=None,
            power_percent=None,
            media=ShotMediaRange(
                artifact_id="field-artifact-1",
                source_locator=f"usb:{interrupted_epoch}",
                source_kind="usb_live",
                start_ms=0,
                end_ms=320,
                source_epoch_id=interrupted_epoch,
            ),
        )
        records.record_shot_start(interrupted, source_start_event_id="start-shot-2")
        records.record_shot_settle(
            replace(
                interrupted,
                settled_ms=240,
                live_evidence_digest="f" * 64,
                live_job_id=f"capture-interrupted:{interrupted_epoch}:shot-2",
            ),
            source_epoch_id=interrupted_epoch,
        )
        records.end_session("session-1")
        second: ShotAnalysisJob | None = None
    else:
        second = _record_ended_usb_session(
            records,
            jobs,
            tmp_path / "capture-2",
            session_id="session-1",
            shot_id="shot-2",
            job_id="job-2",
            epoch="epoch-2",
            artifact_id="field-artifact-1",
            sequence=2,
        )
    manifest = records.get_postgame_manifest(first.session_id)
    assert manifest is not None
    records.save_postgame_clip(
        request_id=f"save-first-with-{blocker}",
        session_id=first.session_id,
        shot_id=first.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    worker = _worker(records, jobs, tmp_path / "artifacts")

    first_result = worker.process_one("worker-1", first.session_id, NOW)
    assert first_result is not None
    if blocker == "failed":
        assert second is not None
        failed_claim = jobs.claim_for_session(
            "failing-worker",
            first.session_id,
            NOW + timedelta(seconds=1),
            expected_job_ids=manifest.expected_jobs,
        )
        assert failed_claim is not None and failed_claim.job.job_id == second.job_id
        assert jobs.fail_claim(
            failed_claim,
            "fixture failure",
            now_utc=NOW + timedelta(seconds=1),
        )
        assert worker.process_one("worker-1", first.session_id, NOW + timedelta(seconds=2)) is None

    clips = records.list_postgame_clips(first.session_id)
    assert clips[0].local_path == (first_result.artifact_path / "overlay.mp4").resolve()
    assert clips[0].save_status == "SAVED"
    if blocker == "interrupted":
        assert len(clips) == 1
    else:
        assert clips[1].local_path is None
    outbox = records.list_saved_media_outbox()
    assert len(outbox) == 1
    assert (outbox[0].session_id, outbox[0].shot_id) == (first.session_id, first.shot_id)
    assert records.is_postgame_finalized(first.session_id) is False


def test_two_sessions_remain_isolated_and_a_stale_claim_is_recovered(
    tmp_path: Path,
) -> None:
    session_ids = iter(("session-1", "session-2"))
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: next(session_ids),
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    first = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )
    second = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-2",
        session_id="session-2",
        shot_id="shot-1",
        job_id="job-2",
        epoch="epoch-2",
        artifact_id="field-artifact-2",
    )
    stale = jobs.claim_for_session(
        "stale-worker",
        second.session_id,
        NOW,
        expected_job_ids=(second.job_id,),
    )
    assert stale is not None
    worker = _worker(records, jobs, tmp_path / "artifacts")

    first_result = worker.process_one("worker-current", first.session_id, NOW)
    second_result = worker.process_one(
        "worker-current", second.session_id, NOW + timedelta(seconds=121)
    )

    assert first_result is not None
    assert second_result is not None
    assert first_result.artifact_id != second_result.artifact_id
    assert first_result.artifact_path.parent == second_result.artifact_path.parent
    assert jobs.complete_claim(stale, "stale-artifact") is False
    with sqlite3.connect(records.path) as connection:
        finalizations = connection.execute(
            "SELECT session_id,manifest_json FROM field_postgame_finalizations ORDER BY session_id"
        ).fetchall()
        outbox = connection.execute(
            "SELECT session_id,manifest_json,status FROM local_postgame_outbox ORDER BY session_id"
        ).fetchall()
    assert [row[0] for row in finalizations] == ["session-1", "session-2"]
    assert [row[0] for row in outbox] == ["session-1", "session-2"]
    assert all(row[2] == "pending" for row in outbox)
    results_by_session = {
        "session-1": first_result,
        "session-2": second_result,
    }
    for (session_id, manifest_json), outbox_row in zip(finalizations, outbox, strict=True):
        assert outbox_row[1] == manifest_json
        local_manifest = json.loads(manifest_json)
        assert local_manifest["report"]["session_id"] == session_id
        assert {clip["job_id"] for clip in local_manifest["clips"]} == {
            "job-1" if session_id == "session-1" else "job-2"
        }
        result = results_by_session[session_id]
        job_id = "job-1" if session_id == "session-1" else "job-2"
        assert local_manifest["clips"][0]["clip_id"] == session_scoped_clip_id(
            session_id,
            "shot-1",
            job_id,
            result.artifact_id,
        )


@pytest.mark.parametrize(
    "tombstone",
    [
        "capture-interrupted:epoch-1:shot-1",
        "abstained:capture_interrupted:epoch-1:shot-1",
        "abstained:incomplete_freeze:epoch-1:shot-1",
    ],
)
def test_interrupted_capture_tombstone_is_operationally_closed_without_a_fake_clip(
    tmp_path: Path,
    tombstone: str,
) -> None:
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    records.open_or_resume_session(artifact_id="field-artifact-1")
    original = FieldShotRecord(
        session_id="session-1",
        shot_id="shot-1",
        sequence=1,
        actor_role=None,
        started_ms=80,
        settled_ms=None,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=ShotMediaRange(
            artifact_id="field-artifact-1",
            source_locator="usb:epoch-1",
            source_kind="usb_live",
            start_ms=0,
            end_ms=320,
            source_epoch_id="epoch-1",
        ),
    )
    records.record_shot_start(original, source_start_event_id="start-shot-1")
    records.record_shot_settle(
        replace(
            original,
            settled_ms=240,
            live_evidence_digest="e" * 64,
            live_job_id=tombstone,
        ),
        source_epoch_id="epoch-1",
    )
    records.end_session("session-1")
    outbox = SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3")
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=outbox,
    )

    result = _worker(
        records,
        jobs,
        tmp_path / "artifacts",
        analysis_publisher=publisher,
    ).process_one("worker-1", "session-1", NOW)

    assert result is None
    projection = outbox.entries("session-1")[-1].payload
    assert projection["status"] == "FAILED_RETRYING"
    assert projection["total_shots"] == 1
    assert projection["retrying_shots"] == 1
    assert projection["available_shots"] == 0
    assert projection["frozen_shot_ids"] == ["shot-1"]
    assert projection["published_clips"] == []
    assert records.has_sync_job("session-1") is False
    assert not records.is_postgame_finalized("session-1")


@pytest.fixture
def owner_deleted_partial_session(tmp_path, request):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionBinding
    from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
        BIRTH,
        SCOPE,
        SESSION,
        task,
    )

    records = SQLiteFieldRecordStore(tmp_path / "field.db", now_utc=lambda: NOW)
    open_request = records.reserve_managed_open(
        SCOPE, artifact_id="field-artifact-1", rule_profile_id=None
    )
    records.record_managed_binding(
        SCOPE,
        open_request.open_request_id,
        ManagedSessionBinding(
            "1.0", SESSION, BIRTH, "managed_session_birth_v1", "table-1", "device-1", "open"
        ),
    )
    records.materialize_managed_session(SCOPE, open_request.open_request_id)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    first = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "first",
        session_id=SESSION,
        shot_id="shot-0001",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
        end_session=False,
    )
    second = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "second",
        session_id=SESSION,
        shot_id="shot-0002",
        job_id="job-2",
        epoch="epoch-2",
        artifact_id="field-artifact-1",
        sequence=2,
        end_session=False,
    )
    if getattr(request, "param", None) == "interruption":
        interrupted = FieldShotRecord(
            SESSION,
            "shot-0003",
            3,
            records.load_session(SESSION).match.active_role,
            500,
            None,
            None,
            None,
            None,
            ShotMediaRange(
                "field-artifact-1", "usb:broken-epoch", "usb_live", 500, 501, "broken-epoch"
            ),
        )
        records.record_shot_start(interrupted, source_start_event_id="start-broken")
        records.record_shot_settle(
            replace(
                interrupted,
                settled_ms=600,
                media=replace(interrupted.media, end_ms=600),
                live_evidence_digest="f" * 64,
                live_job_id="capture-interrupted:broken-epoch:shot-0003",
            ),
            source_epoch_id="broken-epoch",
        )
    records.end_session(SESSION)
    worker = _worker(records, jobs, tmp_path / "postgame" / SESSION)
    manifest = records.get_postgame_manifest(SESSION)
    records.save_postgame_clip(
        request_id="save-1",
        session_id=SESSION,
        shot_id=first.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    result = worker.process_one("worker-1", SESSION, NOW)
    assert result is not None
    original = records.list_postgame_clips(SESSION)[0]
    deletion = records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    # Logical deletion must permit finalization even before physical completion.
    (result.artifact_path / "overlay.mp4").unlink()
    segments = json.loads((first.media_root / "interval.json").read_text())["segment_sha256"]
    assert segments
    for name in segments:
        (first.media_root / name).unlink()
    return records, jobs, worker, first, second, original, deletion


def test_owner_deleted_available_shot_finalizes_with_frozen_history_and_unsaved_only_cleanup(
    owner_deleted_partial_session,
):
    records, jobs, worker, first, second, original, deletion = owner_deleted_partial_session
    manifest = records.get_postgame_manifest(first.session_id)
    # Keep this review active to exercise late analysis after owner erasure.
    # S1 permits completing a partial review independently of finalization.
    assert records.active_postgame_review("field-artifact-1") == manifest
    assert not records.is_postgame_finalized(first.session_id)
    result = worker.process_one("worker-2", second.session_id, NOW)
    assert result is not None
    assert records.is_postgame_finalized(first.session_id)
    with records._connect() as db:
        raw = db.execute("SELECT manifest_json FROM field_postgame_finalizations").fetchone()[0]
    historical = json.loads(raw)["clips"][0]
    assert historical == {
        "clip_id": original.clip_id,
        "job_id": first.job_id,
        "shot_id": first.shot_id,
        "artifact_id": original.artifact_id,
        "path": str(original.local_path),
        "sha256": original.content_sha256,
    }
    assert set(json.loads(raw)) == {"clips", "report"}
    with records._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM local_postgame_outbox").fetchone()[0] == 1
    records.complete_postgame_review("field-artifact-1", first.session_id, manifest.manifest_sha256)
    assert not records.is_postgame_discarded(first.session_id, first.shot_id)
    assert records.is_postgame_discarded(second.session_id, second.shot_id)
    assert len(records.list_saved_media_outbox()) == 1
    assert records.get_owner_delete(deletion.clip_id).status == "pending"
    assert jobs.snapshot(first.job_id).status is ShotAnalysisJobStatus.AVAILABLE


def test_owner_deleted_shot_refuses_regeneration_before_touching_source(
    owner_deleted_partial_session,
    monkeypatch,
):
    import billiards_ai.lab.mainline1_shot_worker as module

    records, jobs, worker, first, second, _, _ = owner_deleted_partial_session
    with sqlite3.connect(jobs._database) as db:
        db.execute(
            "UPDATE shot_analysis_jobs SET status='pending',attempt_count=0 WHERE job_id=?",
            (first.job_id,),
        )
    before = jobs.snapshot(first.job_id)
    opened = []

    def loader(job):
        opened.append(job.job_id)
        raise ValueError("unexpected raw media read")

    monkeypatch.setattr(module, "load_materialized_interval", loader)
    assert worker.process_one("retry-worker", first.session_id, NOW) is None
    assert opened == [second.job_id]
    assert jobs.snapshot(first.job_id) == before


@pytest.mark.parametrize("changed", ["availability_path", "live_media", "frozen_job_artifact"])
def test_deleted_finalization_revalidates_retained_authority(
    owner_deleted_partial_session, changed, monkeypatch
):
    import billiards_ai.lab.mainline1_shot_worker as module

    monkeypatch.setattr(
        module,
        "verify_mainline1_shot_local_artifact",
        lambda *_: pytest.fail("deleted artifact reopened"),
    )
    records, jobs, worker, first, _, _, _ = owner_deleted_partial_session
    if changed == "frozen_job_artifact":
        with sqlite3.connect(jobs._database) as db:
            db.execute(
                "UPDATE shot_analysis_jobs SET artifact_id=? WHERE job_id=?",
                ("f" * 64, first.job_id),
            )
    else:
        with records._connect() as db:
            if changed == "availability_path":
                db.execute(
                    "UPDATE field_postgame_clip_availability "
                    "SET local_path=local_path||'.replacement' WHERE shot_id=?",
                    (first.shot_id,),
                )
            else:
                raw = db.execute(
                    "SELECT payload_json FROM field_live_shot_media WHERE shot_id=?",
                    (first.shot_id,),
                ).fetchone()[0]
                payload = json.loads(raw)
                payload["media_sha256"] = "f" * 64
                db.execute(
                    "UPDATE field_live_shot_media SET payload_json=? WHERE shot_id=?",
                    (json.dumps(payload), first.shot_id),
                )
    with pytest.raises(ValueError):
        worker._finalize_if_available(first.session_id)
    assert not records.is_postgame_finalized(first.session_id)


@pytest.mark.parametrize("changed", ["missing_video", "changed_video"])
def test_deleted_shot_exception_does_not_bypass_other_real_artifacts(
    owner_deleted_partial_session,
    changed,
    monkeypatch,
):
    records, _, worker, first, second, _, _ = owner_deleted_partial_session
    # Finish the other real analysis before asking the real finalizer to validate.
    with monkeypatch.context() as patch:
        patch.setattr(worker, "_finalize_if_available", lambda *_: None)
        result = worker.process_one("worker-2", second.session_id, NOW)
    assert result is not None
    overlay = result.artifact_path / "overlay.mp4"
    if changed == "missing_video":
        overlay.unlink()
    else:
        overlay.write_bytes(b"changed synthetic video")
    with pytest.raises(ValueError):
        worker._finalize_if_available(first.session_id)
    assert not records.is_postgame_finalized(first.session_id)


@pytest.mark.parametrize("owner_deleted_partial_session", ["interruption"], indirect=True)
def test_deleted_shot_exception_keeps_capture_interruption_and_review_gate(
    owner_deleted_partial_session,
):
    records, _, worker, first, second, _, _ = owner_deleted_partial_session
    assert worker.process_one("worker-2", second.session_id, NOW) is not None
    assert not records.is_postgame_finalized(first.session_id)
    manifest = records.get_postgame_manifest(first.session_id)
    records.complete_postgame_review("field-artifact-1", first.session_id, manifest.manifest_sha256)
    assert not records.is_postgame_finalized(first.session_id)


def test_deleted_exception_cannot_substitute_another_frozen_expected_job(
    owner_deleted_partial_session,
):
    records, jobs, worker, first, second, _, _ = owner_deleted_partial_session
    with sqlite3.connect(jobs._database) as db:
        payload, digest, artifact = db.execute(
            "SELECT payload_json,payload_sha256,artifact_id FROM shot_analysis_jobs WHERE job_id=?",
            (first.job_id,),
        ).fetchone()
        db.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=?,artifact_id=?,"
            "status='available' WHERE job_id=?",
            (payload, digest, artifact, second.job_id),
        )
    with pytest.raises(ValueError, match="frozen"):
        worker._finalize_if_available(first.session_id)
    assert not records.is_postgame_finalized(first.session_id)
