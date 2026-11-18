"""Independent, read-only consumer for frozen ended-session USB shots."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_fact_report_payload,
)
from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import session_scoped_clip_id
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.edge.route_assistance.shot_diagnostics import is_capture_interruption
from billiards_ai.lab.mainline1_live_publisher import publish_live_shot
from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    publish_recording_playback,
    publish_selected_original_clips,
    validate_recording_capture_media_authority,
    validate_recording_interval_media_authority,
)
from billiards_ai.lab.mainline1_session_analysis import DevelopmentSessionAnalysisPublisher
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    RecordingCaptureMediaSourceV1,
    RecordingEndMediaSourceV1,
    RecordingIntervalMediaSourceV1,
    ShotAnalysisClaim,
    ShotAnalysisJob,
    ShotMediaEncodeClaim,
    ShotMediaEncodeJobV1,
    SQLiteShotAnalysisJobStore,
)
from billiards_ai.lab.mainline1_shot_local_video import (
    Mainline1ShotLocalConfig,
    build_shot_evidence,
    read_shot_local_artifact_inventory,
    verify_mainline1_shot_local_artifact,
)
from billiards_ai.lab.mainline1_shot_product import build_mainline1_shot_product
from billiards_ai.lab.shot_local.observations import (
    DecodedTrackingFrame,
    GroupIdentityClassifier,
    collect_observation_frames,
)
from billiards_ai.lab.shot_local.product import (
    build_canonical_shot_analyses,
    shot_local_product_active_config,
)
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.shared.calibration.homography import TableCalibration


class LiveResourceGate:
    """A live-priority seam; the worker never waits for the live service."""

    def may_run_heavy_analysis(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class SQLiteLiveResourceGate(LiveResourceGate):
    """Pause heavy work whenever local live capture has priority."""

    records: SQLiteFieldRecordStore
    capture_healthy: Callable[[], bool] = lambda: True
    accelerator_available: Callable[[], bool] = lambda: True

    def may_run_heavy_analysis(self) -> bool:
        return (
            not self.records.has_open_session()
            and bool(self.capture_healthy())
            and bool(self.accelerator_available())
        )


@dataclass(frozen=True, slots=True)
class Mainline1ShotWorkerDependencies:
    calibration: TableCalibration
    config: Mainline1ShotLocalConfig
    vision_backend: VisionBackend
    identity_classifier: GroupIdentityClassifier
    fps: float
    capture_setup_id: str
    calibration_artifact_id: str
    config_sha256: str
    model_package_id: str
    identity_training_package_id: str | None = None
    identity_training_package_sha256: str | None = None
    resource_gate: LiveResourceGate = LiveResourceGate()


@dataclass(frozen=True, slots=True)
class ShotAnalysisResult:
    job_id: str
    artifact_id: str
    artifact_path: Path


class MediaEncodingResourceGate:
    """A lightweight live-priority seam independent of model/accelerator state."""

    cooperative_pause = False

    def cancellation_requested(self) -> bool:
        return False

    def may_run_media_encoding(self, session_id: str) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ShotMediaEncodingResult:
    job_id: str
    artifact_id: str
    artifact_path: Path


class Mainline1RecordingMediaWorker:
    """Encode original intervals without constructing semantic dependencies."""

    def __init__(
        self,
        *,
        records: SQLiteFieldRecordStore,
        jobs: SQLiteShotAnalysisJobStore,
        artifact_root: Path,
        resource_gate: MediaEncodingResourceGate | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        lease_now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._records = records
        self._jobs = jobs
        self._artifact_root = artifact_root
        self._resource_gate = (
            MediaEncodingResourceGate() if resource_gate is None else resource_gate
        )
        self._monotonic = monotonic
        self._lease_now_utc = lease_now_utc

    def schedule(self, session_id: str) -> tuple[ShotMediaEncodeJobV1, ...]:
        session = self._records.load_session(session_id)
        manifest = self._records.get_postgame_manifest(session_id)
        if (
            session.status is FieldSessionStatus.OPEN
            and self._records.get_session_policy(session_id) == "recording-v1"
        ):
            self._schedule_recording_intervals(session_id)
            return self._jobs.list_session_media_jobs(session_id)
        if session.status is not FieldSessionStatus.ENDED or manifest is None:
            return ()
        if self._records.get_session_policy(session_id) == "recording-v1":
            self._schedule_recording_end(session_id, manifest)
        else:
            self._schedule_legacy(session_id, manifest)
        return self._jobs.list_session_media_jobs(session_id)

    def process_one(
        self, worker_id: str, session_id: str, now_utc: datetime
    ) -> ShotMediaEncodingResult | None:
        if not self._resource_gate.may_run_media_encoding(session_id):
            return None
        scheduled = self.schedule(session_id)
        if not scheduled:
            return None
        root = self._records.artifact_path_for_session(session_id)
        try:
            with session_media_lock(root, session_id, exclusive=False, timeout_seconds=0):
                self._check_continue(session_id, None)
                publish_selected_original_clips(
                    self._records,
                    self._jobs,
                    self._records.get_postgame_manifest(session_id),
                )
                claim = self._jobs.claim_media_for_session(
                    worker_id,
                    session_id,
                    now_utc,
                    expected_job_ids=tuple(job.job_id for job in scheduled),
                )
                if claim is None:
                    return None
                return self._process_claim(claim, now_utc)
        except TimeoutError:
            return None
        except RecordingPlaybackError as exc:
            if exc.code == "MEDIA_LIVE_PRIORITY":
                return None
            raise

    def _process_claim(
        self, claim: ShotMediaEncodeClaim, now_utc: datetime
    ) -> ShotMediaEncodingResult | None:
        active_claim = claim
        last_renewed = self._monotonic()

        def continue_encoding(*, force_renew: bool = False) -> None:
            nonlocal active_claim, last_renewed
            while True:
                if self._resource_gate.cancellation_requested():
                    raise RecordingPlaybackError("MEDIA_LIVE_PRIORITY", "media service is stopping")
                if self._is_banned(active_claim.job.session_id, active_claim.job.shot_id):
                    raise RecordingPlaybackError(
                        "MEDIA_SOURCE_REVOKED", "original shot is deleted or discarded"
                    )
                allowed = self._resource_gate.may_run_media_encoding(active_claim.job.session_id)
                if allowed or not self._resource_gate.cooperative_pause:
                    if not allowed:
                        raise RecordingPlaybackError(
                            "MEDIA_LIVE_PRIORITY", "live capture has priority"
                        )
                    break
                observed = self._monotonic()
                if observed - last_renewed >= 30:
                    renewed = self._jobs.renew_media_claim(active_claim, self._lease_now_utc())
                    if renewed is None:
                        raise RecordingPlaybackError("MEDIA_CLAIM_LOST", "media lease was replaced")
                    active_claim, last_renewed = renewed, observed
                time.sleep(0.05)
            observed = self._monotonic()
            if force_renew or observed - last_renewed >= 30:
                renewed = self._jobs.renew_media_claim(active_claim, self._lease_now_utc())
                if renewed is None:
                    raise RecordingPlaybackError(
                        "MEDIA_CLAIM_LOST", "media encoding lease was replaced"
                    )
                active_claim = renewed
                last_renewed = observed

        try:
            self._validate_authority(active_claim.job)
            artifact = publish_recording_playback(
                active_claim.job,
                self._artifact_root,
                authority_check=self._validate_authority,
                continue_check=continue_encoding,
            )
            self._validate_authority(active_claim.job)
            continue_encoding(force_renew=True)
            with self._records.media_publication_guard(active_claim.job) as allowed:
                if not allowed:
                    raise RecordingPlaybackError(
                        "MEDIA_SOURCE_REVOKED", "media publication was revoked"
                    )
                if not self._jobs.complete_claim(
                    active_claim, artifact.artifact_id, lease_now_utc=self._lease_now_utc
                ):
                    return None
            publish_selected_original_clips(
                self._records,
                self._jobs,
                self._records.get_postgame_manifest(active_claim.job.session_id),
            )
            return ShotMediaEncodingResult(
                active_claim.job.job_id, artifact.artifact_id, artifact.path
            )
        except RecordingPlaybackError as exc:
            if exc.code == "MEDIA_CLAIM_LOST":
                return None
            if exc.code == "MEDIA_LIVE_PRIORITY":
                self._jobs.release_media_claim(active_claim)
            elif exc.code == "MEDIA_SOURCE_REVOKED":
                self._jobs.terminal_media_claim(active_claim, str(exc), error_code=exc.code)
            else:
                self._jobs.fail_media_claim(
                    active_claim,
                    str(exc) or type(exc).__name__,
                    error_code=exc.code,
                    now_utc=self._lease_now_utc(),
                )
            return None
        except (OSError, RuntimeError, ValueError, AttributeError, cv2.error) as exc:
            self._jobs.fail_media_claim(
                active_claim,
                str(exc) or type(exc).__name__,
                error_code="MEDIA_ENCODING_FAILED",
                now_utc=self._lease_now_utc(),
            )
            return None

    def _schedule_recording_intervals(self, session_id):
        existing = {job.shot_id for job in self._jobs.list_session_media_jobs(session_id)}
        for shot in self._records.list_shots(session_id):
            if shot.shot_id in existing or self._is_banned(session_id, shot.shot_id):
                continue
            evidence = self._records.get_recording_evidence(session_id, shot.shot_id)
            media, boundary = evidence.get("media"), evidence.get("boundary")
            if (
                evidence.get("media_status") != "RECORDED"
                or not isinstance(media, dict)
                or not isinstance(boundary, dict)
                or boundary.get("complete") is not True
            ):
                continue
            reference = Path(media["reference_path"])
            job = ShotMediaEncodeJobV1.create(
                session_id=session_id,
                shot_id=shot.shot_id,
                source_epoch_id=boundary["source_epoch_id"],
                start_frame_id=media["start_frame_id"],
                end_frame_id=media["end_frame_id"],
                media_root=reference.parent,
                media_sha256=media["media_sha256"],
                source=RecordingIntervalMediaSourceV1(
                    kind="recording_interval",
                    shot_evidence_sha256=_canonical_sha256(evidence),
                    reference_path=reference,
                    reference_sha256=media["reference_sha256"],
                ),
            )
            validate_recording_interval_media_authority(self._records, job)
            self._jobs.enqueue_media(job)

    def _schedule_recording_end(self, session_id: str, manifest) -> None:
        if manifest.expected_jobs:
            raise ValueError("ordinary recording manifest must not contain analysis jobs")
        capture = self._records.get_recording_capture_end(session_id)
        if capture is not None:
            self._schedule_recording_capture(session_id, capture)
            return
        recording = self._records.get_recording_end_manifest(session_id)
        if recording is None:
            return
        existing = {job.shot_id for job in self._jobs.list_session_media_jobs(session_id)}
        for reference in recording["shots"]:
            if reference["shot_id"] in existing:
                continue
            evidence = self._records.get_recording_evidence(session_id, reference["shot_id"])
            digest = _canonical_sha256(evidence)
            if digest != reference["payload_sha256"]:
                raise ValueError("recording evidence digest conflicts with frozen end")
            media = evidence["media"]
            boundary = evidence["boundary"]
            if (
                evidence["media_status"] != "RECORDED"
                or not isinstance(media, dict)
                or not isinstance(boundary, dict)
                or media.get("status") != "RECORDED"
                or not boundary.get("complete")
            ):
                continue
            reference_path = Path(media["reference_path"])
            job = ShotMediaEncodeJobV1.create(
                session_id=session_id,
                shot_id=evidence["shot_id"],
                source_epoch_id=boundary["source_epoch_id"],
                start_frame_id=media["start_frame_id"],
                end_frame_id=media["end_frame_id"],
                media_root=reference_path.parent,
                media_sha256=media["media_sha256"],
                source=RecordingEndMediaSourceV1(
                    kind="recording_end",
                    recording_end_sha256=recording["manifest_sha256"],
                    shot_evidence_sha256=digest,
                    reference_path=reference_path,
                    reference_sha256=media["reference_sha256"],
                ),
            )
            if not self._is_banned(session_id, job.shot_id):
                self._jobs.enqueue_media(job)

    def _schedule_recording_capture(self, session_id, capture):
        existing = {job.shot_id for job in self._jobs.list_session_media_jobs(session_id)}
        for shot in self._records.list_shots(session_id):
            if shot.shot_id in existing:
                continue
            evidence = self._records.get_recording_evidence(session_id, shot.shot_id)
            media, boundary = evidence["media"], evidence["boundary"]
            if (
                evidence["media_status"] != "RECORDED"
                or not isinstance(media, dict)
                or not isinstance(boundary, dict)
                or not boundary.get("complete")
            ):
                continue
            reference_path = Path(media["reference_path"])
            job = ShotMediaEncodeJobV1.create(
                session_id=session_id,
                shot_id=shot.shot_id,
                source_epoch_id=boundary["source_epoch_id"],
                start_frame_id=media["start_frame_id"],
                end_frame_id=media["end_frame_id"],
                media_root=reference_path.parent,
                media_sha256=media["media_sha256"],
                source=RecordingCaptureMediaSourceV1(
                    kind="recording_capture",
                    capture_end_sha256=capture["capture_end_sha256"],
                    shot_evidence_sha256=_canonical_sha256(evidence),
                    reference_path=reference_path,
                    reference_sha256=media["reference_sha256"],
                ),
            )
            if not self._is_banned(session_id, shot.shot_id):
                validate_recording_capture_media_authority(self._records, job)
                self._jobs.enqueue_media(job)

    def _schedule_legacy(self, session_id: str, manifest) -> None:
        canonical = {
            clip.job_id
            for clip in self._records.list_postgame_clips(session_id)
            if clip.clip_id is not None
        }
        for source_job_id in manifest.expected_jobs:
            snapshot = self._jobs.snapshot(source_job_id)
            source = snapshot.job
            if (
                not isinstance(source, ShotAnalysisJob)
                or source.job_id != source_job_id
                or source.session_id != session_id
                or not self._legacy_source_is_valid(session_id, source)
            ):
                raise ValueError("legacy manifest source job lineage conflicts")
            if source_job_id in canonical or self._is_banned(session_id, source.shot_id):
                continue
            job = ShotMediaEncodeJobV1.create(
                session_id=source.session_id,
                shot_id=source.shot_id,
                source_epoch_id=source.source_epoch_id,
                start_frame_id=source.start_frame_id,
                end_frame_id=source.end_frame_id,
                media_root=source.media_root,
                media_sha256=source.media_sha256,
                source=LegacyFrozenJobMediaSourceV1(
                    kind="legacy_frozen_job",
                    postgame_manifest_sha256=manifest.manifest_sha256,
                    source_job_id=source.job_id,
                    source_job_payload_sha256=self._jobs.payload_sha256(source.job_id),
                ),
            )
            self._jobs.enqueue_media(job)

    def _validate_authority(self, job: ShotMediaEncodeJobV1) -> None:
        if self._resource_gate.cooperative_pause:
            if self._resource_gate.cancellation_requested():
                raise RecordingPlaybackError("MEDIA_LIVE_PRIORITY", "media service is stopping")
            if self._is_banned(job.session_id, job.shot_id):
                raise RecordingPlaybackError(
                    "MEDIA_SOURCE_REVOKED", "original shot is deleted or discarded"
                )
        else:
            self._check_continue(job.session_id, job.shot_id)
        if isinstance(job.source, RecordingIntervalMediaSourceV1):
            validate_recording_interval_media_authority(self._records, job)
            return
        session = self._records.load_session(job.session_id)
        manifest = self._records.get_postgame_manifest(job.session_id)
        if session.status is not FieldSessionStatus.ENDED or manifest is None:
            raise RecordingPlaybackError(
                "MEDIA_SOURCE_REVOKED", "media source session is not frozen and ended"
            )
        if isinstance(job.source, RecordingCaptureMediaSourceV1):
            validate_recording_capture_media_authority(self._records, job)
        elif isinstance(job.source, RecordingEndMediaSourceV1):
            recording = self._records.get_recording_end_manifest(job.session_id)
            if recording is None or recording["manifest_sha256"] != job.source.recording_end_sha256:
                raise RecordingPlaybackError(
                    "MEDIA_SOURCE_REVOKED", "recording end authority changed"
                )
            reference = next(
                (item for item in recording["shots"] if item["shot_id"] == job.shot_id),
                None,
            )
            evidence = self._records.get_recording_evidence(job.session_id, job.shot_id)
            digest = _canonical_sha256(evidence)
            media = evidence["media"]
            boundary = evidence["boundary"]
            if (
                reference is None
                or reference["payload_sha256"] != job.source.shot_evidence_sha256
                or digest != job.source.shot_evidence_sha256
                or evidence["media_status"] != "RECORDED"
                or media.get("status") != "RECORDED"
                or not boundary.get("complete")
                or (
                    boundary["source_epoch_id"],
                    media["start_frame_id"],
                    media["end_frame_id"],
                    media["media_sha256"],
                    Path(media["reference_path"]),
                    media["reference_sha256"],
                )
                != (
                    job.source_epoch_id,
                    job.start_frame_id,
                    job.end_frame_id,
                    job.media_sha256,
                    job.source.reference_path,
                    job.source.reference_sha256,
                )
            ):
                raise RecordingPlaybackError(
                    "MEDIA_SOURCE_REVOKED", "recording evidence authority changed"
                )
        else:
            if (
                manifest.manifest_sha256 != job.source.postgame_manifest_sha256
                or job.source.source_job_id not in manifest.expected_jobs
                or self._jobs.payload_sha256(job.source.source_job_id)
                != job.source.source_job_payload_sha256
            ):
                raise RecordingPlaybackError(
                    "MEDIA_SOURCE_REVOKED", "legacy frozen task authority changed"
                )
            source = self._jobs.snapshot(job.source.source_job_id).job
            if (
                not isinstance(source, ShotAnalysisJob)
                or (
                    source.session_id,
                    source.shot_id,
                    source.source_epoch_id,
                    source.start_frame_id,
                    source.end_frame_id,
                    source.media_root,
                    source.media_sha256,
                )
                != (
                    job.session_id,
                    job.shot_id,
                    job.source_epoch_id,
                    job.start_frame_id,
                    job.end_frame_id,
                    job.media_root,
                    job.media_sha256,
                )
                or not self._legacy_source_is_valid(job.session_id, source)
            ):
                raise RecordingPlaybackError(
                    "MEDIA_SOURCE_REVOKED", "legacy source lineage changed"
                )

    def _check_continue(self, session_id: str, shot_id: str | None) -> None:
        if not self._resource_gate.may_run_media_encoding(session_id):
            raise RecordingPlaybackError(
                "MEDIA_LIVE_PRIORITY", "live capture has priority over media encoding"
            )
        if shot_id is not None and self._is_banned(session_id, shot_id):
            raise RecordingPlaybackError(
                "MEDIA_SOURCE_REVOKED", "original shot is deleted or discarded"
            )

    def _is_banned(self, session_id: str, shot_id: str) -> bool:
        return self._records.media_encoding_state(session_id, shot_id)["banned"]

    def _legacy_source_is_valid(self, session_id: str, source: ShotAnalysisJob) -> bool:
        shot = next(
            (row for row in self._records.list_shots(session_id) if row.shot_id == source.shot_id),
            None,
        )
        media = self._records.get_live_shot_media(session_id, source.shot_id)
        return bool(
            source.session_id == session_id
            and shot is not None
            and (shot.session_id, shot.shot_id) == (session_id, source.shot_id)
            and shot.settled_ms is not None
            and shot.media.source_kind == "usb_live"
            and shot.live_job_id == source.job_id
            and media is not None
            and (
                media.session_id,
                media.shot_id,
                media.job_id,
                media.source_epoch_id,
                media.media_sha256,
            )
            == (
                source.session_id,
                source.shot_id,
                source.job_id,
                source.source_epoch_id,
                source.media_sha256,
            )
            and Path(media.media_path).resolve() == source.media_root.resolve()
        )


def load_materialized_interval(job: ShotAnalysisJob) -> FrozenCaptureInterval:
    """Rebuild and validate the immutable interval without trusting a path alone."""
    root = job.media_root
    if not root.is_absolute() or root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise ValueError("job media root is missing or unsafe")
    manifest_path = root / "interval.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("job interval manifest is missing or unsafe")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "1.0" or raw.get("media_sha256") != job.media_sha256:
            raise ValueError("job interval manifest identity is invalid")
        storage_kind = raw.get("storage_kind", "copied_segments")
        if storage_kind == "session_segment_reference":
            segment_root = Path(str(raw["segment_root"]))
            if (
                not segment_root.is_absolute()
                or segment_root != segment_root.resolve()
                or segment_root.is_symlink()
                or not segment_root.is_dir()
            ):
                raise ValueError("job session segment root is missing or unsafe")
        elif storage_kind == "copied_segments":
            segment_root = root.resolve()
        else:
            raise ValueError("job interval storage kind is invalid")
        frozen = FrozenCaptureInterval(
            root=segment_root,
            source_epoch_id=str(raw["source_epoch_id"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            start_frame_id=int(raw["start_frame_id"]),
            end_frame_id=int(raw["end_frame_id"]),
            frame_count=int(raw["frame_count"]),
            segment_sha256={str(name): str(value) for name, value in raw["segment_sha256"].items()},
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("job interval manifest is invalid") from exc
    if (
        frozen.source_epoch_id != job.source_epoch_id
        or frozen.start_frame_id != job.start_frame_id
        or frozen.end_frame_id != job.end_frame_id
        or frozen.media_sha256 != job.media_sha256
    ):
        raise ValueError("job interval identity does not match job")
    if frozen.frame_count != frozen.end_frame_id - frozen.start_frame_id + 1:
        raise ValueError("job interval frame count is invalid")
    for name in frozen.segment_sha256:
        candidate = frozen.root / name
        if (
            Path(name).name != name
            or candidate.resolve().parent != frozen.root
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise ValueError("job interval segment is missing or unsafe")
    # frames() re-hashes every segment and proves contiguous decode order.
    frames = tuple(frozen.frames())
    if len(frames) != frozen.frame_count:
        raise ValueError("job interval decoded frame count is invalid")
    for frame in frames:
        image = cv2.imdecode(np.frombuffer(frame.mjpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (frozen.height, frozen.width):
            raise ValueError("job interval JPEG dimensions are invalid")
    return frozen


class Mainline1ShotWorker:
    """One-shot worker with no live-match mutation surface."""

    def __init__(
        self,
        *,
        records: SQLiteFieldRecordStore,
        jobs: SQLiteShotAnalysisJobStore,
        artifact_root: Path,
        dependencies: Mainline1ShotWorkerDependencies,
        session_analysis_publisher: DevelopmentSessionAnalysisPublisher | None = None,
    ) -> None:
        self._records = records
        self._jobs = jobs
        self._artifact_root = artifact_root
        self._deps = dependencies
        self._session_analysis_publisher = session_analysis_publisher

    def process_one(
        self, worker_id: str, session_id: str, now_utc: datetime
    ) -> ShotAnalysisResult | None:
        if (
            not self._deps.resource_gate.may_run_heavy_analysis()
            or self._records.load_session(session_id).status is not FieldSessionStatus.ENDED
            or self._records.get_postgame_manifest(session_id) is None
        ):
            return None
        # Bind to the same capture root the HTTP server uses; do not infer it
        # from the worker's per-session postgame output directory.
        root = self._records.artifact_path_for_session(session_id)
        try:
            with session_media_lock(root, session_id, exclusive=False, timeout_seconds=0):
                if self._records.is_postgame_finalized(session_id):
                    self._publish_session_analysis(session_id, now_utc)
                    return None
                return self._process_one_pinned(worker_id, session_id, now_utc)
        except TimeoutError:
            return None

    def _process_one_pinned(
        self, worker_id: str, session_id: str, now_utc: datetime
    ) -> ShotAnalysisResult | None:
        if not self._deps.resource_gate.may_run_heavy_analysis():
            return None
        session = self._records.load_session(session_id)
        manifest = self._records.get_postgame_manifest(session_id)
        if session.status is not FieldSessionStatus.ENDED or manifest is None:
            return None
        self._publish_session_analysis(session_id, now_utc)
        shots = self._records.list_shots(session_id)
        # Serialize admission against the field-store deletion/selection fence.
        # The jobs transaction finishes before releasing this lock, so deletion
        # either precedes admission or observes an already admitted reader.
        with self._records._connect() as field:
            field.execute("BEGIN IMMEDIATE")
            eligible = tuple(
                shot.live_job_id
                for shot in shots
                if shot.live_job_id in manifest.expected_jobs
                and not self._records._is_postgame_discarded_in(
                    field,
                    session_id,
                    shot.shot_id,
                )
                and field.execute(
                    "SELECT 1 FROM field_owner_media_deletions WHERE session_id=? AND shot_id=?",
                    (session_id, shot.shot_id),
                ).fetchone()
                is None
            )
            claim = self._jobs.claim_for_session(
                worker_id,
                session_id,
                now_utc,
                expected_job_ids=eligible,
            )
        if claim is None:
            self._finalize_if_available(session_id)
            self._publish_session_analysis(session_id, now_utc)
            return None
        try:
            if self._records.owner_delete_for_shot(session_id, claim.job.shot_id) is not None:
                raise ValueError("postgame clip has been deleted by owner")
            if self._records.is_postgame_discarded(session_id, claim.job.shot_id):
                raise ValueError("postgame clip has been discarded")
            self._validate_lineage(claim)
            self._check_gate()
            interval = load_materialized_interval(claim.job)
            decoded = _decode_interval(interval)
            self._check_gate()
            observations = collect_observation_frames(
                frames=decoded,
                vision_backend=self._deps.vision_backend,
                identity_classifier=self._deps.identity_classifier,
                calibration=self._deps.calibration,
            )
            self._check_gate()
            results = build_shot_evidence(observations=observations, config=self._deps.config)
            if len(results) != 1 or not results[0].window.complete:
                raise ValueError("frozen USB interval did not reproduce one complete shot")
            timing: dict[str, float] = {}
            analyses = build_canonical_shot_analyses(
                evidence_results=results,
                observations=observations,
                images_by_frame={frame.frame_id: frame.image for frame in decoded},
                event_config=self._deps.config.event,
                active_config=shot_local_product_active_config(),
                timing_ms=timing,
            )
            self._check_gate()
            product = build_mainline1_shot_product(
                decoded_frames=decoded,
                observations=observations,
                results=results,
                analyses=analyses,
                calibration=self._deps.calibration,
                config=self._deps.config,
                fps=self._deps.fps,
            )
            artifact = publish_live_shot(
                product=product,
                identity=_artifact_identity(claim.job),
                output_root=self._artifact_root,
            )
            verify_mainline1_shot_local_artifact(artifact.path)
            if not self._jobs.complete_claim(claim, artifact.artifact_id):
                return None
            self._finalize_if_available(session_id)
            self._publish_session_analysis(session_id, now_utc)
            return ShotAnalysisResult(claim.job.job_id, artifact.artifact_id, artifact.path)
        except (OSError, RuntimeError, ValueError, AttributeError, cv2.error) as exc:
            self._jobs.fail_claim(
                claim,
                str(exc) or type(exc).__name__,
                now_utc=datetime.now(now_utc.tzinfo),
            )
            self._publish_session_analysis(session_id, now_utc)
            return None

    def _publish_session_analysis(self, session_id: str, now_utc: datetime) -> None:
        if self._session_analysis_publisher is not None:
            self._session_analysis_publisher.refresh_and_publish(session_id, now_utc)

    def _finalize_if_available(self, session_id: str) -> None:
        if self._records.is_postgame_finalized(session_id):
            return
        manifest = self._records.get_postgame_manifest(session_id)
        if manifest is None:
            return
        shots = self._records.list_shots(session_id)
        has_capture_interruption = any(is_capture_interruption(shot) for shot in shots)
        snapshots = tuple(self._jobs.snapshot(job_id) for job_id in manifest.expected_jobs)
        clips = []
        for expected_job_id, snapshot in zip(manifest.expected_jobs, snapshots, strict=True):
            if snapshot.job.job_id != expected_job_id or snapshot.job.session_id != session_id:
                raise ValueError("published job identity does not match frozen manifest")
            if self._records.is_postgame_discarded(session_id, snapshot.job.shot_id):
                # Selection revokes this head, not later selected heads. Keep
                # whole-session finalization incomplete while publishing each
                # surviving available clip through its original save receipt.
                continue
            if snapshot.status.value != "available" or not snapshot.artifact_id:
                continue
            deleted = self._records.owner_delete_for_shot(session_id, snapshot.job.shot_id)
            if deleted is not None:
                task = deleted.task
                # Replay admission validates original availability, selection, facts
                # and live-media fingerprints without reopening removed video.
                deleted = self._records.accept_owner_delete_task(
                    task,
                    managed_scope=ManagedSessionScope(
                        task.environment_id, task.device_id, task.table_id
                    ),
                )
                if (
                    deleted.session_id,
                    deleted.shot_id,
                    deleted.job_id,
                    deleted.postgame_manifest_sha256,
                ) != (
                    session_id,
                    snapshot.job.shot_id,
                    snapshot.job.job_id,
                    manifest.manifest_sha256,
                ):
                    raise ValueError("deleted artifact identity does not match frozen job")
                self._validate_job_lineage(snapshot.job)
                if deleted.artifact_id != snapshot.artifact_id:
                    selected = self._records.get_saved_media_outbox(deleted.clip_id)
                    if selected is None or selected.local_path.name != "original.mp4":
                        raise ValueError("deleted artifact identity does not match frozen job")
                    # Original selection and analysis have separate immutable
                    # media identities. Keep the selected receipt, and prove the
                    # successful analysis through its retained content inventory.
                    canonical = read_shot_local_artifact_inventory(
                        self._artifact_root / snapshot.artifact_id
                    )
                    expected = _artifact_identity(snapshot.job)
                    if canonical.get(
                        "artifact_id_algorithm"
                    ) != "identity-semantic-inventory-v2" or any(
                        canonical.get(key) != value for key, value in expected.items()
                    ):
                        raise ValueError("deleted analysis inventory conflicts with frozen job")
                clips.append(
                    {
                        "clip_id": deleted.clip_id,
                        "job_id": deleted.job_id,
                        "shot_id": deleted.shot_id,
                        "artifact_id": deleted.artifact_id,
                        "path": str(deleted.local_path),
                        "sha256": deleted.content_sha256,
                    }
                )
                continue
            assert snapshot.artifact_id is not None
            artifact_path = self._artifact_root / snapshot.artifact_id
            artifact = verify_mainline1_shot_local_artifact(artifact_path)
            expected_identity = _artifact_identity(snapshot.job)
            if any(artifact.get(key) != value for key, value in expected_identity.items()):
                raise ValueError("published artifact identity does not match frozen job")
            overlay = artifact_path / "overlay.mp4"
            resolved_overlay = overlay.resolve()
            content_sha256 = hashlib.sha256(overlay.read_bytes()).hexdigest()
            byte_size = overlay.stat().st_size
            self._records.record_postgame_clip_available(
                session_id=session_id,
                shot_id=snapshot.job.shot_id,
                postgame_manifest_sha256=manifest.manifest_sha256,
                job_id=snapshot.job.job_id,
                artifact_id=snapshot.artifact_id,
                local_path=resolved_overlay,
                content_sha256=content_sha256,
                byte_size=byte_size,
                content_type="video/mp4",
            )
            clips.append(
                {
                    "clip_id": session_scoped_clip_id(
                        session_id,
                        snapshot.job.shot_id,
                        snapshot.job.job_id,
                        snapshot.artifact_id,
                    ),
                    "job_id": snapshot.job.job_id,
                    "shot_id": snapshot.job.shot_id,
                    "artifact_id": snapshot.artifact_id,
                    "path": str(resolved_overlay),
                    "sha256": content_sha256,
                }
            )
        if has_capture_interruption or len(clips) != len(snapshots):
            return
        report = compose_field_fact_report(self._records.load_session(session_id), shots)
        local_manifest = json.dumps(
            {"report": field_fact_report_payload(report), "clips": clips},
            sort_keys=True,
            separators=(",", ":"),
        )
        self._records.record_postgame_finalization(session_id, local_manifest)
        # The transaction also creates local_postgame_outbox. USB media must not
        # enter CloudBase-facing field_sync_jobs before its source contract exists.

    def _validate_lineage(self, claim: ShotAnalysisClaim) -> None:
        self._validate_job_lineage(claim.job)

    def _validate_job_lineage(self, job: ShotAnalysisJob) -> None:
        if (
            job.capture_setup_id != self._deps.capture_setup_id
            or job.calibration_artifact_id != self._deps.calibration_artifact_id
            or job.config_sha256 != self._deps.config_sha256
            or job.model_package_id != self._deps.model_package_id
            or getattr(self._deps.vision_backend.package_identity, "package_id", None)
            != job.model_package_id
            or not job.identity_training_package_id
            or not job.identity_training_package_sha256
            or job.identity_training_package_id != self._deps.identity_training_package_id
            or job.identity_training_package_sha256 != self._deps.identity_training_package_sha256
        ):
            raise ValueError("worker immutable dependencies do not match job identity")
        manifest = self._records.get_postgame_manifest(job.session_id)
        if manifest is None or job.job_id not in manifest.expected_jobs:
            raise ValueError("job is not frozen in its ended session manifest")
        shot = next(
            (row for row in self._records.list_shots(job.session_id) if row.shot_id == job.shot_id),
            None,
        )
        media = self._records.get_live_shot_media(job.session_id, job.shot_id)
        if (
            shot is None
            or shot.settled_ms is None
            or shot.media.source_kind != "usb_live"
            or shot.live_job_id != job.job_id
            or media is None
            or (
                media.session_id,
                media.shot_id,
                media.job_id,
                media.source_epoch_id,
                media.media_sha256,
            )
            != (
                job.session_id,
                job.shot_id,
                job.job_id,
                job.source_epoch_id,
                job.media_sha256,
            )
            or Path(media.media_path).resolve() != job.media_root.resolve()
        ):
            raise ValueError("USB postgame job lineage does not match persisted field facts")

    def _check_gate(self) -> None:
        if not self._deps.resource_gate.may_run_heavy_analysis():
            raise ValueError("live resource gate paused postgame analysis")


def _decode_interval(interval: FrozenCaptureInterval) -> tuple[DecodedTrackingFrame, ...]:
    decoded = []
    for frame in interval.frames():
        image = cv2.imdecode(np.frombuffer(frame.mjpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("frozen USB frame cannot be decoded")
        decoded.append(
            DecodedTrackingFrame(
                frame_id=frame.capture_frame_id,
                timestamp_ms=round(frame.captured_at_ns / 1_000_000),
                image=image,
            )
        )
    return tuple(decoded)


def _artifact_identity(job: ShotAnalysisJob) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "kind": "mainline1_shot_local_video",
        "artifact_role": "usb_postgame_shot",
        "source_kind": "usb_live",
        "job_id": job.job_id,
        "session_id": job.session_id,
        "live_shot_id": job.shot_id,
        "source_epoch_id": job.source_epoch_id,
        "media_sha256": job.media_sha256,
        "capture_setup_id": job.capture_setup_id,
        "calibration_artifact_id": job.calibration_artifact_id,
        "config_sha256": job.config_sha256,
        "model_package_id": job.model_package_id,
        "identity_training_package_id": job.identity_training_package_id,
        "identity_training_package_sha256": job.identity_training_package_sha256,
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
