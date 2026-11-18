"""Independent bounded publication of original shot intervals for local replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from billiards_ai.edge.route_assistance.postgame import (
    ClipSaveStatus,
    PostgameClip,
    PostgameSessionManifest,
    session_scoped_clip_id,
)
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.lab.mainline1_realtime_ring import (
    FrozenCaptureInterval,
    RealtimeRingError,
    _same_file_identity,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    RecordingCaptureMediaSourceV1,
    RecordingEndMediaSourceV1,
    RecordingIntervalMediaSourceV1,
    ShotAnalysisJob,
    ShotAnalysisJobStatus,
    ShotMediaEncodeJobSnapshot,
    ShotMediaEncodeJobV1,
    SQLiteShotAnalysisJobStore,
)

_REFERENCE_LIMIT_BYTES = 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_OUTPUT_NAME = "original.mp4"
_MANIFEST_NAME = "manifest.json"


class RecordingPlaybackError(RuntimeError):
    """Typed, honest media publication failure suitable for bounded retry policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RecordingPlaybackArtifact:
    artifact_id: str
    path: Path
    media_path: Path
    manifest_sha256: str
    content_sha256: str
    byte_size: int
    frame_count: int
    width: int
    height: int
    fps: float
    first_captured_at_ns: int
    last_captured_at_ns: int


@dataclass(frozen=True, slots=True)
class _GeneratedMediaReceipt:
    """Ephemeral proof from this controlled encode, never a persistent stat cache."""

    job_sha256: str
    content_sha256: str
    byte_size: int
    device: int
    inode: int
    mtime_ns: int


def _reuse_generated_media(artifact_path, job, receipt, *, continue_check):
    artifact = recording_playback_inventory(artifact_path, job)
    before = artifact.media_path.stat()
    same_origin = (
        receipt.job_sha256 == hashlib.sha256(_canonical(job.identity()).encode()).hexdigest()
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (receipt.device, receipt.inode, receipt.byte_size, receipt.mtime_ns)
        and artifact.content_sha256 == receipt.content_sha256
        and artifact.byte_size == receipt.byte_size
    )
    if same_origin:
        observed = _hash_media_file(artifact.media_path, continue_check=continue_check)
        after = artifact.media_path.stat()
        _read_safe_small_file(artifact.path / _MANIFEST_NAME, artifact.manifest_sha256)
        if _same_file_identity(before, after) and observed == {
            "content_sha256": receipt.content_sha256,
            "byte_size": receipt.byte_size,
        }:
            return artifact
    # Existing/replaced/changed bytes have no matching in-process proof.
    return verify_recording_playback_artifact(artifact_path, job, continue_check=continue_check)


@dataclass(frozen=True, slots=True)
class RecordingPlaybackRow:
    """One frozen original shot plus its independent local-playback state."""

    shot_id: str
    sequence: int
    encoding_status: str
    media_job_id: str | None
    terminal_failure: bool
    error_code: str | None
    playback_clip: PostgameClip | None
    canonical_clip: PostgameClip | None


def project_recording_playback(
    records,
    jobs: SQLiteShotAnalysisJobStore,
    manifest: PostgameSessionManifest,
    *,
    clip_id: str | None = None,
    continue_check=lambda: None,
) -> tuple[RecordingPlaybackRow, ...]:
    """Resolve typed media tasks without creating jobs or canonical availability.

    Frozen recording inventory, not queue rows, determines which shots appear.  A
    completed media task is playable only while its original source authority and
    immutable artifact both still validate.
    """

    session = records.load_session(manifest.session_id)
    if session.status is not FieldSessionStatus.ENDED:
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "playback source session is not ended")
    field_shots = {shot.shot_id: shot for shot in records.list_shots(manifest.session_id)}
    canonical = {clip.shot_id: clip for clip in records.list_postgame_clips(manifest.session_id)}
    media_snapshots = jobs.session_media_snapshots(manifest.session_id)
    by_shot = {}
    for snapshot in media_snapshots:
        if snapshot.job.shot_id in by_shot:
            raise RecordingPlaybackError(
                "MEDIA_TASK_INVENTORY_INVALID", "multiple media tasks target one shot"
            )
        by_shot[snapshot.job.shot_id] = snapshot

    recording = records.get_recording_end_manifest(manifest.session_id)
    if records.get_session_policy(manifest.session_id) == "recording-v1":
        capture = records.get_recording_capture_end(manifest.session_id)
        if capture is not None:
            inventory = tuple(field_shots)
        elif recording is not None:
            inventory = tuple(item["shot_id"] for item in recording["shots"])
        else:
            raise RecordingPlaybackError(
                "MEDIA_SOURCE_REVOKED", "recording source authority is unavailable"
            )
    else:
        if recording is not None:
            raise RecordingPlaybackError(
                "MEDIA_SOURCE_REVOKED", "legacy playback has recording-end authority"
            )
        source_shots: dict[str, str] = {}
        for source_job_id in manifest.expected_jobs:
            source = jobs.snapshot(source_job_id).job
            if not isinstance(source, ShotAnalysisJob) or source.session_id != manifest.session_id:
                raise RecordingPlaybackError(
                    "MEDIA_SOURCE_REVOKED", "legacy frozen task authority changed"
                )
            if source.shot_id in source_shots:
                raise RecordingPlaybackError(
                    "MEDIA_TASK_INVENTORY_INVALID", "legacy shot task inventory conflicts"
                )
            source_shots[source.shot_id] = source_job_id
        inventory = tuple(
            shot.shot_id
            for shot in records.list_shots(manifest.session_id)
            if shot.shot_id in source_shots
        )

    result = []
    for shot_id in inventory:
        if continue_check is not None:
            continue_check()
        snapshot = by_shot.get(shot_id)
        if clip_id is not None and (
            snapshot is None
            or snapshot.status is not ShotAnalysisJobStatus.AVAILABLE
            or not isinstance(snapshot.artifact_id, str)
            or session_scoped_clip_id(
                manifest.session_id, shot_id, snapshot.job.job_id, snapshot.artifact_id
            )
            != clip_id
        ):
            continue
        shot = field_shots.get(shot_id)
        if shot is None:
            raise RecordingPlaybackError(
                "MEDIA_SOURCE_REVOKED", "frozen playback shot is unavailable"
            )
        if snapshot is not None and not isinstance(snapshot, ShotMediaEncodeJobSnapshot):
            raise RecordingPlaybackError(
                "MEDIA_TASK_INVENTORY_INVALID", "playback task kind conflicts"
            )
        canonical_clip = canonical.get(shot_id)
        if records.owner_delete_for_shot(
            manifest.session_id, shot_id
        ) is not None or records.is_postgame_discarded(manifest.session_id, shot_id):
            result.append(
                RecordingPlaybackRow(
                    shot_id=shot_id,
                    sequence=shot.sequence,
                    encoding_status="DELETED",
                    media_job_id=None if snapshot is None else snapshot.job.job_id,
                    terminal_failure=True,
                    error_code="MEDIA_SOURCE_REVOKED",
                    playback_clip=None,
                    canonical_clip=canonical_clip,
                )
            )
            continue
        playable = None
        source_terminal = False
        source_error = None
        if records.get_session_policy(manifest.session_id) == "recording-v1" and snapshot is None:
            evidence = records.get_recording_evidence(manifest.session_id, shot_id)
            if evidence.get("media_status") == "FAILED":
                source_terminal = True
                media_failure = evidence.get("media")
                source_error = (
                    media_failure.get("failure_code")
                    if isinstance(media_failure, dict)
                    else "RECORDING_MEDIA_UNAVAILABLE"
                )
        if snapshot is not None:
            _validate_playback_authority(records, jobs, manifest, snapshot.job, recording)
            if snapshot.status is ShotAnalysisJobStatus.AVAILABLE:
                try:
                    if not isinstance(snapshot.artifact_id, str):
                        raise RecordingPlaybackError(
                            "MEDIA_ARTIFACT_INVALID", "completed task has no artifact"
                        )
                    root = records.artifact_path_for_session(manifest.session_id)
                    artifact = verify_recording_playback_artifact(
                        root / "postgame" / manifest.session_id / snapshot.artifact_id,
                        snapshot.job,
                        validate_video=False,
                        continue_check=continue_check,
                    )
                except RecordingPlaybackError as exc:
                    # Output corruption revokes only this row's local playback.
                    # GET does not rewrite the winning task or hide healthy siblings.
                    result.append(
                        RecordingPlaybackRow(
                            shot_id=shot_id,
                            sequence=shot.sequence,
                            encoding_status="FAILED",
                            media_job_id=snapshot.job.job_id,
                            terminal_failure=True,
                            error_code=exc.code,
                            playback_clip=None,
                            canonical_clip=canonical_clip,
                        )
                    )
                    continue
                playable = PostgameClip(
                    session_id=manifest.session_id,
                    shot_id=shot_id,
                    postgame_manifest_sha256=manifest.manifest_sha256,
                    job_id=snapshot.job.job_id,
                    save_status=ClipSaveStatus.NOT_SAVED,
                    clip_id=session_scoped_clip_id(
                        manifest.session_id,
                        shot_id,
                        snapshot.job.job_id,
                        artifact.artifact_id,
                    ),
                    artifact_id=artifact.artifact_id,
                    local_path=artifact.media_path,
                    content_sha256=artifact.content_sha256,
                    byte_size=artifact.byte_size,
                    content_type="video/mp4",
                    sequence=shot.sequence,
                    can_save=False,
                    ownership_unavailable_reason="RAW_PLAYBACK_NOT_SAVEABLE",
                )
        if source_terminal:
            status = "FAILED"
        elif snapshot is None and canonical_clip is not None and canonical_clip.clip_id is not None:
            status = "CANONICAL_AVAILABLE"
        else:
            status = _playback_status(snapshot)
        result.append(
            RecordingPlaybackRow(
                shot_id=shot_id,
                sequence=shot.sequence,
                encoding_status=status,
                media_job_id=None if snapshot is None else snapshot.job.job_id,
                terminal_failure=(
                    source_terminal if snapshot is None else snapshot.terminal_failure
                ),
                error_code=source_error if snapshot is None else snapshot.error_code,
                playback_clip=(
                    playable
                    if snapshot is not None
                    else canonical_clip
                    if canonical_clip is not None and canonical_clip.clip_id is not None
                    else None
                ),
                canonical_clip=canonical_clip,
            )
        )
    return tuple(result)


def publish_selected_original_clips(records, jobs, manifest):
    """Join trusted selection to available original media, independently of analysis.

    Callers hold the original session media pin. This explicit mutation is used
    by the media worker and kiosk save path; read-only projections never call it.
    """
    if manifest is None or records.get_session_policy(manifest.session_id) != "legacy":
        return
    pending = {
        row.shot_id
        for row in records.list_postgame_clips(manifest.session_id)
        if row.save_status is ClipSaveStatus.SAVE_PENDING and row.can_save
    }
    if not pending:
        return
    by_shot = {}
    for job in jobs.list_session_media_jobs(manifest.session_id):
        if job.shot_id not in pending:
            continue
        if job.shot_id in by_shot:
            raise RecordingPlaybackError("MEDIA_TASK_INVENTORY_INVALID", "selected task conflicts")
        by_shot[job.shot_id] = job
    for job in by_shot.values():
        snapshot = jobs.snapshot(job.job_id)
        if snapshot.status is not ShotAnalysisJobStatus.AVAILABLE or not snapshot.artifact_id:
            continue
        if not isinstance(job.source, LegacyFrozenJobMediaSourceV1):
            raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "selected source kind changed")
        _validate_playback_authority(records, jobs, manifest, job, None)
        root = records.artifact_path_for_session(manifest.session_id)
        artifact = recording_playback_inventory(
            root / "postgame" / manifest.session_id / snapshot.artifact_id,
            job,
        )
        records.record_selected_original_clip_available(
            job,
            jobs.snapshot(job.source.source_job_id).job,
            artifact,
        )


def _playback_status(snapshot: ShotMediaEncodeJobSnapshot | None) -> str:
    if snapshot is None:
        return "NOT_SCHEDULED"
    if snapshot.status is ShotAnalysisJobStatus.AVAILABLE:
        return "AVAILABLE"
    if snapshot.terminal_failure:
        return "FAILED"
    return {
        ShotAnalysisJobStatus.PENDING: "PENDING",
        ShotAnalysisJobStatus.RUNNING: "ENCODING",
        ShotAnalysisJobStatus.FAILED_RETRYING: "FAILED_RETRYING",
    }[snapshot.status]


def validate_recording_interval_media_authority(records, job):
    """Validate one immutable committed interval while recording or after capture ends."""
    source = job.source
    state = records.media_encoding_state(job.session_id, job.shot_id)
    if (
        not isinstance(source, RecordingIntervalMediaSourceV1)
        or state["policy"] != "recording-v1"
        or state["status"] not in {"open", "ended"}
        or state["banned"]
    ):
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "recording interval authority changed")
    evidence = records.get_recording_evidence(job.session_id, job.shot_id)
    media, boundary = evidence.get("media"), evidence.get("boundary")
    if (
        hashlib.sha256(_canonical(evidence).encode()).hexdigest() != source.shot_evidence_sha256
        or evidence.get("media_status") != "RECORDED"
        or not isinstance(media, dict)
        or media.get("status") != "RECORDED"
        or not isinstance(boundary, dict)
        or boundary.get("complete") is not True
        or (
            boundary.get("source_epoch_id"),
            media.get("start_frame_id"),
            media.get("end_frame_id"),
            media.get("media_sha256"),
            Path(media.get("reference_path", "")),
            media.get("reference_sha256"),
        )
        != (
            job.source_epoch_id,
            job.start_frame_id,
            job.end_frame_id,
            job.media_sha256,
            source.reference_path,
            source.reference_sha256,
        )
    ):
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "committed shot evidence changed")
    interval = load_recording_playback_interval(job)
    capture = records.get_recording_capture_end(job.session_id)
    if capture is not None:
        cutoff = capture["cutoff"]
        if (
            cutoff is None
            or capture["source_epoch_id"] != job.source_epoch_id
            or not capture["first_frame_id"]
            <= job.start_frame_id
            <= job.end_frame_id
            <= cutoff["frame_id"]
            or boundary["ended_ms"] * 1_000_000 > cutoff["captured_at_ns"] + 500_000
            or interval.root != Path(capture["raw_manifest_path"]).parent
        ):
            raise RecordingPlaybackError(
                "MEDIA_SOURCE_REVOKED", "interval conflicts with capture cutoff"
            )
    elif state["status"] == "ended":
        frozen = records.get_recording_end_manifest(job.session_id)
        if (
            frozen is None
            or frozen["source_epoch_id"] != job.source_epoch_id
            or interval.root != Path(frozen["raw_manifest_path"]).parent
            or not any(
                row["shot_id"] == job.shot_id
                and row["payload_sha256"] == source.shot_evidence_sha256
                for row in frozen["shots"]
            )
        ):
            raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "ended interval authority missing")


def validate_recording_capture_media_authority(records, job):
    """A single complete interval may publish before the final shot inventory."""
    source = job.source
    capture = records.get_recording_capture_end(job.session_id)
    if (
        not isinstance(source, RecordingCaptureMediaSourceV1)
        or capture is None
        or records.get_session_policy(job.session_id) != "recording-v1"
        or records.load_session(job.session_id).status is not FieldSessionStatus.ENDED
        or capture["capture_end_sha256"] != source.capture_end_sha256
    ):
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "original capture authority changed")
    cutoff = capture["cutoff"]
    evidence = records.get_recording_evidence(job.session_id, job.shot_id)
    media = None if evidence is None else evidence.get("media")
    boundary = None if evidence is None else evidence.get("boundary")
    if (
        cutoff is None
        or evidence is None
        or hashlib.sha256(_canonical(evidence).encode()).hexdigest() != source.shot_evidence_sha256
        or evidence.get("media_status") != "RECORDED"
        or not isinstance(media, dict)
        or media.get("status") != "RECORDED"
        or not isinstance(boundary, dict)
        or boundary.get("complete") is not True
        or not (
            capture["first_frame_id"]
            <= job.start_frame_id
            <= job.end_frame_id
            <= cutoff["frame_id"]
        )
        or boundary["ended_ms"] * 1_000_000 > cutoff["captured_at_ns"] + 500_000
        or (
            capture["source_epoch_id"],
            boundary["source_epoch_id"],
            media["start_frame_id"],
            media["end_frame_id"],
            media["media_sha256"],
            Path(media["reference_path"]),
            media["reference_sha256"],
        )
        != (
            job.source_epoch_id,
            job.source_epoch_id,
            job.start_frame_id,
            job.end_frame_id,
            job.media_sha256,
            source.reference_path,
            source.reference_sha256,
        )
    ):
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "original shot evidence changed")
    interval = load_recording_playback_interval(job)
    if interval.root != Path(capture["raw_manifest_path"]).parent:
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "original recording root changed")


def _validate_playback_authority(records, jobs, manifest, job, recording) -> None:
    if records.owner_delete_for_shot(
        job.session_id, job.shot_id
    ) is not None or records.is_postgame_discarded(job.session_id, job.shot_id):
        raise RecordingPlaybackError(
            "MEDIA_SOURCE_REVOKED", "original shot is deleted or discarded"
        )
    if isinstance(job.source, RecordingIntervalMediaSourceV1):
        validate_recording_interval_media_authority(records, job)
        return
    if isinstance(job.source, RecordingCaptureMediaSourceV1):
        validate_recording_capture_media_authority(records, job)
        return
    if isinstance(job.source, RecordingEndMediaSourceV1):
        if recording is None or recording["manifest_sha256"] != job.source.recording_end_sha256:
            raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "recording end authority changed")
        reference = next(
            (item for item in recording["shots"] if item["shot_id"] == job.shot_id), None
        )
        evidence = records.get_recording_evidence(job.session_id, job.shot_id)
        digest = hashlib.sha256(_canonical(evidence).encode("utf-8")).hexdigest()
        media = evidence.get("media")
        boundary = evidence.get("boundary")
        if (
            reference is None
            or reference["payload_sha256"] != job.source.shot_evidence_sha256
            or digest != job.source.shot_evidence_sha256
            or evidence.get("media_status") != "RECORDED"
            or not isinstance(media, dict)
            or media.get("status") != "RECORDED"
            or not isinstance(boundary, dict)
            or not boundary.get("complete")
            or (
                boundary.get("source_epoch_id"),
                media.get("start_frame_id"),
                media.get("end_frame_id"),
                media.get("media_sha256"),
                Path(media.get("reference_path", "")),
                media.get("reference_sha256"),
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
        return
    if (
        not isinstance(job.source, LegacyFrozenJobMediaSourceV1)
        or manifest.manifest_sha256 != job.source.postgame_manifest_sha256
        or job.source.source_job_id not in manifest.expected_jobs
        or jobs.payload_sha256(job.source.source_job_id) != job.source.source_job_payload_sha256
    ):
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "legacy frozen task authority changed")
    source = jobs.snapshot(job.source.source_job_id).job
    shot = next(
        (row for row in records.list_shots(job.session_id) if row.shot_id == job.shot_id), None
    )
    media = records.get_live_shot_media(job.session_id, job.shot_id)
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
        or shot is None
        or shot.settled_ms is None
        or shot.media.source_kind != "usb_live"
        or shot.live_job_id != source.job_id
        or media is None
        or (
            media.session_id,
            media.shot_id,
            media.job_id,
            media.source_epoch_id,
            media.media_sha256,
        )
        != (
            source.session_id,
            source.shot_id,
            source.job_id,
            source.source_epoch_id,
            source.media_sha256,
        )
        or Path(media.media_path).resolve() != source.media_root.resolve()
    ):
        raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "legacy source lineage changed")


def _recover_generated_artifact(root, job, *, continue_check):
    matches = []
    for candidate in root.iterdir():
        continue_check()
        if not (
            candidate.name.startswith("recording-playback-")
            and _is_sha256(candidate.name.removeprefix("recording-playback-"))
            and candidate.is_dir()
            and not candidate.is_symlink()
        ):
            continue
        try:
            manifest = json.loads(_read_safe_small_file(candidate / _MANIFEST_NAME, None))
        except (OSError, ValueError, RecordingPlaybackError):
            continue  # Unknown/corrupt metadata is never authority to reuse bytes.
        if (
            isinstance(manifest, dict)
            and isinstance(manifest.get("job"), dict)
            and manifest["job"].get("job_id") == job.job_id
        ):
            matches.append(candidate)
    if len(matches) > 1:
        raise RecordingPlaybackError(
            "MEDIA_ARTIFACT_CONFLICT", "multiple outputs target the same job"
        )
    return (
        verify_recording_playback_artifact(matches[0], job, continue_check=continue_check)
        if matches
        else None
    )


def publish_recording_playback(
    job: ShotMediaEncodeJobV1,
    artifact_root: Path,
    *,
    authority_check: Callable[[ShotMediaEncodeJobV1], None],
    continue_check: Callable[[], None] = lambda: None,
    writer_factory: Callable[..., Any] = cv2.VideoWriter,
) -> RecordingPlaybackArtifact:
    """Stream one verified original interval into an atomic H.264 replay artifact."""

    try:
        authority_check(job)
        interval = load_recording_playback_interval(job)
        timing = _measure_timing(interval, continue_check)
    except RecordingPlaybackError:
        raise
    except RealtimeRingError as exc:
        raise RecordingPlaybackError("MEDIA_SOURCE_INVALID", str(exc)) from exc

    pending = None
    try:
        root = artifact_root.absolute()
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            raise RecordingPlaybackError("MEDIA_ARTIFACT_ROOT_UNSAFE", "artifact root is unsafe")
        recovered = _recover_generated_artifact(root, job, continue_check=continue_check)
        if recovered is not None:
            authority_check(job)
            continue_check()
            return recovered
        pending = root / f".pending-recording-playback-{job.job_id}-{uuid4().hex}"
        pending.mkdir()
    except RecordingPlaybackError:
        raise
    except Exception as exc:
        failure = RecordingPlaybackError("MEDIA_PUBLICATION_FAILED", str(exc))
        if pending is not None and pending.exists():
            try:
                shutil.rmtree(pending)
            except OSError as cleanup_error:
                failure.add_note(f"temporary playback cleanup failed: {cleanup_error}")
        raise failure from exc
    assert pending is not None
    output = pending / _OUTPUT_NAME
    try:
        _encode_interval(
            interval,
            output,
            timing["fps"],
            continue_check=continue_check,
            writer_factory=writer_factory,
        )
        before_decode = output.stat()
        output_info = _verify_mp4(
            output,
            expected_frames=interval.frame_count,
            width=interval.width,
            height=interval.height,
            fps=timing["fps"],
            continue_check=continue_check,
        )
        verified = output.stat()
        # The decoder and receipt must refer to the same bytes, not a weak timestamp cache.
        if (
            not _same_file_identity(before_decode, verified)
            or _hash_media_file(output, continue_check=continue_check) != output_info
            or not _same_file_identity(verified, output.stat())
        ):
            raise RecordingPlaybackError(
                "MEDIA_OUTPUT_INVALID", "encoded bytes changed during validation"
            )
        receipt = _GeneratedMediaReceipt(
            hashlib.sha256(_canonical(job.identity()).encode()).hexdigest(),
            output_info["content_sha256"],
            output_info["byte_size"],
            verified.st_dev,
            verified.st_ino,
            verified.st_mtime_ns,
        )
        core = {
            "job": job.identity(),
            "source_interval": {
                "media_sha256": interval.media_sha256,
                "width": interval.width,
                "height": interval.height,
                "frame_count": interval.frame_count,
                "first_captured_at_ns": timing["first_captured_at_ns"],
                "last_captured_at_ns": timing["last_captured_at_ns"],
                "fps": timing["fps"],
            },
            "output": {
                "name": _OUTPUT_NAME,
                "content_type": "video/mp4",
                "content_sha256": output_info["content_sha256"],
                "byte_size": output_info["byte_size"],
                "codec": "avc1",
                "frame_count": interval.frame_count,
                "width": interval.width,
                "height": interval.height,
                "fps": timing["fps"],
            },
        }
        artifact_id = (
            "recording-playback-" + hashlib.sha256(_canonical(core).encode("utf-8")).hexdigest()
        )
        manifest = {
            "schema_version": "recording-playback-artifact/v1",
            "artifact_id": artifact_id,
            **core,
        }
        manifest_bytes = (_canonical(manifest) + "\n").encode("utf-8")
        _write_fsynced(pending / _MANIFEST_NAME, manifest_bytes)
        _fsync_file(output)
        _fsync_directory(pending)
        authority_check(job)
        continue_check()
        destination = root / artifact_id
        if destination.exists():
            artifact = verify_recording_playback_artifact(
                destination, job, continue_check=continue_check
            )
        else:
            published_own_file = False
            try:
                os.rename(pending, destination)
                published_own_file = True
                _fsync_directory(root)
            except OSError:
                if not destination.exists():
                    raise
            artifact = (
                _reuse_generated_media(destination, job, receipt, continue_check=continue_check)
                if published_own_file
                else verify_recording_playback_artifact(
                    destination, job, continue_check=continue_check
                )
            )
        return artifact
    except RecordingPlaybackError:
        raise
    except RealtimeRingError as exc:
        raise RecordingPlaybackError("MEDIA_SOURCE_INVALID", str(exc)) from exc
    except Exception as exc:
        raise RecordingPlaybackError("MEDIA_PUBLICATION_FAILED", str(exc)) from exc
    finally:
        primary_error = sys.exc_info()[1]
        try:
            shutil.rmtree(pending)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            if primary_error is not None:
                primary_error.add_note(f"temporary playback cleanup failed: {cleanup_error}")
            else:
                raise RecordingPlaybackError(
                    "MEDIA_TEMP_CLEANUP_FAILED", "temporary playback output could not be removed"
                ) from cleanup_error


def load_recording_playback_interval(job: ShotMediaEncodeJobV1) -> FrozenCaptureInterval:
    """Validate the immutable reference for either approved original source kind."""

    reference_path = (
        job.source.reference_path
        if isinstance(
            job.source,
            (
                RecordingEndMediaSourceV1,
                RecordingCaptureMediaSourceV1,
                RecordingIntervalMediaSourceV1,
            ),
        )
        else job.media_root / "interval.json"
    )
    expected_reference_sha256 = (
        job.source.reference_sha256
        if isinstance(
            job.source,
            (
                RecordingEndMediaSourceV1,
                RecordingCaptureMediaSourceV1,
                RecordingIntervalMediaSourceV1,
            ),
        )
        else None
    )
    raw = _read_safe_small_file(reference_path, expected_reference_sha256)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RecordingPlaybackError(
            "MEDIA_REFERENCE_INVALID", "recording interval reference is invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise RecordingPlaybackError(
            "MEDIA_REFERENCE_INVALID", "interval reference is not an object"
        )
    if isinstance(
        job.source,
        (RecordingEndMediaSourceV1, RecordingCaptureMediaSourceV1, RecordingIntervalMediaSourceV1),
    ):
        expected_fields = {
            "schema_version",
            "session_id",
            "shot_id",
            "source_epoch_id",
            "width",
            "height",
            "start_frame_id",
            "end_frame_id",
            "frame_count",
            "media_sha256",
            "segment_sha256",
            "storage_kind",
            "segment_root",
        }
        if (
            set(payload) != expected_fields
            or payload.get("schema_version") != "recording-reference/v1"
            or payload.get("storage_kind") != "session_segment_reference"
            or payload.get("session_id") != job.session_id
            or payload.get("shot_id") != job.shot_id
            or type(payload.get("segment_root")) is not str
        ):
            raise RecordingPlaybackError(
                "MEDIA_REFERENCE_LINEAGE_INVALID", "recording reference lineage conflicts"
            )
        source_root = Path(payload["segment_root"])
    else:
        base_fields = {
            "schema_version",
            "source_epoch_id",
            "width",
            "height",
            "start_frame_id",
            "end_frame_id",
            "frame_count",
            "media_sha256",
            "segment_sha256",
        }
        reference_fields = base_fields | {"storage_kind", "segment_root"}
        if (
            set(payload) not in {frozenset(base_fields), frozenset(reference_fields)}
            or payload.get("schema_version") != "1.0"
            or (
                set(payload) == reference_fields
                and (
                    payload.get("storage_kind") != "session_segment_reference"
                    or type(payload.get("segment_root")) is not str
                )
            )
        ):
            raise RecordingPlaybackError(
                "MEDIA_REFERENCE_LINEAGE_INVALID", "legacy interval lineage conflicts"
            )
        source_root = (
            Path(payload["segment_root"]) if set(payload) == reference_fields else job.media_root
        )
    _validate_interval_fields(payload, job)
    source_root = _safe_real_directory(source_root, "source segment root")
    segments = payload["segment_sha256"]
    assert isinstance(segments, dict)
    for name, digest in segments.items():
        if (
            type(name) is not str
            or not name
            or Path(name).name != name
            or Path(name).is_absolute()
            or type(digest) is not str
            or not _is_sha256(digest)
        ):
            raise RecordingPlaybackError("MEDIA_REFERENCE_INVALID", "segment identity is invalid")
        path = source_root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True).parent != source_root
        ):
            raise RecordingPlaybackError("MEDIA_SOURCE_UNSAFE", "source segment is unsafe")
    interval = FrozenCaptureInterval(
        root=source_root,
        source_epoch_id=job.source_epoch_id,
        width=payload["width"],
        height=payload["height"],
        start_frame_id=job.start_frame_id,
        end_frame_id=job.end_frame_id,
        frame_count=payload["frame_count"],
        segment_sha256=dict(segments),
    )
    if interval.media_sha256 != job.media_sha256:
        raise RecordingPlaybackError(
            "MEDIA_REFERENCE_LINEAGE_INVALID", "interval content identity conflicts"
        )
    return interval


def recording_playback_inventory(
    artifact_path: Path,
    job: ShotMediaEncodeJobV1,
) -> RecordingPlaybackArtifact:
    """Validate retained metadata only; cleanup may already have detached media."""
    root = _safe_real_directory(artifact_path, "recording playback artifact")
    manifest_path = root / _MANIFEST_NAME
    raw = _read_safe_small_file(manifest_path, None)
    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RecordingPlaybackError(
            "MEDIA_ARTIFACT_INVALID", "artifact manifest is invalid"
        ) from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "artifact_id",
        "job",
        "source_interval",
        "output",
    }:
        raise RecordingPlaybackError("MEDIA_ARTIFACT_INVALID", "artifact manifest shape is invalid")
    if (
        manifest["schema_version"] != "recording-playback-artifact/v1"
        or manifest["job"] != job.identity()
        or type(manifest["artifact_id"]) is not str
        or root.name != manifest["artifact_id"]
    ):
        raise RecordingPlaybackError("MEDIA_ARTIFACT_INVALID", "artifact identity conflicts")
    core = {
        "job": manifest["job"],
        "source_interval": manifest["source_interval"],
        "output": manifest["output"],
    }
    expected_id = (
        "recording-playback-" + hashlib.sha256(_canonical(core).encode("utf-8")).hexdigest()
    )
    if manifest["artifact_id"] != expected_id:
        raise RecordingPlaybackError("MEDIA_ARTIFACT_INVALID", "artifact content key conflicts")
    source = manifest["source_interval"]
    output = manifest["output"]
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "media_sha256",
            "width",
            "height",
            "frame_count",
            "first_captured_at_ns",
            "last_captured_at_ns",
            "fps",
        }
        or not isinstance(output, dict)
        or set(output)
        != {
            "name",
            "content_type",
            "content_sha256",
            "byte_size",
            "codec",
            "frame_count",
            "width",
            "height",
            "fps",
        }
        or output.get("name") != _OUTPUT_NAME
        or output.get("content_type") != "video/mp4"
        or output.get("codec") != "avc1"
        or source.get("media_sha256") != job.media_sha256
        or source.get("frame_count") != job.end_frame_id - job.start_frame_id + 1
        or output.get("frame_count") != source.get("frame_count")
        or output.get("width") != source.get("width")
        or output.get("height") != source.get("height")
        or output.get("fps") != source.get("fps")
        or not isinstance(output.get("content_sha256"), str)
        or not _is_sha256(output["content_sha256"])
        or type(output.get("byte_size")) is not int
        or output["byte_size"] <= 0
    ):
        raise RecordingPlaybackError("MEDIA_ARTIFACT_INVALID", "artifact inventory conflicts")
    media_path = root / _OUTPUT_NAME
    return RecordingPlaybackArtifact(
        artifact_id=expected_id,
        path=root,
        media_path=media_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256=output["content_sha256"],
        byte_size=output["byte_size"],
        frame_count=source["frame_count"],
        width=source["width"],
        height=source["height"],
        fps=source["fps"],
        first_captured_at_ns=source["first_captured_at_ns"],
        last_captured_at_ns=source["last_captured_at_ns"],
    )


def verify_recording_playback_artifact(
    artifact_path: Path,
    job: ShotMediaEncodeJobV1,
    *,
    validate_video: bool = True,
    continue_check: Callable[[], None] = lambda: None,
) -> RecordingPlaybackArtifact:
    artifact = recording_playback_inventory(artifact_path, job)
    observed = (
        _verify_mp4(
            artifact.media_path,
            expected_frames=artifact.frame_count,
            width=artifact.width,
            height=artifact.height,
            fps=artifact.fps,
            continue_check=continue_check,
        )
        if validate_video
        else _hash_media_file(artifact.media_path, continue_check=continue_check)
    )
    if (
        artifact.content_sha256 != observed["content_sha256"]
        or artifact.byte_size != observed["byte_size"]
    ):
        raise RecordingPlaybackError("MEDIA_ARTIFACT_INVALID", "artifact media hash conflicts")
    return artifact


def _validate_interval_fields(payload: dict[str, Any], job: ShotMediaEncodeJobV1) -> None:
    for name in ("source_epoch_id", "media_sha256"):
        if type(payload.get(name)) is not str:
            raise RecordingPlaybackError("MEDIA_REFERENCE_INVALID", f"{name} is invalid")
    for name in ("width", "height", "start_frame_id", "end_frame_id", "frame_count"):
        if type(payload.get(name)) is not int:
            raise RecordingPlaybackError("MEDIA_REFERENCE_INVALID", f"{name} is invalid")
    if not isinstance(payload.get("segment_sha256"), dict):
        raise RecordingPlaybackError("MEDIA_REFERENCE_INVALID", "segment inventory is invalid")
    if (
        payload["source_epoch_id"] != job.source_epoch_id
        or payload["start_frame_id"] != job.start_frame_id
        or payload["end_frame_id"] != job.end_frame_id
        or payload["frame_count"] != job.end_frame_id - job.start_frame_id + 1
        or payload["media_sha256"] != job.media_sha256
        or payload["width"] <= 0
        or payload["height"] <= 0
    ):
        raise RecordingPlaybackError(
            "MEDIA_REFERENCE_LINEAGE_INVALID", "interval identity conflicts"
        )


def _measure_timing(
    interval: FrozenCaptureInterval, continue_check: Callable[[], None]
) -> dict[str, int | float]:
    count = 0
    first = None
    previous = None
    minimum_delta = None
    maximum_delta = None
    for frame in interval.frames(before_segment=continue_check):
        continue_check()
        if previous is not None and frame.captured_at_ns <= previous:
            raise RecordingPlaybackError(
                "MEDIA_TIMING_INVALID", "capture timestamps are not strict"
            )
        if previous is not None:
            delta = frame.captured_at_ns - previous
            minimum_delta = delta if minimum_delta is None else min(minimum_delta, delta)
            maximum_delta = delta if maximum_delta is None else max(maximum_delta, delta)
        first = frame.captured_at_ns if first is None else first
        previous = frame.captured_at_ns
        count += 1
    if count != interval.frame_count or first is None or previous is None or count < 2:
        raise RecordingPlaybackError("MEDIA_TIMING_INVALID", "capture timing is incomplete")
    fps = (count - 1) * 1_000_000_000 / (previous - first)
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        raise RecordingPlaybackError("MEDIA_TIMING_INVALID", "derived capture cadence is invalid")
    average_delta = (previous - first) / (count - 1)
    if (
        minimum_delta is None
        or maximum_delta is None
        or minimum_delta < average_delta * 0.5
        or maximum_delta > average_delta * 2.0
    ):
        raise RecordingPlaybackError("MEDIA_TIMING_INVALID", "capture cadence is discontinuous")
    return {
        "first_captured_at_ns": first,
        "last_captured_at_ns": previous,
        "fps": round(fps, 6),
    }


def _encode_interval(
    interval: FrozenCaptureInterval,
    output: Path,
    fps: float,
    *,
    continue_check: Callable[[], None],
    writer_factory: Callable[..., Any],
) -> None:
    writer = writer_factory(
        str(output), cv2.VideoWriter_fourcc(*"avc1"), fps, (interval.width, interval.height)
    )
    try:
        if not writer.isOpened():
            raise RecordingPlaybackError(
                "MEDIA_H264_ENCODER_UNAVAILABLE", "H.264 (avc1) encoder is unavailable"
            )
        count = 0
        for frame in interval.frames(before_segment=continue_check):
            continue_check()
            image = cv2.imdecode(np.frombuffer(frame.mjpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RecordingPlaybackError("MEDIA_FRAME_UNDECODABLE", "source JPEG is invalid")
            if image.shape[:2] != (interval.height, interval.width):
                raise RecordingPlaybackError("MEDIA_GEOMETRY_INVALID", "source geometry changed")
            writer.write(image)
            count += 1
        if count != interval.frame_count:
            raise RecordingPlaybackError("MEDIA_FRAME_COUNT_INVALID", "source frame count changed")
    finally:
        writer.release()


def _verify_mp4(
    path: Path,
    *,
    expected_frames: int,
    width: int,
    height: int,
    fps: float,
    continue_check: Callable[[], None] = lambda: None,
) -> dict[str, int | str]:
    if (
        type(expected_frames) is not int
        or type(width) is not int
        or type(height) is not int
        or type(fps) not in {int, float}
        or expected_frames <= 0
        or width <= 0
        or height <= 0
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise RecordingPlaybackError("MEDIA_ARTIFACT_INVALID", "media inventory types are invalid")
    observed = _hash_media_file(path, continue_check=continue_check)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RecordingPlaybackError("MEDIA_OUTPUT_INVALID", "encoded media cannot be opened")
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).lower()
        if codec not in {"avc1", "h264", "x264"}:
            raise RecordingPlaybackError("MEDIA_CODEC_INVALID", "encoded video track is not AVC")
        count = 0
        while True:
            continue_check()
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape[:2] != (height, width):
                raise RecordingPlaybackError("MEDIA_GEOMETRY_INVALID", "encoded geometry changed")
            count += 1
        observed_fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if count != expected_frames:
        raise RecordingPlaybackError("MEDIA_FRAME_COUNT_INVALID", "encoded frame count conflicts")
    if not math.isfinite(observed_fps) or abs(observed_fps - fps) > max(0.5, fps * 0.05):
        raise RecordingPlaybackError("MEDIA_TIMING_INVALID", "encoded cadence conflicts")
    return observed


def _hash_media_file(
    path: Path, *, continue_check: Callable[[], None] = lambda: None
) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise RecordingPlaybackError("MEDIA_OUTPUT_INVALID", "encoded media is missing")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                continue_check()
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise RecordingPlaybackError(
            "MEDIA_OUTPUT_INVALID", "encoded media cannot be read"
        ) from exc
    if size <= 0:
        raise RecordingPlaybackError("MEDIA_OUTPUT_INVALID", "encoded media is empty")
    return {"content_sha256": digest.hexdigest(), "byte_size": size}


def _read_safe_small_file(path: Path, expected_sha256: str | None) -> bytes:
    descriptor = None
    try:
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise RecordingPlaybackError("MEDIA_REFERENCE_UNSAFE", "metadata path is unsafe")
        original = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or not _same_file_identity(original, before, path_and_handle=True)
                or before.st_size > _REFERENCE_LIMIT_BYTES
            ):
                raise RecordingPlaybackError("MEDIA_REFERENCE_UNSAFE", "metadata path is unsafe")
            raw = stream.read(_REFERENCE_LIMIT_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = path.lstat()
        if not _same_file_identity(before, after) or not _same_file_identity(
            after, current, path_and_handle=True
        ):
            raise RecordingPlaybackError("MEDIA_REFERENCE_UNSAFE", "metadata changed while reading")
    except OSError as exc:
        raise RecordingPlaybackError("MEDIA_REFERENCE_UNSAFE", "metadata cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > _REFERENCE_LIMIT_BYTES:
        raise RecordingPlaybackError("MEDIA_REFERENCE_UNSAFE", "metadata exceeds size limit")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RecordingPlaybackError("MEDIA_REFERENCE_HASH_MISMATCH", "metadata hash changed")
    return raw


def _safe_real_directory(path: Path, label: str) -> Path:
    try:
        if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
            raise RecordingPlaybackError("MEDIA_SOURCE_UNSAFE", f"{label} is unsafe")
    except OSError as exc:
        raise RecordingPlaybackError("MEDIA_SOURCE_UNSAFE", f"{label} is unavailable") from exc
    if not path.is_dir():
        raise RecordingPlaybackError("MEDIA_SOURCE_UNSAFE", f"{label} is not a directory")
    return path


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    # Windows _commit/FlushFileBuffers requires write access to this owned output.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
