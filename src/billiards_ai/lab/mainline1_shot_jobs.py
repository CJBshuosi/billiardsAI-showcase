"""Durable immutable media and work records for Mainline 1 shot analysis."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from billiards_ai.lab.mainline1_realtime_ring import (
    FrozenCaptureInterval,
    RealtimeRingError,
)
from billiards_ai.lab.mainline1_session_recording import RecordedShotInterval

_LEASE_SECONDS = 120
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERVAL_MANIFEST = "interval.json"
_QUARANTINE_MANIFEST = "quarantine.json"


class ShotAnalysisJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AVAILABLE = "available"
    FAILED_RETRYING = "failed_retrying"


@dataclass(frozen=True, slots=True)
class RecordingEndMediaSourceV1:
    kind: Literal["recording_end"]
    recording_end_sha256: str
    shot_evidence_sha256: str
    reference_path: Path
    reference_sha256: str

    def __post_init__(self) -> None:
        if self.kind != "recording_end" or not isinstance(self.reference_path, Path):
            raise ValueError("recording media source kind is invalid")
        _require_sha256(self.recording_end_sha256, "recording end")
        _require_sha256(self.shot_evidence_sha256, "shot evidence")
        _require_sha256(self.reference_sha256, "recording reference")


@dataclass(frozen=True, slots=True)
class RecordingCaptureMediaSourceV1:
    kind: Literal["recording_capture"]
    capture_end_sha256: str
    shot_evidence_sha256: str
    reference_path: Path
    reference_sha256: str

    def __post_init__(self) -> None:
        if self.kind != "recording_capture" or not isinstance(self.reference_path, Path):
            raise ValueError("recording capture media source kind is invalid")
        _require_sha256(self.capture_end_sha256, "capture end")
        _require_sha256(self.shot_evidence_sha256, "shot evidence")
        _require_sha256(self.reference_sha256, "recording reference")


@dataclass(frozen=True, slots=True)
class RecordingIntervalMediaSourceV1:
    """Complete committed shot evidence, independent of the later match cutoff."""

    kind: Literal["recording_interval"]
    shot_evidence_sha256: str
    reference_path: Path
    reference_sha256: str

    def __post_init__(self) -> None:
        if self.kind != "recording_interval" or not isinstance(self.reference_path, Path):
            raise ValueError("recording interval media source kind is invalid")
        _require_sha256(self.shot_evidence_sha256, "shot evidence")
        _require_sha256(self.reference_sha256, "recording reference")


@dataclass(frozen=True, slots=True)
class LegacyFrozenJobMediaSourceV1:
    kind: Literal["legacy_frozen_job"]
    postgame_manifest_sha256: str
    source_job_id: str
    source_job_payload_sha256: str

    def __post_init__(self) -> None:
        if (
            self.kind != "legacy_frozen_job"
            or type(self.source_job_id) is not str
            or not self.source_job_id
        ):
            raise ValueError("legacy media source identity is invalid")
        _require_sha256(self.postgame_manifest_sha256, "postgame manifest")
        _require_sha256(self.source_job_payload_sha256, "source job payload")


type ShotMediaEncodeSourceV1 = (
    RecordingEndMediaSourceV1
    | RecordingCaptureMediaSourceV1
    | RecordingIntervalMediaSourceV1
    | LegacyFrozenJobMediaSourceV1
)


@dataclass(frozen=True, slots=True)
class ShotMediaEncodeJobV1:
    contract: Literal["recording-encode-job/v1"]
    job_id: str
    session_id: str
    shot_id: str
    source_epoch_id: str
    start_frame_id: int
    end_frame_id: int
    media_root: Path
    media_sha256: str
    source: ShotMediaEncodeSourceV1
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.contract != "recording-encode-job/v1":
            raise ValueError("shot media encode contract is invalid")
        if any(
            type(value) is not str or not value
            for value in (self.job_id, self.session_id, self.shot_id, self.source_epoch_id)
        ) or not isinstance(self.media_root, Path):
            raise ValueError("shot media encode identities must not be empty")
        if (
            type(self.start_frame_id) is not int
            or type(self.end_frame_id) is not int
            or self.start_frame_id < 0
            or self.end_frame_id < self.start_frame_id
        ):
            raise ValueError("shot media encode frame range is invalid")
        _require_sha256(self.media_sha256, "media")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 20:
            raise ValueError("shot media encode max attempts is invalid")
        if not isinstance(
            self.source,
            (
                RecordingEndMediaSourceV1,
                RecordingCaptureMediaSourceV1,
                RecordingIntervalMediaSourceV1,
                LegacyFrozenJobMediaSourceV1,
            ),
        ):
            raise ValueError("shot media encode source is invalid")
        if isinstance(
            self.source,
            (
                RecordingEndMediaSourceV1,
                RecordingCaptureMediaSourceV1,
                RecordingIntervalMediaSourceV1,
            ),
        ) and (self.source.reference_path != self.media_root / _INTERVAL_MANIFEST):
            raise ValueError("recording media reference path is invalid")
        if self.job_id != _shot_media_encode_job_id(self):
            raise ValueError("shot media encode job ID is invalid")

    def identity(self) -> dict[str, object]:
        """Return the immutable, JSON-compatible media worker identity."""

        return _media_job_payload(self)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        shot_id: str,
        source_epoch_id: str,
        start_frame_id: int,
        end_frame_id: int,
        media_root: Path,
        media_sha256: str,
        source: ShotMediaEncodeSourceV1,
        max_attempts: int = 3,
    ) -> ShotMediaEncodeJobV1:
        values = dict(
            contract="recording-encode-job/v1",
            job_id="",
            session_id=session_id,
            shot_id=shot_id,
            source_epoch_id=source_epoch_id,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            media_root=media_root,
            media_sha256=media_sha256,
            source=source,
            max_attempts=max_attempts,
        )
        pending = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(pending, name, value)
        values["job_id"] = _shot_media_encode_job_id(pending)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ShotAnalysisJob:
    job_id: str
    session_id: str
    shot_id: str
    source_epoch_id: str
    start_frame_id: int
    end_frame_id: int
    media_root: Path
    media_sha256: str
    capture_setup_id: str
    calibration_artifact_id: str
    config_sha256: str
    model_package_id: str
    identity_training_package_id: str | None = None
    identity_training_package_sha256: str | None = None

    def __post_init__(self) -> None:
        identities = (
            self.job_id,
            self.session_id,
            self.shot_id,
            self.source_epoch_id,
            self.capture_setup_id,
            self.calibration_artifact_id,
            self.model_package_id,
        )
        if any(not value for value in identities):
            raise ValueError("shot analysis job identities must not be empty")
        if self.start_frame_id < 0 or self.end_frame_id < self.start_frame_id:
            raise ValueError("shot analysis job frame range is invalid")
        _require_sha256(self.media_sha256, "media")
        _require_sha256(self.config_sha256, "config")
        if (self.identity_training_package_id is None) != (
            self.identity_training_package_sha256 is None
        ):
            raise ValueError("identity training package identity must be complete")
        if self.identity_training_package_sha256 is not None:
            _require_sha256(self.identity_training_package_sha256, "identity training package")

    def identity(self) -> dict[str, object]:
        """Return the immutable, JSON-compatible worker identity."""

        return _canonical_job_payload(self)


@dataclass(frozen=True, slots=True)
class EnqueueShotAnalysisJobResult:
    job: ShotAnalysisJob | ShotMediaEncodeJobV1
    status: ShotAnalysisJobStatus
    created: bool


@dataclass(frozen=True, slots=True)
class ShotAnalysisClaim:
    """A leased job plus its unforgeable, per-attempt completion fence."""

    job: ShotAnalysisJob
    worker_id: str
    lease_token: str
    attempt_count: int
    lease_expires_at_utc: datetime


@dataclass(frozen=True, slots=True)
class ShotAnalysisJobSnapshot:
    job: ShotAnalysisJob
    status: ShotAnalysisJobStatus
    artifact_id: str | None


@dataclass(frozen=True, slots=True)
class ShotMediaEncodeClaim:
    job: ShotMediaEncodeJobV1
    worker_id: str
    lease_token: str
    attempt_count: int
    lease_expires_at_utc: datetime


@dataclass(frozen=True, slots=True)
class ShotMediaEncodeJobSnapshot:
    job: ShotMediaEncodeJobV1
    status: ShotAnalysisJobStatus
    artifact_id: str | None
    terminal_failure: bool
    error_code: str | None
    failure_reason: str | None
    attempt_count: int


@dataclass(frozen=True, slots=True)
class ShotMaterializationQuarantine:
    root: Path
    reason: str
    manifest_sha256: str
    observed_segment_sha256: dict[str, str]


class SQLiteShotAnalysisJobStore:
    """SQLite-backed, restart-safe queue with expiring worker leases."""

    def __init__(self, database: Path) -> None:
        self._database = database
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shot_analysis_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'available', 'failed_retrying')
                    ),
                    claimed_by TEXT,
                    claimed_at_utc TEXT,
                    artifact_id TEXT,
                    failure_reason TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS ix_shot_analysis_jobs_claim
                ON shot_analysis_jobs(status, claimed_at_utc);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(shot_analysis_jobs)")
            }
            additions = {
                "lease_token": "TEXT",
                "next_attempt_at_utc": "TEXT",
                "job_kind": "TEXT",
                "terminal_failure": "INTEGER NOT NULL DEFAULT 0",
                "error_code": "TEXT",
                "max_attempts": "INTEGER",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE shot_analysis_jobs ADD COLUMN {name} {definition}"
                    )

            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_shot_jobs_kind_claim "
                "ON shot_analysis_jobs(job_kind,terminal_failure,status,next_attempt_at_utc)"
            )

    @classmethod
    def open_existing(cls, database: Path):
        """Recover an existing current-schema queue without creating or migrating it."""
        instance = object.__new__(cls)
        instance._database = database
        instance._existing_only = True
        with closing(instance._connect()) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(shot_analysis_jobs)")
            }
            if columns != {
                "job_id",
                "payload_json",
                "payload_sha256",
                "status",
                "claimed_by",
                "claimed_at_utc",
                "artifact_id",
                "failure_reason",
                "attempt_count",
                "lease_token",
                "next_attempt_at_utc",
                "job_kind",
                "terminal_failure",
                "error_code",
                "max_attempts",
            }:
                raise ValueError("original jobs schema is unavailable")
        return instance

    def enqueue(self, job: ShotAnalysisJob | ShotMediaEncodeJobV1) -> EnqueueShotAnalysisJobResult:
        payload_json = _canonical_json(_queue_job_payload(job))
        payload_sha256 = _sha256_bytes(payload_json.encode("utf-8"))
        job_kind = "media" if isinstance(job, ShotMediaEncodeJobV1) else "analysis"
        max_attempts = job.max_attempts if isinstance(job, ShotMediaEncodeJobV1) else None
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT payload_json, payload_sha256, status, job_kind "
                    "FROM shot_analysis_jobs WHERE job_id = ?",
                    (job.job_id,),
                ).fetchone()
                if existing is not None:
                    _verify_stored_payload(existing[0], existing[1])
                    if (
                        existing[1] != payload_sha256
                        or existing[0] != payload_json
                        or (existing[3] or "analysis") != job_kind
                    ):
                        raise ValueError("shot analysis job content key conflict")
                    connection.commit()
                    return EnqueueShotAnalysisJobResult(
                        job=job,
                        status=ShotAnalysisJobStatus(existing[2]),
                        created=False,
                    )
                connection.execute(
                    "INSERT INTO shot_analysis_jobs "
                    "(job_id,payload_json,payload_sha256,status,job_kind,max_attempts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        job.job_id,
                        payload_json,
                        payload_sha256,
                        ShotAnalysisJobStatus.PENDING.value,
                        job_kind,
                        max_attempts,
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return EnqueueShotAnalysisJobResult(
            job=job,
            status=ShotAnalysisJobStatus.PENDING,
            created=True,
        )

    def enqueue_media(self, job: ShotMediaEncodeJobV1) -> EnqueueShotAnalysisJobResult:
        if not isinstance(job, ShotMediaEncodeJobV1):
            raise TypeError("media enqueue requires a shot media encode job")
        return self.enqueue(job)

    def claim(self, worker_id: str, now_utc: datetime) -> ShotAnalysisJob | None:
        if not worker_id:
            raise ValueError("shot analysis claim requires a worker ID")
        now = _utc_text(now_utc)
        stale_before = _utc_text(now_utc - timedelta(seconds=_LEASE_SECONDS))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT job_id, payload_json, payload_sha256 FROM shot_analysis_jobs "
                    "WHERE COALESCE(job_kind,'analysis')='analysis' AND ("
                    "status IN (?, ?) OR (status = ? AND claimed_at_utc < ?)) "
                    "ORDER BY attempt_count, rowid LIMIT 1",
                    (
                        ShotAnalysisJobStatus.PENDING.value,
                        ShotAnalysisJobStatus.FAILED_RETRYING.value,
                        ShotAnalysisJobStatus.RUNNING.value,
                        stale_before,
                    ),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                _verify_stored_payload(row[1], row[2])
                job = _analysis_job_from_payload(json.loads(row[1]))
                connection.execute(
                    "UPDATE shot_analysis_jobs SET status = ?, claimed_by = ?, "
                    "claimed_at_utc = ?, artifact_id = NULL, failure_reason = NULL, "
                    "attempt_count = attempt_count + 1, lease_token = ? WHERE job_id = ?",
                    (ShotAnalysisJobStatus.RUNNING.value, worker_id, now, uuid4().hex, row[0]),
                )
                connection.commit()
                return job
            except BaseException:
                connection.rollback()
                raise

    def claim_for_session(
        self,
        worker_id: str,
        session_id: str,
        now_utc: datetime,
        *,
        expected_job_ids: tuple[str, ...],
    ) -> ShotAnalysisClaim | None:
        """Claim only a due job belonging to one frozen ended-session manifest.

        The field-store manifest is the authority for eligibility; this queue
        method adds a second session fence so a worker can never drain another
        session while resuming after a restart.
        """
        if not worker_id or not session_id:
            raise ValueError("session-scoped claim requires worker and session IDs")
        if not expected_job_ids:
            return None
        expected = set(expected_job_ids)
        now = _utc_text(now_utc)
        stale_before = _utc_text(now_utc - timedelta(seconds=_LEASE_SECONDS))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Restrict by frozen IDs before decoding any queue payload. Foreign
                # corrupt due rows are irrelevant to this session. Chunk below the
                # portable SQLite variable limit; never truncate expected membership.
                rows = []
                expected_ids = sorted(expected)
                for offset in range(0, len(expected_ids), 800):
                    chunk = expected_ids[offset : offset + 800]
                    placeholders = ",".join("?" for _ in chunk)
                    rows.extend(
                        connection.execute(
                            # Only placeholders are generated; requested IDs remain bound.
                            "SELECT job_id, payload_json, "  # nosec B608
                            "payload_sha256, attempt_count, rowid "
                            "FROM shot_analysis_jobs WHERE COALESCE(job_kind,'analysis')="
                            "'analysis' AND job_id IN (" + placeholders + ") AND "
                            "(status = ? OR (status = ? AND (next_attempt_at_utc IS NULL OR "
                            "next_attempt_at_utc <= ?)) OR (status = ? AND claimed_at_utc < ?))",
                            (
                                *chunk,
                                ShotAnalysisJobStatus.PENDING.value,
                                ShotAnalysisJobStatus.FAILED_RETRYING.value,
                                now,
                                ShotAnalysisJobStatus.RUNNING.value,
                                stale_before,
                            ),
                        ).fetchall()
                    )
                rows.sort(key=lambda row: (row[3], row[4]))
                selected = None
                for row in rows:
                    _verify_stored_payload(row[1], row[2])
                    candidate = _analysis_job_from_payload(json.loads(row[1]))
                    if candidate.session_id == session_id and candidate.job_id in expected:
                        selected = (row, candidate)
                        break
                if selected is None:
                    connection.commit()
                    return None
                row, job = selected
                token = uuid4().hex
                attempt = int(row[3]) + 1
                connection.execute(
                    "UPDATE shot_analysis_jobs SET status=?, claimed_by=?, claimed_at_utc=?, "
                    "lease_token=?, attempt_count=?, artifact_id=NULL, failure_reason=NULL "
                    "WHERE job_id=?",
                    (
                        ShotAnalysisJobStatus.RUNNING.value,
                        worker_id,
                        now,
                        token,
                        attempt,
                        job.job_id,
                    ),
                )
                connection.commit()
                return ShotAnalysisClaim(
                    job=job,
                    worker_id=worker_id,
                    lease_token=token,
                    attempt_count=attempt,
                    lease_expires_at_utc=now_utc + timedelta(seconds=_LEASE_SECONDS),
                )
            except BaseException:
                connection.rollback()
                raise

    def claim_media_for_session(
        self,
        worker_id: str,
        session_id: str,
        now_utc: datetime,
        *,
        expected_job_ids: tuple[str, ...] | None = None,
    ) -> ShotMediaEncodeClaim | None:
        if not worker_id or not session_id:
            raise ValueError("media claim requires worker and session IDs")
        expected = None if expected_job_ids is None else set(expected_job_ids)
        if expected is not None and not expected:
            return None
        now = _utc_text(now_utc)
        stale_before = _utc_text(now_utc - timedelta(seconds=_LEASE_SECONDS))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "UPDATE shot_analysis_jobs SET status=?,terminal_failure=1,"
                    "error_code=COALESCE(error_code,'MEDIA_LEASE_EXHAUSTED'),"
                    "failure_reason=COALESCE(failure_reason,'media worker lease expired at "
                    "retry limit'),next_attempt_at_utc=NULL,claimed_by=NULL,"
                    "claimed_at_utc=NULL,lease_token=NULL WHERE job_kind='media' "
                    "AND terminal_failure=0 AND status=? AND claimed_at_utc<? "
                    "AND max_attempts IS NOT NULL AND attempt_count>=max_attempts",
                    (
                        ShotAnalysisJobStatus.FAILED_RETRYING.value,
                        ShotAnalysisJobStatus.RUNNING.value,
                        stale_before,
                    ),
                )
                rows = connection.execute(
                    "SELECT job_id,payload_json,payload_sha256,attempt_count,rowid "
                    "FROM shot_analysis_jobs WHERE job_kind='media' AND terminal_failure=0 "
                    "AND max_attempts IS NOT NULL AND attempt_count<max_attempts "
                    "AND (status=? OR (status=? AND (next_attempt_at_utc IS NULL OR "
                    "next_attempt_at_utc<=?)) OR (status=? AND claimed_at_utc<?)) "
                    "ORDER BY attempt_count,rowid",
                    (
                        ShotAnalysisJobStatus.PENDING.value,
                        ShotAnalysisJobStatus.FAILED_RETRYING.value,
                        now,
                        ShotAnalysisJobStatus.RUNNING.value,
                        stale_before,
                    ),
                ).fetchall()
                selected = None
                for row in rows:
                    if expected is not None and row[0] not in expected:
                        continue
                    _verify_stored_payload(row[1], row[2])
                    candidate = _media_job_from_payload(json.loads(row[1]))
                    if candidate.session_id == session_id:
                        selected = (row, candidate)
                        break
                if selected is None:
                    connection.commit()
                    return None
                row, job = selected
                token = uuid4().hex
                attempt = int(row[3]) + 1
                connection.execute(
                    "UPDATE shot_analysis_jobs SET status=?,claimed_by=?,claimed_at_utc=?,"
                    "lease_token=?,attempt_count=?,artifact_id=NULL,failure_reason=NULL,"
                    "error_code=NULL WHERE job_id=? AND job_kind='media' AND terminal_failure=0",
                    (
                        ShotAnalysisJobStatus.RUNNING.value,
                        worker_id,
                        now,
                        token,
                        attempt,
                        job.job_id,
                    ),
                )
                connection.commit()
                return ShotMediaEncodeClaim(
                    job=job,
                    worker_id=worker_id,
                    lease_token=token,
                    attempt_count=attempt,
                    lease_expires_at_utc=now_utc + timedelta(seconds=_LEASE_SECONDS),
                )
            except BaseException:
                connection.rollback()
                raise

    def complete_claim(
        self,
        claim: ShotAnalysisClaim | ShotMediaEncodeClaim,
        artifact_id: str,
        *,
        lease_now_utc: Callable[[], datetime] | None = None,
    ) -> bool:
        """Commit only the current lease generation; stale workers lose safely."""
        if not artifact_id:
            raise ValueError("shot completion requires an artifact ID")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json, payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (claim.job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("shot analysis job was not found")
            _verify_stored_payload(row[0], row[1])
            if _queue_job_from_payload(json.loads(row[0])) != claim.job:
                raise ValueError("shot job claim payload conflicts")
            job_kind = "media" if isinstance(claim, ShotMediaEncodeClaim) else "analysis"
            if lease_now_utc is not None and lease_now_utc() >= claim.lease_expires_at_utc:
                connection.rollback()
                return False
            cursor = connection.execute(
                "UPDATE shot_analysis_jobs SET status=?, artifact_id=?, claimed_by=NULL, "
                "claimed_at_utc=NULL, lease_token=NULL, failure_reason=NULL, "
                "next_attempt_at_utc=NULL,error_code=NULL,terminal_failure=0 "
                "WHERE job_id=? AND status=? AND claimed_by=? AND lease_token=? "
                "AND attempt_count=? AND COALESCE(job_kind,'analysis')=?",
                (
                    ShotAnalysisJobStatus.AVAILABLE.value,
                    artifact_id,
                    claim.job.job_id,
                    ShotAnalysisJobStatus.RUNNING.value,
                    claim.worker_id,
                    claim.lease_token,
                    claim.attempt_count,
                    job_kind,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def renew_media_claim(
        self, claim: ShotMediaEncodeClaim, now_utc: datetime
    ) -> ShotMediaEncodeClaim | None:
        """Extend only the current media lease generation; stale workers lose safely."""

        now = _utc_text(now_utc)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (claim.job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("shot media encode job was not found")
            _verify_stored_payload(row[0], row[1])
            if _media_job_from_payload(json.loads(row[0])) != claim.job:
                raise ValueError("shot media claim payload conflicts")
            cursor = connection.execute(
                "UPDATE shot_analysis_jobs SET claimed_at_utc=? WHERE job_id=? AND status=? "
                "AND claimed_by=? AND lease_token=? AND attempt_count=? "
                "AND job_kind='media' AND terminal_failure=0",
                (
                    now,
                    claim.job.job_id,
                    ShotAnalysisJobStatus.RUNNING.value,
                    claim.worker_id,
                    claim.lease_token,
                    claim.attempt_count,
                ),
            )
            connection.commit()
        if cursor.rowcount != 1:
            return None
        return replace(claim, lease_expires_at_utc=now_utc + timedelta(seconds=_LEASE_SECONDS))

    def fail_claim(
        self,
        claim: ShotAnalysisClaim,
        reason: str,
        *,
        now_utc: datetime,
        retry_delay: timedelta = timedelta(seconds=30),
    ) -> bool:
        """Persist a bounded retry due-time, fenced to the active lease."""
        if not reason or retry_delay.total_seconds() < 0:
            raise ValueError("retrying failure requires a reason and non-negative delay")
        due = _utc_text(now_utc + retry_delay)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json, payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (claim.job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("shot analysis job was not found")
            _verify_stored_payload(row[0], row[1])
            if _analysis_job_from_payload(json.loads(row[0])) != claim.job:
                raise ValueError("shot analysis claim payload conflicts")
            cursor = connection.execute(
                "UPDATE shot_analysis_jobs SET status=?, failure_reason=?, next_attempt_at_utc=?, "
                "claimed_by=NULL, claimed_at_utc=NULL, lease_token=NULL "
                "WHERE job_id=? AND status=? AND claimed_by=? AND lease_token=? "
                "AND attempt_count=? AND COALESCE(job_kind,'analysis')='analysis'",
                (
                    ShotAnalysisJobStatus.FAILED_RETRYING.value,
                    reason,
                    due,
                    claim.job.job_id,
                    ShotAnalysisJobStatus.RUNNING.value,
                    claim.worker_id,
                    claim.lease_token,
                    claim.attempt_count,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def fail_media_claim(
        self,
        claim: ShotMediaEncodeClaim,
        reason: str,
        *,
        error_code: str,
        now_utc: datetime,
        retry_delay: timedelta = timedelta(seconds=30),
    ) -> bool:
        if not reason or not error_code or len(error_code) > 128 or retry_delay.total_seconds() < 0:
            raise ValueError("media failure requires a typed reason and non-negative delay")
        terminal = claim.attempt_count >= claim.job.max_attempts
        due = None if terminal else _utc_text(now_utc + retry_delay)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (claim.job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("shot media encode job was not found")
            _verify_stored_payload(row[0], row[1])
            if _media_job_from_payload(json.loads(row[0])) != claim.job:
                raise ValueError("shot media claim payload conflicts")
            cursor = connection.execute(
                "UPDATE shot_analysis_jobs SET status=?,failure_reason=?,error_code=?,"
                "next_attempt_at_utc=?,terminal_failure=?,claimed_by=NULL,claimed_at_utc=NULL,"
                "lease_token=NULL WHERE job_id=? AND status=? AND claimed_by=? AND "
                "lease_token=? AND attempt_count=? AND job_kind='media'",
                (
                    ShotAnalysisJobStatus.FAILED_RETRYING.value,
                    reason,
                    error_code,
                    due,
                    int(terminal),
                    claim.job.job_id,
                    ShotAnalysisJobStatus.RUNNING.value,
                    claim.worker_id,
                    claim.lease_token,
                    claim.attempt_count,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_media_claim(self, claim: ShotMediaEncodeClaim) -> bool:
        """Release a paused media attempt without consuming its bounded retry budget."""

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (claim.job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("shot media encode job was not found")
            _verify_stored_payload(row[0], row[1])
            if _media_job_from_payload(json.loads(row[0])) != claim.job:
                raise ValueError("shot media claim payload conflicts")
            cursor = connection.execute(
                "UPDATE shot_analysis_jobs SET status=?,claimed_by=NULL,claimed_at_utc=NULL,"
                "lease_token=NULL,attempt_count=attempt_count-1,artifact_id=NULL,"
                "failure_reason=NULL,error_code=NULL,next_attempt_at_utc=NULL "
                "WHERE job_id=? AND status=? AND claimed_by=? AND lease_token=? "
                "AND attempt_count=? AND job_kind='media' AND terminal_failure=0",
                (
                    ShotAnalysisJobStatus.PENDING.value,
                    claim.job.job_id,
                    ShotAnalysisJobStatus.RUNNING.value,
                    claim.worker_id,
                    claim.lease_token,
                    claim.attempt_count,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def terminal_media_claim(
        self, claim: ShotMediaEncodeClaim, reason: str, *, error_code: str
    ) -> bool:
        """Fence permanently revoked source work without inventing a new queue status."""

        if not reason or not error_code or len(error_code) > 128:
            raise ValueError("terminal media failure requires a typed reason")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (claim.job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("shot media encode job was not found")
            _verify_stored_payload(row[0], row[1])
            if _media_job_from_payload(json.loads(row[0])) != claim.job:
                raise ValueError("shot media claim payload conflicts")
            cursor = connection.execute(
                "UPDATE shot_analysis_jobs SET status=?,terminal_failure=1,failure_reason=?,"
                "error_code=?,next_attempt_at_utc=NULL,claimed_by=NULL,claimed_at_utc=NULL,"
                "lease_token=NULL WHERE job_id=? AND status=? AND claimed_by=? AND "
                "lease_token=? AND attempt_count=? AND job_kind='media'",
                (
                    ShotAnalysisJobStatus.FAILED_RETRYING.value,
                    reason,
                    error_code,
                    claim.job.job_id,
                    ShotAnalysisJobStatus.RUNNING.value,
                    claim.worker_id,
                    claim.lease_token,
                    claim.attempt_count,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def list_session_jobs(self, session_id: str) -> tuple[ShotAnalysisJob, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs "
                "WHERE COALESCE(job_kind,'analysis')='analysis' ORDER BY rowid"
            ).fetchall()
        jobs = []
        for row in rows:
            _verify_stored_payload(row[0], row[1])
            job = _analysis_job_from_payload(json.loads(row[0]))
            if job.session_id == session_id:
                jobs.append(job)
        return tuple(jobs)

    def session_media_progress(self, session_id: str) -> dict:
        """One closed metadata snapshot; no media files or per-job connections."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json,payload_sha256,status,attempt_count,"
                "terminal_failure,error_code "
                "FROM shot_analysis_jobs WHERE job_kind='media' "
                "AND json_extract(payload_json,'$.session_id')=? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        items = []
        for raw, digest, status, attempts, terminal, error in rows:
            _verify_stored_payload(raw, digest)
            job = _media_job_from_payload(json.loads(raw))
            if job.session_id != session_id:
                raise ValueError("media progress session identity changed")
            items.append(
                dict(
                    job_id=job.job_id,
                    shot_id=job.shot_id,
                    status=status,
                    attempts=attempts,
                    terminal_failure=bool(terminal),
                    error_code=error,
                )
            )
        return {
            "generated_count": sum(item["status"] == "available" for item in items),
            "encoding_count": sum(item["status"] == "running" for item in items),
            "queued_count": sum(item["status"] == "pending" for item in items),
            "retrying_count": sum(
                item["status"] == "failed_retrying" and not item["terminal_failure"]
                for item in items
            ),
            "failed_count": sum(item["terminal_failure"] for item in items),
            "items": items,
        }

    def list_session_media_jobs(self, session_id: str) -> tuple[ShotMediaEncodeJobV1, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs "
                "WHERE job_kind='media' ORDER BY rowid"
            ).fetchall()
        jobs = []
        for row in rows:
            _verify_stored_payload(row[0], row[1])
            job = _media_job_from_payload(json.loads(row[0]))
            if job.session_id == session_id:
                jobs.append(job)
        return tuple(jobs)

    def list_session_work_items(
        self, session_id: str
    ) -> tuple[ShotAnalysisJob | ShotMediaEncodeJobV1, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs ORDER BY rowid"
            ).fetchall()
        jobs = []
        for row in rows:
            _verify_stored_payload(row[0], row[1])
            job = _queue_job_from_payload(json.loads(row[0]))
            if job.session_id == session_id:
                jobs.append(job)
        return tuple(jobs)

    def session_media_snapshots(self, session_id):
        """One closed, fresh metadata query; never opens a sibling media file."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json,payload_sha256,status,artifact_id,terminal_failure,"
                "error_code,failure_reason,attempt_count FROM shot_analysis_jobs "
                "WHERE job_kind='media' AND json_extract(payload_json,'$.session_id')=? "
                "ORDER BY rowid",
                (session_id,),
            ).fetchall()
        snapshots = tuple(self._snapshot_row(row) for row in rows)
        if any(
            not isinstance(s, ShotMediaEncodeJobSnapshot) or s.job.session_id != session_id
            for s in snapshots
        ):
            raise ValueError("media snapshot session or kind conflicts")
        return snapshots

    def snapshot(self, job_id: str) -> ShotAnalysisJobSnapshot | ShotMediaEncodeJobSnapshot:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json,payload_sha256,status,artifact_id,terminal_failure,"
                "error_code,failure_reason,attempt_count "
                "FROM shot_analysis_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise ValueError("shot analysis job was not found")
        return self._snapshot_row(row)

    @staticmethod
    def _snapshot_row(row):
        _verify_stored_payload(row[0], row[1])
        job = _queue_job_from_payload(json.loads(row[0]))
        if isinstance(job, ShotMediaEncodeJobV1):
            return ShotMediaEncodeJobSnapshot(
                job,
                ShotAnalysisJobStatus(row[2]),
                row[3],
                bool(row[4]),
                row[5],
                row[6],
                int(row[7]),
            )
        return ShotAnalysisJobSnapshot(job, ShotAnalysisJobStatus(row[2]), row[3])

    def payload_sha256(self, job_id: str) -> str:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise ValueError("shot job was not found")
        _verify_stored_payload(row[0], row[1])
        _queue_job_from_payload(json.loads(row[0]))
        return row[1]

    def complete(self, job_id: str, artifact_id: str) -> bool:
        raise ValueError("shot completion requires the current lease claim token")

    def fail_retrying(self, job_id: str, reason: str) -> None:
        raise ValueError("shot retry failure requires the current lease claim token")

    def status(self, job_id: str) -> ShotAnalysisJobStatus:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status FROM shot_analysis_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise ValueError("shot analysis job was not found")
        return ShotAnalysisJobStatus(row[0])

    def _connect(self) -> sqlite3.Connection:
        if getattr(self, "_existing_only", False):
            path = self._database
            if (
                not path.is_absolute()
                or not path.is_file()
                or path.resolve(strict=True) != path
                or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))
            ):
                raise ValueError("original jobs database is unavailable")
            connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5.0)
        else:
            connection = sqlite3.connect(self._database, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def materialize_frozen_interval(
    interval: FrozenCaptureInterval, destination: Path
) -> FrozenCaptureInterval:
    """Atomically copy and verify a ring interval into immutable job media."""

    if destination.exists():
        return _validate_existing_materialization(interval, destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    pending = parent / f".pending-{destination.name}-{uuid4().hex}"
    try:
        pending.mkdir()
        for name, declared_sha256 in interval.segment_sha256.items():
            _require_segment_name(name)
            observed_sha256 = _copy_and_fsync(interval.root / name, pending / name)
            if observed_sha256 != declared_sha256:
                raise RealtimeRingError("capture ring segment hash changed")
        copied = replace(
            interval,
            root=pending,
            segment_sha256=dict(interval.segment_sha256),
        )
        tuple(copied.frames())
        _write_json_fsynced(pending / _INTERVAL_MANIFEST, _interval_payload(interval))
        _fsync_directory(pending)
        os.rename(pending, destination)
        _fsync_directory(parent)
    except OSError as exc:
        raise RealtimeRingError("frozen capture interval cannot be materialized") from exc
    finally:
        shutil.rmtree(pending, ignore_errors=True)
    return replace(
        interval,
        root=destination,
        segment_sha256=dict(interval.segment_sha256),
    )


def materialize_recorded_interval_reference(
    interval: RecordedShotInterval,
    destination: Path,
    *,
    recording_identity: bool = False,
) -> FrozenCaptureInterval:
    """Persist only an immutable pointer to session segments already on disk."""

    segment_root = interval.root.resolve()
    if interval.root.is_symlink() or not segment_root.is_dir():
        raise RealtimeRingError("session recording root is missing or unsafe")
    frozen = FrozenCaptureInterval(
        root=segment_root,
        source_epoch_id=interval.source_epoch_id,
        width=interval.width,
        height=interval.height,
        start_frame_id=interval.start_frame_id,
        end_frame_id=interval.end_frame_id,
        frame_count=interval.frame_count,
        segment_sha256=dict(interval.segment_sha256),
    )
    tuple(frozen.frames())
    payload = {
        **_interval_payload(frozen),
        "storage_kind": "session_segment_reference",
        "segment_root": str(segment_root),
    }
    if recording_identity:
        payload.update(
            schema_version="recording-reference/v1",
            session_id=interval.session_id,
            shot_id=interval.shot_id,
        )
    if destination.exists():
        try:
            stored = json.loads((destination / _INTERVAL_MANIFEST).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RealtimeRingError("recorded interval reference is invalid") from exc
        if stored != payload:
            raise RealtimeRingError("recorded interval reference conflicts")
        return frozen
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    pending = parent / f".pending-{destination.name}-{uuid4().hex}"
    try:
        pending.mkdir()
        _write_json_fsynced(pending / _INTERVAL_MANIFEST, payload)
        _fsync_directory(pending)
        os.rename(pending, destination)
        _fsync_directory(parent)
    except OSError as exc:
        raise RealtimeRingError("recorded interval reference cannot be persisted") from exc
    finally:
        shutil.rmtree(pending, ignore_errors=True)
    return frozen


def quarantine_frozen_interval(
    interval: FrozenCaptureInterval,
    destination: Path,
    reason: str,
) -> ShotMaterializationQuarantine:
    """Atomically retain raw diagnostic bytes when normal materialization fails."""

    if not reason:
        raise ValueError("shot quarantine requires a failure reason")
    if destination.exists():
        return _validate_existing_quarantine(interval, destination, reason)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    pending = parent / f".pending-{destination.name}-{uuid4().hex}"
    try:
        pending.mkdir()
        observed: dict[str, str] = {}
        segment_rows: dict[str, dict[str, str]] = {}
        for name, declared_sha256 in interval.segment_sha256.items():
            _require_segment_name(name)
            observed_sha256 = _copy_and_fsync(interval.root / name, pending / name)
            if _sha256_file(pending / name) != observed_sha256:
                raise RealtimeRingError("quarantined segment hash verification failed")
            observed[name] = observed_sha256
            segment_rows[name] = {
                "declared_sha256": declared_sha256,
                "observed_sha256": observed_sha256,
            }
        manifest = {
            "schema_version": "1.0",
            "kind": "mainline1_shot_materialization_quarantine",
            "reason": reason,
            "interval": _interval_payload(interval),
            "segments": segment_rows,
        }
        manifest_bytes = _write_json_fsynced(pending / _QUARANTINE_MANIFEST, manifest)
        _fsync_directory(pending)
        os.rename(pending, destination)
        _fsync_directory(parent)
    except OSError as exc:
        raise RealtimeRingError("frozen capture interval cannot be quarantined") from exc
    finally:
        shutil.rmtree(pending, ignore_errors=True)
    return ShotMaterializationQuarantine(
        root=destination,
        reason=reason,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        observed_segment_sha256=observed,
    )


def _validate_existing_materialization(
    interval: FrozenCaptureInterval, destination: Path
) -> FrozenCaptureInterval:
    if not destination.is_dir() or destination.is_symlink():
        raise RealtimeRingError("immutable shot media destination is invalid")
    copied = replace(
        interval,
        root=destination,
        segment_sha256=dict(interval.segment_sha256),
    )
    tuple(copied.frames())
    manifest_path = destination / _INTERVAL_MANIFEST
    try:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealtimeRingError("immutable shot media manifest is invalid") from exc
    if stored != _interval_payload(interval):
        raise RealtimeRingError("immutable shot media destination conflicts")
    return copied


def _validate_existing_quarantine(
    interval: FrozenCaptureInterval,
    destination: Path,
    reason: str,
) -> ShotMaterializationQuarantine:
    manifest_path = destination / _QUARANTINE_MANIFEST
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealtimeRingError("shot quarantine manifest is invalid") from exc
    if manifest.get("reason") != reason or manifest.get("interval") != _interval_payload(interval):
        raise RealtimeRingError("shot quarantine destination conflicts")
    segment_rows = manifest.get("segments")
    if not isinstance(segment_rows, dict) or set(segment_rows) != set(interval.segment_sha256):
        raise RealtimeRingError("shot quarantine segment manifest conflicts")
    observed: dict[str, str] = {}
    for name, declared_sha256 in interval.segment_sha256.items():
        row = segment_rows[name]
        if not isinstance(row, dict) or row.get("declared_sha256") != declared_sha256:
            raise RealtimeRingError("shot quarantine segment manifest conflicts")
        observed_sha256 = row.get("observed_sha256")
        if not isinstance(observed_sha256, str) or not _is_sha256(observed_sha256):
            raise RealtimeRingError("shot quarantine segment manifest conflicts")
        _require_segment_name(name)
        if _sha256_file(destination / name) != observed_sha256:
            raise RealtimeRingError("quarantined segment hash verification failed")
        observed[name] = observed_sha256
    return ShotMaterializationQuarantine(
        root=destination,
        reason=reason,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        observed_segment_sha256=observed,
    )


def _copy_and_fsync(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_stream, destination.open("xb") as target_stream:
            for chunk in iter(lambda: source_stream.read(_COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
                target_stream.write(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())
    except OSError as exc:
        raise RealtimeRingError("capture ring segment cannot be copied") from exc
    return digest.hexdigest()


def _write_json_fsynced(path: Path, payload: dict[str, Any]) -> bytes:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return encoded


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


def _interval_payload(interval: FrozenCaptureInterval) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source_epoch_id": interval.source_epoch_id,
        "width": interval.width,
        "height": interval.height,
        "start_frame_id": interval.start_frame_id,
        "end_frame_id": interval.end_frame_id,
        "frame_count": interval.frame_count,
        "media_sha256": interval.media_sha256,
        "segment_sha256": dict(interval.segment_sha256),
    }


def _canonical_job_payload(job: ShotAnalysisJob) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "job_id": job.job_id,
        "session_id": job.session_id,
        "shot_id": job.shot_id,
        "source_epoch_id": job.source_epoch_id,
        "start_frame_id": job.start_frame_id,
        "end_frame_id": job.end_frame_id,
        "media_root": str(job.media_root),
        "media_sha256": job.media_sha256,
        "capture_setup_id": job.capture_setup_id,
        "calibration_artifact_id": job.calibration_artifact_id,
        "config_sha256": job.config_sha256,
        "model_package_id": job.model_package_id,
        "identity_training_package_id": job.identity_training_package_id,
        "identity_training_package_sha256": job.identity_training_package_sha256,
    }


def _media_source_payload(source: ShotMediaEncodeSourceV1) -> dict[str, object]:
    if isinstance(source, RecordingIntervalMediaSourceV1):
        return {
            "kind": source.kind,
            "shot_evidence_sha256": source.shot_evidence_sha256,
            "reference_path": str(source.reference_path),
            "reference_sha256": source.reference_sha256,
        }
    if isinstance(source, RecordingCaptureMediaSourceV1):
        return {
            "kind": source.kind,
            "capture_end_sha256": source.capture_end_sha256,
            "shot_evidence_sha256": source.shot_evidence_sha256,
            "reference_path": str(source.reference_path),
            "reference_sha256": source.reference_sha256,
        }
    if isinstance(source, RecordingEndMediaSourceV1):
        return {
            "kind": source.kind,
            "recording_end_sha256": source.recording_end_sha256,
            "shot_evidence_sha256": source.shot_evidence_sha256,
            "reference_path": str(source.reference_path),
            "reference_sha256": source.reference_sha256,
        }
    return {
        "kind": source.kind,
        "postgame_manifest_sha256": source.postgame_manifest_sha256,
        "source_job_id": source.source_job_id,
        "source_job_payload_sha256": source.source_job_payload_sha256,
    }


def _media_job_identity_payload(job: ShotMediaEncodeJobV1) -> dict[str, object]:
    return {
        "contract": job.contract,
        "schema_version": "1.0",
        "session_id": job.session_id,
        "shot_id": job.shot_id,
        "source_epoch_id": job.source_epoch_id,
        "start_frame_id": job.start_frame_id,
        "end_frame_id": job.end_frame_id,
        "media_root": str(job.media_root),
        "media_sha256": job.media_sha256,
        "source": _media_source_payload(job.source),
    }


def _shot_media_encode_job_id(job: ShotMediaEncodeJobV1) -> str:
    encoded = _canonical_json(_media_job_identity_payload(job)).encode("utf-8")
    digest = hashlib.sha256(b"billiards-ai:recording-encode-job/v1\0" + encoded).hexdigest()
    return f"media-encode-{digest}"


def _media_job_payload(job: ShotMediaEncodeJobV1) -> dict[str, object]:
    return {
        **_media_job_identity_payload(job),
        "job_id": job.job_id,
        "max_attempts": job.max_attempts,
    }


def _queue_job_payload(
    job: ShotAnalysisJob | ShotMediaEncodeJobV1,
) -> dict[str, object]:
    if isinstance(job, ShotMediaEncodeJobV1):
        return _media_job_payload(job)
    if isinstance(job, ShotAnalysisJob):
        return _canonical_job_payload(job)
    raise TypeError("unsupported shot queue job")


def _analysis_job_from_payload(payload: dict[str, object]) -> ShotAnalysisJob:
    payload = dict(payload)
    if payload.pop("schema_version", None) != "1.0" or "contract" in payload:
        raise ValueError("shot analysis job payload schema is invalid")
    payload["media_root"] = Path(str(payload["media_root"]))
    return ShotAnalysisJob(**payload)  # type: ignore[arg-type]


def _media_job_from_payload(payload: dict[str, object]) -> ShotMediaEncodeJobV1:
    expected = {
        "contract",
        "schema_version",
        "job_id",
        "session_id",
        "shot_id",
        "source_epoch_id",
        "start_frame_id",
        "end_frame_id",
        "media_root",
        "media_sha256",
        "max_attempts",
        "source",
    }
    if (
        set(payload) != expected
        or payload.get("contract") != "recording-encode-job/v1"
        or payload.get("schema_version") != "1.0"
        or not isinstance(payload.get("source"), dict)
        or any(
            type(payload.get(name)) is not str
            for name in (
                "contract",
                "schema_version",
                "job_id",
                "session_id",
                "shot_id",
                "source_epoch_id",
                "media_root",
                "media_sha256",
            )
        )
        or type(payload.get("start_frame_id")) is not int
        or type(payload.get("end_frame_id")) is not int
        or type(payload.get("max_attempts")) is not int
    ):
        raise ValueError("shot media encode payload schema is invalid")
    raw_source = dict(payload["source"])  # type: ignore[arg-type]
    kind = raw_source.get("kind")
    if kind == "recording_end":
        if set(raw_source) != {
            "kind",
            "recording_end_sha256",
            "shot_evidence_sha256",
            "reference_path",
            "reference_sha256",
        }:
            raise ValueError("recording media source payload is invalid")
        if any(type(value) is not str for value in raw_source.values()):
            raise ValueError("recording media source payload is invalid")
        raw_source["reference_path"] = Path(raw_source["reference_path"])
        source: ShotMediaEncodeSourceV1 = RecordingEndMediaSourceV1(**raw_source)  # type: ignore[arg-type]
    elif kind == "recording_capture":
        if set(raw_source) != {
            "kind",
            "capture_end_sha256",
            "shot_evidence_sha256",
            "reference_path",
            "reference_sha256",
        } or any(type(value) is not str for value in raw_source.values()):
            raise ValueError("recording capture media source payload is invalid")
        raw_source["reference_path"] = Path(raw_source["reference_path"])
        source = RecordingCaptureMediaSourceV1(**raw_source)  # type: ignore[arg-type]
    elif kind == "recording_interval":
        if set(raw_source) != {
            "kind",
            "shot_evidence_sha256",
            "reference_path",
            "reference_sha256",
        } or any(type(value) is not str for value in raw_source.values()):
            raise ValueError("recording interval media source payload is invalid")
        raw_source["reference_path"] = Path(raw_source["reference_path"])
        source = RecordingIntervalMediaSourceV1(**raw_source)  # type: ignore[arg-type]
    elif kind == "legacy_frozen_job":
        if set(raw_source) != {
            "kind",
            "postgame_manifest_sha256",
            "source_job_id",
            "source_job_payload_sha256",
        }:
            raise ValueError("legacy media source payload is invalid")
        if any(type(value) is not str for value in raw_source.values()):
            raise ValueError("legacy media source payload is invalid")
        source = LegacyFrozenJobMediaSourceV1(**raw_source)  # type: ignore[arg-type]
    else:
        raise ValueError("shot media source tag is invalid")
    values = dict(payload)
    values.pop("schema_version")
    values["media_root"] = Path(values["media_root"])  # type: ignore[arg-type]
    values["source"] = source
    try:
        return ShotMediaEncodeJobV1(**values)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as exc:
        raise ValueError("shot media encode payload is invalid") from exc


def _queue_job_from_payload(
    payload: dict[str, object],
) -> ShotAnalysisJob | ShotMediaEncodeJobV1:
    contract = payload.get("contract")
    if contract is None:
        return _analysis_job_from_payload(payload)
    if contract == "recording-encode-job/v1":
        return _media_job_from_payload(payload)
    raise ValueError("shot queue job contract is invalid")


def _verify_stored_payload(payload_json: str, payload_sha256: str) -> None:
    if _sha256_bytes(payload_json.encode("utf-8")) != payload_sha256:
        raise ValueError("shot analysis job payload hash mismatch")


def _require_segment_name(name: str) -> None:
    path = Path(name)
    if not name or path.name != name or path.is_absolute():
        raise RealtimeRingError("capture ring segment name is invalid")


def _require_sha256(value: str, kind: str) -> None:
    if type(value) is not str or not _is_sha256(value):
        raise ValueError(f"shot analysis job {kind} SHA-256 is invalid")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("shot analysis lease time must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RealtimeRingError("capture ring segment cannot be hashed") from exc
    return digest.hexdigest()
