"""Read-only postgame manifests, aggregation, and session-scoped identifiers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.records import (
    ShotOwnershipReceipt,
    ownership_receipt_json,
    ownership_receipt_sha256,
)


class SessionAnalysisStatus(StrEnum):
    PROCESSING = "PROCESSING"
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    FAILED_RETRYING = "FAILED_RETRYING"


class ClipSaveStatus(StrEnum):
    NOT_SAVED = "NOT_SAVED"
    SAVE_PENDING = "SAVE_PENDING"
    SAVED = "SAVED"
    SAVE_FAILED_RETRYING = "SAVE_FAILED_RETRYING"
    SAVE_BLOCKED = "SAVE_BLOCKED"


MAX_SAVED_CLIP_BYTES = 256 * 1024 * 1024
MAX_SAVED_CLIP_DURATION_MS = 300_000
MAX_JS_SAFE_INTEGER = 9_007_199_254_740_991
_SAVE_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")


def validate_save_request_id(value: str) -> str:
    """Return one cloud-safe request identity or fail before persistence."""
    if not isinstance(value, str) or _SAVE_REQUEST_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("clip save request id is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ClipSaveReceipt:
    request_id: str
    session_id: str
    shot_id: str
    postgame_manifest_sha256: str
    status: ClipSaveStatus
    clip_id: str | None = None

    schema_version: str = "1.0"
    owner_user_id: str | None = None
    actor_role: WristbandRole | None = None
    ownership_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version == "2.0":
            if (
                not isinstance(self.owner_user_id, str)
                or not self.owner_user_id.strip()
                or any(ord(c) < 32 for c in self.owner_user_id)
                or self.actor_role not in {WristbandRole.RED, WristbandRole.BLUE}
                or not isinstance(self.ownership_receipt_sha256, str)
                or len(self.ownership_receipt_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.ownership_receipt_sha256)
            ):
                raise ValueError("invalid clip save ownership")
        elif self.schema_version != "1.0" or any(
            value is not None
            for value in (self.owner_user_id, self.actor_role, self.ownership_receipt_sha256)
        ):
            raise ValueError("invalid clip save ownership version")
        validate_save_request_id(self.request_id)
        if self.status not in {ClipSaveStatus.SAVE_PENDING, ClipSaveStatus.SAVED}:
            raise ValueError("clip save receipt status is invalid")
        if (self.status is ClipSaveStatus.SAVE_PENDING) != (self.clip_id is None):
            raise ValueError("clip save receipt clip identity is invalid")


def clip_save_receipt_from_json(raw: str) -> ClipSaveReceipt:
    """Explicit legacy loader; reading an old receipt never upgrades its authority."""
    try:
        payload = json.loads(raw)
        legacy = {
            "request_id",
            "session_id",
            "shot_id",
            "postgame_manifest_sha256",
            "status",
            "clip_id",
        }
        version = payload.get("schema_version", "1.0")
        allowed = legacy if "schema_version" not in payload else legacy | {"schema_version"}
        if version == "2.0":
            allowed |= {"owner_user_id", "actor_role", "ownership_receipt_sha256"}
        if set(payload) != allowed:
            raise ValueError("invalid clip save ownership fields")
        payload["status"] = ClipSaveStatus(payload["status"])
        if payload.get("actor_role") is not None:
            payload["actor_role"] = WristbandRole(payload["actor_role"])
        return ClipSaveReceipt(**payload)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError("invalid clip save ownership receipt") from exc


@dataclass(frozen=True, slots=True)
class PostgameClip:
    session_id: str
    shot_id: str
    postgame_manifest_sha256: str
    job_id: str
    save_status: ClipSaveStatus
    clip_id: str | None = None
    artifact_id: str | None = None
    local_path: Path | None = None
    content_sha256: str | None = None
    byte_size: int | None = None
    content_type: str | None = None
    sequence: int = 0
    actor_role: WristbandRole | None = None
    owner_user_id: str | None = None
    can_save: bool = False
    ownership_unavailable_reason: str | None = "SHOT_OWNERSHIP_UNAVAILABLE"
    owner_delete_status: str | None = None


@dataclass(frozen=True, slots=True)
class LocalDiscardClaim:
    """Immutable discard authority and a fenced attempt; never a media lock."""

    session_id: str
    shot_id: str
    postgame_manifest_sha256: str
    job_id: str
    artifact_id: str
    clip_id: str
    local_path: Path
    content_sha256: str
    byte_size: int
    content_type: str
    attempts: int
    claim_token: str
    lease_until_utc: str
    progress_json: str


@dataclass(frozen=True, slots=True)
class SavedMediaOutboxItem:
    clip_id: str
    session_id: str
    shot_id: str
    postgame_manifest_sha256: str
    job_id: str
    artifact_id: str
    local_path: Path
    content_sha256: str
    byte_size: int
    content_type: str
    source_kind: str
    start_ms: int
    end_ms: int
    duration_ms: int
    save_request_id: str
    save_receipt_sha256: str
    status: str
    attempts: int
    claim_token: str | None
    lease_until_utc: str | None
    next_attempt_at_utc: str
    last_error_code: str | None
    owner_user_id: str | None = None
    actor_role: WristbandRole | None = None
    ownership_receipt_sha256: str | None = None
    ownership_receipt: ShotOwnershipReceipt | None = None

    def __post_init__(self) -> None:
        text_values = (
            self.clip_id,
            self.session_id,
            self.shot_id,
            self.postgame_manifest_sha256,
            self.job_id,
            self.artifact_id,
            self.save_request_id,
        )
        if any(
            not isinstance(value, str)
            or not value
            or any(ord(character) < 32 for character in value)
            for value in text_values
        ):
            raise ValueError("saved media identity is invalid")
        if self.clip_id != session_scoped_clip_id(
            self.session_id,
            self.shot_id,
            self.job_id,
            self.artifact_id,
        ):
            raise ValueError("saved media identity is invalid")
        if (
            not self.local_path.is_absolute()
            or len(self.postgame_manifest_sha256) != 64
            or len(self.content_sha256) != 64
            or len(self.artifact_id) != 64
            or len(self.save_receipt_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for digest in (
                    self.postgame_manifest_sha256,
                    self.artifact_id,
                    self.content_sha256,
                    self.save_receipt_sha256,
                )
                for character in digest
            )
            or type(self.byte_size) is not int
            or self.byte_size <= 0
            or self.byte_size > MAX_SAVED_CLIP_BYTES
            or self.content_type != "video/mp4"
            or self.source_kind != "usb_live"
            or type(self.start_ms) is not int
            or type(self.end_ms) is not int
            or type(self.duration_ms) is not int
            or self.start_ms < 0
            or self.start_ms > MAX_JS_SAFE_INTEGER
            or self.end_ms <= self.start_ms
            or self.end_ms > MAX_JS_SAFE_INTEGER
            or self.duration_ms != self.end_ms - self.start_ms
            or self.duration_ms > MAX_SAVED_CLIP_DURATION_MS
        ):
            raise ValueError("saved media descriptor is invalid")

    @property
    def idempotency_key(self) -> str:
        """The remote receiver must use this key when retrying a lost acknowledgement."""
        return self.clip_id

    @property
    def descriptor_json(self) -> str:
        """Canonical append-only cloud request, derived only from frozen local facts."""
        receipt = self.ownership_receipt
        if (
            receipt is None
            or receipt.evidence_status != "bound_at_start"
            or receipt.source_kind != "usb_live"
            or receipt.actor_user_id != self.owner_user_id
            or receipt.actor_role != self.actor_role
            or receipt.session_id != self.session_id
            or receipt.shot_id != self.shot_id
            or ownership_receipt_sha256(receipt) != self.ownership_receipt_sha256
        ):
            raise ValueError("saved media ownership is unavailable")
        return json.dumps(
            {
                "schema_version": "2.0",
                "owner_user_id": self.owner_user_id,
                "actor_role": self.actor_role,
                "ownership_receipt_sha256": self.ownership_receipt_sha256,
                "ownership_receipt": json.loads(ownership_receipt_json(receipt)),
                "publication_kind": "saved_postgame_clip",
                "clip_id": self.clip_id,
                "session_id": self.session_id,
                "shot_id": self.shot_id,
                "job_id": self.job_id,
                "artifact_id": self.artifact_id,
                "postgame_manifest_sha256": self.postgame_manifest_sha256,
                "save_request_id": self.save_request_id,
                "save_receipt_sha256": self.save_receipt_sha256,
                "source_kind": self.source_kind,
                "start_ms": self.start_ms,
                "end_ms": self.end_ms,
                "duration_ms": self.duration_ms,
                "content_sha256": self.content_sha256,
                "byte_size": self.byte_size,
                "content_type": self.content_type,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class PostgameSessionManifest:
    session_id: str
    ended_at_utc: datetime
    deadline_at_utc: datetime
    expected_jobs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.session_id or self.deadline_at_utc < self.ended_at_utc:
            raise ValueError("postgame manifest identity or deadline is invalid")
        if len(self.expected_jobs) != len(set(self.expected_jobs)) or any(
            not job_id for job_id in self.expected_jobs
        ):
            raise ValueError("postgame expected jobs must be unique and non-empty")

    @property
    def manifest_sha256(self) -> str:
        """Identity of the frozen job head, independent of generated analysis JSON."""
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (self.ended_at_utc, self.deadline_at_utc)
        ):
            raise ValueError("postgame manifest timestamps must be timezone aware")
        payload = {
            "session_id": self.session_id,
            "ended_at_utc": self.ended_at_utc.astimezone(UTC).isoformat(),
            "deadline_at_utc": self.deadline_at_utc.astimezone(UTC).isoformat(),
            "expected_jobs": self.expected_jobs,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PostgameJobSnapshot:
    job_id: str
    status: str
    artifact_id: str | None


@dataclass(frozen=True, slots=True)
class SessionAnalysisProjection:
    status: SessionAnalysisStatus
    expected_count: int
    available_count: int


def project_session_analysis(
    manifest: PostgameSessionManifest,
    jobs: tuple[PostgameJobSnapshot, ...],
    now_utc: datetime,
    *,
    finalized: bool = False,
) -> SessionAnalysisProjection:
    """Reduce immutable expected heads with the specified, order-independent precedence."""
    expected = set(manifest.expected_jobs)
    by_job = {job.job_id: job for job in jobs if job.job_id in expected}
    available = sum(job.status == "available" and bool(job.artifact_id) for job in by_job.values())
    if available == len(expected):
        status = SessionAnalysisStatus.AVAILABLE if finalized else SessionAnalysisStatus.PROCESSING
    elif available:
        status = SessionAnalysisStatus.PARTIAL
    elif (
        any(job.status == "failed_retrying" for job in by_job.values())
        or now_utc >= manifest.deadline_at_utc
    ):
        status = SessionAnalysisStatus.FAILED_RETRYING
    else:
        status = SessionAnalysisStatus.PROCESSING
    return SessionAnalysisProjection(status, len(expected), available)


def session_scoped_clip_id(session_id: str, shot_id: str, job_id: str, artifact_id: str) -> str:
    """Stable ordinary-media ID that cannot collide across session rollovers."""
    encoded = json.dumps(
        [session_id, shot_id, job_id, artifact_id], separators=(",", ":"), ensure_ascii=True
    ).encode()
    return f"clip-{hashlib.sha256(encoded).hexdigest()}"
