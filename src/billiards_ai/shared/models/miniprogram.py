"""Strict CloudBase boundary contracts for the long-term user mini program."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)


def _safe_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError("control characters are not allowed")
    return normalized


def _relative_media_path(value: str) -> str:
    if value.startswith(("/", "\\")) or "\\" in value:
        raise ValueError("media path must be a portable relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("media path contains an unsafe segment")
    return value


def _aware_utc(value: object) -> datetime:
    if isinstance(value, (bool, int, float)):
        raise ValueError("datetime must not be numeric")
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC)


SafeIdentifier = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128),
    AfterValidator(_safe_text),
]
SafeMediaPath = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=512),
    AfterValidator(_safe_text),
    AfterValidator(_relative_media_path),
]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
PositiveStrictInt = Annotated[StrictInt, Field(ge=1)]
NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]
Role = Literal["red", "blue"]
BallGroup = Literal["stripe", "solid", "open", "unknown"]
LimitationCode = Literal[
    "development_replay_media",
    "no_global_fixed_ball_id",
    "shot_local_fragment_and_cushion_limits",
    "no_automatic_foul_verdict",
]
AnalysisStatus = Literal["PROCESSING", "AVAILABLE", "PARTIAL", "FAILED_RETRYING"]
CanonicalMediaSource = Literal["development_replay", "usb_live"]


class _StrictMiniProgramModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class MiniProgramParticipantV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    user_id: SafeIdentifier
    role: Role
    group: BallGroup


class MiniProgramPowerV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    band: Literal["light", "medium_light", "medium", "medium_firm", "firm"]
    approximate_percent: Annotated[StrictInt, Field(ge=1, le=100)]
    personalized: Literal[False] = False


class MiniProgramShotV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    shot_id: SafeIdentifier
    sequence: PositiveStrictInt
    actor_role: Role | None
    started_ms: NonNegativeStrictInt
    settled_ms: PositiveStrictInt
    recommendation_id: SafeIdentifier | None
    live_job_id: SafeIdentifier | None = None
    power: MiniProgramPowerV1 | None
    limitations: tuple[LimitationCode, ...]

    @model_validator(mode="after")
    def shot_timing_and_limitations_are_consistent(self) -> MiniProgramShotV1:
        if self.settled_ms <= self.started_ms:
            raise ValueError("shot must settle after it starts")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("shot limitations must be unique")
        return self


class MiniProgramSessionManifestV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: SafeIdentifier
    artifact_id: SafeIdentifier
    status: Literal["ended"]
    created_at_utc: datetime
    ended_at_utc: datetime
    postgame_manifest_sha256: Sha256Digest | None = None
    evidence_source_kind: CanonicalMediaSource
    participants: Annotated[tuple[MiniProgramParticipantV1, ...], Field(max_length=2)]
    breaker: Role | None
    active_role_at_end: Role | None
    shots: tuple[MiniProgramShotV1, ...]
    limitations: tuple[LimitationCode, ...]

    _created_at_utc = field_validator("created_at_utc", mode="before")(_aware_utc)
    _ended_at_utc = field_validator("ended_at_utc", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def session_lineage_is_consistent(self) -> MiniProgramSessionManifestV1:
        if self.ended_at_utc < self.created_at_utc:
            raise ValueError("session cannot end before it starts")
        roles = tuple(item.role for item in self.participants)
        users = tuple(item.user_id for item in self.participants)
        if len(roles) != len(set(roles)) or len(users) != len(set(users)):
            raise ValueError("session participants must have unique roles and users")
        sequences = tuple(shot.sequence for shot in self.shots)
        if sequences != tuple(range(1, len(sequences) + 1)):
            raise ValueError("session shot sequences must be contiguous and ordered")
        shot_ids = tuple(shot.shot_id for shot in self.shots)
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("session shot IDs must be unique")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("session limitations must be unique")
        return self


class MiniProgramClipV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    clip_id: SafeIdentifier
    session_id: SafeIdentifier
    shot_id: SafeIdentifier
    artifact_id: SafeIdentifier
    relative_path: SafeMediaPath
    source_kind: CanonicalMediaSource
    start_ms: NonNegativeStrictInt
    end_ms: PositiveStrictInt
    duration_ms: Annotated[StrictInt, Field(ge=1, le=300_000)]
    content_sha256: Sha256Digest
    byte_size: PositiveStrictInt
    media_type: Literal["video/mp4"]

    @model_validator(mode="after")
    def media_timing_is_consistent(self) -> MiniProgramClipV1:
        if self.end_ms <= self.start_ms:
            raise ValueError("media range must end after it starts")
        if self.duration_ms != self.end_ms - self.start_ms:
            raise ValueError("media duration must match its range")
        return self


class SessionAnalysisProjectionV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: SafeIdentifier
    revision: PositiveStrictInt
    status: AnalysisStatus
    total_shots: NonNegativeStrictInt | None
    available_shots: NonNegativeStrictInt | None
    retrying_shots: NonNegativeStrictInt | None
    report_available: bool
    deadline_at: datetime
    updated_at: datetime

    _deadline_at = field_validator("deadline_at", mode="before")(_aware_utc)
    _updated_at = field_validator("updated_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def counts_match_status(self) -> SessionAnalysisProjectionV1:
        counts = (self.total_shots, self.available_shots, self.retrying_shots)
        if any(item is None for item in counts):
            if any(item is not None for item in counts) or self.status != "PROCESSING":
                raise ValueError("analysis counts must be absent only while processing")
            if self.report_available:
                raise ValueError("analysis status conflicts with availability")
            return self
        if self.available_shots + self.retrying_shots > self.total_shots:
            raise ValueError("analysis counts exceed frozen shot count")
        if self.status == "PROCESSING" and (
            self.available_shots != 0 or self.retrying_shots != 0 or self.report_available
        ):
            raise ValueError("analysis status conflicts with processing counts")
        if self.status == "AVAILABLE" and (
            self.available_shots != self.total_shots
            or self.retrying_shots != 0
            or not self.report_available
        ):
            raise ValueError("available analysis must be complete")
        if self.status == "PARTIAL" and (
            self.available_shots < 1
            or self.available_shots >= self.total_shots
            or not self.report_available
        ):
            raise ValueError("analysis status conflicts with partial availability")
        if self.status == "FAILED_RETRYING" and (
            self.available_shots != 0 or self.retrying_shots < 1 or self.report_available
        ):
            raise ValueError("analysis status conflicts with retry state")
        return self


class EdgeSyncEnvelopeV1(_StrictMiniProgramModel):
    schema_version: Literal["1.0"] = "1.0"
    session: MiniProgramSessionManifestV1
    media: tuple[MiniProgramClipV1, ...]

    @model_validator(mode="after")
    def media_belongs_to_the_session(self) -> EdgeSyncEnvelopeV1:
        shot_ids = {shot.shot_id for shot in self.session.shots}
        clip_ids: set[str] = set()
        for clip in self.media:
            if clip.session_id != self.session.session_id:
                raise ValueError("media session must match the envelope session")
            if clip.artifact_id != self.session.artifact_id:
                raise ValueError("media artifact must match the envelope artifact")
            if clip.shot_id not in shot_ids:
                raise ValueError("media shot must belong to the envelope session")
            if clip.clip_id in clip_ids:
                raise ValueError("media clip IDs must be unique")
            clip_ids.add(clip.clip_id)
            if clip.source_kind != self.session.evidence_source_kind:
                raise ValueError("media evidence source must match the explicit session source")
        if not self.media and self.session.evidence_source_kind == "usb_live":
            if self.session.postgame_manifest_sha256 is None:
                raise ValueError("facts-only USB sync requires a frozen postgame manifest")
            if any(shot.live_job_id is None for shot in self.session.shots):
                raise ValueError("facts-only USB sync requires every shot live job")
        return self


def canonical_sync_hash(value: EdgeSyncEnvelopeV1) -> str:
    """Return the stable lowercase SHA-256 digest for a validated sync envelope."""

    payload = value.model_dump(mode="json", exclude_none=False)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
