"""Strict local contracts for field sessions, shots, and facts-only reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Literal

from billiards_ai.edge.route_assistance.control import (
    MatchControlSnapshot,
    WristbandRole,
)

PowerBand = Literal["light", "medium_light", "medium", "medium_firm", "firm"]
MediaSourceKind = Literal["development_replay", "usb_live"]
RuleDecisionStatus = Literal["applied", "abstained", "manual", "stale"]
LiveMatchMode = Literal["assistance", "observation"]
LiveShotFactOutboxStatus = Literal["pending", "leased", "acknowledged"]


def require_session_policy(value: str) -> str:
    if type(value) is not str or value not in {"legacy", "recording-v1"}:
        raise ValueError("unsupported session recording policy")
    return value


@dataclass(frozen=True, slots=True)
class RecordingBoundaryV1:
    shot_id: str
    source_epoch_id: str
    start_frame_id: int
    end_frame_id: int
    started_ms: int
    ended_ms: int
    complete: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.shot_id) is not str
            or not self.shot_id
            or type(self.source_epoch_id) is not str
            or not self.source_epoch_id
            or any(
                type(v) is not int or not 0 <= v <= 9_007_199_254_740_991
                for v in (
                    self.start_frame_id,
                    self.end_frame_id,
                    self.started_ms,
                    self.ended_ms,
                )
            )
            or self.end_frame_id < self.start_frame_id
            or self.ended_ms < self.started_ms
            or (self.complete and self.ended_ms == self.started_ms)
            or type(self.complete) is not bool
            or (not self.complete and (type(self.reason) is not str or not self.reason))
            or (self.reason is not None and len(self.reason) > 300)
            or (self.complete and self.reason is not None)
        ):
            raise ValueError("invalid recording boundary")


@dataclass(frozen=True, slots=True)
class RecordedMediaV1:
    boundary: RecordingBoundaryV1
    status: Literal["RECORDED", "FAILED"]
    evidence_ref_id: str | None = None
    reference_path: str | None = None
    reference_sha256: str | None = None
    media_sha256: str | None = None
    start_frame_id: int | None = None
    end_frame_id: int | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, RecordingBoundaryV1):
            raise ValueError("invalid recording media boundary")
        if (
            self.status not in {"RECORDED", "FAILED"}
            or (
                self.status == "RECORDED"
                and (
                    not self.boundary.complete
                    or self.failure_code is not None
                    or self.reference_path is None
                )
            )
            or (
                self.status == "FAILED"
                and (
                    type(self.failure_code) is not str
                    or not self.failure_code
                    or len(self.failure_code) > 128
                )
            )
        ):
            raise ValueError("invalid recording media status")
        reference = (
            self.evidence_ref_id,
            self.reference_path,
            self.reference_sha256,
            self.media_sha256,
            self.start_frame_id,
            self.end_frame_id,
        )
        if any(v is not None for v in reference):
            if (
                any(v is None for v in reference)
                or type(self.evidence_ref_id) is not str
                or not self.evidence_ref_id
                or type(self.reference_path) is not str
                or not Path(self.reference_path).is_absolute()
                or any(
                    type(v) is not str
                    or len(v) != 64
                    or any(c not in "0123456789abcdef" for c in v)
                    for v in (self.reference_sha256, self.media_sha256)
                )
                or type(self.start_frame_id) is not int
                or type(self.end_frame_id) is not int
                or not 0 <= self.start_frame_id <= self.boundary.start_frame_id
                or not self.start_frame_id <= self.end_frame_id <= 9_007_199_254_740_991
                or self.end_frame_id < self.boundary.end_frame_id
            ):
                raise ValueError("invalid recorded interval reference")


def recording_evidence_from_payload(value: object) -> dict:
    """Revalidate the versioned local evidence at every database read/write."""
    keys = {
        "contract",
        "session_id",
        "shot_id",
        "start",
        "boundary",
        "media",
        "media_status",
        "analysis_status",
        "encoding_status",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or (
            value["contract"] != "recording-evidence/v1"
            or value["analysis_status"] != "NOT_REQUESTED"
            or value["encoding_status"] != "NOT_SCHEDULED"
            or value["media_status"] not in {"PENDING", "RECORDED", "FAILED"}
        )
    ):
        raise ValueError("invalid recording metadata schema")
    for key in ("session_id", "shot_id"):
        if type(value[key]) is not str or not value[key]:
            raise ValueError("invalid recording metadata identity")
    start = value["start"]
    if type(start) is not dict or set(start) != {"source_epoch_id", "frame_id", "started_ms"}:
        raise ValueError("invalid recording start schema")
    if (
        type(start["source_epoch_id"]) is not str
        or not start["source_epoch_id"]
        or any(
            type(start[key]) is not int or not 0 <= start[key] <= 9_007_199_254_740_991
            for key in ("frame_id", "started_ms")
        )
    ):
        raise ValueError("invalid recording start metadata")
    boundary = None
    if value["boundary"] is not None:
        try:
            boundary = RecordingBoundaryV1(**value["boundary"])
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid recording boundary schema") from exc
        if (
            boundary.shot_id,
            boundary.source_epoch_id,
            boundary.start_frame_id,
            boundary.started_ms,
        ) != (value["shot_id"], start["source_epoch_id"], start["frame_id"], start["started_ms"]):
            raise ValueError("recording boundary metadata lineage mismatch")
    media = value["media"]
    if media is None:
        if value["media_status"] != "PENDING":
            raise ValueError("recording status has no media evidence")
    else:
        if type(media) is not dict or boundary is None:
            raise ValueError("recording media has no boundary")
        try:
            parsed = RecordedMediaV1(
                **(media | {"boundary": RecordingBoundaryV1(**media["boundary"])})
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid recording media schema") from exc
        if parsed.boundary != boundary or parsed.status != value["media_status"]:
            raise ValueError("recording media metadata lineage mismatch")
    return value


class FieldSessionStatus(StrEnum):
    OPEN = "open"
    ENDED = "ended"


class SessionLimitCode(StrEnum):
    DEVELOPMENT_REPLAY_MEDIA = "development_replay_media"
    NO_GLOBAL_FIXED_BALL_ID = "no_global_fixed_ball_id"
    SHOT_LOCAL_FRAGMENT_AND_CUSHION_LIMITS = "shot_local_fragment_and_cushion_limits"
    NO_AUTOMATIC_FOUL_VERDICT = "no_automatic_foul_verdict"


def _require_text(value: str, *, name: str) -> None:
    if not value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not be empty or contain control characters")


def _require_utc_text(value: str, *, name: str) -> None:
    _require_text(value, name=name)
    if not value.endswith("Z"):
        raise ValueError(f"{name} must be a UTC timestamp ending in Z")


def _require_unique_limitations(limitations: tuple[SessionLimitCode, ...]) -> None:
    if len(limitations) != len(set(limitations)):
        raise ValueError("session limitations must be unique")


@dataclass(frozen=True, slots=True)
class ShotMediaRange:
    artifact_id: str
    source_locator: str
    source_kind: MediaSourceKind
    start_ms: int
    end_ms: int
    source_epoch_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, name="media artifact ID")
        if self.source_kind == "usb_live":
            _require_text(self.source_epoch_id or "", name="USB source epoch")
            if self.source_locator != f"usb:{self.source_epoch_id}":
                raise ValueError("USB source locator must bind its epoch")
        elif self.source_kind == "development_replay":
            if self.source_locator != "presentation.mp4" or self.source_epoch_id is not None:
                raise ValueError("development replay must use presentation.mp4 without an epoch")
        else:
            raise ValueError("media source must use a canonical source kind")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("media range must end after it starts")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class LiveShotMediaReference:
    """Immutable materialized media attachment, separate from start-time ownership."""

    session_id: str
    shot_id: str
    source_epoch_id: str
    match_revision: int
    job_id: str
    media_path: str
    media_sha256: str

    def __post_init__(self) -> None:
        for name in ("session_id", "shot_id", "source_epoch_id", "job_id", "media_path"):
            _require_text(getattr(self, name), name=name)
        if self.match_revision < 0 or not Path(self.media_path).is_absolute():
            raise ValueError("USB media requires a revision and absolute immutable path")
        if len(self.media_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.media_sha256
        ):
            raise ValueError("USB media hash must be SHA-256")


@dataclass(frozen=True, slots=True)
class FieldSessionRecord:
    session_id: str
    artifact_id: str
    status: FieldSessionStatus
    created_at_utc: str
    ended_at_utc: str | None = None
    match: MatchControlSnapshot = field(default_factory=MatchControlSnapshot)

    def __post_init__(self) -> None:
        _require_text(self.session_id, name="session ID")
        _require_text(self.artifact_id, name="artifact ID")
        _require_utc_text(self.created_at_utc, name="created timestamp")
        if self.status is FieldSessionStatus.OPEN and self.ended_at_utc is not None:
            raise ValueError("open session cannot have an ended timestamp")
        if self.status is FieldSessionStatus.ENDED and self.ended_at_utc is None:
            raise ValueError("ended session requires an ended timestamp")
        if self.ended_at_utc is not None:
            _require_utc_text(self.ended_at_utc, name="ended timestamp")

    @classmethod
    def open(
        cls,
        *,
        session_id: str,
        artifact_id: str,
        created_at_utc: str,
    ) -> FieldSessionRecord:
        return cls(
            session_id=session_id,
            artifact_id=artifact_id,
            status=FieldSessionStatus.OPEN,
            created_at_utc=created_at_utc,
        )


@dataclass(frozen=True, slots=True)
class FieldShotRecord:
    session_id: str
    shot_id: str
    sequence: int
    actor_role: WristbandRole | None
    started_ms: int
    settled_ms: int | None
    recommendation_id: str | None
    power_band: PowerBand | None
    power_percent: int | None
    media: ShotMediaRange
    limitations: tuple[SessionLimitCode, ...] = ()
    live_evidence_digest: str | None = None
    live_job_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.session_id, name="shot session ID")
        _require_text(self.shot_id, name="shot ID")
        if self.sequence <= 0 or self.started_ms < 0:
            raise ValueError("shot sequence must be positive and start must be non-negative")
        if self.settled_ms is not None and self.settled_ms <= self.started_ms:
            raise ValueError("shot must settle after it starts")
        if (self.power_band is None) != (self.power_percent is None):
            raise ValueError("shot power band and percent must appear together")
        if self.power_percent is not None and not 1 <= self.power_percent <= 100:
            raise ValueError("shot power percent must be between one and one hundred")
        if self.recommendation_id is not None:
            _require_text(self.recommendation_id, name="recommendation ID")
        if not self.media.start_ms <= self.started_ms < self.media.end_ms:
            raise ValueError("shot start must be inside its media range")
        if self.settled_ms is not None and self.settled_ms > self.media.end_ms:
            raise ValueError("shot settle must be inside its media range")
        _require_unique_limitations(self.limitations)
        if (self.live_evidence_digest is None) != (self.live_job_id is None):
            raise ValueError("live evidence digest and job identity must appear together")
        if self.live_evidence_digest is not None:
            if self.media.source_kind != "usb_live" or self.settled_ms is None:
                raise ValueError("live evidence requires settled USB media")
            if len(self.live_evidence_digest) != 64 or any(
                c not in "0123456789abcdef" for c in self.live_evidence_digest
            ):
                raise ValueError("live evidence digest must be SHA-256")
            _require_text(self.live_job_id or "", name="live job identity")


@dataclass(frozen=True, slots=True)
class LiveShotFact:
    """Small per-shot cloud projection; recording bytes stay on the field machine."""

    session_id: str
    shot_id: str
    sequence: int
    start_frame_id: int
    settle_frame_id: int
    mode: LiveMatchMode
    actor_role: WristbandRole
    actor_user_id: str
    observed_outcome: str
    rule_revision: int
    recording_reference: str
    idempotency_key: str

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "shot_id",
            "actor_user_id",
            "observed_outcome",
            "recording_reference",
            "idempotency_key",
        ):
            _require_text(getattr(self, name), name=name)
        if (
            self.sequence <= 0
            or self.start_frame_id < 0
            or self.settle_frame_id <= self.start_frame_id
            or self.rule_revision < 0
            or self.mode not in {"assistance", "observation"}
            or not isinstance(self.actor_role, WristbandRole)
        ):
            raise ValueError("live shot fact is invalid")
        if self.idempotency_key != self._expected_idempotency_key():
            raise ValueError("live shot fact idempotency key is invalid")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        shot_id: str,
        sequence: int,
        start_frame_id: int,
        settle_frame_id: int,
        mode: LiveMatchMode,
        actor_role: WristbandRole,
        actor_user_id: str,
        observed_outcome: str,
        rule_revision: int,
        recording_reference: str,
    ) -> LiveShotFact:
        key = cls._idempotency_key(
            session_id=session_id,
            shot_id=shot_id,
            sequence=sequence,
            start_frame_id=start_frame_id,
            settle_frame_id=settle_frame_id,
        )
        return cls(
            session_id=session_id,
            shot_id=shot_id,
            sequence=sequence,
            start_frame_id=start_frame_id,
            settle_frame_id=settle_frame_id,
            mode=mode,
            actor_role=actor_role,
            actor_user_id=actor_user_id,
            observed_outcome=observed_outcome,
            rule_revision=rule_revision,
            recording_reference=recording_reference,
            idempotency_key=key,
        )

    def _expected_idempotency_key(self) -> str:
        return self._idempotency_key(
            session_id=self.session_id,
            shot_id=self.shot_id,
            sequence=self.sequence,
            start_frame_id=self.start_frame_id,
            settle_frame_id=self.settle_frame_id,
        )

    @staticmethod
    def _idempotency_key(
        *,
        session_id: str,
        shot_id: str,
        sequence: int,
        start_frame_id: int,
        settle_frame_id: int,
    ) -> str:
        identity = {
            "session_id": session_id,
            "shot_id": shot_id,
            "sequence": sequence,
            "start_frame_id": start_frame_id,
            "settle_frame_id": settle_frame_id,
        }
        return (
            "live-shot-"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class LiveShotFactOutboxItem:
    fact: LiveShotFact
    payload_json: str
    status: LiveShotFactOutboxStatus
    attempts: int
    lease_until_utc: str | None

    @property
    def idempotency_key(self) -> str:
        return self.fact.idempotency_key


@dataclass(frozen=True, slots=True)
class FieldFactReport:
    session_id: str
    artifact_id: str
    created_at_utc: str
    ended_at_utc: str
    match: MatchControlSnapshot
    shots: tuple[FieldShotRecord, ...]
    limitations: tuple[SessionLimitCode, ...]

    def __post_init__(self) -> None:
        _require_text(self.session_id, name="report session ID")
        _require_text(self.artifact_id, name="report artifact ID")
        _require_utc_text(self.created_at_utc, name="report created timestamp")
        _require_utc_text(self.ended_at_utc, name="report ended timestamp")
        if any(shot.session_id != self.session_id for shot in self.shots):
            raise ValueError("report shots must belong to its session")
        _require_unique_limitations(self.limitations)


@dataclass(frozen=True, slots=True)
class ShotOwnershipReceipt:
    """Internal start-time binding, never a reviewed-fact or publication approval."""

    session_id: str
    shot_id: str
    sequence: int
    actor_user_id: str | None
    actor_role: WristbandRole | None
    match_revision: int
    source_artifact_id: str
    source_mainline1_artifact_id: str | None
    source_started_ms: int
    recorded_at_utc: str
    evidence_status: Literal["bound_at_start", "unknown"]
    source_kind: Literal["development_replay", "usb_live"] = "development_replay"
    revision: int = 1
    supersedes_revision: None = None
    source_start_event_id: str | None = None
    evidence_reason: str | None = None
    source_epoch_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "shot_id", "source_artifact_id"):
            _require_text(getattr(self, name), name=name)
        if self.sequence <= 0 or self.match_revision < 0 or self.source_started_ms < 0:
            raise ValueError("invalid ownership sequence/revision/time")
        _require_utc_text(self.recorded_at_utc, name="ownership service time")
        source = self.source_mainline1_artifact_id
        if source is not None and (
            len(source) != 64 or any(c not in "0123456789abcdef" for c in source)
        ):
            raise ValueError("invalid ownership Mainline 1 source")
        if self.evidence_status == "bound_at_start":
            if (
                not self.actor_user_id
                or not isinstance(self.actor_role, WristbandRole)
                or (source is None and self.source_kind != "usb_live")
            ):
                raise ValueError("bound ownership requires user, role and source")
        elif (
            self.evidence_status != "unknown"
            or self.actor_user_id is not None
            or self.actor_role is not None
        ):
            raise ValueError("unknown ownership cannot assert a user or role")
        if (
            self.source_kind not in {"development_replay", "usb_live"}
            or self.revision != 1
            or self.supersedes_revision is not None
        ):
            raise ValueError("unsupported ownership source or revision")
        if self.source_kind == "usb_live":
            _require_text(self.source_epoch_id or "", name="USB ownership epoch")
            _require_text(self.source_start_event_id or "", name="USB start event")
        elif self.source_epoch_id is not None:
            raise ValueError("replay ownership cannot assert a USB epoch")


@dataclass(frozen=True, slots=True)
class RuleDecisionReceipt:
    """Immutable audit receipt for one realtime rules-engine attempt."""

    session_id: str
    shot_id: str
    source_epoch_id: str
    start_frame_id: int
    settled_frame_id: int
    settled_evidence_digest: str
    profile_id: str
    facts_json: str
    fact_digest: str
    decision_json: str
    decision_digest: str
    status: RuleDecisionStatus
    applied_command_id: str | None
    elapsed_ms: float
    created_at_utc: str

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "shot_id",
            "source_epoch_id",
            "profile_id",
            "facts_json",
            "fact_digest",
            "decision_json",
            "decision_digest",
        ):
            _require_text(getattr(self, name), name=name)
        if self.start_frame_id < 0 or self.settled_frame_id < self.start_frame_id:
            raise ValueError("rule receipt frame interval is invalid")
        for digest in (
            self.settled_evidence_digest,
            self.fact_digest,
            self.decision_digest,
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("rule receipt digest must be SHA-256")
        if self.status == "applied":
            _require_text(self.applied_command_id or "", name="applied rule command")
        elif self.applied_command_id is not None:
            raise ValueError("non-applied rule receipt cannot name a command")
        if self.elapsed_ms < 0:
            raise ValueError("rule receipt elapsed time must be non-negative")
        _require_utc_text(self.created_at_utc, name="rule receipt timestamp")


@dataclass(frozen=True, slots=True)
class CommandLedgerEntry:
    """One locally ordered phone or rules-engine authority result."""

    entry_id: str
    session_id: str
    ledger_order: int
    origin: Literal["phone", "rules_engine"]
    principal_id: str
    origin_sequence: int | None
    command_type: str
    payload_digest: str
    outcome: Literal["applied", "rejected"]
    result_code: str
    resulting_revision: int
    decision_ref_json: str | None
    completed_at_utc: str

    def __post_init__(self) -> None:
        for name in (
            "entry_id",
            "session_id",
            "principal_id",
            "command_type",
            "payload_digest",
            "result_code",
        ):
            _require_text(getattr(self, name), name=name)
        if self.ledger_order <= 0 or self.resulting_revision < 0:
            raise ValueError("ledger order/revision is invalid")
        if self.origin == "phone":
            if self.origin_sequence is None or self.origin_sequence <= 0:
                raise ValueError("phone ledger entry requires a positive origin sequence")
            if self.decision_ref_json is not None:
                raise ValueError("phone ledger entry cannot reference a rule decision")
        elif self.origin_sequence is not None or self.decision_ref_json is None:
            raise ValueError("rules ledger entry requires only a decision reference")
        if len(self.payload_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.payload_digest
        ):
            raise ValueError("ledger payload digest must be SHA-256")
        _require_utc_text(self.completed_at_utc, name="ledger completion timestamp")


@dataclass(frozen=True, slots=True)
class RecordingEndV1:
    """Frozen references to original shot authority and the sealed raw manifest."""

    contract: str
    session_id: str
    source_epoch_id: str
    policy: str
    end_reason: str
    cutoff: dict[str, int] | None
    raw_manifest_path: str
    raw_manifest_sha256: str
    raw_segments: list[dict[str, object]]
    first_frame_id: int | None
    last_frame_id: int | None
    width: int
    height: int
    shots: list[dict[str, str]]
    media_status: str
    analysis_status: str
    encoding_status: str
    ended_at_utc: str | None

    def _expected_identity(self):
        return (
            "recording-end/v1",
            "recording-v1",
            "USER_ENDED_RECORDING",
            "NOT_REQUESTED",
            "NOT_SCHEDULED",
        )

    def __post_init__(self):
        if (
            self.contract,
            self.policy,
            self.end_reason,
            self.analysis_status,
            self.encoding_status,
        ) != self._expected_identity():
            raise ValueError("invalid recording end contract")
        for value in (self.session_id, self.source_epoch_id, self.raw_manifest_path):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid recording end identity")
        if self.media_status not in {"RECORDED", "RETAINED_INCOMPLETE", "EMPTY"}:
            raise ValueError("invalid recording end media status")
        if self.cutoff is not None:
            if (
                not isinstance(self.cutoff, dict)
                or set(self.cutoff) != {"frame_id", "captured_at_ns"}
                or any(type(v) is not int or v < 0 for v in self.cutoff.values())
                or self.last_frame_id != self.cutoff["frame_id"]
            ):
                raise ValueError("invalid recording cutoff")
            if (
                type(self.first_frame_id) is not int
                or not 0 <= self.first_frame_id <= self.last_frame_id
                or type(self.width) is not int
                or self.width <= 0
                or type(self.height) is not int
                or self.height <= 0
            ):
                raise ValueError("invalid recording geometry or frame range")
        elif (
            self.first_frame_id is not None
            or self.last_frame_id is not None
            or self.width != 0
            or self.height != 0
            or self.shots
            or self.media_status != "EMPTY"
        ):
            raise ValueError("invalid empty recording")
        if not isinstance(self.shots, list):
            raise ValueError("invalid recording shot references")
        ids = set()
        digests = [self.raw_manifest_sha256]
        if not isinstance(self.raw_segments, list):
            raise ValueError("invalid recording segment receipts")
        names = set()
        for segment in self.raw_segments:
            if (
                not isinstance(segment, dict)
                or set(segment)
                != {"name", "sha256", "size", "device", "inode", "mtime_ns", "ctime_ns"}
                or not isinstance(segment["name"], str)
                or not segment["name"]
                or segment["name"] in names
                or any(
                    type(segment[k]) is not int or segment[k] < 0
                    for k in ("size", "device", "inode", "mtime_ns", "ctime_ns")
                )
            ):
                raise ValueError("invalid recording segment receipt")
            names.add(segment["name"])
            digests.append(segment["sha256"])
        for shot in self.shots:
            if (
                not isinstance(shot, dict)
                or set(shot) != {"shot_id", "payload_sha256"}
                or not isinstance(shot["shot_id"], str)
                or not shot["shot_id"]
                or shot["shot_id"] in ids
            ):
                raise ValueError("invalid recording shot reference")
            ids.add(shot["shot_id"])
            digests.append(shot["payload_sha256"])
        if any(
            not isinstance(d, str) or len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
            for d in digests
        ):
            raise ValueError("invalid recording end digest")
        if self.ended_at_utc is not None:
            _require_utc_text(self.ended_at_utc, name="recording end timestamp")


@dataclass(frozen=True, slots=True)
class RecordingInterruptedEndV2(RecordingEndV1):
    """Only the committed prefix is final; actual capture tail remains unknown."""

    tail_completeness: str

    def _expected_identity(self):
        return (
            "recording-end/v2",
            "recording-v1",
            "PROCESS_INTERRUPTED",
            "NOT_REQUESTED",
            "NOT_SCHEDULED",
        )

    def __post_init__(self):
        RecordingEndV1.__post_init__(self)
        if self.tail_completeness != "UNKNOWN":
            raise ValueError("invalid interrupted recording completeness")


def recording_end_from_payload(payload):
    cls = (
        RecordingInterruptedEndV2
        if isinstance(payload, dict) and payload.get("contract") == "recording-end/v2"
        else RecordingEndV1
    )
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(cls)}:
        raise ValueError("invalid recording end fields")
    try:
        return asdict(cls(**payload))
    except (TypeError, AttributeError) as exc:
        raise ValueError("invalid recording end payload") from exc


def ownership_receipt_from_payload(
    payload: object, *, allow_legacy: bool = False
) -> ShotOwnershipReceipt:
    """Load immutable ownership; legacy reads preserve absent optional evidence."""
    if allow_legacy and isinstance(payload, dict):
        payload = dict(payload)
        for item in fields(ShotOwnershipReceipt):
            if item.name not in payload and item.default is not MISSING:
                payload[item.name] = item.default
    if not isinstance(payload, dict) or set(payload) != {
        item.name for item in fields(ShotOwnershipReceipt)
    }:
        raise ValueError("invalid ownership receipt fields")
    values = dict(payload)
    for name in ("session_id", "shot_id", "source_artifact_id", "recorded_at_utc"):
        if not isinstance(values[name], str):
            raise ValueError("invalid ownership receipt text")
        _require_text(values[name], name=name)
    for name in (
        "actor_user_id",
        "source_mainline1_artifact_id",
        "source_start_event_id",
        "evidence_reason",
        "source_epoch_id",
    ):
        if values[name] is not None:
            if not isinstance(values[name], str):
                raise ValueError("invalid ownership receipt text")
            _require_text(values[name], name=name)
    try:
        for value in values.values():
            if isinstance(value, str):
                value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("ownership receipt text must be valid UTF-8") from exc
    for name in ("sequence", "match_revision", "source_started_ms", "revision"):
        value = values[name]
        if type(value) is not int or not 0 <= value <= 9_007_199_254_740_991:
            raise ValueError("invalid ownership receipt integer")
    try:
        if values["actor_role"] is not None:
            values["actor_role"] = WristbandRole(values["actor_role"])
        return ShotOwnershipReceipt(**values)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError("invalid ownership receipt") from exc


def ownership_receipt_json(receipt: ShotOwnershipReceipt) -> str:
    payload = asdict(receipt)
    ownership_receipt_from_payload(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ownership_receipt_sha256(receipt: ShotOwnershipReceipt) -> str:
    return hashlib.sha256(ownership_receipt_json(receipt).encode("utf-8")).hexdigest()


def validate_shot_ownership(
    receipt: ShotOwnershipReceipt,
    shot: FieldShotRecord,
    media: LiveShotMediaReference | None = None,
) -> ShotOwnershipReceipt:
    """Verify start authority against the frozen shot, and materialized USB lineage."""
    ownership_receipt_json(receipt)
    if (
        receipt.evidence_status != "bound_at_start"
        or receipt.actor_role not in {WristbandRole.RED, WristbandRole.BLUE}
        or not receipt.actor_user_id
        or receipt.source_kind != "usb_live"
        or (
            receipt.session_id,
            receipt.shot_id,
            receipt.sequence,
            receipt.actor_role,
            receipt.source_started_ms,
            receipt.source_artifact_id,
            receipt.source_epoch_id,
        )
        != (
            shot.session_id,
            shot.shot_id,
            shot.sequence,
            shot.actor_role,
            shot.started_ms,
            shot.media.artifact_id,
            shot.media.source_epoch_id,
        )
        or shot.media.source_kind != "usb_live"
    ):
        raise ValueError("shot ownership is unavailable")
    if media is not None and (
        (media.session_id, media.shot_id, media.source_epoch_id, media.match_revision, media.job_id)
        != (
            shot.session_id,
            shot.shot_id,
            receipt.source_epoch_id,
            receipt.match_revision,
            shot.live_job_id,
        )
        or type(media.match_revision) is not int
        or not 0 <= media.match_revision <= 9_007_199_254_740_991
    ):
        raise ValueError("shot ownership media lineage is unavailable")
    return receipt
