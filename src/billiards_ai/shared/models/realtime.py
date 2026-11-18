"""Realtime observation and feedback contracts consumed by CoachRuntime."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator
from pydantic_core import to_jsonable_python

from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr, Sha256, require_utc
from billiards_ai.shared.models.evidence import FinalizedShotEvidence
from billiards_ai.shared.models.overlay import (
    CoachingDeliveryStatus,
    CoachingOutcomeStatus,
    OverlayPayloadV1,
)

MAX_REALTIME_COACHING_EVENT_BYTES = 65_536


class RealtimeObservationStatus(StrEnum):
    OBSERVED = "OBSERVED"
    ABSTAINED = "ABSTAINED"
    UNSUPPORTED = "UNSUPPORTED"


class RealtimeShotObservation(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID4
    session_id: UUID4
    shot_sequence: StrictInt = Field(ge=1)
    monotonic_timestamp_ms: StrictInt = Field(ge=0)
    shot: FinalizedShotEvidence
    status: RealtimeObservationStatus
    evidence_sha256: Sha256
    perception_model_version: NonEmptyStr
    calibration_revision: StrictInt = Field(ge=1)
    observed_at: datetime

    _observed_at_utc = field_validator("observed_at")(require_utc)

    @model_validator(mode="after")
    def canonical_shot_matches_runtime_identity(self) -> RealtimeShotObservation:
        if self.shot.session_id != self.session_id:
            raise ValueError("realtime observation and shot session must match")
        if self.shot.sequence != self.shot_sequence:
            raise ValueError("realtime observation and shot sequence must match")
        if (
            self.status is RealtimeObservationStatus.ABSTAINED
            and self.shot.confidence.value != "abstain"
        ):
            raise ValueError("abstained observation requires an abstained shot")
        return self


class RealtimeFeedbackKind(StrEnum):
    OVERLAY = "OVERLAY"
    SPEECH = "SPEECH"
    TEXT = "TEXT"
    SAFE_PAUSE = "SAFE_PAUSE"


class RealtimeCoachingEvent(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID4
    session_id: UUID4
    shot_sequence: StrictInt = Field(ge=1)
    kind: RealtimeFeedbackKind
    sequence: StrictInt = Field(ge=1)
    payload_code: NonEmptyStr
    created_at: datetime

    _created_at_utc = field_validator("created_at")(require_utc)


class CoachingEventType(StrEnum):
    SETUP = "SETUP"
    PRE_SHOT_CUE = "PRE_SHOT_CUE"
    POST_SHOT_FEEDBACK = "POST_SHOT_FEEDBACK"
    PROGRESSION = "PROGRESSION"
    REGRESSION = "REGRESSION"
    ABSTENTION = "ABSTENTION"
    REST_SUGGESTION = "REST_SUGGESTION"
    SESSION_SUMMARY = "SESSION_SUMMARY"
    SYSTEM_STATUS = "SYSTEM_STATUS"


class SpeechCueV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    locale: Literal["zh-CN"] = "zh-CN"
    cue_code: NonEmptyStr
    outcome_status: CoachingOutcomeStatus
    delivery_status: CoachingDeliveryStatus
    text: NonEmptyStr


class RealtimeCoachingLineageV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    admission_id: NonEmptyStr
    bundle_id: UUID4
    bundle_content_sha256: Sha256
    projection_sha256: Sha256
    session_id: UUID4
    plan_id: UUID4
    plan_version: StrictInt = Field(ge=1)
    block_id: UUID4
    block_order: StrictInt = Field(ge=1)
    drill_id: NonEmptyStr
    drill_revision: StrictInt = Field(ge=1)
    drill_content_sha256: Sha256
    shot_id: UUID4 | None = None
    shot_sequence: StrictInt | None = Field(default=None, ge=1)
    evidence_sha256: Sha256 | None = None
    decision_id: Sha256 | None = None
    effect_id: Sha256

    @model_validator(mode="after")
    def optional_shot_lineage_is_atomic(self) -> RealtimeCoachingLineageV1:
        shot_values = (self.shot_id, self.shot_sequence, self.evidence_sha256)
        if any(item is not None for item in shot_values) and not all(
            item is not None for item in shot_values
        ):
            raise ValueError("shot lineage must be wholly present or absent")
        if self.decision_id is not None and self.shot_id is None:
            raise ValueError("decision lineage requires shot lineage")
        return self


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(domain: str, material: object) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + _canonical_bytes(material)).hexdigest()


def _event_identity_material(values: dict[str, object]) -> dict[str, object]:
    return {
        key: values[key]
        for key in (
            "schema_version",
            "sequence",
            "event_type",
            "outcome_status",
            "delivery_status",
            "lineage",
            "created_monotonic_ms",
        )
    }


def _uuid4_from_digest(digest: str) -> UUID:
    value = bytearray(bytes.fromhex(digest)[:16])
    value[6] = (value[6] & 0x0F) | 0x40
    value[8] = (value[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(value))


class RealtimeCoachingEventV2(CoreContractModel):
    schema_version: Literal["2.0"] = "2.0"
    event_id: UUID4
    sequence: StrictInt = Field(ge=1)
    event_type: CoachingEventType
    outcome_status: CoachingOutcomeStatus
    delivery_status: CoachingDeliveryStatus
    lineage: RealtimeCoachingLineageV1
    overlay: OverlayPayloadV1 | None = None
    speech: SpeechCueV1 | None = None
    text_summary: NonEmptyStr
    created_monotonic_ms: StrictInt = Field(ge=0)
    idempotency_key: Sha256
    content_sha256: Sha256

    @model_validator(mode="after")
    def statuses_lineage_and_hashes_are_consistent(self) -> RealtimeCoachingEventV2:
        if self.overlay is not None and (
            self.overlay.outcome_status is not self.outcome_status
            or self.overlay.delivery_status is not self.delivery_status
        ):
            raise ValueError("overlay status must match event status")
        if self.speech is not None and (
            self.speech.outcome_status is not self.outcome_status
            or self.speech.delivery_status is not self.delivery_status
        ):
            raise ValueError("speech status must match event status")
        if self.speech is not None and self.speech.text != self.text_summary:
            raise ValueError("speech text must match event text summary")
        if (
            self.delivery_status is not CoachingDeliveryStatus.NORMAL
            or self.outcome_status
            in {CoachingOutcomeStatus.ABSTAINED, CoachingOutcomeStatus.UNSUPPORTED}
        ) and self.speech is None:
            raise ValueError("abstention, degraded, and error events require a speech cue")
        allowed_statuses = {
            CoachingEventType.SETUP: {
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.NORMAL)
            },
            CoachingEventType.PRE_SHOT_CUE: {
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.NORMAL)
            },
            CoachingEventType.POST_SHOT_FEEDBACK: {
                (CoachingOutcomeStatus.SUCCESS, CoachingDeliveryStatus.NORMAL),
                (CoachingOutcomeStatus.FAILURE, CoachingDeliveryStatus.NORMAL),
            },
            CoachingEventType.PROGRESSION: {
                (CoachingOutcomeStatus.SUCCESS, CoachingDeliveryStatus.NORMAL)
            },
            CoachingEventType.REGRESSION: {
                (CoachingOutcomeStatus.FAILURE, CoachingDeliveryStatus.NORMAL)
            },
            CoachingEventType.ABSTENTION: {
                (CoachingOutcomeStatus.ABSTAINED, CoachingDeliveryStatus.NORMAL),
                (CoachingOutcomeStatus.UNSUPPORTED, CoachingDeliveryStatus.NORMAL),
            },
            CoachingEventType.REST_SUGGESTION: {
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.NORMAL)
            },
            CoachingEventType.SESSION_SUMMARY: {
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.NORMAL)
            },
            CoachingEventType.SYSTEM_STATUS: {
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.NORMAL),
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.DEGRADED),
                (CoachingOutcomeStatus.INFORMATIONAL, CoachingDeliveryStatus.ERROR),
                (CoachingOutcomeStatus.ABSTAINED, CoachingDeliveryStatus.DEGRADED),
                (CoachingOutcomeStatus.UNSUPPORTED, CoachingDeliveryStatus.DEGRADED),
            },
        }
        if (self.outcome_status, self.delivery_status) not in allowed_statuses[self.event_type]:
            raise ValueError("event type, outcome, and delivery statuses are inconsistent")
        if self.outcome_status is not CoachingOutcomeStatus.INFORMATIONAL and (
            self.lineage.shot_id is None or self.lineage.decision_id is None
        ):
            raise ValueError("machine outcome requires shot and decision lineage")
        values = self.model_dump(mode="json")
        identity_digest = _sha256(
            "realtime-coaching-event-identity/v2",
            _event_identity_material(values),
        )
        if self.event_id != _uuid4_from_digest(identity_digest):
            raise ValueError("event identity mismatch")
        expected_idempotency = _sha256(
            "realtime-coaching-event-idempotency/v2",
            _event_identity_material(values),
        )
        if self.idempotency_key != expected_idempotency:
            raise ValueError("event idempotency identity mismatch")
        content = {key: value for key, value in values.items() if key != "content_sha256"}
        if self.content_sha256 != _sha256("realtime-coaching-event-content/v2", content):
            raise ValueError("event content hash mismatch")
        return self


RealtimeCoachingEventAny = RealtimeCoachingEvent | RealtimeCoachingEventV2


def issue_realtime_coaching_event_v2(
    *,
    sequence: int,
    event_type: CoachingEventType,
    outcome_status: CoachingOutcomeStatus,
    delivery_status: CoachingDeliveryStatus,
    lineage: RealtimeCoachingLineageV1,
    overlay: OverlayPayloadV1 | None,
    speech: SpeechCueV1 | None,
    text_summary: str,
    created_monotonic_ms: int,
) -> RealtimeCoachingEventV2:
    if type(lineage) is not RealtimeCoachingLineageV1:
        raise TypeError("lineage must be an exact RealtimeCoachingLineageV1")
    lineage = RealtimeCoachingLineageV1.model_validate(lineage.model_dump(), strict=True)
    if overlay is not None:
        if type(overlay) is not OverlayPayloadV1:
            raise TypeError("overlay must be an exact OverlayPayloadV1")
        overlay = OverlayPayloadV1.model_validate(overlay.model_dump(), strict=True)
    if speech is not None:
        if type(speech) is not SpeechCueV1:
            raise TypeError("speech must be an exact SpeechCueV1")
        speech = SpeechCueV1.model_validate(speech.model_dump(), strict=True)
    material: dict[str, object] = {
        "schema_version": "2.0",
        "sequence": sequence,
        "event_type": event_type,
        "outcome_status": outcome_status,
        "delivery_status": delivery_status,
        "lineage": lineage,
        "created_monotonic_ms": created_monotonic_ms,
    }
    identity = _event_identity_material(material)
    identity_digest = _sha256("realtime-coaching-event-identity/v2", identity)
    values: dict[str, object] = {
        **material,
        "event_id": _uuid4_from_digest(identity_digest),
        "overlay": overlay,
        "speech": speech,
        "text_summary": text_summary,
        "idempotency_key": _sha256("realtime-coaching-event-idempotency/v2", identity),
    }
    content = {**values}
    values["content_sha256"] = _sha256("realtime-coaching-event-content/v2", content)
    event = RealtimeCoachingEventV2(**values)  # type: ignore[arg-type]
    if len(_canonical_bytes(event.model_dump(mode="json"))) > MAX_REALTIME_COACHING_EVENT_BYTES:
        raise ValueError("realtime coaching event exceeds maximum wire size")
    return event


def canonical_realtime_coaching_event_bytes(event: RealtimeCoachingEventAny) -> bytes:
    if type(event) not in {RealtimeCoachingEvent, RealtimeCoachingEventV2}:
        raise TypeError("event must be an exact realtime coaching event")
    validated = type(event).model_validate(event.model_dump(), strict=True)
    data = _canonical_bytes(validated.model_dump(mode="json"))
    if len(data) > MAX_REALTIME_COACHING_EVENT_BYTES:
        raise ValueError("realtime coaching event exceeds maximum wire size")
    return data


def decode_realtime_coaching_event(data: bytes) -> RealtimeCoachingEventAny:
    if type(data) is not bytes:
        raise TypeError("realtime coaching event input must be bytes")
    if len(data) > MAX_REALTIME_COACHING_EVENT_BYTES:
        raise ValueError("realtime coaching event exceeds maximum wire size")
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid realtime coaching event JSON") from exc
    if type(raw) is not dict:
        raise ValueError("realtime coaching event must be an object")
    version = raw.get("schema_version")
    if version == "1.0":
        event: RealtimeCoachingEventAny = RealtimeCoachingEvent.model_validate_json(data)
    elif version == "2.0":
        event = RealtimeCoachingEventV2.model_validate_json(data)
    else:
        raise ValueError("unsupported realtime coaching event schema version")
    if canonical_realtime_coaching_event_bytes(event) != data:
        raise ValueError("realtime coaching event JSON must be canonical")
    return event
