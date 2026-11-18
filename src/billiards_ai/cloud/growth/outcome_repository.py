"""Exactly-once PlanOutcome history and single-process atomic reference storage."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import UUID

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from billiards_ai.cloud.growth.repository import stored_model_values
from billiards_ai.shared.models.events import (
    DomainEvent,
    NonEmptyStr,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.growth import PlanOutcome

_UUID4 = TypeAdapter(UUID4)
_PLAN_OUTCOME_EVENT_TYPE = "coach.runtime.plan.completed.v1"
_PLAN_OUTCOME_AGGREGATE_TYPE = "plan_outcome"
PLAN_OUTCOME_EVENT_CLOCK_POLICY_VERSION = "plan-outcome-event-clock-v1"
PLAN_OUTCOME_EVENT_CLOCK_SKEW_TOLERANCE = timedelta(seconds=2)


class OutcomeIngestErrorCode(StrEnum):
    INVALID_OUTCOME = "INVALID_OUTCOME"
    INVALID_EVENT = "INVALID_EVENT"
    RESOURCE_NOT_OWNED = "RESOURCE_NOT_OWNED"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    EVENT_ID_CONFLICT = "EVENT_ID_CONFLICT"
    OUTCOME_ID_CONFLICT = "OUTCOME_ID_CONFLICT"
    BUSINESS_IDENTITY_CONFLICT = "BUSINESS_IDENTITY_CONFLICT"
    HISTORY_ID_CONFLICT = "HISTORY_ID_CONFLICT"
    RECEIPT_ID_CONFLICT = "RECEIPT_ID_CONFLICT"
    CLOCK_POLICY_VIOLATION = "CLOCK_POLICY_VIOLATION"


class OutcomeIngestError(ValueError):
    """Stable failure that never embeds a rejected user payload."""

    def __init__(self, code: OutcomeIngestErrorCode):
        self.code = code
        super().__init__(code.value)


class OutcomeRepositoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OutcomeIngestStatus(StrEnum):
    CREATED = "CREATED"
    REPLAYED = "REPLAYED"


class OutcomeInboxReceipt(OutcomeRepositoryModel):
    receipt_id: UUID4
    history_id: UUID4
    event_id: UUID4
    outcome_id: UUID4
    user_id: UUID4
    idempotency_key: NonEmptyStr
    received_at: datetime

    _received_at_utc = field_validator("received_at")(require_utc)


class PlanOutcomeHistoryRecord(OutcomeRepositoryModel):
    history_id: UUID4
    receipt_id: UUID4
    outcome: PlanOutcome
    source_event: DomainEvent
    received_at: datetime

    _received_at_utc = field_validator("received_at")(require_utc)


class PlanOutcomeIngestResult(OutcomeRepositoryModel):
    status: OutcomeIngestStatus
    record: PlanOutcomeHistoryRecord
    receipt: OutcomeInboxReceipt


class PlanOutcomeRepository(Protocol):
    def append(
        self,
        outcome: PlanOutcome,
        *,
        source_event: DomainEvent,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> PlanOutcomeIngestResult: ...

    def list_for_user(self, user_id: UUID) -> tuple[PlanOutcomeHistoryRecord, ...]: ...


def revalidate_plan_outcome(outcome: PlanOutcome) -> PlanOutcome:
    try:
        values = stored_model_values(outcome)
        if not isinstance(values, dict):
            raise TypeError("PlanOutcome must be an object")
        return PlanOutcome.model_validate(values, strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
        raise OutcomeIngestError(OutcomeIngestErrorCode.INVALID_OUTCOME) from exc


def revalidate_outcome_event(source_event: DomainEvent) -> DomainEvent:
    try:
        values = stored_model_values(source_event)
        if not isinstance(values, dict):
            raise TypeError("DomainEvent must be an object")
        return DomainEvent.model_validate(values, strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
        raise OutcomeIngestError(OutcomeIngestErrorCode.INVALID_EVENT) from exc


def plan_outcome_event_payload(outcome: PlanOutcome) -> dict[str, object]:
    """Bind every outcome field plus a digest independent of its opaque content hash."""

    full_outcome = outcome.model_dump(mode="json")
    authoritative_outcome = outcome.model_dump(mode="json", exclude={"content_sha256"})
    return {
        "plan_outcome": full_outcome,
        "authoritative_sha256": canonical_sha256(authoritative_outcome),
    }


def _event_payload_matches_outcome(payload: object, outcome: PlanOutcome) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "plan_outcome",
        "authoritative_sha256",
    }:
        return False
    embedded = payload.get("plan_outcome")
    digest = payload.get("authoritative_sha256")
    if not isinstance(embedded, dict) or not isinstance(digest, str):
        return False
    authoritative = dict(embedded)
    authoritative.pop("content_sha256", None)
    return (
        digest == canonical_sha256(authoritative)
        and embedded == outcome.model_dump(mode="json")
        and payload == plan_outcome_event_payload(outcome)
    )


def validate_outcome_event_lineage(
    outcome: PlanOutcome, source_event: DomainEvent
) -> tuple[PlanOutcome, DomainEvent]:
    validated_outcome = revalidate_plan_outcome(outcome)
    validated_event = revalidate_outcome_event(source_event)
    if validated_event.event_type != _PLAN_OUTCOME_EVENT_TYPE:
        raise OutcomeIngestError(OutcomeIngestErrorCode.INVALID_EVENT)
    if (
        validated_event.aggregate_type != _PLAN_OUTCOME_AGGREGATE_TYPE
        or validated_event.aggregate_id != validated_outcome.outcome_id
        or validated_event.aggregate_version != validated_outcome.plan_version
        or validated_event.causation_id != validated_outcome.plan_id
        or validated_event.correlation_id != validated_outcome.session_id
        or not _event_payload_matches_outcome(validated_event.payload, validated_outcome)
    ):
        raise OutcomeIngestError(OutcomeIngestErrorCode.LINEAGE_MISMATCH)
    if validated_event.occurred_at < validated_outcome.completed_at:
        raise OutcomeIngestError(OutcomeIngestErrorCode.CLOCK_POLICY_VIOLATION)
    return validated_outcome, validated_event


def _validate_received_clock(
    outcome: PlanOutcome,
    source_event: DomainEvent,
    received_at: datetime,
) -> datetime:
    try:
        received_at = require_utc(received_at)
    except (AttributeError, TypeError, ValueError) as exc:
        raise OutcomeIngestError(OutcomeIngestErrorCode.CLOCK_POLICY_VIOLATION) from exc
    if not (
        outcome.completed_at
        <= source_event.occurred_at
        <= received_at + PLAN_OUTCOME_EVENT_CLOCK_SKEW_TOLERANCE
    ):
        raise OutcomeIngestError(OutcomeIngestErrorCode.CLOCK_POLICY_VIOLATION)
    return received_at


@dataclass(frozen=True, slots=True)
class _OutcomeRepositoryState:
    """All six authority tables; a commit publishes one new state reference."""

    history: tuple[tuple[UUID, PlanOutcomeHistoryRecord], ...] = ()
    receipts: tuple[tuple[UUID, OutcomeInboxReceipt], ...] = ()
    by_idempotency: tuple[tuple[str, UUID], ...] = ()
    by_event_id: tuple[tuple[UUID, UUID], ...] = ()
    by_outcome_id: tuple[tuple[UUID, UUID], ...] = ()
    by_business_identity: tuple[tuple[tuple[UUID, UUID], UUID], ...] = ()

    def as_tuple(self) -> tuple[object, ...]:
        return (
            self.history,
            self.receipts,
            self.by_idempotency,
            self.by_event_id,
            self.by_outcome_id,
            self.by_business_identity,
        )


class InMemoryPlanOutcomeRepository:
    """Single-process reference repository; durable storage belongs to Story 8.6."""

    def __init__(self, *, failure_injector: Callable[[str], None] | None = None):
        self._lock = RLock()
        self._failure_injector = failure_injector
        self._commit_in_progress = False
        self._state = _OutcomeRepositoryState()

    def _inject(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)

    @staticmethod
    def _result(
        state: _OutcomeRepositoryState,
        history_id: UUID,
        status: OutcomeIngestStatus,
    ) -> PlanOutcomeIngestResult:
        record = dict(state.history)[history_id]
        return PlanOutcomeIngestResult(
            status=status,
            record=record,
            receipt=dict(state.receipts)[record.receipt_id],
        )

    @classmethod
    def _resolve_state(
        cls,
        state: _OutcomeRepositoryState,
        outcome: PlanOutcome,
        source_event: DomainEvent,
    ) -> PlanOutcomeIngestResult | None:
        by_idempotency = dict(state.by_idempotency)
        by_event_id = dict(state.by_event_id)
        by_outcome_id = dict(state.by_outcome_id)
        by_business_identity = dict(state.by_business_identity)
        history = dict(state.history)
        history_id = by_idempotency.get(source_event.idempotency_key)
        if history_id is not None:
            existing = history[history_id]
            if existing.outcome == outcome and existing.source_event == source_event:
                return cls._result(state, history_id, OutcomeIngestStatus.REPLAYED)
            raise OutcomeIngestError(OutcomeIngestErrorCode.IDEMPOTENCY_CONFLICT)

        reused_event = by_event_id.get(source_event.event_id)
        if reused_event is not None:
            raise OutcomeIngestError(OutcomeIngestErrorCode.EVENT_ID_CONFLICT)
        reused_outcome = by_outcome_id.get(outcome.outcome_id)
        if reused_outcome is not None:
            raise OutcomeIngestError(OutcomeIngestErrorCode.OUTCOME_ID_CONFLICT)
        business_identity = (outcome.plan_id, outcome.session_id)
        reused_business = by_business_identity.get(business_identity)
        if reused_business is not None:
            raise OutcomeIngestError(OutcomeIngestErrorCode.BUSINESS_IDENTITY_CONFLICT)
        return None

    def append(
        self,
        outcome: PlanOutcome,
        *,
        source_event: DomainEvent,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> PlanOutcomeIngestResult:
        validated_outcome, validated_event = validate_outcome_event_lineage(outcome, source_event)
        with self._lock:
            if self._commit_in_progress:
                raise RuntimeError("nested outcome repository commit")
            if existing := self._resolve_state(self._state, validated_outcome, validated_event):
                return deepcopy(existing)

        # External callbacks run outside the commit lock.  A callback or competing
        # writer may commit meanwhile, so identity resolution is repeated below.
        received_at = _validate_received_clock(validated_outcome, validated_event, clock())
        try:
            history_id = _UUID4.validate_python(uuid_factory(), strict=True)
            receipt_id = _UUID4.validate_python(uuid_factory(), strict=True)
            record = PlanOutcomeHistoryRecord(
                history_id=history_id,
                receipt_id=receipt_id,
                outcome=validated_outcome,
                source_event=validated_event,
                received_at=received_at,
            )
            receipt = OutcomeInboxReceipt(
                receipt_id=receipt_id,
                history_id=history_id,
                event_id=validated_event.event_id,
                outcome_id=validated_outcome.outcome_id,
                user_id=validated_outcome.user_id,
                idempotency_key=validated_event.idempotency_key,
                received_at=received_at,
            )
        except (TypeError, ValidationError, ValueError) as exc:
            raise OutcomeIngestError(OutcomeIngestErrorCode.INVALID_EVENT) from exc

        with self._lock:
            if self._commit_in_progress:
                raise RuntimeError("nested outcome repository commit")
            state = self._state
            if existing := self._resolve_state(state, validated_outcome, validated_event):
                return deepcopy(existing)
            history = dict(state.history)
            receipts = dict(state.receipts)
            if history_id in history:
                raise OutcomeIngestError(OutcomeIngestErrorCode.HISTORY_ID_CONFLICT)
            if receipt_id in receipts:
                raise OutcomeIngestError(OutcomeIngestErrorCode.RECEIPT_ID_CONFLICT)

            self._commit_in_progress = True
            try:
                history[history_id] = record
                self._inject("history")
                receipts[receipt_id] = receipt
                self._inject("receipt")
                by_idempotency = dict(state.by_idempotency)
                by_idempotency[validated_event.idempotency_key] = history_id
                self._inject("idempotency")
                by_event_id = dict(state.by_event_id)
                by_event_id[validated_event.event_id] = history_id
                self._inject("event")
                by_outcome_id = dict(state.by_outcome_id)
                by_outcome_id[validated_outcome.outcome_id] = history_id
                self._inject("outcome")
                by_business_identity = dict(state.by_business_identity)
                by_business_identity[(validated_outcome.plan_id, validated_outcome.session_id)] = (
                    history_id
                )
                self._inject("business")

                staged = _OutcomeRepositoryState(
                    history=tuple(history.items()),
                    receipts=tuple(receipts.items()),
                    by_idempotency=tuple(by_idempotency.items()),
                    by_event_id=tuple(by_event_id.items()),
                    by_outcome_id=tuple(by_outcome_id.items()),
                    by_business_identity=tuple(by_business_identity.items()),
                )
                self._state = staged
            finally:
                self._commit_in_progress = False
            return deepcopy(self._result(self._state, history_id, OutcomeIngestStatus.CREATED))

    def list_for_user(self, user_id: UUID) -> tuple[PlanOutcomeHistoryRecord, ...]:
        with self._lock:
            history = dict(self._state.history)
            values = (record for record in history.values() if record.outcome.user_id == user_id)
            return deepcopy(
                tuple(
                    sorted(
                        values,
                        key=lambda record: (
                            record.outcome.started_at,
                            record.outcome.completed_at,
                            str(record.outcome.plan_id),
                            str(record.outcome.session_id),
                            str(record.outcome.outcome_id),
                        ),
                    )
                )
            )

    def list_receipts(self) -> tuple[OutcomeInboxReceipt, ...]:
        with self._lock:
            receipts = dict(self._state.receipts)
            return deepcopy(
                tuple(
                    sorted(
                        receipts.values(),
                        key=lambda receipt: (receipt.received_at, str(receipt.receipt_id)),
                    )
                )
            )

    def list_outbox_events(self) -> tuple[()]:
        """Outcome ingestion emits no growth snapshot event."""

        return ()

    def snapshot(self) -> tuple[object, ...]:
        with self._lock:
            return deepcopy(self._state.as_tuple())
