"""Append-only, single-process reference repository for Story 9.6 drafts."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Literal, Self
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, ValidationError, model_validator

from billiards_ai.cloud.coach_agent.context import _stored_values
from billiards_ai.cloud.coach_agent.prompts import (
    TRAINING_PLANNING_PROMPT_V1_SHA256,
)
from billiards_ai.cloud.coach_agent.training_planning import (
    TrainingPlanDraft,
    TrainingPlanningFailure,
)
from billiards_ai.shared.models.coaching import AgentRun, AgentRunStatus
from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
)


class TrainingPlanRepositoryErrorCode(StrEnum):
    INVALID_RECORD = "INVALID_RECORD"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    AUTHORITY_UNCHANGED = "AUTHORITY_UNCHANGED"
    NOT_FOUND = "NOT_FOUND"


class TrainingPlanRepositoryError(RuntimeError):
    def __init__(self, code: TrainingPlanRepositoryErrorCode):
        self.code = code
        super().__init__(code.value)


class TrainingPlanLifecycleStatus(StrEnum):
    DRAFT = "DRAFT"
    SUPERSEDED = "SUPERSEDED"


class TrainingPlanLifecycleRecord(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    record_id: UUID4
    sequence: StrictInt = Field(ge=1)
    user_id: UUID4
    plan_id: UUID4
    plan_version: StrictInt = Field(ge=1)
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    assessment_content_sha256: Sha256
    catalog_version: NonEmptyStr
    catalog_heads_sha256: Sha256
    status: TrainingPlanLifecycleStatus
    superseded_by_plan_id: UUID4 | None = None
    recorded_at: datetime
    previous_record_sha256: Sha256 | None = None
    record_sha256: Sha256

    @model_validator(mode="after")
    def lifecycle_is_closed(self) -> Self:
        from billiards_ai.shared.models.events import require_utc

        require_utc(self.recorded_at)
        if (self.status is TrainingPlanLifecycleStatus.SUPERSEDED) != (
            self.superseded_by_plan_id is not None
        ):
            raise ValueError("superseded lifecycle must name the replacement plan")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_sha256"}))
        if self.record_sha256 != expected:
            raise ValueError("training plan lifecycle hash mismatch")
        return self


class TrainingPlanSaveCommand(CoreContractModel):
    draft: TrainingPlanDraft
    agent_run: AgentRun
    expected_current_plan_version: StrictInt = Field(ge=0)
    idempotency_key: NonEmptyStr


class TrainingPlanSaveResult(CoreContractModel):
    draft: TrainingPlanDraft
    agent_run: AgentRun
    lifecycle_records: tuple[TrainingPlanLifecycleRecord, ...] = Field(min_length=1)


class TrainingPlanFailureSaveCommand(CoreContractModel):
    user_id: UUID4
    failure: TrainingPlanningFailure
    idempotency_key: NonEmptyStr


class TrainingPlanFailureSaveResult(CoreContractModel):
    user_id: UUID4
    failure: TrainingPlanningFailure


type TrainingPlanPersistResult = TrainingPlanSaveResult | TrainingPlanFailureSaveResult


def _uuid4_from_payload(value: object) -> UUID:
    raw = bytearray(bytes.fromhex(canonical_sha256(value))[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _clone[T: CoreContractModel](model: type[T], value: T) -> T:
    try:
        return model.model_validate(_stored_values(value), strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise TrainingPlanRepositoryError(TrainingPlanRepositoryErrorCode.INVALID_RECORD) from None


def _record(
    *,
    draft: TrainingPlanDraft,
    status: TrainingPlanLifecycleStatus,
    sequence: int,
    previous_record_sha256: str | None,
    superseded_by_plan_id: UUID | None = None,
    recorded_at: datetime | None = None,
) -> TrainingPlanLifecycleRecord:
    payload = {
        "schema_version": "1.0",
        "record_id": _uuid4_from_payload(
            {
                "user_id": draft.plan.user_id,
                "plan_id": draft.plan.plan_id,
                "plan_version": draft.plan.version,
                "status": status.value,
                "sequence": sequence,
                "superseded_by_plan_id": superseded_by_plan_id,
            }
        ),
        "sequence": sequence,
        "user_id": draft.plan.user_id,
        "plan_id": draft.plan.plan_id,
        "plan_version": draft.plan.version,
        "assessment_id": draft.assessment_id,
        "assessment_revision": draft.assessment_revision,
        "assessment_content_sha256": draft.assessment_content_sha256,
        "catalog_version": draft.catalog_version,
        "catalog_heads_sha256": draft.catalog_heads_sha256,
        "status": status,
        "superseded_by_plan_id": superseded_by_plan_id,
        "recorded_at": draft.created_at if recorded_at is None else recorded_at,
        "previous_record_sha256": previous_record_sha256,
    }
    encoded = TrainingPlanLifecycleRecord.model_construct(
        **payload, record_sha256="0" * 64
    ).model_dump(mode="json", exclude={"record_sha256"})
    return TrainingPlanLifecycleRecord.model_validate(
        payload | {"record_sha256": canonical_sha256(encoded)}, strict=True
    )


def _run_has_exact_common_lineage(run: AgentRun) -> bool:
    return (
        run.task_type == "TRAINING_PLAN"
        and run.prompt_version == "training-plan-v1"
        and run.prompt_sha256 == TRAINING_PLANNING_PROMPT_V1_SHA256
        and run.repair_count == 0
        and type(run.started_at) is datetime
        and run.started_at.tzinfo is UTC
        and type(run.completed_at) is datetime
        and run.completed_at.tzinfo is UTC
        and run.started_at <= run.completed_at
    )


def _valid_success_lineage(draft: TrainingPlanDraft, run: AgentRun) -> bool:
    return (
        _run_has_exact_common_lineage(run)
        and run.status is AgentRunStatus.SUCCEEDED
        and run.run_id == draft.agent_run_id
        and run.prompt_version == draft.prompt_version
        and run.prompt_sha256 == draft.prompt_sha256
        and run.output_sha256 == draft.content_sha256
        and run.context_sha256 == draft.context_sha256
        and run.completed_at == draft.created_at
    )


def _valid_failure_lineage(failure: TrainingPlanningFailure) -> bool:
    run = failure.agent_run
    return (
        _run_has_exact_common_lineage(run)
        and run.status is AgentRunStatus.FAILED
        and run.output_sha256 is None
    )


class InMemoryTrainingPlanRepository:
    """Atomic reference adapter; deliberately not a PostgreSQL claim."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._drafts: dict[UUID, list[TrainingPlanDraft]] = defaultdict(list)
        self._runs: dict[UUID, list[AgentRun]] = defaultdict(list)
        self._lifecycle: dict[UUID, list[TrainingPlanLifecycleRecord]] = defaultdict(list)
        self._current: dict[UUID, TrainingPlanDraft] = {}
        self._receipts: dict[tuple[UUID, str], tuple[str, TrainingPlanPersistResult]] = {}

    def save(self, command: TrainingPlanSaveCommand) -> TrainingPlanSaveResult:
        try:
            exact_command = TrainingPlanSaveCommand.model_validate(
                _stored_values(command), strict=True
            )
            draft = TrainingPlanDraft.model_validate(
                _stored_values(exact_command.draft), strict=True
            )
            run = AgentRun.model_validate(_stored_values(exact_command.agent_run), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise TrainingPlanRepositoryError(
                TrainingPlanRepositoryErrorCode.INVALID_RECORD
            ) from None
        if not _valid_success_lineage(draft, run):
            raise TrainingPlanRepositoryError(TrainingPlanRepositoryErrorCode.INVALID_RECORD)
        user_id = draft.plan.user_id
        fingerprint = canonical_sha256(exact_command.model_dump(mode="json"))
        receipt_key = (user_id, exact_command.idempotency_key)
        with self._lock:
            receipt = self._receipts.get(receipt_key)
            if receipt is not None:
                if receipt[0] != fingerprint or not isinstance(receipt[1], TrainingPlanSaveResult):
                    raise TrainingPlanRepositoryError(
                        TrainingPlanRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return _clone(TrainingPlanSaveResult, receipt[1])
            current = self._current.get(user_id)
            current_version = 0 if current is None else current.plan.version
            if exact_command.expected_current_plan_version != current_version:
                raise TrainingPlanRepositoryError(TrainingPlanRepositoryErrorCode.VERSION_CONFLICT)
            if current is not None and (
                current.assessment_id == draft.assessment_id
                and current.assessment_revision == draft.assessment_revision
                and current.assessment_content_sha256 == draft.assessment_content_sha256
                and current.catalog_version == draft.catalog_version
                and current.catalog_heads_sha256 == draft.catalog_heads_sha256
                and current.context_sha256 == draft.context_sha256
            ):
                raise TrainingPlanRepositoryError(
                    TrainingPlanRepositoryErrorCode.AUTHORITY_UNCHANGED
                )
            if draft.plan.version != current_version + 1:
                raise TrainingPlanRepositoryError(TrainingPlanRepositoryErrorCode.VERSION_CONFLICT)

            existing_records = self._lifecycle[user_id]
            previous_hash = None if not existing_records else existing_records[-1].record_sha256
            new_records: list[TrainingPlanLifecycleRecord] = []
            sequence = len(existing_records) + 1
            if current is not None:
                superseded = _record(
                    draft=current,
                    status=TrainingPlanLifecycleStatus.SUPERSEDED,
                    sequence=sequence,
                    previous_record_sha256=previous_hash,
                    superseded_by_plan_id=draft.plan.plan_id,
                    recorded_at=draft.created_at,
                )
                new_records.append(superseded)
                previous_hash = superseded.record_sha256
                sequence += 1
            active = _record(
                draft=draft,
                status=TrainingPlanLifecycleStatus.DRAFT,
                sequence=sequence,
                previous_record_sha256=previous_hash,
            )
            new_records.append(active)
            result = TrainingPlanSaveResult(
                draft=draft,
                agent_run=run,
                lifecycle_records=tuple(new_records),
            )

            self._drafts[user_id].append(draft)
            self._runs[user_id].append(run)
            self._lifecycle[user_id].extend(new_records)
            self._current[user_id] = draft
            self._receipts[receipt_key] = (fingerprint, result)
            return _clone(TrainingPlanSaveResult, result)

    def save_failure(
        self, command: TrainingPlanFailureSaveCommand
    ) -> TrainingPlanFailureSaveResult:
        try:
            exact_command = TrainingPlanFailureSaveCommand.model_validate(
                _stored_values(command), strict=True
            )
            failure = TrainingPlanningFailure.model_validate(
                _stored_values(exact_command.failure), strict=True
            )
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise TrainingPlanRepositoryError(
                TrainingPlanRepositoryErrorCode.INVALID_RECORD
            ) from None
        if not _valid_failure_lineage(failure):
            raise TrainingPlanRepositoryError(TrainingPlanRepositoryErrorCode.INVALID_RECORD)
        fingerprint = canonical_sha256(exact_command.model_dump(mode="json"))
        receipt_key = (exact_command.user_id, exact_command.idempotency_key)
        with self._lock:
            receipt = self._receipts.get(receipt_key)
            if receipt is not None:
                if receipt[0] != fingerprint or not isinstance(
                    receipt[1], TrainingPlanFailureSaveResult
                ):
                    raise TrainingPlanRepositoryError(
                        TrainingPlanRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return _clone(TrainingPlanFailureSaveResult, receipt[1])
            result = TrainingPlanFailureSaveResult(
                user_id=exact_command.user_id,
                failure=failure,
            )
            self._runs[exact_command.user_id].append(failure.agent_run)
            self._receipts[receipt_key] = (fingerprint, result)
            return _clone(TrainingPlanFailureSaveResult, result)

    def get_current(self, user_id: UUID) -> TrainingPlanDraft:
        with self._lock:
            value = self._current.get(user_id)
            if value is None:
                raise TrainingPlanRepositoryError(TrainingPlanRepositoryErrorCode.NOT_FOUND)
            return _clone(TrainingPlanDraft, value)

    def list_drafts(self, user_id: UUID) -> tuple[TrainingPlanDraft, ...]:
        with self._lock:
            return tuple(_clone(TrainingPlanDraft, item) for item in self._drafts[user_id])

    def list_agent_runs(self, user_id: UUID) -> tuple[AgentRun, ...]:
        with self._lock:
            return tuple(_clone(AgentRun, item) for item in self._runs[user_id])

    def list_lifecycle(self, user_id: UUID) -> tuple[TrainingPlanLifecycleRecord, ...]:
        with self._lock:
            return tuple(
                _clone(TrainingPlanLifecycleRecord, item) for item in self._lifecycle[user_id]
            )


__all__ = [
    "InMemoryTrainingPlanRepository",
    "TrainingPlanLifecycleRecord",
    "TrainingPlanLifecycleStatus",
    "TrainingPlanFailureSaveCommand",
    "TrainingPlanFailureSaveResult",
    "TrainingPlanPersistResult",
    "TrainingPlanRepositoryError",
    "TrainingPlanRepositoryErrorCode",
    "TrainingPlanSaveCommand",
    "TrainingPlanSaveResult",
]
