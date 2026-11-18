"""Append-only in-memory assessment authority for Story 9.5."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, ValidationError, field_validator, model_validator

from billiards_ai.cloud.coach_agent.assessment_validation import (
    AssessmentValidationReport,
    AssessmentValidationStatus,
)
from billiards_ai.shared.models.coaching import (
    AgentRun,
    CoachAssessmentStatus,
    CoachAssessmentV2,
)
from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)


class AssessmentRepositoryErrorCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    TRANSITION_NOT_ALLOWED = "TRANSITION_NOT_ALLOWED"
    INVALID_RECORD = "INVALID_RECORD"


class AssessmentRepositoryError(RuntimeError):
    def __init__(self, code: AssessmentRepositoryErrorCode):
        self.code = code
        super().__init__(code.value)


class AssessmentRepairStatus(StrEnum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    SUCCEEDED = "SUCCEEDED"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    VALIDATION_FAILED = "VALIDATION_FAILED"


class AssessmentActorRole(StrEnum):
    PROGRAM_VALIDATOR = "PROGRAM_VALIDATOR"
    HUMAN_REVIEWER = "HUMAN_REVIEWER"
    SYSTEM = "SYSTEM"
    AI_PROVIDER = "AI_PROVIDER"


class AssessmentPrincipal(CoreContractModel):
    principal_id: NonEmptyStr

    @field_validator("principal_id")
    @classmethod
    def principal_id_is_bounded_and_safe(cls, value: str) -> str:
        if len(value) > 128 or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("principal identity is invalid")
        return value


class AssessmentAuthorizationReceipt(CoreContractModel):
    assessment_id: UUID4
    principal_id: NonEmptyStr
    actor_role: AssessmentActorRole
    target_status: CoachAssessmentStatus
    issued_at: datetime
    receipt_sha256: Sha256

    _issued_at_utc = field_validator("issued_at")(require_utc)

    @field_validator("principal_id")
    @classmethod
    def principal_id_is_bounded_and_safe(cls, value: str) -> str:
        return AssessmentPrincipal(principal_id=value).principal_id

    @classmethod
    def build(
        cls,
        *,
        assessment_id: UUID,
        principal_id: str,
        actor_role: AssessmentActorRole,
        target_status: CoachAssessmentStatus,
        issued_at: datetime,
    ) -> AssessmentAuthorizationReceipt:
        payload = {
            "assessment_id": assessment_id,
            "principal_id": principal_id,
            "actor_role": actor_role,
            "target_status": target_status,
            "issued_at": require_utc(issued_at),
        }
        return cls(**payload, receipt_sha256=canonical_sha256(payload))

    @model_validator(mode="after")
    def receipt_is_hash_bound(self) -> Self:
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"receipt_sha256"}))
        if self.receipt_sha256 != expected:
            raise ValueError("authorization receipt hash mismatch")
        return self


class AssessmentAuthorizationPort(Protocol):
    def authorize(
        self,
        *,
        principal: AssessmentPrincipal,
        assessment_id: UUID,
        target_status: CoachAssessmentStatus,
        requested_at: datetime,
    ) -> AssessmentAuthorizationReceipt: ...


@dataclass(frozen=True, slots=True)
class AssessmentValidationClaim:
    command_sha256: str
    idempotency_key: str
    run_id: UUID
    event: asyncio.Event
    owner: bool
    replay: AssessmentValidationRecord | None = None


class AssessmentRepairAudit(CoreContractModel):
    status: AssessmentRepairStatus
    request_id: UUID4
    request_sha256: Sha256
    output_sha256: Sha256 | None = None
    provider_name: NonEmptyStr
    provider_model: NonEmptyStr
    started_at: datetime
    completed_at: datetime
    audit_sha256: Sha256

    _started_at_utc = field_validator("started_at")(require_utc)
    _completed_at_utc = field_validator("completed_at")(require_utc)

    @model_validator(mode="after")
    def audit_is_closed(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("repair audit timeline is invalid")
        output_expected = self.status in {
            AssessmentRepairStatus.SUCCEEDED,
            AssessmentRepairStatus.INVALID_OUTPUT,
        }
        if output_expected != (self.output_sha256 is not None):
            raise ValueError("repair output audit mismatch")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"audit_sha256"}))
        if self.audit_sha256 != expected:
            raise ValueError("repair audit hash mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        status: AssessmentRepairStatus,
        request_id: UUID,
        request_sha256: str,
        output_sha256: str | None,
        provider_name: str,
        provider_model: str,
        started_at: datetime,
        completed_at: datetime,
    ) -> AssessmentRepairAudit:
        payload = {
            "status": status,
            "request_id": request_id,
            "request_sha256": request_sha256,
            "output_sha256": output_sha256,
            "provider_name": provider_name,
            "provider_model": provider_model,
            "started_at": require_utc(started_at),
            "completed_at": require_utc(completed_at),
        }
        return cls(**payload, audit_sha256=canonical_sha256(payload))


def _immutable_assessment_payload(assessment: CoachAssessmentV2) -> dict[str, object]:
    return assessment.model_dump(
        mode="json",
        exclude={"revision", "status", "content_sha256"},
    )


def _clone_model[T: CoreContractModel](model: type[T], value: object) -> T:
    if not isinstance(value, model):
        raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
    try:
        return model.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD) from None


def _command_sha256(command: Mapping[str, object]) -> str:
    try:
        return canonical_sha256(dict(command))
    except (RecursionError, TypeError, ValueError):
        raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD) from None


def _transition_is_authorized(
    *,
    from_status: CoachAssessmentStatus,
    to_status: CoachAssessmentStatus,
    actor_role: str,
) -> bool:
    if actor_role == "PROGRAM_VALIDATOR":
        return from_status is CoachAssessmentStatus.DRAFT and to_status in {
            CoachAssessmentStatus.VALIDATED,
            CoachAssessmentStatus.REVIEW_REQUIRED,
        }
    if actor_role == "HUMAN_REVIEWER":
        return from_status in {
            CoachAssessmentStatus.VALIDATED,
            CoachAssessmentStatus.REVIEW_REQUIRED,
        } and to_status in {
            CoachAssessmentStatus.APPROVED,
            CoachAssessmentStatus.REJECTED,
        }
    if actor_role == "SYSTEM":
        return (
            from_status is not CoachAssessmentStatus.SUPERSEDED
            and to_status is CoachAssessmentStatus.SUPERSEDED
        )
    return False


class AssessmentValidationRecord(CoreContractModel):
    assessment_id: UUID4
    result_code: Literal["REVIEW_REQUIRED", "NO_RELIABLE_ASSESSMENT"]
    assessment: CoachAssessmentV2 | None
    agent_run: AgentRun
    validation_reports: tuple[AssessmentValidationReport, ...] = Field(min_length=1, max_length=2)
    repair_status: AssessmentRepairStatus
    repair_audit: AssessmentRepairAudit | None = None
    repair_request_id: UUID4 | None = None
    repair_request_sha256: Sha256 | None = None
    repair_output_sha256: Sha256 | None = None
    repair_audit_sha256: Sha256 | None = None
    command_sha256: Sha256
    promotion_transition_sha256: Sha256 | None = None
    record_sha256: Sha256

    @model_validator(mode="after")
    def record_is_identity_closed(self) -> Self:
        agent_run_sha256 = canonical_sha256(self.agent_run.model_dump(mode="json"))
        if (
            self.agent_run.repair_count != 0
            or self.agent_run.completed_at < self.agent_run.started_at
        ):
            raise ValueError("provider AgentRun repair state is immutable")
        if any(
            item.agent_run_id != self.agent_run.run_id or item.agent_run_sha256 != agent_run_sha256
            for item in self.validation_reports
        ):
            raise ValueError("validation record run identity mismatch")
        if any(item.assessment_id != self.assessment_id for item in self.validation_reports):
            raise ValueError("validation reports must use one assessment identity")
        if any(
            item.assessment_revision != 1
            or item.validated_at < self.agent_run.completed_at
            or item.context_sha256 != self.validation_reports[0].context_sha256
            for item in self.validation_reports
        ):
            raise ValueError("validation report revision or timeline is invalid")
        if any(
            current.validated_at < previous.validated_at
            for previous, current in zip(
                self.validation_reports,
                self.validation_reports[1:],
                strict=False,
            )
        ):
            raise ValueError("validation report timeline is not monotonic")
        if (
            self.validation_reports[0].repair_audit_sha256 is not None
            or self.validation_reports[0].previous_report_sha256 is not None
        ):
            raise ValueError("initial validation report cannot claim repair lineage")
        if self.assessment is not None:
            if (
                self.result_code != "REVIEW_REQUIRED"
                or self.assessment.assessment_id != self.assessment_id
                or self.assessment.agent_run_id != self.agent_run.run_id
                or self.assessment.status is not CoachAssessmentStatus.REVIEW_REQUIRED
                or self.agent_run.status.value != "SUCCEEDED"
                or self.agent_run.repair_count != 0
                or self.validation_reports[-1].status is not AssessmentValidationStatus.PASSED
                or self.promotion_transition_sha256 is None
            ):
                raise ValueError("successful validation record is not artifact closed")
            if self.repair_status is AssessmentRepairStatus.NOT_ATTEMPTED:
                if len(self.validation_reports) != 1:
                    raise ValueError("unrepaired success requires one report")
            elif self.repair_status is AssessmentRepairStatus.SUCCEEDED:
                if (
                    len(self.validation_reports) != 2
                    or self.validation_reports[0].status is not AssessmentValidationStatus.FAILED
                ):
                    raise ValueError("repaired success report matrix is invalid")
            else:
                raise ValueError("failed repair cannot produce a reliable assessment")
        elif (
            self.result_code != "NO_RELIABLE_ASSESSMENT"
            or self.validation_reports[-1].status is not AssessmentValidationStatus.FAILED
            or self.promotion_transition_sha256 is not None
        ):
            raise ValueError("failed validation record cannot claim reliable output")
        elif self.repair_status is AssessmentRepairStatus.SUCCEEDED:
            raise ValueError("successful repair requires a reliable assessment")
        initial_schema_failure = any(
            violation.code.value == "SCHEMA" for violation in self.validation_reports[0].violations
        )
        if (
            not initial_schema_failure
            and self.agent_run.output_sha256 is not None
            and self.agent_run.output_sha256 != self.validation_reports[0].assessment_content_sha256
        ):
            raise ValueError("provider run must remain bound to its original draft")
        if (self.agent_run.status.value == "SUCCEEDED") != (
            self.agent_run.output_sha256 is not None
        ):
            raise ValueError("provider run result matrix is invalid")
        if self.repair_status is AssessmentRepairStatus.NOT_ATTEMPTED:
            if (
                any(
                    value is not None
                    for value in (
                        self.repair_request_id,
                        self.repair_request_sha256,
                        self.repair_output_sha256,
                        self.repair_audit_sha256,
                    )
                )
                or self.repair_audit is not None
                or len(self.validation_reports) != 1
            ):
                raise ValueError("non-repair record cannot claim repair lineage")
        elif (
            self.repair_audit is None
            or (
                self.repair_status is AssessmentRepairStatus.VALIDATION_FAILED
                and self.repair_audit.status is not AssessmentRepairStatus.SUCCEEDED
            )
            or (
                self.repair_status is not AssessmentRepairStatus.VALIDATION_FAILED
                and self.repair_audit.status is not self.repair_status
            )
            or self.repair_request_id is None
            or self.repair_request_sha256 is None
            or self.repair_audit_sha256 is None
            or self.repair_audit.request_id != self.repair_request_id
            or self.repair_audit.request_sha256 != self.repair_request_sha256
            or self.repair_audit.output_sha256 != self.repair_output_sha256
            or self.repair_audit.audit_sha256 != self.repair_audit_sha256
        ):
            raise ValueError("repair record requires request lineage")
        if self.repair_audit is not None and (
            self.repair_audit.request_id == self.agent_run.run_id
            or self.repair_audit.provider_name != self.agent_run.provider_name
            or self.repair_audit.provider_model != self.agent_run.provider_model
            or self.repair_audit.started_at < self.validation_reports[0].validated_at
            or (
                len(self.validation_reports) == 2
                and self.validation_reports[1].validated_at < self.repair_audit.completed_at
            )
        ):
            raise ValueError("repair audit identity or timeline is invalid")
        if self.repair_status is not AssessmentRepairStatus.NOT_ATTEMPTED and (
            self.agent_run.status.value != "SUCCEEDED"
            or not self.validation_reports[0].violations
            or any(
                violation.code.value != "SELECTION_POLICY"
                for violation in self.validation_reports[0].violations
            )
        ):
            raise ValueError("repair is only allowed for a selection-policy failure")
        if self.repair_status is AssessmentRepairStatus.SUCCEEDED and (
            len(self.validation_reports) != 2
            or self.repair_output_sha256 is None
            or self.validation_reports[0].status is not AssessmentValidationStatus.FAILED
        ):
            raise ValueError("successful repair audit is incomplete")
        if (
            self.repair_status
            in {
                AssessmentRepairStatus.PROVIDER_FAILED,
                AssessmentRepairStatus.INVALID_OUTPUT,
            }
            and len(self.validation_reports) != 1
        ):
            raise ValueError("provider repair failure cannot claim a second validation")
        if self.repair_status is AssessmentRepairStatus.VALIDATION_FAILED and (
            len(self.validation_reports) != 2
            or self.repair_output_sha256 is None
            or self.validation_reports[-1].repair_audit_sha256 != self.repair_audit_sha256
            or self.validation_reports[-1].previous_report_sha256
            != self.validation_reports[0].report_sha256
        ):
            raise ValueError("failed repaired validation audit is incomplete")
        if self.repair_status is AssessmentRepairStatus.SUCCEEDED and (
            self.validation_reports[-1].repair_audit_sha256 != self.repair_audit_sha256
            or self.validation_reports[-1].previous_report_sha256
            != self.validation_reports[0].report_sha256
        ):
            raise ValueError("repaired report chain is incomplete")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"record_sha256"}))
        if self.record_sha256 != expected_hash:
            raise ValueError("validation record hash mismatch")
        return self


class AssessmentStatusTransitionRecord(CoreContractModel):
    assessment_id: UUID4
    from_revision: StrictInt = Field(ge=1)
    to_revision: StrictInt = Field(ge=2)
    from_status: CoachAssessmentStatus
    to_status: CoachAssessmentStatus
    actor_role: AssessmentActorRole
    principal_id: NonEmptyStr
    authorization_receipt_sha256: Sha256 | None = None
    occurred_at: datetime
    command_sha256: Sha256
    from_content_sha256: Sha256
    assessment_content_sha256: Sha256
    validation_report_sha256: Sha256 | None = None
    agent_run_sha256: Sha256 | None = None
    previous_transition_sha256: Sha256 | None = None
    transition_sha256: Sha256

    _occurred_at_utc = field_validator("occurred_at")(require_utc)

    @field_validator("principal_id")
    @classmethod
    def principal_id_is_bounded_and_safe(cls, value: str) -> str:
        return AssessmentPrincipal(principal_id=value).principal_id

    @model_validator(mode="after")
    def transition_is_contiguous(self) -> Self:
        if self.to_revision != self.from_revision + 1:
            raise ValueError("assessment transition revisions must be contiguous")
        if not _transition_is_authorized(
            from_status=self.from_status,
            to_status=self.to_status,
            actor_role=self.actor_role.value,
        ):
            raise ValueError("assessment transition authority is invalid")
        if (self.actor_role is AssessmentActorRole.PROGRAM_VALIDATOR) != (
            self.validation_report_sha256 is not None
            and self.agent_run_sha256 is not None
            and self.authorization_receipt_sha256 is None
        ):
            raise ValueError("transition authority audit mismatch")
        if self.actor_role is not AssessmentActorRole.PROGRAM_VALIDATOR and (
            self.authorization_receipt_sha256 is None
        ):
            raise ValueError("review transition requires authorization receipt")
        if self.actor_role is AssessmentActorRole.PROGRAM_VALIDATOR:
            if self.principal_id != "assessment-validator":
                raise ValueError("program transition principal is invalid")
        else:
            receipt_payload = {
                "assessment_id": self.assessment_id,
                "principal_id": self.principal_id,
                "actor_role": self.actor_role,
                "target_status": self.to_status,
                "issued_at": self.occurred_at,
            }
            if self.authorization_receipt_sha256 != canonical_sha256(receipt_payload):
                raise ValueError("transition authorization receipt is detached")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"transition_sha256"}))
        if self.transition_sha256 != expected:
            raise ValueError("transition hash mismatch")
        return self


class InMemoryAssessmentRepository:
    """Single-process atomic reference repository; it does not claim DB durability."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._revisions: dict[UUID, list[CoachAssessmentV2]] = defaultdict(list)
        self._records: dict[UUID, list[AssessmentValidationRecord]] = defaultdict(list)
        self._idempotency: dict[str, tuple[str, object]] = {}
        self._run_records: dict[UUID, AssessmentValidationRecord] = {}
        self._transitions: dict[UUID, list[AssessmentStatusTransitionRecord]] = defaultdict(list)
        self._leases: dict[UUID, AssessmentValidationClaim] = {}

    @staticmethod
    def _validate_idempotency_key(key: str) -> None:
        if (
            not isinstance(key, str)
            or not key
            or key != key.strip()
            or len(key) > 256
            or any(ord(character) < 32 or ord(character) == 127 for character in key)
        ):
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)

    def lookup_idempotency(self, key: str, command: Mapping[str, object]) -> object | None:
        self._validate_idempotency_key(key)
        digest = _command_sha256(command)
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is None:
                return None
            if existing[0] != digest:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT)
            value = existing[1]
            if isinstance(value, AssessmentValidationRecord):
                return _clone_model(AssessmentValidationRecord, value)
            if isinstance(value, CoachAssessmentV2):
                return _clone_model(CoachAssessmentV2, value)
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)

    def claim_validation(
        self,
        *,
        run_id: UUID,
        idempotency_key: str,
        command: Mapping[str, object],
    ) -> AssessmentValidationClaim:
        self._validate_idempotency_key(idempotency_key)
        digest = _command_sha256(command)
        with self._lock:
            existing = self._idempotency.get(idempotency_key)
            if existing is not None:
                if existing[0] != digest or not isinstance(existing[1], AssessmentValidationRecord):
                    raise AssessmentRepositoryError(
                        AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return AssessmentValidationClaim(
                    command_sha256=digest,
                    idempotency_key=idempotency_key,
                    run_id=run_id,
                    event=asyncio.Event(),
                    owner=False,
                    replay=_clone_model(AssessmentValidationRecord, existing[1]),
                )
            if run_id in self._run_records:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT)
            active = self._leases.get(run_id)
            if active is not None:
                if active.command_sha256 != digest or active.idempotency_key != idempotency_key:
                    raise AssessmentRepositoryError(
                        AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return AssessmentValidationClaim(
                    command_sha256=digest,
                    idempotency_key=idempotency_key,
                    run_id=run_id,
                    event=active.event,
                    owner=False,
                )
            claim = AssessmentValidationClaim(
                command_sha256=digest,
                idempotency_key=idempotency_key,
                run_id=run_id,
                event=asyncio.Event(),
                owner=True,
            )
            self._leases[run_id] = claim
            return claim

    async def wait_for_validation(
        self, claim: AssessmentValidationClaim
    ) -> AssessmentValidationRecord:
        if claim.replay is not None:
            return claim.replay
        await claim.event.wait()
        with self._lock:
            replay = self._idempotency.get(claim.idempotency_key)
            if (
                replay is None
                or replay[0] != claim.command_sha256
                or not isinstance(replay[1], AssessmentValidationRecord)
            ):
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            return _clone_model(AssessmentValidationRecord, replay[1])

    def abort_validation(self, claim: AssessmentValidationClaim) -> None:
        with self._lock:
            active = self._leases.get(claim.run_id)
            if active is claim:
                del self._leases[claim.run_id]
                active.event.set()

    def get_run_record(self, run_id: UUID) -> AssessmentValidationRecord | None:
        with self._lock:
            value = self._run_records.get(run_id)
            return None if value is None else _clone_model(AssessmentValidationRecord, value)

    def commit_validation(
        self,
        *,
        record: AssessmentValidationRecord,
        revisions: tuple[CoachAssessmentV2, ...],
        expected_revision: int,
        idempotency_key: str,
        command: Mapping[str, object],
        claim: AssessmentValidationClaim,
    ) -> AssessmentValidationRecord:
        with self._lock:
            record = _clone_model(AssessmentValidationRecord, record)
            digest = _command_sha256(command)
            if (
                record.command_sha256 != digest
                or claim.command_sha256 != digest
                or claim.idempotency_key != idempotency_key
                or self._leases.get(record.agent_run.run_id) is not claim
            ):
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            replay = self.lookup_idempotency(idempotency_key, command)
            if replay is not None:
                if not isinstance(replay, AssessmentValidationRecord):
                    raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
                return replay
            if record.agent_run.run_id in self._run_records:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT)
            stored = self._revisions[record.assessment_id]
            current_revision = stored[-1].revision if stored else 0
            if current_revision != expected_revision:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.VERSION_CONFLICT)
            next_revision = current_revision + 1
            staged: list[CoachAssessmentV2] = []
            for revision in revisions:
                exact = _clone_model(CoachAssessmentV2, revision)
                if exact.assessment_id != record.assessment_id or exact.revision != next_revision:
                    raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.VERSION_CONFLICT)
                staged.append(exact)
                next_revision += 1
            if record.assessment != (staged[-1] if staged else None):
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            if staged:
                if (
                    len(staged) != 2
                    or staged[0].status is not CoachAssessmentStatus.DRAFT
                    or staged[1].status is not CoachAssessmentStatus.REVIEW_REQUIRED
                    or _immutable_assessment_payload(staged[0])
                    != _immutable_assessment_payload(staged[1])
                    or record.validation_reports[-1].assessment_content_sha256
                    != staged[0].content_sha256
                ):
                    raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
                transition_payload = {
                    "assessment_id": record.assessment_id,
                    "from_revision": staged[0].revision,
                    "to_revision": staged[1].revision,
                    "from_status": staged[0].status,
                    "to_status": staged[1].status,
                    "actor_role": AssessmentActorRole.PROGRAM_VALIDATOR,
                    "principal_id": "assessment-validator",
                    "authorization_receipt_sha256": None,
                    "occurred_at": max(
                        staged[1].created_at,
                        record.validation_reports[-1].validated_at,
                    ),
                    "command_sha256": digest,
                    "from_content_sha256": staged[0].content_sha256,
                    "assessment_content_sha256": staged[1].content_sha256,
                    "validation_report_sha256": record.validation_reports[-1].report_sha256,
                    "agent_run_sha256": canonical_sha256(record.agent_run.model_dump(mode="json")),
                    "previous_transition_sha256": None,
                }
                transition = AssessmentStatusTransitionRecord(
                    **transition_payload,
                    transition_sha256=canonical_sha256(transition_payload),
                )
                if record.promotion_transition_sha256 != transition.transition_sha256:
                    raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            else:
                transition = None
            stored.extend(_clone_model(CoachAssessmentV2, item) for item in staged)
            stored_record = _clone_model(AssessmentValidationRecord, record)
            self._records[record.assessment_id].append(stored_record)
            self._run_records[record.agent_run.run_id] = stored_record
            if transition is not None:
                self._transitions[record.assessment_id].append(transition)
            self._idempotency[idempotency_key] = (digest, stored_record)
            del self._leases[record.agent_run.run_id]
            claim.event.set()
            return _clone_model(AssessmentValidationRecord, stored_record)

    def append_transition(
        self,
        *,
        assessment: CoachAssessmentV2,
        expected_revision: int,
        idempotency_key: str,
        command: Mapping[str, object],
        authorization_receipt: AssessmentAuthorizationReceipt,
        occurred_at: datetime,
    ) -> CoachAssessmentV2:
        with self._lock:
            digest = _command_sha256(command)
            replay = self.lookup_idempotency(idempotency_key, command)
            if replay is not None:
                if not isinstance(replay, CoachAssessmentV2):
                    raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
                return replay
            exact = _clone_model(CoachAssessmentV2, assessment)
            receipt = _clone_model(AssessmentAuthorizationReceipt, authorization_receipt)
            stored = self._validated_revision_chain(exact.assessment_id)
            if not stored:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.NOT_FOUND)
            if stored[-1].revision != expected_revision:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.VERSION_CONFLICT)
            if exact.revision != expected_revision + 1:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.VERSION_CONFLICT)
            if _immutable_assessment_payload(stored[-1]) != _immutable_assessment_payload(
                exact
            ) or not _transition_is_authorized(
                from_status=stored[-1].status,
                to_status=exact.status,
                actor_role=receipt.actor_role.value,
            ):
                raise AssessmentRepositoryError(
                    AssessmentRepositoryErrorCode.TRANSITION_NOT_ALLOWED
                )
            transitions = self._validated_transition_chain(exact.assessment_id, stored)
            earliest = max(
                exact.created_at,
                transitions[-1].occurred_at if transitions else exact.created_at,
            )
            if require_utc(occurred_at) < earliest:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            if (
                receipt.assessment_id != exact.assessment_id
                or receipt.target_status is not exact.status
                or receipt.issued_at != occurred_at
            ):
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            transition_payload = {
                "assessment_id": exact.assessment_id,
                "from_revision": stored[-1].revision,
                "to_revision": exact.revision,
                "from_status": stored[-1].status,
                "to_status": exact.status,
                "actor_role": receipt.actor_role,
                "principal_id": receipt.principal_id,
                "authorization_receipt_sha256": receipt.receipt_sha256,
                "occurred_at": occurred_at,
                "command_sha256": digest,
                "from_content_sha256": stored[-1].content_sha256,
                "assessment_content_sha256": exact.content_sha256,
                "validation_report_sha256": None,
                "agent_run_sha256": None,
                "previous_transition_sha256": (
                    transitions[-1].transition_sha256 if transitions else None
                ),
            }
            transition = AssessmentStatusTransitionRecord(
                **transition_payload,
                transition_sha256=canonical_sha256(transition_payload),
            )
            self._revisions[exact.assessment_id].append(_clone_model(CoachAssessmentV2, exact))
            self._transitions[exact.assessment_id].append(transition)
            self._idempotency[idempotency_key] = (
                digest,
                _clone_model(CoachAssessmentV2, exact),
            )
            return _clone_model(CoachAssessmentV2, exact)

    def get_head(self, assessment_id: UUID) -> CoachAssessmentV2 | None:
        with self._lock:
            stored = self._validated_revision_chain(assessment_id)
            self._validated_transition_chain(assessment_id, stored)
            return None if not stored else _clone_model(CoachAssessmentV2, stored[-1])

    def get_revisions(self, assessment_id: UUID) -> tuple[CoachAssessmentV2, ...]:
        with self._lock:
            stored = self._validated_revision_chain(assessment_id)
            self._validated_transition_chain(assessment_id, stored)
            return tuple(_clone_model(CoachAssessmentV2, item) for item in stored)

    def list_agent_runs(self, assessment_id: UUID) -> tuple[AgentRun, ...]:
        with self._lock:
            return tuple(
                _clone_model(AgentRun, item.agent_run)
                for item in self._validated_records(assessment_id)
            )

    def list_validation_records(
        self, assessment_id: UUID
    ) -> tuple[AssessmentValidationRecord, ...]:
        with self._lock:
            return tuple(
                _clone_model(AssessmentValidationRecord, item)
                for item in self._validated_records(assessment_id)
            )

    def get_approved_head(self, assessment_id: UUID) -> CoachAssessmentV2 | None:
        with self._lock:
            head = self.get_head(assessment_id)
            if head is None or head.status is not CoachAssessmentStatus.APPROVED:
                return None
            return head

    def list_status_transitions(
        self, assessment_id: UUID
    ) -> tuple[AssessmentStatusTransitionRecord, ...]:
        with self._lock:
            revisions = self._validated_revision_chain(assessment_id)
            return tuple(
                _clone_model(AssessmentStatusTransitionRecord, item)
                for item in self._validated_transition_chain(assessment_id, revisions)
            )

    def _validated_records(self, assessment_id: UUID) -> list[AssessmentValidationRecord]:
        return [
            _clone_model(AssessmentValidationRecord, item)
            for item in self._records.get(assessment_id, ())
        ]

    def _validated_revision_chain(self, assessment_id: UUID) -> list[CoachAssessmentV2]:
        revisions = [
            _clone_model(CoachAssessmentV2, item) for item in self._revisions.get(assessment_id, ())
        ]
        for index, item in enumerate(revisions, start=1):
            if item.assessment_id != assessment_id or item.revision != index:
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            if index > 1 and _immutable_assessment_payload(revisions[index - 2]) != (
                _immutable_assessment_payload(item)
            ):
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
        return revisions

    def _validated_transition_chain(
        self,
        assessment_id: UUID,
        revisions: list[CoachAssessmentV2],
    ) -> list[AssessmentStatusTransitionRecord]:
        transitions = [
            _clone_model(AssessmentStatusTransitionRecord, item)
            for item in self._transitions.get(assessment_id, ())
        ]
        if len(transitions) != max(0, len(revisions) - 1):
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
        previous_hash = None
        previous_time = revisions[0].created_at if revisions else None
        for index, transition in enumerate(transitions):
            source, target = revisions[index], revisions[index + 1]
            if (
                transition.from_revision != source.revision
                or transition.to_revision != target.revision
                or transition.from_status is not source.status
                or transition.to_status is not target.status
                or transition.from_content_sha256 != source.content_sha256
                or transition.assessment_content_sha256 != target.content_sha256
                or transition.previous_transition_sha256 != previous_hash
                or (previous_time is not None and transition.occurred_at < previous_time)
            ):
                raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
            previous_hash = transition.transition_sha256
            previous_time = transition.occurred_at
        return transitions


__all__ = [
    "AssessmentActorRole",
    "AssessmentAuthorizationPort",
    "AssessmentAuthorizationReceipt",
    "AssessmentPrincipal",
    "AssessmentRepairAudit",
    "AssessmentRepairStatus",
    "AssessmentRepositoryError",
    "AssessmentRepositoryErrorCode",
    "AssessmentStatusTransitionRecord",
    "AssessmentValidationRecord",
    "AssessmentValidationClaim",
    "InMemoryAssessmentRepository",
]
