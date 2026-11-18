"""Assessment validation, one-shot repair, and controlled review transitions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Self, cast
from uuid import UUID

from pydantic import UUID4, Field, TypeAdapter, ValidationError, model_validator

from billiards_ai.cloud.coach_agent.assessment_repository import (
    AssessmentActorRole,
    AssessmentAuthorizationPort,
    AssessmentAuthorizationReceipt,
    AssessmentPrincipal,
    AssessmentRepairAudit,
    AssessmentRepairStatus,
    AssessmentRepositoryError,
    AssessmentRepositoryErrorCode,
    AssessmentStatusTransitionRecord,
    AssessmentValidationClaim,
    AssessmentValidationRecord,
    InMemoryAssessmentRepository,
)
from billiards_ai.cloud.coach_agent.assessment_validation import (
    AssessmentValidationReport,
    AssessmentValidationStatus,
    AssessmentValidator,
    AssessmentViolationCode,
)
from billiards_ai.cloud.coach_agent.context import CoachAgentContext, CoachTaskType
from billiards_ai.cloud.coach_agent.growth_analysis import (
    GrowthAnalysisProviderOutput,
    assemble_growth_assessment_draft,
    growth_analysis_provider_payload,
    parse_growth_analysis_provider_output,
)
from billiards_ai.cloud.coach_agent.prompts import get_growth_analysis_prompt
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelProvider,
    CoachModelProviderAuditPort,
    CoachModelRequest,
    CoachModelResponse,
    revalidate_coach_model_result,
)
from billiards_ai.shared.models.coaching import (
    AgentRun,
    CoachAssessmentStatus,
    CoachAssessmentV2,
)
from billiards_ai.shared.models.events import CoreContractModel, canonical_sha256, require_utc
from billiards_ai.shared.models.growth import GrowthEvidenceSnapshotV3

_MAX_REPAIR_PAYLOAD_BYTES = 36_864
_UUID4_ADAPTER = TypeAdapter(UUID4)


def _safe_provider_label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("provider audit label is invalid")
    return value


class AssessmentServiceResultCode(StrEnum):
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NO_RELIABLE_ASSESSMENT = "NO_RELIABLE_ASSESSMENT"


class AssessmentServiceResult(CoreContractModel):
    code: AssessmentServiceResultCode
    assessment: CoachAssessmentV2 | None
    agent_run: AgentRun
    validation_reports: tuple[AssessmentValidationReport, ...] = Field(min_length=1, max_length=2)
    repair_status: AssessmentRepairStatus
    record_sha256: str

    @model_validator(mode="after")
    def result_is_closed(self) -> Self:
        reliable = self.code is AssessmentServiceResultCode.REVIEW_REQUIRED
        if reliable != (self.assessment is not None):
            raise ValueError("assessment result code and artifact differ")
        if reliable:
            assert self.assessment is not None
            if (
                self.assessment.status is not CoachAssessmentStatus.REVIEW_REQUIRED
                or self.validation_reports[-1].status is not AssessmentValidationStatus.PASSED
            ):
                raise ValueError("review-required result is not audit closed")
        elif self.validation_reports[-1].status is not AssessmentValidationStatus.FAILED:
            raise ValueError("unreliable result is not audit closed")
        if self.agent_run.repair_count != 0 or any(
            item.agent_run_sha256 != canonical_sha256(self.agent_run.model_dump(mode="json"))
            for item in self.validation_reports
        ):
            raise ValueError("provider AgentRun was rewritten or detached")
        if self.agent_run.output_sha256 is not None and (
            self.agent_run.output_sha256 != self.validation_reports[0].assessment_content_sha256
        ):
            raise ValueError("provider output is detached from the original draft")
        return self


def _promote(
    assessment: CoachAssessmentV2,
    *,
    revision: int,
    status: CoachAssessmentStatus,
) -> CoachAssessmentV2:
    payload = {key: value for key, value in vars(assessment).items() if key != "content_sha256"} | {
        "revision": revision,
        "status": status,
    }
    encoded = CoachAssessmentV2.model_construct(**payload, content_sha256="0" * 64).model_dump(
        mode="json", exclude={"content_sha256"}
    )
    return CoachAssessmentV2.model_validate(
        payload | {"content_sha256": canonical_sha256(encoded)}, strict=True
    )


def _record_result(record: AssessmentValidationRecord) -> AssessmentServiceResult:
    return AssessmentServiceResult(
        code=AssessmentServiceResultCode(record.result_code),
        assessment=record.assessment,
        agent_run=record.agent_run,
        validation_reports=record.validation_reports,
        repair_status=record.repair_status,
        record_sha256=record.record_sha256,
    )


@dataclass(frozen=True, slots=True)
class _RepairAttempt:
    status: AssessmentRepairStatus
    audit: AssessmentRepairAudit
    draft: CoachAssessmentV2 | None = None


def _validation_command(
    *,
    context: object,
    draft: object,
    agent_run: AgentRun,
) -> dict[str, object]:
    try:
        context_payload: object = CoachAgentContext.model_validate(
            dict(vars(context)), strict=True
        ).model_dump(mode="json")
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        context_payload = {
            "schema_failure": True,
            "invalid_shape_sha256": _invalid_object_fingerprint(context),
        }
    try:
        draft_payload: object = CoachAssessmentV2.model_validate(
            dict(vars(draft)), strict=True
        ).model_dump(mode="json")
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        draft_payload = {
            "schema_failure": True,
            "invalid_shape_sha256": _invalid_object_fingerprint(draft),
        }
    return {
        "schema_version": "assessment-validation-command-v1",
        "context": context_payload,
        "draft": draft_payload,
        "agent_run": agent_run.model_dump(mode="json"),
    }


def _invalid_object_fingerprint(value: object) -> str:
    """Bound an invalid command to its complete top-level shape without storing payloads."""

    try:
        fields = vars(value)
    except (TypeError, ValueError):
        fields = {}
    field_hashes: list[tuple[str, str]] = []
    for name, field_value in sorted(fields.items(), key=lambda item: str(item[0])):
        safe_name = str(name)[:128]
        try:
            digest = canonical_sha256(field_value)
        except (RecursionError, TypeError, ValueError):
            invalid_type = f"{type(field_value).__module__}.{type(field_value).__qualname__}"
            digest = canonical_sha256(
                {
                    "invalid_type": invalid_type,
                    "field": safe_name,
                }
            )
        field_hashes.append((safe_name, digest))
    return canonical_sha256(
        {
            "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": field_hashes,
        }
    )


def _build_record(**payload: object) -> AssessmentValidationRecord:
    return AssessmentValidationRecord(
        **payload,
        record_sha256=canonical_sha256(payload),
    )


def _program_transition_sha256(
    *,
    draft: CoachAssessmentV2,
    promoted: CoachAssessmentV2,
    report: AssessmentValidationReport,
    agent_run: AgentRun,
    command_sha256: str,
) -> str:
    payload = {
        "assessment_id": draft.assessment_id,
        "from_revision": draft.revision,
        "to_revision": promoted.revision,
        "from_status": draft.status,
        "to_status": promoted.status,
        "actor_role": AssessmentActorRole.PROGRAM_VALIDATOR,
        "principal_id": "assessment-validator",
        "authorization_receipt_sha256": None,
        "occurred_at": max(promoted.created_at, report.validated_at),
        "command_sha256": command_sha256,
        "from_content_sha256": draft.content_sha256,
        "assessment_content_sha256": promoted.content_sha256,
        "validation_report_sha256": report.report_sha256,
        "agent_run_sha256": canonical_sha256(agent_run.model_dump(mode="json")),
        "previous_transition_sha256": None,
    }
    transition = AssessmentStatusTransitionRecord(
        **payload,
        transition_sha256=canonical_sha256(payload),
    )
    return transition.transition_sha256


class AssessmentValidationService:
    """Validate once, repair at most once, and atomically store the audit outcome."""

    def __init__(
        self,
        *,
        repository: InMemoryAssessmentRepository,
        repair_provider: CoachModelProvider,
        uuid_factory: Callable[[], UUID],
        clock: Callable[[], datetime],
        validator: AssessmentValidator | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        if (
            not isinstance(timeout_ms, int)
            or isinstance(timeout_ms, bool)
            or not 1 <= timeout_ms <= 600_000
        ):
            raise ValueError("timeout_ms must be a bounded strict integer")
        self._repository = repository
        self._provider = repair_provider
        self._uuid_factory = uuid_factory
        self._clock = clock
        self._validator = validator or AssessmentValidator()
        self._timeout_ms = timeout_ms
        audit_port = cast(CoachModelProviderAuditPort, repair_provider)
        try:
            self._repair_provider_name = _safe_provider_label(audit_port.audit_provider_name)
            self._repair_provider_model = _safe_provider_label(audit_port.audit_model_name)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("repair provider audit metadata is invalid") from None

    def _time(self) -> datetime:
        try:
            return require_utc(self._clock())
        except (TypeError, ValueError):
            raise ValueError("clock must return UTC datetime values") from None

    def _uuid(self) -> UUID:
        try:
            return _UUID4_ADAPTER.validate_python(self._uuid_factory(), strict=True)
        except (TypeError, ValidationError, ValueError):
            raise ValueError("uuid_factory must return UUID4 values") from None

    async def validate_and_stage(
        self,
        *,
        context: CoachAgentContext,
        draft: CoachAssessmentV2,
        agent_run: AgentRun,
        idempotency_key: str,
    ) -> AssessmentServiceResult:
        try:
            agent_run = AgentRun.model_validate(dict(vars(agent_run)), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD) from None
        try:
            exact_context = CoachAgentContext.model_validate(dict(vars(context)), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            exact_context = None
        exact_draft: CoachAssessmentV2 | None
        try:
            exact_draft = CoachAssessmentV2.model_validate(dict(vars(draft)), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            exact_draft = None
        if exact_draft is not None and exact_draft.revision != 1:
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.VERSION_CONFLICT)
        command = _validation_command(context=context, draft=draft, agent_run=agent_run)
        claim = self._repository.claim_validation(
            run_id=agent_run.run_id,
            idempotency_key=idempotency_key,
            command=command,
        )
        if not claim.owner:
            return _record_result(await self._repository.wait_for_validation(claim))
        try:
            return await self._execute(
                context=context,
                exact_context=exact_context,
                draft=draft,
                exact_draft=exact_draft,
                agent_run=agent_run,
                idempotency_key=idempotency_key,
                command=command,
                claim=claim,
            )
        except BaseException:
            self._repository.abort_validation(claim)
            raise

    async def _execute(
        self,
        *,
        context: object,
        exact_context: CoachAgentContext | None,
        draft: object,
        exact_draft: CoachAssessmentV2 | None,
        agent_run: AgentRun,
        idempotency_key: str,
        command: Mapping[str, object],
        claim: AssessmentValidationClaim,
    ) -> AssessmentServiceResult:
        initial = self._validator.validate(
            assessment=draft,
            context=context,
            agent_run=agent_run,
            validated_at=self._time(),
        )
        reports = [initial]
        repair_status = AssessmentRepairStatus.NOT_ATTEMPTED
        repair_audit = None
        accepted_draft = exact_draft
        repairable = (
            exact_draft is not None
            and exact_context is not None
            and initial.status is AssessmentValidationStatus.FAILED
            and bool(initial.violations)
            and {item.code for item in initial.violations}
            <= {AssessmentViolationCode.SELECTION_POLICY}
        )
        if repairable:
            repair = await self._repair(exact_context, exact_draft, agent_run, initial)
            repair_status = repair.status
            repair_audit = repair.audit
            if repair.draft is not None:
                repaired_draft = repair.draft
                repaired_report = self._validator.validate(
                    assessment=repaired_draft,
                    context=context,
                    agent_run=agent_run,
                    validated_at=self._time(),
                    repair_audit_sha256=repair.audit.audit_sha256,
                    previous_report_sha256=initial.report_sha256,
                )
                reports.append(repaired_report)
                if repaired_report.status is AssessmentValidationStatus.PASSED:
                    repair_status = AssessmentRepairStatus.SUCCEEDED
                    accepted_draft = repaired_draft
                else:
                    repair_status = AssessmentRepairStatus.VALIDATION_FAILED
        if reports[-1].status is AssessmentValidationStatus.PASSED:
            assert accepted_draft is not None
            review_required = _promote(
                accepted_draft,
                revision=accepted_draft.revision + 1,
                status=CoachAssessmentStatus.REVIEW_REQUIRED,
            )
            code = AssessmentServiceResultCode.REVIEW_REQUIRED
            revisions = (accepted_draft, review_required)
            assessment = review_required
        else:
            code = AssessmentServiceResultCode.NO_RELIABLE_ASSESSMENT
            revisions = ()
            assessment = None
        command_sha256 = canonical_sha256(dict(command))
        promotion_sha256 = (
            None
            if assessment is None or accepted_draft is None
            else _program_transition_sha256(
                draft=accepted_draft,
                promoted=assessment,
                report=reports[-1],
                agent_run=agent_run,
                command_sha256=command_sha256,
            )
        )
        record_payload = {
            "assessment_id": initial.assessment_id,
            "result_code": code.value,
            "assessment": assessment,
            "agent_run": agent_run,
            "validation_reports": tuple(reports),
            "repair_status": repair_status,
            "repair_audit": repair_audit,
            "repair_request_id": None if repair_audit is None else repair_audit.request_id,
            "repair_request_sha256": (
                None if repair_audit is None else repair_audit.request_sha256
            ),
            "repair_output_sha256": (None if repair_audit is None else repair_audit.output_sha256),
            "repair_audit_sha256": (None if repair_audit is None else repair_audit.audit_sha256),
            "command_sha256": command_sha256,
            "promotion_transition_sha256": promotion_sha256,
        }
        record = _build_record(**record_payload)
        stored = self._repository.commit_validation(
            record=record,
            revisions=revisions,
            expected_revision=0,
            idempotency_key=idempotency_key,
            command=command,
            claim=claim,
        )
        return _record_result(stored)

    async def _repair(
        self,
        context: CoachAgentContext,
        draft: CoachAssessmentV2,
        run: AgentRun,
        report: AssessmentValidationReport,
    ) -> _RepairAttempt:
        request_id = self._uuid()
        started_at = self._time()
        prompt = get_growth_analysis_prompt("growth-analysis-v1")

        def audit(
            status: AssessmentRepairStatus,
            *,
            request_sha256: str,
            output_sha256: str | None = None,
        ) -> AssessmentRepairAudit:
            return AssessmentRepairAudit.build(
                status=status,
                request_id=request_id,
                request_sha256=request_sha256,
                output_sha256=output_sha256,
                provider_name=self._repair_provider_name,
                provider_model=self._repair_provider_model,
                started_at=started_at,
                completed_at=max(started_at, self._time()),
            )

        fallback_request_sha256 = canonical_sha256(
            {"request_id": request_id, "status": "request-not-created"}
        )
        if (
            not isinstance(context.growth_snapshot, GrowthEvidenceSnapshotV3)
            or run.repair_count != 0
            or run.provider_name != self._repair_provider_name
            or run.provider_model != self._repair_provider_model
        ):
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=fallback_request_sha256,
                ),
            )
        try:
            full_candidates = growth_analysis_provider_payload(context.growth_snapshot)[
                "metric_candidates"
            ]
            metric_by_code = {
                item.metric.metric_code.value: item.metric_identity_sha256
                for item in context.growth_snapshot.metric_assessments
            }
            invalid_selection = [
                {
                    "metric_identity_sha256": metric_by_code[
                        finding.finding_code.split(":", 1)[-1]
                    ],
                    "finding_type": finding.finding_type.value,
                }
                for finding in draft.findings
                if finding.finding_code.split(":", 1)[-1] in metric_by_code
            ]
            payload = {
                "schema_version": "1.0",
                "metric_candidates": [
                    {
                        "metric_identity_sha256": candidate["metric_identity_sha256"],
                        "allowed_finding_types": candidate["allowed_finding_types"],
                    }
                    for candidate in full_candidates
                ],
                "invalid_selection": invalid_selection,
                "validation_violation_codes": sorted(
                    {item.code.value for item in report.violations}
                ),
            }
        except (AttributeError, RecursionError, TypeError, ValueError):
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=fallback_request_sha256,
                ),
            )
        if (
            len(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > _MAX_REPAIR_PAYLOAD_BYTES
        ):
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=fallback_request_sha256,
                ),
            )
        request = CoachModelRequest(
            task_type=CoachTaskType.GROWTH_ASSESSMENT,
            prompt_version=prompt.version,
            system_prompt=prompt.system_prompt,
            input_payload=payload,
            output_schema=GrowthAnalysisProviderOutput.model_json_schema(),
            request_id=request_id,
            timeout_ms=self._timeout_ms,
        )
        request_sha256 = canonical_sha256(request.model_dump(mode="json"))
        try:
            raw = await self._provider.generate_structured(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=request_sha256,
                ),
            )
        try:
            result = revalidate_coach_model_result(raw)
        except ValueError:
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=request_sha256,
                ),
            )
        if isinstance(result, CoachModelFailure):
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=request_sha256,
                ),
            )
        if (
            not isinstance(result, CoachModelResponse)
            or result.provider != run.provider_name
            or result.model != run.provider_model
        ):
            return _RepairAttempt(
                AssessmentRepairStatus.PROVIDER_FAILED,
                audit(
                    AssessmentRepairStatus.PROVIDER_FAILED,
                    request_sha256=request_sha256,
                ),
            )
        output_sha256 = canonical_sha256(result.structured_output)
        try:
            output = parse_growth_analysis_provider_output(result)
        except ValueError:
            return _RepairAttempt(
                AssessmentRepairStatus.INVALID_OUTPUT,
                audit(
                    AssessmentRepairStatus.INVALID_OUTPUT,
                    request_sha256=request_sha256,
                    output_sha256=output_sha256,
                ),
            )
        repaired = assemble_growth_assessment_draft(
            snapshot=context.growth_snapshot,
            provider_output=output,
            assessment_id=draft.assessment_id,
            agent_run_id=run.run_id,
            created_at=draft.created_at,
        )
        return _RepairAttempt(
            AssessmentRepairStatus.SUCCEEDED,
            audit(
                AssessmentRepairStatus.SUCCEEDED,
                request_sha256=request_sha256,
                output_sha256=output_sha256,
            ),
            draft=repaired,
        )


class AssessmentReviewService:
    """Apply only explicitly authorized immutable review status transitions."""

    def __init__(
        self,
        *,
        repository: InMemoryAssessmentRepository,
        authorization_port: AssessmentAuthorizationPort,
    ) -> None:
        self._repository = repository
        self._authorization_port = authorization_port

    def transition(
        self,
        *,
        assessment_id: UUID,
        expected_revision: int,
        target_status: CoachAssessmentStatus,
        principal: AssessmentPrincipal,
        idempotency_key: str,
        occurred_at: datetime,
    ) -> CoachAssessmentV2:
        try:
            assessment_id = _UUID4_ADAPTER.validate_python(assessment_id, strict=True)
            occurred_at = require_utc(occurred_at)
            principal = AssessmentPrincipal.model_validate(dict(vars(principal)), strict=True)
        except (TypeError, ValidationError, ValueError):
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD) from None
        if (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 1
            or not isinstance(target_status, CoachAssessmentStatus)
        ):
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.INVALID_RECORD)
        try:
            raw_receipt = self._authorization_port.authorize(
                principal=principal,
                assessment_id=assessment_id,
                target_status=target_status,
                requested_at=occurred_at,
            )
            receipt = AssessmentAuthorizationReceipt.model_validate(
                dict(vars(raw_receipt)), strict=True
            )
        except AssessmentRepositoryError:
            raise
        except Exception:
            raise AssessmentRepositoryError(
                AssessmentRepositoryErrorCode.TRANSITION_NOT_ALLOWED
            ) from None
        if receipt.principal_id != principal.principal_id:
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.TRANSITION_NOT_ALLOWED)
        command = {
            "schema_version": "assessment-transition-command-v1",
            "assessment_id": str(assessment_id),
            "expected_revision": expected_revision,
            "target_status": target_status.value,
            "principal": principal.model_dump(mode="json"),
            "authorization_receipt": receipt.model_dump(mode="json"),
            "occurred_at": occurred_at.isoformat(),
        }
        replay = self._repository.lookup_idempotency(idempotency_key, command)
        if isinstance(replay, CoachAssessmentV2):
            return replay
        head = self._repository.get_head(assessment_id)
        if head is None:
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.NOT_FOUND)
        if head.revision != expected_revision:
            raise AssessmentRepositoryError(AssessmentRepositoryErrorCode.VERSION_CONFLICT)
        next_assessment = _promote(
            head,
            revision=expected_revision + 1,
            status=target_status,
        )
        return self._repository.append_transition(
            assessment=next_assessment,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command=command,
            authorization_receipt=receipt,
            occurred_at=occurred_at,
        )


__all__ = [
    "AssessmentActorRole",
    "AssessmentAuthorizationPort",
    "AssessmentAuthorizationReceipt",
    "AssessmentPrincipal",
    "AssessmentReviewService",
    "AssessmentServiceResult",
    "AssessmentServiceResultCode",
    "AssessmentValidationService",
]
