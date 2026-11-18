from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.assessment_service import (
    AssessmentActorRole,
    AssessmentAuthorizationReceipt,
    AssessmentPrincipal,
)
from billiards_ai.cloud.coach_agent.assessment_validation import (
    ASSESSMENT_VALIDATION_POLICY_VERSION,
    AssessmentValidationReport,
    AssessmentValidationStatus,
    AssessmentViolationCode,
    ValidationViolation,
)
from billiards_ai.shared.models.coaching import CoachAssessmentStatus


def _uid(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def test_validation_report_is_strict_frozen_canonical_and_hash_bound() -> None:
    violations = (
        ValidationViolation(code=AssessmentViolationCode.FINDING_POLICY, location="findings"),
        ValidationViolation(code=AssessmentViolationCode.EVIDENCE_REFS, location="findings.0"),
    )
    report = AssessmentValidationReport.build(
        assessment_id=_uid(1),
        assessment_revision=1,
        assessment_content_sha256="a" * 64,
        context_sha256="b" * 64,
        agent_run_id=_uid(2),
        agent_run_sha256="d" * 64,
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
        violations=violations,
    )

    assert report.policy_version == ASSESSMENT_VALIDATION_POLICY_VERSION
    assert report.status is AssessmentValidationStatus.FAILED
    assert report.violations == tuple(sorted(violations, key=lambda item: item.sort_key()))
    assert len(report.report_sha256) == 64
    assert (
        report.model_dump_json()
        == AssessmentValidationReport.model_validate_json(
            report.model_dump_json(), strict=True
        ).model_dump_json()
    )
    with pytest.raises(ValidationError):
        report.status = AssessmentValidationStatus.PASSED
    with pytest.raises(ValidationError):
        AssessmentValidationReport.model_validate(
            report.model_dump() | {"detail": "secret prompt"}, strict=True
        )


def test_passed_report_cannot_hide_violations_or_use_non_utc_time() -> None:
    with pytest.raises(ValidationError):
        AssessmentValidationReport(
            assessment_id=_uid(1),
            assessment_revision=1,
            assessment_content_sha256="a" * 64,
            context_sha256="b" * 64,
            agent_run_id=_uid(2),
            agent_run_sha256="d" * 64,
            policy_version=ASSESSMENT_VALIDATION_POLICY_VERSION,
            status=AssessmentValidationStatus.PASSED,
            validated_at=datetime(2026, 8, 16, 10),
            violations=(
                ValidationViolation(
                    code=AssessmentViolationCode.SCHEMA,
                    location="assessment",
                ),
            ),
            report_sha256="c" * 64,
        )


def test_authorization_receipt_is_trusted_identity_and_hash_bound() -> None:
    principal = AssessmentPrincipal(principal_id="human-reviewer")
    receipt = AssessmentAuthorizationReceipt.build(
        assessment_id=_uid(1),
        principal_id=principal.principal_id,
        actor_role=AssessmentActorRole.HUMAN_REVIEWER,
        target_status=CoachAssessmentStatus.APPROVED,
        issued_at=datetime(2026, 8, 16, 11, tzinfo=UTC),
    )

    assert receipt.principal_id == principal.principal_id
    with pytest.raises(ValidationError):
        AssessmentAuthorizationReceipt.model_validate(
            receipt.model_dump(mode="python") | {"actor_role": AssessmentActorRole.AI_PROVIDER},
            strict=True,
        )
