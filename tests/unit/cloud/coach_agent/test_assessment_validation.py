from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from billiards_ai.cloud.coach_agent.assessment_validation import (
    AssessmentValidationStatus,
    AssessmentValidator,
    AssessmentViolationCode,
)
from billiards_ai.cloud.coach_agent.growth_analysis import GrowthAnalysisCompleted
from billiards_ai.shared.models.coaching import CoachAssessmentStatus, CoachAssessmentV2
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.growth_metrics import MetricReliability
from tests.unit.cloud.coach_agent.test_growth_analysis import (
    ScriptedProvider,
    _context,
    _project,
    _response,
    _window_batches,
    _workflow,
)


async def _valid_bundle(
    finding_type: str = "IMPROVEMENT",
) -> tuple[object, CoachAssessmentV2, object]:
    snapshot = _project(_window_batches(0, 5))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": finding_type,
                },
                "secondary": None,
            }
        )
    )
    context = _context(snapshot)
    result = await _workflow(provider).analyze(context)
    assert isinstance(result, GrowthAnalysisCompleted)
    return context, result.assessment, result.agent_run


def _replace_assessment(assessment: CoachAssessmentV2, **updates: object) -> CoachAssessmentV2:
    payload = {
        key: value for key, value in vars(assessment).items() if key != "content_sha256"
    } | updates
    json_payload = CoachAssessmentV2.model_construct(**payload, content_sha256="0" * 64).model_dump(
        mode="json", exclude={"content_sha256"}
    )
    return CoachAssessmentV2.model_validate(
        payload | {"content_sha256": canonical_sha256(json_payload)}, strict=True
    )


@pytest.mark.asyncio
async def test_exact_story_94_draft_passes_all_programmatic_validators() -> None:
    context, assessment, audit = await _valid_bundle()

    report = AssessmentValidator().validate(
        assessment=assessment,
        context=context,
        agent_run=audit,
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    assert report.status is AssessmentValidationStatus.PASSED
    assert report.violations == ()


@pytest.mark.asyncio
async def test_numeric_claim_and_causal_language_fail_closed() -> None:
    context, assessment, audit = await _valid_bundle()
    finding = assessment.findings[0].model_copy(
        update={
            "summary": "成功率提高到 99%",
            "explanation": "因为训练，所以该能力提高。",
        }
    )
    malformed = _replace_assessment(assessment, findings=(finding,))

    report = AssessmentValidator().validate(
        assessment=malformed,
        context=context,
        agent_run=audit.model_copy(update={"output_sha256": malformed.content_sha256}),
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    codes = {item.code for item in report.violations}
    assert AssessmentViolationCode.NUMERIC_INTEGRITY in codes
    assert AssessmentViolationCode.FINDING_POLICY in codes


@pytest.mark.asyncio
async def test_evidence_subset_and_lineage_drift_are_rejected() -> None:
    context, assessment, audit = await _valid_bundle()
    finding = assessment.findings[0]
    assert len(finding.evidence_ref_ids) >= 2
    subset = finding.model_copy(update={"evidence_ref_ids": finding.evidence_ref_ids[:1]})
    malformed = _replace_assessment(assessment, findings=(subset,))
    wrong_audit = audit.model_copy(
        update={
            "context_sha256": "f" * 64,
            "output_sha256": malformed.content_sha256,
        }
    )

    report = AssessmentValidator().validate(
        assessment=malformed,
        context=context,
        agent_run=wrong_audit,
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    codes = {item.code for item in report.violations}
    assert AssessmentViolationCode.EVIDENCE_REFS in codes
    assert AssessmentViolationCode.CONTEXT_LINEAGE in codes


@pytest.mark.asyncio
async def test_unsafe_construct_is_mapped_to_schema_violation_without_payload() -> None:
    context, assessment, audit = await _valid_bundle()
    recursive: list[object] = []
    recursive.append(recursive)
    malformed = CoachAssessmentV2.model_construct(
        **(dict(vars(assessment)) | {"findings": recursive})
    )

    report = AssessmentValidator().validate(
        assessment=malformed,
        context=context,
        agent_run=audit,
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    assert report.status is AssessmentValidationStatus.FAILED
    assert {item.code for item in report.violations} == {AssessmentViolationCode.SCHEMA}
    assert "recursive" not in report.model_dump_json().lower()


@pytest.mark.asyncio
async def test_prompt_provider_snapshot_and_status_lineage_are_independently_checked() -> None:
    context, assessment, audit = await _valid_bundle()
    changed = _replace_assessment(
        assessment,
        status=CoachAssessmentStatus.REVIEW_REQUIRED,
        snapshot_content_sha256="e" * 64,
    )
    wrong_run = audit.model_copy(
        update={
            "prompt_sha256": "d" * 64,
            "provider_name": " scripted\nsecret ",
            "output_sha256": changed.content_sha256,
        }
    )

    report = AssessmentValidator().validate(
        assessment=changed,
        context=context,
        agent_run=wrong_run,
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    codes = {item.code for item in report.violations}
    assert AssessmentViolationCode.SNAPSHOT_LINEAGE in codes
    assert AssessmentViolationCode.PROMPT_LINEAGE in codes
    assert AssessmentViolationCode.PROVIDER_LINEAGE in codes
    assert AssessmentViolationCode.FINDING_POLICY in codes


@pytest.mark.asyncio
async def test_strength_cannot_raise_reliability_or_change_deterministic_finding_id() -> None:
    context, assessment, audit = await _valid_bundle("STRENGTH")
    finding = assessment.findings[0]
    replacement = finding.model_copy(
        update={
            "finding_id": UUID("00000000-0000-4000-8000-000000111111"),
            "reliability": (
                MetricReliability.LOW
                if finding.reliability is MetricReliability.HIGH
                else MetricReliability.HIGH
            ),
        }
    )
    malformed = _replace_assessment(
        assessment,
        findings=(replacement,),
        primary_finding_id=replacement.finding_id,
    )
    report = AssessmentValidator().validate(
        assessment=malformed,
        context=context,
        agent_run=audit.model_copy(update={"output_sha256": malformed.content_sha256}),
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    codes = {item.code for item in report.violations}
    assert AssessmentViolationCode.RELIABILITY in codes
    assert AssessmentViolationCode.FINDING_POLICY in codes


@pytest.mark.asyncio
async def test_every_finding_field_is_rebuilt_from_v3_and_sensitive_text_is_rejected() -> None:
    context, assessment, audit = await _valid_bundle()
    finding = assessment.findings[0].model_copy(
        update={
            "summary": "一位学员 because 张三@example.com",
            "limitations": ("call 13800138000 therefore caused by practice",),
        }
    )
    malformed = _replace_assessment(assessment, findings=(finding,))
    report = AssessmentValidator().validate(
        assessment=malformed,
        context=context,
        agent_run=audit.model_copy(update={"output_sha256": malformed.content_sha256}),
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    assert report.status is AssessmentValidationStatus.FAILED
    codes = {item.code for item in report.violations}
    assert AssessmentViolationCode.NUMERIC_INTEGRITY in codes
    assert AssessmentViolationCode.FINDING_POLICY in codes


@pytest.mark.asyncio
async def test_substantive_finding_must_be_selected_exactly_once() -> None:
    context, assessment, audit = await _valid_bundle()
    malformed = _replace_assessment(
        assessment,
        primary_finding_id=None,
        secondary_finding_id=None,
    )
    report = AssessmentValidator().validate(
        assessment=malformed,
        context=context,
        agent_run=audit.model_copy(update={"output_sha256": malformed.content_sha256}),
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    assert AssessmentViolationCode.SELECTION_POLICY in {item.code for item in report.violations}


@pytest.mark.asyncio
async def test_known_zero_evidence_metric_matches_its_insufficient_finding() -> None:
    snapshot = _project(_window_batches(0, 5))
    metric = next(item for item in snapshot.metric_assessments if not item.evidence_refs)
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "INSUFFICIENT_EVIDENCE",
                },
                "secondary": None,
            }
        )
    )
    context = _context(snapshot)
    generated = await _workflow(provider).analyze(context)
    assert isinstance(generated, GrowthAnalysisCompleted)

    report = AssessmentValidator().validate(
        assessment=generated.assessment,
        context=context,
        agent_run=generated.agent_run,
        validated_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
    )

    assert report.status is AssessmentValidationStatus.PASSED
    assert report.violations == ()
