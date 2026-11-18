from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.coaching import (
    CoachAssessmentStatus,
    CoachAssessmentV2,
    CoachingFindingV2,
    FindingSupportStatus,
    FindingType,
    GrowthTrendLineageReferenceV1,
    TrendDirection,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.growth import MetricSupportStatus
from billiards_ai.shared.models.growth_metrics import MetricReliability

REF = UUID("00000000-0000-4000-8000-000000000011")
TREND = UUID("00000000-0000-4000-8000-000000000012")


def _trend() -> GrowthTrendLineageReferenceV1:
    return GrowthTrendLineageReferenceV1(
        lineage_id=TREND,
        source_snapshot_schema_version="3.0",
        metric_code="pot.success_rate",
        metric_identity_sha256="1" * 64,
        trend_direction=TrendDirection.IMPROVING,
        trend_support_status=MetricSupportStatus.SUPPORTED,
        reliability=MetricReliability.HIGH,
        trend_policy_version="trend-v1",
        evidence_ref_ids=(REF,),
    )


def _finding(**updates: object) -> CoachingFindingV2:
    values = {
        "finding_id": UUID("00000000-0000-4000-8000-000000000013"),
        "finding_type": FindingType.IMPROVEMENT,
        "finding_code": "POT_SUCCESS_TREND",
        "affected_skill_codes": ("POTTING",),
        "support_status": FindingSupportStatus.SUPPORTED,
        "reliability": MetricReliability.HIGH,
        "evidence_ref_ids": (REF,),
        "trend_lineage_id": TREND,
        "summary": "Supported improvement.",
        "explanation": "The supplied trend projection reports improvement.",
        "limitations": (),
    }
    values.update(updates)
    return CoachingFindingV2.model_validate(values)


def _assessment(**updates: object) -> CoachAssessmentV2:
    payload = {
        "schema_version": "2.0",
        "assessment_id": UUID("00000000-0000-4000-8000-000000000014"),
        "user_id": UUID("00000000-0000-4000-8000-000000000001"),
        "snapshot_id": UUID("00000000-0000-4000-8000-000000000015"),
        "snapshot_version": 3,
        "snapshot_schema_version": "3.0",
        "snapshot_content_sha256": "2" * 64,
        "revision": 1,
        "agent_run_id": UUID("00000000-0000-4000-8000-000000000016"),
        "status": CoachAssessmentStatus.DRAFT,
        "created_at": datetime(2026, 8, 9, tzinfo=UTC),
        "snapshot_evidence_ref_ids": (REF,),
        "snapshot_trend_lineage": (_trend(),),
        "findings": (_finding(),),
        "primary_finding_id": UUID("00000000-0000-4000-8000-000000000013"),
        "secondary_finding_id": None,
    }
    payload.update(updates)
    return CoachAssessmentV2.model_validate(payload | {"content_sha256": canonical_sha256(payload)})


def test_v2_assessment_is_hash_bound_frozen_and_extra_forbidden() -> None:
    assessment = _assessment()
    assert assessment.schema_version == "2.0"
    with pytest.raises(ValidationError):
        CoachAssessmentV2.model_validate(
            assessment.model_dump(mode="python") | {"unknown": "forbidden"}
        )
    with pytest.raises(ValidationError, match="content hash"):
        CoachAssessmentV2.model_validate(
            assessment.model_dump(mode="python") | {"content_sha256": "0" * 64}
        )


@pytest.mark.parametrize(
    ("finding_type", "trend_lineage_id"),
    [(FindingType.IMPROVEMENT, None), (FindingType.DECLINE, None), (FindingType.PLATEAU, None)],
)
def test_trend_finding_requires_d1_lineage(
    finding_type: FindingType, trend_lineage_id: UUID | None
) -> None:
    with pytest.raises(ValidationError, match="trend lineage"):
        _finding(finding_type=finding_type, trend_lineage_id=trend_lineage_id)


def test_assessment_rejects_evidence_or_trend_not_in_exact_snapshot() -> None:
    other = UUID("00000000-0000-4000-8000-000000000099")
    with pytest.raises(ValidationError, match="snapshot evidence"):
        _assessment(findings=(_finding(evidence_ref_ids=(other,)),))
    with pytest.raises(ValidationError, match="trend lineage"):
        _assessment(findings=(_finding(trend_lineage_id=other),))


def test_insufficient_or_unknown_trend_cannot_claim_substantive_conclusion() -> None:
    with pytest.raises(ValidationError, match="limited or unsupported"):
        _assessment(
            snapshot_trend_lineage=(
                _trend().model_copy(
                    update={
                        "trend_direction": TrendDirection.UNKNOWN,
                        "trend_support_status": MetricSupportStatus.INSUFFICIENT_EVIDENCE,
                        "reliability": MetricReliability.UNKNOWN,
                        "evidence_ref_ids": (),
                    }
                ),
            ),
            findings=(_finding(),),
        )


def test_trend_finding_cannot_rewrite_source_reliability() -> None:
    with pytest.raises(ValidationError, match="limited or unsupported"):
        _assessment(findings=(_finding(reliability=MetricReliability.LOW),))


def test_insufficient_finding_uses_objective_unknown_reliability() -> None:
    finding = _finding(
        finding_type=FindingType.INSUFFICIENT_EVIDENCE,
        support_status=FindingSupportStatus.UNSUPPORTED,
        reliability=MetricReliability.UNKNOWN,
        evidence_ref_ids=(),
        trend_lineage_id=None,
        limitations=("No comparable window.",),
    )

    assert finding.reliability is MetricReliability.UNKNOWN


def test_supported_finding_and_trend_reject_unknown_reliability() -> None:
    with pytest.raises(ValidationError, match="UNKNOWN"):
        _finding(reliability=MetricReliability.UNKNOWN)
    with pytest.raises(ValidationError, match="UNKNOWN"):
        GrowthTrendLineageReferenceV1.model_validate(
            _trend().model_dump(mode="python") | {"reliability": MetricReliability.UNKNOWN}
        )


def test_unknown_trend_cannot_claim_known_reliability() -> None:
    with pytest.raises(ValidationError, match="UNKNOWN reliability"):
        GrowthTrendLineageReferenceV1.model_validate(
            _trend().model_dump(mode="python")
            | {
                "trend_direction": TrendDirection.UNKNOWN,
                "trend_support_status": MetricSupportStatus.INSUFFICIENT_EVIDENCE,
                "reliability": MetricReliability.HIGH,
                "evidence_ref_ids": (),
            }
        )
