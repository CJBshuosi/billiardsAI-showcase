"""Pure deterministic matching for the frozen traceable recommendation catalog."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ValidationError

from billiards_ai.cloud.engine.typical_shot_selector import _select_typical_shots
from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.shared.models.quality import ShotAnalysisStatus
from billiards_ai.shared.models.recommendation import (
    Recommendation,
    RecommendationCatalogVersion,
    RecommendationClaimCode,
    RecommendationDrill,
    RecommendationEvidence,
    RecommendationEvidenceKind,
    RecommendationPracticeCode,
    RecommendationReason,
    RecommendationResult,
    RecommendationRuleId,
    RecommendationRuleVersion,
    RecommendationScope,
    RecommendationSemanticCore,
    RecommendationSourceLineage,
    RecommendationStatus,
)
from billiards_ai.shared.models.report import ReportDocument
from billiards_ai.shared.models.review import ApprovedSessionData, ApprovedShotData
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, TargetZoneSource
from billiards_ai.shared.models.typical_shot import (
    TypicalShotSelection,
    TypicalShotSelectionReason,
    TypicalShotSelectionResult,
)

_FOUL_REASONS = frozenset(
    {
        TypicalShotSelectionReason.SEVERE_FOUL,
        TypicalShotSelectionReason.REPEATED_FOUL,
    }
)
_MISSED_REASONS = frozenset(
    {
        TypicalShotSelectionReason.SEVERE_MISSED_POT,
        TypicalShotSelectionReason.REPEATED_MISSED_POT,
    }
)
_DEVIATION_REASONS = frozenset(
    {
        TypicalShotSelectionReason.SEVERE_DEVIATION,
        TypicalShotSelectionReason.REPEATED_LARGE_DEVIATION,
    }
)
_NEGATIVE_REASONS = _FOUL_REASONS | _MISSED_REASONS | _DEVIATION_REASONS


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _invalid_result() -> RecommendationResult:
    return RecommendationResult.build(
        catalog_version=RecommendationCatalogVersion.V1,
        status=RecommendationStatus.INVALID_INPUT,
        reason=RecommendationReason.INVALID_IDENTIFIER_OR_INPUT,
        source_lineage=None,
        recommendations=(),
    )


def _selection_mismatch(lineage: RecommendationSourceLineage) -> RecommendationResult:
    return RecommendationResult.build(
        catalog_version=RecommendationCatalogVersion.V1,
        status=RecommendationStatus.SELECTION_MISMATCH,
        reason=RecommendationReason.SELECTION_RESULT_MISMATCH,
        source_lineage=lineage,
        recommendations=(),
    )


def _lineage(
    approved: ApprovedSessionData,
    selection: TypicalShotSelectionResult,
) -> RecommendationSourceLineage:
    return RecommendationSourceLineage(
        session_id=approved.session_id,
        report_id=approved.report.report_id,
        report_revision=approved.report.revision,
        selection_rule_version=selection.rule_version,
    )


def _normalize_approved_projection(
    value: object,
    *,
    session_id: UUID,
    report_id: UUID,
    revision: int,
) -> ApprovedSessionData:
    if not isinstance(value, ApprovedSessionData):
        raise TypeError("repository must return ApprovedSessionData")
    ApprovedSessionData.model_validate(_stored_model_values(value))
    report = ReportDocument.model_validate(_stored_model_values(value.report))
    shots = tuple(
        ApprovedShotData.model_validate(_stored_model_values(shot)) for shot in value.shots
    )
    if (
        value.session_id != session_id
        or report.session_id != session_id
        or report.report_id != report_id
        or report.revision != revision
        or any(shot.session_id != session_id for shot in shots)
    ):
        raise ValueError("approved projection must match requested lineage")
    expected_count = report.session_statistics.total_shot_count
    if (
        len(shots) != expected_count
        or {shot.sequence for shot in shots} != set(range(1, expected_count + 1))
        or len({shot.shot_id for shot in shots}) != expected_count
    ):
        raise ValueError("approved projection requires a complete unique shot set")
    return ApprovedSessionData(session_id=session_id, report=report, shots=shots)


def _evidence(
    selection: TypicalShotSelection,
    shot: ApprovedShotData,
) -> RecommendationEvidence | None:
    common = dict(
        shot_id=shot.shot_id,
        sequence=shot.sequence,
        confidence=shot.confidence,
        analysis_status=shot.analysis_status,
        review_status=shot.review_status,
        selection_rank=selection.selection_rank,
        selection_reason=selection.selection_reason,
        selection_rule_version=selection.rule_version,
    )
    try:
        if selection.selection_reason in _FOUL_REASONS:
            if shot.is_foul is not True:
                return None
            return RecommendationEvidence(
                evidence_kind=RecommendationEvidenceKind.FOUL,
                is_foul=True,
                **common,
            )
        if selection.selection_reason in _MISSED_REASONS:
            if shot.is_foul is not False or shot.potted_balls:
                return None
            return RecommendationEvidence(
                evidence_kind=RecommendationEvidenceKind.MISSED_POT,
                is_foul=False,
                potted_balls=(),
                **common,
            )
        if selection.selection_reason in _DEVIATION_REASONS:
            if (
                shot.is_foul is not False
                or shot.deviation_mm is None
                or shot.target_zone_source is TargetZoneSource.NONE
            ):
                return None
            return RecommendationEvidence(
                evidence_kind=RecommendationEvidenceKind.LARGE_DEVIATION,
                is_foul=False,
                deviation_mm=shot.deviation_mm,
                target_zone_source=shot.target_zone_source,
                **common,
            )
    except ValidationError:
        return None
    return None


def _semantic_core(kind: RecommendationEvidenceKind) -> RecommendationSemanticCore:
    rule_id, claim_code, practice_code = {
        RecommendationEvidenceKind.FOUL: (
            RecommendationRuleId.REDUCE_FOUL_RISK,
            RecommendationClaimCode.REDUCE_FOUL_RISK,
            RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL,
        ),
        RecommendationEvidenceKind.MISSED_POT: (
            RecommendationRuleId.STABILIZE_POTTING,
            RecommendationClaimCode.STABILIZE_POTTING,
            RecommendationPracticeCode.TRACK_BASIC_STRAIGHT_POTS,
        ),
        RecommendationEvidenceKind.LARGE_DEVIATION: (
            RecommendationRuleId.CONTROL_CUE_BALL_POSITION,
            RecommendationClaimCode.CONTROL_CUE_BALL_POSITION,
            RecommendationPracticeCode.TRACK_TARGET_ZONE_FINISHES,
        ),
    }[kind]
    return RecommendationSemanticCore(
        rule_id=rule_id,
        rule_version=RecommendationRuleVersion.V1,
        scope=RecommendationScope.SESSION_PRIMARY,
        claim_code=claim_code,
        practice_code=practice_code,
    )


def _has_unusable_negative_evidence(shots: tuple[ApprovedShotData, ...]) -> bool:
    eligible_confidence = {Confidence.HIGH, Confidence.MEDIUM}
    eligible_review = {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
    for shot in shots:
        negative = (
            shot.is_foul is True
            or (shot.is_foul is False and not shot.potted_balls)
            or (shot.deviation_mm is not None and shot.is_foul is not True)
        )
        eligible = (
            shot.confidence in eligible_confidence
            and shot.review_status in eligible_review
            and shot.analysis_status is ShotAnalysisStatus.ANALYZED
        )
        if negative and not eligible:
            return True
    return False


def _deviation_sources_are_reliable(approved: ApprovedSessionData) -> bool:
    sources = {
        shot.target_zone_source
        for shot in approved.shots
        if shot.is_foul is not True
        and shot.deviation_mm is not None
        and shot.confidence in {Confidence.HIGH, Confidence.MEDIUM}
        and shot.review_status in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
        and shot.analysis_status is ShotAnalysisStatus.ANALYZED
    }
    return len(sources) == 1 and TargetZoneSource.NONE not in sources


def generate_recommendations(
    repository: object,
    session_id: object,
    report_id: object,
    revision: object,
    selection_result: object,
    catalog_version: object,
) -> RecommendationResult:
    """Load one current approved head, authenticate selection, then match V1."""

    try:
        get_approved_data = getattr(repository, "get_approved_session_data", None)
    except Exception:
        return _invalid_result()
    if (
        not callable(get_approved_data)
        or not isinstance(session_id, UUID)
        or session_id.version != 4
        or not isinstance(report_id, UUID)
        or report_id.version != 4
        or type(revision) is not int
        or revision < 1
        or not isinstance(selection_result, TypicalShotSelectionResult)
        or catalog_version is not RecommendationCatalogVersion.V1
    ):
        return _invalid_result()
    try:
        validated_selection = TypicalShotSelectionResult.model_validate(
            _stored_model_values(selection_result)
        )
    except (TypeError, ValueError, ValidationError):
        return _invalid_result()

    try:
        repository_projection = get_approved_data(session_id, report_id, revision)
    except ReportNotDeliverableError:
        raise
    except Exception:
        return _invalid_result()
    try:
        approved = _normalize_approved_projection(
            repository_projection,
            session_id=session_id,
            report_id=report_id,
            revision=revision,
        )
        recomputed = _select_typical_shots(approved, validated_selection.rule_version)
    except (TypeError, ValueError, ValidationError):
        return _invalid_result()
    lineage = _lineage(approved, recomputed)
    if validated_selection != recomputed:
        return _selection_mismatch(lineage)

    shots_by_id = {shot.shot_id: shot for shot in approved.shots}
    unsupported_negative = False
    for selection in recomputed.selections:
        if selection.selection_reason not in _NEGATIVE_REASONS:
            continue
        shot = shots_by_id.get(selection.shot_id)
        if shot is None or shot.sequence != selection.sequence:
            return _selection_mismatch(lineage)
        if selection.selection_reason in _DEVIATION_REASONS and not _deviation_sources_are_reliable(
            approved
        ):
            unsupported_negative = True
            continue
        evidence = _evidence(selection, shot)
        if evidence is None:
            return _selection_mismatch(lineage)
        core = _semantic_core(evidence.evidence_kind)
        recommendation = Recommendation(
            semantic_core=core,
            evidence=(evidence,),
            drill=RecommendationDrill(practice_code=core.practice_code),
        )
        return RecommendationResult.build(
            catalog_version=RecommendationCatalogVersion.V1,
            status=RecommendationStatus.READY,
            source_lineage=lineage,
            recommendations=(recommendation,),
        )

    reason = (
        RecommendationReason.UNSUPPORTED_EVIDENCE
        if unsupported_negative
        else (
            RecommendationReason.INSUFFICIENT_EVIDENCE
            if not recomputed.selections or _has_unusable_negative_evidence(approved.shots)
            else RecommendationReason.NO_NEGATIVE_RULE_MATCH
        )
    )
    return RecommendationResult.build(
        catalog_version=RecommendationCatalogVersion.V1,
        status=RecommendationStatus.NO_RELIABLE_RECOMMENDATION,
        reason=reason,
        source_lineage=lineage,
        recommendations=(),
    )
