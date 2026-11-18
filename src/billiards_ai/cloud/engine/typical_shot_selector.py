"""Pure deterministic typical-shot selection from approved projections only."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from billiards_ai.cloud.review.repository import ApprovedDataRepository
from billiards_ai.shared.models.quality import ShotAnalysisStatus
from billiards_ai.shared.models.report import ReportDocument, ReportStatus
from billiards_ai.shared.models.review import ApprovedSessionData, ApprovedShotData
from billiards_ai.shared.models.shot import Confidence, ReviewStatus
from billiards_ai.shared.models.typical_shot import (
    TypicalShotRuleVersion,
    TypicalShotSelection,
    TypicalShotSelectionReason,
    TypicalShotSelectionResult,
)

_LARGE_DEVIATION_MM_V1 = 300
_ELIGIBLE_CONFIDENCE = frozenset({Confidence.HIGH, Confidence.MEDIUM})
_ELIGIBLE_REVIEW_STATUS = frozenset({ReviewStatus.APPROVED, ReviewStatus.CORRECTED})


class _IssueCategory(StrEnum):
    FOUL = "foul"
    LARGE_DEVIATION = "large_deviation"
    MISSED_POT = "missed_pot"


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _validate_approved_data(data: ApprovedSessionData) -> ApprovedSessionData:
    if not isinstance(data, ApprovedSessionData):
        raise TypeError("selector input must be ApprovedSessionData")

    session_values = _stored_model_values(data)
    ApprovedSessionData.model_validate(session_values)
    report = ReportDocument.model_validate(_stored_model_values(data.report))
    shots = tuple(
        ApprovedShotData.model_validate(_stored_model_values(shot)) for shot in data.shots
    )
    if report.report_status not in {ReportStatus.APPROVED, ReportStatus.CORRECTED}:
        raise ValueError("approved session data requires an approved or corrected report")
    if report.session_id != data.session_id:
        raise ValueError("report session must match approved session data")
    if any(shot.session_id != data.session_id for shot in shots):
        raise ValueError("shot session must match approved session data")
    return ApprovedSessionData(session_id=data.session_id, report=report, shots=shots)


def _is_eligible(shot: ApprovedShotData) -> bool:
    return (
        shot.confidence in _ELIGIBLE_CONFIDENCE
        and shot.review_status in _ELIGIBLE_REVIEW_STATUS
        and shot.analysis_status is ShotAnalysisStatus.ANALYZED
    )


def _matches_issue(shot: ApprovedShotData, category: _IssueCategory) -> bool:
    if category is _IssueCategory.FOUL:
        return shot.is_foul is True
    if category is _IssueCategory.LARGE_DEVIATION:
        return (
            shot.is_foul is not True
            and shot.deviation_mm is not None
            and shot.deviation_mm >= _LARGE_DEVIATION_MM_V1
        )
    return (
        shot.is_foul is False
        and not shot.potted_balls
        and (shot.deviation_mm is None or shot.deviation_mm < _LARGE_DEVIATION_MM_V1)
    )


def _issue_sort_key(
    shot: ApprovedShotData,
    category: _IssueCategory,
) -> tuple[int, int, str]:
    if category is _IssueCategory.LARGE_DEVIATION:
        return (-shot.deviation_mm, shot.sequence, str(shot.shot_id))
    confidence_rank = 0 if shot.confidence is Confidence.HIGH else 1
    return (confidence_rank, shot.sequence, str(shot.shot_id))


def _reason_for(
    category: _IssueCategory,
    *,
    repeated: bool,
) -> TypicalShotSelectionReason:
    if repeated:
        return {
            _IssueCategory.FOUL: TypicalShotSelectionReason.REPEATED_FOUL,
            _IssueCategory.LARGE_DEVIATION: (TypicalShotSelectionReason.REPEATED_LARGE_DEVIATION),
            _IssueCategory.MISSED_POT: TypicalShotSelectionReason.REPEATED_MISSED_POT,
        }[category]
    return {
        _IssueCategory.FOUL: TypicalShotSelectionReason.SEVERE_FOUL,
        _IssueCategory.LARGE_DEVIATION: TypicalShotSelectionReason.SEVERE_DEVIATION,
        _IssueCategory.MISSED_POT: TypicalShotSelectionReason.SEVERE_MISSED_POT,
    }[category]


def _category_candidates(
    shots: tuple[ApprovedShotData, ...],
    category: _IssueCategory,
) -> tuple[ApprovedShotData, ...]:
    return tuple(
        sorted(
            (shot for shot in shots if _matches_issue(shot, category)),
            key=lambda shot: _issue_sort_key(shot, category),
        )
    )


def _select_typical_shots(
    data: ApprovedSessionData,
    rule_version: TypicalShotRuleVersion,
) -> TypicalShotSelectionResult:
    """Pure selection from one repository-approved projection."""

    if rule_version is not TypicalShotRuleVersion.V1:
        raise ValueError("unsupported typical-shot rule version")
    approved = _validate_approved_data(data)
    eligible = tuple(shot for shot in approved.shots if _is_eligible(shot))
    categories = (
        _IssueCategory.FOUL,
        _IssueCategory.LARGE_DEVIATION,
        _IssueCategory.MISSED_POT,
    )
    candidates = {category: _category_candidates(eligible, category) for category in categories}

    chosen: list[tuple[ApprovedShotData, TypicalShotSelectionReason]] = []
    chosen_ids: set[object] = set()
    for category in categories:
        if candidates[category] and len(chosen) < 2:
            representative = candidates[category][0]
            chosen.append((representative, _reason_for(category, repeated=False)))
            chosen_ids.add(representative.shot_id)

    repeated_count = 0
    for category in categories:
        if len(candidates[category]) < 3 or repeated_count == 2:
            continue
        representative = next(
            (
                candidate
                for candidate in candidates[category]
                if candidate.shot_id not in chosen_ids
            ),
            None,
        )
        if representative is None:
            continue
        chosen.append((representative, _reason_for(category, repeated=True)))
        chosen_ids.add(representative.shot_id)
        repeated_count += 1

    positive_candidates = tuple(
        sorted(
            (
                shot
                for shot in eligible
                if shot.is_foul is False
                and bool(shot.potted_balls)
                and shot.confidence is Confidence.HIGH
            ),
            key=lambda shot: (
                shot.deviation_mm is None,
                shot.deviation_mm if shot.deviation_mm is not None else 0,
                shot.sequence,
                str(shot.shot_id),
            ),
        )
    )
    positive = next(
        (candidate for candidate in positive_candidates if candidate.shot_id not in chosen_ids),
        None,
    )
    if positive is not None:
        chosen.append((positive, TypicalShotSelectionReason.POSITIVE_POTTING))

    return TypicalShotSelectionResult(
        session_id=approved.session_id,
        report_id=approved.report.report_id,
        report_revision=approved.report.revision,
        rule_version=rule_version,
        selections=tuple(
            TypicalShotSelection(
                shot_id=shot.shot_id,
                sequence=shot.sequence,
                selection_rank=rank,
                selection_reason=reason,
                rule_version=rule_version,
            )
            for rank, (shot, reason) in enumerate(chosen, start=1)
        ),
    )


def select_typical_shots(
    repository: ApprovedDataRepository,
    session_id: UUID,
    report_id: UUID,
    revision: int,
    rule_version: TypicalShotRuleVersion,
) -> TypicalShotSelectionResult:
    """Load through the approved-only repository, then select without further I/O."""

    get_approved_data = getattr(repository, "get_approved_session_data", None)
    if not callable(get_approved_data):
        raise TypeError("repository must implement ApprovedDataRepository")
    approved = get_approved_data(session_id, report_id, revision)
    return _select_typical_shots(approved, rule_version)
