"""Deterministic typical-shot selector tests for Story 5.1."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from billiards_ai.cloud.engine import select_typical_shots
from billiards_ai.cloud.review import ReportNotDeliverableError
from billiards_ai.shared.models import (
    ApprovedSessionData,
    ApprovedShotData,
    Confidence,
    ReportDocument,
    ReportNotDeliverableReason,
    ReportStatus,
    ReviewStatus,
    ShotAnalysisStatus,
    TypicalShotRuleVersion,
    TypicalShotSelectionReason,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"


class _ApprovedRepository:
    def __init__(self, data: ApprovedSessionData):
        self.data = data
        self.calls: list[tuple[UUID, UUID, int]] = []

    def get_approved_session_data(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
    ) -> ApprovedSessionData:
        self.calls.append((session_id, report_id, revision))
        return self.data


class _BlockedRepository:
    def __init__(self, failure: ReportNotDeliverableError):
        self.failure = failure

    def get_approved_session_data(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
    ) -> ApprovedSessionData:
        raise self.failure


def _report(*, status: ReportStatus = ReportStatus.APPROVED) -> ReportDocument:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["report_status"] = status.value
    return ReportDocument.model_validate_json(json.dumps(payload))


def _shot(
    sequence: int,
    *,
    shot_id_suffix: int | None = None,
    confidence: Confidence = Confidence.HIGH,
    analysis_status: ShotAnalysisStatus = ShotAnalysisStatus.ANALYZED,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
    is_foul: bool | None = False,
    potted_balls: tuple[int, ...] = (),
    deviation_mm: int | None = None,
) -> ApprovedShotData:
    report = _report()
    suffix = sequence if shot_id_suffix is None else shot_id_suffix
    return ApprovedShotData(
        shot_id=UUID(f"00000000-0000-4000-8000-{suffix:012d}"),
        session_id=report.session_id,
        sequence=sequence,
        struck_ball=1,
        potted_balls=potted_balls,
        cushion_count=0,
        travel_distance_mm=500,
        is_foul=is_foul,
        deviation_mm=deviation_mm,
        confidence=confidence,
        analysis_status=analysis_status,
        review_status=review_status,
    )


def _approved(*shots: ApprovedShotData, report: ReportDocument | None = None):
    selected_report = report or _report()
    return ApprovedSessionData(
        session_id=selected_report.session_id,
        report=selected_report,
        shots=shots,
    )


def _select(
    data: ApprovedSessionData,
    rule_version: TypicalShotRuleVersion = TypicalShotRuleVersion.V1,
):
    repository = _ApprovedRepository(data)
    return select_typical_shots(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        rule_version,
    )


def test_public_selector_uses_approved_repository_as_its_only_data_entry():
    approved = _approved(_shot(1, is_foul=True))
    repository = _ApprovedRepository(approved)

    result = select_typical_shots(
        repository,
        approved.session_id,
        approved.report.report_id,
        approved.report.revision,
        TypicalShotRuleVersion.V1,
    )

    assert repository.calls == [
        (approved.session_id, approved.report.report_id, approved.report.revision)
    ]
    assert result.session_id == approved.session_id


def test_public_selector_propagates_approved_repository_gate_failure_unchanged():
    report = _report()
    failure = ReportNotDeliverableError(ReportNotDeliverableReason.REVIEW_PENDING)
    repository = _BlockedRepository(failure)

    with pytest.raises(ReportNotDeliverableError) as raised:
        select_typical_shots(
            repository,
            report.session_id,
            report.report_id,
            report.revision,
            TypicalShotRuleVersion.V1,
        )

    assert raised.value is failure


def test_public_selector_rejects_directly_constructed_approved_session_data():
    approved = _approved(_shot(1, is_foul=True))

    with pytest.raises(TypeError, match="ApprovedDataRepository"):
        select_typical_shots(
            approved,
            approved.session_id,
            approved.report.report_id,
            approved.report.revision,
            TypicalShotRuleVersion.V1,
        )


def test_selector_is_versioned_stable_and_independent_of_input_order():
    shots = (
        _shot(1, is_foul=True),
        _shot(2, deviation_mm=600),
        _shot(3),
        _shot(4, confidence=Confidence.MEDIUM),
        _shot(5),
        _shot(6, potted_balls=(7,), deviation_mm=20),
    )

    first = _select(_approved(*shots))
    repeated = _select(_approved(*reversed(shots)))

    assert first == repeated
    assert first.rule_version is TypicalShotRuleVersion.V1
    assert first.session_id == _report().session_id
    assert first.report_id == _report().report_id
    assert first.report_revision == _report().revision
    assert [item.sequence for item in first.selections] == [1, 2, 3, 6]
    assert [item.selection_reason for item in first.selections] == [
        TypicalShotSelectionReason.SEVERE_FOUL,
        TypicalShotSelectionReason.SEVERE_DEVIATION,
        TypicalShotSelectionReason.REPEATED_MISSED_POT,
        TypicalShotSelectionReason.POSITIVE_POTTING,
    ]
    assert [item.selection_rank for item in first.selections] == [1, 2, 3, 4]
    assert all(item.rule_version is TypicalShotRuleVersion.V1 for item in first.selections)


def test_severity_ties_break_by_sequence_then_shot_id():
    later = _shot(4, shot_id_suffix=4, deviation_mm=700)
    larger_id = _shot(2, shot_id_suffix=9, deviation_mm=700)
    smaller_id = _shot(2, shot_id_suffix=3, deviation_mm=700)

    by_sequence = _select(_approved(later, larger_id))
    by_id = _select(_approved(larger_id, smaller_id))

    assert by_sequence.selections[0].shot_id == larger_id.shot_id
    assert by_id.selections[0].shot_id == smaller_id.shot_id


def test_one_issue_cannot_fill_the_result_and_repetition_needs_three_candidates():
    two_fouls = (_shot(1, is_foul=True), _shot(2, is_foul=True))
    five_fouls = tuple(_shot(sequence, is_foul=True) for sequence in range(1, 6))

    below_threshold = _select(_approved(*two_fouls))
    repeated = _select(_approved(*five_fouls))

    assert [item.selection_reason for item in below_threshold.selections] == [
        TypicalShotSelectionReason.SEVERE_FOUL
    ]
    assert [item.selection_reason for item in repeated.selections] == [
        TypicalShotSelectionReason.SEVERE_FOUL,
        TypicalShotSelectionReason.REPEATED_FOUL,
    ]
    assert len(repeated.selections) < 3


def test_two_repeated_issue_categories_each_select_next_stable_representative():
    fouls = tuple(_shot(sequence, is_foul=True) for sequence in range(1, 4))
    deviations = tuple(_shot(sequence, deviation_mm=700) for sequence in range(4, 7))
    positive = _shot(7, potted_balls=(5,), deviation_mm=10)

    result = _select(_approved(*reversed((*fouls, *deviations, positive))))

    assert [item.sequence for item in result.selections] == [1, 4, 2, 5, 7]
    assert [item.selection_reason for item in result.selections] == [
        TypicalShotSelectionReason.SEVERE_FOUL,
        TypicalShotSelectionReason.SEVERE_DEVIATION,
        TypicalShotSelectionReason.REPEATED_FOUL,
        TypicalShotSelectionReason.REPEATED_LARGE_DEVIATION,
        TypicalShotSelectionReason.POSITIVE_POTTING,
    ]


@pytest.mark.parametrize(
    "excluded",
    [
        _shot(1, confidence=Confidence.LOW, is_foul=True),
        _shot(1, confidence=Confidence.ABSTAIN, is_foul=True),
        _shot(1, review_status=ReviewStatus.PENDING, is_foul=True),
        _shot(1, review_status=ReviewStatus.IN_REVIEW, is_foul=True),
        _shot(1, review_status=ReviewStatus.REJECTED, is_foul=True),
        _shot(1, analysis_status=ShotAnalysisStatus.PENDING, is_foul=True),
        _shot(
            1,
            analysis_status=ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE,
            is_foul=True,
        ),
    ],
)
def test_low_confidence_unreviewed_or_incomplete_shots_never_appear(excluded):
    result = _select(_approved(excluded))

    assert result.selections == ()


def test_sparse_candidates_return_fewer_than_three_without_low_quality_padding():
    only_objective_miss = _shot(1)
    low_quality_positive = _shot(2, confidence=Confidence.LOW, potted_balls=(3,))

    result = _select(_approved(only_objective_miss, low_quality_positive))

    assert [item.shot_id for item in result.selections] == [only_objective_miss.shot_id]
    assert result.selections[0].selection_reason is TypicalShotSelectionReason.SEVERE_MISSED_POT


def test_positive_selection_falls_back_when_best_candidate_is_already_severe():
    severe_positive = _shot(1, potted_balls=(3,), deviation_mm=400)
    next_positive = _shot(2, potted_balls=(5,), deviation_mm=None)

    result = _select(_approved(severe_positive, next_positive))

    assert [item.sequence for item in result.selections] == [1, 2]
    assert [item.selection_reason for item in result.selections] == [
        TypicalShotSelectionReason.SEVERE_DEVIATION,
        TypicalShotSelectionReason.POSITIVE_POTTING,
    ]


def test_selector_fails_closed_for_wrong_session_draft_or_tampered_projection():
    valid = _shot(1, is_foul=True)
    wrong_session = valid.model_copy(
        update={"session_id": UUID("30000000-0000-4000-8000-000000000001")}
    )
    draft = _report(status=ReportStatus.DRAFT_PENDING_REVIEW)
    tampered = valid.model_copy(update={"trajectory": ((0, 0), (1, 1))})

    with pytest.raises(ValueError, match="session"):
        _select(_approved(wrong_session))
    with pytest.raises(ValueError, match="approved|corrected"):
        _select(_approved(valid, report=draft))
    with pytest.raises(ValueError, match="trajectory"):
        _select(_approved(tampered))


def test_selector_rejects_unknown_rule_version():
    approved = _approved(_shot(1, is_foul=True))

    with pytest.raises(ValueError, match="rule version"):
        _select(approved, "future")
