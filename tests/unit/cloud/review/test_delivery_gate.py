"""Fail-closed report delivery and approved-data repository tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.review import (
    InMemoryApprovedDataRepository,
    ReportNotDeliverableError,
    require_deliverable_report,
)
from billiards_ai.shared.models import (
    Confidence,
    ReportDocument,
    ReportNotDeliverableReason,
    ReportStatus,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    Shot,
    TargetZoneSource,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
V11_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "v1.1" / "complete.json"


def make_report(*, status: ReportStatus = ReportStatus.APPROVED) -> ReportDocument:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    draft = ReportDocument.model_validate_json(json.dumps(payload))
    return ReportDocument.model_validate(draft.model_dump() | {"report_status": status})


def make_v11_report() -> ReportDocument:
    draft = ReportDocument.model_validate_json(V11_FIXTURE.read_text(encoding="utf-8"))
    return ReportDocument.model_validate(
        draft.model_dump(mode="python") | {"report_status": ReportStatus.APPROVED}
    )


def make_task(
    report: ReportDocument,
    *,
    status: ReviewTaskStatus = ReviewTaskStatus.APPROVED,
    report_revision: int | None = None,
) -> ReviewTask:
    return ReviewTask(
        id=uuid4(),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision if report_revision is None else report_revision,
        status=status,
    )


def make_shot(
    report: ReportDocument,
    *,
    sequence: int = 1,
    shot_id: UUID | None = None,
    confidence: Confidence = Confidence.HIGH,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
) -> Shot:
    return Shot(
        id=shot_id or uuid4(),
        session_id=report.session_id,
        sequence=sequence,
        t_start_ms=(sequence - 1) * 1000,
        t_end_ms=sequence * 1000,
        ball_positions_before=None,
        ball_positions_after=None,
        cue_ball_end=None,
        confidence=confidence,
        review_status=review_status,
        raw_observation={"provider": "must-not-leak"},
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )


def make_shots(report: ReportDocument) -> tuple[Shot, ...]:
    return tuple(
        make_shot(report, sequence=index)
        for index in range(1, report.session_statistics.total_shot_count + 1)
    )


def make_v11_shots(report: ReportDocument) -> tuple[Shot, ...]:
    return tuple(
        make_shot(report, sequence=sequence, shot_id=shot_id)
        for sequence, shot_id in enumerate(report.objective_statistics.shot_ids, start=1)
    )


def test_v11_delivery_requires_exact_objective_and_typical_card_shot_lineage():
    report = make_v11_report()
    task = make_task(report)
    exact_shots = make_v11_shots(report)

    assert require_deliverable_report(report, task, exact_shots) is report

    wrong_ids = tuple(
        make_shot(report, sequence=sequence)
        for sequence in range(1, report.session_statistics.total_shot_count + 1)
    )
    with pytest.raises(ReportNotDeliverableError) as objective_mismatch:
        require_deliverable_report(report, task, wrong_ids)
    assert objective_mismatch.value.reason is ReportNotDeliverableReason.SHOT_SET_INCOMPLETE

    payload = report.model_dump(mode="python", exclude={"content_sha256"})
    payload["typical_shot_analysis"]["cards"][0]["sequence"] += 1
    wrong_card = ReportDocument.build_v11(**payload)
    with pytest.raises(ReportNotDeliverableError) as card_mismatch:
        require_deliverable_report(wrong_card, task, exact_shots)
    assert card_mismatch.value.reason is ReportNotDeliverableReason.SHOT_SET_INCOMPLETE


@pytest.mark.parametrize(
    ("task_status", "report_status"),
    [
        (ReviewTaskStatus.APPROVED, ReportStatus.APPROVED),
        (ReviewTaskStatus.CORRECTED, ReportStatus.CORRECTED),
    ],
)
def test_only_matching_approved_or_corrected_revision_is_deliverable(
    task_status: ReviewTaskStatus,
    report_status: ReportStatus,
):
    report = make_report(status=report_status)

    delivered = require_deliverable_report(
        report, make_task(report, status=task_status), make_shots(report)
    )

    assert delivered is report


@pytest.mark.parametrize(
    ("task_status", "reason"),
    [
        (ReviewTaskStatus.PENDING_REVIEW, ReportNotDeliverableReason.REVIEW_PENDING),
        (ReviewTaskStatus.IN_REVIEW, ReportNotDeliverableReason.REVIEW_IN_PROGRESS),
        (ReviewTaskStatus.REJECTED, ReportNotDeliverableReason.REVIEW_REJECTED),
    ],
)
def test_non_approved_review_task_is_rejected(task_status, reason):
    report = make_report()

    with pytest.raises(ReportNotDeliverableError) as raised:
        require_deliverable_report(
            report, make_task(report, status=task_status), make_shots(report)
        )

    assert raised.value.reason is reason
    assert "provider" not in str(raised.value)


@pytest.mark.parametrize(
    ("report", "task_factory", "reason"),
    [
        (
            make_report(status=ReportStatus.DRAFT_PENDING_REVIEW),
            lambda report: make_task(report),
            ReportNotDeliverableReason.REPORT_NOT_APPROVED,
        ),
        (
            make_report(),
            lambda report: make_task(report, report_revision=report.revision + 1),
            ReportNotDeliverableReason.REVISION_MISMATCH,
        ),
        (
            make_report(status=ReportStatus.CORRECTED),
            lambda report: make_task(report, status=ReviewTaskStatus.APPROVED),
            ReportNotDeliverableReason.REPORT_STATUS_MISMATCH,
        ),
    ],
)
def test_report_revision_and_status_must_match_authoritative_task(report, task_factory, reason):
    with pytest.raises(ReportNotDeliverableError) as raised:
        require_deliverable_report(report, task_factory(report), make_shots(report))

    assert raised.value.reason is reason


@pytest.mark.parametrize("review_status", [ReviewStatus.PENDING, ReviewStatus.IN_REVIEW])
def test_unprocessed_phase_zero_shot_blocks_delivery(review_status: ReviewStatus):
    report = make_report()
    shots = list(make_shots(report))
    shots[0] = make_shot(report, confidence=Confidence.ABSTAIN, review_status=review_status)

    with pytest.raises(ReportNotDeliverableError) as raised:
        require_deliverable_report(report, make_task(report), tuple(shots))

    assert raised.value.reason is ReportNotDeliverableReason.UNPROCESSED_CRITICAL_SHOT


def test_unknown_constructed_task_or_shot_status_fails_closed():
    report = make_report()
    unknown_task = make_task(report).model_copy(update={"status": "future_status"})
    unknown_shot = make_shot(report).model_copy(update={"review_status": "future_status"})

    with pytest.raises(ReportNotDeliverableError) as task_error:
        require_deliverable_report(report, unknown_task, make_shots(report))
    with pytest.raises(ReportNotDeliverableError) as shot_error:
        require_deliverable_report(
            report, make_task(report), (unknown_shot, *make_shots(report)[1:])
        )

    assert task_error.value.reason is ReportNotDeliverableReason.UNKNOWN_REVIEW_STATUS
    assert shot_error.value.reason is ReportNotDeliverableReason.UNKNOWN_SHOT_REVIEW_STATUS


def test_repository_cannot_be_bypassed_by_wrong_id_or_caller_approval_flag():
    report = make_report()
    repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(make_task(report),),
        shots=make_shots(report),
    )

    with pytest.raises(TypeError, match="approved"):
        repository.get_approved_report(report.report_id, report.revision, approved=True)
    with pytest.raises(ReportNotDeliverableError) as boolean_revision:
        repository.get_approved_report(report.report_id, True)
    with pytest.raises(ReportNotDeliverableError) as wrong_id:
        repository.get_approved_report(uuid4(), report.revision)

    assert boolean_revision.value.reason is ReportNotDeliverableReason.REPORT_NOT_FOUND
    assert wrong_id.value.reason is ReportNotDeliverableReason.REPORT_NOT_FOUND


@pytest.mark.parametrize(
    "task_status",
    [ReviewTaskStatus.PENDING_REVIEW, ReviewTaskStatus.IN_REVIEW, ReviewTaskStatus.REJECTED],
)
def test_repository_never_returns_unapproved_report_or_shots(task_status: ReviewTaskStatus):
    report = make_report()
    repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(make_task(report, status=task_status),),
        shots=make_shots(report),
    )

    with pytest.raises(ReportNotDeliverableError):
        repository.get_approved_session_data(report.session_id, report.report_id, report.revision)


def test_repository_returns_read_only_projection_without_raw_fields():
    report = make_report()
    shots = make_shots(report)
    repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(make_task(report),),
        shots=shots,
    )

    approved = repository.get_approved_session_data(
        report.session_id,
        report.report_id,
        report.revision,
    )
    payload = approved.model_dump(mode="json")
    serialized = json.dumps(payload)

    assert approved.report == report
    assert approved.report is not report
    assert len(approved.shots) == report.session_statistics.total_shot_count
    assert all(
        name not in serialized for name in ("raw_observation", "trajectory", "media_locator")
    )
    assert {UUID(item["shot_id"]) for item in payload["shots"]} == {shot.id for shot in shots}


def test_repository_projection_preserves_reviewed_target_zone_source():
    report = make_report()
    shots = list(make_shots(report))
    shots[0] = shots[0].model_copy(
        update={
            "deviation_mm": 400,
            "target_zone_source": TargetZoneSource.POST_HOC_INFERRED,
        }
    )
    repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(make_task(report),),
        shots=tuple(shots),
    )

    approved = repository.get_approved_session_data(
        report.session_id, report.report_id, report.revision
    )

    assert approved.shots[0].target_zone_source is TargetZoneSource.POST_HOC_INFERRED


def test_repository_rejects_stale_and_same_revision_forked_session_heads():
    old = make_report()
    newer = ReportDocument.model_validate(
        old.model_dump(mode="python")
        | {
            "report_id": uuid4(),
            "revision": old.revision + 1,
            "parent_revision_id": old.report_id,
        }
    )
    stale_repository = InMemoryApprovedDataRepository(
        reports=(old, newer),
        review_tasks=(make_task(old), make_task(newer)),
        shots=make_shots(old),
    )

    with pytest.raises(ReportNotDeliverableError) as stale:
        stale_repository.get_approved_session_data(old.session_id, old.report_id, old.revision)
    assert stale.value.reason is ReportNotDeliverableReason.REVISION_MISMATCH

    fork = ReportDocument.model_validate(old.model_dump(mode="python") | {"report_id": uuid4()})
    fork_repository = InMemoryApprovedDataRepository(
        reports=(old, fork),
        review_tasks=(make_task(old), make_task(fork)),
        shots=make_shots(old),
    )
    with pytest.raises(ReportNotDeliverableError) as forked:
        fork_repository.get_approved_session_data(old.session_id, old.report_id, old.revision)
    assert forked.value.reason is ReportNotDeliverableReason.REVISION_MISMATCH


def test_repository_checks_current_head_before_stale_revision_delivery_gate():
    old = make_report()
    newer = ReportDocument.model_validate(
        old.model_dump(mode="python")
        | {
            "report_id": uuid4(),
            "revision": old.revision + 1,
            "parent_revision_id": old.report_id,
        }
    )
    repository = InMemoryApprovedDataRepository(
        reports=(old, newer),
        review_tasks=(
            make_task(old, status=ReviewTaskStatus.PENDING_REVIEW),
            make_task(newer),
        ),
        shots=make_shots(old),
    )

    with pytest.raises(ReportNotDeliverableError) as stale:
        repository.get_approved_session_data(old.session_id, old.report_id, old.revision)

    assert stale.value.reason is ReportNotDeliverableReason.REVISION_MISMATCH


@pytest.mark.parametrize(
    ("malformed_kind", "expected_reason"),
    [
        ("report", ReportNotDeliverableReason.REPORT_NOT_FOUND),
        ("task", ReportNotDeliverableReason.REVIEW_TASK_NOT_FOUND),
        ("shot", ReportNotDeliverableReason.SHOT_SET_INCOMPLETE),
    ],
)
def test_repository_deeply_revalidates_stored_models(malformed_kind, expected_reason):
    report = make_report()
    task = make_task(report)
    shots = list(make_shots(report))
    if malformed_kind == "report":
        report = ReportDocument.model_construct(
            **(report.model_dump(mode="python") | {"revision": True})
        )
    elif malformed_kind == "task":
        task = ReviewTask.model_construct(
            **(task.model_dump(mode="python") | {"report_revision": True})
        )
    else:
        shots[0] = Shot.model_construct(**(shots[0].model_dump(mode="python") | {"sequence": True}))
    repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(task,),
        shots=tuple(shots),
    )

    with pytest.raises(ReportNotDeliverableError) as malformed:
        repository.get_approved_session_data(report.session_id, report.report_id, 1)

    assert malformed.value.reason is expected_reason
