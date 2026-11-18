"""Approved-only Story 5.3 visualization projection repository."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"


def _report():
    from billiards_ai.shared.models import ReportDocument

    return ReportDocument.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def _quality(*, trajectory_confidence=0.9):
    from billiards_ai.shared.models import FieldQuality, FinalFieldSource, ShotFieldQuality

    supported = FieldQuality(source=FinalFieldSource.CV, confidence=0.9)
    unknown = FieldQuality(source=FinalFieldSource.UNKNOWN, confidence=0.0)
    return ShotFieldQuality(
        ball_positions_before=supported,
        ball_positions_after=supported,
        cue_ball_end=supported,
        struck_ball=supported,
        potted_balls=supported,
        cushion_count=unknown,
        travel_distance_mm=supported,
        is_foul=unknown,
        trajectory=FieldQuality(
            source=(
                FinalFieldSource.UNKNOWN if trajectory_confidence is None else FinalFieldSource.CV
            ),
            confidence=0.0 if trajectory_confidence is None else trajectory_confidence,
        ),
    )


def _shots(report, calibration):
    from billiards_ai.shared.models import Confidence, Point, ReviewStatus, Shot, ShotAnalysisStatus

    return tuple(
        Shot(
            id=uuid4(),
            session_id=report.session_id,
            sequence=sequence,
            t_start_ms=(sequence - 1) * 1000,
            t_end_ms=sequence * 1000,
            ball_positions_before={0: Point(x_mm=100, y_mm=100), 3: Point(x_mm=900, y_mm=500)},
            ball_positions_after={0: Point(x_mm=700, y_mm=400), 3: Point(x_mm=930, y_mm=510)},
            cue_ball_end=Point(x_mm=700, y_mm=400),
            struck_ball=3,
            potted_balls=[],
            cushion_count=None,
            travel_distance_mm=650,
            is_foul=None,
            confidence=Confidence.HIGH,
            field_quality=_quality(trajectory_confidence=0.9 if sequence == 1 else None),
            analysis_status=ShotAnalysisStatus.ANALYZED,
            trajectory=(
                [Point(x_mm=100, y_mm=100), Point(x_mm=700, y_mm=400)] if sequence == 1 else None
            ),
            trajectory_calibration=__import__(
                "billiards_ai.shared.models", fromlist=["CalibrationLineage"]
            ).CalibrationLineage.from_record(calibration),
            review_status=ReviewStatus.PENDING,
            created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
        )
        for sequence in range(1, report.session_statistics.total_shot_count + 1)
    )


def _calibration():
    from billiards_ai.shared.models import (
        CalibrationRecord,
        CalibrationReference,
        CalibrationResolution,
        CalibrationStatus,
        CameraProfile,
        DistortionParameters,
        Point,
        TableProfile,
    )

    now = datetime(2026, 7, 29, 9, tzinfo=UTC)
    return CalibrationRecord(
        calibration_id=uuid4(),
        revision_id=uuid4(),
        revision_number=1,
        device_id="camera-3",
        camera_profile=CameraProfile(
            camera_profile_id="overhead-v1",
            position_fingerprint="a" * 64,
            lens_fingerprint="b" * 64,
        ),
        resolution=CalibrationResolution(width_px=1920, height_px=1080),
        table_profile=TableProfile(
            table_profile_id="dilan-table-3",
            length_mm=2540,
            width_mm=1270,
            configuration_fingerprint="c" * 64,
        ),
        distortion=DistortionParameters(),
        corners_px=((0.0, 0.0), (1919.0, 0.0), (1919.0, 1079.0), (0.0, 1079.0)),
        homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        validation_references=(
            CalibrationReference(pixel=(100.0, 100.0), table=Point(x_mm=100, y_mm=100)),
        ),
        source_image_sha256="d" * 64,
        status=CalibrationStatus.ACTIVE,
        validated_at=now,
        max_reference_error_mm=0.0,
        created_at=now,
    )


def _decision(shot, *, reviewed=True):
    from billiards_ai.shared.models import (
        AccessActorRole,
        ReviewerIdentity,
        ShotReviewDecision,
        ShotReviewOutcome,
    )

    return ShotReviewDecision(
        shot_id=shot.id,
        outcome=ShotReviewOutcome.APPROVED,
        reviewer=ReviewerIdentity(
            principal_id="reviewer-53",
            roles=(AccessActorRole.REVIEWER,),
            credential_id="credential-53",
        ),
        reviewed_at=datetime(2026, 7, 29, 11, tzinfo=UTC),
        trajectory_reviewed=reviewed,
    )


def _repository(*, task_status=None, reviewed=True):
    from billiards_ai.cloud.review import apply_shot_reviews, submit_review
    from billiards_ai.cloud.review.visualization_repository import (
        InMemoryApprovedVisualizationRepository,
    )
    from billiards_ai.shared.models import (
        ReviewTask,
        ReviewTaskStatus,
    )

    calibration = _calibration()
    report = _report()
    source_shots = _shots(report, calibration)
    task = ReviewTask(
        id=uuid4(),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        status=ReviewTaskStatus.IN_REVIEW,
    )
    decisions = tuple(
        _decision(shot, reviewed=shot.trajectory is not None) for shot in source_shots
    )
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=source_shots,
        decisions=decisions,
    )
    submission = submit_review(
        report=report,
        review_task=task,
        application=application,
        completed_at=datetime(2026, 7, 29, 12, tzinfo=UTC),
    )
    report = submission.report
    shots = submission.shots
    task = submission.review_task.model_copy(
        update={"status": task_status or submission.review_task.status}
    )
    return (
        InMemoryApprovedVisualizationRepository(
            reports=(report,),
            review_tasks=(task,),
            shots=shots,
            review_submissions=(submission,) if reviewed else (),
        ),
        report,
        shots,
        calibration,
    )


def test_projection_reuses_gate_and_exposes_only_exact_reviewed_visualization_data():
    repository, report, shots, calibration = _repository()
    projection = repository.get_approved_visualization_projection(
        report.session_id,
        report.report_id,
        report.revision,
        shots[0].id,
    )

    assert projection.session_id == report.session_id
    assert projection.report_id == report.report_id
    assert projection.report_revision == report.revision
    assert projection.shot_id == shots[0].id
    assert projection.actual_path.points == tuple(shots[0].trajectory)
    assert projection.actual_path.confidence == shots[0].field_quality.trajectory.confidence
    assert projection.calibration_id == calibration.calibration_id
    assert projection.calibration_revision_id == calibration.revision_id
    assert projection.table_configuration_fingerprint == (
        calibration.table_profile.configuration_fingerprint
    )
    serialized = projection.model_dump_json().lower()
    assert all(
        forbidden not in serialized
        for forbidden in ("raw_observation", "pixel", "media", "url", "provider_text")
    )


@pytest.mark.parametrize("status", ["PENDING_REVIEW", "IN_REVIEW", "REJECTED"])
def test_unified_delivery_gate_errors_propagate_unchanged(status):
    from billiards_ai.cloud.review import ReportNotDeliverableError
    from billiards_ai.shared.models import ReviewTaskStatus

    repository, report, shots, _ = _repository(task_status=ReviewTaskStatus[status])
    with pytest.raises(ReportNotDeliverableError) as raised:
        repository.get_approved_visualization_projection(
            report.session_id,
            report.report_id,
            report.revision,
            shots[0].id,
        )
    assert raised.value.reason.value in {
        "review_pending",
        "review_in_progress",
        "review_rejected",
    }


def test_revision_session_shot_and_boolean_revision_must_match_exactly():
    from billiards_ai.cloud.review import ReportNotDeliverableError

    repository, report, shots, _ = _repository()
    cases = (
        (uuid4(), report.report_id, report.revision, shots[0].id),
        (report.session_id, report.report_id, report.revision + 1, shots[0].id),
        (report.session_id, report.report_id, report.revision, uuid4()),
        (report.session_id, report.report_id, True, shots[0].id),
    )
    for args in cases:
        with pytest.raises(ReportNotDeliverableError):
            repository.get_approved_visualization_projection(*args)


def test_unreviewed_trajectory_is_projected_as_unavailable_without_geometry():
    from billiards_ai.shared.models import ApprovedActualPathStatus

    repository, report, shots, _ = _repository(reviewed=False)
    projection = repository.get_approved_visualization_projection(
        report.session_id,
        report.report_id,
        report.revision,
        shots[0].id,
    )
    assert projection.actual_path.status is ApprovedActualPathStatus.NOT_REVIEWED
    assert projection.actual_path.points == ()


def test_reviewed_missing_trajectory_preserves_calibration_and_reports_not_observed():
    from billiards_ai.shared.models import (
        ApprovedActualPathReason,
        ApprovedActualPathStatus,
    )

    repository, report, shots, calibration = _repository()
    projection = repository.get_approved_visualization_projection(
        report.session_id,
        report.report_id,
        report.revision,
        shots[1].id,
    )

    assert projection.actual_path.status is ApprovedActualPathStatus.UNAVAILABLE
    assert projection.actual_path.reason is ApprovedActualPathReason.NOT_OBSERVED
    assert projection.actual_path.points == ()
    assert projection.calibration_revision_id == calibration.revision_id


def test_general_approved_projection_remains_trajectory_free():
    from billiards_ai.cloud.review import InMemoryApprovedDataRepository

    _, report, shots, _ = _repository()
    general = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(
            __import__("billiards_ai.shared.models", fromlist=["ReviewTask"]).ReviewTask(
                id=uuid4(),
                session_id=report.session_id,
                report_id=report.report_id,
                report_revision=report.revision,
                status=__import__(
                    "billiards_ai.shared.models", fromlist=["ReviewTaskStatus"]
                ).ReviewTaskStatus.APPROVED,
            ),
        ),
        shots=shots,
    ).get_approved_session_data(report.session_id, report.report_id, report.revision)
    assert "trajectory" not in general.shots[0].model_dump()


def test_authorized_path_cannot_be_swapped_after_the_reviewer_saw_it():
    from billiards_ai.shared.models import ApprovedActualPathStatus, Point

    repository, report, shots, _ = _repository()
    shots[0].trajectory = [Point(x_mm=100, y_mm=100), Point(x_mm=500, y_mm=500)]
    projection = repository.get_approved_visualization_projection(
        report.session_id,
        report.report_id,
        report.revision,
        shots[0].id,
    )

    assert projection.actual_path.status is ApprovedActualPathStatus.NOT_REVIEWED
    assert projection.actual_path.points == ()


def test_same_session_same_max_revision_fork_has_no_unique_current_head():
    from billiards_ai.cloud.review import ReportNotDeliverableError
    from billiards_ai.cloud.review.visualization_repository import (
        InMemoryApprovedVisualizationRepository,
    )

    repository, report, shots, _ = _repository()
    fork = report.model_copy(update={"report_id": uuid4()})
    forked = InMemoryApprovedVisualizationRepository(
        reports=repository._reports + (fork,),
        review_tasks=repository._review_tasks,
        shots=shots,
        review_submissions=repository._review_submissions,
    )
    with pytest.raises(ReportNotDeliverableError):
        forked.get_approved_visualization_projection(
            report.session_id,
            report.report_id,
            report.revision,
            shots[0].id,
        )
