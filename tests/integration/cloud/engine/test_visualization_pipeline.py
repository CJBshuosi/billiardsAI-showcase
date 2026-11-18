"""Synthetic Story 5.3 frame-to-approved-overlay acceptance flow."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

CALIBRATION_ID = UUID("60000000-0000-4000-8000-000000000053")
CALIBRATION_REVISION_ID = UUID("60000000-0000-4000-8000-000000000054")


def _frame(timestamp_ms: int, cue_x: int, *, cue_visible: bool = True):
    from billiards_ai.shared.models import (
        BallCandidate,
        BallColor,
        CalibrationLineage,
        CalibrationStatus,
        DetectionSource,
        FrameDetection,
        FrameDetections,
        FrameQuality,
        PixelObservation,
        Point,
    )

    positions = {3: (900, 500)}
    if cue_visible:
        positions[0] = (cue_x, 100)
    colors = {0: BallColor.WHITE, 3: BallColor.RED}
    detections = tuple(
        FrameDetection(
            candidates=(BallCandidate(ball_id=ball_id, color=colors[ball_id], confidence=0.99),),
            position=Point(x_mm=x_mm, y_mm=y_mm),
            confidence=0.99,
            source=DetectionSource.CLASSIC_CV_CIRCLE_COLOR,
            raw_observation=PixelObservation(
                center_px=(float(x_mm), float(y_mm)),
                radius_px=10.0,
                circularity=0.99,
                color=colors[ball_id],
            ),
        )
        for ball_id, (x_mm, y_mm) in sorted(positions.items())
    )
    return FrameDetections(
        timestamp_ms=timestamp_ms,
        detections=detections,
        missing_ball_ids=tuple(ball_id for ball_id in range(16) if ball_id not in positions),
        frame_quality=FrameQuality.GOOD,
        detector_version="synthetic/5.3-e2e",
        calibration=CalibrationLineage(
            calibration_id=CALIBRATION_ID,
            revision_id=CALIBRATION_REVISION_ID,
            revision_number=1,
            status=CalibrationStatus.ACTIVE,
            table_profile_id="dilan-table-3",
            table_length_mm=2540,
            table_width_mm=1270,
            table_configuration_fingerprint="e" * 64,
        ),
    )


class ProjectionRepository:
    def __init__(self, projection):
        self.projection = projection

    def get_approved_visualization_projection(self, session_id, report_id, revision, shot_id):
        assert session_id == self.projection.session_id
        assert report_id == self.projection.report_id
        assert revision == self.projection.report_revision
        assert shot_id == self.projection.shot_id
        return self.projection


def test_synthetic_frames_flow_to_reviewed_deterministic_visualization_asset():
    from billiards_ai.cloud.engine.visualization import generate_visualization_asset
    from billiards_ai.cloud.perception.finalizer import finalize_shot
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.cloud.perception.segmenter import segment_shots
    from billiards_ai.cloud.review import apply_shot_reviews, review_session_evidence
    from billiards_ai.shared.models import (
        AccessActorRole,
        ApprovedActualPath,
        ApprovedActualPathStatus,
        ApprovedVisualizationProjection,
        BallPosition,
        CalibrationStatus,
        IdealPathBall,
        IdealPathEvent,
        IdealPathEventKind,
        IdealPathPoint,
        IdealPathProvenance,
        IdealPathResult,
        IdealPathStatus,
        ReviewerIdentity,
        ShotReviewDecision,
        ShotReviewOutcome,
        TypicalShotRuleVersion,
        TypicalShotSelection,
        TypicalShotSelectionReason,
        TypicalShotSelectionResult,
        VisualizationStatus,
    )

    frames = tuple(_frame(time, 100) for time in range(0, 500, 100)) + (
        _frame(500, 101),
        _frame(600, 140),
        _frame(700, 0, cue_visible=False),
        _frame(800, 500),
        _frame(900, 500),
        _frame(1000, 500),
        _frame(1100, 500),
        _frame(1200, 500),
        _frame(1300, 500),
    )
    segment = segment_shots(frames).segments[0]
    observation = observe_shot(segment)
    session_id = uuid4()
    shot = finalize_shot(
        shot_id=uuid4(),
        session_id=session_id,
        sequence=1,
        observation=observation,
        created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
    )
    evidence = review_session_evidence(session_id=session_id, shots=(shot,))
    decision = ShotReviewDecision(
        shot_id=shot.id,
        outcome=ShotReviewOutcome.APPROVED,
        reviewer=ReviewerIdentity(
            principal_id="reviewer-53",
            roles=(AccessActorRole.REVIEWER,),
            credential_id="credential-53",
        ),
        reviewed_at=datetime(2026, 7, 29, 11, tzinfo=UTC),
        trajectory_reviewed=True,
    )
    reviewed = apply_shot_reviews(
        session_id=session_id,
        shots=(shot,),
        decisions=(decision,),
    ).shots[0]
    assert evidence.shots[0].actual_path.points == tuple(reviewed.trajectory)

    report_id = UUID("30000000-0000-4000-8000-000000000053")
    calibration_id = CALIBRATION_ID
    calibration_revision_id = CALIBRATION_REVISION_ID
    fingerprint = "e" * 64
    projection = ApprovedVisualizationProjection(
        session_id=session_id,
        report_id=report_id,
        report_revision=1,
        shot_id=reviewed.id,
        shot_sequence=reviewed.sequence,
        struck_ball=3,
        ball_positions=tuple(
            BallPosition(ball_id=ball_id, position=point)
            for ball_id, point in sorted(reviewed.ball_positions_before.items())
        ),
        actual_path=ApprovedActualPath(
            status=ApprovedActualPathStatus.AVAILABLE,
            points=tuple(reviewed.trajectory),
            source=reviewed.field_quality.trajectory.source,
            confidence=reviewed.field_quality.trajectory.confidence,
        ),
        calibration_id=calibration_id,
        calibration_revision_id=calibration_revision_id,
        calibration_revision_number=1,
        calibration_status=CalibrationStatus.ACTIVE,
        table_profile_id="dilan-table-3",
        table_length_mm=2540,
        table_width_mm=1270,
        table_configuration_fingerprint=fingerprint,
    )
    selection = TypicalShotSelectionResult(
        session_id=session_id,
        report_id=report_id,
        report_revision=1,
        rule_version=TypicalShotRuleVersion.V1,
        selections=(
            TypicalShotSelection(
                shot_id=reviewed.id,
                sequence=1,
                selection_rank=1,
                selection_reason=TypicalShotSelectionReason.SEVERE_MISSED_POT,
                rule_version=TypicalShotRuleVersion.V1,
            ),
        ),
    )
    cue_start = reviewed.ball_positions_before[0]
    object_start = reviewed.ball_positions_before[3]
    ideal = IdealPathResult(
        status=IdealPathStatus.SUCCEEDED,
        reason=None,
        adapter_version="fake-1.0",
        engine_name="fake-physics",
        engine_version="1.0",
        source_configuration_version="chinese-eight-ball-9ft-v1",
        provenance=IdealPathProvenance(
            calibration_id=calibration_id,
            calibration_revision_id=calibration_revision_id,
            calibration_revision_number=1,
            table_profile_id="dilan-table-3",
            table_configuration_fingerprint=fingerprint,
            cue_ball=IdealPathBall(
                ball_id=0,
                position=IdealPathPoint(x_mm=cue_start.x_mm, y_mm=cue_start.y_mm),
            ),
            object_balls=(
                IdealPathBall(
                    ball_id=3,
                    position=IdealPathPoint(x_mm=object_start.x_mm, y_mm=object_start.y_mm),
                ),
            ),
            target_ball_id=3,
        ),
        events=(
            IdealPathEvent(
                sequence=1,
                kind=IdealPathEventKind.CUE_STRIKE,
                ball_id=0,
                position=IdealPathPoint(x_mm=cue_start.x_mm, y_mm=cue_start.y_mm),
            ),
            IdealPathEvent(
                sequence=2,
                kind=IdealPathEventKind.BALL_CONTACT,
                ball_id=0,
                other_ball_id=3,
                position=IdealPathPoint(x_mm=850, y_mm=480),
            ),
            IdealPathEvent(
                sequence=3,
                kind=IdealPathEventKind.POCKETED,
                ball_id=3,
                pocket="TOP_RIGHT",
                position=IdealPathPoint(x_mm=2540, y_mm=1270),
            ),
        ),
    )

    result = generate_visualization_asset(
        repository=ProjectionRepository(projection),
        session_id=session_id,
        report_id=report_id,
        revision=1,
        shot_id=reviewed.id,
        selection=selection,
        ideal_result=ideal,
    )
    assert result.status is VisualizationStatus.SUCCEEDED
    assert result.asset.source.shot_id == reviewed.id
    assert result.asset.actual.confidence == reviewed.field_quality.trajectory.confidence
