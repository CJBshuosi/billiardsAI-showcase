"""Story 5.3 calibrated cue-ball trajectory production and review authorization."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest


def _frame(timestamp_ms: int, positions: dict[int, tuple[int, int]]):
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
        detector_version="synthetic/5.3",
        calibration=CalibrationLineage(
            calibration_id=UUID("60000000-0000-4000-8000-000000000001"),
            revision_id=UUID("60000000-0000-4000-8000-000000000002"),
            revision_number=2,
            status=CalibrationStatus.ACTIVE,
            table_profile_id="synthetic-9ft",
            table_length_mm=2540,
            table_width_mm=1270,
            table_configuration_fingerprint="c" * 64,
        ),
    )


def _segment_with_path():
    from billiards_ai.cloud.perception.segmenter import segment_shots

    rest = {0: (100, 100), 3: (900, 500)}
    samples = [(time, rest) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (101, 100), 3: (900, 500)}),
        (600, {0: (120, 110), 3: (900, 500)}),
        (700, {3: (910, 500)}),  # short cue-ball gap: never forward-filled
        (800, {0: (300, 180), 3: (930, 500)}),
        (900, {0: (300, 180), 3: (930, 500)}),
        (1000, {0: (300, 180), 3: (930, 500)}),
        (1100, {0: (300, 180), 3: (930, 500)}),
        (1200, {0: (300, 180), 3: (930, 500)}),
        (1300, {0: (300, 180), 3: (930, 500)}),
    ]
    result = segment_shots(tuple(_frame(timestamp, positions) for timestamp, positions in samples))
    assert result.failures == ()
    return result.segments[0]


def test_segment_observer_and_finalizer_preserve_real_calibrated_path_without_invention():
    from billiards_ai.cloud.perception.finalizer import finalize_shot
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.shared.models import FinalFieldSource, Point

    segment = _segment_with_path()
    assert tuple(sample.position for sample in segment.cue_ball_path) == (
        Point(x_mm=100, y_mm=100),
        Point(x_mm=101, y_mm=100),
        Point(x_mm=120, y_mm=110),
        Point(x_mm=300, y_mm=180),
    )
    assert tuple(sample.timestamp_ms for sample in segment.cue_ball_path) == (400, 500, 600, 800)

    observation = observe_shot(segment)
    assert observation.trajectory.value == tuple(
        sample.position for sample in segment.cue_ball_path
    )
    assert observation.trajectory.unsupported_reason is None

    shot = finalize_shot(
        shot_id=uuid4(),
        session_id=uuid4(),
        sequence=1,
        observation=observation,
        created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
    )
    assert shot.trajectory == list(observation.trajectory.value)
    assert shot.field_quality.trajectory.source is FinalFieldSource.CV
    assert shot.field_quality.trajectory.confidence >= 0.8


def test_calibration_lineage_survives_segment_observation_and_finalization():
    import billiards_ai.shared.models as models

    lineage_type = getattr(models, "CalibrationLineage", None)
    assert lineage_type is not None
    lineage = lineage_type(
        calibration_id=uuid4(),
        revision_id=uuid4(),
        revision_number=2,
        status=models.CalibrationStatus.ACTIVE,
        table_profile_id="table-53",
        table_length_mm=2540,
        table_width_mm=1270,
        table_configuration_fingerprint="c" * 64,
    )
    segment = _segment_with_path().model_copy(update={"calibration": lineage})
    observation = __import__(
        "billiards_ai.cloud.perception.observer", fromlist=["observe_shot"]
    ).observe_shot(segment)
    shot = __import__(
        "billiards_ai.cloud.perception.finalizer", fromlist=["finalize_shot"]
    ).finalize_shot(
        shot_id=uuid4(),
        session_id=uuid4(),
        sequence=1,
        observation=observation,
        created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
    )

    assert observation.trajectory_calibration == lineage
    assert shot.trajectory_calibration == lineage


def test_finalizer_fail_closes_missing_single_point_and_low_confidence_paths():
    from billiards_ai.cloud.perception.finalizer import finalize_shot
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.shared.models import (
        ObservationConfidence,
        ObservationField,
        ObservationSource,
        Point,
        TrajectoryUnavailableReason,
        UnsupportedReason,
    )

    observation = observe_shot(_segment_with_path())
    variants = (
        (
            ObservationField[tuple[Point, ...] | None](
                value=None,
                confidence=ObservationConfidence.UNKNOWN,
                source=ObservationSource.UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
                unsupported_reason=UnsupportedReason.TRAJECTORY_NOT_OBSERVED,
            ),
            TrajectoryUnavailableReason.NOT_OBSERVED,
        ),
        (
            ObservationField[tuple[Point, ...] | None](
                value=(Point(x_mm=100, y_mm=100),),
                confidence=ObservationConfidence.HIGH,
                source=ObservationSource.SEGMENT_CUE_BALL_PATH,
                unsupported_reason=None,
            ),
            TrajectoryUnavailableReason.INSUFFICIENT_POINTS,
        ),
        (
            ObservationField[tuple[Point, ...] | None](
                value=(Point(x_mm=100, y_mm=100), Point(x_mm=200, y_mm=200)),
                confidence=ObservationConfidence.LOW,
                source=ObservationSource.SEGMENT_CUE_BALL_PATH,
                unsupported_reason=None,
            ),
            TrajectoryUnavailableReason.LOW_CONFIDENCE,
        ),
    )

    for trajectory, expected_reason in variants:
        shot = finalize_shot(
            shot_id=uuid4(),
            session_id=uuid4(),
            sequence=1,
            observation=observation.model_copy(update={"trajectory": trajectory}),
            created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
        )
        assert shot.trajectory is None
        assert shot.trajectory_calibration == observation.trajectory_calibration
        assert shot.trajectory_unavailable_reason is expected_reason


def test_review_evidence_displays_trajectory_and_approval_requires_explicit_review():
    from billiards_ai.cloud.perception.finalizer import finalize_shot
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.cloud.review import apply_shot_reviews, review_session_evidence
    from billiards_ai.shared.models import (
        AccessActorRole,
        ReviewerIdentity,
        ShotReviewDecision,
        ShotReviewOutcome,
    )

    session_id = uuid4()
    shot = finalize_shot(
        shot_id=uuid4(),
        session_id=session_id,
        sequence=1,
        observation=observe_shot(_segment_with_path()),
        created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
    )
    evidence = review_session_evidence(session_id=session_id, shots=(shot,))
    assert evidence.shots[0].actual_path.points == tuple(shot.trajectory)
    assert evidence.shots[0].actual_path.source == shot.field_quality.trajectory.source
    assert evidence.shots[0].actual_path.confidence == shot.field_quality.trajectory.confidence

    common = {
        "shot_id": shot.id,
        "outcome": ShotReviewOutcome.APPROVED,
        "reviewer": ReviewerIdentity(
            principal_id="reviewer-53",
            roles=(AccessActorRole.REVIEWER,),
            credential_id="credential-53",
        ),
        "reviewed_at": datetime(2026, 7, 29, 11, tzinfo=UTC),
    }
    with pytest.raises(ValueError, match="trajectory|path|review"):
        apply_shot_reviews(
            session_id=session_id,
            shots=(shot,),
            decisions=(ShotReviewDecision(**common),),
        )
    application = apply_shot_reviews(
        session_id=session_id,
        shots=(shot,),
        decisions=(ShotReviewDecision(trajectory_reviewed=True, **common),),
    )
    assert application.shots[0].trajectory == shot.trajectory
