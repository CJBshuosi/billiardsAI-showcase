"""Synthetic snapshot-only acceptance tests for Story 2.7."""

import pytest


def _segment(before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]):
    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.segmentation import BallPosition, ShotSegment, TableSnapshot

    def snapshot(timestamp_ms: int, positions: dict[int, tuple[int, int]]):
        return TableSnapshot(
            timestamp_ms=timestamp_ms,
            positions=tuple(
                BallPosition(ball_id=ball_id, position=Point(x_mm=x_mm, y_mm=y_mm))
                for ball_id, (x_mm, y_mm) in sorted(positions.items())
            ),
        )

    return ShotSegment(
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before=snapshot(400, before),
        ball_positions_after=snapshot(900, after),
    )


def test_observer_derives_reviewable_snapshot_facts_and_preserves_unknown_verdicts():
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.shared.models.observation import (
        ObservationConfidence,
        ObservationSource,
        UnsupportedReason,
    )
    from billiards_ai.shared.models.shot import TargetZoneSource

    segment = _segment(
        before={0: (100, 100), 1: (400, 400), 2: (700, 700), 3: (900, 900)},
        after={0: (250, 100), 1: (490, 520), 3: (900, 900)},
    )

    first = observe_shot(segment)
    second = observe_shot(segment)

    assert first == second
    assert first.t_start_ms == 500
    assert first.t_end_ms == 900
    assert first.ball_positions_before.value == segment.ball_positions_before.positions
    assert first.ball_positions_after.value == segment.ball_positions_after.positions
    assert first.potted_balls.value == (2,)
    assert first.cue_ball_end.value.x_mm == 250
    assert tuple((item.ball_id, item.distance_mm) for item in first.ball_movements.value) == (
        (0, 150),
        (1, 150),
        (3, 0),
    )
    assert first.travel_distance_mm.value == 150
    assert first.travel_distance_mm.source is ObservationSource.ENDPOINT_DISPLACEMENT
    assert first.cushion_count.value is None
    assert first.cushion_count.unsupported_reason is (
        UnsupportedReason.CUSHION_CONTACT_NOT_OBSERVABLE
    )
    assert first.struck_ball.value is None
    assert first.struck_ball.confidence is ObservationConfidence.UNKNOWN
    assert tuple(item.ball_id for item in first.struck_ball_candidates) == (1, 2)
    assert first.struck_ball_candidates[0].unsupported_reason is (
        UnsupportedReason.FIRST_CONTACT_NOT_OBSERVABLE
    )
    assert first.is_foul.value is None
    assert first.is_foul.unsupported_reason is UnsupportedReason.FOUL_NOT_OBSERVABLE
    assert first.foul_evidence == ()
    assert first.target_zone_source is TargetZoneSource.NONE
    assert first.deviation_mm is None


def test_observer_treats_only_missing_before_keys_as_potted_and_rejects_new_after_keys():
    from billiards_ai.cloud.perception.observer import observe_shot

    potted = observe_shot(_segment(before={0: (100, 100), 1: (400, 400)}, after={0: (200, 100)}))

    assert potted.potted_balls.value == (1,)
    assert tuple(item.ball_id for item in potted.ball_positions_after.value) == (0,)

    with pytest.raises(ValueError, match="after snapshot contains ball IDs absent from before"):
        observe_shot(_segment(before={0: (100, 100)}, after={0: (200, 100), 1: (400, 400)}))


def test_potted_cue_ball_is_supported_foul_evidence_but_endpoint_distance_stay_unknown():
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.shared.models.observation import (
        FoulEvidenceKind,
        ObservationConfidence,
        UnsupportedReason,
    )

    observation = observe_shot(
        _segment(before={0: (100, 100), 1: (400, 400)}, after={1: (450, 400)})
    )

    assert observation.potted_balls.value == (0,)
    assert observation.cue_ball_end.value is None
    assert observation.cue_ball_end.unsupported_reason is UnsupportedReason.CUE_BALL_POTTED
    assert observation.travel_distance_mm.value is None
    assert observation.travel_distance_mm.confidence is ObservationConfidence.UNKNOWN
    assert observation.is_foul.value is True
    assert observation.is_foul.confidence is ObservationConfidence.HIGH
    assert len(observation.foul_evidence) == 1
    assert observation.foul_evidence[0].kind is FoulEvidenceKind.CUE_BALL_POTTED
    assert observation.foul_evidence[0].supports_foul is True


def test_segmented_terminal_missing_object_ball_is_exposed_as_auditable_pot():
    from billiards_ai.cloud.perception.observer import observe_shot

    observation = observe_shot(
        _segment(before={0: (100, 100), 1: (400, 400)}, after={0: (200, 100)})
    )

    assert observation.potted_balls.value == (1,)
    assert observation.struck_ball_candidates[0].ball_id == 1


def test_stationary_object_balls_produce_no_struck_candidate_without_erasing_unknown_reason():
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.shared.models.observation import UnsupportedReason

    observation = observe_shot(
        _segment(
            before={0: (100, 100), 1: (400, 400), 2: (700, 700)},
            after={0: (200, 100), 1: (400, 400), 2: (700, 700)},
        )
    )

    assert observation.struck_ball.value is None
    assert observation.struck_ball_candidates == ()
    assert observation.struck_ball.unsupported_reason is (
        UnsupportedReason.NO_OBJECT_BALL_MOVEMENT_EVIDENCE
    )


def test_missing_cue_ball_in_both_snapshots_stays_unknown_without_claiming_a_scratch():
    from billiards_ai.cloud.perception.observer import observe_shot
    from billiards_ai.shared.models.observation import UnsupportedReason

    observation = observe_shot(_segment(before={1: (400, 400)}, after={1: (450, 400)}))

    assert observation.potted_balls.value == ()
    assert observation.cue_ball_end.value is None
    assert observation.cue_ball_end.unsupported_reason is UnsupportedReason.CUE_BALL_NOT_OBSERVED
    assert observation.travel_distance_mm.value is None
    assert observation.travel_distance_mm.unsupported_reason is (
        UnsupportedReason.CUE_BALL_NOT_OBSERVED
    )
    assert observation.is_foul.value is None
    assert observation.foul_evidence == ()
