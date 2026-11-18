"""S1 must not trade an easy current pot for speculative position."""

import importlib.util

import pytest

from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


def scene(cue=(1270, 300), target=(1270, 800), *, obstacle=None):
    balls = [
        CanonicalBall(
            track_id=name,
            identity="cue" if kind is BallKind.CUE else "unknown",
            kind=kind,
            center=Point(x_mm=x, y_mm=y),
            radius_mm=28.575,
            identity_confidence=0.99,
            detection_confidence=0.99,
        )
        for name, kind, (x, y) in (
            ("cue", BallKind.CUE, cue),
            ("current", BallKind.SOLID, target),
            ("next", BallKind.SOLID, (1850, 950)),
        )
    ]
    if obstacle:
        balls.append(
            CanonicalBall(
                track_id="obstacle",
                identity="unknown",
                kind=BallKind.STRIPE,
                center=Point(x_mm=obstacle[0], y_mm=obstacle[1]),
                radius_mm=28.575,
                identity_confidence=0.99,
                detection_confidence=0.99,
            )
        )
    return CanonicalTableState(
        frame_id=42,
        timestamp_ms=700,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=tuple(balls),
        source_refs=("pixels:epoch-1#42",),
    )


def recommend(state, **kwargs):
    spec = importlib.util.find_spec("billiards_ai.lab.planning.s1")
    assert spec is not None, "S1 snapshot planner is not implemented"
    from billiards_ai.lab.planning.s1 import recommend_s1

    return recommend_s1(state, EightBallContext(group="solid"), **kwargs)


def test_near_straight_pot_has_broad_region_with_consistent_next_shot():
    result = recommend(scene())
    assert result.route.target_track_id == "current"
    assert result.route.pocket.value == "top_middle"
    assert result.region is not None
    assert result.region.radius_mm >= 100
    assert result.region.next_target_track_id == "next"
    assert result.region.contains(Point(x_mm=1270, y_mm=743))
    assert result.region.evidence == "s1_pooltool_local_check/1;field_unvalidated"
    assert result.region.physical_witnesses >= 2
    assert result.source_frame_id == 42


def test_oblique_pot_remains_available_without_claiming_s1_stop_region():
    result = recommend(scene(cue=(800, 350)))
    assert result.route is not None
    assert result.region is None
    assert result.region_reason == "outside_s1_near_straight"


def test_occupied_near_contact_area_cannot_be_painted_as_safe():
    result = recommend(scene(obstacle=(1370, 740)))
    assert result.route is not None
    assert result.region is None


def test_no_next_ball_has_pot_only_result():
    state = scene()
    state = state.model_copy(update={"balls": state.balls[:2]})
    result = recommend(state)
    assert result.route is not None
    assert result.region is None
    assert result.region_reason == "no_clear_next_shot"


def test_low_confidence_scene_abstains_instead_of_ignoring_obstacles():
    state = scene()
    state = state.model_copy(
        update={
            "balls": (
                state.balls[0],
                state.balls[1].model_copy(update={"detection_confidence": 0.2}),
                state.balls[2],
            )
        }
    )
    result = recommend(state)
    assert result.route is None
    assert result.reason == "untrusted_table_snapshot"


def test_cancellation_prevents_speculative_search():
    with pytest.raises(InterruptedError):
        recommend(scene(), cancelled=lambda: True)


def test_motion_is_not_a_planning_input():
    with pytest.raises(ValueError, match="stationary"):
        recommend(scene().model_copy(update={"motion_status": MotionStatus.MOVING}))


def test_open_table_next_shot_keeps_the_current_potted_group(monkeypatch):
    from billiards_ai.lab.planning import s1_validation
    from billiards_ai.lab.planning.s1 import recommend_s1

    monkeypatch.setattr(s1_validation, "stop_witnesses", lambda *args: 3)
    state = scene()
    state = state.model_copy(
        update={
            "balls": (*state.balls[:2], state.balls[2].model_copy(update={"kind": BallKind.STRIPE}))
        }
    )
    result = recommend_s1(state, EightBallContext(group="open"))
    assert result.route.target_track_id == "current"
    assert result.region is None, "opposite-group next shot has no legal positional value"
