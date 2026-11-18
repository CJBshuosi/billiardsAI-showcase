from __future__ import annotations

import pytest

from billiards_ai.lab.field_assistance.guidance import (
    build_field_guidance,
    generic_power_display,
)
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PlannedShot,
    PocketId,
)
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


def _ball(track_id: str, kind: BallKind, x_mm: int, y_mm: int) -> CanonicalBall:
    return CanonicalBall(
        track_id=track_id,
        identity="cue" if kind is BallKind.CUE else "unknown",
        kind=kind,
        identity_confidence=0.9,
        center=Point(x_mm=x_mm, y_mm=y_mm),
        radius_mm=28.575,
        detection_confidence=0.9,
    )


def _state(*, target_kind: BallKind = BallKind.SOLID) -> CanonicalTableState:
    return CanonicalTableState(
        frame_id=8,
        timestamp_ms=240,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=(
            _ball("scene-0002-ball-01", BallKind.CUE, 500, 500),
            _ball("scene-0002-ball-02", target_kind, 1500, 650),
        ),
        source_refs=("artifact:detections.json#frame=8",),
    )


def _candidate(total_mm: float) -> PlannedShot:
    return PlannedShot(
        candidate_id=f"candidate-{total_mm}",
        target_track_id="target",
        target_identity="unknown",
        target_kind=BallKind.SOLID,
        pocket=PocketId.TOP_RIGHT,
        status=GuidanceStatus.PLANNED,
        ghost_ball_center=Point(x_mm=100, y_mm=100),
        cue_path=(Point(x_mm=0, y_mm=0), Point(x_mm=100, y_mm=100)),
        object_path=(Point(x_mm=100, y_mm=100), Point(x_mm=200, y_mm=200)),
        cue_travel_mm=total_mm,
        object_travel_mm=0,
        evidence_confidence=0.9,
        evidence_refs=("synthetic",),
    )


def test_field_guidance_keeps_only_rank_one_and_marks_power_generic() -> None:
    result = build_field_guidance(
        state=_state(),
        context=EightBallContext(group="solid"),
        scene_generation=2,
    )

    assert result.recommendation.status in {
        GuidanceStatus.GEOMETRY_ONLY,
        GuidanceStatus.SIMULATED,
    }
    assert 1 <= len(result.recommendation.candidates) <= 3
    assert result.scene is not None
    assert len(result.scene.routes) == 1
    assert result.display is not None
    assert result.display.scene_generation == 2
    assert result.display.candidate_id == result.scene.routes[0].candidate_id
    assert result.display.power.personalized is False


@pytest.mark.parametrize(
    ("travel_mm", "band", "percent"),
    (
        (700, "light", 20),
        (701, "medium_light", 35),
        (1101, "medium", 50),
        (1601, "medium_firm", 70),
        (2201, "firm", 90),
    ),
)
def test_generic_power_thresholds_are_deterministic(
    travel_mm: float,
    band: str,
    percent: int,
) -> None:
    display = generic_power_display(_candidate(travel_mm))

    assert display.band == band
    assert display.approximate_percent == percent
    assert display.personalized is False


def test_unknown_legal_inventory_abstains_without_a_visible_route() -> None:
    result = build_field_guidance(
        state=_state(target_kind=BallKind.UNKNOWN),
        context=EightBallContext(group="solid"),
        scene_generation=2,
    )

    assert result.recommendation.status is GuidanceStatus.ABSTAINED
    assert result.display is None
    assert result.scene is None
    assert result.abstained_reason == "group_inventory_ambiguous"


def test_negative_scene_generation_is_rejected() -> None:
    with pytest.raises(ValueError, match="scene generation"):
        build_field_guidance(
            state=_state(),
            context=EightBallContext(group="solid"),
            scene_generation=-1,
        )
