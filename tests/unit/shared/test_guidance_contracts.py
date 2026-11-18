import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    GuidanceScene,
    GuidanceStatus,
    PhysicsResult,
    PhysicsStatus,
    PlannedShot,
    PocketId,
    Recommendation,
)
from billiards_ai.shared.models.table_state import BallKind


def _shot() -> PlannedShot:
    return PlannedShot(
        candidate_id="candidate-1",
        target_track_id="object-1",
        target_identity="1",
        target_kind=BallKind.SOLID,
        pocket=PocketId.TOP_RIGHT,
        status=GuidanceStatus.PLANNED,
        ghost_ball_center=Point(x_mm=1900, y_mm=500),
        cue_path=(Point(x_mm=500, y_mm=500), Point(x_mm=1900, y_mm=500)),
        object_path=(Point(x_mm=1957, y_mm=500), Point(x_mm=2540, y_mm=1270)),
        cut_angle_deg=12.0,
        cue_travel_mm=1400,
        object_travel_mm=966,
        evidence_confidence=0.91,
        score_components={"pot_tolerance": 0.8},
        score=0.8,
        evidence_refs=("frame:12",),
    )


def test_score_components_are_bounded() -> None:
    payload = _shot().model_dump()
    payload["score_components"] = {"pot_tolerance": 1.1}
    with pytest.raises(ValidationError, match="score components"):
        PlannedShot.model_validate(payload)


def test_recommendation_requires_evidence_and_at_most_three_candidates() -> None:
    with pytest.raises(ValidationError, match="at most three"):
        Recommendation(
            status=GuidanceStatus.GEOMETRY_ONLY,
            candidates=tuple(_shot().model_copy(update={"candidate_id": str(i)}) for i in range(4)),
            calibration_id="cal-1",
            evidence_refs=("frame:12",),
            planner_config_hash="a" * 64,
        )


def test_scene_is_device_independent_table_mm_contract() -> None:
    scene = GuidanceScene(
        recommendation_id="rec-1",
        calibration_id="cal-1",
        routes=(_shot(),),
        evidence_refs=("frame:12",),
    )
    assert scene.coordinate_space == "table_mm"
    assert "pixel" not in scene.model_dump_json()


def test_physics_failure_requires_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        PhysicsResult(
            candidate_id="candidate-1",
            status=PhysicsStatus.FAILED,
            backend_version="pooltool/0.6.0",
        )
