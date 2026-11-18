from billiards_ai.lab.planning.ranking import rank_candidates
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import GuidanceStatus, PlannedShot, PocketId
from billiards_ai.shared.models.table_state import BallKind


def _candidate(candidate_id: str) -> PlannedShot:
    return PlannedShot(
        candidate_id=candidate_id,
        target_track_id="target",
        target_identity="1",
        target_kind=BallKind.SOLID,
        pocket=PocketId.TOP_RIGHT,
        status=GuidanceStatus.PLANNED,
        ghost_ball_center=Point(x_mm=1900, y_mm=600),
        cue_path=(Point(x_mm=500, y_mm=600), Point(x_mm=1900, y_mm=600)),
        object_path=(Point(x_mm=1957, y_mm=600), Point(x_mm=2540, y_mm=1270)),
        cut_angle_deg=20,
        cue_travel_mm=1400,
        object_travel_mm=900,
        evidence_confidence=0.9,
        evidence_refs=("frame:1",),
    )


def test_ranker_uses_stable_candidate_id_tiebreak_and_top_three() -> None:
    recommendation = rank_candidates(
        [_candidate("d"), _candidate("c"), _candidate("a"), _candidate("b")],
        physics={},
        calibration_id="cal-1",
    )
    assert [item.candidate_id for item in recommendation.candidates] == ["a", "b", "c"]
    assert recommendation.status is GuidanceStatus.GEOMETRY_ONLY
    assert all(item.score_components for item in recommendation.candidates)


def test_empty_candidates_abstain() -> None:
    recommendation = rank_candidates([], physics={}, calibration_id="cal-1")
    assert recommendation.status is GuidanceStatus.ABSTAINED
    assert recommendation.candidates == ()
    assert recommendation.fallback_reason == "no_supported_candidates"
