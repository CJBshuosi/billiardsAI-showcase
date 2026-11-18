"""Deterministic direct-pocket geometry in calibrated table millimetres."""

from __future__ import annotations

import hashlib
import json
from math import acos, degrees, hypot

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import GuidanceStatus, PlannedShot, PocketId
from billiards_ai.shared.models.table_state import CanonicalBall, CanonicalTableState

POCKET_CENTERS = {
    PocketId.BOTTOM_LEFT: Point(x_mm=0, y_mm=0),
    PocketId.BOTTOM_MIDDLE: Point(x_mm=TABLE_LENGTH_MM // 2, y_mm=0),
    PocketId.BOTTOM_RIGHT: Point(x_mm=TABLE_LENGTH_MM, y_mm=0),
    PocketId.TOP_LEFT: Point(x_mm=0, y_mm=TABLE_WIDTH_MM),
    PocketId.TOP_MIDDLE: Point(x_mm=TABLE_LENGTH_MM // 2, y_mm=TABLE_WIDTH_MM),
    PocketId.TOP_RIGHT: Point(x_mm=TABLE_LENGTH_MM, y_mm=TABLE_WIDTH_MM),
}


def ghost_ball_center(object_center: Point, pocket: Point, radius_mm: float) -> Point:
    dx = pocket.x_mm - object_center.x_mm
    dy = pocket.y_mm - object_center.y_mm
    length = hypot(dx, dy)
    if length <= 2 * radius_mm:
        raise ValueError("object ball is too close to pocket for stable ghost geometry")
    return Point(
        x_mm=round(object_center.x_mm - 2 * radius_mm * dx / length),
        y_mm=round(object_center.y_mm - 2 * radius_mm * dy / length),
    )


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x_mm - start.x_mm
    dy = end.y_mm - start.y_mm
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return float(point.distance_to(start))
    projection = ((point.x_mm - start.x_mm) * dx + (point.y_mm - start.y_mm) * dy) / length_sq
    projection = max(0.0, min(1.0, projection))
    nearest_x = start.x_mm + projection * dx
    nearest_y = start.y_mm + projection * dy
    return hypot(point.x_mm - nearest_x, point.y_mm - nearest_y)


def _corridor_blocked(
    balls: tuple[CanonicalBall, ...],
    start: Point,
    end: Point,
    *,
    excluded: set[str],
    moving_radius_mm: float,
    clearance_mm: float,
) -> bool:
    for ball in balls:
        if ball.track_id in excluded:
            continue
        threshold = moving_radius_mm + ball.radius_mm + clearance_mm
        if _point_segment_distance(ball.center, start, end) <= threshold:
            return True
    return False


def _cut_angle(cue: Point, ghost: Point, object_center: Point, pocket: Point) -> float:
    incoming = (ghost.x_mm - cue.x_mm, ghost.y_mm - cue.y_mm)
    outgoing = (pocket.x_mm - object_center.x_mm, pocket.y_mm - object_center.y_mm)
    incoming_length = hypot(*incoming)
    outgoing_length = hypot(*outgoing)
    if incoming_length == 0 or outgoing_length == 0:
        raise ValueError("shot vectors must have non-zero length")
    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (
        incoming_length * outgoing_length
    )
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _candidate_id(state: CanonicalTableState, target_track_id: str, pocket: PocketId) -> str:
    payload = {
        "schema": "direct-geometry/1",
        "calibration_id": state.calibration_id,
        "frame_id": state.frame_id,
        "target_track_id": target_track_id,
        "pocket": pocket.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def plan_direct_shot(
    *,
    state: CanonicalTableState,
    target_track_id: str,
    pocket: PocketId,
    clearance_mm: float = 3.0,
) -> PlannedShot:
    state.require_plannable()
    cue = state.cue_ball()
    target = state.ball_by_track(target_track_id)
    pocket_center = POCKET_CENTERS[pocket]
    ghost = ghost_ball_center(target.center, pocket_center, target.radius_mm)
    angle = _cut_angle(cue.center, ghost, target.center, pocket_center)
    reasons: list[str] = []
    if not ghost.is_on_table():
        reasons.append("ghost_ball_out_of_bounds")
    if _corridor_blocked(
        state.balls,
        cue.center,
        ghost,
        excluded={cue.track_id, target.track_id},
        moving_radius_mm=cue.radius_mm,
        clearance_mm=clearance_mm,
    ):
        reasons.append("cue_path_blocked")
    if _corridor_blocked(
        state.balls,
        target.center,
        pocket_center,
        excluded={cue.track_id, target.track_id},
        moving_radius_mm=target.radius_mm,
        clearance_mm=clearance_mm,
    ):
        reasons.append("object_path_blocked")
    return PlannedShot(
        candidate_id=_candidate_id(state, target_track_id, pocket),
        target_track_id=target.track_id,
        target_identity=target.identity,
        target_kind=target.kind,
        pocket=pocket,
        status=GuidanceStatus.UNSUPPORTED if reasons else GuidanceStatus.PLANNED,
        ghost_ball_center=ghost,
        cue_path=(cue.center, ghost),
        object_path=(target.center, pocket_center),
        cut_angle_deg=angle,
        cue_travel_mm=hypot(ghost.x_mm - cue.center.x_mm, ghost.y_mm - cue.center.y_mm),
        object_travel_mm=hypot(
            pocket_center.x_mm - target.center.x_mm,
            pocket_center.y_mm - target.center.y_mm,
        ),
        evidence_confidence=min(
            cue.detection_confidence,
            target.detection_confidence,
            target.identity_confidence,
        ),
        reasons=tuple(sorted(reasons)),
        evidence_refs=state.source_refs,
    )
