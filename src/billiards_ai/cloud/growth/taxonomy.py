"""Approved provisional R&D v1 ability taxonomy and geometry classification."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

from billiards_ai.shared.models.growth_metrics import (
    CALCULATION_POLICY_VERSION as CALCULATION_POLICY_VERSION,
)
from billiards_ai.shared.models.growth_metrics import (
    TAXONOMY_VERSION as TAXONOMY_VERSION,
)
from billiards_ai.shared.models.growth_metrics import (
    AngleBand,
    DistanceBand,
    RailRelation,
)
from billiards_ai.shared.models.metric_evidence import (
    PocketCode,
    ReviewedPoint,
    ReviewedTableGeometry,
)

POCKET_AIM_POINTS: dict[PocketCode, tuple[float, float]] = {
    PocketCode.BOTTOM_LEFT: (0.0, 0.0),
    PocketCode.BOTTOM_CENTER: (1270.0, 0.0),
    PocketCode.BOTTOM_RIGHT: (2540.0, 0.0),
    PocketCode.TOP_LEFT: (0.0, 1270.0),
    PocketCode.TOP_CENTER: (1270.0, 1270.0),
    PocketCode.TOP_RIGHT: (2540.0, 1270.0),
}


def round_half_up(value: float, places: int) -> float:
    if not math.isfinite(value):
        raise ValueError("taxonomy values must be finite")
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def classify_angle_band(angle_deg: float) -> AngleBand:
    angle = round_half_up(angle_deg, 3)
    if not 0.0 <= angle <= 90.0:
        raise ValueError("cut angle must be within 0-90 degrees")
    if angle <= 5.0:
        return AngleBand.STRAIGHT
    if angle <= 30.0:
        return AngleBand.SMALL
    if angle <= 50.0:
        return AngleBand.MEDIUM
    if angle <= 75.0:
        return AngleBand.LARGE
    return AngleBand.EXTREME


def classify_distance_band(distance_mm: float) -> DistanceBand:
    distance = round_half_up(distance_mm, 3)
    if distance <= 0.0:
        raise ValueError("shot distance must be positive")
    if distance < 450.0:
        return DistanceBand.VERY_SHORT
    if distance < 900.0:
        return DistanceBand.SHORT
    if distance < 1300.0:
        return DistanceBand.MEDIUM
    if distance < 1800.0:
        return DistanceBand.LONG
    return DistanceBand.VERY_LONG


def shot_distance_mm(cue: ReviewedPoint, object_ball: ReviewedPoint) -> float:
    if cue.calibration_key != object_ball.calibration_key:
        raise ValueError("shot-distance points must share calibration lineage")
    distance = math.hypot(object_ball.x_mm - cue.x_mm, object_ball.y_mm - cue.y_mm)
    if distance == 0.0:
        raise ValueError("cue and object ball centers must differ")
    return round_half_up(distance, 3)


def cut_angle_deg(
    cue: ReviewedPoint,
    object_ball: ReviewedPoint,
    pocket: PocketCode,
) -> float:
    if cue.calibration_key != object_ball.calibration_key:
        raise ValueError("cut-angle points must share calibration lineage")
    pocket_x, pocket_y = POCKET_AIM_POINTS[pocket]
    incoming = (object_ball.x_mm - cue.x_mm, object_ball.y_mm - cue.y_mm)
    outgoing = (pocket_x - object_ball.x_mm, pocket_y - object_ball.y_mm)
    incoming_norm = math.hypot(*incoming)
    outgoing_norm = math.hypot(*outgoing)
    if incoming_norm == 0.0 or outgoing_norm == 0.0:
        raise ValueError("cut angle requires non-zero vectors")
    cosine = max(
        -1.0,
        min(
            1.0,
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (incoming_norm * outgoing_norm),
        ),
    )
    angle = round_half_up(math.degrees(math.acos(cosine)), 3)
    if angle > 90.0:
        raise ValueError("R&D v1 does not support cut angles above 90 degrees")
    return angle


def _on_horizontal_rail_segment(x_mm: float) -> bool:
    return 43.0 <= x_mm <= 1224.0 or 1316.0 <= x_mm <= 2497.0


def _on_vertical_rail_segment(y_mm: float) -> bool:
    return 43.0 <= y_mm <= 1227.0


def classify_rail_relation(
    object_ball: ReviewedPoint,
    table: ReviewedTableGeometry,
) -> RailRelation | None:
    if object_ball.calibration_key != table.calibration_key:
        raise ValueError("rail classification requires matching calibration lineage")
    distances = {
        "left": object_ball.x_mm,
        "right": table.table_length_mm - object_ball.x_mm,
        "bottom": object_ball.y_mm,
        "top": table.table_width_mm - object_ball.y_mm,
    }
    nearest = min(distances.values())
    nearest_sides = tuple(side for side, distance in distances.items() if distance == nearest)
    valid = any(
        _on_vertical_rail_segment(object_ball.y_mm)
        if side in {"left", "right"}
        else _on_horizontal_rail_segment(object_ball.x_mm)
        for side in nearest_sides
    )
    if not valid:
        return None
    edge_gap = round_half_up(max(0.0, nearest - table.ball_diameter_mm / 2.0), 3)
    if edge_gap <= 2.0:
        return RailRelation.FROZEN_TO_RAIL
    if edge_gap <= 57.15:
        return RailRelation.NEAR_RAIL
    return RailRelation.OPEN_TABLE
