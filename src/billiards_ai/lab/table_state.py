"""Convert tracked event-domain balls into the canonical planning state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from billiards_ai.lab.events.models import TrackedFrame
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
    kind_for_identity,
)

GROUP_KINDS = {
    "cue": BallKind.CUE,
    "eight": BallKind.EIGHT,
    "solid": BallKind.SOLID,
    "stripe": BallKind.STRIPE,
    "unknown": BallKind.UNKNOWN,
}


def build_table_state(
    *,
    tracked_frame: TrackedFrame,
    identities: Mapping[str, Mapping[str, Any]],
    calibration_id: str,
    stationary: bool,
    source_refs: tuple[str, ...],
    radius_mm: float = 28.575,
) -> CanonicalTableState:
    """Build a new immutable state without modifying tracking evidence."""
    balls: list[CanonicalBall] = []
    for tracked in sorted(tracked_frame.balls, key=lambda item: item.track_id):
        result = identities.get(tracked.track_id, {})
        exact_identity = result.get("exact_identity", tracked.identity)
        identity = exact_identity if isinstance(exact_identity, str) else tracked.identity
        group = str(result.get("group", "unknown"))
        if identity != "unknown":
            kind = kind_for_identity(identity)
        else:
            kind = GROUP_KINDS.get(group, BallKind.UNKNOWN)
        identity_confidence = float(result.get("confidence", tracked.confidence))
        balls.append(
            CanonicalBall(
                track_id=tracked.track_id,
                identity=identity,
                kind=kind,
                identity_confidence=identity_confidence,
                center=Point(x_mm=tracked.x_mm, y_mm=tracked.y_mm),
                radius_mm=radius_mm,
                detection_confidence=tracked.confidence,
            )
        )
    return CanonicalTableState(
        frame_id=tracked_frame.frame_id,
        timestamp_ms=tracked_frame.timestamp_ms,
        calibration_id=calibration_id,
        motion_status=MotionStatus.STATIONARY if stationary else MotionStatus.MOVING,
        balls=tuple(balls),
        source_refs=source_refs,
    )
