"""Conservative, local-only S1 direct-pot and near-contact target-zone planning.

Regions are intended targets supported by a near-straight stop-shot template,
not predicted endpoints or measured skill. Field accuracy is not asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, hypot, pi, sin

from billiards_ai.lab.planning.geometry import POCKET_CENTERS, plan_direct_shot
from billiards_ai.lab.planning.rules import EightBallContext, legal_targets
from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import GuidanceStatus, PlannedShot, PocketId
from billiards_ai.shared.models.table_state import BallKind, CanonicalTableState


@dataclass(frozen=True)
class StopRegion:
    center: Point
    radius_mm: int
    next_target_track_id: str
    next_pocket: PocketId
    evidence: str = "s1_pooltool_local_check/1;field_unvalidated"
    physical_witnesses: int = 0

    def contains(self, point: Point) -> bool:
        return hypot(point.x_mm - self.center.x_mm, point.y_mm - self.center.y_mm) <= self.radius_mm


@dataclass(frozen=True)
class SnapshotRecommendation:
    source_frame_id: int
    calibration_id: str
    route: PlannedShot | None
    region: StopRegion | None = None
    reason: str | None = None
    region_reason: str | None = None


def _check(cancelled):
    if cancelled():
        raise InterruptedError("snapshot guidance cancelled")


def _difficulty(route: PlannedShot) -> float:
    # A geometric ordering, never a calibrated probability of success.
    return (
        float(route.cut_angle_deg) / 75
        + float(route.cue_travel_mm) / 3000
        + float(route.object_travel_mm) / 2500
    )


def _candidates(state, context, cancelled):
    rows = []
    for target in legal_targets(state, context).track_ids:
        for pocket in PocketId:
            _check(cancelled)
            try:
                route = plan_direct_shot(state=state, target_track_id=target, pocket=pocket)
            except ValueError:
                continue
            ghost = route.ghost_ball_center
            radius = state.cue_ball().radius_mm
            if (
                route.status is GuidanceStatus.PLANNED
                and route.cut_angle_deg <= 75
                and radius <= ghost.x_mm <= TABLE_LENGTH_MM - radius
                and radius <= ghost.y_mm <= TABLE_WIDTH_MM - radius
            ):
                rows.append(route)
    return sorted(rows, key=lambda row: (_difficulty(row), row.candidate_id))


def _region(state, route, context, cancelled):
    if route.cut_angle_deg > 5:
        return None, "outside_s1_near_straight"
    if route.cue_travel_mm > 1200 or route.object_travel_mm > 900:
        return None, "outside_s1_distance"
    center = route.ghost_ball_center
    radius = 100
    cue = state.cue_ball()
    margin = radius + cue.radius_mm + 20
    if not (
        margin <= center.x_mm <= TABLE_LENGTH_MM - margin
        and margin <= center.y_mm <= TABLE_WIDTH_MM - margin
    ):
        return None, "near_rail_or_pocket"
    remaining = tuple(ball for ball in state.balls if ball.track_id != route.target_track_id)
    for ball in remaining:
        if ball.kind is not BallKind.CUE and (
            hypot(ball.center.x_mm - center.x_mm, ball.center.y_mm - center.y_mm)
            <= radius + cue.radius_mm + ball.radius_mm + 20
        ):
            return None, "occupied_stop_region"
    if any(center.distance_to(pocket) < margin + 100 for pocket in POCKET_CENTERS.values()):
        return None, "near_rail_or_pocket"
    after = state.model_copy(update={"balls": remaining})
    # Hypothetical successor only; never an authoritative match group update.
    next_context = context
    if context.group == "open":
        target_kind = next(
            ball.kind for ball in state.balls if ball.track_id == route.target_track_id
        )
        if target_kind not in {BallKind.SOLID, BallKind.STRIPE}:
            return None, "unresolved_successor_group"
        next_context = EightBallContext(group=target_kind.value)
    # Check the same next target/pocket across centre and two rings, not an
    # unrelated best next shot at every point. This is finite geometric sampling.
    samples = [center] + [
        Point(
            x_mm=round(center.x_mm + r * cos(i * pi / 8)),
            y_mm=round(center.y_mm + r * sin(i * pi / 8)),
        )
        for r in (50, 100)
        for i in range(16)
    ]
    next_options = _candidates(
        after.model_copy(
            update={
                "balls": tuple(
                    ball.model_copy(update={"center": center})
                    if ball.kind is BallKind.CUE
                    else ball
                    for ball in remaining
                )
            }
        ),
        next_context,
        cancelled,
    )
    for option in next_options:
        _check(cancelled)
        valid = True
        for point in samples:
            _check(cancelled)
            hypothetical = after.model_copy(
                update={
                    "balls": tuple(
                        ball.model_copy(update={"center": point})
                        if ball.kind is BallKind.CUE
                        else ball
                        for ball in remaining
                    )
                }
            )
            trial = plan_direct_shot(
                state=hypothetical, target_track_id=option.target_track_id, pocket=option.pocket
            )
            if (
                trial.status is not GuidanceStatus.PLANNED
                or trial.cut_angle_deg > 45
                or trial.object_travel_mm > 1200
                or trial.cue_travel_mm > 1500
            ):
                valid = False
                break
        if valid:
            from billiards_ai.lab.planning.s1_validation import stop_witnesses

            try:
                witnesses = stop_witnesses(state, route, center, radius, cancelled)
            except InterruptedError:
                raise
            except Exception:
                return None, "s1_physics_unavailable"
            if witnesses < 2:
                return None, "s1_stop_not_supported"
            return StopRegion(
                center, radius, option.target_track_id, option.pocket, physical_witnesses=witnesses
            ), None
    return None, "no_clear_next_shot"


def recommend_s1(
    state: CanonicalTableState,
    context: EightBallContext,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> SnapshotRecommendation:
    _check(cancelled)
    state.require_plannable()
    if (
        any(
            ball.detection_confidence < 0.85
            or ball.identity_confidence < 0.85
            or ball.kind is BallKind.UNKNOWN
            or not ball.center.is_on_table()
            for ball in state.balls
        )
        or not state.calibration_id
    ):
        return SnapshotRecommendation(
            state.frame_id, state.calibration_id, None, reason="untrusted_table_snapshot"
        )
    for index, ball in enumerate(state.balls):
        if any(
            hypot(ball.center.x_mm - other.center.x_mm, ball.center.y_mm - other.center.y_mm)
            < ball.radius_mm + other.radius_mm
            for other in state.balls[index + 1 :]
        ):
            return SnapshotRecommendation(
                state.frame_id, state.calibration_id, None, reason="overlapping_ball_snapshot"
            )
    candidates = _candidates(state, context, cancelled)
    if not candidates:
        return SnapshotRecommendation(
            state.frame_id, state.calibration_id, None, reason="no_reliable_direct_pot"
        )
    easiest = candidates[0]
    fallback_reason = "no_clear_next_shot"
    for candidate in candidates:
        # Only near-equal current difficulty may benefit from position value.
        if _difficulty(candidate) > _difficulty(easiest) + 0.05:
            break
        region, reason = _region(state, candidate, context, cancelled)
        if candidate is easiest:
            fallback_reason = reason
        if region is not None:
            return SnapshotRecommendation(state.frame_id, state.calibration_id, candidate, region)
    return SnapshotRecommendation(
        state.frame_id, state.calibration_id, easiest, region_reason=fallback_reason
    )
