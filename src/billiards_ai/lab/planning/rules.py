"""Conservative eight-ball target legality for local guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from billiards_ai.shared.models.guidance import GuidanceStatus
from billiards_ai.shared.models.table_state import BallKind, CanonicalTableState


@dataclass(frozen=True, slots=True)
class EightBallContext:
    group: Literal["open", "solid", "stripe"]


@dataclass(frozen=True, slots=True)
class LegalTargetResult:
    status: GuidanceStatus
    track_ids: tuple[str, ...]
    reason: str | None = None


def legal_targets(
    state: CanonicalTableState,
    context: EightBallContext,
) -> LegalTargetResult:
    state.require_plannable()
    if context.group == "open":
        track_ids = tuple(
            sorted(
                ball.track_id
                for ball in state.balls
                if ball.kind in {BallKind.SOLID, BallKind.STRIPE}
            )
        )
        return LegalTargetResult(GuidanceStatus.PLANNED, track_ids)

    assigned_kind = BallKind.SOLID if context.group == "solid" else BallKind.STRIPE
    assigned = tuple(sorted(ball.track_id for ball in state.balls if ball.kind is assigned_kind))
    if assigned:
        return LegalTargetResult(GuidanceStatus.PLANNED, assigned)
    if any(ball.kind is BallKind.UNKNOWN for ball in state.balls):
        return LegalTargetResult(
            GuidanceStatus.ABSTAINED,
            (),
            "group_inventory_ambiguous",
        )
    eight = tuple(sorted(ball.track_id for ball in state.balls if ball.kind is BallKind.EIGHT))
    if eight:
        return LegalTargetResult(GuidanceStatus.PLANNED, eight)
    return LegalTargetResult(GuidanceStatus.ABSTAINED, (), "no_legal_target_visible")
