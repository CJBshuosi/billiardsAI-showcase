"""Cushion contacts require both rail proximity and a direction reversal."""

from __future__ import annotations

from collections.abc import Sequence

from billiards_ai.lab.events.models import EventStatus, EvidenceEvent, TrackedFrame
from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM


def observe_cushion_contacts(
    frames: Sequence[TrackedFrame], *, rail_distance_mm: int
) -> tuple[EvidenceEvent, ...]:
    events = []
    for previous, current, following in zip(frames, frames[1:], frames[2:], strict=False):
        states = previous.by_identity(), current.by_identity(), following.by_identity()
        for identity in sorted(states[0].keys() & states[1].keys() & states[2].keys()):
            before, at_rail, after = (state[identity] for state in states)
            near_vertical = min(at_rail.x_mm, TABLE_LENGTH_MM - at_rail.x_mm) <= rail_distance_mm
            near_horizontal = min(at_rail.y_mm, TABLE_WIDTH_MM - at_rail.y_mm) <= rail_distance_mm
            x_reversed = (at_rail.x_mm - before.x_mm) * (after.x_mm - at_rail.x_mm) < 0
            y_reversed = (at_rail.y_mm - before.y_mm) * (after.y_mm - at_rail.y_mm) < 0
            if not ((near_vertical and x_reversed) or (near_horizontal and y_reversed)):
                continue
            events.append(
                EvidenceEvent(
                    event_type="cushion",
                    status=EventStatus.OBSERVED,
                    timestamp_ms=current.timestamp_ms,
                    actor_identity=identity,
                    target_identity=None,
                    evidence_frame_ids=(previous.frame_id, current.frame_id, following.frame_id),
                    confidence=min(before.confidence, at_rail.confidence, after.confidence),
                )
            )
    return tuple(events)
