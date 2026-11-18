"""Shot boundary observation from calibrated ball motion."""

from __future__ import annotations

import math
from collections.abc import Sequence

from billiards_ai.lab.events.models import ShotInterval, TrackedFrame


def _frame_is_moving(previous: TrackedFrame, current: TrackedFrame, threshold: float) -> bool:
    before = previous.by_identity()
    after = current.by_identity()
    return any(
        math.hypot(
            after[identity].x_mm - before[identity].x_mm,
            after[identity].y_mm - before[identity].y_mm,
        )
        >= threshold
        for identity in before.keys() & after.keys()
    )


def segment_shots(
    frames: Sequence[TrackedFrame], *, motion_threshold_mm: float, still_duration_ms: int
) -> tuple[ShotInterval, ...]:
    if len(frames) < 2:
        return ()
    shots = []
    start_index: int | None = None
    last_motion_ms: int | None = None
    for index in range(1, len(frames)):
        moving = _frame_is_moving(frames[index - 1], frames[index], motion_threshold_mm)
        if moving:
            if start_index is None:
                start_index = index - 1
            last_motion_ms = frames[index].timestamp_ms
        elif (
            start_index is not None
            and last_motion_ms is not None
            and frames[index].timestamp_ms - last_motion_ms >= still_duration_ms
        ):
            shots.append(
                ShotInterval(
                    start_ms=frames[start_index].timestamp_ms,
                    end_ms=frames[index].timestamp_ms,
                    evidence_frame_ids=(frames[start_index].frame_id, frames[index].frame_id),
                    complete=True,
                )
            )
            start_index = None
            last_motion_ms = None
    if start_index is not None:
        shots.append(
            ShotInterval(
                start_ms=frames[start_index].timestamp_ms,
                end_ms=frames[-1].timestamp_ms,
                evidence_frame_ids=(frames[start_index].frame_id, frames[-1].frame_id),
                complete=False,
            )
        )
    return tuple(shots)
