"""Pot observation from persistent disappearance at a calibrated pocket."""

from __future__ import annotations

import math
from collections.abc import Sequence

from billiards_ai.lab.events.models import BallState, EventStatus, EvidenceEvent, TrackedFrame
from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM

_POCKET_NAMES = (
    "top-left",
    "top-middle",
    "top-right",
    "bottom-left",
    "bottom-middle",
    "bottom-right",
)


def _pockets(table_length_mm: int, table_width_mm: int) -> tuple[tuple[int, int], ...]:
    return (
        (0, 0),
        (table_length_mm // 2, 0),
        (table_length_mm, 0),
        (0, table_width_mm),
        (table_length_mm // 2, table_width_mm),
        (table_length_mm, table_width_mm),
    )


def _pocket_id(
    x_mm: int,
    y_mm: int,
    radius_mm: float,
    *,
    table_length_mm: int,
    table_width_mm: int,
) -> str | None:
    candidates = sorted(
        (
            (math.hypot(x_mm - x, y_mm - y), name)
            for name, (x, y) in zip(
                _POCKET_NAMES,
                _pockets(table_length_mm, table_width_mm),
                strict=True,
            )
        ),
        key=lambda item: (item[0], item[1]),
    )
    return candidates[0][1] if candidates[0][0] <= radius_mm else None


def _track_key(ball: BallState) -> str:
    return ball.track_id or f"identity:{ball.identity}"


def observe_pots(
    frames: Sequence[TrackedFrame],
    *,
    persistence_frames: int,
    pocket_radius_mm: float = 90,
    table_length_mm: int = TABLE_LENGTH_MM,
    table_width_mm: int = TABLE_WIDTH_MM,
) -> tuple[EvidenceEvent, ...]:
    if persistence_frames <= 0:
        raise ValueError("pot persistence must be positive")
    track_keys = sorted({_track_key(ball) for frame in frames for ball in frame.balls})
    events = []
    for track_key in track_keys:
        for index in range(len(frames) - persistence_frames):
            present = next(
                (ball for ball in frames[index].balls if _track_key(ball) == track_key),
                None,
            )
            if present is None:
                continue
            pocket_id = _pocket_id(
                present.x_mm,
                present.y_mm,
                pocket_radius_mm,
                table_length_mm=table_length_mm,
                table_width_mm=table_width_mm,
            )
            if pocket_id is None:
                continue
            missing_window = frames[index + 1 : index + 1 + persistence_frames]
            if any(
                _track_key(ball) == track_key
                for frame in frames[index + 1 :]
                for ball in frame.balls
            ):
                continue
            events.append(
                EvidenceEvent(
                    event_type="pot",
                    status=EventStatus.OBSERVED,
                    timestamp_ms=frames[index + 1].timestamp_ms,
                    actor_identity=present.identity,
                    target_identity=None,
                    evidence_frame_ids=(
                        frames[index].frame_id,
                        frames[index + 1].frame_id,
                        missing_window[-1].frame_id,
                    ),
                    confidence=present.confidence,
                    reason=f"pocket:{pocket_id}",
                )
            )
            break
    return tuple(events)
