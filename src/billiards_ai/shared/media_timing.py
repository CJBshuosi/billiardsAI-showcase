"""Exact timestamp conversion and VFR sampling shared by media consumers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from billiards_ai.shared.models.media import TimeBase


@dataclass(frozen=True, slots=True)
class FrameSample:
    requested_timestamp_ms: int
    source_pts: int
    decode_index: int


def pts_to_milliseconds(pts: int, time_base: TimeBase) -> int:
    """Convert PTS using exact integer arithmetic and half-away-from-zero rounding."""

    numerator = pts * time_base.numerator * 1_000
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    rounded = (2 * magnitude + time_base.denominator) // (2 * time_base.denominator)
    return sign * rounded


def select_frame_samples(
    frame_pts: Sequence[int | None],
    *,
    time_base: TimeBase,
    interval_ms: int,
) -> tuple[FrameSample, ...]:
    """Select the first decodable VFR frame at or after every interval target."""

    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    valid = tuple((index, pts) for index, pts in enumerate(frame_pts) if pts is not None)
    if not valid:
        raise ValueError("at least one valid PTS is required")
    if any(current[1] <= previous[1] for previous, current in zip(valid, valid[1:], strict=False)):
        raise ValueError("valid PTS values must be strictly increasing")

    first_pts = valid[0][1]
    last_pts = valid[-1][1]
    samples: list[FrameSample] = []
    candidate_index = 0
    requested_ms = 0
    while (
        requested_ms * time_base.denominator <= (last_pts - first_pts) * time_base.numerator * 1_000
    ):
        while (
            valid[candidate_index][1] - first_pts
        ) * time_base.numerator * 1_000 < requested_ms * time_base.denominator:
            candidate_index += 1
        decode_index, source_pts = valid[candidate_index]
        samples.append(
            FrameSample(
                requested_timestamp_ms=requested_ms,
                source_pts=source_pts,
                decode_index=decode_index,
            )
        )
        requested_ms += interval_ms
    return tuple(samples)
