"""Pure planning for bounded, overlapping propagation windows."""

from __future__ import annotations

from billiards_ai.lab.propagation.contracts import PropagationWindow


def plan_windows(
    *,
    asset_id: str,
    start_frame: int,
    end_frame: int,
    window_size: int = 30,
    overlap: int = 10,
) -> tuple[PropagationWindow, ...]:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("frame interval must be non-empty and non-negative")
    if window_size <= 0:
        raise ValueError("window size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be non-negative and smaller than window size")

    windows: list[PropagationWindow] = []
    cursor = start_frame
    while cursor < end_frame:
        window_end = min(cursor + window_size, end_frame)
        windows.append(
            PropagationWindow(
                asset_id=asset_id,
                ordinal=len(windows),
                start_frame=cursor,
                end_frame=window_end,
            )
        )
        if window_end == end_frame:
            break
        cursor = window_end - overlap
    return tuple(windows)
