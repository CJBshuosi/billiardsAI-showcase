"""Server-owned monotonic replay clock and timeline projection."""

from __future__ import annotations

import bisect
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplayTimelineFrame:
    frame_id: int
    timestamp_ms: int
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayTimeline:
    frames: tuple[ReplayTimelineFrame, ...]
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ReplayProjection:
    position_ms: int
    playing: bool
    frame: ReplayTimelineFrame


def load_replay_timeline(path: Path, *, duration_ms: int) -> ReplayTimeline:
    """Load a verified-shape timeline for local UI projection."""

    source = path.resolve()
    if path.is_symlink() or not source.is_file():
        raise ValueError("replay timeline is missing or unsafe")
    if duration_ms <= 0:
        raise ValueError("media duration must be positive")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("replay timeline is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("replay timeline schema is unsupported")
    rows = payload.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("replay timeline frames are invalid")
    frames = []
    try:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("row is invalid")
            frame_id = int(row["frame_id"])
            timestamp_ms = int(row["timestamp_ms"])
            if frame_id < 0 or timestamp_ms < 0:
                raise ValueError("negative row")
            frames.append(
                ReplayTimelineFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    payload=dict(row),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("replay timeline frame is invalid") from exc
    identities = [(frame.frame_id, frame.timestamp_ms) for frame in frames]
    if any(
        current_id <= previous_id or current_ms < previous_ms
        for (previous_id, previous_ms), (current_id, current_ms) in zip(
            identities,
            identities[1:],
            strict=False,
        )
    ):
        raise ValueError("replay timeline frames must be ordered")
    if frames[-1].timestamp_ms > duration_ms:
        raise ValueError("replay timeline exceeds media duration")
    return ReplayTimeline(frames=tuple(frames), duration_ms=duration_ms)


class ReplayClock:
    """Synchronize multiple local clients to one injected monotonic timebase."""

    def __init__(
        self,
        timeline: ReplayTimeline,
        *,
        monotonic: Callable[[], float],
    ) -> None:
        self._timeline = timeline
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._accumulated_ms = 0
        self._started_at: float | None = None
        initial = float(monotonic())
        if not math.isfinite(initial):
            raise ValueError("monotonic clock must be finite")
        self._last_clock_value = initial

    @property
    def timeline(self) -> ReplayTimeline:
        """Return the immutable verified timeline owned by this clock."""

        return self._timeline

    def _now_locked(self) -> float:
        value = float(self._monotonic())
        if not math.isfinite(value) or value < self._last_clock_value:
            raise ValueError("monotonic clock moved backwards or became invalid")
        self._last_clock_value = value
        return value

    def _position_locked(self, now: float) -> int:
        position = self._accumulated_ms
        if self._started_at is not None:
            position += round((now - self._started_at) * 1000)
        if position >= self._timeline.duration_ms:
            self._accumulated_ms = self._timeline.duration_ms
            self._started_at = None
            return self._timeline.duration_ms
        return max(0, position)

    def start(self) -> None:
        with self._lock:
            now = self._now_locked()
            if self._started_at is None and self._accumulated_ms < self._timeline.duration_ms:
                self._started_at = now

    def pause(self) -> None:
        with self._lock:
            now = self._now_locked()
            self._accumulated_ms = self._position_locked(now)
            self._started_at = None

    def reset(self) -> None:
        with self._lock:
            self._now_locked()
            self._accumulated_ms = 0
            self._started_at = None

    def projection(self) -> ReplayProjection:
        with self._lock:
            now = self._now_locked()
            position = self._position_locked(now)
            timestamps = [frame.timestamp_ms for frame in self._timeline.frames]
            index = max(0, bisect.bisect_right(timestamps, position) - 1)
            return ReplayProjection(
                position_ms=position,
                playing=self._started_at is not None,
                frame=self._timeline.frames[index],
            )
