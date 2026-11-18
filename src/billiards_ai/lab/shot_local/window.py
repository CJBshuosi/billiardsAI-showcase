"""Identity-free shot segmentation and stable pre-shot anchor construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from billiards_ai.lab.shot_local.models import (
    ObservationFrame,
    ShotAnchor,
    ShotWindow,
)


@dataclass(frozen=True, slots=True)
class ShotWindowConfig:
    motion_threshold_mm: float
    motion_confirm_frames: int
    still_duration_ms: int

    def __post_init__(self) -> None:
        if self.motion_threshold_mm <= 0:
            raise ValueError("motion threshold must be positive")
        if self.motion_confirm_frames <= 0:
            raise ValueError("motion confirmation must be positive")
        if self.still_duration_ms < 0:
            raise ValueError("still duration must be non-negative")


class ShotWindowTransitionKind(StrEnum):
    SHOT_STARTED = "SHOT_STARTED"
    SHOT_SETTLED = "SHOT_SETTLED"
    STABLE_RECOVERED = "STABLE_RECOVERED"


@dataclass(frozen=True, slots=True)
class ShotWindowTransition:
    kind: ShotWindowTransitionKind
    window: ShotWindow | None = None


class ShotWindowTransitionCore:
    """Stateful form of the canonical stable-moving-stable policy."""

    def __init__(self, config: ShotWindowConfig, *, initial_shot_number: int = 0) -> None:
        if type(initial_shot_number) is not int or initial_shot_number < 0:
            raise ValueError("initial shot number must be a non-negative integer")
        self._config = config
        self._previous: ObservationFrame | None = None
        self._pending_start: ObservationFrame | None = None
        self._pending_before: ObservationFrame | None = None
        self._moving_transition_count = 0
        self._active_start: ObservationFrame | None = None
        self._active_before: ObservationFrame | None = None
        self._still_since_ms: int | None = None
        self._shot_number = initial_shot_number
        self._recovering = False
        self._recovery_still_since_ms: int | None = None

    @property
    def active(self) -> bool:
        return self._active_start is not None

    @property
    def last_frame(self) -> ObservationFrame | None:
        return self._previous

    def push(self, frame: ObservationFrame) -> tuple[ShotWindowTransition, ...]:
        previous = self._previous
        if previous is None:
            self._previous = frame
            return ()
        _validate_frame_pair(previous, frame)
        moving = _frame_change_mm(previous, frame) >= self._config.motion_threshold_mm
        self._previous = frame

        if self._recovering:
            if moving:
                self._recovery_still_since_ms = None
                return ()
            if self._recovery_still_since_ms is None:
                self._recovery_still_since_ms = previous.timestamp_ms
            if frame.timestamp_ms - self._recovery_still_since_ms < self._config.still_duration_ms:
                return ()
            self._recovering = False
            self._recovery_still_since_ms = None
            return (ShotWindowTransition(ShotWindowTransitionKind.STABLE_RECOVERED),)

        if self._active_start is None:
            if not moving:
                self._pending_start = None
                self._pending_before = None
                self._moving_transition_count = 0
                return ()
            if self._pending_start is None:
                self._pending_start = frame
                self._pending_before = previous
                self._moving_transition_count = 1
            else:
                self._moving_transition_count += 1
            if self._moving_transition_count < self._config.motion_confirm_frames:
                return ()
            self._active_start = self._pending_start
            self._active_before = self._pending_before
            return (
                ShotWindowTransition(
                    ShotWindowTransitionKind.SHOT_STARTED,
                    self.incomplete_window(),
                ),
            )

        if moving:
            self._still_since_ms = None
            return ()
        if self._still_since_ms is None:
            self._still_since_ms = previous.timestamp_ms
        if frame.timestamp_ms - self._still_since_ms < self._config.still_duration_ms:
            return ()

        window = self._complete_window(frame)
        self._clear_shot_state()
        return (ShotWindowTransition(ShotWindowTransitionKind.SHOT_SETTLED, window),)

    def incomplete_window(self) -> ShotWindow | None:
        if self._active_start is None or self._active_before is None or self._previous is None:
            return None
        return ShotWindow(
            shot_id=f"shot-{self._shot_number + 1:04d}",
            start_frame_id=self._active_start.frame_id,
            end_frame_id=self._previous.frame_id,
            before_frame_id=self._active_before.frame_id,
            after_frame_id=None,
            start_ms=self._active_start.timestamp_ms,
            end_ms=self._previous.timestamp_ms,
            complete=False,
        )

    def interrupt(self) -> ShotWindow | None:
        window = self.incomplete_window()
        self._clear_shot_state()
        self._recovering = True
        self._recovery_still_since_ms = None
        return window

    def _complete_window(self, frame: ObservationFrame) -> ShotWindow:
        assert self._active_start is not None
        assert self._active_before is not None
        return ShotWindow(
            shot_id=f"shot-{self._shot_number + 1:04d}",
            start_frame_id=self._active_start.frame_id,
            end_frame_id=frame.frame_id,
            before_frame_id=self._active_before.frame_id,
            after_frame_id=frame.frame_id,
            start_ms=self._active_start.timestamp_ms,
            end_ms=frame.timestamp_ms,
            complete=True,
        )

    def _clear_shot_state(self) -> None:
        if self._active_start is not None:
            self._shot_number += 1
        self._pending_start = None
        self._pending_before = None
        self._moving_transition_count = 0
        self._active_start = None
        self._active_before = None
        self._still_since_ms = None


def _directed_distance(first: ObservationFrame, second: ObservationFrame) -> float:
    if not first.observations:
        return 0.0
    if not second.observations:
        return math.inf
    return max(
        min(
            math.hypot(item.x_mm - candidate.x_mm, item.y_mm - candidate.y_mm)
            for candidate in second.observations
        )
        for item in first.observations
    )


def _frame_change_mm(first: ObservationFrame, second: ObservationFrame) -> float:
    if len(first.observations) != len(second.observations):
        return math.inf
    if not first.observations:
        return 0.0
    return max(_directed_distance(first, second), _directed_distance(second, first))


def _validate_frame_order(frames: Sequence[ObservationFrame]) -> None:
    for previous, current in pairwise(frames):
        _validate_frame_pair(previous, current)


def _validate_frame_pair(previous: ObservationFrame, current: ObservationFrame) -> None:
    if current.frame_id <= previous.frame_id or current.timestamp_ms < previous.timestamp_ms:
        raise ValueError("observation frames must be ordered")


def segment_shot_windows(
    frames: Sequence[ObservationFrame],
    config: ShotWindowConfig,
) -> tuple[ShotWindow, ...]:
    """Find stable–moving–stable windows without associating physical-ball IDs."""

    if len(frames) < 2:
        return ()
    _validate_frame_order(frames)

    core = ShotWindowTransitionCore(config)
    windows: list[ShotWindow] = []
    for frame in frames:
        for transition in core.push(frame):
            if transition.kind is ShotWindowTransitionKind.SHOT_SETTLED:
                assert transition.window is not None
                windows.append(transition.window)
    incomplete = core.incomplete_window()
    if incomplete is not None:
        windows.append(incomplete)
    return tuple(windows)


def build_shot_anchors(
    window: ShotWindow,
    frames: Sequence[ObservationFrame],
) -> tuple[ShotAnchor, ...]:
    """Snapshot the exact stable frame immediately before one shot."""

    matches = [frame for frame in frames if frame.frame_id == window.before_frame_id]
    if len(matches) != 1:
        raise ValueError("shot before frame must exist exactly once")
    ordered = sorted(
        matches[0].observations,
        key=lambda item: (item.group.value, item.x_mm, item.y_mm, item.detection_id),
    )
    return tuple(
        ShotAnchor(
            shot_id=window.shot_id,
            anchor_id=f"{window.shot_id}-anchor-{index:02d}",
            source_detection_id=item.detection_id,
            group=item.group,
            x_mm=item.x_mm,
            y_mm=item.y_mm,
            confidence=item.confidence,
        )
        for index, item in enumerate(ordered, start=1)
    )
