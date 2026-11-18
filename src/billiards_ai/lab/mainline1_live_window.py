"""Incremental shot boundaries backed by the canonical segmentation policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from billiards_ai.lab.shot_local.models import ObservationFrame, ShotWindow
from billiards_ai.lab.shot_local.window import (
    ShotWindowConfig,
    ShotWindowTransitionCore,
    ShotWindowTransitionKind,
)


class LiveShotBoundaryKind(StrEnum):
    SHOT_STARTED = "SHOT_STARTED"
    SHOT_SETTLED = "SHOT_SETTLED"
    SHOT_INTERRUPTED = "SHOT_INTERRUPTED"


@dataclass(frozen=True, slots=True)
class LiveShotBoundary:
    kind: LiveShotBoundaryKind
    status: str
    window: ShotWindow | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"observed", "abstained"}:
            raise ValueError("live boundary status must be observed or abstained")
        if self.kind is LiveShotBoundaryKind.SHOT_SETTLED:
            if self.status != "observed" or self.window is None or not self.window.complete:
                raise ValueError("settled boundary requires a complete observed window")
        if self.kind is LiveShotBoundaryKind.SHOT_INTERRUPTED:
            if self.status != "abstained" or not self.reason:
                raise ValueError("interrupted boundary requires an abstention reason")


class IncrementalShotWindowCoordinator:
    """Emit live lifecycle boundaries without inventing a second shot policy."""

    def __init__(self, config: ShotWindowConfig, *, initial_shot_number: int = 0) -> None:
        self._core = ShotWindowTransitionCore(config, initial_shot_number=initial_shot_number)

    def push(self, frame: ObservationFrame) -> tuple[LiveShotBoundary, ...]:
        boundaries: list[LiveShotBoundary] = []
        for transition in self._core.push(frame):
            if transition.kind is ShotWindowTransitionKind.SHOT_STARTED:
                boundaries.append(
                    LiveShotBoundary(
                        kind=LiveShotBoundaryKind.SHOT_STARTED,
                        status="observed",
                        window=transition.window,
                    )
                )
            elif transition.kind is ShotWindowTransitionKind.SHOT_SETTLED:
                boundaries.append(
                    LiveShotBoundary(
                        kind=LiveShotBoundaryKind.SHOT_SETTLED,
                        status="observed",
                        window=transition.window,
                    )
                )
        return tuple(boundaries)

    def interrupt(self, reason: str) -> LiveShotBoundary:
        if not reason:
            raise ValueError("interrupt reason must not be empty")
        return LiveShotBoundary(
            kind=LiveShotBoundaryKind.SHOT_INTERRUPTED,
            status="abstained",
            window=self._core.interrupt(),
            reason=reason,
        )


class VisualShotBoundaryDetector:
    """Detect stable-moving-stable windows from lightweight visual motion flags."""

    def __init__(
        self,
        config: ShotWindowConfig,
        *,
        initial_shot_number: int = 0,
        motion_confirmation_window_ms: int = 0,
    ) -> None:
        if type(initial_shot_number) is not int or initial_shot_number < 0:
            raise ValueError("initial shot number must be a non-negative integer")
        if type(motion_confirmation_window_ms) is not int or motion_confirmation_window_ms < 0:
            raise ValueError("motion confirmation window must be non-negative")
        self._config = config
        self._confirmation_window_ms = motion_confirmation_window_ms
        self._pending_motion_ms = None
        self._shot_number = initial_shot_number
        self._previous_frame_id: int | None = None
        self._previous_timestamp_ms: int | None = None
        self._pending_start_frame_id: int | None = None
        self._pending_start_ms: int | None = None
        self._pending_before_frame_id: int | None = None
        self._moving_count = 0
        self._active_start_frame_id: int | None = None
        self._active_start_ms: int | None = None
        self._active_before_frame_id: int | None = None
        self._still_since_ms: int | None = None

    def push(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        moving: bool,
    ) -> tuple[LiveShotBoundary, ...]:
        if frame_id < 0 or timestamp_ms < 0 or type(moving) is not bool:
            raise ValueError("visual boundary sample is invalid")
        previous_frame_id = self._previous_frame_id
        previous_timestamp_ms = self._previous_timestamp_ms
        if previous_frame_id is not None and (
            frame_id <= previous_frame_id or timestamp_ms < previous_timestamp_ms
        ):
            raise ValueError("visual boundary samples must be ordered")
        self._previous_frame_id = frame_id
        self._previous_timestamp_ms = timestamp_ms
        if previous_frame_id is None:
            return ()

        if self._active_start_frame_id is None:
            if (
                self._confirmation_window_ms
                and self._pending_motion_ms is not None
                and timestamp_ms - self._pending_motion_ms >= self._confirmation_window_ms
            ):
                self._clear_pending_start()
            if not moving:
                if not (
                    self._confirmation_window_ms
                    and self._pending_motion_ms is not None
                    and timestamp_ms - self._pending_motion_ms < self._confirmation_window_ms
                ):
                    self._clear_pending_start()
                return ()
            if self._pending_start_frame_id is None:
                self._pending_start_frame_id = frame_id
                self._pending_start_ms = timestamp_ms
                self._pending_before_frame_id = previous_frame_id
                self._moving_count = 1
            else:
                self._moving_count += 1
            self._pending_motion_ms = timestamp_ms
            if self._moving_count < self._config.motion_confirm_frames:
                return ()
            self._active_start_frame_id = self._pending_start_frame_id
            self._active_start_ms = self._pending_start_ms
            self._active_before_frame_id = self._pending_before_frame_id
            return (
                LiveShotBoundary(
                    kind=LiveShotBoundaryKind.SHOT_STARTED,
                    status="observed",
                    window=self._window(frame_id, timestamp_ms, complete=False),
                ),
            )

        if moving:
            self._still_since_ms = None
            return ()
        if self._still_since_ms is None:
            assert previous_timestamp_ms is not None
            self._still_since_ms = previous_timestamp_ms
        if timestamp_ms - self._still_since_ms < self._config.still_duration_ms:
            return ()
        window = self._window(frame_id, timestamp_ms, complete=True)
        self._shot_number += 1
        self._active_start_frame_id = None
        self._active_start_ms = None
        self._active_before_frame_id = None
        self._still_since_ms = None
        self._clear_pending_start()
        return (
            LiveShotBoundary(
                kind=LiveShotBoundaryKind.SHOT_SETTLED,
                status="observed",
                window=window,
            ),
        )

    def interrupt(self, reason: str) -> LiveShotBoundary:
        if not reason:
            raise ValueError("interrupt reason must not be empty")
        window = None
        if self._active_start_frame_id is not None:
            assert self._previous_frame_id is not None
            assert self._previous_timestamp_ms is not None
            window = self._window(
                self._previous_frame_id,
                self._previous_timestamp_ms,
                complete=False,
            )
            self._shot_number += 1
        self._active_start_frame_id = None
        self._active_start_ms = None
        self._active_before_frame_id = None
        self._still_since_ms = None
        self._clear_pending_start()
        return LiveShotBoundary(
            kind=LiveShotBoundaryKind.SHOT_INTERRUPTED,
            status="abstained",
            window=window,
            reason=reason,
        )

    def _window(self, frame_id: int, timestamp_ms: int, *, complete: bool) -> ShotWindow:
        assert self._active_start_frame_id is not None
        assert self._active_start_ms is not None
        assert self._active_before_frame_id is not None
        return ShotWindow(
            shot_id=f"shot-{self._shot_number + 1:04d}",
            start_frame_id=self._active_start_frame_id,
            end_frame_id=frame_id,
            before_frame_id=self._active_before_frame_id,
            after_frame_id=frame_id if complete else None,
            start_ms=self._active_start_ms,
            end_ms=timestamp_ms,
            complete=complete,
        )

    def _clear_pending_start(self) -> None:
        self._pending_motion_ms = None
        self._pending_start_frame_id = None
        self._pending_start_ms = None
        self._pending_before_frame_id = None
        self._moving_count = 0
