"""Capture-cadence optical-flow tracking between expensive detector anchors."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.shot_local.models import BallGroup
from billiards_ai.shared.calibration.homography import TableCalibration, pixel_to_table


class TrackerHealthState(StrEnum):
    RUNNING = "running"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class RealtimeTrackerConfig:
    working_width: int = 960
    working_height: int = 540
    minimum_features: int = 4
    maximum_features: int = 24
    forward_backward_gate_px: float = 1.5
    ambiguity_distance_px: float = 24.0
    processing_budget_ms: float = 16.7
    enforce_processing_budget: bool = True
    maximum_source_lag_frames: int = 2
    history_frames: int = 240
    allow_accounted_frame_gaps: bool = False
    maximum_accounted_frame_gap: int = 8
    perf_counter_ns: Callable[[], int] = field(
        default=time.perf_counter_ns,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.working_width <= 0 or self.working_height <= 0:
            raise ValueError("tracker working geometry must be positive")
        if self.minimum_features < 1 or self.maximum_features < self.minimum_features:
            raise ValueError("tracker feature limits are invalid")
        if (
            not math.isfinite(self.forward_backward_gate_px)
            or self.forward_backward_gate_px <= 0
            or not math.isfinite(self.ambiguity_distance_px)
            or self.ambiguity_distance_px <= 0
            or not math.isfinite(self.processing_budget_ms)
            or self.processing_budget_ms <= 0
        ):
            raise ValueError("tracker thresholds must be finite and positive")
        if (
            self.maximum_source_lag_frames < 0
            or self.history_frames < 2
            or self.maximum_accounted_frame_gap < 2
        ):
            raise ValueError("tracker lag and history limits are invalid")


@dataclass(frozen=True, slots=True)
class AnchorBall:
    track_id: str
    center_px: tuple[float, float]
    confidence: float
    group: BallGroup = BallGroup.UNKNOWN
    bbox_xyxy_px: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.track_id or not 0 <= self.confidence <= 1:
            raise ValueError("anchor ball identity and confidence are required")
        if not np.isfinite(self.center_px).all():
            raise ValueError("anchor ball center must be finite")


@dataclass(frozen=True, slots=True)
class AnchorObservation:
    capture_frame_id: int
    captured_at_ns: int
    balls: tuple[AnchorBall, ...]

    def __post_init__(self) -> None:
        if self.capture_frame_id < 0 or self.captured_at_ns < 0:
            raise ValueError("anchor frame identity is invalid")
        identifiers = [ball.track_id for ball in self.balls]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("anchor ball track IDs must be unique")


@dataclass(frozen=True, slots=True)
class TrackedBall:
    track_id: str
    group: BallGroup
    state: str
    provenance: str
    center_px: tuple[float, float] | None
    table_mm: tuple[int, int] | None
    confidence: float
    anchor_frame_id: int


@dataclass(frozen=True, slots=True)
class TrackedObservation:
    capture_frame_id: int
    captured_at_ns: int
    balls: tuple[TrackedBall, ...]


@dataclass(frozen=True, slots=True)
class TrackerHealth:
    state: TrackerHealthState
    processed_frames: int
    p50_processing_ms: float | None
    p95_processing_ms: float | None
    maximum_processing_ms: float | None
    maximum_source_lag_frames: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class TrackerDiagnosticSnapshot:
    state: TrackerHealthState
    processed_frames: int
    maximum_source_lag_frames: int
    reason: str | None
    # This cached statistic has its own processed_frames/time boundary; it may be absent or stale.
    last_computed_health: TrackerHealth | None


@dataclass(slots=True)
class _FlowBall:
    track_id: str
    group: BallGroup
    center_working: np.ndarray
    points: np.ndarray
    confidence: float
    anchor_frame_id: int


class RealtimeBallTracker:
    """Track every source frame and make uncertainty explicit."""

    def __init__(self, calibration: TableCalibration, config: RealtimeTrackerConfig) -> None:
        self._calibration = calibration
        self._config = config
        self._state = TrackerHealthState.RUNNING
        self._reason: str | None = None
        self._last_frame_id: int | None = None
        self._previous_gray: np.ndarray | None = None
        self._source_geometry: tuple[int, int] | None = None
        self._balls: dict[str, _FlowBall] = {}
        self._durations_ms: list[float] = []
        self._maximum_lag = 0
        self._last_computed_health: TrackerHealth | None = None
        self._history: deque[tuple[CaptureFrame, np.ndarray]] = deque(maxlen=config.history_frames)
        self._observations: deque[TrackedObservation] = deque(maxlen=config.history_frames)
        self._current: TrackedObservation | None = None

    @property
    def current(self) -> TrackedObservation | None:
        return self._current

    @property
    def history(self) -> tuple[tuple[CaptureFrame, np.ndarray], ...]:
        return tuple(self._history)

    @property
    def observations(self) -> tuple[TrackedObservation, ...]:
        return tuple(self._observations)

    @property
    def health(self) -> TrackerHealth:
        durations = np.asarray(self._durations_ms, dtype=np.float64)
        result = TrackerHealth(
            state=self._state,
            processed_frames=len(self._durations_ms),
            p50_processing_ms=(None if not durations.size else float(np.percentile(durations, 50))),
            p95_processing_ms=(None if not durations.size else float(np.percentile(durations, 95))),
            maximum_processing_ms=(None if not durations.size else float(durations.max())),
            maximum_source_lag_frames=self._maximum_lag,
            reason=self._reason,
        )
        self._last_computed_health = result
        return result

    @property
    def diagnostic_snapshot(self) -> TrackerDiagnosticSnapshot:
        """Read current state and cached statistics without walking timing history."""
        return TrackerDiagnosticSnapshot(
            state=self._state,
            processed_frames=len(self._durations_ms),
            maximum_source_lag_frames=self._maximum_lag,
            reason=self._reason,
            last_computed_health=self._last_computed_health,
        )

    def apply_anchor(
        self,
        anchor: AnchorObservation,
        frame: CaptureFrame,
    ) -> TrackedObservation:
        return self.apply_anchor_working(anchor, frame, self._decode_working(frame))

    def apply_anchor_working(
        self,
        anchor: AnchorObservation,
        frame: CaptureFrame,
        gray: np.ndarray,
    ) -> TrackedObservation:
        if frame.capture_frame_id != anchor.capture_frame_id:
            raise ValueError("anchor must be bound to its source frame")
        self._validate_working_gray(gray)
        scale_x = self._config.working_width / frame.width
        scale_y = self._config.working_height / frame.height
        balls: dict[str, _FlowBall] = {}
        for ball in anchor.balls:
            center = np.asarray(
                (ball.center_px[0] * scale_x, ball.center_px[1] * scale_y),
                dtype=np.float32,
            )
            points = self._features(gray, center, ball.bbox_xyxy_px, scale_x, scale_y)
            balls[ball.track_id] = _FlowBall(
                track_id=ball.track_id,
                group=ball.group,
                center_working=center,
                points=points,
                confidence=ball.confidence,
                anchor_frame_id=anchor.capture_frame_id,
            )
        self._balls = balls
        self._last_frame_id = frame.capture_frame_id
        self._previous_gray = gray
        self._source_geometry = (frame.width, frame.height)
        self._history.append((frame, gray))
        self._current = self._observation(
            frame,
            states={track_id: "confirmed" for track_id in balls},
            provenance="anchor",
        )
        self._observations.append(self._current)
        return self._current

    def track(
        self,
        frame: CaptureFrame,
        *,
        latest_capture_frame_id: int | None = None,
    ) -> TrackedObservation:
        started_at = self._config.perf_counter_ns()
        gray = self._decode_working(frame)
        return self.track_working(
            frame,
            gray,
            latest_capture_frame_id=latest_capture_frame_id,
            started_at_ns=started_at,
        )

    def track_working(
        self,
        frame: CaptureFrame,
        gray: np.ndarray,
        *,
        latest_capture_frame_id: int | None = None,
        started_at_ns: int | None = None,
    ) -> TrackedObservation:
        if self._state is TrackerHealthState.DEGRADED:
            raise ValueError("realtime tracker is degraded")
        if self._last_frame_id is None or self._previous_gray is None:
            raise ValueError("realtime tracker requires an anchor")
        frame_step = frame.capture_frame_id - self._last_frame_id
        if frame_step != 1 and (not self._config.allow_accounted_frame_gaps or frame_step <= 0):
            self._degrade("tracker source frames must be contiguous")
            raise ValueError("tracker source frames must be contiguous")
        if frame_step > self._config.maximum_accounted_frame_gap:
            self._degrade("tracker source gap exceeds safe flow window")
            raise ValueError("tracker source gap exceeds safe flow window")
        if self._source_geometry != (frame.width, frame.height):
            self._degrade("tracker source geometry changed")
            raise ValueError("tracker source geometry changed")

        self._validate_working_gray(gray)
        started_at = self._config.perf_counter_ns() if started_at_ns is None else started_at_ns
        current_gray = gray
        states = self._advance_balls(self._previous_gray, current_gray)
        for track_id, state in states.items():
            if state == "lost":
                # These features belong to the previous image, which advances
                # below. Only a trusted anchor can reacquire this ball.
                self._balls[track_id].points = np.empty((0, 1, 2), dtype=np.float32)
                self._balls[track_id].confidence = 0.0
        self._mark_ambiguities(states)
        self._previous_gray = current_gray
        self._last_frame_id = frame.capture_frame_id
        self._history.append((frame, current_gray))
        row = self._observation(frame, states=states, provenance="flow")
        self._current = row
        self._observations.append(row)

        elapsed_ms = (self._config.perf_counter_ns() - started_at) / 1_000_000
        self._durations_ms.append(elapsed_ms)
        latest = (
            frame.capture_frame_id if latest_capture_frame_id is None else latest_capture_frame_id
        )
        self._maximum_lag = max(self._maximum_lag, latest - frame.capture_frame_id)
        if (
            self._config.enforce_processing_budget
            and self.health.p95_processing_ms is not None
            and self.health.p95_processing_ms > self._config.processing_budget_ms
        ):
            self._degrade("tracker processing p95 exceeds realtime budget")
        elif self._maximum_lag > self._config.maximum_source_lag_frames:
            self._degrade("tracker source lag exceeds realtime budget")
        return row

    def _decode_working(self, frame: CaptureFrame) -> np.ndarray:
        return frame.decode_working_gray(
            width=self._config.working_width,
            height=self._config.working_height,
        )

    def _validate_working_gray(self, gray: np.ndarray) -> None:
        if gray.dtype != np.uint8 or gray.shape != (
            self._config.working_height,
            self._config.working_width,
        ):
            raise ValueError("tracker working grayscale geometry is invalid")

    def _features(
        self,
        gray: np.ndarray,
        center: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        mask = np.zeros_like(gray)
        if bbox is None:
            radius = max(6, round(min(gray.shape) / 80))
            cv2.circle(mask, tuple(np.rint(center).astype(int)), radius, 255, -1)
        else:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                mask,
                (max(0, round(x1 * scale_x)), max(0, round(y1 * scale_y))),
                (
                    min(gray.shape[1] - 1, round(x2 * scale_x)),
                    min(gray.shape[0] - 1, round(y2 * scale_y)),
                ),
                255,
                -1,
            )
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._config.maximum_features,
            qualityLevel=0.01,
            minDistance=2,
            mask=mask,
            blockSize=3,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32)

    def _advance_balls(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> dict[str, str]:
        states = {
            track_id: "lost"
            for track_id, ball in self._balls.items()
            if len(ball.points) < self._config.minimum_features
        }
        eligible = [
            (track_id, ball)
            for track_id, ball in self._balls.items()
            if len(ball.points) >= self._config.minimum_features
        ]
        if not eligible:
            return states
        lengths = [len(ball.points) for _, ball in eligible]
        offsets = np.cumsum((0, *lengths))
        points = np.concatenate([ball.points for _, ball in eligible], axis=0)
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if forward is None or forward_status is None:
            states.update({track_id: "lost" for track_id, _ in eligible})
            return states
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            forward,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if backward is None or backward_status is None:
            states.update({track_id: "lost" for track_id, _ in eligible})
            return states
        all_errors = np.linalg.norm(
            points.reshape(-1, 2) - backward.reshape(-1, 2),
            axis=1,
        )
        for index, (track_id, ball) in enumerate(eligible):
            start, end = int(offsets[index]), int(offsets[index + 1])
            errors = all_errors[start:end]
            valid = (
                forward_status[start:end].reshape(-1).astype(bool)
                & backward_status[start:end].reshape(-1).astype(bool)
                & np.isfinite(errors)
                & (errors <= self._config.forward_backward_gate_px)
            )
            states[track_id] = self._apply_flow_result(
                ball,
                points[start:end],
                forward[start:end],
                errors,
                valid,
                current_gray,
            )
        return states

    def _apply_flow_result(
        self,
        ball: _FlowBall,
        old: np.ndarray,
        forward: np.ndarray,
        errors: np.ndarray,
        valid: np.ndarray,
        current_gray: np.ndarray,
    ) -> str:
        valid_count = int(valid.sum())
        if valid_count < self._config.minimum_features:
            return "lost"
        old_points = old.reshape(-1, 2)[valid]
        new_points = forward.reshape(-1, 2)[valid]
        displacement = np.median(new_points - old_points, axis=0)
        next_center = ball.center_working + displacement
        if not (
            0 <= next_center[0] < current_gray.shape[1]
            and 0 <= next_center[1] < current_gray.shape[0]
        ):
            return "lost"
        support_ratio = valid_count / len(ball.points)
        mean_error = float(errors[valid].mean())
        flow_confidence = support_ratio * max(
            0.0,
            1.0 - mean_error / self._config.forward_backward_gate_px,
        )
        ball.center_working = next_center.astype(np.float32)
        ball.points = new_points.reshape(-1, 1, 2).astype(np.float32)
        ball.confidence = min(ball.confidence, ball.confidence * flow_confidence)
        return "confirmed"

    def _mark_ambiguities(self, states: dict[str, str]) -> None:
        candidates = list(self._balls.values())
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1 :]:
                threshold = self._config.ambiguity_distance_px
                if "lost" in {states[first.track_id], states[second.track_id]}:
                    threshold *= 2
                if np.linalg.norm(first.center_working - second.center_working) < threshold:
                    states[first.track_id] = "ambiguous"
                    states[second.track_id] = "ambiguous"

    def _observation(
        self,
        frame: CaptureFrame,
        *,
        states: dict[str, str],
        provenance: str,
    ) -> TrackedObservation:
        scale_x = frame.width / self._config.working_width
        scale_y = frame.height / self._config.working_height
        output: list[TrackedBall] = []
        for track_id in sorted(self._balls):
            ball = self._balls[track_id]
            state = states[track_id]
            center_px: tuple[float, float] | None = None
            table_mm: tuple[int, int] | None = None
            if state == "confirmed":
                center_px = (
                    float(ball.center_working[0] * scale_x),
                    float(ball.center_working[1] * scale_y),
                )
                point = pixel_to_table(self._calibration, *center_px)
                table_mm = (point.x_mm, point.y_mm)
            output.append(
                TrackedBall(
                    track_id=track_id,
                    group=ball.group,
                    state=state,
                    provenance=provenance,
                    center_px=center_px,
                    table_mm=table_mm,
                    confidence=ball.confidence if state == "confirmed" else 0.0,
                    anchor_frame_id=ball.anchor_frame_id,
                )
            )
        return TrackedObservation(
            capture_frame_id=frame.capture_frame_id,
            captured_at_ns=frame.captured_at_ns,
            balls=tuple(output),
        )

    def _degrade(self, reason: str) -> None:
        self._state = TrackerHealthState.DEGRADED
        self._reason = reason
