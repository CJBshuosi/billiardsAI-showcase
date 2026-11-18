"""Product multi-rate Mainline 1 capture and perception coordinator."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.records import (
    RecordedMediaV1,
    RecordingBoundaryV1,
    require_session_policy,
)
from billiards_ai.lab.events.models import TrackedFrame
from billiards_ai.lab.field_assistance.visual_motion import (
    RecordingMotionGap,
    RecordingVisualMotion,
    VisualMotionConfig,
    table_visual_motion_detected,
)
from billiards_ai.lab.mainline1_live_window import (
    IncrementalShotWindowCoordinator,
    LiveShotBoundary,
    LiveShotBoundaryKind,
    VisualShotBoundaryDetector,
)
from billiards_ai.lab.mainline1_realtime_anchor import (
    AnchorReplayCoordinator,
    AnchorReplayHealth,
    AnchorSchedule,
    AnchorWorkerHealth,
    HeavyAnchorResult,
    HeavyAnchorWorker,
)
from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureBackend,
    CaptureCadenceMonitor,
    CaptureContractError,
    CaptureFrame,
    CaptureHealth,
    RealtimeCaptureConfig,
)
from billiards_ai.lab.mainline1_realtime_diagnostics import (
    RealtimeShotDiagnostics,
    diagnose_realtime_shot,
)
from billiards_ai.lab.mainline1_realtime_refinement import (
    FactCoverageVerdict,
    RealtimeEvidenceCoverage,
    RealtimeShotRefiner,
)
from billiards_ai.lab.mainline1_realtime_ring import (
    AsyncMjpegSegmentRing,
    FrozenCaptureInterval,
    MjpegSegmentRing,
    RealtimeRingHealth,
)
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorBall,
    AnchorObservation,
    RealtimeTrackerConfig,
    TrackedObservation,
    TrackerHealth,
    TrackerHealthState,
)
from billiards_ai.lab.mainline1_recording_processing import (
    RecordingProcessingJournal,
    recording_processing_config_sha256,
)
from billiards_ai.lab.mainline1_session_recording import (
    CommittedRecordingSource,
    RecordedShotInterval,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJob,
    _require_sha256,
    materialize_frozen_interval,
    materialize_recorded_interval_reference,
)
from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
from billiards_ai.lab.shot_local.active_tracks import assign_calibrated_points
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.models import BallGroup, ObservationFrame, ShotWindow
from billiards_ai.lab.shot_local.observations import GroupIdentityClassifier
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.lab.vision.contracts import BallInstanceObservation
from billiards_ai.shared.calibration.homography import TableCalibration, pixel_to_table


@dataclass(frozen=True, slots=True)
class ProductRealtimeRuntimeConfig:
    ring_root: Path
    pre_roll_frames: int = 120
    post_roll_frames: int = 120
    maximum_shot_frames: int = 1800
    maximum_ring_bytes: int = 2 * 1024**3
    maximum_pending_ring_frames: int = 180
    tracker_width: int = 960
    tracker_height: int = 540
    tracker_history_seconds: float = 12.0
    capture_cancel_timeout_seconds: float = 0.5
    background_shutdown_timeout_seconds: float = 5.0
    stop_after_terminal_shots: int | None = None
    diagnostic_duration_seconds: float | None = None
    initial_shot_number: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.initial_shot_number) is not int
            or self.initial_shot_number < 0
            or self.pre_roll_frames < 0
            or self.post_roll_frames < 0
            or self.maximum_shot_frames <= 0
            or self.maximum_ring_bytes <= 0
            or self.maximum_pending_ring_frames <= 0
            or self.tracker_width <= 0
            or self.tracker_height <= 0
            or not math.isfinite(self.tracker_history_seconds)
            or self.tracker_history_seconds <= 0
            or not math.isfinite(self.capture_cancel_timeout_seconds)
            or self.capture_cancel_timeout_seconds <= 0
            or not math.isfinite(self.background_shutdown_timeout_seconds)
            or self.background_shutdown_timeout_seconds <= 0
            or (
                self.diagnostic_duration_seconds is not None
                and (
                    not math.isfinite(self.diagnostic_duration_seconds)
                    or self.diagnostic_duration_seconds <= 0
                )
            )
            or (self.stop_after_terminal_shots is not None and self.stop_after_terminal_shots <= 0)
        ):
            raise ValueError("product realtime runtime configuration is invalid")


@dataclass(frozen=True, slots=True)
class LiveSettledShot:
    shot_id: str
    start_frame_id: int
    settled_frame_id: int
    settled_at_ns: int
    observations: tuple[ObservationFrame, ...]
    publishable: bool
    shot_abstention_reason: str | None
    fact_coverage: Mapping[str, FactCoverageVerdict]
    source_epoch_id: str
    job_id: str
    rule_frames: tuple[TrackedFrame, ...] = ()
    rule_coverage: RealtimeEvidenceCoverage | None = None
    event_config: ShotLocalEventConfig | None = None
    event_config_digest: str | None = None


@dataclass(frozen=True, slots=True)
class ProductRealtimeEvent:
    kind: Literal[
        "SHOT_STARTED",
        "SHOT_SETTLED",
        "SHOT_ENQUEUED",
        "SHOT_INCOMPLETE",
        "SHOT_RECORDING_BOUNDARY",
        "SHOT_MEDIA_RECORDED",
        "CAPTURE_DEGRADED",
    ]
    frame_id: int | None = None
    reason: str | None = None
    shot_id: str | None = None
    settled: LiveSettledShot | None = None
    started_ms: int | None = None
    job: ShotAnalysisJob | None = None
    recording_boundary: RecordingBoundaryV1 | None = None
    recorded_media: RecordedMediaV1 | None = None


class ProductRealtimeEventChannelFailure(RuntimeError):
    """Terminal fail-closed condition for a live authority-event channel."""


class ProductRealtimeEventChannel:
    """Bounded nonblocking authority-event handoff for live consumers."""

    def __init__(
        self,
        *,
        maximum_pending_events: int = 64,
        shot_admission: Callable[[], bool] | None = None,
        recording_evidence_sink: Callable[[ProductRealtimeEvent], None] | None = None,
    ) -> None:
        if maximum_pending_events <= 0:
            raise ValueError("event channel capacity must be positive")
        self._maximum_pending_events = maximum_pending_events
        self._events: deque[ProductRealtimeEvent] = deque()
        self._condition = threading.Condition()
        self._failure_reason: str | None = None
        self._shot_admission = shot_admission
        self._recording_evidence_sink = recording_evidence_sink
        self.first_admitted_frame_id: int | None = None if shot_admission else 0

    def record_evidence(self, event: ProductRealtimeEvent) -> None:
        if self._recording_evidence_sink is not None and event.kind in {
            "SHOT_STARTED",
            "SHOT_RECORDING_BOUNDARY",
            "SHOT_MEDIA_RECORDED",
        }:
            self._recording_evidence_sink(event)

    def can_start_shots(self) -> bool:
        return self._shot_admission is None or self._shot_admission()

    def admit_frame(self, frame_id: int) -> bool:
        if self.first_admitted_frame_id is not None:
            return frame_id >= self.first_admitted_frame_id
        if not self.can_start_shots():
            return False
        self.first_admitted_frame_id = frame_id
        return True

    def publish(self, event: ProductRealtimeEvent) -> None:
        with self._condition:
            if self._failure_reason is not None:
                raise ProductRealtimeEventChannelFailure(self._failure_reason)
            if len(self._events) >= self._maximum_pending_events:
                reason = "product realtime event channel capacity exhausted"
                self._failure_reason = reason
                self._condition.notify_all()
                raise ProductRealtimeEventChannelFailure(reason)
            self._events.append(event)
            self._condition.notify()

    def receive(self, *, timeout_seconds: float | None = None) -> ProductRealtimeEvent | None:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("event receive timeout must not be negative")
        with self._condition:
            ready = self._condition.wait_for(
                lambda: bool(self._events) or self._failure_reason is not None,
                timeout=timeout_seconds,
            )
            if not ready:
                return None
            if self._events:
                return self._events.popleft()
            assert self._failure_reason is not None
            raise ProductRealtimeEventChannelFailure(self._failure_reason)

    @property
    def pending_events(self) -> int:
        with self._condition:
            return len(self._events)

    def fail(self, reason: str) -> None:
        if not reason:
            raise ValueError("event channel failure requires a reason")
        with self._condition:
            if self._failure_reason is None:
                self._failure_reason = reason
                self._condition.notify_all()

    def require_healthy(self) -> None:
        with self._condition:
            reason = self._failure_reason
        if reason is not None:
            raise ProductRealtimeEventChannelFailure(reason)


class ShotAnalysisJobStore(Protocol):
    def enqueue(self, job: ShotAnalysisJob) -> object: ...


class ShotEvidenceStore(Protocol):
    @property
    def health(self) -> RealtimeRingHealth: ...

    @property
    def maximum_queue_depth(self) -> int: ...

    def append(self, frame: CaptureFrame) -> None: ...

    def begin_shot(self, frame_id: int) -> None: ...

    def settle_shot(self, frame_id: int) -> None: ...

    def freeze_shot(self) -> FrozenCaptureInterval | RecordedShotInterval: ...

    def release_shot(self) -> None: ...

    def close(self) -> None: ...


class ShotBoundaryStore(Protocol):
    def record(self, boundary: LiveShotBoundary, *, recovered: bool = False) -> object: ...


@dataclass(frozen=True, slots=True)
class ProductRealtimeDiagnostics:
    accepted_anchor_frames: int
    accepted_anchor_frame_ids: tuple[int, ...]
    maximum_anchor_gap_ms: float | None
    anchor_frames_with_balls: int
    anchor_frames_with_cue_ball: int
    maximum_anchor_balls: int
    tracked_observation_frames: int
    tracked_frames_with_confirmed_balls: int
    tracked_frames_with_cue_ball: int
    tracked_frames_without_confirmed_balls: int
    maximum_confirmed_balls: int
    confirmed_ball_count_change_frames: int
    maximum_tracking_frame_gap: int
    maximum_frame_change_mm: float
    maximum_frame_change_frame_id: int | None
    maximum_cue_frame_change_mm: float
    maximum_cue_frame_change_frame_id: int | None
    motion_threshold_mm: float | None
    motion_threshold_crossings: int
    maximum_stage_duration_ms: Mapping[str, float]
    maximum_stage_frame_id: Mapping[str, int]
    maximum_frame_processing_ms: float
    maximum_frame_processing_frame_id: int | None
    first_shot_diagnostics: RealtimeShotDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class _CueBallCandidate:
    center_px: tuple[float, float]
    bbox_xyxy_px: tuple[float, float, float, float]
    confidence: float
    detection_id: str | None


def _detection_is_on_table(
    detection: BallInstanceObservation,
    calibration: TableCalibration,
) -> bool:
    point = pixel_to_table(calibration, detection.center.x, detection.center.y)
    return (
        0 <= point.x_mm <= calibration.table_length_mm
        and 0 <= point.y_mm <= calibration.table_width_mm
    )


def _detect_unique_cue_ball(
    image: np.ndarray,
    detections: tuple[BallInstanceObservation, ...],
    calibration: TableCalibration,
) -> _CueBallCandidate | None:
    """Supplement a detector-missed cue only from one unambiguous table-local blob.

    The RF-DETR candidate remains authoritative for every detected ball.  This
    fallback considers both a detector-missed ball and a detector proposal
    whose downstream identity classification missed the unique cue.
    """

    if image.ndim != 3 or image.shape[2] != 3 or not detections:
        return None
    diameters = np.asarray(
        [
            math.sqrt(
                (detection.bbox.right - detection.bbox.left)
                * (detection.bbox.bottom - detection.bbox.top)
            )
            for detection in detections
        ],
        dtype=np.float64,
    )
    diameters = diameters[np.isfinite(diameters) & (diameters > 0)]
    if not diameters.size:
        return None
    expected_diameter = float(np.median(diameters))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 105), (179, 95, 255))
    table_mask = np.zeros(white.shape, dtype=np.uint8)
    table_polygon = np.rint(np.asarray(calibration.corners_px)).astype(np.int32)
    cv2.fillConvexPoly(table_mask, table_polygon, 255)
    white = cv2.bitwise_and(white, table_mask)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    candidates: list[tuple[float, _CueBallCandidate]] = []
    for contour in cv2.findContours(
        white,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        left, top, width, height = cv2.boundingRect(contour)
        diameter = float(max(width, height))
        if (
            diameter < expected_diameter * 0.45
            or diameter > expected_diameter * 1.55
            or min(width, height) / diameter < 0.65
        ):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0 else 4.0 * math.pi * area / perimeter**2
        if circularity < 0.35:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            continue
        if (
            cv2.pointPolygonTest(
                table_polygon.astype(np.float32),
                (float(center_x), float(center_y)),
                True,
            )
            < expected_diameter * 0.25
        ):
            continue
        nearest_detection = min(
            detections,
            key=lambda detection: math.hypot(
                center_x - detection.center.x,
                center_y - detection.center.y,
            ),
        )
        nearest_distance = math.hypot(
            center_x - nearest_detection.center.x,
            center_y - nearest_detection.center.y,
        )
        white_fraction = min(1.0, area / (math.pi * radius**2))
        if white_fraction < 0.35:
            continue
        score = circularity * white_fraction
        candidates.append(
            (
                score,
                _CueBallCandidate(
                    center_px=(float(center_x), float(center_y)),
                    bbox_xyxy_px=(
                        float(left),
                        float(top),
                        float(left + width),
                        float(top + height),
                    ),
                    confidence=min(0.90, max(0.55, score)),
                    detection_id=(
                        nearest_detection.detection_id
                        if nearest_distance < expected_diameter * 0.60
                        else None
                    ),
                ),
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.20:
        return None
    return candidates[0][1]


def _observation_set_change_mm(
    previous: ObservationFrame,
    current: ObservationFrame,
) -> float:
    """Mirror the identity-free shot-window distance for diagnostic reporting."""

    if not previous.observations:
        return 0.0

    def directed(first: ObservationFrame, second: ObservationFrame) -> float:
        return max(
            min(
                math.hypot(
                    item.x_mm - candidate.x_mm,
                    item.y_mm - candidate.y_mm,
                )
                for candidate in second.observations
            )
            for item in first.observations
        )

    return max(directed(previous, current), directed(current, previous))


class ProductRealtimeRuntime:
    """Keep source capture independent from scheduled heavyweight inference."""

    def __init__(
        self,
        *,
        source: CaptureBackend,
        capture_config: RealtimeCaptureConfig,
        calibration: TableCalibration,
        detector: VisionBackend,
        config: ProductRealtimeRuntimeConfig,
        identity_classifier: GroupIdentityClassifier | None = None,
        shot_config: Mainline1ShotLocalConfig | None = None,
        job_store: ShotAnalysisJobStore | None = None,
        job_media_root: Path | None = None,
        quarantine_root: Path | None = None,
        session_id: str | None = None,
        capture_setup_id: str | None = None,
        calibration_artifact_id: str | None = None,
        config_sha256: str | None = None,
        identity_training_package_id: str | None = None,
        identity_training_package_sha256: str | None = None,
        event_channel: ProductRealtimeEventChannel | None = None,
        evidence_store: ShotEvidenceStore | None = None,
        boundary_ledger: ShotBoundaryStore | None = None,
        visual_motion_config: VisualMotionConfig | None = None,
        session_policy: str = "legacy",
    ) -> None:
        if (identity_training_package_id is None) != (identity_training_package_sha256 is None):
            raise ValueError("identity training package identity must be complete")
        if identity_training_package_sha256 is not None:
            if (
                not isinstance(identity_training_package_id, str)
                or not identity_training_package_id
            ):
                raise ValueError("identity training package ID is invalid")
            _require_sha256(identity_training_package_sha256, "identity training package")
        self._session_policy = require_session_policy(session_policy)
        if session_policy == "recording-v1" and (
            shot_config is None
            or session_id is None
            or job_media_root is None
            or evidence_store is None
            or visual_motion_config is None
            or job_store is not None
            or identity_classifier is not None
            or detector is not None
        ):
            raise ValueError("recording policy requires raw media, not semantic dependencies")
        self._recording_windows = deque()
        self._recording_active = None
        self._recording_user_end = False
        self._recording_end_reason = "USER_ENDED_RECORDING"
        job_dependencies = (
            job_store,
            job_media_root,
            quarantine_root,
            session_id,
            capture_setup_id,
            calibration_artifact_id,
            config_sha256,
        )
        if session_policy == "legacy" and (
            (shot_config is None and any(value is not None for value in job_dependencies))
            or (shot_config is not None and any(value is None for value in job_dependencies))
        ):
            raise ValueError("shot config and durable job dependencies must be supplied together")
        self._source = source
        self._capture_config = capture_config
        self._calibration = calibration
        self._detector = detector
        self._config = config
        self._shot_config = shot_config
        self._job_store = job_store
        self._job_media_root = job_media_root
        self._quarantine_root = quarantine_root
        self._session_id = session_id
        self._capture_setup_id = capture_setup_id
        self._calibration_artifact_id = calibration_artifact_id
        self._config_sha256 = config_sha256
        self._identity_training_package_id = identity_training_package_id
        self._identity_training_package_sha256 = identity_training_package_sha256
        self._event_channel = event_channel
        self._boundary_ledger = boundary_ledger
        self._visual_motion_config = visual_motion_config
        self._recording_processing = None
        self._recording_replay_through = None
        self._recording_replaying = False
        self._recording_original_input = None
        if session_policy == "recording-v1" and isinstance(source, CommittedRecordingSource):
            if boundary_ledger is None or not hasattr(boundary_ledger, "path"):
                raise ValueError("disk processing requires a durable boundary ledger")
            if event_channel is None or event_channel._recording_evidence_sink is None:
                raise ValueError(
                    "persistent recording processing requires a synchronous evidence sink"
                )
            if (
                source.progress.session_id != session_id
                or boundary_ledger.session_id != session_id
                or boundary_ledger.source_epoch_id != source.progress.source_epoch_id
            ):
                raise ValueError("recording processing session/epoch lineage mismatch")
            digest = recording_processing_config_sha256(
                capture_config=capture_config,
                calibration=calibration,
                window=shot_config.window,
                motion=visual_motion_config,
                runtime_config=config,
                config_sha256=config_sha256,
            )
            self._recording_processing = RecordingProcessingJournal(
                path=boundary_ledger.path,
                session_id=session_id,
                source_epoch_id=source._source_epoch_id,
                config_sha256=digest,
                initial_shot_number=config.initial_shot_number,
            )
            prior = self._recording_processing.state
            self._recording_original_input = prior["input_prefix"]
            if prior["input_prefix"] is not None:
                source.validate_input_prefix(prior["input_prefix"])
            if prior["checkpoint"] is not None:
                source.validate_input_prefix(prior["checkpoint"])
                self._recording_replay_through = prior["checkpoint"]["last_frame_id"]
            config = replace(config, initial_shot_number=prior["initial_shot_number"])
            self._config = config
        self._visual_previous_bgr = None
        self._recording_motion = (
            RecordingVisualMotion(
                calibration,
                visual_motion_config,
                still_duration_ms=shot_config.window.still_duration_ms,
            )
            if session_policy == "recording-v1"
            else None
        )
        self._event_channel_failed = False
        self._capture_monitor = CaptureCadenceMonitor(
            expected_fps=capture_config.fps,
            allow_accounted_frame_gaps=bool(getattr(source, "allow_accounted_frame_gaps", False)),
        )
        source_allows_accounted_gaps = bool(getattr(source, "allow_accounted_frame_gaps", False))
        tracker_config = RealtimeTrackerConfig(
            working_width=config.tracker_width,
            working_height=config.tracker_height,
            history_frames=max(
                2,
                math.ceil(capture_config.fps * config.tracker_history_seconds),
            ),
            allow_accounted_frame_gaps=source_allows_accounted_gaps,
            # A coalescing live subscription always operates on the newest
            # available frame. Its per-frame compute time is useful telemetry,
            # but crossing a 60-FPS budget does not make the observation false.
            # Ball loss and ambiguity still fail closed inside the tracker.
            enforce_processing_budget=not source_allows_accounted_gaps,
        )
        self._tracker_diagnostic_config = tracker_config
        self._first_shot_diagnostics: RealtimeShotDiagnostics | None = None
        self._replay = (
            None
            if session_policy == "recording-v1"
            else AnchorReplayCoordinator(calibration, tracker_config)
        )
        self._worker = (
            None
            if session_policy == "recording-v1"
            else HeavyAnchorWorker(
                detector,
                AnchorSchedule(
                    source_fps=round(capture_config.fps),
                    idle_fps=4,
                    active_fps=8,
                ),
                identity_classifier=identity_classifier,
            )
        )
        self._ring: ShotEvidenceStore = evidence_store or AsyncMjpegSegmentRing(
            MjpegSegmentRing(
                config.ring_root,
                expected_fps=round(capture_config.fps),
                pre_roll_frames=config.pre_roll_frames,
                post_roll_frames=config.post_roll_frames,
                maximum_shot_frames=config.maximum_shot_frames,
                maximum_bytes=config.maximum_ring_bytes,
                idle_onset_lookback_frames=config.maximum_pending_ring_frames,
            ),
            maximum_pending_frames=config.maximum_pending_ring_frames,
        )
        self._accepted_anchors: list[AnchorObservation] = []
        self._tracked_by_frame: dict[int, TrackedObservation] = {}
        self._last_online_frame_id: int | None = None
        self._last_motion_row: TrackedObservation | None = None
        self._active_until_frame_id = -1
        self._visual_boundary_detector = (
            None
            if shot_config is None or visual_motion_config is None
            else VisualShotBoundaryDetector(
                shot_config.window,
                initial_shot_number=config.initial_shot_number,
                motion_confirmation_window_ms=(
                    shot_config.window.still_duration_ms if session_policy == "recording-v1" else 0
                ),
            )
        )
        self._window_coordinator = (
            None
            if shot_config is None or self._visual_boundary_detector is not None
            else IncrementalShotWindowCoordinator(
                shot_config.window, initial_shot_number=config.initial_shot_number
            )
        )
        self._pending_window: ShotWindow | None = None
        self._next_track_id = 0
        self.maximum_capture_queue_lag_frames = 0
        self.maximum_detector_queue_depth = 1
        self._capture_cancelled = False
        self._retain_unsafe_interval = False
        self._diagnostic_previous: ObservationFrame | None = None
        self._diagnostic_tracked_frames = 0
        self._diagnostic_frames_with_balls = 0
        self._diagnostic_frames_with_cue = 0
        self._diagnostic_frames_without_balls = 0
        self._diagnostic_maximum_balls = 0
        self._diagnostic_count_changes = 0
        self._diagnostic_maximum_frame_gap = 0
        self._diagnostic_maximum_change_mm = 0.0
        self._diagnostic_maximum_change_frame_id: int | None = None
        self._diagnostic_maximum_cue_change_mm = 0.0
        self._diagnostic_maximum_cue_change_frame_id: int | None = None
        self._diagnostic_threshold_crossings = 0
        self._maximum_stage_duration_ms: dict[str, float] = {}
        self._maximum_stage_frame_id: dict[str, int] = {}
        self._latest_capture_lock = threading.Lock()
        self._latest_capture_frame: CaptureFrame | None = None

    @property
    def calibration(self) -> TableCalibration:
        return self._calibration

    @property
    def current_capture_frame(self) -> CaptureFrame | None:
        """Return the latest validated native MJPEG frame without decoding or copying it."""
        with self._latest_capture_lock:
            return self._latest_capture_frame

    @property
    def capture_health(self) -> CaptureHealth:
        return self._capture_monitor.health

    @property
    def anchor_health(self) -> AnchorWorkerHealth:
        return None if self._worker is None else self._worker.health

    @property
    def ring_health(self) -> RealtimeRingHealth:
        return self._ring.health

    @property
    def maximum_ring_queue_depth(self) -> int:
        return self._ring.maximum_queue_depth

    @property
    def current_tracking(self) -> TrackedObservation | None:
        return None if self._replay is None else self._replay.current

    @property
    def current_observation(self) -> ObservationFrame | None:
        """Return only a route-eligible calibrated observation from the live tracker."""
        current = self.current_tracking
        health = self.tracker_health
        if current is None or health is None or health.state is not TrackerHealthState.RUNNING:
            return None
        observation = self._observation_frame(current)
        return observation if observation.observations else None

    @property
    def tracker_health(self) -> TrackerHealth | None:
        return None if self._replay is None else self._replay.tracker_health

    @property
    def anchor_replay_health(self) -> AnchorReplayHealth:
        return None if self._replay is None else self._replay.health

    @property
    def anchors(self) -> tuple[AnchorObservation, ...]:
        return tuple(self._accepted_anchors)

    @property
    def diagnostics(self) -> ProductRealtimeDiagnostics:
        anchors = tuple(self._accepted_anchors)
        anchor_ball_counts = tuple(len(anchor.balls) for anchor in anchors)
        ordered_anchor_timestamps = tuple(
            anchor.captured_at_ns
            for anchor in sorted(anchors, key=lambda row: row.capture_frame_id)
        )
        return ProductRealtimeDiagnostics(
            accepted_anchor_frames=len(anchors),
            accepted_anchor_frame_ids=tuple(
                anchor.capture_frame_id
                for anchor in sorted(anchors, key=lambda row: row.capture_frame_id)
            ),
            maximum_anchor_gap_ms=(
                None
                if len(ordered_anchor_timestamps) < 2
                else max(
                    (current - previous) / 1_000_000
                    for previous, current in zip(
                        ordered_anchor_timestamps[:-1],
                        ordered_anchor_timestamps[1:],
                        strict=True,
                    )
                )
            ),
            anchor_frames_with_balls=sum(count > 0 for count in anchor_ball_counts),
            anchor_frames_with_cue_ball=sum(
                any(ball.group is BallGroup.CUE for ball in anchor.balls) for anchor in anchors
            ),
            maximum_anchor_balls=max(anchor_ball_counts, default=0),
            tracked_observation_frames=self._diagnostic_tracked_frames,
            tracked_frames_with_confirmed_balls=self._diagnostic_frames_with_balls,
            tracked_frames_with_cue_ball=self._diagnostic_frames_with_cue,
            tracked_frames_without_confirmed_balls=self._diagnostic_frames_without_balls,
            maximum_confirmed_balls=self._diagnostic_maximum_balls,
            confirmed_ball_count_change_frames=self._diagnostic_count_changes,
            maximum_tracking_frame_gap=self._diagnostic_maximum_frame_gap,
            maximum_frame_change_mm=self._diagnostic_maximum_change_mm,
            maximum_frame_change_frame_id=self._diagnostic_maximum_change_frame_id,
            maximum_cue_frame_change_mm=self._diagnostic_maximum_cue_change_mm,
            maximum_cue_frame_change_frame_id=(self._diagnostic_maximum_cue_change_frame_id),
            motion_threshold_mm=(
                None if self._shot_config is None else self._shot_config.window.motion_threshold_mm
            ),
            motion_threshold_crossings=self._diagnostic_threshold_crossings,
            maximum_stage_duration_ms=dict(self._maximum_stage_duration_ms),
            maximum_stage_frame_id=dict(self._maximum_stage_frame_id),
            maximum_frame_processing_ms=self._maximum_stage_duration_ms.get(
                "frame_processing",
                0.0,
            ),
            maximum_frame_processing_frame_id=self._maximum_stage_frame_id.get("frame_processing"),
            first_shot_diagnostics=self._first_shot_diagnostics,
        )

    def run(self) -> tuple[ProductRealtimeEvent, ...]:
        if self._session_policy == "recording-v1":
            return self._run_recording()
        stop = threading.Event()
        events: list[ProductRealtimeEvent] = []
        terminal_errors: list[ProductRealtimeEvent] = []
        failure: BaseException | None = None
        stop_reason = "source_stopped"
        first_captured_at_ns: int | None = None
        try:
            for frame in self._source.frames(stop):
                if self._event_channel is not None:
                    self._event_channel.require_healthy()
                frame_processing_started_ns = time.perf_counter_ns()
                if first_captured_at_ns is None:
                    first_captured_at_ns = frame.captured_at_ns
                with self._measure_stage("capture_health", frame.capture_frame_id):
                    self._validate_frame_mode(frame)
                    self._capture_monitor.observe(frame)
                with self._latest_capture_lock:
                    self._latest_capture_frame = frame
                pending_frames = (
                    int(getattr(self._source, "pending_frame_count", 0))
                    + self._replay.preprocessing_lag_frames
                )
                self.maximum_capture_queue_lag_frames = max(
                    self.maximum_capture_queue_lag_frames,
                    pending_frames,
                )
                # Queue depth is transient latency telemetry, not a capture failure.
                # Both queues retain ordered source frames, and the settled-shot
                # refinement below already abstains when tracking coverage is
                # incomplete or degraded.  Stopping here used to cancel the native
                # source and unnecessarily remove the live camera from the screen.
                with self._measure_stage("ring_enqueue", frame.capture_frame_id):
                    self._ring.append(frame)
                if self._visual_boundary_detector is not None:
                    with self._measure_stage("visual_boundary", frame.capture_frame_id):
                        self._handle_visual_boundary(frame, events)
                with self._measure_stage("tracker_retain", frame.capture_frame_id):
                    self._replay.retain(frame)
                with self._measure_stage("anchor_transfer", frame.capture_frame_id):
                    self._accept_completed_anchors()
                with self._measure_stage("tracking_window", frame.capture_frame_id):
                    self._handle_tracking(events)
                with self._measure_stage("shot_finalize", frame.capture_frame_id):
                    self._finalize_if_ready(frame, events, stop)
                with self._measure_stage("anchor_schedule", frame.capture_frame_id):
                    self._worker.submit(
                        frame,
                        active=frame.capture_frame_id <= self._active_until_frame_id,
                    )
                self._record_stage_duration(
                    "frame_processing",
                    frame.capture_frame_id,
                    (time.perf_counter_ns() - frame_processing_started_ns) / 1_000_000,
                )
                terminal_limit = self._config.stop_after_terminal_shots
                if (
                    terminal_limit is not None
                    and sum(event.kind in {"SHOT_ENQUEUED", "SHOT_INCOMPLETE"} for event in events)
                    >= terminal_limit
                ):
                    stop_reason = "terminal_shot_limit"
                    break
                diagnostic_duration = self._config.diagnostic_duration_seconds
                if (
                    diagnostic_duration is not None
                    and first_captured_at_ns is not None
                    and frame.captured_at_ns - first_captured_at_ns
                    >= diagnostic_duration * 1_000_000_000
                ):
                    stop_reason = "diagnostic_timeout"
                    break
        except KeyboardInterrupt:
            stop_reason = "operator_interrupt"
        except Exception as exc:
            failure = exc
            stop_reason = f"{type(exc).__name__}: {exc}"
            terminal_errors.append(
                ProductRealtimeEvent(
                    kind="CAPTURE_DEGRADED",
                    frame_id=(
                        None
                        if self._capture_monitor.health.received_frames == 0
                        else self._capture_monitor.health.received_frames - 1
                    ),
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            stop.set()
            try:
                self._cancel_source()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                terminal_errors.append(
                    ProductRealtimeEvent(
                        kind="CAPTURE_DEGRADED",
                        reason=f"capture cleanup failed: {type(exc).__name__}: {exc}",
                    )
                )
            try:
                self._worker.close(timeout_seconds=self._config.background_shutdown_timeout_seconds)
                self._accept_completed_anchors()
                self._replay.close(timeout_seconds=self._config.background_shutdown_timeout_seconds)
                self._accepted_anchors.extend(self._replay.poll_accepted())
                self._handle_tracking(events)
                # The last anchor can first reveal a settled window during
                # shutdown. Use only already captured post-roll, before closing
                # the ring; scheduling latency must not make it incomplete.
                latest = self.current_capture_frame
                if failure is None and latest is not None:
                    self._finalize_if_ready(latest, events, stop)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                terminal_errors.append(
                    ProductRealtimeEvent(
                        kind="CAPTURE_DEGRADED",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
            # Drain observations before interrupting their window state. An
            # in-flight first anchor can reveal motion only after cancellation.
            self._interrupt_active(events, stop_reason)
            if self._pending_window is not None:
                self._emit_event(
                    events,
                    ProductRealtimeEvent(
                        kind="SHOT_INCOMPLETE",
                        shot_id=self._pending_window.shot_id,
                        frame_id=self._pending_window.end_frame_id,
                        reason="source_stopped_before_post_roll",
                    ),
                    fail_on_channel_error=False,
                )
                self._retain_unsafe_interval = True
                self._pending_window = None
            # Field consumers revoke their channel on degradation. Publish the
            # original shot's terminal boundary before that revocation signal.
            for event in terminal_errors:
                self._emit_event(events, event, fail_on_channel_error=False)
            self._ring.close()
            self._capture_monitor.close()
        if failure is not None and not events:
            raise failure
        return tuple(events)

    def request_recording_end(self, *, reason="USER_ENDED_RECORDING"):
        if self._session_policy != "recording-v1":
            raise ValueError("manual recording seal requires recording policy")
        if reason not in {"USER_ENDED_RECORDING", "PROCESS_INTERRUPTED"}:
            raise ValueError("invalid recording end reason")
        self._recording_end_reason = reason
        self._recording_user_end = True

    @property
    def recording_processing_progress(self):
        journal = self._recording_processing
        return None if journal is None else journal.state

    def _admit_recording_segment(self, frame):
        journal = self._recording_processing
        if journal is None:
            return
        segment = self._source.current_segment_input
        original = self._recording_original_input
        if original is None or segment["segment_index"] > original["segment_index"]:
            journal.admit_segment(segment)
            self._recording_original_input = segment
        self._recording_replaying = (
            self._recording_replay_through is not None
            and frame.capture_frame_id <= self._recording_replay_through
        )

    def _checkpoint_recording_segment(self, frame):
        if self._recording_processing is None or self._recording_replaying:
            return
        segment = self._source.current_segment_input
        if frame.capture_frame_id == segment["last_frame_id"]:
            if frame.captured_at_ns != segment["last_captured_at_ns"]:
                raise ValueError("recording processing input timestamp changed")
            self._recording_processing.checkpoint(segment)

    def _run_recording(self):
        stop = threading.Event()
        events = []
        reason = "source_stopped"
        failure = None
        ownership = ExitStack()
        try:
            if self._recording_processing is not None:
                ownership.enter_context(
                    session_media_lock(
                        self._recording_processing.path.parent.resolve(),
                        "recording-detector",
                        exclusive=True,
                    )
                )
            for frame in self._source.frames(stop):
                self._admit_recording_segment(frame)
                if self._event_channel is not None:
                    self._event_channel.require_healthy()
                self._validate_frame_mode(frame)
                self._capture_monitor.observe(frame)
                with self._latest_capture_lock:
                    self._latest_capture_frame = frame
                self._ring.append(frame)
                self._handle_visual_boundary(frame, events)
                if (
                    self._recording_active is not None
                    and frame.capture_frame_id - self._recording_active.start_frame_id + 1
                    > self._config.maximum_shot_frames
                ):
                    self._apply_recording_boundary(
                        self._visual_boundary_detector.interrupt("maximum_shot_frames"),
                        events,
                    )
                self._finalize_recording_windows(frame.capture_frame_id, events)
                self._checkpoint_recording_segment(frame)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failure = exc
        finally:
            stop.set()
            if failure is None and self._recording_user_end:
                reason = self._recording_end_reason
            terminal_input = (
                self._recording_processing is None
                or (failure is None and not self._source.progress.cancelled)
                or self._source.terminal_capture_failure is not None
            )
            if terminal_input:
                # Only real input exhaustion/capture interruption seals windows.
                # Processing, sink, checkpoint and reference I/O faults leave
                # evidence replayable rather than inventing an immutable cutoff.
                self._recording_replaying = False
            try:
                self._cancel_source()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                    reason = f"capture cleanup failed: {type(exc).__name__}: {exc}"
            try:
                last = self._latest_capture_frame
                if self._recording_active is not None and terminal_input:
                    interrupted = self._visual_boundary_detector.interrupt(reason)
                    if interrupted.window is not None:
                        self._apply_recording_boundary(interrupted, events, closing=True)
                if last is not None and terminal_input:
                    self._finalize_recording_windows(last.capture_frame_id, events, closing=True)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                    reason = f"recording finalization failed: {type(exc).__name__}: {exc}"
            finally:
                try:
                    self._ring.close()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                        reason = f"recording cleanup failed: {type(exc).__name__}: {exc}"
                finally:
                    try:
                        self._capture_monitor.close()
                    finally:
                        ownership.close()
            if failure is not None:
                self._emit_event(
                    events,
                    ProductRealtimeEvent(
                        kind="CAPTURE_DEGRADED",
                        reason=reason,
                    ),
                    fail_on_channel_error=False,
                )
        if failure is not None and not events:
            raise failure
        return tuple(events)

    def _apply_recording_boundary(self, boundary, events, *, closing=False):
        window = boundary.window
        if boundary.kind is LiveShotBoundaryKind.SHOT_STARTED:
            self._recording_active = window
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_STARTED",
                    shot_id=window.shot_id,
                    frame_id=window.start_frame_id,
                    started_ms=window.start_ms,
                ),
                fail_on_channel_error=not closing,
            )
            return
        metadata = RecordingBoundaryV1(
            window.shot_id,
            self._latest_capture_frame.source_epoch_id,
            window.start_frame_id,
            window.end_frame_id,
            window.start_ms,
            window.end_ms,
            window.complete,
            None if window.complete else boundary.reason,
        )
        self._recording_active = None
        capacity_exceeded = len(self._recording_windows) >= 32
        if not capacity_exceeded:
            # Retain the real boundary before UI delivery can fail. The shutdown
            # path must still finalize its media after the active shot is cleared.
            self._recording_windows.append(metadata)
        try:
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_RECORDING_BOUNDARY",
                    shot_id=window.shot_id,
                    frame_id=window.end_frame_id,
                    recording_boundary=metadata,
                ),
                fail_on_channel_error=not closing,
            )
        finally:
            if capacity_exceeded:
                self._emit_event(
                    events,
                    ProductRealtimeEvent(
                        kind="SHOT_MEDIA_RECORDED",
                        shot_id=window.shot_id,
                        recorded_media=RecordedMediaV1(
                            metadata,
                            "FAILED",
                            failure_code="RECORDING_WINDOW_CAPACITY",
                        ),
                    ),
                    fail_on_channel_error=False,
                )

    def _finalize_recording_windows(self, available_frame_id, events, *, closing=False):
        while self._recording_windows:
            boundary = self._recording_windows[0]
            target = boundary.end_frame_id + self._config.post_roll_frames
            if not closing and available_frame_id < target:
                return
            self._recording_windows.popleft()
            complete = boundary.complete and available_frame_id >= target
            try:
                frozen = self._ring.freeze_window(
                    shot_id=boundary.shot_id,
                    source_epoch_id=boundary.source_epoch_id,
                    start_frame_id=boundary.start_frame_id,
                    end_frame_id=min(target, available_frame_id),
                    timeout_seconds=self._config.background_shutdown_timeout_seconds,
                )
                identity = dict(
                    session_id=self._session_id,
                    boundary=asdict(boundary),
                    media_sha256=frozen.media_sha256,
                )
                ref_id = (
                    "recording-"
                    + hashlib.sha256(
                        json.dumps(
                            identity,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                )
                destination = self._job_media_root / ref_id
                materialize_recorded_interval_reference(
                    frozen, destination, recording_identity=True
                )
                reference = destination / "interval.json"
                media = RecordedMediaV1(
                    boundary,
                    "RECORDED" if complete else "FAILED",
                    ref_id,
                    str(reference),
                    hashlib.sha256(reference.read_bytes()).hexdigest(),
                    frozen.media_sha256,
                    frozen.start_frame_id,
                    frozen.end_frame_id,
                    None if complete else "INCOMPLETE_RECORDING_WINDOW",
                )
            except Exception as exc:
                if self._recording_processing is not None:
                    # Retry from the last durable checkpoint; an I/O exception
                    # is not evidence that the recorded shot itself is invalid.
                    raise
                media = RecordedMediaV1(
                    boundary,
                    "FAILED",
                    failure_code=f"RECORDING_{type(exc).__name__}",
                )
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_MEDIA_RECORDED",
                    shot_id=boundary.shot_id,
                    frame_id=boundary.end_frame_id,
                    recorded_media=media,
                ),
                fail_on_channel_error=not closing,
            )

    def _emit_event(
        self,
        events: list[ProductRealtimeEvent],
        event: ProductRealtimeEvent,
        *,
        fail_on_channel_error: bool = True,
    ) -> None:
        channel = self._event_channel
        if getattr(self, "_recording_replaying", False) and event.kind != "CAPTURE_DEGRADED":
            return
        if self._session_policy == "recording-v1" and channel is not None:
            channel.record_evidence(event)
        events.append(event)
        if channel is None or self._event_channel_failed:
            return
        try:
            channel.publish(event)
        except RuntimeError:
            self._event_channel_failed = True
            if fail_on_channel_error:
                raise

    @contextmanager
    def _measure_stage(self, name: str, frame_id: int) -> Iterator[None]:
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self._record_stage_duration(
                name,
                frame_id,
                (time.perf_counter_ns() - started_ns) / 1_000_000,
            )

    def _record_stage_duration(
        self,
        name: str,
        frame_id: int,
        duration_ms: float,
    ) -> None:
        if duration_ms > self._maximum_stage_duration_ms.get(name, -1.0):
            self._maximum_stage_duration_ms[name] = duration_ms
            self._maximum_stage_frame_id[name] = frame_id

    def _validate_frame_mode(self, frame: CaptureFrame) -> None:
        if (frame.width, frame.height) != (
            self._capture_config.width,
            self._capture_config.height,
        ):
            raise CaptureContractError("native capture geometry changed")

    def _accept_completed_anchors(self) -> None:
        for result in self._worker.poll():
            anchor = self._to_anchor(result)
            self._replay.submit(anchor)
        self._accepted_anchors.extend(self._replay.poll_accepted())

    def _to_anchor(self, result: HeavyAnchorResult) -> AnchorObservation:
        detections = tuple(
            detection
            for detection in result.observation.detections
            if _detection_is_on_table(detection, self._calibration)
        )
        identities = result.identities
        observed_centers = tuple((item.center.x, item.center.y) for item in detections)
        previous = self._replay.current
        previous_balls = (
            ()
            if previous is None
            else tuple(ball for ball in previous.balls if ball.center_px is not None)
        )
        assignments = assign_calibrated_points(
            tuple(ball.center_px for ball in previous_balls if ball.center_px is not None),
            observed_centers,
            maximum_distance_mm=100.0,
        )
        track_ids_by_detection = {
            observed_index: previous_balls[previous_index].track_id
            for previous_index, observed_index in assignments
        }
        previous_balls_by_detection = {
            observed_index: previous_balls[previous_index]
            for previous_index, observed_index in assignments
        }
        balls: list[AnchorBall] = []
        for detection_index, detection in enumerate(detections):
            track_id = track_ids_by_detection.get(detection_index)
            if track_id is None:
                track_id = f"realtime-ball-{self._next_track_id:04d}"
                self._next_track_id += 1
            identity = identities.get(detection.detection_id, {})
            if identity:
                try:
                    group = BallGroup(str(identity.get("group", "unknown")))
                except ValueError:
                    group = BallGroup.UNKNOWN
            else:
                matched = previous_balls_by_detection.get(detection_index)
                group = BallGroup.UNKNOWN if matched is None else matched.group
            balls.append(
                AnchorBall(
                    track_id=track_id,
                    center_px=(detection.center.x, detection.center.y),
                    confidence=detection.confidence,
                    group=group,
                    bbox_xyxy_px=(
                        detection.bbox.left,
                        detection.bbox.top,
                        detection.bbox.right,
                        detection.bbox.bottom,
                    ),
                )
            )
        if not any(ball.group is BallGroup.CUE for ball in balls):
            candidate = _detect_unique_cue_ball(
                result.capture_frame.decode_bgr(),
                detections,
                self._calibration,
            )
            if candidate is not None:
                if candidate.detection_id is not None:
                    balls = [
                        replace(ball, group=BallGroup.CUE)
                        if detection.detection_id == candidate.detection_id
                        else ball
                        for ball, detection in zip(balls, detections, strict=True)
                    ]
                    candidate = None
            if candidate is not None:
                previous_cue = tuple(
                    ball
                    for ball in previous_balls
                    if ball.group is BallGroup.CUE
                    and ball.track_id not in {item.track_id for item in balls}
                )
                track_id = (
                    previous_cue[0].track_id
                    if len(previous_cue) == 1
                    else f"realtime-ball-{self._next_track_id:04d}"
                )
                if len(previous_cue) != 1:
                    self._next_track_id += 1
                balls.append(
                    AnchorBall(
                        track_id=track_id,
                        center_px=candidate.center_px,
                        confidence=candidate.confidence,
                        group=BallGroup.CUE,
                        bbox_xyxy_px=candidate.bbox_xyxy_px,
                    )
                )
        return AnchorObservation(
            capture_frame_id=result.capture_frame.capture_frame_id,
            captured_at_ns=result.capture_frame.captured_at_ns,
            balls=tuple(balls),
        )

    def _handle_tracking(self, events: list[ProductRealtimeEvent]) -> None:
        for row in self._replay.tracked_history:
            if row.capture_frame_id not in self._tracked_by_frame:
                self._observe_diagnostics(self._observation_frame(row))
            self._tracked_by_frame[row.capture_frame_id] = row
        current = self._replay.current
        if current is not None:
            self._observe_motion(current)
        if self._window_coordinator is None:
            return
        if self._event_channel is not None and not self._event_channel.can_start_shots():
            # Drain pre-match observations without assigning shot IDs or jobs.
            if self._tracked_by_frame:
                self._last_online_frame_id = max(self._tracked_by_frame)
            return
        pending_rows = sorted(
            (
                row
                for row in self._tracked_by_frame.values()
                if self._last_online_frame_id is None
                or row.capture_frame_id > self._last_online_frame_id
            ),
            key=lambda row: row.capture_frame_id,
        )
        for row in pending_rows:
            if self._event_channel is not None and not self._event_channel.admit_frame(
                row.capture_frame_id
            ):
                self._last_online_frame_id = row.capture_frame_id
                continue
            observation = self._observation_frame(row)
            for boundary in self._window_coordinator.push(observation):
                self._apply_boundary(boundary, events)
            self._last_online_frame_id = row.capture_frame_id

    def _handle_visual_boundary(
        self,
        frame: CaptureFrame,
        events: list[ProductRealtimeEvent],
    ) -> None:
        detector = self._visual_boundary_detector
        config = self._visual_motion_config
        recording_motion = getattr(self, "_recording_motion", None)
        if detector is None or config is None:
            return
        # Ordinary detection consumes saved pixels independently of live rule
        # admission. A later roster/readiness change cannot discard old media;
        # recording ownership still stays unknown without a trusted start receipt.
        if (
            self._session_policy != "recording-v1"
            and self._event_channel is not None
            and not self._event_channel.admit_frame(frame.capture_frame_id)
        ):
            self._visual_previous_bgr = None
            if recording_motion is not None:
                recording_motion.reset()
            return
        current = frame.decode_bgr()
        if recording_motion is not None:
            try:
                moving = recording_motion.push(current, timestamp_ms=frame.timestamp_ms)
            except RecordingMotionGap:
                self._apply_boundary(detector.interrupt("recording_motion_sample_gap"), events)
                recording_motion.reset()
                recording_motion.push(current, timestamp_ms=frame.timestamp_ms)
                return
        else:
            previous = self._visual_previous_bgr
            self._visual_previous_bgr = current
            if previous is None:
                return
            moving = table_visual_motion_detected(previous, current, self._calibration, config)
        for boundary in detector.push(
            frame_id=frame.capture_frame_id,
            timestamp_ms=frame.timestamp_ms,
            moving=moving,
        ):
            self._apply_boundary(boundary, events)

    def _apply_boundary(
        self,
        boundary: LiveShotBoundary,
        events: list[ProductRealtimeEvent],
    ) -> None:
        window = boundary.window
        if window is None:
            return
        if self._boundary_ledger is not None and not getattr(self, "_recording_replaying", False):
            self._boundary_ledger.record(boundary)
        if self._session_policy == "recording-v1":
            self._apply_recording_boundary(boundary, events)
            return
        if boundary.kind is LiveShotBoundaryKind.SHOT_STARTED:
            if self._pending_window is not None:
                superseded = self._pending_window
                self._emit_event(
                    events,
                    ProductRealtimeEvent(
                        kind="SHOT_INCOMPLETE",
                        shot_id=superseded.shot_id,
                        frame_id=superseded.end_frame_id,
                        reason=json.dumps(
                            {
                                "error_type": "IncompleteShotEvidence",
                                "message": ("next shot started before prior post-roll completed"),
                                "stage": "post_roll_superseded",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
                self._release_materialized_shot()
            self._ring.begin_shot(window.start_frame_id)
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_STARTED",
                    shot_id=window.shot_id,
                    frame_id=window.start_frame_id,
                    started_ms=window.start_ms,
                ),
            )
        elif boundary.kind is LiveShotBoundaryKind.SHOT_SETTLED:
            self._ring.settle_shot(window.end_frame_id)
            self._pending_window = window

    def _observe_diagnostics(self, current: ObservationFrame) -> None:
        confirmed = current.observations
        confirmed_count = len(confirmed)
        self._diagnostic_tracked_frames += 1
        self._diagnostic_maximum_balls = max(
            self._diagnostic_maximum_balls,
            confirmed_count,
        )
        if confirmed_count:
            self._diagnostic_frames_with_balls += 1
        else:
            self._diagnostic_frames_without_balls += 1
        if any(ball.group is BallGroup.CUE for ball in confirmed):
            self._diagnostic_frames_with_cue += 1

        previous = self._diagnostic_previous
        self._diagnostic_previous = current
        if previous is None:
            return
        self._diagnostic_maximum_frame_gap = max(
            self._diagnostic_maximum_frame_gap,
            current.frame_id - previous.frame_id,
        )
        if len(previous.observations) != confirmed_count:
            self._diagnostic_count_changes += 1
            self._diagnostic_threshold_crossings += 1
            return
        change_mm = _observation_set_change_mm(previous, current)
        if change_mm > self._diagnostic_maximum_change_mm:
            self._diagnostic_maximum_change_mm = change_mm
            self._diagnostic_maximum_change_frame_id = current.frame_id
        threshold = (
            None if self._shot_config is None else self._shot_config.window.motion_threshold_mm
        )
        if threshold is not None and change_mm >= threshold:
            self._diagnostic_threshold_crossings += 1

        previous_cue = tuple(ball for ball in previous.observations if ball.group is BallGroup.CUE)
        current_cue = tuple(ball for ball in confirmed if ball.group is BallGroup.CUE)
        if len(previous_cue) == len(current_cue) == 1:
            cue_change_mm = math.hypot(
                previous_cue[0].x_mm - current_cue[0].x_mm,
                previous_cue[0].y_mm - current_cue[0].y_mm,
            )
            if cue_change_mm > self._diagnostic_maximum_cue_change_mm:
                self._diagnostic_maximum_cue_change_mm = cue_change_mm
                self._diagnostic_maximum_cue_change_frame_id = current.frame_id

    def _observe_motion(self, current: TrackedObservation) -> None:
        previous = self._last_motion_row
        self._last_motion_row = current
        if previous is None or previous.capture_frame_id >= current.capture_frame_id:
            return
        before = {
            ball.track_id: ball.table_mm
            for ball in previous.balls
            if ball.state == "confirmed" and ball.table_mm is not None
        }
        displacement = max(
            (
                math.dist(before[ball.track_id], ball.table_mm)
                for ball in current.balls
                if ball.state == "confirmed"
                and ball.table_mm is not None
                and ball.track_id in before
            ),
            default=0.0,
        )
        if displacement >= 5.0:
            self._active_until_frame_id = current.capture_frame_id + 120

    @staticmethod
    def _observation_frame(row: TrackedObservation) -> ObservationFrame:
        from billiards_ai.lab.shot_local.models import CalibratedBallObservation

        return ObservationFrame(
            frame_id=row.capture_frame_id,
            timestamp_ms=round(row.captured_at_ns / 1_000_000),
            observations=tuple(
                CalibratedBallObservation(
                    detection_id=f"realtime-{row.capture_frame_id:08d}-{ball.track_id}",
                    x_mm=ball.table_mm[0],
                    y_mm=ball.table_mm[1],
                    confidence=ball.confidence,
                    group=ball.group,
                    group_confidence=ball.confidence,
                )
                for ball in row.balls
                if ball.state == "confirmed" and ball.table_mm is not None
            ),
        )

    def _record_first_shot_diagnostics(
        self,
        *,
        frozen: FrozenCaptureInterval | RecordedShotInterval,
        tracked: tuple[TrackedObservation, ...],
        anchors: tuple[AnchorObservation, ...],
        settled: LiveSettledShot,
    ) -> None:
        if getattr(self, "_first_shot_diagnostics", None) is not None:
            return
        try:
            coverage = settled.rule_coverage
            diagnostic = diagnose_realtime_shot(
                shot_id=settled.shot_id,
                interval_start=frozen.start_frame_id,
                interval_end=frozen.end_frame_id,
                declared_source_count=frozen.frame_count,
                shot_start=settled.start_frame_id,
                shot_end=settled.settled_frame_id,
                source_ids=(
                    None
                    if coverage is None or settled.shot_abstention_reason == "ring_incomplete"
                    else coverage.source_frame_ids
                ),
                tracked=tracked,
                rule_frames=settled.rule_frames,
                anchors=anchors,
                publishable=settled.publishable,
                abstention_reason=settled.shot_abstention_reason,
                tracker_snapshot=getattr(self._replay, "tracker_diagnostic_snapshot", None),
                tracker_config=getattr(self, "_tracker_diagnostic_config", None),
            )
        except Exception as exc:
            # A diagnostic failure cannot suppress or modify the settled event/job.
            diagnostic = RealtimeShotDiagnostics(
                status="UNAVAILABLE",
                shot_id=settled.shot_id,
                error_type=type(exc).__name__[:128],
            )
        self._first_shot_diagnostics = diagnostic

    def _finalize_if_ready(
        self,
        frame: CaptureFrame,
        events: list[ProductRealtimeEvent],
        stop: threading.Event,
    ) -> None:
        del stop
        window = self._pending_window
        if (
            window is None
            or frame.capture_frame_id < window.end_frame_id + self._config.post_roll_frames
        ):
            return
        if not self._replay.wait_until_preprocessed(
            frame.capture_frame_id,
            timeout_seconds=self._config.background_shutdown_timeout_seconds,
        ):
            raise CaptureContractError("working-frame preprocessing did not cover settled shot")
        self._accept_completed_anchors()
        self._handle_tracking(events)
        self._retain_unsafe_interval = True
        try:
            frozen = self._ring.freeze_shot()
        except BaseException as exc:
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_INCOMPLETE",
                    shot_id=window.shot_id,
                    frame_id=window.end_frame_id,
                    reason=_structured_failure_reason("freeze", exc),
                ),
            )
            self._release_materialized_shot()
            return
        assert self._job_store is not None
        assert self._job_media_root is not None
        assert self._quarantine_root is not None
        job = self._analysis_job(window, frozen)
        tracked = tuple(
            self._tracked_by_frame[frame_id]
            for frame_id in range(frozen.start_frame_id, frozen.end_frame_id + 1)
            if frame_id in self._tracked_by_frame
        )
        anchors = tuple(
            anchor
            for anchor in self._accepted_anchors
            if frozen.start_frame_id <= anchor.capture_frame_id <= frozen.end_frame_id
        )
        try:
            refined = RealtimeShotRefiner(
                maximum_shot_frames=self._config.maximum_shot_frames
            ).refine(interval=frozen, tracked=tracked, anchors=anchors)
            settled = LiveSettledShot(
                shot_id=window.shot_id,
                start_frame_id=window.start_frame_id,
                settled_frame_id=window.end_frame_id,
                settled_at_ns=window.end_ms * 1_000_000,
                observations=refined.observations(),
                publishable=refined.publishable,
                shot_abstention_reason=refined.shot_abstention_reason,
                fact_coverage=dict(refined.fact_coverage),
                source_epoch_id=frozen.source_epoch_id,
                job_id=job.job_id,
                rule_frames=tuple(getattr(refined, "rule_frames", lambda: ())()),
                rule_coverage=getattr(refined, "coverage", None),
                event_config=self._shot_config.event,
                event_config_digest=_event_config_digest(self._shot_config.event),
            )
        except Exception as exc:
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_INCOMPLETE",
                    shot_id=window.shot_id,
                    frame_id=window.end_frame_id,
                    reason=_structured_failure_reason("live_settle", exc),
                ),
            )
            self._release_materialized_shot()
            return
        self._record_first_shot_diagnostics(
            frozen=frozen,
            tracked=tracked,
            anchors=anchors,
            settled=settled,
        )
        self._emit_event(
            events,
            ProductRealtimeEvent(
                kind="SHOT_SETTLED",
                shot_id=window.shot_id,
                frame_id=window.end_frame_id,
                settled=settled,
            ),
        )
        try:
            if isinstance(frozen, RecordedShotInterval):
                materialize_recorded_interval_reference(frozen, job.media_root)
            else:
                materialize_frozen_interval(frozen, job.media_root)
        except Exception as materialization_error:
            reason = _structured_failure_reason("materialization", materialization_error)
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_INCOMPLETE",
                    shot_id=window.shot_id,
                    frame_id=window.end_frame_id,
                    reason=reason,
                ),
            )
            self._discard_incomplete_shot(job.media_root)
            return
        self._retain_unsafe_interval = False
        try:
            self._job_store.enqueue(job)
        except Exception as exc:
            self._emit_event(
                events,
                ProductRealtimeEvent(
                    kind="SHOT_INCOMPLETE",
                    shot_id=window.shot_id,
                    frame_id=window.end_frame_id,
                    reason=_structured_failure_reason("enqueue", exc),
                ),
            )
            self._discard_incomplete_shot(job.media_root)
            return
        self._emit_event(
            events,
            ProductRealtimeEvent(
                kind="SHOT_ENQUEUED",
                shot_id=window.shot_id,
                frame_id=window.end_frame_id,
                job=job,
            ),
        )
        self._release_materialized_shot()

    def _analysis_job(
        self,
        window: ShotWindow,
        frozen: FrozenCaptureInterval | RecordedShotInterval,
    ) -> ShotAnalysisJob:
        assert self._session_id is not None
        assert self._capture_setup_id is not None
        assert self._calibration_artifact_id is not None
        assert self._config_sha256 is not None
        assert self._job_media_root is not None
        identity = {
            "session_id": self._session_id,
            "shot_id": window.shot_id,
            "source_epoch_id": frozen.source_epoch_id,
            "start_frame_id": frozen.start_frame_id,
            "end_frame_id": frozen.end_frame_id,
            "media_sha256": frozen.media_sha256,
            "capture_setup_id": self._capture_setup_id,
            "calibration_artifact_id": self._calibration_artifact_id,
            "config_sha256": self._config_sha256,
            "model_package_id": self._detector.package_identity.package_id,
        }
        job_id = (
            "job-"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        return ShotAnalysisJob(
            job_id=job_id,
            session_id=self._session_id,
            shot_id=window.shot_id,
            source_epoch_id=frozen.source_epoch_id,
            start_frame_id=frozen.start_frame_id,
            end_frame_id=frozen.end_frame_id,
            media_root=self._job_media_root / job_id,
            media_sha256=frozen.media_sha256,
            capture_setup_id=self._capture_setup_id,
            calibration_artifact_id=self._calibration_artifact_id,
            config_sha256=self._config_sha256,
            model_package_id=self._detector.package_identity.package_id,
            identity_training_package_id=self._identity_training_package_id,
            identity_training_package_sha256=self._identity_training_package_sha256,
        )

    def _release_materialized_shot(self) -> None:
        self._ring.release_shot()
        self._pending_window = None
        self._retain_unsafe_interval = False

    def _discard_incomplete_shot(self, media_root: Path) -> None:
        assert self._job_media_root is not None
        expected_parent = self._job_media_root.resolve()
        target = media_root.resolve()
        if target.parent != expected_parent:
            self._release_materialized_shot()
            raise RuntimeError("incomplete shot media escaped the job media root")
        try:
            if target.exists():
                shutil.rmtree(target)
        finally:
            self._release_materialized_shot()

    def _cancel_source(self) -> None:
        if self._capture_cancelled:
            return
        cancel_started = time.monotonic()
        self._source.cancel()
        cancel_elapsed = time.monotonic() - cancel_started
        if cancel_elapsed > self._config.capture_cancel_timeout_seconds:
            raise CaptureContractError("capture cancellation exceeded 500 ms product boundary")
        self._capture_cancelled = True

    def _interrupt_active(
        self,
        events: list[ProductRealtimeEvent],
        reason: str,
    ) -> None:
        coordinator = self._visual_boundary_detector or self._window_coordinator
        if coordinator is None or self._pending_window is not None:
            return
        interrupted = coordinator.interrupt(reason)
        if interrupted.window is None:
            return
        window = interrupted.window
        self._emit_event(
            events,
            ProductRealtimeEvent(
                kind="SHOT_INCOMPLETE",
                shot_id=window.shot_id,
                frame_id=window.end_frame_id,
                reason=reason,
            ),
            fail_on_channel_error=False,
        )
        self._retain_unsafe_interval = True


def _structured_failure_reason(stage: str, error: BaseException) -> str:
    return json.dumps(
        {
            "error_type": type(error).__name__,
            "message": str(error),
            "stage": stage,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _event_config_digest(config: ShotLocalEventConfig) -> str:
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
