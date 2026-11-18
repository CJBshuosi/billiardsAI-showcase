"""Dependency-light LAN server for synchronized field-assistance replay."""

from __future__ import annotations

import hashlib
import hmac
import io
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import sqlite3
import stat
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.control import (
    AssignGroups,
    BallGroupAssignment,
    ContinueTurn,
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    SetBreaker,
    SetParticipant,
    SwitchTurn,
    WristbandRole,
    apply_control,
)
from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_fact_report_payload,
    field_sync_envelope,
)
from billiards_ai.edge.route_assistance.local_media_lock import (
    _assert_directory_chain,
    session_media_lock,
)
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionCloseState,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import (
    PostgameClip,
    PostgameJobSnapshot,
    PostgameSessionManifest,
    project_session_analysis,
    validate_save_request_id,
)
from billiards_ai.edge.route_assistance.recording import (
    RecommendationPower,
    ShotBoundary,
    ShotBoundaryKind,
    ShotTimelineIndex,
    build_media_range,
    load_recommendation_power,
)
from billiards_ai.edge.route_assistance.recording_capture import (
    public_recording_capture,
    public_recording_end,
)
from billiards_ai.edge.route_assistance.records import (
    FieldSessionRecord,
    FieldSessionStatus,
    FieldShotRecord,
    SessionLimitCode,
)
from billiards_ai.edge.route_assistance.runtime_resources import managed_live_artifact_id
from billiards_ai.edge.route_assistance.sync import SyncMediaItem
from billiards_ai.lab.field_assistance.cloud_gateway import (
    FieldCloudGateway,
    FieldJoinCoordinator,
)
from billiards_ai.lab.field_assistance.http_transport import (
    _RANGE_PATTERN as _RANGE_PATTERN,
)
from billiards_ai.lab.field_assistance.http_transport import (
    FieldHTTPResponseMixin,
    _ClosingRequestReader,
)
from billiards_ai.lab.field_assistance.live_source import (
    LiveProjectionSource,
    ReplayProjectionSource,
    USBLiveProjectionSource,
    confirmation_reference,
    session_has_live_activity,
)
from billiards_ai.lab.field_assistance.live_state import (
    ReplayClock,
    ReplayTimeline,
    load_replay_timeline,
)
from billiards_ai.lab.field_assistance.mainline1_replay import (
    Mainline1ReplayInputs,
    load_mainline1_replay_inputs,
)
from billiards_ai.lab.field_assistance.managed_lifecycle import (
    poll_managed_close,
    resolve_command_confirmation,
)
from billiards_ai.lab.field_assistance.managed_startup import ManagedStartupCoordinator
from billiards_ai.lab.field_assistance.pages import render_screen_page, render_tablet_page
from billiards_ai.lab.field_assistance.recording_recovery import RecordingRecovery
from billiards_ai.lab.field_assistance.replay import verify_field_route_replay_artifact
from billiards_ai.lab.field_assistance.runtime_resources import verify_managed_runtime_resources
from billiards_ai.lab.field_assistance.scenario_playback import (
    AuthenticatedScenarioPlayback,
    PlaybackGate,
)
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    project_recording_playback,
    publish_selected_original_clips,
)
from billiards_ai.lab.mainline1_recording_processing import RecordingProcessingJournal
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import (
    Mainline1RecordingMediaWorker,
    MediaEncodingResourceGate,
)
from billiards_ai.shared.models.miniprogram import canonical_sync_hash

_MAX_CONTROL_BYTES = 32 * 1024
_FRAME_PATTERN = re.compile(r"/media/frame/(\d+)\.jpg\Z")
_POSTGAME_MEDIA_PATTERN = re.compile(r"/media/postgame/(clip-[a-f0-9]{64})\.mp4\Z")
_CLOUD_SESSION_RETRY_SECONDS = 10
_JOIN_REQUEST_REFRESH_SECONDS = 5
_SESSION_ENDED_RECEIPT_SECONDS = 3.0
_CAPTURE_RETRY_INITIAL_SECONDS = 0.2
_CAPTURE_RETRY_MAX_SECONDS = 1.0


def _portable_review_path_snapshot(root: Path, parts: tuple[str, ...]):
    """Reject links/reparse points and retain each component's filesystem identity."""
    snapshots = []
    current = root
    for index, part in enumerate((None, *parts)):
        if part is not None:
            current = current / part
        info = current.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(current, "is_junction", lambda: False)()
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not info.st_ino
        ):
            raise ValueError("linked or ambiguous review media path")
        expected_type = stat.S_ISREG if index == len(parts) else stat.S_ISDIR
        if not expected_type(info.st_mode):
            raise ValueError("invalid review media component")
        if current.resolve(strict=True) != current:
            raise ValueError("review media containment changed")
        snapshots.append((info.st_dev, info.st_ino, info.st_mode))
    return tuple(snapshots)


@contextmanager
def _open_review_media(artifact_path: Path, clip: PostgameClip, *, continue_check=lambda: None):
    """Open beneath the trusted artifact root without following any child symlink.

    The verified descriptor is retained for streaming, so pathname replacement
    cannot switch the file after verification. Reads and hashing are chunked.
    """
    if (
        clip.local_path is None
        or clip.content_type != "video/mp4"
        or clip.byte_size is None
        or clip.byte_size <= 0
        or not clip.session_id
        or clip.session_id in {".", ".."}
        or "\\" in clip.session_id
        or Path(clip.session_id).name != clip.session_id
    ):
        raise ValueError("invalid review media")
    root = artifact_path.absolute()
    relative = clip.local_path.relative_to(root / "postgame" / clip.session_id)
    if not relative.parts or any(part in {"..", "."} for part in relative.parts):
        raise ValueError("invalid review media path")
    parts = ("postgame", clip.session_id, *relative.parts)
    descriptor_walk = (
        all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
        and os.open in os.supports_dir_fd
    )
    directory_fd = None
    media_fd = None
    try:
        if descriptor_walk:
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for part in parts[:-1]:
                child_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
                )
                os.close(directory_fd)
                directory_fd = child_fd
            media_fd = os.open(
                relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
        else:
            snapshots = _portable_review_path_snapshot(root, parts)
            expected = clip.local_path.lstat()
            media_fd = os.open(
                clip.local_path,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0),
            )
        before = os.fstat(media_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != clip.byte_size:
            raise ValueError("invalid review media size")
        if not descriptor_walk:
            if (
                (before.st_dev, before.st_ino, before.st_mode) != snapshots[-1]
                or (before.st_dev, before.st_ino, before.st_size)
                != (expected.st_dev, expected.st_ino, expected.st_size)
                or snapshots != _portable_review_path_snapshot(root, parts)
            ):
                raise ValueError("review media identity changed")
        with os.fdopen(media_fd, "rb") as source:
            media_fd = None
            digest = hashlib.sha256()
            remaining = clip.byte_size
            while remaining:
                continue_check()
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("truncated review media")
                digest.update(chunk)
                remaining -= len(chunk)
            continue_check()
            after = os.fstat(source.fileno())
            if digest.hexdigest() != clip.content_sha256 or (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("review media changed")
            if not descriptor_walk and snapshots != _portable_review_path_snapshot(root, parts):
                raise ValueError("review media containment changed")
            source.seek(0)
            yield source
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if media_fd is not None:
            os.close(media_fd)


def _role_label(role: str | WristbandRole | None) -> str:
    return "红方" if role == WristbandRole.RED or role == "red" else "蓝方"


def _group_label(group: object) -> str:
    return "花色" if group == BallGroupAssignment.STRIPE or group == "stripe" else "全色"


def _describe_applied_command(
    envelope: FieldCommandEnvelope,
    application: FieldCommandApplication,
    match: MatchControlSnapshot,
) -> str | None:
    """Describe the latest applied phone command for the public field display."""

    if application.outcome != "applied":
        return None
    command_type = envelope.command_type
    if command_type == "claim_role":
        return f"{_role_label(envelope.payload.get('role'))}已加入本场"
    if command_type == "release_role":
        return "腕带已释放"
    if command_type == "set_breaker":
        return f"{_role_label(envelope.payload.get('role'))}开球已设置"
    if command_type == "assign_groups":
        red_group = envelope.payload.get("red_group")
        blue_group = "solid" if red_group == "stripe" else "stripe"
        return f"球组已分配：红方{_group_label(red_group)}，蓝方{_group_label(blue_group)}"
    if command_type == "continue_turn":
        return f"{_role_label(match.active_role)}继续击球"
    if command_type == "switch_turn":
        return f"已切换至{_role_label(match.active_role)}击球"
    if command_type == "end_session":
        return "本场已结束"
    return None


class _BackgroundFieldCommandWorker:
    """Apply cloud controls exactly once locally before acknowledging them."""

    def __init__(
        self,
        *,
        gateway: FieldCloudGateway,
        store: SQLiteFieldRecordStore,
        match_lock: threading.RLock,
        before_application: Callable[[FieldCommandEnvelope], None] | None = None,
        confirmation_resolver: Callable | None = None,
        command_reducer: Callable = reduce_field_command,
        match_changed: Callable[[str, MatchControlSnapshot], None] | None = None,
        application_changed: Callable[[FieldCommandEnvelope, FieldCommandApplication], None]
        | None = None,
        health_changed: Callable[[str], None] | None = None,
        maintenance: Callable[[], None] | None = None,
        wake: threading.Event | None = None,
        # Five attested operations (open, refresh, claim, stale close, complete),
        # each with a 20s assertion and 20s request, plus local teardown margin.
        # Extra pending maintenance may exceed this budget: stop then fails closed.
        join_timeout_seconds: float = 5 * 2 * 20.0 + 5.0,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._match_lock = match_lock
        self._before_application = before_application or (lambda _envelope: None)
        self._confirmation_resolver = confirmation_resolver
        self._command_reducer = command_reducer
        self._match_changed = match_changed or (lambda _session_id, _match: None)
        self._application_changed = application_changed or (lambda _envelope, _application: None)
        self._health_changed = health_changed or (lambda _status: None)
        self._maintenance = maintenance or (lambda: None)
        self._wake = wake or threading.Event()
        self._stopped = threading.Event()
        self._join_timeout_seconds = join_timeout_seconds
        self._offline = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="field-commands")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._join_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("field command worker did not terminate")

    def claim_and_apply_once(self, *, raise_errors: bool = False) -> bool:
        if self._stopped.is_set():
            return False
        try:
            envelope = self._gateway.claim_field_command()
            if self._stopped.is_set():
                return False
            if envelope is None:
                self._offline = False
                self._health_changed("connected")
                return False
            missing_ended_session = False
            read_only_replay = False
            with self._match_lock:
                if self._stopped.is_set():
                    return False
                try:
                    self._before_application(envelope)
                except ValueError as exc:
                    if str(exc) != "LIVE_AUTHORITY_UNAVAILABLE":
                        raise
                    reference = (
                        None
                        if self._confirmation_resolver is None
                        else self._confirmation_resolver(envelope)
                    )
                    application = self._store.get_field_command_application(
                        envelope,
                        confirmation_ref=reference,
                    )
                    if application is None:
                        raise
                    # An exact durable receipt has no remaining local mutation.
                    # Do not rerun match, playback, projection or health callbacks.
                    read_only_replay = True
                if not read_only_replay:
                    if self._stopped.is_set():
                        return False
                    reference = (
                        None
                        if self._confirmation_resolver is None
                        else self._confirmation_resolver(envelope)
                    )
                    if self._stopped.is_set():
                        return False
                    try:
                        application = self._store.apply_field_command(
                            envelope, self._command_reducer, confirmation_ref=reference
                        )
                    except ValueError as exc:
                        missing_ended_session = (
                            not envelope.session_id.startswith("managed-")
                            and envelope.command_type == "end_session"
                            and str(exc) == "field session does not exist"
                        )
                        if not missing_ended_session:
                            raise
                    else:
                        if envelope.session_id.startswith(
                            "managed-"
                        ) and application.result_code in {
                            "COMMAND_OUT_OF_ORDER",
                            "COMMAND_IDEMPOTENCY_CONFLICT",
                        }:
                            raise OSError("managed command requires ordered recovery")
                        if self._stopped.is_set():
                            return False
                        match = self._store.load_session(envelope.session_id).match
                        if self._stopped.is_set():
                            return False
                        self._match_changed(envelope.session_id, match)
                        if self._stopped.is_set():
                            return False
                        self._application_changed(envelope, application)
            if self._stopped.is_set():
                return False
            if missing_ended_session:
                lease = self._gateway.close_field_session(envelope.session_id)
                if lease.session_id != envelope.session_id or lease.status != "ended":
                    raise OSError("cloud gateway returned invalid stale-session close")
                application = FieldCommandApplication(
                    envelope.command_id,
                    "applied",
                    "STALE_SESSION_CLOSED",
                    envelope.expected_revision,
                )
            if self._stopped.is_set():
                return False
            self._gateway.complete_field_command(application)
        except (OSError, ValueError):
            if self._stopped.is_set():
                return False
            self._offline = True
            self._health_changed("offline")
            if raise_errors:
                raise
            return False
        if self._stopped.is_set():
            return False
        self._offline = False
        if not read_only_replay:
            self._health_changed("connected")
        return True

    def _run(self) -> None:
        retry_seconds = 0.25
        while not self._stopped.is_set():
            try:
                self._maintenance()
            except (OSError, ValueError):
                self._offline = True
                if not self._stopped.is_set():
                    self._health_changed("offline")
            if self._stopped.is_set():
                break
            applied = self.claim_and_apply_once()
            if applied:
                retry_seconds = 0.25
                continue
            wait_seconds = retry_seconds if self._offline else 0.25
            self._wake.wait(timeout=wait_seconds)
            self._wake.clear()
            retry_seconds = min(30.0, retry_seconds * 2) if self._offline else 0.25


@dataclass(frozen=True, slots=True)
class _PublicImageAsset:
    image_bytes: bytes
    content_type: str


class _FieldMediaResourceGate(MediaEncodingResourceGate):
    """Prioritize writer capacity without banning media for an entire live match."""

    def __init__(self, state) -> None:
        self._state = state

    @property
    def cooperative_pause(self):
        return getattr(self._state, "session_policy", "legacy") == "recording-v1"

    def cancellation_requested(self):
        state = self._state
        return (
            state._media_stopped.is_set()
            or state.projection_lifecycle_closing
            or state.projection_lifecycle_closed
        )

    def may_run_media_encoding(self, session_id: str) -> bool:
        state = self._state
        try:
            with state.match_lock:
                if state._media_stopped.is_set():
                    return False
                if state.projection_lifecycle_closing or state.projection_lifecycle_closed:
                    return False
                if getattr(state, "session_policy", "legacy") == "recording-v1":
                    source = state.projection_source
                    # Copy the reference only; do not project or read SQLite under this lock.
                else:
                    source = None
                if source is None and state.session_id == session_id:
                    return state.record_store.ended_session_allows_media_encoding(session_id)
                if source is None:
                    target = state.record_store.load_session(session_id)
                    if target.status is not FieldSessionStatus.ENDED:
                        return False
                    current = state.record_store.load_session(state.session_id)
                    if current.status is not FieldSessionStatus.OPEN:
                        return False
                    if state.record_store.get_session_policy(current.session_id) != "recording-v1":
                        return False
                    if session_has_live_activity(state.record_store, current.session_id):
                        return False
                    if not getattr(state.projection_source, "media_encoding_idle", lambda: False)():
                        return False
                    projection = state.projection_source.current()
                    frame = projection.frame.payload
                    return bool(
                        frame.get("session_id") == current.session_id
                        and frame.get("source_kind") == "usb_live"
                        and frame.get("phase") == "stable"
                        and frame.get("rule_status") != "DEGRADED"
                        and frame.get("guidance_visible") is False
                    )
            return source.recording_media_capacity_available()
        except (OSError, RuntimeError, ValueError, AttributeError, sqlite3.Error):
            return False


def _load_join_qr_asset(path: Path) -> _PublicImageAsset:
    if path.is_symlink() or not path.is_file():
        raise ValueError("permanent table code image is missing or unsafe")
    size = path.stat().st_size
    if size <= 0 or size > 2 * 1024 * 1024:
        raise ValueError("permanent table code image size is invalid")
    payload = path.read_bytes()
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        content_type = "image/png"
    elif payload.startswith(b"\xff\xd8\xff"):
        content_type = "image/jpeg"
    else:
        raise ValueError("permanent table code image format is unsupported")
    return _PublicImageAsset(payload, content_type)


def probe_video_duration_ms(path: Path) -> int:
    """Read a positive media duration without trusting container metadata alone."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("field replay presentation cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError("field replay presentation duration is invalid")
    return max(1, round(frame_count / fps * 1000))


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_control(payload: object):
    if not isinstance(payload, dict) or not isinstance(payload.get("command"), str):
        raise ValueError("control command is invalid")
    command = payload["command"]
    try:
        if command == "set_participant":
            return SetParticipant(
                WristbandRole(payload["role"]),
                str(payload["user_id"]),
            )
        if command == "set_breaker":
            return SetBreaker(WristbandRole(payload["role"]))
        if command == "assign_groups":
            return AssignGroups(BallGroupAssignment(payload["red_group"]))
        if command == "continue_turn":
            return ContinueTurn()
        if command == "switch_turn":
            return SwitchTurn()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("control command arguments are invalid") from exc
    raise ValueError("control command is unsupported")


@dataclass(slots=True)
class _ServerState:
    artifact_path: Path
    control_token: str
    replay_clock: ReplayClock
    match: MatchControlSnapshot
    match_lock: threading.RLock
    frame_store: _ReplayFrameStore
    record_store: SQLiteFieldRecordStore
    session_id: str
    artifact_id: str
    shot_index: ShotTimelineIndex
    recommendation_power: dict[str, RecommendationPower]
    duration_ms: int
    join_qr_asset: _PublicImageAsset | None = None
    join_coordinator: FieldJoinCoordinator | None = None
    monotonic: Callable[[], float] = time.monotonic
    managed_session_scope: ManagedSessionScope | None = None
    # Only scenes materialized in this invocation have a trusted current model/jobs context.
    resource_session_ids: tuple[str, ...] | None = None
    recovery_diagnostics: dict[str, str] = field(default_factory=dict)
    cloud_session_status: str = "unavailable"
    cloud_session_next_attempt: float = 0
    join_requests_next_refresh: float = 0
    join_error: str | None = None
    pending_cloud_closes: dict[str, float] = field(default_factory=dict)
    cloud_close_error: str | None = None
    cloud_control_status: str = "unavailable"
    sync_wakeup: Callable[[], None] | None = None
    record_cursor_ms: int = -1
    last_applied_action: str | None = None
    session_ended_at_monotonic: float | None = None
    sync_error: str | None = None
    source_mainline1_artifact_id: str | None = None
    verified_start_events: tuple[tuple[int, str], ...] = ()
    source_validation_reason: str = "SOURCE_INPUTS_MISSING"
    ownership_recovery_ambiguous: bool = False
    replay_has_started: bool = False
    projection_source: LiveProjectionSource | None = None
    media_job_store: SQLiteShotAnalysisJobStore | None = None
    _media_thread: threading.Thread | None = field(default=None, init=False)
    _media_thread_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _media_stopped: threading.Event = field(default_factory=threading.Event, init=False)
    cloud_io_lock: threading.Lock = field(default_factory=threading.Lock)
    projection_lifecycle_closed: bool = field(default=False, init=False)
    projection_lifecycle_closing: bool = field(default=False, init=False)
    _projection_close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _pending_rollover_session: FieldSessionRecord | None = field(default=None, init=False)
    _pending_projection_cleanup: LiveProjectionSource | None = field(default=None, init=False)
    _background_recordings: list[LiveProjectionSource] = field(default_factory=list, init=False)
    _recording_recoveries: list[RecordingRecovery] = field(default_factory=list, init=False)
    review_session_id: str | None = field(default=None, init=False)
    _review_scope: tuple[str, str] | None = field(default=None, init=False)
    _kiosk_nonce: str = field(default="", init=False, repr=False)
    _review_recovery_required: bool = field(default=False, init=False)
    _review_live_active: bool = field(default=False, init=False)
    _media_candidate_cursor: int = field(default=0, init=False)
    postgame_now_utc: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.projection_source is None:
            self.projection_source = ReplayProjectionSource(self.replay_clock, self.frame_store)
        # No persisted replay-clock history exists. Never attach pre-restart,
        # unprocessed starts to a roster recovered from a later match revision.
        self.ownership_recovery_ambiguous = bool(
            self.match.revision or self.record_store.list_shots(self.session_id)
        )

    def maintain_service(self) -> None:
        """Run local service lifecycle work from ``serve_forever``."""
        self._maintain_recording_media()

    def _maintain_recording_media(self) -> None:
        if self.media_job_store is None:
            return
        with self._media_thread_lock:
            if self._media_stopped.is_set():
                return
            if self._media_thread is not None and self._media_thread.is_alive():
                return
            self._media_thread = threading.Thread(
                target=self._run_recording_media,
                name="field-recording-media",
                daemon=True,
            )
            self._media_thread.start()

    def close_media_worker(self) -> None:
        with self._media_thread_lock:
            self._media_stopped.set()
            thread = self._media_thread
        if thread is not None:
            thread.join(timeout=6)
            if thread.is_alive():
                raise TimeoutError("recording media worker did not terminate")

    def _media_candidate_sessions(self):
        """Private encoding work is independent of the user-visible review gate."""
        with self.match_lock:
            current = (
                self.session_id
                if isinstance(self.projection_source, USBLiveProjectionSource)
                and getattr(self, "session_policy", "legacy") == "recording-v1"
                else None
            )
            scope, artifact = self.managed_session_scope, self.artifact_id
        review = (
            self.record_store.active_postgame_review(artifact)
            if scope is None
            else self.record_store.active_postgame_review_for_scope(scope)
        )
        candidates = (
            [current]
            if current is not None
            and (scope is not None or not self.record_store.has_managed_session_identity(current))
            else []
        )
        if (
            review is not None
            and review.session_id not in candidates
            and (
                scope is not None
                or not self.record_store.has_managed_session_identity(review.session_id)
            )
        ):
            candidates.append(review.session_id)
        if candidates:
            cursor = getattr(self, "_media_candidate_cursor", 0) % len(candidates)
            self._media_candidate_cursor = cursor + 1
            candidates = candidates[cursor:] + candidates[:cursor]
        return tuple(candidates)

    def _media_jobs_for_session(self, session_id):
        jobs = self.media_job_store
        scope = self.managed_session_scope
        if jobs is None or scope is None or session_id == self.session_id:
            return jobs
        # Historical managed work must keep its original queue, never the
        # currently running scene's invocation roots. Resolution validates the
        # retained scope, manifest, epochs and source lineage before opening it.
        from billiards_ai.lab.field_assistance.managed_resource_recovery import (
            resolve_managed_session_resources,
            resolve_ordinary_recording_jobs,
        )

        if self.record_store.get_session_policy(session_id) == "recording-v1":
            try:
                original_jobs = resolve_ordinary_recording_jobs(
                    self.record_store, scope, session_id
                )
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                self.recovery_diagnostics[f"media:{session_id}"] = (
                    "POSTGAME_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
                )
                return None
            self.recovery_diagnostics.pop(f"media:{session_id}", None)
            return original_jobs
        original = resolve_managed_session_resources(
            self.record_store, scope, session_id, purpose="media"
        )
        return SQLiteShotAnalysisJobStore.open_existing(original.jobs_db)

    def _run_recording_media(self) -> None:
        try:
            candidates = self._media_candidate_sessions()
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            return
        for session_id in candidates:
            try:
                original_root = self.record_store.artifact_path_for_session(session_id)
                jobs = self._media_jobs_for_session(session_id)
                if jobs is None:
                    continue
                worker = Mainline1RecordingMediaWorker(
                    records=self.record_store,
                    jobs=jobs,
                    artifact_root=original_root / "postgame" / session_id,
                    resource_gate=_FieldMediaResourceGate(self),
                )
                result = worker.process_one(
                    f"field-media-{os.getpid()}", session_id, datetime.now(UTC)
                )
                self.recovery_diagnostics.pop(f"media:{session_id}", None)
                if result is not None:
                    return
            except (OSError, RuntimeError, ValueError) as exc:
                self.recovery_diagnostics[f"media:{session_id}"] = (
                    f"MEDIA_WORKER_RETRYING:{type(exc).__name__}"
                )

    def _active_review(self) -> PostgameSessionManifest | None:
        """Call under match_lock. The durable queue, never current match, is authority."""
        self._review_recovery_required = False
        self._review_live_active = False
        if isinstance(self.projection_source, USBLiveProjectionSource):
            try:
                if session_has_live_activity(self.record_store, self.session_id):
                    self._review_live_active = True
                    return None
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                return None
        if self.managed_session_scope is None:
            manifest = self.record_store.active_postgame_review(self.artifact_id)
            if manifest is not None and self.record_store.has_managed_session_identity(
                manifest.session_id
            ):
                manifest = None
                self._review_recovery_required = True
                self.recovery_diagnostics["review"] = "MANAGED_REVIEW_SCOPE_UNAVAILABLE"
        else:
            try:
                manifest = self.record_store.active_postgame_review_for_scope(
                    self.managed_session_scope
                )
                if manifest is None:
                    pending = self.record_store.pending_postgame_review_session_ids_for_scope(
                        self.managed_session_scope
                    )
                    self._review_recovery_required = bool(pending)
                    for session_id in pending:
                        self.recovery_diagnostics[session_id] = (
                            "POSTGAME_REVIEW_QUEUE_RECOVERY_REQUIRED"
                        )
                else:
                    original_root = self.record_store.artifact_path_for_session(manifest.session_id)
                    if not original_root.is_absolute():
                        raise ValueError("managed review original root is not absolute")
                    _assert_directory_chain(original_root)
                    if self.recovery_diagnostics.get(manifest.session_id) == (
                        "POSTGAME_REVIEW_QUEUE_RECOVERY_REQUIRED"
                    ):
                        self.recovery_diagnostics.pop(manifest.session_id)
                self.recovery_diagnostics.pop("review", None)
            except (OSError, ValueError):
                # Keep the original queue intact and expose no inferred authority.
                manifest = None
                self._review_recovery_required = True
                self.recovery_diagnostics["review"] = "POSTGAME_REVIEW_ORIGINAL_CONTEXT_UNAVAILABLE"
        scope = (manifest.session_id, manifest.manifest_sha256) if manifest else None
        if scope != self._review_scope:
            self._review_scope = scope
            self.review_session_id = manifest.session_id if manifest else None
            self._kiosk_nonce = secrets.token_urlsafe(32) if manifest else ""
        return manifest

    @staticmethod
    def _recording_processing_progress(capture):
        cutoff = capture["cutoff"]
        processed, remaining, unavailable = None, None, False
        try:
            state = RecordingProcessingJournal.read_state(
                Path(capture["raw_manifest_path"]).parent.parent / "shot-boundaries.sqlite3",
                capture["session_id"],
                capture["source_epoch_id"],
            )
            checkpoint = state["checkpoint"]
            processed = None if checkpoint is None else checkpoint["last_frame_id"]
            if processed is not None and (
                cutoff is None or not capture["first_frame_id"] <= processed <= cutoff["frame_id"]
            ):
                raise ValueError("processing checkpoint outside original capture")
            remaining = (
                0
                if cutoff is None
                else cutoff["frame_id"]
                - (processed if processed is not None else capture["first_frame_id"] - 1)
            )
        except (OSError, ValueError, sqlite3.Error):
            unavailable = True
        if capture["processing_status"] == "COMPLETE":
            processed = None if cutoff is None else cutoff["frame_id"]
            remaining, unavailable = 0, False
        return dict(
            status=capture["processing_status"],
            tail_status=capture["tail_status"],
            processed_frame_id=processed,
            cutoff_frame_id=None if cutoff is None else cutoff["frame_id"],
            remaining_frames=remaining,
            progress_unavailable=unavailable,
            failure_code=capture["failure_code"],
        )

    def postgame_review_payload(self, *, kiosk: bool = False) -> dict[str, object]:
        with self.match_lock:
            manifest = self._active_review()
            if manifest is None:
                return self._postgame_review_payload_for(None, kiosk=kiosk)
            root = self.record_store.artifact_path_for_session(manifest.session_id)
        # Keep the original session alive while hashing, without blocking capture.
        try:
            with session_media_lock(root, manifest.session_id, exclusive=False, timeout_seconds=0):
                payload = self._postgame_review_payload_for(manifest, kiosk=kiosk)
                with self.match_lock:
                    if self._active_review() != manifest:
                        raise ValueError("review identity changed during projection")
                    if kiosk:
                        payload["kiosk_nonce"] = self._kiosk_nonce
                    return payload
        except TimeoutError as exc:
            raise ValueError("review media is busy") from exc

    def _postgame_review_payload_for(self, manifest, *, kiosk=False):
        diagnostics = (
            {"recovery_diagnostics": dict(self.recovery_diagnostics)}
            if self.managed_session_scope is not None or self.recovery_diagnostics
            else {}
        )
        if manifest is None:
            live_progress = {}
            ordinary_live = (
                self._review_live_active
                and getattr(self, "session_policy", "legacy") == "recording-v1"
            )
            if ordinary_live:
                try:
                    # Jobs first, then starts: new immutable shots may arrive between stores.
                    progress = self.media_job_store.session_media_progress(self.session_id)
                    discovered = len(self.record_store.list_shots(self.session_id))
                    live_progress = {
                        key: value for key, value in progress.items() if key != "items"
                    }
                    live_progress.update(
                        discovered_count=discovered,
                        pending_count=max(
                            0,
                            discovered
                            - progress["generated_count"]
                            - progress["failed_count"]
                            - progress["encoding_count"],
                        ),
                        progress_status="READY",
                    )
                except (OSError, RuntimeError, ValueError, AttributeError, sqlite3.Error):
                    live_progress = {"progress_status": "UNAVAILABLE"}
            return {
                "phase": (
                    "LIVE_MATCH"
                    if self._review_live_active
                    else "RECOVERY_REQUIRED"
                    if self._review_recovery_required
                    else "REVIEW_COMPLETE"
                ),
                "review_session_id": None,
                "postgame_manifest_sha256": None,
                "analysis_status": "NOT_REQUESTED" if ordinary_live else None,
                "expected_count": None if ordinary_live else 0,
                "available_count": 0,
                "can_complete": False,
                "clips": [],
                **diagnostics,
                **live_progress,
            }
        media_jobs = self._media_jobs_for_session(manifest.session_id)
        media_context_available = not (
            self.managed_session_scope is not None
            and manifest.session_id != self.session_id
            and media_jobs is None
        )
        if self.managed_session_scope is not None:
            diagnostics = {"recovery_diagnostics": dict(self.recovery_diagnostics)}
        if self.record_store.get_session_policy(manifest.session_id) == "recording-v1":
            recording = self.record_store.get_recording_end_manifest(manifest.session_id)
            capture = self.record_store.get_recording_capture_end(manifest.session_id)
            if recording is None and capture is None:
                raise ValueError("recording source authority is unavailable")
            complete = recording is not None and (
                capture is None or capture["processing_status"] == "COMPLETE"
            )
            inventory = (
                tuple(
                    dict(shot_id=shot.shot_id)
                    for shot in self.record_store.list_shots(manifest.session_id)
                )
                if capture is not None
                else recording["shots"]
            )
            try:
                playback = (
                    ()
                    if media_jobs is None
                    else project_recording_playback(
                        self.record_store,
                        media_jobs,
                        manifest,
                        continue_check=self._check_review_running,
                    )
                )
            except RecordingPlaybackError as exc:
                raise ValueError("recording playback projection is unavailable") from exc
            by_shot = {row.shot_id: row for row in playback}
            clips = []
            for sequence, frozen_shot in enumerate(inventory, start=1):
                row = by_shot.get(frozen_shot["shot_id"])
                clip = None if row is None else row.playback_clip
                status = "NOT_SCHEDULED" if row is None else row.encoding_status
                clips.append(
                    {
                        "shot_id": frozen_shot["shot_id"],
                        "sequence": sequence if row is None else row.sequence,
                        "actor_role": None,
                        "can_save": False,
                        "ownership_unavailable_reason": "SHOT_OWNERSHIP_UNAVAILABLE",
                        "clip_id": None if clip is None else clip.clip_id,
                        "save_status": "NOT_SAVED",
                        "owner_delete_status": None,
                        "encoding_status": status,
                        "terminal_failure": False if row is None else row.terminal_failure,
                        "error_code": None if row is None else row.error_code,
                        "media_url": (
                            None
                            if clip is None
                            else f"/media/postgame/{clip.clip_id}.mp4?manifest="
                            f"{manifest.manifest_sha256}"
                        ),
                    }
                )
            available_count = sum(row["media_url"] is not None for row in clips)
            from billiards_ai.lab.field_assistance.snapshot_review import snapshot_shot_review

            for clip in clips:
                guidance_review = snapshot_shot_review(
                    capture,
                    self.record_store.get_recording_evidence(manifest.session_id, clip["shot_id"]),
                )
                if guidance_review is not None:
                    clip["guidance_review"] = guidance_review
            statuses = {row["encoding_status"] for row in clips}
            encoding_status = (
                "AVAILABLE"
                if clips and available_count == len(clips)
                else "FAILED"
                if clips
                and all(row["media_url"] is not None or row["terminal_failure"] for row in clips)
                else "FAILED_RETRYING"
                if "FAILED_RETRYING" in statuses
                else "PROCESSING"
                if statuses - {"NOT_SCHEDULED"}
                else "NOT_SCHEDULED"
            )
            payload = {
                "schema_version": "recording-review/v2",
                "phase": (
                    "RECORDING_INTERRUPTED"
                    if complete
                    and capture is not None
                    and capture["contract"] == "recording-capture-end/v2"
                    else "RECORDING_ENDED"
                )
                if complete
                else "RECORDING_PROCESSING",
                "review_session_id": manifest.session_id,
                "postgame_manifest_sha256": manifest.manifest_sha256,
                "recording_end": None if recording is None else public_recording_end(recording),
                "recording_capture": (
                    None if capture is None else public_recording_capture(capture)
                ),
                "processing": (
                    dict(status="COMPLETE", remaining_frames=0)
                    if capture is None
                    else self._recording_processing_progress(capture)
                ),
                "discovered_count": len(inventory),
                "playback": {
                    "schema_version": (
                        "recording-playback/v1" if capture is None else "recording-playback/v2"
                    ),
                    "encoding_status": encoding_status,
                    "expected_count": len(clips) if complete else None,
                    "discovered_count": len(clips),
                    "available_count": available_count,
                },
                "analysis_status": "NOT_REQUESTED",
                "encoding_status": encoding_status,
                "expected_count": len(inventory) if complete else None,
                "available_count": available_count,
                "can_complete": complete
                and not (capture is not None and capture["contract"] == "recording-capture-end/v2")
                and media_context_available
                and (
                    capture is None
                    or all(row["media_url"] is not None or row["terminal_failure"] for row in clips)
                ),
                "clips": clips,
                **diagnostics,
            }
            if kiosk:
                payload["kiosk_nonce"] = self._kiosk_nonce
            return payload
        clips = self.record_store.list_postgame_clips(manifest.session_id)
        try:
            playback_rows = (
                ()
                if media_jobs is None
                else project_recording_playback(
                    self.record_store,
                    media_jobs,
                    manifest,
                    continue_check=self._check_review_running,
                )
            )
        except RecordingPlaybackError as exc:
            raise ValueError("recording playback projection is unavailable") from exc
        playback_by_shot = {row.shot_id: row for row in playback_rows}
        # Read-only fallback until a job snapshot provider is wired: availability,
        # finalization, and frozen deadline. GET never invokes a publisher.
        analysis_snapshots = (
            tuple(
                PostgameJobSnapshot(
                    snapshot.job.job_id,
                    snapshot.status.value,
                    snapshot.artifact_id,
                )
                for job_id in manifest.expected_jobs
                for snapshot in (media_jobs.snapshot(job_id),)
            )
            if media_jobs is not None
            else tuple(
                PostgameJobSnapshot(c.job_id, "available", c.artifact_id)
                for c in clips
                if c.clip_id
            )
        )
        projection = project_session_analysis(
            manifest,
            analysis_snapshots,
            self.postgame_now_utc(),
            finalized=self.record_store.is_postgame_finalized(manifest.session_id),
        )
        phase = {
            "AVAILABLE": "POSTGAME_REVIEW",
            "PARTIAL": "POSTGAME_REVIEW",
            "PROCESSING": "POSTGAME_PROCESSING",
            "FAILED_RETRYING": "FAILED_RETRYING",
        }[projection.status]
        payload = {
            "phase": phase,
            "review_session_id": manifest.session_id,
            "postgame_manifest_sha256": manifest.manifest_sha256,
            "analysis_status": projection.status,
            "expected_count": projection.expected_count,
            "available_count": projection.available_count,
            "can_complete": True,
            "playback": {
                "schema_version": "recording-playback/v1",
                "expected_count": len(playback_rows),
                "available_count": sum(row.playback_clip is not None for row in playback_rows),
                "encoding_status": (
                    "AVAILABLE"
                    if playback_rows and all(row.playback_clip is not None for row in playback_rows)
                    else "PROCESSING"
                ),
            },
            "clips": [
                {
                    "shot_id": c.shot_id,
                    "sequence": c.sequence,
                    "actor_role": c.actor_role,
                    "can_save": c.can_save,
                    "ownership_unavailable_reason": c.ownership_unavailable_reason,
                    "clip_id": c.clip_id,
                    "playback_clip_id": (
                        playback_by_shot[c.shot_id].playback_clip.clip_id
                        if c.shot_id in playback_by_shot
                        and playback_by_shot[c.shot_id].playback_clip is not None
                        else c.clip_id
                    ),
                    "save_status": c.save_status,
                    "owner_delete_status": c.owner_delete_status,
                    "encoding_status": (
                        playback_by_shot[c.shot_id].encoding_status
                        if c.shot_id in playback_by_shot
                        else "NOT_SCHEDULED"
                    ),
                    "terminal_failure": (
                        playback_by_shot[c.shot_id].terminal_failure
                        if c.shot_id in playback_by_shot
                        else False
                    ),
                    "error_code": (
                        playback_by_shot[c.shot_id].error_code
                        if c.shot_id in playback_by_shot
                        else None
                    ),
                    "media_url": (
                        f"/media/postgame/"
                        f"{playback_by_shot[c.shot_id].playback_clip.clip_id}.mp4"
                        f"?manifest={manifest.manifest_sha256}"
                    )
                    if c.shot_id in playback_by_shot
                    and playback_by_shot[c.shot_id].playback_clip is not None
                    and c.owner_delete_status is None
                    else (f"/media/postgame/{c.clip_id}.mp4?manifest={manifest.manifest_sha256}")
                    if c.clip_id and c.owner_delete_status is None
                    else None,
                }
                for c in clips
            ],
            **diagnostics,
        }
        if kiosk:
            payload["kiosk_nonce"] = self._kiosk_nonce
        return payload

    def mutate_postgame_review(self, payload: dict[str, str], nonce: str, *, complete: bool):
        with self.match_lock:
            manifest = self._active_review()
            if (
                manifest is None
                or not nonce
                or not hmac.compare_digest(nonce.encode(), self._kiosk_nonce.encode())
            ):
                raise PermissionError("kiosk nonce rejected")
            if (
                payload["session_id"] != manifest.session_id
                or payload["postgame_manifest_sha256"] != manifest.manifest_sha256
            ):
                raise ValueError("stale review identity")
            media_jobs = self._media_jobs_for_session(manifest.session_id)
            if (
                self.managed_session_scope is not None
                and manifest.session_id != self.session_id
                and media_jobs is None
                and self.record_store.get_session_policy(manifest.session_id) == "recording-v1"
            ):
                raise ValueError("original postgame media context is unavailable")
            if complete:
                # Commit the read/publication fence first. Existing shared
                # readers stop at their next chunk check; cleanup takes the exclusive
                # pin before touching bytes. Selected pending media is retained.
                self.record_store.complete_postgame_review(
                    self.record_store.load_session(manifest.session_id).artifact_id,
                    manifest.session_id,
                    manifest.manifest_sha256,
                    **(
                        {"media_jobs": media_jobs}
                        if self.record_store.get_session_policy(manifest.session_id)
                        == "recording-v1"
                        else {}
                    ),
                )
            else:
                receipt = self.record_store.save_postgame_clip(**payload)
                if media_jobs is not None:
                    try:
                        with session_media_lock(
                            self.record_store.artifact_path_for_session(manifest.session_id),
                            manifest.session_id,
                            exclusive=False,
                            timeout_seconds=0,
                        ):
                            publish_selected_original_clips(
                                self.record_store,
                                media_jobs,
                                manifest,
                            )
                    except TimeoutError as exc:
                        raise ValueError("postgame media is busy") from exc
        if complete:
            return self.postgame_review_payload(kiosk=True)
        if self.sync_wakeup is not None:
            self.sync_wakeup()
        return asdict(receipt)

    def _check_review_running(self):
        if self.projection_lifecycle_closing or self.projection_lifecycle_closed:
            raise RecordingPlaybackError("MEDIA_SOURCE_REVOKED", "review service is closing")

    def review_clip(self, clip_id: str, digest: str) -> PostgameClip:
        with self.match_lock:
            manifest = self._active_review()
            if manifest is None or digest != manifest.manifest_sha256:
                raise ValueError("review media is unavailable")
            media_jobs = self._media_jobs_for_session(manifest.session_id)
            if (
                self.managed_session_scope is not None
                and manifest.session_id != self.session_id
                and media_jobs is None
                and self.record_store.get_session_policy(manifest.session_id) == "recording-v1"
            ):
                raise ValueError("original postgame media context is unavailable")
            root = self.record_store.artifact_path_for_session(manifest.session_id)
        with session_media_lock(root, manifest.session_id, exclusive=False, timeout_seconds=0):
            clip = next(
                (
                    c
                    for c in self.record_store.list_postgame_clips(manifest.session_id)
                    if c.clip_id == clip_id
                ),
                None,
            )
            if clip is None and media_jobs is not None:
                try:
                    rows = project_recording_playback(
                        self.record_store,
                        media_jobs,
                        manifest,
                        clip_id=clip_id,
                        continue_check=self._check_review_running,
                    )
                except RecordingPlaybackError:
                    rows = ()
                clip = next(
                    (
                        r.playback_clip
                        for r in rows
                        if r.playback_clip is not None and r.playback_clip.clip_id == clip_id
                    ),
                    None,
                )
            with self.match_lock:
                if (
                    clip is not None
                    and self._active_review() == manifest
                    and not self.record_store.media_encoding_state(clip.session_id, clip.shot_id)[
                        "banned"
                    ]
                    and clip.owner_delete_status is None
                ):
                    return clip
        raise ValueError("review media is unavailable")

    def close_projection(self) -> None:
        with self.match_lock:
            self.projection_lifecycle_closing = True
            revoke = getattr(self.projection_source, "_revoke", None)
            if revoke is not None:
                revoke()
            pending = self._pending_projection_cleanup
            if pending is not None:
                pending._revoke()
            recoveries = tuple(self._recording_recoveries)
            for recovery in recoveries:
                recovery.close()
            retained = tuple(self._background_recordings)
            for owner in retained:
                owner._revoke()
        # Only serialize cleanup outside the match lock: runners and consumers
        # may still need that lock to commit their final immutable receipts.
        with self._projection_close_lock:
            with self.match_lock:
                if self.projection_lifecycle_closed:
                    return
                source = self.projection_source
            failures = []
            for recovery in recoveries:
                try:
                    recovery.shutdown()
                except BaseException as exc:
                    failures.append(exc)
            owners = tuple(
                dict.fromkeys(
                    id(owner) for owner in (source, pending, *retained) if owner is not None
                )
            )
            by_id = {
                id(owner): owner for owner in (source, pending, *retained) if owner is not None
            }
            # Stop every detector before any shared hub/camera is released.
            for identity in owners:
                owner = by_id[identity]
                if retained and isinstance(owner, USBLiveProjectionSource):
                    try:
                        owner.close()
                    except BaseException as exc:
                        failures.append(exc)
            if failures:
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup("projection analysis cleanup failed", failures)
            for identity in owners:
                owner = by_id[identity]
                try:
                    shutdown = getattr(owner, "shutdown", owner.close)
                    shutdown()
                except BaseException as exc:
                    failures.append(exc)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("projection cleanup failed", failures)
            with self.match_lock:
                self._pending_projection_cleanup = None
                self._background_recordings.clear()
                self._recording_recoveries.clear()
                self.projection_lifecycle_closed = True

    def _require_projection_authority(self):
        if self.projection_lifecycle_closing or self.projection_lifecycle_closed:
            raise ValueError("LIVE_AUTHORITY_UNAVAILABLE")

    def before_field_command(self, envelope: FieldCommandEnvelope) -> None:
        """Only local arrived boundaries; callers hold the same match lock."""
        with self.match_lock:
            self._require_projection_authority()
            if envelope.session_id != self.session_id:
                return
            self._advance_records(self.projection_source.current().position_ms)

    def set_command_match(self, session_id: str, match: MatchControlSnapshot) -> None:
        with self.match_lock:
            if session_id != self.session_id:
                return
            self.match = match

    def set_cloud_control_status(self, status: str) -> None:
        with self.match_lock:
            self.cloud_control_status = status

    def handle_command_application(
        self, envelope: FieldCommandEnvelope, application: FieldCommandApplication
    ) -> None:
        with self.match_lock:
            self._require_projection_authority()
            if envelope.session_id != self.session_id:
                return
            self.last_applied_action = _describe_applied_command(envelope, application, self.match)
            if envelope.command_type == "set_breaker" and application.outcome == "applied":
                self.projection_source.apply_control({"command": "start"})
            if envelope.command_type == "end_session" and application.outcome == "applied":
                if self.session_ended_at_monotonic is None:
                    self.projection_source.end_session()
                    self.session_ended_at_monotonic = self.monotonic()
                    try:
                        self._enqueue_session_sync()
                    except (OSError, ValueError):
                        self.sync_error = "本场事实正在本机等待重新同步"

    def _pair_for(self, sequence: int) -> tuple[ShotBoundary, ShotBoundary]:
        matching = tuple(
            boundary for boundary in self.shot_index.boundaries if boundary.sequence == sequence
        )
        if (
            len(matching) != 2
            or matching[0].kind is not ShotBoundaryKind.STARTED
            or matching[1].kind is not ShotBoundaryKind.SETTLED
        ):
            raise ValueError("authoritative shot boundary pair is invalid")
        return matching[0], matching[1]

    def _existing_shot(self, sequence: int) -> FieldShotRecord | None:
        return next(
            (
                shot
                for shot in self.record_store.list_shots(self.session_id)
                if shot.sequence == sequence
            ),
            None,
        )

    def _record_boundary(
        self, boundary: ShotBoundary, *, ownership_ambiguous: bool = False
    ) -> None:
        started, settled = self._pair_for(boundary.sequence)
        media = build_media_range(
            artifact_id=self.artifact_id,
            started_ms=started.timestamp_ms,
            settled_ms=settled.timestamp_ms,
            duration_ms=self.duration_ms,
        )
        power = (
            None
            if started.recommendation_id is None
            else self.recommendation_power.get(started.recommendation_id)
        )
        existing = self._existing_shot(boundary.sequence)
        if boundary.kind is ShotBoundaryKind.STARTED and existing is not None:
            # Optional provenance may have appeared/disappeared across restart.
            # Replay never upgrades a durable start's original ownership receipt.
            return
        actor = self.match.active_role if existing is None else existing.actor_role
        shot = FieldShotRecord(
            session_id=self.session_id,
            shot_id=f"shot-{boundary.sequence:04d}",
            sequence=boundary.sequence,
            actor_role=actor,
            started_ms=started.timestamp_ms,
            settled_ms=(
                settled.timestamp_ms if boundary.kind is ShotBoundaryKind.SETTLED else None
            ),
            recommendation_id=started.recommendation_id,
            power_band=None if power is None else power.band,
            power_percent=None if power is None else power.approximate_percent,
            media=media,
            limitations=(SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,),
        )
        if boundary.kind is ShotBoundaryKind.STARTED:
            event_id = dict(self.verified_start_events).get(boundary.sequence)
            self.record_store.record_shot_start(
                shot,
                source_mainline1_artifact_id=self.source_mainline1_artifact_id,
                source_start_event_id=event_id,
                ownership_ambiguous=ownership_ambiguous or event_id is None,
                ownership_reason=(
                    "TEMPORAL_ORDER_UNPROVEN"
                    if ownership_ambiguous
                    else (
                        "SOURCE_START_UNVERIFIED"
                        if self.source_validation_reason == "SOURCE_START_VERIFIED"
                        else self.source_validation_reason
                    )
                    if event_id is None
                    else "ACTOR_UNKNOWN"
                ),
            )
        else:
            self.record_store.record_shot_settle(shot)

    def _advance_records(self, position_ms: int) -> None:
        if not self.replay_has_started:
            self.replay_has_started = self.replay_clock.projection().playing or position_ms > 0
        if not self.replay_has_started:
            return
        if self.record_store.load_session(self.session_id).status is FieldSessionStatus.ENDED:
            return
        if position_ms < self.record_cursor_ms:
            self.record_cursor_ms = -1
            self.ownership_recovery_ambiguous = True
        boundaries = self.shot_index.boundaries_between(
            after_ms=self.record_cursor_ms,
            through_ms=position_ms,
        )
        new_starts = [
            b
            for b in boundaries
            if b.kind is ShotBoundaryKind.STARTED and self._existing_shot(b.sequence) is None
        ]
        for boundary in boundaries:
            self._record_boundary(
                boundary,
                ownership_ambiguous=(
                    self.ownership_recovery_ambiguous
                    or len(new_starts) > 1
                    or boundary.timestamp_ms == position_ms
                    or boundary.timestamp_ms == 0
                ),
            )
        self.record_cursor_ms = position_ms

    def _close_managed_session(
        self, session: FieldSessionRecord | None, gateway: FieldCloudGateway
    ) -> ManagedSessionCloseState:
        # A retained ended scene and an explicitly configured persistent scope
        # are both required. Neither the name nor gateway attestation grants it.
        if session is None or session.status is not FieldSessionStatus.ENDED:
            raise ValueError("managed close requires retained ended local session")
        if self.managed_session_scope is None:
            raise ValueError("managed close requires configured scope")
        request = self.record_store.get_materialized_managed_open_request(
            self.managed_session_scope, session.session_id
        )
        return poll_managed_close(self.record_store, request, gateway, self.match_lock)

    def _maintain_join(self, *, ended: bool) -> None:
        # Serialize cloud maintenance independently of physical-shot authority.
        # Another HTTP request can project immediately while a cloud call waits.
        if not self.cloud_io_lock.acquire(blocking=False):
            return
        try:
            with self.match_lock:
                coordinator = self.join_coordinator
                if coordinator is None:
                    return
                session = self.record_store.load_session(self.session_id)
                session_id, revision, status = (
                    session.session_id,
                    session.match.revision,
                    session.status,
                )
                now = self.monotonic()
                for pending_id in self.record_store.list_pending_cloud_close_session_ids():
                    self.pending_cloud_closes.setdefault(pending_id, now)
                ended = status is FieldSessionStatus.ENDED
                if (
                    ended
                    and self.cloud_session_status == "open"
                    and session_id not in self.pending_cloud_closes
                ):
                    self.record_store.enqueue_cloud_close_intent(session_id)
                    self.pending_cloud_closes[session_id] = now
                closes = tuple(key for key, due in self.pending_cloud_closes.items() if now >= due)
                should_open = (
                    not ended
                    and self.cloud_session_status != "open"
                    and now >= self.cloud_session_next_attempt
                )

            def is_current():
                # Called only under match_lock. Do not apply a late lease or scan
                # result to a changed session, roster revision, or ended session.
                return (
                    self.join_coordinator is coordinator
                    and self.session_id == session_id
                    and (current := self.record_store.load_session(session_id)).match.revision
                    == revision
                    and current.status is status
                )

            for pending_session_id in closes:
                try:
                    try:
                        pending_session = self.record_store.load_session(pending_session_id)
                    except ValueError as exc:
                        if str(exc) != "field session does not exist":
                            raise
                        pending_session = None
                    managed = pending_session_id.startswith("managed-")
                    if managed:
                        close = self._close_managed_session(pending_session, coordinator.gateway)
                        if close.coverage != "sealed":
                            with self.match_lock:
                                self.pending_cloud_closes[pending_session_id] = (
                                    self.monotonic() + _CLOUD_SESSION_RETRY_SECONDS
                                )
                            # Return to the worker loop to apply/ACK one command.
                            # Business evidence gates must not block prefix draining.
                            continue
                    pending_match = None if pending_session is None else pending_session.match
                    if pending_match is not None and pending_match.ranking_intent:
                        if (
                            pending_session.status is not FieldSessionStatus.ENDED
                            or pending_match.mode is not MatchMode.OBSERVATION
                            or pending_match.match_format is not MatchFormat.SINGLE
                            or pending_match.rack_phase.value != "ended"
                            or pending_match.winner_role is None
                        ):
                            raise ValueError("ranked terminal result is incomplete")
                        if not self.record_store.is_session_sync_complete(pending_session_id):
                            raise OSError("ranked session evidence is still syncing")
                        red_won = pending_match.winner_role is WristbandRole.RED
                        coordinator.gateway.propose_match_result(
                            session_id=pending_session_id,
                            winner_role=pending_match.winner_role,
                            red_score=1 if red_won else 0,
                            blue_score=0 if red_won else 1,
                            finish_reason="normal",
                        )
                    if not managed:
                        coordinator.gateway.close_field_session(pending_session_id)
                except (OSError, ValueError):
                    with self.match_lock:
                        self.pending_cloud_closes[pending_session_id] = (
                            self.monotonic() + _CLOUD_SESSION_RETRY_SECONDS
                        )
                        self.cloud_close_error = "上一场证据、赛果或云端关闭正在后台重试"
                else:
                    with self.match_lock:
                        self.record_store.complete_cloud_close_intent(pending_session_id)
                        self.pending_cloud_closes.pop(pending_session_id, None)
                        if not self.pending_cloud_closes:
                            self.cloud_close_error = None
                        if pending_session_id == session_id and ended and is_current():
                            self.cloud_session_status = "ended"
            if should_open:
                try:
                    if session_id.startswith("managed-"):
                        raise OSError("managed startup requires explicit reservation recovery")
                    lease = coordinator.gateway.open_field_session(session_id)
                    if lease.session_id != session_id:
                        raise OSError("cloud lease session identity mismatch")
                except OSError:
                    with self.match_lock:
                        if is_current():
                            self.join_error = "云端加入服务暂时不可用，系统正在自动重试"
                            self.cloud_session_status = "retrying"
                            self.cloud_session_next_attempt = (
                                self.monotonic() + _CLOUD_SESSION_RETRY_SECONDS
                            )
                else:
                    with self.match_lock:
                        self.record_store.record_cloud_session_open(session_id)
                        if (
                            self.record_store.load_session(session_id).status
                            is FieldSessionStatus.ENDED
                        ):
                            # The lease opened after local end/rollover: close the
                            # original cloud session without touching the new one.
                            self.record_store.enqueue_cloud_close_intent(session_id)
                            self.pending_cloud_closes[session_id] = self.monotonic()
                        elif is_current():
                            self.cloud_session_status = lease.status
                            self.join_error = None
                            self.join_requests_next_refresh = self.monotonic()
            with self.match_lock:
                should_refresh = (
                    is_current()
                    and not ended
                    and self.cloud_session_status == "open"
                    and self.monotonic() >= self.join_requests_next_refresh
                )
            if should_refresh:
                try:
                    requests = coordinator.gateway.refresh_join_requests(session_id)
                except OSError:
                    requests = None
                with self.match_lock:
                    if is_current():
                        if requests is not None:
                            coordinator.pending = tuple(
                                item for item in requests if item.status == "pending"
                            )
                            coordinator.confirmed_requests = tuple(
                                item
                                for item in requests
                                if item.status == "confirmed" and item.role is not None
                            )
                        self.join_requests_next_refresh = (
                            self.monotonic() + _JOIN_REQUEST_REFRESH_SECONDS
                        )
        finally:
            self.cloud_io_lock.release()

    def _rollover_ended_session(
        self, session: FieldSessionRecord
    ) -> tuple[FieldSessionRecord, bool]:
        if (
            self.projection_lifecycle_closing
            or self.projection_lifecycle_closed
            or session.status is not FieldSessionStatus.ENDED
            or session.session_id.startswith("managed-")
        ):
            return session, False
        ended_at = self.session_ended_at_monotonic
        if ended_at is None:
            self.session_ended_at_monotonic = self.monotonic()
            return session, False
        if self.monotonic() - ended_at < _SESSION_ENDED_RECEIPT_SECONDS:
            return session, False
        if not self._can_rollover(session):
            self.sync_error = "本场事实正在本机等待重新同步"
            return session, False
        self._prepare_projection_rollover()
        gateway = None if self.join_coordinator is None else self.join_coordinator.gateway
        if self._pending_rollover_session is None:
            self._pending_rollover_session = self.record_store.open_or_resume_session(
                artifact_id=self.artifact_id,
                rule_profile_id=getattr(self, "rule_profile_id", None),
                session_policy=getattr(self, "session_policy", "legacy"),
            )
        next_session = self._pending_rollover_session
        self.record_store.bind_artifact_path(self.artifact_id, self.artifact_path)
        if not self._rollover_projection(next_session):
            return session, False
        self.session_id = next_session.session_id
        self.match = next_session.match
        self._pending_rollover_session = None
        self.record_cursor_ms = -1
        self.replay_has_started = False
        self.ownership_recovery_ambiguous = False
        self.last_applied_action = None
        self.session_ended_at_monotonic = None
        self.sync_error = None
        self.cloud_session_status = "unavailable"
        self.cloud_session_next_attempt = 0
        self.join_requests_next_refresh = 0
        self.join_error = None
        self.join_coordinator = (
            None
            if gateway is None
            else FieldJoinCoordinator(session_id=next_session.session_id, gateway=gateway)
        )
        return next_session, True

    def _can_rollover(self, session: FieldSessionRecord) -> bool:
        return self.record_store.has_sync_job(session.session_id)

    def _prepare_projection_rollover(self) -> None:
        """Complete old-source cleanup before advancing the current match."""

    def _reset_projection(self) -> None:
        self.projection_source.apply_control({"command": "reset"})

    def _rollover_projection(self, session: FieldSessionRecord) -> bool:
        self._reset_projection()
        return True

    def payload(self) -> dict[str, object]:
        self.maintain_background()
        with self.match_lock:
            session = self.record_store.load_session(self.session_id)
            session, rolled_over = self._rollover_ended_session(session)
        if rolled_over:
            self._maintain_join(ended=False)
        return self._projection_payload()

    def _projection_payload(self) -> dict[str, object]:
        with self.match_lock:
            session = self.record_store.load_session(self.session_id)
            projection = self.projection_source.current()
            self._advance_records(projection.position_ms)
            match = _jsonable(asdict(self.match))
            join_status = self.cloud_session_status
            display_phase = self._display_phase(
                session.status, str(projection.frame.payload.get("phase", ""))
            )
            pending = () if self.join_coordinator is None else self.join_coordinator.pending
            confirmed_requests = (
                () if self.join_coordinator is None else self.join_coordinator.confirmed_requests
            )
            confirmed_by_role = {item.role: item for item in confirmed_requests}
            red_ready = self.match.red_user_id is not None
            blue_ready = self.match.blue_user_id is not None
            if pending:
                visible_names = "、".join(item.display_name for item in pending[:2])
                suffix = "等用户" if len(pending) > 2 else ""
                join_prompt = f"{visible_names}{suffix} 已加入，请在手机认领红方或蓝方"
            elif red_ready and blue_ready:
                join_prompt = "双方已就绪"
            elif red_ready:
                join_prompt = "红方已就绪，等待另一位用户扫码"
            elif blue_ready:
                join_prompt = "蓝方已就绪，等待另一位用户扫码"
            else:
                join_prompt = "微信扫码加入本桌"
        return {
            "schema_version": "1.0",
            "playback": {
                "position_ms": projection.position_ms,
                "playing": projection.playing,
            },
            "frame": projection.frame.payload,
            "match": match,
            "session": {
                "session_id": session.session_id,
                "status": session.status.value,
            },
            "display": {
                "phase": display_phase,
                "table_code_image": "/api/table-code.png",
                "join_prompt": join_prompt,
                "last_action": self.last_applied_action,
                "red_slot": (
                    f"红方：{confirmed_by_role[WristbandRole.RED].display_name}"
                    if WristbandRole.RED in confirmed_by_role
                    else "红方已就绪"
                    if red_ready
                    else "红方待认领"
                ),
                "blue_slot": (
                    f"蓝方：{confirmed_by_role[WristbandRole.BLUE].display_name}"
                    if WristbandRole.BLUE in confirmed_by_role
                    else "蓝方已就绪"
                    if blue_ready
                    else "蓝方待认领"
                ),
            },
            "cloud_control": {
                "status": self.cloud_control_status,
                "available": self.cloud_control_status == "connected",
            },
            "join": {
                "available": self.join_coordinator is not None,
                "status": join_status,
                "qr_available": self.join_qr_asset is not None,
                "expires_at": None,
                "error": self.join_error or self.cloud_close_error,
                "pending": (_jsonable([asdict(item) for item in pending])),
            },
        }

    def _display_phase(self, status: FieldSessionStatus, frame_phase: str) -> str:
        if status is FieldSessionStatus.ENDED:
            return "SESSION_ENDED"
        if self.join_error is not None or self.cloud_close_error is not None:
            return "DEGRADED"
        if self.match.red_user_id is None or self.match.blue_user_id is None:
            return "WAITING_FOR_PLAYERS"
        if self.match.breaker is None or (
            self.match.rule_profile_id is not None and not self.match.settings_frozen
        ):
            return "WAITING_FOR_SETUP"
        if frame_phase == "moving":
            return "BALLS_MOVING"
        return "WAITING_FOR_SHOT"

    def _save_participant(self, role: WristbandRole, user_id: str) -> None:
        with self.match_lock:
            self._advance_records(self.replay_clock.projection().position_ms)
            if self.record_store.list_shots(self.session_id):
                raise ValueError("participant assignment is frozen after start")
            previous_revision = self.match.revision
            updated = apply_control(self.match, SetParticipant(role, user_id))
            self.record_store.save_match(
                self.session_id,
                updated,
                expected_revision=previous_revision,
            )
            self.match = updated

    def _prepare_sync_media(
        self, session: FieldSessionRecord, shots: tuple[FieldShotRecord, ...]
    ) -> tuple[SyncMediaItem, ...]:
        source_path = (
            self.record_store.artifact_path_for_session(session.session_id) / "presentation.mp4"
        )
        target_dir = self.record_store.path.parent / "sync-media" / session.session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("field replay presentation cannot be decoded for sync")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError("field replay presentation geometry is invalid for sync")
        prepared: list[SyncMediaItem] = []
        try:
            for shot in shots:
                clip_id = f"clip-{shot.shot_id}"
                destination = (target_dir / f"{clip_id}.mp4").resolve()
                temporary = target_dir / f".{clip_id}.tmp.mp4"
                start_frame = max(0, math.floor(shot.media.start_ms * fps / 1000))
                end_frame = max(start_frame + 1, math.ceil(shot.media.end_ms * fps / 1000))
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise ValueError("field sync clip encoder is unavailable")
                try:
                    written_frames = 0
                    for _frame_id in range(start_frame, end_frame):
                        ok, frame = capture.read()
                        if not ok:
                            break
                        writer.write(frame)
                        written_frames += 1
                finally:
                    writer.release()
                if written_frames != end_frame - start_frame:
                    temporary.unlink(missing_ok=True)
                    raise ValueError("field sync clip did not contain the complete frame range")
                if not temporary.is_file() or temporary.stat().st_size <= 0:
                    raise ValueError("field sync clip could not be prepared")
                temporary.replace(destination)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                prepared.append(
                    SyncMediaItem(
                        clip_id=clip_id,
                        shot_id=shot.shot_id,
                        local_path=destination,
                        content_type="video/mp4",
                        content_sha256=digest,
                        byte_size=destination.stat().st_size,
                    )
                )
        finally:
            capture.release()
        return tuple(prepared)

    def _enqueue_session_sync(self, session_id: str | None = None) -> None:
        target_session_id = session_id or self.session_id
        session = self.record_store.load_session(target_session_id)
        shots = self.record_store.list_shots(target_session_id)
        report = compose_field_fact_report(session, shots)
        media = self._prepare_sync_media(session, shots)
        envelope = field_sync_envelope(report, media, evidence_source_kind="development_replay")
        manifest_json = json.dumps(
            envelope.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.record_store.enqueue_sync_job(
            session_id=target_session_id,
            manifest_hash=canonical_sync_hash(envelope),
            manifest_json=manifest_json,
            media=media,
        )
        if self.sync_wakeup is not None:
            self.sync_wakeup()

    def _retry_unsynced_sessions(self) -> None:
        with self.match_lock:
            failures = False
            for session in self.record_store.list_ended_sessions_without_sync():
                try:
                    self._enqueue_session_sync(session.session_id)
                except (OSError, ValueError):
                    failures = True
            self.sync_error = "本场事实正在本机等待重新同步" if failures else None

    def maintain_background(self) -> None:
        with self.match_lock:
            if self.projection_lifecycle_closing or self.projection_lifecycle_closed:
                return
            session = self.record_store.load_session(self.session_id)
        self._maintain_join(ended=session.status is FieldSessionStatus.ENDED)
        self._retry_unsynced_sessions()

    def mutate(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("control payload is invalid")
        with self.match_lock:
            self._require_projection_authority()
            projection = self.projection_source.current()
            self._advance_records(projection.position_ms)
            command = payload.get("command")
            if command == "start":
                self.replay_clock.start()
                return
            if command == "pause":
                self.replay_clock.pause()
                return
            if command == "reset":
                self.replay_clock.reset()
                self.record_cursor_ms = -1
                self.replay_has_started = False
                self.ownership_recovery_ambiguous = bool(
                    self.record_store.list_shots(self.session_id)
                )
                return
            if command not in {
                "emergency_set_participant",
                "emergency_set_breaker",
                "emergency_assign_groups",
            }:
                raise ValueError("administrator control command is unsupported")
            reason = payload.get("reason")
            if not isinstance(reason, str) or not 8 <= len(reason.strip()) <= 300:
                raise ValueError("administrator emergency reason is required")
            if self.record_store.list_shots(self.session_id):
                raise ValueError("administrator emergency correction is frozen after start")
            translated = dict(payload)
            translated["command"] = command.removeprefix("emergency_")
            translated.pop("reason", None)
            parsed = _parse_control(translated)
            previous_revision = self.match.revision
            updated = apply_control(self.match, parsed)
            self.record_store.apply_admin_override(
                self.session_id,
                updated,
                expected_revision=previous_revision,
                command_type=command,
                reason=reason,
                payload={key: value for key, value in translated.items() if key != "command"},
            )
            self.match = updated

    def join_qr_image(self) -> tuple[bytes, str]:
        if self.join_qr_asset is None:
            raise ValueError("permanent table code image is unavailable")
        return self.join_qr_asset.image_bytes, self.join_qr_asset.content_type

    def session_payload(self) -> dict[str, object]:
        with self.match_lock:
            session = self.record_store.load_session(self.session_id)
            return _jsonable(asdict(session))  # type: ignore[return-value]

    def shots_payload(self) -> dict[str, object]:
        with self.match_lock:
            shots = self.record_store.list_shots(self.session_id)
            return {
                "schema_version": "1.0",
                "session_id": self.session_id,
                "shots": _jsonable([asdict(shot) for shot in shots]),
            }

    def report_payload(self) -> dict[str, object]:
        with self.match_lock:
            session = self.record_store.load_session(self.session_id)
            if session.status is not FieldSessionStatus.ENDED:
                raise ValueError("field fact report requires an ended session")
            report = compose_field_fact_report(
                session,
                self.record_store.list_shots(self.session_id),
            )
            return field_fact_report_payload(report)


class _LiveServerState(_ServerState):
    """Retain the established session, cloud, QR, record and rollover authority."""

    def _advance_records(self, position_ms):
        # The concurrent channel consumer alone records USB boundaries.
        return

    def wait_until_capture_ready(self, *, timeout_seconds: float) -> str:
        source = self.projection_source
        if not source.wait_until_ready(timeout_seconds=timeout_seconds):
            reason = getattr(source, "failure_reason", None)
            detail = "" if not reason else f": {reason}"
            raise RuntimeError(f"live USB source did not complete validated startup{detail}")
        with self.match_lock:
            if (
                self.projection_lifecycle_closing
                or self.projection_lifecycle_closed
                or source is not self.projection_source
                or source.restart_required
            ):
                raise RuntimeError("live USB source changed during validated startup")
            source_epoch_id = getattr(source, "source_epoch_id", None)
            if not isinstance(source_epoch_id, str) or not source_epoch_id:
                raise RuntimeError("validated live USB source epoch is missing")
            return source_epoch_id

    def payload(self):
        # Source.current alone cannot guard DB reads performed by session/cloud
        # maintenance before projection. Keep this entire HTTP boundary safe.
        if not getattr(self, "_projection_store_failed", False):
            try:
                return super().payload()
            except Exception as exc:
                with self.match_lock:
                    self.projection_source.fail(
                        f"HTTP projection failure: {type(exc).__name__}: {exc}"
                    )
                    self._projection_store_failed = True
        with self.match_lock:
            projection = self.projection_source.current()
            frame = projection.frame.payload
            return {
                "schema_version": "1.0",
                "playback": {"position_ms": projection.position_ms, "playing": False},
                "frame": frame,
                "match": _jsonable(asdict(MatchControlSnapshot())),
                "session": {"session_id": frame["session_id"], "status": "unavailable"},
                "display": {
                    "phase": "DEGRADED",
                    "table_code_image": "/api/table-code.png",
                    "join_prompt": "现场状态不可用",
                    "last_action": "",
                    "red_slot": "红方状态不可用",
                    "blue_slot": "蓝方状态不可用",
                },
                "cloud_control": {"status": "unavailable", "available": False},
                "join": {
                    "available": False,
                    "status": "unavailable",
                    "qr_available": self.join_qr_asset is not None,
                    "expires_at": None,
                    "error": frame["degraded_reason"],
                    "pending": [],
                },
                **{
                    key: frame[key]
                    for key in (
                        "session_id",
                        "rule_status",
                        "guidance",
                        "confirmation_ref",
                        "rule_pending_reason",
                        "degraded_reason",
                        "route_abstained_reason",
                    )
                },
            }

    def before_field_command(self, envelope):
        with self.match_lock:
            self._require_projection_authority()
            if envelope.session_id != self.session_id or envelope.command_type != "end_session":
                return
            session = self.record_store.load_session(self.session_id)
            match = session.match
            participants = {match.red_user_id, match.blue_user_id}
            # This runs under the same service lock as ledger application. Never
            # freeze capture for a retry, ID collision, or out-of-order command.
            if (
                session.status is FieldSessionStatus.OPEN
                and envelope.actor_user_id in participants
                and envelope.expected_revision == match.revision
                and not envelope.payload
                and self.record_store.is_new_ordered_field_command(envelope)
            ):
                self.projection_source.prepare_session_end()

    def _can_rollover(self, session):
        # Ended sessions and immutable USB jobs are durable. The current cloud
        # media contract accepts replay MP4 only; publication is a worker concern.
        if session.status is not FieldSessionStatus.ENDED:
            return False
        if getattr(self, "session_policy", "legacy") == "recording-v1":
            capture = self.record_store.get_recording_capture_end(session.session_id)
            if capture is not None:
                return capture["tail_status"] == "SAVED"
            return self.record_store.get_recording_end_manifest(session.session_id) is not None
        return True

    def _release_projection_for_replacement(self):
        source = self.projection_source
        if (
            getattr(source, "session_policy", "legacy") == "recording-v1"
            and self.record_store.get_recording_capture_end(source.session_id) is not None
        ):
            if source not in self._background_recordings:
                self._background_recordings.append(source)
        else:
            source.close()

    def _prepare_projection_rollover(self):
        self._require_projection_authority()
        self._release_projection_for_replacement()

    def _rollover_projection(self, session):
        failure = getattr(self, "_rollover_failure", None)
        target = (session.session_id, self.projection_source.source_epoch_id, 0)
        if (
            failure is not None
            and getattr(
                failure,
                "_pending_recorder_handoff_target",
                None,
            )
            != target
        ):
            # A general constructor/start failure is not proof of a retryable
            # recorder handoff. Keep it and any candidate cleanup owner intact.
            raise failure
        try:
            self._reset_projection(session_id=session.session_id)
        except Exception as exc:
            self._rollover_failure = exc
            if getattr(exc, "_pending_recorder_handoff_target", None) != target:
                raise
            return False
        self._rollover_failure = None
        return True

    def _reset_projection(self, *, recover_capture: bool = False, session_id=None):
        if self.projection_lifecycle_closing or self.projection_lifecycle_closed:
            return
        if self._pending_projection_cleanup is not None:
            raise self._projection_start_failure
        self._release_projection_for_replacement()
        replacement = self.live_source_factory(
            self.session_id if session_id is None else session_id,
            recover_capture=recover_capture,
        )
        # A partial start still owns threads and must remain reachable by the
        # service's shutdown, without publishing an unmatched active context.
        self._pending_projection_cleanup = replacement
        try:
            replacement.start()
        except BaseException as exc:
            self._projection_start_failure = exc
            raise
        self.projection_source = replacement
        self.frame_store = replacement
        self._pending_projection_cleanup = None

    def maintain_service(self):
        self._maintain_recording_media()
        with self.match_lock:
            self._background_recordings[:] = [
                source
                for source in self._background_recordings
                if not (
                    source._closed
                    and getattr(source.runtime, "snapshot_guidance_cleanup_complete", True)
                    and all(
                        thread is None or not thread.is_alive()
                        for thread in (
                            source._runner,
                            source._consumer,
                            source._recording_finalizer,
                            source.runtime._recording_tail_thread,
                        )
                    )
                )
            ]
            if self.projection_lifecycle_closing or self.projection_lifecycle_closed:
                return
            if getattr(self, "_projection_store_failed", False):
                return
            if self._pending_rollover_session is not None:
                self._rollover_ended_session(self.record_store.load_session(self.session_id))
                return
            now = self.monotonic()
            if not self.projection_source.restart_required:
                if self.projection_source.poll_capture_ready():
                    self.capture_recovery_attempts = 0
                    self.capture_recovery_not_before = 0.0
                return
            if getattr(self, "managed_blackbox", False):
                # Finite files may only restart at frame zero after a fresh managed birth.
                return
            if now < self.capture_recovery_not_before:
                return
            self._reset_projection(recover_capture=True)
            self.capture_recovery_attempts += 1
            delay = min(
                _CAPTURE_RETRY_MAX_SECONDS,
                _CAPTURE_RETRY_INITIAL_SECONDS * 2 ** (self.capture_recovery_attempts - 1),
            )
            # Closing a native capture can take a measurable amount of time under
            # load. Start the backoff after the replacement has actually started,
            # otherwise close/start latency consumes the delay and retries can
            # collapse into a restart storm.
            self.capture_recovery_not_before = self.monotonic() + delay

    def _projection_payload(self):
        with self.match_lock:
            self.maintain_service()
            self.match = self.record_store.load_session(self.session_id).match
            payload = super()._projection_payload()
            frame = payload["frame"]
            failure = getattr(self, "_rollover_failure", None)
            if failure is not None:
                frame["rule_status"] = "DEGRADED"
                frame["degraded_reason"] = (
                    f"projection rollover pending: {type(failure).__name__}: {failure}"
                )
            for key in (
                "session_id",
                "rule_status",
                "guidance",
                "confirmation_ref",
                "rule_pending_reason",
                "degraded_reason",
                "route_abstained_reason",
            ):
                payload[key] = frame[key]
            if frame["rule_status"] == "DEGRADED":
                payload["display"]["phase"] = "DEGRADED"
            if self.last_applied_action is None:
                payload["display"]["last_action"] = frame.get("capture_action")
            health = getattr(self.projection_source, "subsystem_health", lambda: {})()
            payload.update(health)
            payload["session_policy"] = getattr(self, "session_policy", "legacy")
            payload["media_encoding_idle"] = bool(
                not session_has_live_activity(self.record_store, self.session_id)
                and getattr(self.projection_source, "media_encoding_idle", lambda: False)()
            )
            payload["sync_health"] = {
                "status": (
                    "degraded"
                    if self.sync_error is not None
                    else "ready"
                    if self.cloud_control_status == "connected"
                    else "unavailable"
                ),
                "reason": self.sync_error,
            }
            return payload

    def mutate(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("control payload is invalid")
        with self.match_lock:
            self._require_projection_authority()
            command = payload.get("command")
            if command in {"continue_turn", "switch_turn"}:
                self.projection_source.apply_control(payload)
                self.match = self.record_store.load_session(self.session_id).match
            elif command == "end_session":
                if getattr(self, "session_policy", "legacy") == "recording-v1":
                    target = payload.get("session_id")
                    if not isinstance(target, str) or not target:
                        raise ValueError("RECORDING_SESSION_ID_REQUIRED")
                    original = self.record_store.load_session(target)
                    if original.artifact_id != self.artifact_id:
                        raise ValueError("RECORDING_SESSION_MISMATCH")
                    capture = self.record_store.get_recording_capture_end(target)
                    if capture is not None and original.status is FieldSessionStatus.ENDED:
                        return {"recording_end": public_recording_capture(capture)}
                    frozen = self.record_store.get_recording_end_manifest(target)
                    if frozen is not None:
                        return {"recording_end": frozen}
                    if target != self.session_id:
                        raise ValueError("RECORDING_SESSION_MISMATCH")
                try:
                    self.projection_source.prepare_session_end()
                    if getattr(self, "session_policy", "legacy") == "legacy" and any(
                        shot.settled_ms is None
                        for shot in self.record_store.list_shots(self.session_id)
                    ):
                        raise ValueError("SHOT_UNSETTLED")
                    if not self.projection_source.ready_to_end:
                        raise ValueError("LIVE_AUTHORITY_UNAVAILABLE")
                    # Keep stopped capture uncommitted/DEGRADED until SQLite
                    # durably ends the session, just as in the cloud command path.
                    self.record_store.end_session(self.session_id)
                    self.projection_source.end_session()
                    self.session_ended_at_monotonic = self.monotonic()
                    self.sync_error = (
                        "本场采集已停止；请在赛后大屏查看回放与处理状态"
                        if getattr(self, "session_policy", "legacy") == "recording-v1"
                        else "USB 媒体等待分析任务发布后同步"
                    )
                    if getattr(self, "session_policy", "legacy") == "recording-v1":
                        return {
                            "recording_end": public_recording_capture(
                                self.record_store.get_recording_capture_end(self.session_id)
                            )
                        }
                except Exception as exc:
                    cause = exc.__cause__ or exc
                    if getattr(self, "session_policy", "legacy") == "legacy":
                        self.projection_source.fail(
                            f"local session end failure: {type(cause).__name__}: {cause}"
                        )
                    if isinstance(exc, ValueError):
                        raise
                    raise ValueError(str(exc)) from exc
            elif isinstance(command, str) and command.startswith("emergency_"):
                # Replay-only clock controls stay unavailable for live capture.
                super().mutate(payload)
                observer = getattr(self, "scenario_playback_observer", None)
                if observer is not None:
                    observer.invalidate()
            else:
                raise ValueError("unsupported live control")

    def resolve_command_confirmation(self, envelope):
        return resolve_command_confirmation(envelope)

    def reduce_live_command(self, match, envelope, **kwargs):
        self._require_projection_authority()
        if (
            envelope.command_type == "set_breaker"
            and match.rule_profile_id is not None
            and set(envelope.payload) == {"role"}
        ):
            # The legacy command can select a breaker but cannot freeze the
            # assistance/observation mode.  A profiled commercial match must
            # never enter the half-started state that shows "mode pending".
            raise ValueError("MATCH_SETTINGS_REQUIRED")
        if envelope.command_type == "end_session":
            if (
                getattr(self, "session_policy", "legacy") == "recording-v1"
                and self.projection_source.ready_to_end
            ):
                kwargs["has_unsettled_shot"] = False
            reduction = reduce_field_command(match, envelope, **kwargs)
            if not self.projection_source.ready_to_end:
                raise ValueError("LIVE_AUTHORITY_UNAVAILABLE")
            return reduction
        if envelope.command_type in {"continue_turn", "switch_turn"}:
            if self.projection_source.current().frame.payload["rule_status"] == "DEGRADED":
                raise ValueError("LIVE_AUTHORITY_UNAVAILABLE")
            if set(envelope.payload) != {"confirmation_ref"}:
                raise ValueError("TURN_CONFIRMATION_REQUIRED")
            confirmation_reference(envelope.payload["confirmation_ref"])
            envelope = replace(envelope, payload={})
        return reduce_field_command(match, envelope, **kwargs)

    def handle_command_application(self, envelope, application):
        with self.match_lock:
            self._require_projection_authority()
            if envelope.session_id != self.session_id:
                return
            self.last_applied_action = _describe_applied_command(envelope, application, self.match)
            if application.outcome != "applied":
                if envelope.command_type == "end_session" and self.projection_source.ready_to_end:
                    self.projection_source.fail(f"session end rejected: {application.result_code}")
                return
            observer = getattr(self, "scenario_playback_observer", None)
            if observer is not None and not (
                self.projection_lifecycle_closing or self.projection_lifecycle_closed
            ):
                observer.observe(
                    envelope, application, self.record_store.load_session(self.session_id)
                )
            if envelope.command_type in {"continue_turn", "switch_turn"}:
                reference = confirmation_reference(envelope.payload.get("confirmation_ref"))
                self.projection_source.refresh_confirmed_turn(reference)
            elif envelope.command_type == "end_session":
                self.projection_source.end_session()
                if self.session_ended_at_monotonic is None:
                    self.session_ended_at_monotonic = self.monotonic()
                self.sync_error = (
                    "录制已结束，未请求分析，尚未调度编码"
                    if getattr(self, "session_policy", "legacy") == "recording-v1"
                    else "USB 媒体等待分析任务发布后同步"
                )

    def _enqueue_session_sync(self, session_id=None):
        raise ValueError("USB media publication awaits the analysis worker")


class FieldAssistanceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        server_address,
        state: _ServerState | None = None,
        *,
        state_factory: Callable[[], _ServerState] | None = None,
    ) -> None:
        if (state is None) == (state_factory is None):
            raise ValueError("provide exactly one field state or state factory")
        self.field_state = state
        self._request_lock = threading.Lock()
        self._request_connections = {}
        self._requests_closing = threading.Event()
        super().__init__(server_address, _FieldAssistanceHandler)
        if state_factory is not None:
            try:
                # Own the bound listener throughout any durable startup transition.
                self.field_state = state_factory()
            except BaseException as primary:
                _cleanup_failed_startup(primary, (("listener", self.server_close),))
                raise

    def process_request(self, request, client_address):
        with self._request_lock:
            if self._requests_closing.is_set():
                self.shutdown_request(request)
                return
            request.settimeout(15)
            thread = threading.Thread(
                target=self.process_request_thread, args=(request, client_address), daemon=True
            )
            self._request_connections[request] = thread
            try:
                thread.start()
            except BaseException:
                self._request_connections.pop(request, None)
                self.shutdown_request(request)
                raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._request_lock:
                self._request_connections.pop(request, None)

    def handle_error(self, request, client_address):
        if self._requests_closing.is_set() and isinstance(sys.exception(), OSError):
            return
        super().handle_error(request, client_address)

    def service_actions(self) -> None:
        self.field_state.maintain_service()

    def server_close(self) -> None:
        failures = []
        self._requests_closing.set()
        listener_failure = None
        try:
            super().server_close()
        except BaseException as exc:
            listener_failure = exc
        if self.field_state is not None and hasattr(self.field_state, "match_lock"):
            with self.field_state.match_lock:
                self.field_state.projection_lifecycle_closing = True
        with self._request_lock:
            owned = tuple(self._request_connections.items())
        for connection, _ in owned:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError as exc:
                failures.append(exc)
        for cleanup in ("close_media_worker", "close_projection"):
            try:
                if self.field_state is not None:
                    getattr(self.field_state, cleanup)()
            except BaseException as exc:
                failures.append(exc)
        # Preserve the existing projection-before-listener error reporting order,
        # while stopping the listener first to prevent new accepted connections.
        if listener_failure is not None:
            failures.append(listener_failure)
        deadline = time.monotonic() + 5
        for _, thread in owned:
            if thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))
        if any(
            thread.is_alive() and thread is not threading.current_thread() for _, thread in owned
        ):
            failures.append(TimeoutError("owned HTTP requests did not terminate"))
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("field service cleanup failed", failures)


class _ReplayFrameStore:
    def __init__(self, path: Path) -> None:
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError("field replay presentation cannot be decoded")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next_frame_id = 0
        self._lock = threading.Lock()

    def jpeg(self, frame_id: int) -> bytes:
        if frame_id < 0 or frame_id >= self._frame_count:
            raise ValueError("replay frame is outside media")
        with self._lock:
            if frame_id != self._next_frame_id:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, image = self._capture.read()
            self._next_frame_id = frame_id + 1
            if not ok:
                raise ValueError("replay frame cannot be decoded")
            encoded, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not encoded:
                raise ValueError("replay frame cannot be encoded")
            return payload.tobytes()

    def close(self) -> None:
        with self._lock:
            self._capture.release()


class _FieldAssistanceHandler(FieldHTTPResponseMixin, BaseHTTPRequestHandler):
    server: FieldAssistanceHTTPServer

    def setup(self):
        super().setup()
        original = self.rfile
        try:
            # Keep the normal socket's 15s read/write timeout; only cancellation
            # polling is short. This reader does not retain a makefile socket ref.
            self.rfile = io.BufferedReader(
                _ClosingRequestReader(self.connection, self.server._requests_closing)
            )
        finally:
            original.close()

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _is_loopback_peer(self) -> bool:
        try:
            address = ipaddress.ip_address(self.client_address[0])
            mapped = getattr(address, "ipv4_mapped", None)
            return (mapped or address).is_loopback
        except (ValueError, TypeError, IndexError, AttributeError):
            return False

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlsplit(self.path)
        route = parsed_url.path
        if route == "/api/postgame-review":
            try:
                payload = self.server.field_state.postgame_review_payload(
                    kiosk=self._is_loopback_peer()
                )
            except ValueError:
                self._error(HTTPStatus.CONFLICT, "review is unavailable")
            else:
                self._send_json(HTTPStatus.OK, payload)
            return
        clip_match = _POSTGAME_MEDIA_PATTERN.fullmatch(route)
        if clip_match:
            self._serve_postgame_media(clip_match.group(1), parsed_url.query)
            return
        if route == "/screen":
            self._send_bytes(
                HTTPStatus.OK,
                render_screen_page().encode(),
                content_type="text/html; charset=utf-8",
            )
            return
        if route == "/tablet":
            self._send_bytes(
                HTTPStatus.OK,
                render_tablet_page(
                    live=isinstance(self.server.field_state, _LiveServerState)
                ).encode(),
                content_type="text/html; charset=utf-8",
            )
            return
        if route in {"/api/screen", "/api/tablet"}:
            try:
                payload = self.server.field_state.payload()
            except ValueError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
            else:
                self._send_json(HTTPStatus.OK, payload)
            return
        if route == "/api/table-code.png":
            try:
                payload, content_type = self.server.field_state.join_qr_image()
            except ValueError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
            else:
                self._send_bytes(HTTPStatus.OK, payload, content_type=content_type)
            return
        local_fact_routes = {
            "/api/local/session": self.server.field_state.session_payload,
            "/api/local/shots": self.server.field_state.shots_payload,
            "/api/local/report": self.server.field_state.report_payload,
        }
        if route in local_fact_routes:
            provided = self.headers.get("X-Control-Token", "")
            expected = self.server.field_state.control_token
            if not hmac.compare_digest(provided, expected):
                self._error(HTTPStatus.FORBIDDEN, "control token rejected")
                return
            try:
                payload = local_fact_routes[route]()
            except ValueError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
            else:
                self._send_json(HTTPStatus.OK, payload)
            return
        if route == "/media/presentation.mp4":
            self._serve_media()
            return
        if route == "/media/live.mjpg":
            self._serve_live_mjpeg(parsed_url.query)
            return
        frame_match = _FRAME_PATTERN.fullmatch(route)
        if frame_match is not None:
            try:
                payload = self.server.field_state.frame_store.jpeg(int(frame_match.group(1)))
            except ValueError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
            else:
                self._send_bytes(
                    HTTPStatus.OK,
                    payload,
                    content_type="image/jpeg",
                )
            return
        self._error(HTTPStatus.NOT_FOUND, "route not found")

    def _serve_live_mjpeg(self, query: str) -> None:
        state = self.server.field_state
        if not isinstance(state, _LiveServerState):
            self._error(HTTPStatus.NOT_FOUND, "live camera stream is unavailable")
            return
        source = state.projection_source
        requested_epoch = parse_qs(query).get("epoch", [""])[0]
        if not requested_epoch or requested_epoch != getattr(source, "source_epoch_id", None):
            self._error(HTTPStatus.NOT_FOUND, "live camera epoch is unavailable")
            return
        snapshot = getattr(source, "mjpeg_snapshot", None)
        if snapshot is None:
            self._error(HTTPStatus.NOT_FOUND, "live camera stream is unavailable")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=billiards-frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True
        last_frame_id = None
        try:
            while source is state.projection_source and not self.server._requests_closing.is_set():
                current = snapshot()
                if current is None:
                    time.sleep(0.005)
                    continue
                frame_id, jpeg = current
                if frame_id == last_frame_id:
                    time.sleep(0.005)
                    continue
                part = (
                    b"--billiards-frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.write(part)
                self.wfile.flush()
                last_frame_id = frame_id
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError, ValueError):
            return

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route in {"/api/postgame-review/save", "/api/postgame-review/complete"}:
            self._postgame_write(complete=route.endswith("/complete"))
            return
        if route != "/api/control":
            self._error(HTTPStatus.NOT_FOUND, "route not found")
            return
        provided = self.headers.get("X-Control-Token", "")
        expected = self.server.field_state.control_token
        if not hmac.compare_digest(provided, expected):
            self._error(HTTPStatus.FORBIDDEN, "control token rejected")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "content length is invalid")
            return
        if length < 0 or length > _MAX_CONTROL_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "control body is too large")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            result = self.server.field_state.mutate(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json(
            HTTPStatus.OK, self.server.field_state.payload() if result is None else result
        )

    def _postgame_write(self, *, complete: bool) -> None:
        if not self._is_loopback_peer():
            self._error(HTTPStatus.FORBIDDEN, "kiosk peer rejected")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[
            0
        ].strip().lower() != "application/json" or self.headers.get("Transfer-Encoding"):
            self._error(HTTPStatus.BAD_REQUEST, "JSON body required")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "content length is invalid")
            return
        if length <= 0 or length > _MAX_CONTROL_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "review body is too large")
            return
        try:

            def unique_object(pairs):
                result = dict(pairs)
                if len(result) != len(pairs):
                    raise ValueError("duplicate identity field")
                return result

            payload = json.loads(self.rfile.read(length), object_pairs_hook=unique_object)
            keys = {"session_id", "postgame_manifest_sha256"}
            if not complete:
                keys |= {"request_id", "shot_id"}
            if (
                not isinstance(payload, dict)
                or set(payload) != keys
                or any(
                    not isinstance(value, str) or not value.strip() or len(value) > 256
                    for value in payload.values()
                )
            ):
                raise ValueError("review identity is invalid")
            if not complete:
                validate_save_request_id(payload["request_id"])
        except (ValueError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "review body is invalid")
            return
        try:
            result = self.server.field_state.mutate_postgame_review(
                payload, self.headers.get("X-Kiosk-Nonce", ""), complete=complete
            )
        except PermissionError:
            self._error(HTTPStatus.FORBIDDEN, "kiosk nonce rejected")
        except ValueError:
            self._error(HTTPStatus.CONFLICT, "review identity conflicts")
        else:
            self._send_json(HTTPStatus.OK, result)

    def _serve_postgame_media(self, clip_id: str, query: str) -> None:
        state = self.server.field_state
        try:
            params = parse_qs(query, strict_parsing=True)
            if set(params) != {"manifest"} or len(params["manifest"]) != 1:
                raise ValueError
            digest = params["manifest"][0]
            clip = state.review_clip(clip_id, digest)
            original_root = state.record_store.artifact_path_for_session(clip.session_id)
            with session_media_lock(
                original_root,
                clip.session_id,
                exclusive=False,
                timeout_seconds=0,
            ):
                state.review_clip(clip_id, digest)
                with _open_review_media(
                    original_root, clip, continue_check=state._check_review_running
                ) as source:
                    # Hashing is outside match_lock; recheck selection before response.
                    state.review_clip(clip_id, digest)

                    def still_allowed():
                        # Fresh per-chunk revocation, no file I/O or projection lock.
                        try:
                            return (
                                not state.projection_lifecycle_closing
                                and not state.record_store.media_encoding_state(
                                    clip.session_id, clip.shot_id
                                )["banned"]
                            )
                        except (OSError, ValueError, sqlite3.Error):
                            return False

                    self._stream_media(source, clip.byte_size, continue_check=still_allowed)
        except (OSError, ValueError, RecordingPlaybackError):
            self._error(HTTPStatus.NOT_FOUND, "review media is unavailable")

    def _serve_media(self) -> None:
        media = self.server.field_state.artifact_path / "presentation.mp4"
        if not media.is_file():
            self._error(HTTPStatus.NOT_FOUND, "presentation media is unavailable")
            return
        size = media.stat().st_size
        with media.open("rb") as source:
            self._stream_media(source, size)


def _verified_start_events(
    inputs: Mainline1ReplayInputs | None,
    replay_manifest: dict[str, object],
    timeline: ReplayTimeline,
    index: ShotTimelineIndex,
) -> tuple[tuple[tuple[int, str], ...], str]:
    """One startup-only check of explicitly located immutable source evidence.

    Optional attribution evidence cannot break otherwise valid base replay. The
    field replay's own manifest is still checked by its existing mandatory gate.
    Observed motion is NOT reviewed legality, pot completeness or publication.
    """
    if inputs is None:
        return (), "SOURCE_INPUTS_MISSING"
    try:
        verified = load_mainline1_replay_inputs(
            artifact_path=inputs.artifact_path, video_path=inputs.video_path
        )
        if verified != inputs:
            return (), "SOURCE_LINK_MISMATCH"
        links = {
            "source_mainline1_artifact_id": verified.artifact_id,
            "video_sha256": hashlib.sha256(verified.video_path.read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256(verified.events_path.read_bytes()).hexdigest(),
            "shot_windows_sha256": hashlib.sha256(
                verified.shot_windows_path.read_bytes()
            ).hexdigest(),
            "observations_sha256": hashlib.sha256(
                verified.observations_path.read_bytes()
            ).hexdigest(),
        }
        if any(replay_manifest.get(key) != value for key, value in links.items()):
            return (), "SOURCE_LINK_MISMATCH"
        windows = json.loads(verified.shot_windows_path.read_text())["shots"]
        events = json.loads(verified.events_path.read_text())["events"]
        frames = json.loads(verified.observations_path.read_text())["frames"]
        if not all(isinstance(rows, list) for rows in (windows, events, frames)):
            return (), "SOURCE_CONTENT_INVALID"
        frame_times = {row["frame_id"]: row["timestamp_ms"] for row in frames}
        timeline_frames = {frame.timestamp_ms: frame for frame in timeline.frames}
        if len(frame_times) != len(frames) or len({row["event_id"] for row in events}) != len(
            events
        ):
            return (), "SOURCE_CONTENT_INVALID"
        result = []
        for start in index.boundaries:
            if start.kind is not ShotBoundaryKind.STARTED or start.sequence > len(windows):
                continue
            window = windows[start.sequence - 1]
            # Exact window order + shot identity + decision frame, not a search
            # for any event with a matching timestamp from a different shot.
            matches = [
                e
                for e in events
                if e["shot_id"] == window["shot_id"] and e["event_type"] == "shot_start"
            ]
            if len(matches) != 1:
                continue
            event = matches[0]
            frame = timeline_frames.get(start.timestamp_ms)
            if (
                event["status"] == "observed"
                and frame is not None
                and event["decision_frame_id"] == frame.frame_id
                and frame_times.get(frame.frame_id) == start.timestamp_ms
                and window["start_frame_id"] <= frame.frame_id <= window["end_frame_id"]
                and frame.payload.get("shot_sequence") == start.sequence - 1
                and "SHOT_STARTED" in frame.payload.get("effects", ())
            ):
                result.append((start.sequence, event["event_id"]))
        return tuple(result), "SOURCE_START_VERIFIED" if result else "SOURCE_START_UNVERIFIED"
    except (OSError, ValueError, KeyError, TypeError):
        return (), "SOURCE_CONTENT_INVALID"


def build_field_assistance_server(
    *,
    artifact_path: Path,
    host: str,
    port: int,
    control_token: str,
    monotonic: Callable[[], float] = time.monotonic,
    duration_probe: Callable[[Path], int] = probe_video_duration_ms,
    state_db_path: Path | None = None,
    cloud_gateway: FieldCloudGateway | None = None,
    join_qr_asset_path: Path | None = None,
    sync_wakeup: Callable[[], None] | None = None,
    mainline1_inputs: Mainline1ReplayInputs | None = None,
) -> FieldAssistanceHTTPServer:
    """Verify immutable evidence, then bind a local synchronized HTTP server."""

    if not control_token or len(control_token) > 512:
        raise ValueError("control token length is invalid")
    artifact = artifact_path.resolve()
    manifest = verify_field_route_replay_artifact(artifact)
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("field replay artifact ID is invalid")
    media = artifact / "presentation.mp4"
    timeline_path = artifact / "timeline.json"
    if media.is_symlink() or not media.is_file():
        raise ValueError("field replay presentation is missing or unsafe")
    duration_ms = int(duration_probe(media))
    timeline = load_replay_timeline(timeline_path, duration_ms=duration_ms)
    shot_index = ShotTimelineIndex.from_timeline(timeline)
    verified_starts, source_reason = _verified_start_events(
        mainline1_inputs, manifest, timeline, shot_index
    )
    recommendations = artifact / "recommendations.json"
    recommendation_power = (
        load_recommendation_power(recommendations) if recommendations.is_file() else {}
    )
    record_store = SQLiteFieldRecordStore(
        state_db_path or (artifact.parent / "field-state.sqlite3")
    )
    recoverable = record_store.list_ended_sessions_without_sync(artifact_id)
    session = (
        recoverable[-1]
        if recoverable
        else record_store.open_or_resume_session(artifact_id=artifact_id)
    )
    record_store.bind_artifact_path(artifact_id, artifact)
    state = _ServerState(
        artifact_path=artifact,
        control_token=control_token,
        replay_clock=ReplayClock(timeline, monotonic=monotonic),
        match=session.match,
        match_lock=threading.RLock(),
        frame_store=_ReplayFrameStore(media),
        record_store=record_store,
        session_id=session.session_id,
        artifact_id=artifact_id,
        shot_index=shot_index,
        recommendation_power=recommendation_power,
        duration_ms=duration_ms,
        source_mainline1_artifact_id=manifest.get("source_mainline1_artifact_id"),
        verified_start_events=verified_starts,
        source_validation_reason=source_reason,
        join_qr_asset=(
            None if join_qr_asset_path is None else _load_join_qr_asset(join_qr_asset_path)
        ),
        join_coordinator=(
            None
            if cloud_gateway is None
            else FieldJoinCoordinator(session_id=session.session_id, gateway=cloud_gateway)
        ),
        monotonic=monotonic,
        sync_wakeup=sync_wakeup,
        pending_cloud_closes={
            session_id: 0 for session_id in record_store.list_pending_cloud_close_session_ids()
        },
    )
    state._retry_unsynced_sessions()
    return FieldAssistanceHTTPServer((host, port), state)


def build_live_field_assistance_server(
    *,
    artifact_path: Path,
    state_db_path: Path,
    host: str,
    port: int,
    control_token: str,
    source_epoch_factory: Callable[[], str],
    calibration_id: str | None,
    runtime_factory: Callable | None,
    join_qr_asset_path: Path | None = None,
    cloud_gateway: FieldCloudGateway | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sync_wakeup: Callable[[], None] | None = None,
    rule_profile_id: str | None = None,
    session_policy: str = "legacy",
    fresh_session_on_start: bool = False,
    managed_startup: ManagedStartupCoordinator | None = None,
    runtime_preparer: Callable | None = None,
    recording_recovery_preparer: Callable | None = None,
    runtime_resource_inspector: Callable | None = None,
    scenario_playback_gate: PlaybackGate | None = None,
) -> FieldAssistanceHTTPServer:
    """Bind before durable startup, then prepare and open this invocation's source.

    ``runtime_preparer`` returns (calibration_id, runtime_factory, prepared_backend)
    after managed preparation. Its backend is cancelled if construction fails;
    successful capture remains owned by the projection's shutdown lifecycle.
    """
    if session_policy not in {"legacy", "recording-v1"} or (
        session_policy == "recording-v1" and rule_profile_id is not None
    ):
        raise ValueError("invalid explicit session recording policy")
    if not control_token or len(control_token) > 512:
        raise ValueError("control token length is invalid")
    if scenario_playback_gate is not None and (
        managed_startup is None
        or not fresh_session_on_start
        or scenario_playback_gate.is_released()
    ):
        raise ValueError("scenario playback gate requires a fresh managed blackbox scene")
    artifact = artifact_path.resolve()
    artifact_id = managed_live_artifact_id(str(artifact))
    if runtime_resource_inspector is not None and managed_startup is None:
        raise ValueError("managed runtime resource inspection requires managed startup")

    def build_state():
        resource_spec = None if runtime_resource_inspector is None else runtime_resource_inspector()
        if runtime_resource_inspector is not None and resource_spec is None:
            raise ValueError("managed resource inspector returned no specification")
        store = SQLiteFieldRecordStore(state_db_path)
        if session_policy == "recording-v1":
            candidates = store.recording_interruption_candidates(artifact_id)
            if candidates:
                if managed_startup is not None:
                    retained_request = store.get_active_managed_open_request(managed_startup.scope)
                    if retained_request is None or retained_request.phase != "materialized":
                        raise ValueError("interrupted managed scene has no original binding")
                    retained_session = store.validate_managed_local_session(
                        managed_startup.scope, retained_request.open_request_id
                    )
                    if any(sid != retained_session.session_id for sid, _ in candidates):
                        raise ValueError("interrupted recording is outside managed scope")
                recovery_factory = runtime_factory
                if recovery_factory is None and recording_recovery_preparer is not None:
                    recovery_factory = recording_recovery_preparer()
                recover_interrupted = getattr(
                    recovery_factory, "recover_interrupted_recordings", None
                )
                if recover_interrupted is None:
                    raise ValueError(
                        "interrupted recording requires explicit disk recovery preparation"
                    )
                recover_interrupted(store, artifact_id)
            store.resume_recording_capture_intents(artifact_id)
        retained = (
            None
            if managed_startup is None
            else store.get_active_managed_open_request(managed_startup.scope)
        )
        if managed_startup is None:
            session = store.open_or_resume_session(
                artifact_id=artifact_id,
                rule_profile_id=rule_profile_id,
                fresh_session_on_start=fresh_session_on_start,
                session_policy=session_policy,
            )
        else:
            session = managed_startup.prepare(
                store,
                artifact_id=artifact_id,
                rule_profile_id=rule_profile_id,
                policy="blackbox_fresh" if fresh_session_on_start else "usb_resume",
                resource_spec=resource_spec,
                **({} if session_policy == "legacy" else {"session_policy": session_policy}),
            )
        binding = None
        resource_spec_record = None
        resource_session_ids = None
        if managed_startup is not None:
            request = store.get_managed_open_request_for_session(
                managed_startup.scope, session.session_id
            )
            store.validate_managed_local_session(managed_startup.scope, request.open_request_id)
            binding = request.binding
            if binding is None or binding.status != "open":
                raise ValueError("managed startup requires verified open binding")
            if runtime_resource_inspector is not None:
                resource_spec_record = store.get_managed_resource_spec_for_session(
                    managed_startup.scope, session.session_id
                )
                if resource_spec_record.spec != resource_spec:
                    raise ValueError("managed startup resource specification changed")
            resource_session_ids = (
                ()
                if retained is not None
                and retained.phase == "materialized"
                and retained.binding.session_id == session.session_id
                else (session.session_id,)
            )
        interrupted_session_id = None
        if session_policy == "recording-v1":
            store.require_recording_recovery_complete(session.session_id)
        persisted_shots = store.list_shots(session.session_id)
        unsettled = tuple(shot for shot in persisted_shots if shot.settled_ms is None)
        if unsettled:
            if managed_startup is not None:
                raise ValueError("managed startup returned an unsettled session")
            if len(unsettled) != 1 or unsettled[0] != persisted_shots[-1]:
                raise ValueError("persisted shot recovery is ambiguous")
            interrupted_session_id = session.session_id
            store.end_session(interrupted_session_id)
            session = store.open_or_resume_session(
                artifact_id=artifact_id,
                rule_profile_id=rule_profile_id,
                session_policy=session_policy,
            )
        artifact.mkdir(parents=True, exist_ok=True)
        store.bind_artifact_path(artifact_id, artifact)
        prepared_source = None
        prepared_calibration_id, prepared_runtime_factory = calibration_id, runtime_factory
        try:
            if resource_spec_record is not None:
                verify_managed_runtime_resources(resource_spec_record.spec, purpose="capture")
            if runtime_preparer is not None:
                prepared_calibration_id, prepared_runtime_factory, prepared_source = (
                    runtime_preparer()
                    if runtime_resource_inspector is None
                    else runtime_preparer(resource_spec=resource_spec_record.spec)
                )
            if resource_spec_record is not None:
                verify_managed_runtime_resources(resource_spec_record.spec, purpose="capture")
                if getattr(prepared_runtime_factory, "managed_resource_spec_sha256", None) != (
                    resource_spec_record.spec_sha256
                ):
                    raise ValueError("prepared managed runtime resource specification changed")
        except BaseException as primary:
            if prepared_source is not None:
                _cleanup_failed_startup(primary, (("prepared capture", prepared_source.cancel),))
            raise
        lock = threading.RLock()
        source_epochs: set[str] = set()
        active_source_epoch_id: str | None = None

        def source_factory(session_id, *, recover_capture: bool = False):
            nonlocal active_source_epoch_id
            if active_source_epoch_id is None or recover_capture:
                source_epoch_id = source_epoch_factory()
                if (
                    not isinstance(source_epoch_id, str)
                    or not source_epoch_id.strip()
                    or source_epoch_id in source_epochs
                ):
                    raise ValueError(
                        "each physical live capture open requires a fresh source epoch"
                    )
                source_epochs.add(source_epoch_id)
                if resource_spec_record is not None:
                    store.register_managed_capture_epoch(
                        managed_startup.scope, request.open_request_id, source_epoch_id
                    )
                active_source_epoch_id = source_epoch_id
            else:
                source_epoch_id = active_source_epoch_id
            return USBLiveProjectionSource(
                store=store,
                session_id=session_id,
                source_epoch_id=source_epoch_id,
                calibration_id=prepared_calibration_id,
                runtime_factory=prepared_runtime_factory,
                match_lock=lock,
            )

        source = None
        try:
            source = source_factory(session.session_id)
            media_jobs_path = getattr(prepared_runtime_factory, "jobs_db_path", None)
            media_job_store = (
                SQLiteShotAnalysisJobStore(media_jobs_path)
                if isinstance(media_jobs_path, Path)
                else None
            )
            state = _LiveServerState(
                artifact_path=artifact,
                control_token=control_token,
                replay_clock=None,
                match=session.match,
                match_lock=lock,
                frame_store=source,
                record_store=store,
                session_id=session.session_id,
                artifact_id=artifact_id,
                shot_index=ShotTimelineIndex(()),
                recommendation_power={},
                duration_ms=0,
                projection_source=source,
                media_job_store=media_job_store,
                monotonic=monotonic,
                sync_wakeup=sync_wakeup,
                managed_session_scope=None if managed_startup is None else managed_startup.scope,
                cloud_session_status="unavailable" if binding is None else binding.status,
                resource_session_ids=resource_session_ids,
                join_qr_asset=None
                if join_qr_asset_path is None
                else _load_join_qr_asset(join_qr_asset_path),
                join_coordinator=None
                if cloud_gateway is None
                else FieldJoinCoordinator(session_id=session.session_id, gateway=cloud_gateway),
                pending_cloud_closes={
                    session_id: 0 for session_id in store.list_pending_cloud_close_session_ids()
                },
            )
        except BaseException as primary:
            cleanups = []
            if source is not None:
                cleanups.append(("live projection", source.shutdown))
            if prepared_source is not None:
                cleanups.append(("prepared capture", prepared_source.cancel))
            _cleanup_failed_startup(primary, cleanups)
            raise
        state.live_source_factory = source_factory
        if session_policy == "recording-v1":
            state._recording_recoveries = [
                RecordingRecovery(
                    store=store,
                    capture=capture,
                    runtime_factory=prepared_runtime_factory,
                )
                for capture in store.pending_recording_captures(artifact_id)
            ]
        state.managed_resource_spec_record = resource_spec_record
        state.scenario_playback_observer = (
            None
            if scenario_playback_gate is None
            else AuthenticatedScenarioPlayback(session.session_id, scenario_playback_gate)
        )
        state.managed_blackbox = managed_startup is not None and fresh_session_on_start
        state.rule_profile_id = rule_profile_id
        state.session_policy = session_policy
        state.capture_recovery_attempts = 0
        state.capture_recovery_not_before = 0.0
        if interrupted_session_id is not None:
            state.last_applied_action = "上次异常中断的对局已保存，请重新扫码加入新对局"
        return state

    server = FieldAssistanceHTTPServer((host, port), state_factory=build_state)
    try:
        server.field_state.projection_source.start()
        for recovery in server.field_state._recording_recoveries:
            recovery.start()
        return server
    except BaseException as primary:
        _cleanup_failed_startup(primary, (("service/listener", server.server_close),))
        raise


def _report_startup_cleanup_failure(primary, owner, exc):
    failures = list(getattr(primary, "_field_cleanup_failures", ()))
    if not any(label == owner for label, _ in failures):
        failures.append((owner, exc))
        primary.add_note(f"field startup cleanup failed ({owner}): {type(exc).__name__}: {exc}")
    primary._field_cleanup_failures = tuple(failures)
    print(f"Field startup cleanup failed ({owner}):", file=sys.stderr, flush=True)
    traceback.print_exception(exc, file=sys.stderr, chain=False)


def _cleanup_failed_startup(primary, cleanups):
    """Preserve startup failure and retain only original pending cleanup owners."""
    pending = list(getattr(primary, "_field_pending_cleanup", ()))
    for owner, cleanup in cleanups:
        try:
            cleanup()
        except BaseException as exc:
            _report_startup_cleanup_failure(primary, owner, exc)
            if not any(action == cleanup for _, action in pending):
                pending.append((owner, cleanup))
    primary._field_pending_cleanup = tuple(pending)


def _retry_failed_startup_cleanup(primary):
    """Explicitly drain the same retained owners, without any new allocation."""
    if not hasattr(primary, "_field_pending_cleanup"):
        return
    pending = []
    failures = []
    for owner, cleanup in getattr(primary, "_field_pending_cleanup", ()):
        try:
            cleanup()
        except BaseException as exc:
            pending.append((owner, cleanup))
            failures.append(exc)
            _report_startup_cleanup_failure(primary, owner, exc)
    primary._field_pending_cleanup = tuple(pending)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("field startup cleanup still pending", failures)
