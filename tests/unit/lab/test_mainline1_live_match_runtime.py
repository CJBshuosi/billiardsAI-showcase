from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.lab import mainline1_product_realtime_runtime as product_runtime
from billiards_ai.lab.events.models import BallState, TrackedFrame
from billiards_ai.lab.mainline1_live_window import LiveShotBoundary, LiveShotBoundaryKind
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeEvent,
    ProductRealtimeRuntime,
)
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_refinement import (
    FactCoverageVerdict,
    RealtimeEvidenceCoverage,
)
from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.models import ObservationFrame, ShotWindow


def test_waiting_players_motion_never_reaches_visual_shot_detector(monkeypatch):
    from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEventChannel

    ready = False
    seen = []
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._session_policy = "legacy"
    runtime._event_channel = ProductRealtimeEventChannel(shot_admission=lambda: ready)
    runtime._visual_boundary_detector = SimpleNamespace(push=lambda **kw: seen.append(kw) or ())
    runtime._visual_motion_config = object()
    runtime._calibration = object()
    runtime._visual_previous_bgr = None
    monkeypatch.setattr(product_runtime, "table_visual_motion_detected", lambda *args: True)
    for index in range(5):
        runtime._handle_visual_boundary(_frame(index), [])
    assert seen == []
    ready = True
    runtime._handle_visual_boundary(_frame(5), [])
    assert seen == []  # Establish a fresh baseline after confirmation.
    runtime._handle_visual_boundary(_frame(6), [])
    assert [row["frame_id"] for row in seen] == [6]


class _RecordingRing:
    def __init__(self, frozen: object, operations: list[str]) -> None:
        self._frozen = frozen
        self._operations = operations
        self.released = False

    def freeze_shot(self) -> object:
        self._operations.append("freeze")
        return self._frozen

    def release_shot(self) -> None:
        self._operations.append("release")
        self.released = True


class _RecordingStore:
    def __init__(self, operations: list[str], *, delay_seconds: float = 0.0) -> None:
        self._operations = operations
        self._delay_seconds = delay_seconds
        self.jobs: list[object] = []

    def enqueue(self, job: object) -> SimpleNamespace:
        time.sleep(self._delay_seconds)
        self._operations.append("enqueue")
        self.jobs.append(job)
        return SimpleNamespace(job=job, status="pending", created=True)


class _RefinedShot:
    publishable = True
    shot_abstention_reason = None
    fact_coverage = {
        name: SimpleNamespace(status="observed", reason=None)
        for name in ("first_contact", "cushion_contacts", "pots")
    }
    coverage = RealtimeEvidenceCoverage(
        source_frame_ids=(0, 1, 2, 3, 4),
        anchor_frame_ids=(0, 4),
        maximum_anchor_gap_ms=66.0,
        uncertain_frame_ids=(),
        segment_sha256={"segment-0": "a" * 64},
    )

    @staticmethod
    def observations() -> tuple[ObservationFrame, ...]:
        return (ObservationFrame(frame_id=3, timestamp_ms=50, observations=()),)

    @staticmethod
    def rule_frames() -> tuple[TrackedFrame, ...]:
        return (
            TrackedFrame(
                frame_id=3,
                timestamp_ms=50,
                balls=(BallState("cue", 500, 500, 0.95, track_id="cue-track"),),
            ),
        )

    @staticmethod
    def provenance() -> dict[str, object]:
        return {}


class _Refiner:
    def __init__(self, *, maximum_shot_frames: int) -> None:
        assert maximum_shot_frames == 12

    def refine(self, **kwargs: object) -> _RefinedShot:
        return _RefinedShot()


def _frame(frame_id: int) -> CaptureFrame:
    image = np.full((12, 16, 3), frame_id, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 16_666_667,
        width=16,
        height=12,
        mjpeg=encoded.tobytes(),
    )


def _frozen_interval(root: Path):
    ring = MjpegSegmentRing(
        root,
        expected_fps=4,
        pre_roll_frames=1,
        post_roll_frames=1,
        maximum_shot_frames=12,
        maximum_bytes=1024 * 1024,
    )
    for frame_id in range(5):
        ring.append(_frame(frame_id))
        if frame_id == 1:
            ring.begin_shot(frame_id)
        if frame_id == 3:
            ring.settle_shot(frame_id)
    return ring.freeze_shot()


def _runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: object | None = None,
) -> tuple[ProductRealtimeRuntime, _RecordingRing, list[str]]:
    operations: list[str] = []
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._session_policy = "legacy"
    runtime._pending_window = ShotWindow(
        shot_id="shot-00000001",
        start_frame_id=1,
        end_frame_id=3,
        before_frame_id=0,
        after_frame_id=3,
        start_ms=17,
        end_ms=50,
        complete=True,
    )
    ring = _RecordingRing(_frozen_interval(tmp_path / "ring"), operations)
    runtime._ring = ring
    runtime._replay = SimpleNamespace(wait_until_preprocessed=lambda *args, **kwargs: True)
    runtime._accept_completed_anchors = lambda: None
    runtime._handle_tracking = lambda events: None
    runtime._tracked_by_frame = {}
    runtime._accepted_anchors = []
    runtime._config = SimpleNamespace(
        post_roll_frames=1,
        background_shutdown_timeout_seconds=1.0,
        maximum_shot_frames=12,
        stop_after_terminal_shots=None,
    )
    runtime._job_store = store or _RecordingStore(operations)
    runtime._job_media_root = tmp_path / "analysis-media"
    runtime._quarantine_root = tmp_path / "analysis-quarantine"
    runtime._session_id = "session-0001"
    runtime._capture_setup_id = "setup-1"
    runtime._calibration_artifact_id = "calibration-1"
    runtime._config_sha256 = "1" * 64
    runtime._identity_training_package_id = None
    runtime._identity_training_package_sha256 = None
    runtime._shot_config = SimpleNamespace(
        event=ShotLocalEventConfig(
            first_contact_distance_mm=60,
            first_contact_ambiguity_mm=3,
            cue_motion_min_mm=5,
            contact_time_tolerance_ms=20,
            cushion_distance_mm=40,
            pocket_distance_mm=90,
            pot_missing_frames=2,
            after_match_distance_mm=50,
            table_length_mm=2540,
            table_width_mm=1270,
        )
    )
    runtime._detector = SimpleNamespace(
        package_identity=SimpleNamespace(package_id="model-package-1")
    )
    runtime._retain_unsafe_interval = False
    runtime._event_channel = None
    runtime._event_channel_failed = False
    monkeypatch.setattr(product_runtime, "RealtimeShotRefiner", _Refiner)
    return runtime, ring, operations


def test_finalize_materializes_enqueues_emits_live_settle_then_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, ring, operations = _runtime(tmp_path, monkeypatch)
    original_materialize = product_runtime.materialize_frozen_interval

    def recording_materialize(interval: object, destination: Path) -> object:
        operations.append("materialize")
        return original_materialize(interval, destination)

    monkeypatch.setattr(product_runtime, "materialize_frozen_interval", recording_materialize)
    events = [ProductRealtimeEvent(kind="SHOT_STARTED", shot_id="shot-00000001")]

    runtime._finalize_if_ready(_frame(4), events, threading.Event())

    assert [event.kind for event in events] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
        "SHOT_ENQUEUED",
    ]
    assert operations == ["freeze", "materialize", "enqueue", "release"]
    settled = events[1].settled
    assert settled is not None
    assert settled.shot_id == "shot-00000001"
    assert settled.source_epoch_id == "epoch-1"
    assert settled.start_frame_id == 1
    assert settled.settled_frame_id == 3
    assert settled.settled_at_ns == 50_000_000
    assert settled.observations == _RefinedShot.observations()
    assert settled.rule_frames == _RefinedShot.rule_frames()
    assert settled.rule_coverage == _RefinedShot.coverage
    assert settled.event_config == runtime._shot_config.event
    expected_config_digest = hashlib.sha256(
        json.dumps(
            asdict(runtime._shot_config.event),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert settled.event_config_digest == expected_config_digest
    assert settled.job_id == runtime._job_store.jobs[0].job_id
    assert events[2].settled is None
    assert ring.released is True
    assert runtime._pending_window is None


def test_first_shot_diagnostics_reuses_refinement_and_survives_later_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, ring, operations = _runtime(tmp_path, monkeypatch)
    frame_reads = []
    original_frames = ring._frozen.frames

    def frames():
        frame_reads.append(True)
        return original_frames()

    # Refiner is the existing fake; only materialization should read source frames.
    monkeypatch.setattr(type(ring._frozen), "frames", lambda self: frames())
    calls = []
    original = product_runtime.diagnose_realtime_shot

    def diagnose(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(product_runtime, "diagnose_realtime_shot", diagnose)
    events = []
    runtime._finalize_if_ready(_frame(4), events, threading.Event())
    first = runtime._first_shot_diagnostics
    assert first.status == "OBSERVED_METADATA"
    assert first.source.count == 5 and first.tracked.count == 0
    assert len(calls) == 1 and len(frame_reads) == 1
    settled = next(event.settled for event in events if event.kind == "SHOT_SETTLED")
    runtime._replay.tracker_health = "later tracker replacement"
    runtime._record_first_shot_diagnostics(
        frozen=ring._frozen,
        tracked=(),
        anchors=(),
        settled=settled,
    )
    assert runtime._first_shot_diagnostics is first
    assert len(calls) == 1 and len(frame_reads) == 1
    assert operations == ["freeze", "enqueue", "release"]


def test_diagnostic_failure_preserves_settled_event_and_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, ring, operations = _runtime(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise ValueError("private path must not escape into diagnostics")

    monkeypatch.setattr(product_runtime, "diagnose_realtime_shot", fail)
    events = []
    runtime._finalize_if_ready(_frame(4), events, threading.Event())
    assert [event.kind for event in events] == ["SHOT_SETTLED", "SHOT_ENQUEUED"]
    assert asdict(runtime._first_shot_diagnostics)["error_type"] == "ValueError"
    assert "private path" not in str(asdict(runtime._first_shot_diagnostics))
    assert events[0].settled.observations == _RefinedShot.observations()
    assert events[0].settled.rule_frames == _RefinedShot.rule_frames()
    assert events[0].settled.rule_coverage == _RefinedShot.coverage
    assert runtime._job_store.jobs[0].job_id == events[0].settled.job_id
    assert ring.released
    assert operations == ["freeze", "enqueue", "release"]


def test_freeze_failure_discards_incomplete_shot_and_releases_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, ring, operations = _runtime(tmp_path, monkeypatch)

    def fail_freeze() -> object:
        operations.append("freeze")
        raise product_runtime.RealtimeRingError("incomplete evidence")

    ring.freeze_shot = fail_freeze
    events = [ProductRealtimeEvent(kind="SHOT_STARTED", shot_id="shot-00000001")]

    runtime._finalize_if_ready(_frame(4), events, threading.Event())

    assert [event.kind for event in events] == ["SHOT_STARTED", "SHOT_INCOMPLETE"]
    assert operations == ["freeze", "release"]
    assert ring.released is True
    assert runtime._pending_window is None
    assert not runtime._job_media_root.exists()
    assert not runtime._quarantine_root.exists()


def test_new_shot_supersedes_prior_pending_post_roll_without_stopping_capture(
    tmp_path: Path,
) -> None:
    """Catches a quick next shot colliding with the prior shot's 120-frame post-roll."""
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._session_policy = "legacy"
    ring = MjpegSegmentRing(
        tmp_path / "overlap-ring",
        expected_fps=60,
        pre_roll_frames=0,
        post_roll_frames=120,
        maximum_shot_frames=1_800,
        maximum_bytes=16 * 1024 * 1024,
    )
    for frame_id in range(131):
        ring.append(_frame(frame_id))
    ring.begin_shot(10)
    ring.settle_shot(100)
    runtime._ring = ring
    runtime._pending_window = ShotWindow(
        shot_id="shot-0001",
        start_frame_id=10,
        end_frame_id=100,
        before_frame_id=9,
        after_frame_id=100,
        start_ms=167,
        end_ms=1_667,
        complete=True,
    )
    runtime._boundary_ledger = None
    runtime._event_channel = None
    runtime._event_channel_failed = False
    events: list[ProductRealtimeEvent] = []
    second = ShotWindow(
        shot_id="shot-0002",
        start_frame_id=130,
        end_frame_id=130,
        before_frame_id=129,
        after_frame_id=None,
        start_ms=2_167,
        end_ms=2_167,
        complete=False,
    )

    runtime._apply_boundary(
        LiveShotBoundary(
            kind=LiveShotBoundaryKind.SHOT_STARTED,
            status="observed",
            window=second,
        ),
        events,
    )

    assert [(event.kind, event.shot_id) for event in events] == [
        ("SHOT_INCOMPLETE", "shot-0001"),
        ("SHOT_STARTED", "shot-0002"),
    ]
    assert json.loads(events[0].reason or "") == {
        "error_type": "IncompleteShotEvidence",
        "message": "next shot started before prior post-roll completed",
        "stage": "post_roll_superseded",
    }
    assert runtime._pending_window is None
    ring.settle_shot(130)


@pytest.mark.parametrize("blocked_stage", ["materialize", "enqueue"])
def test_settled_event_is_published_before_artifact_preparation_can_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_stage: str,
) -> None:
    runtime, ring, operations = _runtime(tmp_path, monkeypatch)
    original_materialize = product_runtime.materialize_frozen_interval
    blocker_entered = threading.Event()
    release_blocker = threading.Event()

    def blocking_materialize(interval: object, destination: Path) -> object:
        if blocked_stage == "materialize":
            blocker_entered.set()
            assert release_blocker.wait(timeout=1.0)
        return original_materialize(interval, destination)

    class BlockingStore(_RecordingStore):
        def enqueue(self, job: object) -> SimpleNamespace:
            if blocked_stage == "enqueue":
                blocker_entered.set()
                assert release_blocker.wait(timeout=1.0)
            return super().enqueue(job)

    runtime._job_store = BlockingStore(operations)
    runtime._event_channel = product_runtime.ProductRealtimeEventChannel(maximum_pending_events=4)
    monkeypatch.setattr(product_runtime, "materialize_frozen_interval", blocking_materialize)
    events: list[ProductRealtimeEvent] = []
    failures: list[BaseException] = []

    def finalize() -> None:
        try:
            runtime._finalize_if_ready(_frame(4), events, threading.Event())
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=finalize, daemon=True)
    worker.start()
    assert blocker_entered.wait(timeout=1.0)

    settled_event = runtime._event_channel.receive(timeout_seconds=0.1)
    assert settled_event is not None
    assert settled_event.kind == "SHOT_SETTLED"
    assert settled_event.settled is not None
    assert settled_event.settled.rule_frames == _RefinedShot.rule_frames()
    assert settled_event.settled.rule_coverage == _RefinedShot.coverage
    assert worker.is_alive()
    assert ring.released is False

    release_blocker.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert failures == []
    assert runtime._event_channel.receive(timeout_seconds=0.1).kind == "SHOT_ENQUEUED"
    assert operations.count("release") == 1
    assert ring.released is True


def test_live_settled_shot_new_rule_fields_have_safe_empty_defaults() -> None:
    settled = product_runtime.LiveSettledShot(
        "shot-legacy",
        1,
        2,
        2_000_000,
        (),
        True,
        None,
        {},
        "epoch-legacy",
        "job-legacy",
    )

    assert settled.rule_frames == ()
    assert settled.rule_coverage is None
    assert settled.event_config is None
    assert settled.event_config_digest is None


def test_live_settled_shot_preserves_refiner_abstention_and_fact_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AbstainingRefinedShot:
        publishable = False
        shot_abstention_reason = "trusted_after_missing"
        fact_coverage = {
            "first_contact": FactCoverageVerdict(
                status="abstained",
                reason="trusted_after_missing",
            ),
            "cushion_contacts": FactCoverageVerdict(status="observed"),
            "pots": FactCoverageVerdict(status="abstained", reason="tracker_degraded"),
        }

        @staticmethod
        def observations() -> tuple[ObservationFrame, ...]:
            return _RefinedShot.observations()

    class AbstainingRefiner:
        def __init__(self, *, maximum_shot_frames: int) -> None:
            assert maximum_shot_frames == 12

        def refine(self, **kwargs: object) -> AbstainingRefinedShot:
            return AbstainingRefinedShot()

    runtime, _ring, _operations = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(product_runtime, "RealtimeShotRefiner", AbstainingRefiner)
    events: list[ProductRealtimeEvent] = []

    runtime._finalize_if_ready(_frame(4), events, threading.Event())

    settled = next(event.settled for event in events if event.kind == "SHOT_SETTLED")
    assert settled is not None
    assert settled.publishable is False
    assert settled.shot_abstention_reason == "trusted_after_missing"
    assert settled.fact_coverage == AbstainingRefinedShot.fact_coverage
    assert settled.fact_coverage["first_contact"].status == "abstained"
    assert settled.fact_coverage["pots"].reason == "tracker_degraded"


def test_bounded_event_channel_overflow_fails_closed_without_reordering() -> None:
    channel = product_runtime.ProductRealtimeEventChannel(maximum_pending_events=1)
    started = ProductRealtimeEvent(kind="SHOT_STARTED", shot_id="shot-00000001")
    settled = ProductRealtimeEvent(kind="SHOT_SETTLED", shot_id="shot-00000001")

    channel.publish(started)
    with pytest.raises(RuntimeError, match="event channel capacity"):
        channel.publish(settled)

    assert channel.receive(timeout_seconds=0.01) == started
    with pytest.raises(
        product_runtime.ProductRealtimeEventChannelFailure,
        match="event channel capacity",
    ):
        channel.receive(timeout_seconds=0.01)


def test_default_event_receive_wakes_promptly_when_channel_fails() -> None:
    channel = product_runtime.ProductRealtimeEventChannel(maximum_pending_events=1)
    entered_receive = threading.Event()
    outcome: list[BaseException | ProductRealtimeEvent | None] = []

    def receive_until_terminal() -> None:
        entered_receive.set()
        try:
            outcome.append(channel.receive())
        except BaseException as exc:
            outcome.append(exc)

    consumer = threading.Thread(target=receive_until_terminal, daemon=True)
    consumer.start()
    assert entered_receive.wait(timeout=1.0)
    consumer.join(timeout=0.05)
    assert consumer.is_alive()

    channel.fail("authority event stream unavailable")

    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], product_runtime.ProductRealtimeEventChannelFailure)
    assert str(outcome[0]) == "authority event stream unavailable"


def test_materialization_failure_discards_partial_media_and_releases_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, ring, operations = _runtime(tmp_path, monkeypatch)

    def fail_materialization(interval: object, destination: Path) -> object:
        operations.append("materialize_failed")
        destination.mkdir(parents=True)
        (destination / "partial.jpg").write_bytes(b"partial")
        raise OSError("copy failed")

    monkeypatch.setattr(product_runtime, "materialize_frozen_interval", fail_materialization)
    events: list[ProductRealtimeEvent] = []

    runtime._finalize_if_ready(_frame(4), events, threading.Event())

    assert [event.kind for event in events] == ["SHOT_SETTLED", "SHOT_INCOMPLETE"]
    reason = json.loads(events[1].reason or "")
    assert reason == {
        "error_type": "OSError",
        "message": "copy failed",
        "stage": "materialization",
    }
    assert operations == ["freeze", "materialize_failed", "release"]
    assert ring.released is True
    assert runtime._pending_window is None
    assert not any(runtime._job_media_root.iterdir())


def test_enqueue_failure_discards_materialized_media_and_releases_ring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingStore:
        def enqueue(self, job: object) -> object:
            raise RuntimeError("queue database unavailable")

    runtime, ring, operations = _runtime(tmp_path, monkeypatch, store=FailingStore())
    events: list[ProductRealtimeEvent] = []

    runtime._finalize_if_ready(_frame(4), events, threading.Event())

    assert [event.kind for event in events] == ["SHOT_SETTLED", "SHOT_INCOMPLETE"]
    reason = json.loads(events[1].reason or "")
    assert reason["stage"] == "enqueue"
    assert not list((tmp_path / "analysis-media").glob("*/interval.json"))
    assert operations.count("release") == 1
    assert ring.released is True
    assert runtime._pending_window is None
