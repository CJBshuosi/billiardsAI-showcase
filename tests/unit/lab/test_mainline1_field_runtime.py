from __future__ import annotations

import hashlib
import importlib
import importlib.util
import queue
import threading

import pytest

from billiards_ai.lab.mainline1_capture_hub import CaptureHub, CaptureSubscription
from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEventChannel
from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureFrame,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)


def test_guidance_cleanup_failure_keeps_owner_for_shutdown_retry():
    from types import SimpleNamespace

    from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle

    class Guidance:
        attempts = 0
        cleanup_complete = False

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("guidance cleanup failed once")
            self.cleanup_complete = True

    guidance = Guidance()
    handle = RuntimeHandle(
        SimpleNamespace(cancel=lambda: None), lambda: None, snapshot_guidance=guidance
    )
    handle.close()  # Optional guidance failure cannot abort ordinary recorder cancellation.
    assert handle.snapshot_guidance is guidance, "failed owner must remain reachable"
    assert not handle.snapshot_guidance_cleanup_complete
    handle.shutdown()
    assert guidance.attempts == 2
    assert handle.snapshot_guidance_cleanup_complete


class GatedCapture:
    """A capture-contract source deliberately without native-only start()."""

    identity = NativeCaptureIdentity(
        device_locator="scenario://test",
        native_handle_ref="scenario://test/handle",
        durable_fingerprint="a" * 64,
    )

    def __init__(self, epoch, calls):
        self.epoch = epoch
        self.calls = calls
        self.pending = queue.Queue()
        self.cancelled = threading.Event()
        self.reading = threading.Event()

    def emit(self, frame_id):
        frame = CaptureFrame(
            source_epoch_id=self.epoch,
            capture_frame_id=frame_id,
            captured_at_ns=frame_id * 16_666_667,
            width=16,
            height=12,
            mjpeg=f"jpeg-{frame_id}".encode(),
        )
        self.pending.put(frame)
        return frame

    def frames(self, stop):
        self.calls.append("frames")
        self.reading.set()
        while not stop.is_set() and not self.cancelled.is_set():
            try:
                yield self.pending.get(timeout=0.01)
            except queue.Empty:
                pass

    def cancel(self):
        self.calls.append("cancel")
        self.cancelled.set()


@pytest.mark.parametrize("physical_callbacks", [False, True])
@pytest.mark.parametrize("frozen_digest", [None, "f" * 64])
def test_injected_capture_reaches_runtime_and_rollover_preserves_camera_ownership(
    tmp_path, monkeypatch, physical_callbacks, frozen_digest
):
    """Catches raw-backend analysis, native start coupling, or rollover reopening capture."""
    name = "billiards_ai.lab.mainline1_field_runtime"
    assert importlib.util.find_spec(name) is not None, "reusable live runtime assembly is missing"
    module = importlib.import_module(name)
    calls = []
    sources = []
    options = []
    observed = []
    stages = []
    runtime_constructed = threading.Event()

    def capture_factory(epoch):
        calls.append(("capture", epoch))
        source = GatedCapture(epoch, calls)
        sources.append(source)
        return source

    class Runtime:
        def __init__(self, **kwargs):
            calls.append("runtime")
            options.append(kwargs)
            runtime_constructed.set()

        def run(self):
            observed.append(next(options[-1]["source"].frames(threading.Event())))

    monkeypatch.setattr(module, "ProductRealtimeRuntime", Runtime)
    original_start = CaptureHub.start

    def start(hub):
        calls.append("hub-start")
        original_start(hub)

    monkeypatch.setattr(CaptureHub, "start", start)
    config_path = tmp_path / "config.yaml"
    calibration, detector, identity, shot_config = object(), object(), object(), object()
    factory = module.build_live_runtime_factory(
        capture_factory=capture_factory,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=detector,
        identity_classifier=identity,
        shot_config=shot_config,
        capture_setup_id="setup-1",
        calibration_artifact_id="calibration-1",
        config_path=config_path,
        **(
            {}
            if frozen_digest is None
            else {
                "config_sha256": frozen_digest,
                "identity_training_package_id": "identity-1",
                "identity_training_package_sha256": "e" * 64,
            }
        ),
        ring_root=tmp_path / "rings",
        jobs_root=tmp_path / "jobs",
        capture_starter=(lambda source: calls.append("starter")) if physical_callbacks else None,
        startup_validator=(
            (lambda source: calls.append("validator")) if physical_callbacks else None
        ),
        startup_progress=stages.append,
        capture_diagnostics_factory=(
            None
            if physical_callbacks
            else lambda actual, epoch: {
                "source_sha256": actual.identity.durable_fingerprint,
                "source_epoch_id": epoch,
                "configured": {"width": 16},
            }
        ),
    )
    channel = ProductRealtimeEventChannel()
    first = factory("session-1", channel, 7, "epoch-1")
    if physical_callbacks:
        assert first.capture_diagnostics is None
    else:
        assert first.capture_diagnostics["source_sha256"] == sources[0].identity.durable_fingerprint
        assert first.capture_diagnostics["source_epoch_id"] == "epoch-1"
        first.capture_diagnostics["configured"]["width"] = 99
        assert first.capture_diagnostics["configured"]["width"] == 16
    config_path.write_bytes(b"deferred-config")
    failures = []

    def run_first():
        try:
            first.run()
        except BaseException as exc:
            failures.append(exc)

    runner = threading.Thread(target=run_first, daemon=True)
    runner.start()
    try:
        assert sources[0].reading.wait(timeout=1)
        assert not runtime_constructed.wait(timeout=0.02)
        assert not first.wait_until_ready(timeout_seconds=0.01)
        frame = sources[0].emit(0)
        runner.join(timeout=1)
        assert not runner.is_alive()
        assert failures == []
        assert observed == [frame]
        assert isinstance(options[0]["source"], CaptureSubscription)
        assert options[0]["source"] is not sources[0]
        assert first.current_capture_frame is frame
        assert not first.wait_until_ready(timeout_seconds=0.1)
        assert options[0]["calibration"] is calibration
        assert options[0]["detector"] is detector
        assert options[0]["identity_classifier"] is identity
        assert options[0]["shot_config"] is shot_config
        assert options[0]["event_channel"] is channel
        assert options[0]["identity_training_package_id"] == (
            None if frozen_digest is None else "identity-1"
        )
        assert options[0]["identity_training_package_sha256"] == (
            None if frozen_digest is None else "e" * 64
        )
        assert options[0]["capture_setup_id"] == "setup-1"
        assert options[0]["calibration_artifact_id"] == "calibration-1"
        assert options[0]["config_sha256"] == (
            frozen_digest or hashlib.sha256(b"deferred-config").hexdigest()
        )
        assert options[0]["config"].initial_shot_number == 7
        assert options[0]["config"].ring_root == (
            tmp_path / "rings/sessions/session-1/epoch-1/shot-ring"
        )
        assert options[0]["job_media_root"] == tmp_path / "jobs/media/session-1"
        assert options[0]["quarantine_root"] == tmp_path / "jobs/quarantine/session-1"
        assert options[0]["evidence_store"].delegate is first.capture_hub._recorder
        assert calls.index("hub-start") < calls.index("frames") < calls.index("runtime")
        if physical_callbacks:
            assert calls.index("starter") < calls.index("validator") < calls.index("hub-start")
            assert stages[:4] == [
                "live_capture_starting",
                "live_capture_started",
                "live_capture_validation_started",
                "live_capture_validation_ready",
            ]
        assert stages[-2:] == ["capture_hub_starting", "capture_hub_ready"]

        first.close()
        options[0]["evidence_store"].close()
        assert not sources[0].cancelled.is_set()
        second = factory("session-2", channel, 0, "epoch-1")
        assert second.capture_hub is first.capture_hub
        assert second.source is first.source
        assert second.capture_diagnostics == first.capture_diagnostics
        second_frame = sources[0].emit(1)
        second.run()
        second.close()
        assert observed == [frame, second_frame]
        assert options[1]["job_store"] is options[0]["job_store"]
        assert options[1]["boundary_ledger"] is not options[0]["boundary_ledger"]
        assert options[1]["evidence_store"].delegate is not options[0]["evidence_store"].delegate
        assert options[1]["evidence_store"].delegate is second.capture_hub._recorder
        assert options[1]["config"].ring_root == (
            tmp_path / "rings/sessions/session-2/epoch-1/shot-ring"
        )
        assert calls.count("starter") == int(physical_callbacks)
        assert calls.count("validator") == int(physical_callbacks)
        assert calls.count("frames") == 1
        assert not sources[0].cancelled.is_set()

        third = factory("session-3", channel, 0, "epoch-2")
        assert sources[0].cancelled.is_set()
        assert third.capture_hub is not first.capture_hub
        if not physical_callbacks:
            assert third.capture_diagnostics["source_epoch_id"] == "epoch-2"
        assert [source.epoch for source in sources] == ["epoch-1", "epoch-2"]
        assert [call for call in calls if isinstance(call, tuple)] == [
            ("capture", "epoch-1"),
            ("capture", "epoch-2"),
        ]
        sources[1].emit(0)
        try:
            third.run()
        finally:
            third.shutdown()
        assert sources[1].cancelled.is_set()
        assert calls.count("starter") == 2 * int(physical_callbacks)
        assert calls.count("validator") == 2 * int(physical_callbacks)
    finally:
        first.shutdown()
        runner.join(timeout=1)


@pytest.mark.parametrize("pair", [("identity", None), (None, "a" * 64)])
def test_incomplete_identity_pair_refuses_before_jobs_store_creation(tmp_path, pair):
    from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory

    with pytest.raises(ValueError, match="identity.*complete"):
        build_live_runtime_factory(
            capture_factory=lambda _: None,
            capture_config=RealtimeCaptureConfig(),
            calibration=object(),
            detector=object(),
            identity_classifier=object(),
            shot_config=object(),
            capture_setup_id="setup",
            calibration_artifact_id="calibration",
            config_path=tmp_path / "config",
            ring_root=tmp_path / "rings",
            jobs_root=tmp_path / "jobs",
            identity_training_package_id=pair[0],
            identity_training_package_sha256=pair[1],
        )
    assert not (tmp_path / "jobs").exists()


def _handoff_factory(tmp_path, monkeypatch):
    """Keep actual hub/writers/ledger while controlling the model boundary."""
    module = importlib.import_module("billiards_ai.lab.mainline1_field_runtime")
    writers, sources, options = [], [], []
    real_writer = module.AsyncMjpegSegmentRing

    def writer(ring, **kwargs):
        actual = real_writer(ring, maximum_pending_frames=1)
        writers.append(actual)
        return actual

    def capture(epoch):
        source = GatedCapture(epoch, [])
        sources.append(source)
        return source

    class Runtime:
        def __init__(self, **kwargs):
            options.append(kwargs)

    monkeypatch.setattr(module, "AsyncMjpegSegmentRing", writer)
    monkeypatch.setattr(module, "ProductRealtimeRuntime", Runtime)
    factory = module.build_live_runtime_factory(
        capture_factory=capture,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=object(),
        detector=object(),
        identity_classifier=object(),
        shot_config=object(),
        capture_setup_id="setup",
        calibration_artifact_id="calibration",
        config_path=tmp_path / "unused-config",
        config_sha256="a" * 64,
        ring_root=tmp_path / "rings",
        jobs_root=tmp_path / "jobs",
    )
    return factory, writers, sources, options


def _close_test_writers(writers):
    from billiards_ai.lab.mainline1_realtime_ring import AsyncMjpegSegmentRing, RealtimeRingError

    for writer in writers:
        try:
            AsyncMjpegSegmentRing.close(writer, timeout_seconds=1)
        except RealtimeRingError:
            pass
        writer._thread.join(1)


def test_shared_factory_retries_one_adopted_target_with_matching_session(tmp_path, monkeypatch):
    """Catches pending rollover allocating a third writer or returning the old session."""
    from billiards_ai.lab.mainline1_realtime_capture import CaptureContractError
    from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError

    factory, writers, sources, options = _handoff_factory(tmp_path, monkeypatch)
    channel = ProductRealtimeEventChannel()
    first = factory("session-1", channel, 0, "epoch-1")
    old = writers[0]
    append_entered, release_append = threading.Event(), threading.Event()
    original_append, original_close = old._ring.append, old.close

    def append(frame):
        append_entered.set()
        assert release_append.wait(2)
        original_append(frame)

    monkeypatch.setattr(old._ring, "append", append)
    monkeypatch.setattr(old, "close", lambda: original_close(timeout_seconds=0.01))
    frame = sources[0].emit(0)
    sources[0].pending.get_nowait()
    old.append(frame)
    assert append_entered.wait(1)
    old.append(sources[0].emit(1))
    sources[0].pending.get_nowait()
    try:
        with pytest.raises(RealtimeRingError, match="capacity"):
            factory("session-2", channel, 3, "epoch-1")
        adopted = writers[1]
        assert first.capture_hub._recorder is adopted
        assert len(writers) == 2
        # Requests for another context cannot bypass the pending session-2 handoff.
        with pytest.raises(CaptureContractError, match="pending"):
            factory("session-1", channel, 0, "epoch-1")
        with pytest.raises(RealtimeRingError, match="capacity"):
            factory("session-2", channel, 3, "epoch-1")
        assert len(writers) == 2
        release_append.set()
        monkeypatch.setattr(old, "close", lambda: original_close(timeout_seconds=1))
        recovered = factory("session-2", channel, 3, "epoch-1")
        assert len(writers) == 2
        assert recovered.capture_hub is first.capture_hub
        assert recovered.source is first.source
        recovered.runtime_factory(recovered._analysis_source)
        assert options[-1]["session_id"] == "session-2"
        assert options[-1]["boundary_ledger"].session_id == "session-2"
        assert options[-1]["boundary_ledger"].source_epoch_id == "epoch-1"
        assert options[-1]["evidence_store"].delegate is adopted
        assert options[-1]["config"].initial_shot_number == 3
        assert options[-1]["config"].ring_root == (
            tmp_path / "rings/sessions/session-2/epoch-1/shot-ring"
        )
        routed = threading.Event()
        original_new_append = adopted._ring.append

        def observe(frame):
            original_new_append(frame)
            routed.set()

        monkeypatch.setattr(adopted._ring, "append", observe)
        recovered.capture_hub.start()
        actual = sources[0].emit(0)
        assert routed.wait(1)
        assert recovered.current_capture_frame == actual
        assert len(sources) == 1
        assert not old._thread.is_alive()
        recovered.shutdown()
        assert not adopted._thread.is_alive()
    finally:
        release_append.set()
        monkeypatch.setattr(old, "close", lambda: original_close(timeout_seconds=1))
        first.shutdown()
        _close_test_writers(writers)


def test_shared_factory_closes_unadopted_target_on_rejection(tmp_path, monkeypatch):
    """Catches a closed hub rejecting replacement and leaking its new writer."""
    from billiards_ai.lab.mainline1_realtime_capture import CaptureContractError

    factory, writers, _, _ = _handoff_factory(tmp_path, monkeypatch)
    channel = ProductRealtimeEventChannel()
    first = factory("session-1", channel, 0, "epoch-1")
    first.shutdown()
    try:
        with pytest.raises(CaptureContractError, match="closed"):
            factory("session-2", channel, 0, "epoch-1")
        assert all(not writer._thread.is_alive() for writer in writers)
    finally:
        _close_test_writers(writers)


def test_rejected_target_cleanup_timeout_remains_reachable_from_shutdown(tmp_path, monkeypatch):
    """Catches retaining an unadopted writer without an executable shutdown path."""
    from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError

    factory, writers, sources, _ = _handoff_factory(tmp_path, monkeypatch)
    channel = ProductRealtimeEventChannel()
    first = factory("session-1", channel, 0, "epoch-1")
    first.shutdown()
    module = importlib.import_module("billiards_ai.lab.mainline1_field_runtime")
    constructor = module.AsyncMjpegSegmentRing
    entered, release = threading.Event(), threading.Event()

    def writer(ring, **kwargs):
        candidate = constructor(ring, **kwargs)
        original_append, original_close = ring.append, candidate.close

        def append(frame):
            entered.set()
            assert release.wait(2)
            original_append(frame)

        monkeypatch.setattr(ring, "append", append)
        monkeypatch.setattr(candidate, "close", lambda: original_close(timeout_seconds=0.01))
        candidate.append(sources[0].emit(0))
        sources[0].pending.get_nowait()
        assert entered.wait(1)
        candidate.append(sources[0].emit(1))
        sources[0].pending.get_nowait()
        return candidate

    monkeypatch.setattr(module, "AsyncMjpegSegmentRing", writer)
    try:
        with pytest.raises(BaseExceptionGroup):
            factory("session-2", channel, 0, "epoch-1")
        candidate = writers[1]
        assert not first.capture_hub._uses_recorder(candidate)
        with pytest.raises(RealtimeRingError, match="capacity"):
            first.shutdown()
        with pytest.raises(BaseExceptionGroup):
            factory("session-2", channel, 0, "epoch-1")
        assert len(writers) == 2
        release.set()
        monkeypatch.setattr(
            candidate,
            "close",
            lambda: type(candidate).close(candidate, timeout_seconds=1),
        )
        first.shutdown()
        first.shutdown()
        assert not candidate._thread.is_alive()
        assert candidate._commands.empty()
        assert len(sources) == 1
    finally:
        release.set()
        _close_test_writers(writers)


def test_authoritative_handle_retains_short_backlog_without_losing_preview_or_recording():
    """A slow analysis consumer must not silently lose authoritative frames."""
    from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle

    calls = []
    source = GatedCapture("epoch-1", calls)
    recorded = []
    recorded_all = threading.Event()

    class Recorder:
        def append(self, frame):
            recorded.append(frame)
            if frame.capture_frame_id == 5:
                recorded_all.set()

        def close(self):
            return None

    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        recorder=Recorder(),
    )
    preview = hub.subscribe("preview", maximum_pending_frames=1)
    first_consumed = threading.Event()
    release = threading.Event()
    observed = []

    class Runtime:
        def __init__(self, analysis):
            self.analysis = analysis

        def run(self):
            rows = self.analysis.frames(threading.Event())
            try:
                observed.append(next(rows))
                first_consumed.set()
                assert release.wait(timeout=1)
                while observed[-1].capture_frame_id < 5:
                    observed.append(next(rows))
            finally:
                rows.close()

    handle = RuntimeHandle(source, Runtime, capture_hub=hub, start_capture=True)
    failures = []

    def run():
        try:
            handle.run()
        except BaseException as exc:
            failures.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert source.reading.wait(timeout=1)
        original = [source.emit(0)]
        assert first_consumed.wait(timeout=1)
        original.extend(source.emit(i) for i in range(1, 6))
        assert recorded_all.wait(timeout=1)
        assert calls.count("cancel") == 0
        assert next(preview.frames(threading.Event())) is original[-1]
        release.set()
        runner.join(timeout=1)
        assert not runner.is_alive()
        assert failures == []
        assert len(observed) == len(original)
        assert all(actual is expected for actual, expected in zip(observed, original, strict=True))
        assert all(actual is expected for actual, expected in zip(recorded, original, strict=True))
        assert hub.failure_reason is None
    finally:
        release.set()
        hub.close()
        runner.join(timeout=1)


def test_analysis_activation_waits_for_factory_after_more_than_180_startup_frames():
    """Initialization has no authority; bounded FIFO begins before ready/run."""
    from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle

    source = GatedCapture("epoch-1", [])
    recorded, authority = [], []
    recorded_all = threading.Event()

    class Recorder:
        def append(self, frame):
            recorded.append(frame)
            if frame.capture_frame_id == 200:
                recorded_all.set()

        def close(self):
            pass

    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        recorder=Recorder(),
    )
    preview = hub.subscribe("preview", maximum_pending_frames=1)
    factory_entered, release_factory = threading.Event(), threading.Event()
    reader_started, received = threading.Event(), threading.Event()
    entered_run, release_run = threading.Event(), threading.Event()
    observed, failures = [], []
    readers = []

    def factory(analysis):
        rows = analysis.frames(threading.Event())

        def read():
            reader_started.set()
            try:
                observed.append(next(rows))
                received.set()
            except BaseException as exc:
                failures.append(exc)
            finally:
                rows.close()

        reader = threading.Thread(target=read, daemon=True)
        readers.append(reader)
        reader.start()
        factory_entered.set()
        assert release_factory.wait(1)

        class Runtime:
            def run(self):
                assert handle._ready.is_set()
                assert received.wait(1)
                # Stand-ins for all work that requires an admitted runtime.
                authority.extend(["shot", "event", "job"])
                entered_run.set()
                assert release_run.wait(1)

        return Runtime()

    handle = RuntimeHandle(source, factory, capture_hub=hub)

    def run():
        try:
            handle.run()
        except BaseException as exc:
            failures.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert source.reading.wait(1)
        originals = [source.emit(0)]
        assert factory_entered.wait(1) and reader_started.wait(1)
        originals.extend(source.emit(i) for i in range(1, 201))
        assert recorded_all.wait(1)
        assert not received.wait(0.02)
        assert authority == [] and not handle._ready.is_set()
        assert not handle.wait_until_ready(timeout_seconds=0.01)
        analysis = handle._analysis_source
        assert analysis.pending_frame_count == 1
        assert analysis.pending_bytes == len(originals[-1].mjpeg)
        assert analysis._maximum_pending_frames == 180
        assert analysis._maximum_pending_bytes == 64 * 1024 * 1024
        assert analysis.allow_accounted_frame_gaps is True
        assert next(preview.frames(threading.Event())) is originals[-1]
        assert len(recorded) == len(originals)
        assert all(a is b for a, b in zip(recorded, originals, strict=True))
        release_factory.set()
        assert entered_run.wait(1)
        assert handle.wait_until_ready(timeout_seconds=0.01)
        assert observed == [originals[-1]] and observed[0] is originals[-1]
        assert authority == ["shot", "event", "job"]
    finally:
        release_factory.set()
        release_run.set()
        hub.close()
        runner.join(1)
        for reader in readers:
            reader.join(1)
    assert not runner.is_alive() and failures == []


def test_pre_activation_byte_failure_still_enters_constructed_runtime_cleanup(monkeypatch):
    from billiards_ai.lab import mainline1_field_runtime as module
    from billiards_ai.lab.mainline1_realtime_capture import CaptureContractError

    monkeypatch.setattr(module, "_ANALYSIS_FIFO_MAXIMUM_PENDING_BYTES", 5)
    source = GatedCapture("epoch-1", [])
    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    cleanup = threading.Event()
    failures, authority = [], []

    def factory(analysis):
        class Runtime:
            def run(self):
                assert not handle._ready.is_set()
                authority.extend(analysis.frames(threading.Event()))
                cleanup.set()

        return Runtime()

    handle = module.RuntimeHandle(source, factory, capture_hub=hub)

    def run():
        try:
            handle.run()
        except BaseException as exc:
            failures.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert source.reading.wait(1)
        source.emit(0)  # Six MJPEG bytes: first-frame failure before activation.
        runner.join(1)
        assert not runner.is_alive()
        assert cleanup.is_set() and authority == []
        assert len(failures) == 1 and isinstance(failures[0], CaptureContractError)
        assert "byte capacity" in str(failures[0])
        assert not handle.wait_until_ready(timeout_seconds=0.01)
        assert hub.failure_reason is None
        assert not source.cancelled.is_set()
        assert handle._analysis_source.pending_bytes == 0
    finally:
        hub.close()
        runner.join(1)


@pytest.mark.parametrize("kind", ["fifo", "preview", "direct"])
def test_fifo_keeps_existing_live_processing_budget_policy(tmp_path, monkeypatch, kind):
    """Exercise real config construction without starting analysis workers."""
    from billiards_ai.lab import mainline1_product_realtime_runtime as module
    from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle

    monkeypatch.setattr(module, "HeavyAnchorWorker", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "AnchorReplayCoordinator", lambda *args: object())
    source = GatedCapture("epoch-1", [])
    handle = RuntimeHandle(
        source,
        lambda _: None,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    if kind == "fifo":
        analysis = handle._analysis_source
    elif kind == "preview":
        analysis = handle.capture_hub.subscribe("preview", maximum_pending_frames=1)
    else:
        analysis = source
    try:
        runtime = module.ProductRealtimeRuntime(
            source=analysis,
            capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
            calibration=object(),
            detector=object(),
            evidence_store=object(),
            config=module.ProductRealtimeRuntimeConfig(ring_root=tmp_path / "unused"),
        )
        policy = runtime._tracker_diagnostic_config
        assert policy.allow_accounted_frame_gaps is (kind != "direct")
        assert policy.enforce_processing_budget is (kind == "direct")
        assert policy.processing_budget_ms == 16.7
    finally:
        handle.shutdown()
