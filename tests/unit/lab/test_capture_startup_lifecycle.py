"""Startup authority and bounded observations at the real capture boundary."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from billiards_ai.lab import mainline1_field_runtime as field_runtime
from billiards_ai.lab import mainline1_scenario_capture as scenario
from billiards_ai.lab.mainline1_capture_hub import CaptureHub
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame, RealtimeCaptureConfig


def _frame():
    return CaptureFrame(
        source_epoch_id="epoch",
        capture_frame_id=0,
        captured_at_ns=17,
        width=16,
        height=12,
        mjpeg=b"jpeg",
    )


class GatedSource:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def frames(self, stop):
        self.entered.set()
        while not stop.is_set() and not self.release.wait(0.005):
            pass
        if not stop.is_set():
            yield _frame()
        stop.wait()

    def cancel(self):
        self.release.set()


def _launch(call):
    failures = []

    def run():
        try:
            call()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, failures


@pytest.mark.parametrize("allowance", [2.75, 5.0])
def test_configured_allowance_reaches_first_frame_wait_in_bounded_slices(monkeypatch, allowance):
    """Catches the literal 1s wait and resetting the deadline on each slice."""
    source = GatedSource()
    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(
            width=16,
            height=12,
            fps=60,
            startup_timeout_seconds=allowance,
        ),
    )
    now = [10.0]
    waits = []

    def wait(*, after_frame_id, timeout_seconds):
        assert after_frame_id is None
        waits.append(timeout_seconds)
        now[0] += timeout_seconds
        return None

    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(hub, "wait_for_frame", wait)
    handle = field_runtime.RuntimeHandle(source, lambda _: pytest.fail("no frame"), capture_hub=hub)
    try:
        with pytest.raises(RuntimeError, match="capture hub did not publish a startup frame"):
            handle.run()
        assert sum(waits) == pytest.approx(allowance)
        assert max(waits) <= 0.05
        assert not handle.wait_until_ready(timeout_seconds=0.01)
        assert "startup_diagnostics=" in handle.failure_reason
    finally:
        handle.shutdown()


def test_gated_cold_frame_within_allowance_constructs_runtime(monkeypatch):
    source = GatedSource()
    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(
            width=16,
            height=12,
            fps=60,
            startup_timeout_seconds=0.3,
        ),
    )
    waits = []
    original_wait = hub.wait_for_frame
    runtime_entered, release_runtime = threading.Event(), threading.Event()

    def wait(**kwargs):
        waits.append(kwargs["timeout_seconds"])
        if len(waits) == 3:
            source.release.set()
        return original_wait(**kwargs)

    class Runtime:
        def run(self):
            runtime_entered.set()
            assert release_runtime.wait(1)

    monkeypatch.setattr(hub, "wait_for_frame", wait)
    handle = field_runtime.RuntimeHandle(source, lambda _: Runtime(), capture_hub=hub)
    thread, failures = _launch(handle.run)
    try:
        assert runtime_entered.wait(0.6)
        assert handle.wait_until_ready(timeout_seconds=0.05)
        assert len(waits) == 3
        assert max(waits) <= 0.05
        assert handle.current_capture_frame == _frame()
    finally:
        release_runtime.set()
        thread.join(1)
        handle.shutdown()
    assert not thread.is_alive()
    assert failures == []
    assert not handle.wait_until_ready(timeout_seconds=0.01)


def test_source_failure_retains_original_exception_in_startup_chain():
    original = ValueError("private source failure")

    class FailedSource(GatedSource):
        def frames(self, stop):
            raise original
            yield

    source = FailedSource()
    handle = field_runtime.RuntimeHandle(
        source,
        lambda _: pytest.fail("failed source"),
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    try:
        with pytest.raises(
            RuntimeError, match="capture hub.*ValueError: private source failure"
        ) as e:
            handle.run()
        assert e.value.__cause__ is original
        assert handle.capture_hub.failure_reason == "ValueError: private source failure"
        assert handle.capture_hub.startup_diagnostics["failure_type"] == "ValueError"
        assert not handle.wait_until_ready(timeout_seconds=0.01)
    finally:
        handle.shutdown()


def test_cancel_before_frame_revokes_factory_and_readiness_without_stopping_shared_hub():
    source = GatedSource()
    handle = field_runtime.RuntimeHandle(
        source,
        lambda _: pytest.fail("cancelled factory"),
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    thread, failures = _launch(handle.run)
    assert source.entered.wait(0.5)
    closer, close_failures = _launch(handle.close)
    try:
        thread.join(0.2)
        closer.join(0.2)
        assert not thread.is_alive(), "startup wait did not observe cancellation in <=50ms slices"
        assert not closer.is_alive()
        assert failures == close_failures == []
        assert not handle.wait_until_ready(timeout_seconds=0.01)
        assert handle.capture_hub.latest_frame() is None
    finally:
        source.release.set()
        thread.join(1)
        closer.join(1)
        handle.shutdown()


def test_cancel_racing_factory_enters_existing_cleanup_with_no_frame_authority():
    source = GatedSource()
    source.release.set()
    constructing, finish_construction, cleaned = (threading.Event() for _ in range(3))

    class Runtime:
        def __init__(self, subscription):
            self.subscription = subscription
            constructing.set()
            assert finish_construction.wait(1)

        def run(self):
            try:
                assert list(self.subscription.frames(threading.Event())) == []
            finally:
                cleaned.set()

    handle = field_runtime.RuntimeHandle(
        source, Runtime, capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60)
    )
    thread, failures = _launch(handle.run)
    assert constructing.wait(0.5)
    closer, close_failures = _launch(handle.close)
    try:
        assert handle._cancelled.wait(0.5)
        assert not handle.wait_until_ready(timeout_seconds=0.01)
        finish_construction.set()
        thread.join(1)
        closer.join(1)
        assert cleaned.is_set()
        assert failures == close_failures == []
        assert not handle.wait_until_ready(timeout_seconds=0.01)
    finally:
        finish_construction.set()
        handle.shutdown()
        thread.join(1)
        closer.join(1)


def test_finished_and_cancelled_legacy_handle_never_reports_ready():
    source = GatedSource()
    handle = field_runtime.RuntimeHandle(source, lambda: SimpleNamespace(run=lambda: None))
    handle.run()
    assert not handle.wait_until_ready(timeout_seconds=0.01)
    handle.close()
    assert not handle.wait_until_ready(timeout_seconds=0.01)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_source_failure_during_factory_revokes_input_and_retains_failure_through_cleanup(
    cleanup_fails,
):
    """Catches READY/buffered input after a known source failure during allocation."""
    original = ValueError("capture failed during construction")
    cleanup_failure = OSError("owned worker cleanup failed")
    fail_source, constructing, release_factory = (threading.Event() for _ in range(3))
    cleanup_entered, release_cleanup, cleaned = (threading.Event() for _ in range(3))
    consumed, subscription_failures = [], []

    class Source(GatedSource):
        def frames(self, stop):
            yield _frame()
            assert fail_source.wait(1)
            raise original

    class Runtime:
        def __init__(self, subscription):
            self.subscription = subscription
            constructing.set()
            assert release_factory.wait(1)

        def run(self):
            try:
                try:
                    consumed.extend(self.subscription.frames(threading.Event()))
                except BaseException as exc:
                    subscription_failures.append(exc)
            finally:
                cleanup_entered.set()
                assert release_cleanup.wait(1)
                cleaned.set()
            if cleanup_fails:
                raise cleanup_failure

    handle = field_runtime.RuntimeHandle(
        Source(), Runtime, capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60)
    )
    thread, failures = _launch(handle.run)
    try:
        assert constructing.wait(0.5)
        fail_source.set()
        assert handle.capture_hub.wait_for_frame(after_frame_id=0, timeout_seconds=0.5) is None
        assert handle.capture_hub.failure_reason == "ValueError: capture failed during construction"
        release_factory.set()
        assert cleanup_entered.wait(0.5)
        assert not handle.wait_until_ready(timeout_seconds=0.01)
        assert consumed == []
        assert subscription_failures == []
        release_cleanup.set()
        thread.join(1)
        assert not thread.is_alive()
        assert cleaned.is_set()
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, RuntimeError)
        assert "capture hub did not publish a startup frame" in str(failure)
        assert "ValueError: capture failed during construction" in str(failure)
        assert failure.__cause__ is original
        assert "ValueError: capture failed during construction" in handle.failure_reason
        if cleanup_fails:
            assert failure.__context__ is cleanup_failure
            assert any("OSError: owned worker cleanup failed" in note for note in failure.__notes__)
        assert not handle.wait_until_ready(timeout_seconds=0.01)
    finally:
        fail_source.set()
        release_factory.set()
        release_cleanup.set()
        thread.join(1)
        handle.shutdown()


def test_readiness_rejects_known_hub_failure_while_runtime_is_still_running():
    fail_source, runtime_entered, release_runtime = (threading.Event() for _ in range(3))

    class Source(GatedSource):
        def frames(self, stop):
            yield _frame()
            assert fail_source.wait(1)
            raise ValueError("source failed after runtime entry")

    class Runtime:
        def run(self):
            runtime_entered.set()
            assert release_runtime.wait(1)

    handle = field_runtime.RuntimeHandle(
        Source(),
        lambda _: Runtime(),
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    thread, failures = _launch(handle.run)
    try:
        assert runtime_entered.wait(0.5)
        assert handle.wait_until_ready(timeout_seconds=0.01)
        fail_source.set()
        assert handle.capture_hub.wait_for_frame(after_frame_id=0, timeout_seconds=0.5) is None
        assert handle.capture_hub.failure_reason == "ValueError: source failed after runtime entry"
        assert not handle.wait_until_ready(timeout_seconds=0.01)
    finally:
        fail_source.set()
        release_runtime.set()
        thread.join(1)
        handle.shutdown()
    assert not thread.is_alive()
    assert failures == []


@pytest.mark.parametrize(
    "stage",
    [
        "source_validation",
        "video_open",
        "mode_validation",
        "first_read",
        "first_encode",
        "video_release",
    ],
)
def test_phase_snapshot_remains_safe_and_available_during_blocked_operation(
    tmp_path,
    monkeypatch,
    stage,
):
    """Catches locks across codec/hash calls, cancellation erasing a stall, or payload leaks."""
    path = tmp_path / "secret-token-private-camera.avi"
    path.write_bytes(b"private pixels and credentials")
    backend = scenario.ScenarioVideoCaptureBackend(
        scenario.ScenarioCaptureConfig(
            path,
            RealtimeCaptureConfig(width=16, height=12, fps=60),
        ),
        source_epoch_id="secret-epoch",
    )
    entered, release = threading.Event(), threading.Event()

    def gate(name):
        if stage == name:
            entered.set()
            assert release.wait(1)

    validate = backend._validate_source

    def validation(*, content=False):
        if content:
            gate("source_validation")
        validate(content=content)

    class Capture:
        def __init__(self, _path):
            gate("video_open")

        def isOpened(self):
            gate("mode_validation")
            return True

        def get(self, key):
            return {
                scenario.cv2.CAP_PROP_BACKEND: 0,
                scenario.cv2.CAP_PROP_FRAME_WIDTH: 16,
                scenario.cv2.CAP_PROP_FRAME_HEIGHT: 12,
                scenario.cv2.CAP_PROP_FPS: 60.0,
                scenario.cv2.CAP_PROP_FRAME_COUNT: 1.0,
            }[key]

        def read(self):
            gate("first_read")
            return True, np.zeros((12, 16, 3), dtype=np.uint8)

        def release(self):
            gate("video_release")

    def encode(*_):
        gate("first_encode")
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(backend, "_validate_source", validation)
    monkeypatch.setattr(scenario.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(scenario.cv2, "imencode", encode)
    hub = CaptureHub(source=backend, capture_config=backend._config)
    hub.start()
    if stage == "video_release":
        assert hub.wait_for_frame(after_frame_id=None, timeout_seconds=0.5) is not None
        backend.cancel()
    try:
        assert entered.wait(0.5)
        snapshots = []
        reader, failures = _launch(lambda: snapshots.append(hub.startup_diagnostics))
        reader.join(0.1)
        assert not reader.is_alive(), "diagnostics blocked behind native operation"
        assert failures == []
        before = snapshots[0]
        assert before["source"]["stage"] == stage
        assert before["source"]["status"] == "in_progress"
        backend.cancel()
        after = hub.startup_diagnostics
        assert after["source"]["stage"] == stage
        assert isinstance(after["source"]["cancel_requested_at_ns"], int)
        assert after["source"]["thread_id"] is not None
        encoded = json.dumps(after)
        assert all(
            secret not in encoded
            for secret in (
                str(path),
                "secret",
                "private",
                "credentials",
                "pixels",
                "epoch",
            )
        )
        assert len(encoded) < 2000
        after["source"]["stage"] = "mutated"
        assert hub.startup_diagnostics["source"]["stage"] == stage
    finally:
        release.set()
        hub.close()
    done = backend.startup_diagnostics
    assert done["stage"] == "video_release"
    assert done["status"] == "released"
    assert isinstance(done["released_at_ns"], int)


def test_observations_leave_scripted_clock_frame_bytes_and_closed_gate_unchanged(
    tmp_path, monkeypatch
):
    path = tmp_path / "tiny.avi"
    path.write_bytes(b"immutable fixture")
    read_calls = []

    class Capture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def get(self, key):
            return {
                scenario.cv2.CAP_PROP_BACKEND: 0,
                scenario.cv2.CAP_PROP_FRAME_WIDTH: 16,
                scenario.cv2.CAP_PROP_FRAME_HEIGHT: 12,
                scenario.cv2.CAP_PROP_FPS: 20.0,
                scenario.cv2.CAP_PROP_FRAME_COUNT: 2.0,
            }[key]

        def read(self):
            read_calls.append(True)
            return True, np.full((12, 16, 3), len(read_calls), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(scenario.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(
        scenario.cv2,
        "imencode",
        lambda _, image: (
            True,
            np.array([255, int(image[0, 0, 0]), 217], dtype=np.uint8),
        ),
    )
    clock_values = iter([2_000_000_000, 2_000_000_000, 2_050_000_000, 2_100_000_000])
    gate = scenario.ScenarioPlaybackGate()
    backend = scenario.ScenarioVideoCaptureBackend(
        scenario.ScenarioCaptureConfig(
            path,
            RealtimeCaptureConfig(width=16, height=12, fps=20),
        ),
        source_epoch_id="epoch",
        clock_ns=lambda: next(clock_values),
        playback_gate=gate,
    )
    frames = backend.frames(threading.Event())
    try:
        first = next(frames)
        snapshot = backend.startup_diagnostics
        second = next(frames)
        assert backend.startup_diagnostics == snapshot
        gate.release()
        third = next(frames)
        assert [
            (f.capture_frame_id, f.captured_at_ns, f.mjpeg) for f in (first, second, third)
        ] == [
            (0, 2_000_000_000, b"\xff\x01\xd9"),
            (1, 2_050_000_000, b"\xff\x01\xd9"),
            (2, 2_100_000_000, b"\xff\x02\xd9"),
        ]
        assert len(read_calls) == 2
        assert snapshot["stage"] == "frame_ready"
        assert next(clock_values, None) is None
    finally:
        frames.close()


def test_hub_diagnostics_drop_untrusted_source_payloads():
    class Source(GatedSource):
        startup_diagnostics = {
            "stage": "secret-token",
            "thread_id": "private-path",
            "failure_type": "secretCredentialToken",
            "pixels": b"private",
        }

    hub = CaptureHub(source=Source(), capture_config=RealtimeCaptureConfig())
    assert "secret" not in json.dumps(hub.startup_diagnostics)
    assert "private" not in json.dumps(hub.startup_diagnostics)
    hub.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"stage": [], "status": {}, "failure_stage": []},
        {"thread_id": True, "stage_started_at_ns": 2**100, "failure_type": "x" * 1000},
    ],
)
def test_malformed_optional_source_snapshot_cannot_break_bounded_diagnostics(payload):
    source = GatedSource()
    source.startup_diagnostics = payload
    hub = CaptureHub(source=source, capture_config=RealtimeCaptureConfig())
    try:
        assert len(json.dumps(hub.startup_diagnostics)) < 1000
        assert hub.startup_diagnostics["source"] == {
            "stage_finished_at_ns": None,
            "cancel_requested_at_ns": None,
            "released_at_ns": None,
            **({"stage_started_at_ns": None, "thread_id": None} if "stage" in payload else {}),
        }
    finally:
        hub.close()


def test_broken_optional_snapshot_cannot_mask_original_source_failure():
    class Source(GatedSource):
        @property
        def startup_diagnostics(self):
            raise OSError("secret path")

        def frames(self, stop):
            raise ValueError("original failure")
            yield

    source = Source()
    handle = field_runtime.RuntimeHandle(
        source, lambda _: pytest.fail("no frame"), capture_config=RealtimeCaptureConfig()
    )
    try:
        with pytest.raises(
            RuntimeError, match="capture hub did not publish.*original failure"
        ) as e:
            handle.run()
        assert isinstance(e.value.__cause__, ValueError)
        assert "secret" not in str(e.value)
    finally:
        handle.shutdown()


def test_frame_returned_after_deadline_cannot_create_runtime(monkeypatch):
    source = GatedSource()
    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(
            startup_timeout_seconds=0.1,
        ),
    )
    now = [10.0]

    def late_wait(**_):
        now[0] = 10.2
        return _frame()

    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(hub, "wait_for_frame", late_wait)
    handle = field_runtime.RuntimeHandle(
        source, lambda _: pytest.fail("late frame"), capture_hub=hub
    )
    try:
        with pytest.raises(RuntimeError, match="capture hub did not publish a startup frame"):
            handle.run()
    finally:
        handle.shutdown()


@pytest.mark.parametrize(
    ("preparation_seconds", "hash_seconds", "read_seconds", "succeeds"),
    [(20, 6, 0, True), (None, 6, 0, False), (20, 6, 6, False), (20, 21, 0, False)],
)
def test_file_preparation_and_real_first_frame_have_separate_fixed_deadlines(
    tmp_path, monkeypatch, preparation_seconds, hash_seconds, read_seconds, succeeds
):
    """A real hash/decode crosses a controlled clock, without sleeping for 20s."""
    import cv2

    from billiards_ai.lab import mainline1_capture_hub as hubs

    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    config = RealtimeCaptureConfig(width=32, height=24, fps=20)
    source = scenario.ScenarioVideoCaptureBackend(
        scenario.ScenarioCaptureConfig(
            path, config, startup_preparation_seconds=preparation_seconds
        ),
        source_epoch_id="epoch",
        playback_gate=scenario.ScenarioPlaybackGate(),
    )
    now = [100.0]

    def clock_ns():
        return round(now[0] * 1_000_000_000)

    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(hubs, "monotonic_ns", clock_ns)
    monkeypatch.setattr(scenario, "monotonic_ns", clock_ns)
    original_hash, original_capture = scenario._content_hash, cv2.VideoCapture

    def measured_hash(path):
        result = original_hash(path)
        now[0] += hash_seconds
        return result

    class SlowFirstRead:
        def __init__(self, *args):
            self.capture = original_capture(*args)
            self.first = True

        def __getattr__(self, key):
            return getattr(self.capture, key)

        def read(self):
            result = self.capture.read()
            if self.first:
                self.first = False
                now[0] += read_seconds
            return result

    monkeypatch.setattr(scenario, "_content_hash", measured_hash)
    monkeypatch.setattr(cv2, "VideoCapture", SlowFirstRead)
    hub = CaptureHub(source=source, capture_config=config)
    constructed = []

    def runtime(subscription):
        constructed.append(subscription)
        return SimpleNamespace(run=lambda: None)

    handle = field_runtime.RuntimeHandle(source, runtime, capture_hub=hub)
    try:
        if succeeds:
            handle.run()
            assert len(constructed) == 1
            assert hub.latest_frame().capture_frame_id == 0
            assert hub.preparation_completed_at_ns == 106_000_000_000
        else:
            with pytest.raises(RuntimeError, match="capture hub did not publish"):
                handle.run()
            assert constructed == []
        assert not handle.wait_until_ready(timeout_seconds=0.01)
    finally:
        handle.shutdown()


@pytest.mark.parametrize(
    "receipt", [None, True, "bad", -1, 99_000_000_000, 101_000_000_000, 121_000_000_000]
)
def test_explicit_preparation_rejects_missing_invalid_or_out_of_window_receipt(
    monkeypatch, receipt
):
    from billiards_ai.lab import mainline1_capture_hub as hubs

    class Source(GatedSource):
        startup_preparation_seconds = 20
        preparation_completed_at_ns = receipt

    source = Source()
    source.release.set()
    now = [100.0]
    monkeypatch.setattr(hubs, "monotonic_ns", lambda: round(now[0] * 1_000_000_000))
    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))
    handle = field_runtime.RuntimeHandle(
        source, lambda _: pytest.fail("invalid preparation"), capture_config=RealtimeCaptureConfig()
    )
    original_wait = handle.capture_hub.wait_for_frame

    def wait(**kwargs):
        value = original_wait(**kwargs)
        now[0] += 1
        return value

    monkeypatch.setattr(handle.capture_hub, "wait_for_frame", wait)
    try:
        with pytest.raises(RuntimeError, match="capture hub did not publish"):
            handle.run()
    finally:
        handle.shutdown()


def test_shared_prepared_hub_keeps_original_startup_receipt_and_does_not_reprepare(monkeypatch):
    from billiards_ai.lab import mainline1_capture_hub as hubs

    class Source(GatedSource):
        startup_preparation_seconds = 20
        preparation_completed_at_ns = None

        def frames(self, stop):
            now[0] = 106.0
            self.preparation_completed_at_ns = 106_000_000_000
            now[0] = 107.0
            yield _frame()
            stop.wait()

    now = [100.0]
    monkeypatch.setattr(hubs, "monotonic_ns", lambda: round(now[0] * 1_000_000_000))
    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))
    source = Source()
    hub = CaptureHub(source=source, capture_config=RealtimeCaptureConfig())
    created = []

    def runtime(subscription):
        created.append(subscription)
        return SimpleNamespace(run=lambda: None)

    first = field_runtime.RuntimeHandle(source, runtime, capture_hub=hub)
    second = field_runtime.RuntimeHandle(source, runtime, capture_hub=hub)
    try:
        first.run()
        now[0] = 200.0
        second.run()
        assert len(created) == 2
        assert hub.startup_started_at_ns == 100_000_000_000
        assert hub.first_frame_at_ns == 107_000_000_000
        assert hub.preparation_completed_at_ns == 106_000_000_000
    finally:
        first.close()
        second.shutdown()


def test_preparation_without_completion_times_out_at_its_own_bound(monkeypatch):
    from billiards_ai.lab import mainline1_capture_hub as hubs

    source = GatedSource()
    source.startup_preparation_seconds = 20
    source.preparation_completed_at_ns = None
    now = [100.0]
    monkeypatch.setattr(hubs, "monotonic_ns", lambda: round(now[0] * 1_000_000_000))
    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))
    handle = field_runtime.RuntimeHandle(
        source,
        lambda _: pytest.fail("unfinished preparation"),
        capture_config=RealtimeCaptureConfig(),
    )

    def wait(*, after_frame_id, timeout_seconds):
        now[0] += timeout_seconds
        return None

    monkeypatch.setattr(handle.capture_hub, "wait_for_frame", wait)
    try:
        with pytest.raises(RuntimeError, match="capture hub did not publish"):
            handle.run()
        assert now[0] == pytest.approx(120.0)
        assert handle.capture_hub.latest_frame() is None
    finally:
        handle.shutdown()


def test_preparation_cancellation_retains_blocked_source_until_shutdown_can_join(
    tmp_path, monkeypatch
):
    from billiards_ai.lab.mainline1_realtime_capture import CaptureContractError

    path = tmp_path / "source.avi"
    path.write_bytes(b"hash only: cancellation must not open the decoder")
    source = scenario.ScenarioVideoCaptureBackend(
        scenario.ScenarioCaptureConfig(
            path,
            RealtimeCaptureConfig(cancel_timeout_seconds=0.01),
            startup_preparation_seconds=20,
        ),
        source_epoch_id="epoch",
    )
    original_hash = scenario._content_hash
    entered, release = threading.Event(), threading.Event()

    def blocked_hash(path):
        entered.set()
        assert release.wait(2)
        return original_hash(path)

    monkeypatch.setattr(scenario, "_content_hash", blocked_hash)
    monkeypatch.setattr(scenario.cv2, "VideoCapture", lambda *_: pytest.fail("cancelled decoder"))
    handle = field_runtime.RuntimeHandle(
        source,
        lambda _: pytest.fail("cancelled preparation"),
        capture_config=source._config,
    )
    thread, failures = _launch(handle.run)
    try:
        assert entered.wait(0.5)
        handle.close()
        thread.join(0.2)
        assert not thread.is_alive()
        assert failures == []
        with pytest.raises(CaptureContractError, match="capture hub did not stop"):
            handle.shutdown()
        assert handle.capture_hub._source is source
        assert handle.capture_hub._thread.is_alive()
        assert source.preparation_completed_at_ns is None
        release.set()
        handle.shutdown()
        assert not handle.capture_hub._thread.is_alive()
        assert handle.capture_hub.latest_frame() is None
    finally:
        release.set()
        thread.join(1)
        handle.shutdown()


@pytest.mark.parametrize(
    ("completed_seconds", "first_seconds", "parent_seconds", "succeeds"),
    [(106, 107, 107, True), (106, 107, 200, True), (121, 122, 200, False), (106, 112, 200, False)],
)
def test_completion_published_between_receipt_and_frame_reads_keeps_original_deadlines(
    monkeypatch, completed_seconds, first_seconds, parent_seconds, succeeds
):
    from billiards_ai.lab import mainline1_capture_hub as hubs

    allowed, published = threading.Event(), threading.Event()
    now = [100.0]
    monkeypatch.setattr(hubs, "monotonic_ns", lambda: round(now[0] * 1_000_000_000))
    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class Source(GatedSource):
        startup_preparation_seconds = 20
        preparation_completed_at_ns = None

        def frames(self, stop):
            assert allowed.wait(1)
            now[0] = completed_seconds
            self.preparation_completed_at_ns = round(now[0] * 1_000_000_000)
            now[0] = first_seconds
            yield _frame()
            published.set()
            stop.wait()

    class InterleavedHub(CaptureHub):
        interleaved = False

        @property
        def preparation_completed_at_ns(self):
            value = super().preparation_completed_at_ns
            if value is None and not self.interleaved:
                self.interleaved = True
                allowed.set()
                assert published.wait(1)
                now[0] = parent_seconds
            return value

    source = Source()
    hub = InterleavedHub(source=source, capture_config=RealtimeCaptureConfig())
    created = []

    def runtime(subscription):
        created.append(subscription)
        return SimpleNamespace(run=lambda: None)

    handle = field_runtime.RuntimeHandle(source, runtime, capture_hub=hub)
    try:
        if succeeds:
            handle.run()
            assert len(created) == 1
        else:
            with pytest.raises(RuntimeError, match="capture hub did not publish"):
                handle.run()
            assert created == []
        assert hub.startup_started_at_ns == 100_000_000_000
        assert hub.preparation_completed_at_ns == completed_seconds * 1_000_000_000
        assert hub.first_frame_at_ns == first_seconds * 1_000_000_000
    finally:
        allowed.set()
        handle.shutdown()


@pytest.mark.parametrize(("first_seconds", "succeeds"), [(107, True), (112, False)])
def test_first_frame_published_before_deadline_is_not_rejected_by_stale_empty_read(
    monkeypatch, first_seconds, succeeds
):
    from billiards_ai.lab import mainline1_capture_hub as hubs

    prepared, allowed, published = (threading.Event() for _ in range(3))
    now = [100.0]
    monkeypatch.setattr(hubs, "monotonic_ns", lambda: round(now[0] * 1_000_000_000))
    monkeypatch.setattr(field_runtime, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class Source(GatedSource):
        startup_preparation_seconds = 20
        preparation_completed_at_ns = None

        def frames(self, stop):
            now[0] = 106.0
            self.preparation_completed_at_ns = 106_000_000_000
            prepared.set()
            assert allowed.wait(1)
            now[0] = first_seconds
            yield _frame()
            published.set()
            stop.wait()

    class InterleavedHub(CaptureHub):
        interleaved = False

        @property
        def preparation_completed_at_ns(self):
            assert prepared.wait(1)
            return super().preparation_completed_at_ns

        @property
        def first_frame_at_ns(self):
            value = super().first_frame_at_ns
            if value is None and not self.interleaved:
                self.interleaved = True
                allowed.set()
                assert published.wait(1)
                now[0] = 200.0
            return value

    source = Source()
    hub = InterleavedHub(source=source, capture_config=RealtimeCaptureConfig())
    created = []

    def runtime(subscription):
        created.append(subscription)
        return SimpleNamespace(run=lambda: None)

    handle = field_runtime.RuntimeHandle(source, runtime, capture_hub=hub)
    try:
        if succeeds:
            handle.run()
            assert len(created) == 1
        else:
            with pytest.raises(RuntimeError, match="capture hub did not publish"):
                handle.run()
            assert created == []
        assert hub.first_frame_at_ns == first_seconds * 1_000_000_000
    finally:
        allowed.set()
        handle.shutdown()
