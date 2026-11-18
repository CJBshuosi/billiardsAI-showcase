from __future__ import annotations

import importlib
import importlib.util
import threading
from collections.abc import Iterator

import pytest

from billiards_ai.lab.mainline1_native_capture import NativeCaptureProtocolError
from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureContractError,
    CaptureFrame,
    CaptureHealthState,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)


def _frame(frame_id: int) -> CaptureFrame:
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 16_666_667,
        width=16,
        height=12,
        mjpeg=f"jpeg-{frame_id}".encode(),
    )


class _FiniteCaptureSource:
    def __init__(self, frames: tuple[CaptureFrame, ...]) -> None:
        self._frames = frames
        self.started = threading.Event()
        self.emitted = threading.Event()
        self.frames_calls = 0
        self.cancel_calls = 0

    @property
    def identity(self) -> NativeCaptureIdentity:
        return NativeCaptureIdentity(
            device_locator="pnp://rer-u3",
            native_handle_ref="mf://rer-u3-symbolic-link",
            durable_fingerprint="c" * 64,
        )

    def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
        self.frames_calls += 1
        self.started.set()
        for frame in self._frames:
            if stop.is_set():
                return
            yield frame
        self.emitted.set()
        stop.wait(timeout=1)

    def cancel(self) -> None:
        self.cancel_calls += 1


@pytest.mark.parametrize("disk_input", [False, True])
def test_recording_seal_waits_for_append_and_preserves_preview(disk_input):
    from billiards_ai.lab.mainline1_capture_hub import CaptureHub

    append_entered, append_release, next_frame = (threading.Event() for _ in range(3))

    class Source(_FiniteCaptureSource):
        def frames(self, stop):
            yield _frame(0)
            next_frame.wait(2)
            yield _frame(1)
            stop.wait(2)

    class Recorder:
        def __init__(self):
            self.frames = []
            self.last_admitted_frame = None

        def append(self, frame):
            append_entered.set()
            append_release.wait(2)
            self.frames.append(frame.capture_frame_id)
            self.last_admitted_frame = frame

        def close(self):
            pass

    recorder = Recorder()
    hub = CaptureHub(
        source=Source(()),
        recorder=recorder,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    fifo = hub.subscribe(
        "recording", delivery_mode="fifo", maximum_pending_frames=4, maximum_pending_bytes=1024
    )
    preview = hub.subscribe("preview", maximum_pending_frames=1)
    fifo.activate()
    if disk_input:
        fifo._fail(CaptureContractError("independent legacy consumer failed"))
    result, error = [], []

    def seal():
        try:
            result.append(
                hub.seal_recording_input(recorder, timeout_seconds=1, on_sealed=lambda: None)
                if disk_input
                else hub.seal_recording(fifo, recorder, timeout_seconds=1, on_sealed=lambda: None)
            )
        except BaseException as exc:
            error.append(exc)

    hub.start()
    thread = threading.Thread(target=seal)
    try:
        assert append_entered.wait(1)
        thread.start()
        thread.join(0.05)
        assert thread.is_alive(), "seal crossed an unfinished append"
        append_release.set()
        thread.join(1)
        assert not error
        assert result[0].capture_frame_id == 0
        next_frame.set()
        assert hub.wait_for_frame(after_frame_id=0, timeout_seconds=1).capture_frame_id == 1
        if not disk_input:
            assert [f.capture_frame_id for f in fifo.frames(threading.Event())] == [0]
        assert recorder.frames == [0]
        assert next(preview.frames(threading.Event())).capture_frame_id == 1
    finally:
        append_release.set()
        next_frame.set()
        hub.close()


def test_slow_analysis_subscription_cannot_delay_or_close_latest_frame() -> None:
    """Catches a downstream consumer owning capture cadence or camera lifetime."""
    module_spec = importlib.util.find_spec("billiards_ai.lab.mainline1_capture_hub")
    assert module_spec is not None, "Capture Hub product boundary is missing"
    capture_hub = importlib.import_module("billiards_ai.lab.mainline1_capture_hub")
    source = _FiniteCaptureSource(tuple(_frame(frame_id) for frame_id in range(5)))
    hub = capture_hub.CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    subscription = hub.subscribe("analysis", maximum_pending_frames=1)

    hub.start()
    latest = hub.wait_for_frame(after_frame_id=3, timeout_seconds=1)
    assert latest is not None and latest.capture_frame_id == 4

    consumer_stop = threading.Event()
    analysis_frame = next(subscription.frames(consumer_stop))
    assert analysis_frame.capture_frame_id == 4
    assert subscription.allow_accounted_frame_gaps is True

    subscription.cancel()
    assert source.cancel_calls == 0
    hub.close()

    assert source.frames_calls == 1
    assert source.cancel_calls == 1


def test_physical_capture_failure_is_reported_without_erasing_last_frame() -> None:
    """Catches camera failure being hidden or destroying the last preview state."""
    capture_hub = importlib.import_module("billiards_ai.lab.mainline1_capture_hub")

    class FailingSource(_FiniteCaptureSource):
        def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
            self.frames_calls += 1
            yield _frame(0)
            raise RuntimeError("camera cable lost")

    source = FailingSource(())
    hub = capture_hub.CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )

    hub.start()
    first = hub.wait_for_frame(after_frame_id=None, timeout_seconds=1)
    assert first is not None and first.capture_frame_id == 0
    assert hub.wait_for_frame(after_frame_id=0, timeout_seconds=1) is None

    health = hub.capture_health
    assert health.state is CaptureHealthState.DEGRADED
    assert health.received_frames == 1
    assert health.reason == "RuntimeError: camera cable lost"
    assert hub.latest_frame() == first

    hub.close()


def test_physical_capture_failure_terminates_subscriptions_with_original_error() -> None:
    """Catches CaptureHub hiding a dead camera from the recovery authority."""
    capture_hub = importlib.import_module("billiards_ai.lab.mainline1_capture_hub")

    class FailingSource(_FiniteCaptureSource):
        def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
            self.frames_calls += 1
            yield _frame(0)
            raise NativeCaptureProtocolError("native camera read exceeded timeout")

    source = FailingSource(())
    hub = capture_hub.CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    subscription = hub.subscribe("analysis", maximum_pending_frames=1)
    consumer_stop = threading.Event()
    frames = subscription.frames(consumer_stop)

    hub.start()
    assert next(frames) == _frame(0)
    failures: list[BaseException] = []
    finished = threading.Event()

    def consume_failure() -> None:
        try:
            next(frames)
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    consumer = threading.Thread(target=consume_failure, daemon=True)
    consumer.start()
    try:
        assert finished.wait(timeout=1), "capture failure did not wake the subscription"
        assert len(failures) == 1
        assert isinstance(failures[0], NativeCaptureProtocolError)
        assert str(failures[0]) == "native camera read exceeded timeout"
    finally:
        hub.close()
        consumer.join(timeout=1)


def test_recorder_failure_is_isolated_while_capture_keeps_advancing() -> None:
    """Catches evidence storage failure cancelling a still-healthy live camera."""
    capture_hub = importlib.import_module("billiards_ai.lab.mainline1_capture_hub")

    class FailingRecorder:
        def __init__(self) -> None:
            self.frame_ids: list[int] = []
            self.close_calls = 0

        def append(self, frame: CaptureFrame) -> None:
            self.frame_ids.append(frame.capture_frame_id)
            if frame.capture_frame_id == 1:
                raise OSError("disk stalled")

        def close(self) -> None:
            self.close_calls += 1

    source = _FiniteCaptureSource(tuple(_frame(frame_id) for frame_id in range(5)))
    recorder = FailingRecorder()
    hub = capture_hub.CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        recorder=recorder,
    )

    hub.start()
    latest = hub.wait_for_frame(after_frame_id=3, timeout_seconds=1)

    assert latest is not None and latest.capture_frame_id == 4
    assert recorder.frame_ids == [0, 1]
    assert hub.recording_failure_reason == "OSError: disk stalled"
    assert source.cancel_calls == 0
    hub.close()
    assert recorder.close_calls == 1


def test_recorder_can_roll_to_a_new_session_without_reopening_camera() -> None:
    """Catches match rollover closing and reopening the physical USB source."""
    capture_hub = importlib.import_module("billiards_ai.lab.mainline1_capture_hub")
    release_second = threading.Event()

    class GatedSource(_FiniteCaptureSource):
        def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
            self.frames_calls += 1
            yield _frame(0)
            assert release_second.wait(timeout=1)
            yield _frame(1)
            stop.wait(timeout=1)

    class Recorder:
        def __init__(self) -> None:
            self.frames: list[int] = []
            self.close_calls = 0

        def append(self, frame: CaptureFrame) -> None:
            self.frames.append(frame.capture_frame_id)

        def close(self) -> None:
            self.close_calls += 1

    source = GatedSource(())
    first = Recorder()
    second = Recorder()
    hub = capture_hub.CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        recorder=first,
    )
    hub.start()
    assert hub.wait_for_frame(after_frame_id=None, timeout_seconds=1) == _frame(0)

    hub.replace_recorder(second)
    release_second.set()
    assert hub.wait_for_frame(after_frame_id=0, timeout_seconds=1) == _frame(1)

    assert first.frames == [0]
    assert first.close_calls == 1
    assert second.frames == [1]
    assert source.frames_calls == 1
    assert source.cancel_calls == 0
    hub.close()
    assert second.close_calls == 1
    assert source.cancel_calls == 1


@pytest.mark.parametrize(
    ("frame_cap", "byte_cap", "reason"),
    [(2, 100, "frame capacity"), (10, 15, "byte capacity"), (10, 5, "byte capacity")],
)
def test_fifo_overflow_fails_only_analysis_before_delivering_a_prefix(
    frame_cap,
    byte_cap,
    reason,
) -> None:
    from billiards_ai.lab.mainline1_capture_hub import CaptureHub

    originals = tuple(_frame(i) for i in range(5))
    source = _FiniteCaptureSource(originals)
    recorded = []

    class Recorder:
        def append(self, frame):
            recorded.append(frame)

        def close(self):
            pass

    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        recorder=Recorder(),
    )
    analysis = hub.subscribe(
        "analysis",
        maximum_pending_frames=frame_cap,
        delivery_mode="fifo",
        maximum_pending_bytes=byte_cap,
    )
    preview = hub.subscribe("preview", maximum_pending_frames=1)
    analysis.activate()
    try:
        hub.start()
        assert source.emitted.wait(1)
        assert analysis.pending_frame_count == analysis.pending_bytes == 0
        rows = analysis.frames(threading.Event())
        with pytest.raises(CaptureContractError, match=reason):
            next(rows)
        # Sticky failure is retained even after reading/closing the iterator.
        with pytest.raises(CaptureContractError, match=reason):
            analysis.activate()
        assert next(preview.frames(threading.Event())) is originals[-1]
        assert len(recorded) == len(originals)
        assert all(a is b for a, b in zip(recorded, originals, strict=True))
        assert hub.latest_frame() is originals[-1]
        assert hub.failure_reason is None
        assert source.cancel_calls == 0
    finally:
        hub.close()


@pytest.mark.parametrize("ending", ["cancel", "stop", "close", "eof"])
def test_fifo_read_and_termination_release_pending_frames_and_bytes(ending) -> None:
    from billiards_ai.lab.mainline1_capture_hub import CaptureSubscription

    subscription = CaptureSubscription(
        name="analysis",
        maximum_pending_frames=3,
        delivery_mode="fifo",
        maximum_pending_bytes=100,
    )
    originals = [_frame(i) for i in range(3)]
    subscription.activate()
    for frame in originals:
        subscription._publish(frame)
    stop = threading.Event()
    rows = subscription.frames(stop)
    assert next(rows) is originals[0]
    assert subscription.pending_frame_count == 2
    assert subscription.pending_bytes == sum(len(f.mjpeg) for f in originals[1:])
    if ending == "cancel":
        subscription.cancel()
    elif ending == "stop":
        stop.set()
    elif ending == "close":
        rows.close()
    else:
        subscription._fail(CaptureContractError("capture source ended"))
        assert next(rows) is originals[1]
        assert next(rows) is originals[2]
        with pytest.raises(CaptureContractError, match="capture source ended"):
            next(rows)
    if ending != "eof":
        with pytest.raises(StopIteration):
            next(rows)
    assert subscription.pending_frame_count == subscription.pending_bytes == 0


@pytest.mark.parametrize("source_error", [None, NativeCaptureProtocolError("cable lost")])
def test_fifo_drains_complete_captured_tail_then_exposes_source_termination(source_error):
    from billiards_ai.lab.mainline1_capture_hub import CaptureHub

    originals = tuple(_frame(i) for i in range(3))

    class CompletingSource(_FiniteCaptureSource):
        def frames(self, stop):
            yield from originals
            if source_error is not None:
                raise source_error

    source = CompletingSource(())
    hub = CaptureHub(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    analysis = hub.subscribe(
        "analysis",
        maximum_pending_frames=3,
        delivery_mode="fifo",
        maximum_pending_bytes=100,
    )
    analysis.activate()
    rows = analysis.frames(threading.Event())
    try:
        hub.start()
        for original in originals:
            assert next(rows) is original
        with pytest.raises((CaptureContractError, NativeCaptureProtocolError)) as exc:
            next(rows)
        if source_error is not None:
            assert exc.value is source_error
        else:
            assert "ended" in str(exc.value)
        assert analysis.pending_frame_count == analysis.pending_bytes == 0
    finally:
        hub.close()


@pytest.mark.parametrize("failure", ["fifo_overflow", "writer"])
def test_failed_recording_cannot_be_sealed_as_normal_end(failure):
    from billiards_ai.lab.mainline1_capture_hub import CaptureHub

    class Recorder:
        def append(self, frame):
            if failure == "writer":
                raise OSError("injected recorder append failure")

        def close(self):
            pass

    source = _FiniteCaptureSource(tuple(_frame(i) for i in range(3)))
    recorder = Recorder()
    hub = CaptureHub(
        source=source,
        recorder=recorder,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    fifo = hub.subscribe(
        "recording", delivery_mode="fifo", maximum_pending_frames=1, maximum_pending_bytes=1024
    )
    fifo.activate()
    marked = []
    try:
        hub.start()
        assert source.emitted.wait(1)
        with pytest.raises(CaptureContractError):
            hub.seal_recording(
                fifo, recorder, timeout_seconds=0.1, on_sealed=lambda: marked.append(True)
            )
        assert not marked
        assert hub._uses_recorder(recorder)
    finally:
        hub.close()


def test_recording_seal_cleanup_lock_is_bounded_and_does_not_detach_writer():
    import time

    from billiards_ai.lab.mainline1_capture_hub import CaptureHub

    source = _FiniteCaptureSource(())
    hub = CaptureHub(
        source=source, capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60)
    )
    fifo = hub.subscribe(
        "recording", delivery_mode="fifo", maximum_pending_frames=1, maximum_pending_bytes=1024
    )
    fifo.activate()
    acquired, release = threading.Event(), threading.Event()

    def cleanup():
        with hub._close_lock:
            acquired.set()
            release.wait(1)

    thread = threading.Thread(target=cleanup)
    thread.start()
    try:
        assert acquired.wait(1)
        started = time.monotonic()
        with pytest.raises(CaptureContractError, match="timed out"):
            hub.seal_recording(fifo, None, timeout_seconds=0.02, on_sealed=lambda: None)
        assert time.monotonic() - started < 0.5
        assert not fifo._sealed
    finally:
        release.set()
        thread.join(1)
        hub.close()
