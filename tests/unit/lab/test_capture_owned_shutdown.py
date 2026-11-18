"""Owned cleanup retries must continue the original capture and writer work."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab import mainline1_realtime_ring as ring_module
from billiards_ai.lab.mainline1_capture_hub import CaptureHub
from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureContractError,
    CaptureFrame,
    RealtimeCaptureConfig,
)
from billiards_ai.lab.mainline1_realtime_ring import (
    AsyncMjpegSegmentRing,
    MjpegSegmentRing,
    RealtimeRingError,
)


def _frame(frame_id: int = 0) -> CaptureFrame:
    return CaptureFrame(
        source_epoch_id="epoch-owned",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 500_000_000,
        width=16,
        height=12,
        mjpeg=b"jpeg",
    )


class _GatedSource:
    def __init__(self, *, cancel_failure: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancel_calls = 0
        self.cancel_failure = cancel_failure

    def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
        self.entered.set()
        assert self.release.wait(2)
        if False:
            yield _frame()

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.cancel_failure:
            raise OSError("cancel failed")


class _Recorder:
    def __init__(self, *, failure: bool = False) -> None:
        self.close_calls = 0
        self.failure = failure
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def close(self) -> None:
        self.close_calls += 1
        self.entered.set()
        assert self.release.wait(2)
        if self.failure:
            raise OSError("recorder failed")


def _hub(source: _GatedSource, recorder: _Recorder) -> CaptureHub:
    return CaptureHub(
        source=source,
        recorder=recorder,
        capture_config=RealtimeCaptureConfig(
            width=16,
            height=12,
            fps=2,
            cancel_timeout_seconds=0.01,
        ),
    )


def test_hub_retry_joins_original_worker_and_finalizes_once(monkeypatch) -> None:
    """Catches revocation skipping pending join and recorder/monitor cleanup."""
    source, recorder = _GatedSource(), _Recorder()
    hub = _hub(source, recorder)
    subscription = hub.subscribe("analysis", maximum_pending_frames=1)
    monitor_closes = []
    original_close = hub._monitor.close
    monkeypatch.setattr(hub._monitor, "close", lambda: (monitor_closes.append(1), original_close()))
    hub.start()
    assert source.entered.wait(1)
    original_thread = hub._thread
    try:
        with pytest.raises(CaptureContractError, match="did not stop"):
            hub.close()
        cancelled_at = hub.startup_diagnostics["cancel_requested_at_ns"]
        assert list(subscription.frames(threading.Event())) == []
        with pytest.raises(CaptureContractError, match="closed"):
            hub.subscribe("late", maximum_pending_frames=1)
        assert recorder.close_calls == 0
        source.release.set()
        hub.close()
        assert hub._thread is original_thread
        assert not original_thread.is_alive()
        assert recorder.close_calls == 1
        assert monitor_closes == [1]
        hub.close()
        assert recorder.close_calls == 1
        assert source.cancel_calls == 1
        assert hub.startup_diagnostics["cancel_requested_at_ns"] == cancelled_at
    finally:
        source.release.set()
        original_thread.join(1)


def test_hub_cancel_failure_does_not_abandon_owned_cleanup() -> None:
    """Catches a failed cancel preventing join and recorder finalization."""
    source, recorder = _GatedSource(cancel_failure=True), _Recorder()
    hub = _hub(source, recorder)
    hub.start()
    assert source.entered.wait(1)
    source.release.set()
    for _ in range(2):
        with pytest.raises(OSError, match="cancel failed"):
            hub.close()
    assert not hub._thread.is_alive()
    assert recorder.close_calls == 1
    assert source.cancel_calls == 1


def test_hub_recorder_failure_remains_visible_and_retryable() -> None:
    """Catches silently accepting failed physical recorder finalization."""
    source, recorder = _GatedSource(), _Recorder(failure=True)
    hub = _hub(source, recorder)
    with pytest.raises(OSError, match="recorder failed"):
        hub.close()
    assert hub.recording_failure_reason == "OSError: recorder failed"
    recorder.failure = False
    hub.close()
    hub.close()
    assert recorder.close_calls == 2


def _writer(tmp_path: Path, monkeypatch, *, capacity: int = 1):
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=2,
        pre_roll_frames=2,
        post_roll_frames=0,
        maximum_shot_frames=10,
        maximum_bytes=1024 * 1024,
    )
    entered, release = threading.Event(), threading.Event()
    closes = []
    original_close = ring.close

    def gated_close() -> None:
        closes.append(1)
        entered.set()
        assert release.wait(2)
        original_close()

    monkeypatch.setattr(ring, "close", gated_close)
    writer = AsyncMjpegSegmentRing(ring, maximum_pending_frames=capacity)
    return writer, ring, entered, release, closes


def test_writer_retries_original_queued_close_without_false_success(tmp_path, monkeypatch) -> None:
    """Catches a timed-out close forgetting its receipt while the writer lives."""
    writer, _, entered, release, closes = _writer(tmp_path, monkeypatch)
    try:
        with pytest.raises(RealtimeRingError, match="timeout"):
            writer.close(timeout_seconds=0.01)
        assert entered.wait(1)
        with pytest.raises(RealtimeRingError, match="closed"):
            writer.append(_frame())
        with pytest.raises(RealtimeRingError, match="timeout"):
            writer.close(timeout_seconds=0.01)
        assert writer._thread.is_alive()
        release.set()
        writer.close(timeout_seconds=1)
        writer.close(timeout_seconds=0.01)
        assert not writer._thread.is_alive()
        assert closes == [1]
        assert writer._commands.empty()
    finally:
        release.set()
        writer._thread.join(1)


def test_writer_queue_full_before_close_enqueue_can_retry(tmp_path, monkeypatch) -> None:
    """Catches a full queue permanently losing the one final close command."""
    writer, ring, _, release_close, closes = _writer(tmp_path, monkeypatch)
    append_entered, release_append = threading.Event(), threading.Event()
    original_append = ring.append

    def gated_append(frame) -> None:
        append_entered.set()
        assert release_append.wait(2)
        original_append(frame)

    monkeypatch.setattr(ring, "append", gated_append)
    try:
        writer.append(_frame(0))
        assert append_entered.wait(1)
        writer.append(_frame(1))
        with pytest.raises(RealtimeRingError, match="capacity"):
            writer.close(timeout_seconds=0.01)
        # Health must remain readable while a close waits on a full queue.
        assert writer.health.stored_frames == 0
        release_append.set()
        release_close.set()
        writer.close(timeout_seconds=1)
        assert not writer._thread.is_alive()
        assert closes == [1]
        assert writer.health.stored_frames == 2
    finally:
        release_append.set()
        release_close.set()
        # Baseline leaves the worker waiting forever: explicit test-only stop.
        if writer._thread.is_alive():
            from billiards_ai.lab.mainline1_realtime_ring import _AsyncRingCommand

            writer._commands.put(_AsyncRingCommand("close"), timeout=1)
        writer._thread.join(1)


@pytest.mark.parametrize("fail_operation", ["append", "close"])
def test_writer_failure_stays_visible_after_owned_thread_stops(
    tmp_path,
    monkeypatch,
    fail_operation,
) -> None:
    """Catches prior writer errors blocking its close or disappearing on repeat."""
    writer, ring, entered, release, closes = _writer(tmp_path, monkeypatch)
    failed = threading.Event()

    def fail(*args) -> None:
        failed.set()
        raise OSError("storage failed")

    monkeypatch.setattr(ring, fail_operation, fail)
    try:
        if fail_operation == "append":
            writer.append(_frame())
            assert failed.wait(1)
        release.set()
        for _ in range(2):
            with pytest.raises(RealtimeRingError, match="storage failed"):
                writer.close(timeout_seconds=0.1)
        assert not writer._thread.is_alive()
        assert closes == ([1] if fail_operation == "append" else [])
    finally:
        release.set()
        if writer._thread.is_alive():
            from billiards_ai.lab.mainline1_realtime_ring import _AsyncRingCommand

            writer._commands.put(_AsyncRingCommand("close"), timeout=1)
        writer._thread.join(1)


def _call_close(owner, errors: list[BaseException], done: threading.Event, **kwargs) -> None:
    try:
        owner.close(**kwargs)
    except BaseException as exc:
        errors.append(exc)
    finally:
        done.set()


def test_concurrent_hub_close_waits_without_holding_hub_locks() -> None:
    """Catches concurrent close reporting success before recorder completion."""
    source, recorder = _GatedSource(), _Recorder()
    recorder.release.clear()
    hub = _hub(source, recorder)
    errors, first_done, second_done = [], threading.Event(), threading.Event()
    first = threading.Thread(target=_call_close, args=(hub, errors, first_done), daemon=True)
    second = threading.Thread(target=_call_close, args=(hub, errors, second_done), daemon=True)
    first.start()
    assert recorder.entered.wait(1)
    try:
        second.start()
        assert not second_done.wait(0.02)
        assert hub._condition.acquire(timeout=0.1)
        hub._condition.release()
        assert hub._recorder_lock.acquire(timeout=0.1)
        hub._recorder_lock.release()
        assert hub.startup_diagnostics["cancel_requested_at_ns"] is not None
    finally:
        recorder.release.set()
        first.join(1)
        second.join(1)
    assert first_done.is_set() and second_done.is_set()
    assert errors == []
    assert recorder.close_calls == 1


def test_concurrent_writer_close_waits_for_original_completion(tmp_path, monkeypatch) -> None:
    """Catches concurrent callers accepting a still-live writer as closed."""
    writer, _, entered, release, closes = _writer(tmp_path, monkeypatch)
    errors, done = [], threading.Event()
    first = threading.Thread(
        target=_call_close,
        args=(writer, errors, done),
        kwargs={"timeout_seconds": 1},
        daemon=True,
    )
    first.start()
    assert entered.wait(1)
    try:
        with pytest.raises(RealtimeRingError, match="timeout"):
            writer.close(timeout_seconds=0.01)
        assert not done.is_set()
        assert writer.health.stored_frames == 0
    finally:
        release.set()
        first.join(1)
    assert errors == []
    assert closes == [1]
    assert not writer._thread.is_alive()


def test_append_admission_cannot_land_behind_final_close(tmp_path, monkeypatch) -> None:
    """Catches checking revocation separately from bounded-queue admission."""
    writer, _, close_entered, release_close, closes = _writer(tmp_path, monkeypatch)
    admission_entered, release_admission = threading.Event(), threading.Event()
    original_put = writer._commands.put_nowait

    def gated_put(command) -> None:
        if command.operation == "append":
            admission_entered.set()
            assert release_admission.wait(2)
        original_put(command)

    monkeypatch.setattr(writer._commands, "put_nowait", gated_put)
    errors, done = [], threading.Event()

    def append() -> None:
        try:
            writer.append(_frame())
        except BaseException as exc:
            errors.append(exc)

    appender = threading.Thread(target=append, daemon=True)
    closer = threading.Thread(
        target=_call_close,
        args=(writer, errors, done),
        kwargs={"timeout_seconds": 1},
        daemon=True,
    )
    appender.start()
    assert admission_entered.wait(1)
    closer.start()
    try:
        # If admission owns its check through enqueue, close cannot pass it.
        assert not close_entered.wait(0.02)
    finally:
        release_admission.set()
        release_close.set()
        appender.join(1)
        closer.join(1)
    assert errors == []
    assert done.is_set()
    assert closes == [1]
    assert writer.health.stored_frames == 1
    assert writer._commands.empty()


def test_writer_join_uses_remaining_close_budget(tmp_path, monkeypatch) -> None:
    """Catches restarting the full timeout after queue/completion have used it."""
    writer, ring, _, release, _ = _writer(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(ring_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_put, original_close = writer._commands.put, ring.close
    original_join = writer._thread.join
    join_budgets = []

    def put(command, **kwargs) -> None:
        clock[0] = 3.0
        original_put(command, **kwargs)

    def close() -> None:
        original_close()
        clock[0] = 7.0

    def join(*, timeout) -> None:
        join_budgets.append(timeout)
        original_join(timeout=timeout)

    monkeypatch.setattr(writer._commands, "put", put)
    monkeypatch.setattr(ring, "close", close)
    monkeypatch.setattr(writer._thread, "join", join)
    release.set()
    try:
        writer.close(timeout_seconds=10)
        assert join_budgets == [3.0]
    finally:
        release.set()
        original_join(timeout=1)


def test_hub_retains_both_cancel_and_recorder_failures() -> None:
    """Catches one failed owner cleanup masking another cleanup failure."""
    hub = _hub(_GatedSource(cancel_failure=True), _Recorder(failure=True))
    with pytest.raises(BaseExceptionGroup) as caught:
        hub.close()
    assert [str(error) for error in caught.value.exceptions] == ["cancel failed", "recorder failed"]


def test_writer_queue_wait_does_not_hold_state_lock(tmp_path, monkeypatch) -> None:
    """Catches close blocking health/writer completion while its queue is full."""
    writer, ring, _, release_close, _ = _writer(tmp_path, monkeypatch)
    append_entered, release_append, queue_wait = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_append, original_put = ring.append, writer._commands.put

    def append(frame) -> None:
        append_entered.set()
        assert release_append.wait(2)
        original_append(frame)

    def put(command, **kwargs) -> None:
        if command.operation == "close":
            queue_wait.set()
        original_put(command, **kwargs)

    monkeypatch.setattr(ring, "append", append)
    monkeypatch.setattr(writer._commands, "put", put)
    errors, done = [], threading.Event()
    closer = threading.Thread(
        target=_call_close,
        args=(writer, errors, done),
        kwargs={"timeout_seconds": 1},
        daemon=True,
    )
    try:
        writer.append(_frame(0))
        assert append_entered.wait(1)
        writer.append(_frame(1))
        closer.start()
        assert queue_wait.wait(1)
        assert writer._lock.acquire(timeout=0.02)
        writer._lock.release()
        assert not done.is_set()
    finally:
        release_append.set()
        release_close.set()
        closer.join(1)
        writer._thread.join(1)
    assert errors == []
    assert not writer._thread.is_alive()


def test_writer_completion_does_not_imply_worker_has_exited(tmp_path, monkeypatch) -> None:
    """Catches returning after a receipt while its owning worker is still alive."""
    writer, _, _, release_close, closes = _writer(tmp_path, monkeypatch)
    receipt_set, release_worker = threading.Event(), threading.Event()
    original_put = writer._commands.put

    class Completion:
        def __init__(self, completed):
            self.completed = completed

        def wait(self, timeout):
            return self.completed.wait(timeout)

        def set(self):
            self.completed.set()
            receipt_set.set()
            assert release_worker.wait(2)

    def put(command, **kwargs):
        if command.operation == "close":
            command.completed = Completion(command.completed)
        original_put(command, **kwargs)

    monkeypatch.setattr(writer._commands, "put", put)
    release_close.set()
    try:
        with pytest.raises(RealtimeRingError, match="did not stop"):
            writer.close(timeout_seconds=0.01)
        assert receipt_set.is_set()
        assert writer._thread.is_alive()
        with pytest.raises(RealtimeRingError, match="did not stop"):
            writer.close(timeout_seconds=0.01)
        release_worker.set()
        writer.close(timeout_seconds=1)
        assert not writer._thread.is_alive()
        assert closes == [1]
    finally:
        release_worker.set()
        writer._thread.join(1)


def test_writer_close_budget_includes_admission_serialization(tmp_path, monkeypatch) -> None:
    """Catches an unbounded state-lock wait before the close deadline starts."""
    writer, _, _, release_close, _ = _writer(tmp_path, monkeypatch)
    errors, done = [], threading.Event()
    closer = threading.Thread(
        target=_call_close,
        args=(writer, errors, done),
        kwargs={"timeout_seconds": 0.01},
        daemon=True,
    )
    writer._lock.acquire()
    try:
        closer.start()
        assert done.wait(0.1), "close ignored its budget while serializing admission"
        assert len(errors) == 1
        assert isinstance(errors[0], RealtimeRingError)
        assert "timeout" in str(errors[0])
    finally:
        writer._lock.release()
        release_close.set()
        closer.join(1)
        writer.close(timeout_seconds=1)


@pytest.mark.parametrize("pending_phase", ["queue_full", "queued_receipt", "failed_receipt"])
def test_hub_rollover_retains_original_retired_writer(tmp_path, monkeypatch, pending_phase) -> None:
    """Catches rollover abandoning an old writer after adopting its successor."""
    old, ring, close_entered, release_close, old_closes = _writer(tmp_path / "old", monkeypatch)
    new, _, _, release_new, new_closes = _writer(tmp_path / "new", monkeypatch)
    release_new.set()
    append_entered, release_append = threading.Event(), threading.Event()
    original_append, original_old_close = ring.append, old.close

    def append(frame) -> None:
        append_entered.set()
        assert release_append.wait(2)
        original_append(frame)

    def close_failure() -> None:
        old_closes.append(1)
        raise OSError("retired storage failed")

    monkeypatch.setattr(old, "close", lambda: original_old_close(timeout_seconds=0.01))
    if pending_phase == "queue_full":
        monkeypatch.setattr(ring, "append", append)
        old.append(_frame(0))
        assert append_entered.wait(1)
        old.append(_frame(1))
    elif pending_phase == "failed_receipt":
        monkeypatch.setattr(ring, "close", close_failure)
    hub = _hub(_GatedSource(), old)
    try:
        with pytest.raises(RealtimeRingError):
            hub.replace_recorder(new)
        assert hub._recorder is new
        if pending_phase == "queued_receipt":
            assert close_entered.wait(1)
        # Current cleanup must not hide an unresolved or failed retired owner.
        with pytest.raises(RealtimeRingError):
            hub.close()
        assert not new._thread.is_alive()
        assert new_closes == [1]
        if pending_phase == "failed_receipt":
            with pytest.raises(RealtimeRingError, match="retired storage failed"):
                hub.close()
        else:
            release_append.set()
            release_close.set()
            monkeypatch.setattr(old, "close", lambda: original_old_close(timeout_seconds=1))
            hub.close()
            hub.close()
        assert not old._thread.is_alive()
        assert old_closes == [1]
        assert old._commands.empty()
        assert new_closes == [1]
    finally:
        release_append.set()
        release_close.set()
        for writer in (old, new):
            try:
                AsyncMjpegSegmentRing.close(writer, timeout_seconds=1)
            except RealtimeRingError:
                pass
            writer._thread.join(1)
