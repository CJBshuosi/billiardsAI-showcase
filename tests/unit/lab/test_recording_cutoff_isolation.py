"""A stopped original input owns its cutoff independently of later preview faults."""

import threading

from billiards_ai.lab.mainline1_session_recording import CommittedRecordingSource
from tests.unit.lab.test_recording_committed_source import _frame, _recorder


def test_sealed_recording_waits_for_its_tail_despite_later_preview_failure(tmp_path):
    raw = _recorder(tmp_path)
    for n in range(61):
        raw.append(_frame(n))
    assert raw.wait_until_committed(59, timeout_seconds=1)
    failure = [None]
    source = CommittedRecordingSource(
        recorder=raw,
        session_id="session-1",
        source_epoch_id="epoch-1",
        capture_failure=lambda: failure[0],
    )
    source.seal(_frame(60))
    rows = source.frames(threading.Event())
    assert [next(rows).capture_frame_id for _ in range(60)] == list(range(60))
    failure[0] = "preview failed after original capture stopped"
    done, errors, tail = threading.Event(), [], []

    def read_tail():
        try:
            tail.extend(rows)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    reader = threading.Thread(target=read_tail)
    reader.start()
    try:
        assert not done.wait(0.25), f"old detector terminated before its tail: {errors}"
    finally:
        raw.close()
        reader.join(1)
    assert not reader.is_alive() and not errors
    assert [frame.capture_frame_id for frame in tail] == [60]
    assert source.terminal_capture_failure is None


def test_a_capture_failure_callback_started_before_seal_cannot_terminalize_after_seal(tmp_path):
    raw = _recorder(tmp_path)
    for n in range(61):
        raw.append(_frame(n))
    assert raw.wait_until_committed(59, timeout_seconds=1)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    errors, delivered = [], []

    def later_preview_failure():
        entered.set()
        assert release.wait(2)
        return "shared preview failed after accepted original cutoff"

    source = CommittedRecordingSource(
        recorder=raw,
        session_id="session-1",
        source_epoch_id="epoch-1",
        capture_failure=later_preview_failure,
    )

    def consume():
        try:
            delivered.extend(frame.capture_frame_id for frame in source.frames(threading.Event()))
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    reader = threading.Thread(target=consume)
    reader.start()
    try:
        assert entered.wait(1)
        source.seal(_frame(60))
        release.set()
        assert not done.wait(0.2), errors
    finally:
        release.set()
        raw.close()
        reader.join(2)
    assert not errors
    assert delivered == list(range(61))
    assert source.terminal_capture_failure is None
