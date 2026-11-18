from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureContractError,
    CaptureFrame,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)
from billiards_ai.lab.mainline1_scenario_capture import (
    ScenarioCaptureConfig,
    ScenarioVideoCaptureBackend,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 2_000_000_000
        self.waits: list[float] = []

    def __call__(self) -> int:
        return self.now

    def wait(self, event: threading.Event, seconds: float) -> bool:
        self.waits.append(seconds)
        self.now += round(seconds * 1_000_000_000)
        return event.is_set()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
    assert writer.isOpened()
    try:
        for value in (30, 100, 180):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _config(path: Path) -> ScenarioCaptureConfig:
    return ScenarioCaptureConfig(
        source_path=path,
        capture_config=RealtimeCaptureConfig(width=32, height=24, fps=20.0),
    )


def _backend(path: Path, clock: _Clock | None = None) -> ScenarioVideoCaptureBackend:
    clock = clock or _Clock()
    return ScenarioVideoCaptureBackend(
        _config(path),
        source_epoch_id="scenario-epoch",
        clock_ns=clock,
        waiter=clock.wait,
    )


def test_real_video_emits_only_exact_geometry_consecutive_paced_capture_frames(video: Path) -> None:
    clock = _Clock()
    backend = _backend(video, clock)
    frames = list(backend.frames(threading.Event()))
    assert len(frames) == 3
    assert all(isinstance(frame, CaptureFrame) for frame in frames)
    assert [frame.capture_frame_id for frame in frames] == [0, 1, 2]
    assert [frame.captured_at_ns for frame in frames] == [
        2_000_000_000,
        2_050_000_000,
        2_100_000_000,
    ]
    assert {frame.source_epoch_id for frame in frames} == {"scenario-epoch"}
    assert {(frame.width, frame.height) for frame in frames} == {(32, 24)}
    assert [round(float(frame.decode_bgr().mean())) for frame in frames] == [30, 100, 180]
    assert sum(clock.waits) == pytest.approx(0.1)
    assert isinstance(backend.identity, NativeCaptureIdentity)
    assert backend.identity.durable_fingerprint == hashlib.sha256(video.read_bytes()).hexdigest()


@pytest.mark.parametrize("changes", [{"width": 34}, {"height": 26}, {"fps": 21.0}, {"fps": 20.001}])
def test_rejects_source_mode_mismatch(video: Path, changes: dict[str, float]) -> None:
    config = _config(video)
    config = replace(config, capture_config=replace(config.capture_config, **changes))
    with pytest.raises(CaptureContractError, match="geometry|fps|FPS"):
        list(ScenarioVideoCaptureBackend(config, source_epoch_id="epoch").frames(threading.Event()))


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "parent-symlink"])
def test_rejects_missing_nonregular_and_symlink_sources(video: Path, kind: str) -> None:
    path = video.parent / "unsafe"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(video)
    elif kind == "parent-symlink":
        path.symlink_to(video.parent, target_is_directory=True)
        path = path / video.name
    with pytest.raises((ValueError, CaptureContractError), match="unsafe|regular|symlink|missing"):
        list(_backend(path).frames(threading.Event()))


def test_rejects_undecodable_source(tmp_path: Path) -> None:
    path = tmp_path / "broken.avi"
    path.write_bytes(b"not a video")
    with pytest.raises(CaptureContractError, match="open|decode"):
        list(_backend(path).frames(threading.Event()))


def test_cancel_during_pacing_has_no_extra_frame(video: Path) -> None:
    clock = _Clock()

    def cancel_in_wait(event: threading.Event, seconds: float) -> bool:
        backend.cancel()
        return event.is_set()

    backend = ScenarioVideoCaptureBackend(
        _config(video),
        source_epoch_id="epoch",
        clock_ns=clock,
        waiter=cancel_in_wait,
    )
    frames = backend.frames(threading.Event())
    assert next(frames).capture_frame_id == 0
    assert list(frames) == []


def test_stop_during_pacing_is_checked_within_cancellation_bound(video: Path) -> None:
    clock = _Clock()
    stop = threading.Event()

    def stop_in_wait(event: threading.Event, seconds: float) -> bool:
        assert seconds <= 0.5
        stop.set()
        clock.wait(event, seconds)
        return event.is_set()

    backend = ScenarioVideoCaptureBackend(
        _config(video),
        source_epoch_id="epoch",
        clock_ns=clock,
        waiter=stop_in_wait,
    )
    frames = backend.frames(stop)
    assert next(frames).capture_frame_id == 0
    assert list(frames) == []


def test_cancel_interrupts_real_wait_and_returns_within_bound(video: Path) -> None:
    slow_video = video.with_name("slow.avi")
    writer = cv2.VideoWriter(str(slow_video), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    config = _config(slow_video)
    config = replace(config, capture_config=replace(config.capture_config, fps=1.0))
    started = threading.Event()

    def wait(event: threading.Event, seconds: float) -> bool:
        started.set()
        return event.wait(seconds)

    backend = ScenarioVideoCaptureBackend(config, source_epoch_id="epoch", waiter=wait)
    frames = backend.frames(threading.Event())
    next(frames)
    output: list[CaptureFrame] = []

    def consume() -> None:
        output.extend(frames)

    thread = threading.Thread(target=consume)
    thread.start()
    assert started.wait(0.5)
    begin = time.monotonic()
    backend.cancel()
    thread.join(timeout=config.capture_config.cancel_timeout_seconds)
    assert not thread.is_alive()
    assert time.monotonic() - begin < config.capture_config.cancel_timeout_seconds
    assert output == []


@pytest.mark.parametrize("cancel", [True, False])
def test_already_cancelled_or_stopped_backend_emits_nothing(video: Path, cancel: bool) -> None:
    backend = _backend(video)
    stop = threading.Event()
    if cancel:
        backend.cancel()
    else:
        stop.set()
    assert list(backend.frames(stop)) == []


def test_new_backend_starts_new_epoch_and_frame_ids(video: Path) -> None:
    first = list(_backend(video).frames(threading.Event()))
    clock = _Clock()
    second = list(
        ScenarioVideoCaptureBackend(
            _config(video),
            source_epoch_id="new-epoch",
            clock_ns=clock,
            waiter=clock.wait,
        ).frames(threading.Event())
    )
    assert first[0].capture_frame_id == second[0].capture_frame_id == 0
    assert second[0].source_epoch_id == "new-epoch"


@pytest.mark.parametrize("when", ["before-open", "during", "at-eof"])
@pytest.mark.parametrize("preparation_seconds", [None, 20])
def test_replaced_content_fails_closed_even_with_same_size_and_mtime(
    video: Path,
    when: str,
    preparation_seconds: float | None,
) -> None:
    clock = _Clock()
    backend = ScenarioVideoCaptureBackend(
        replace(_config(video), startup_preparation_seconds=preparation_seconds),
        source_epoch_id="epoch",
        clock_ns=clock,
        waiter=clock.wait,
    )
    frames = backend.frames(threading.Event())
    if when != "before-open":
        next(frames)
    if when == "at-eof":
        next(frames)
        next(frames)
    old_stat = video.stat()
    content = bytearray(video.read_bytes())
    content[-1] ^= 1
    replacement = video.with_suffix(".replacement")
    replacement.write_bytes(content)
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    replacement.replace(video)
    with pytest.raises(CaptureContractError, match="changed|replaced|fingerprint"):
        next(frames)


def test_truncated_video_is_decode_failure_instead_of_clean_eof(video: Path) -> None:
    video.write_bytes(video.read_bytes()[:-250])
    with pytest.raises(CaptureContractError, match="decode|EOF|ended"):
        list(_backend(video).frames(threading.Event()))


def test_failed_jpeg_reencoding_fails_closed(video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The fallback must still fail closed for a source that requires conversion.
    lossless = video.with_name("lossless.avi")
    writer = cv2.VideoWriter(str(lossless), cv2.VideoWriter_fourcc(*"FFV1"), 20.0, (32, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(CaptureContractError, match="encod"):
        list(_backend(lossless).frames(threading.Event()))


def test_empty_epoch_rejected(video: Path) -> None:
    with pytest.raises(ValueError, match="epoch"):
        ScenarioVideoCaptureBackend(_config(video), source_epoch_id=" ")


def test_container_fps_tolerance_still_uses_exact_configured_output_cadence(video: Path) -> None:
    config = _config(video)
    config = replace(config, capture_config=replace(config.capture_config, fps=20.0000005))
    clock = _Clock()
    frames = list(
        ScenarioVideoCaptureBackend(
            config,
            source_epoch_id="epoch",
            clock_ns=clock,
            waiter=clock.wait,
        ).frames(threading.Event())
    )
    assert [frame.captured_at_ns for frame in frames] == [
        2_000_000_000,
        2_049_999_999,
        2_099_999_998,
    ]


def test_late_consumer_does_not_rebase_scheduled_timestamps(video: Path) -> None:
    clock = _Clock()
    frames = _backend(video, clock).frames(threading.Event())
    first = next(frames)
    clock.now += 500_000_000
    rest = list(frames)
    assert [first.captured_at_ns, *(frame.captured_at_ns for frame in rest)] == [
        2_000_000_000,
        2_050_000_000,
        2_100_000_000,
    ]
    assert clock.waits == []


def test_same_epoch_cannot_be_reopened(video: Path) -> None:
    backend = _backend(video)
    list(backend.frames(threading.Event()))
    with pytest.raises(CaptureContractError, match="already opened"):
        list(backend.frames(threading.Event()))


def test_in_place_content_change_with_restored_mtime_fails_closed(video: Path) -> None:
    backend = _backend(video)
    frames = backend.frames(threading.Event())
    next(frames)
    old_stat = video.stat()
    with video.open("r+b") as source:
        source.seek(-1, 2)
        source.write(b"X")
    os.utime(video, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with pytest.raises(CaptureContractError, match="changed|fingerprint"):
        next(frames)


def test_playback_gate_holds_real_frame_zero_without_consuming_source(video: Path) -> None:
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    gate = scenario.ScenarioPlaybackGate()
    clock = _Clock()
    backend = ScenarioVideoCaptureBackend(
        _config(video),
        source_epoch_id="held",
        clock_ns=clock,
        waiter=clock.wait,
        playback_gate=gate,
    )
    frames = backend.frames(threading.Event())
    held = [next(frames) for _ in range(8)]
    assert not gate.is_released()
    assert [round(float(frame.decode_bgr().mean())) for frame in held] == [30] * 8
    assert [frame.capture_frame_id for frame in held] == list(range(8))
    assert [frame.captured_at_ns for frame in held] == [
        2_000_000_000 + index * 50_000_000 for index in range(8)
    ]
    gate.release()
    gate.release()
    rest = list(frames)
    assert gate.is_released()
    assert [round(float(frame.decode_bgr().mean())) for frame in rest] == [100, 180]
    assert [frame.capture_frame_id for frame in rest] == [8, 9]
    with pytest.raises(CaptureContractError, match="already opened"):
        list(backend.frames(threading.Event()))


def test_one_frame_source_does_not_end_while_held_and_fresh_gate_holds_again(tmp_path) -> None:
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    path = tmp_path / "one.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
    assert writer.isOpened()
    writer.write(np.full((24, 32, 3), 42, dtype=np.uint8))
    writer.release()
    for epoch in ("first", "fresh"):
        gate = scenario.ScenarioPlaybackGate()
        clock = _Clock()
        frames = ScenarioVideoCaptureBackend(
            _config(path),
            source_epoch_id=epoch,
            clock_ns=clock,
            waiter=clock.wait,
            playback_gate=gate,
        ).frames(threading.Event())
        assert [next(frames).capture_frame_id for _ in range(5)] == [0, 1, 2, 3, 4]
        gate.release()
        assert list(frames) == []


def test_held_playback_still_checks_source_and_interrupts_pacing(video: Path) -> None:
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    clock = _Clock()
    gate = scenario.ScenarioPlaybackGate()
    backend = ScenarioVideoCaptureBackend(
        _config(video),
        source_epoch_id="held",
        clock_ns=clock,
        waiter=lambda *_: backend.cancel() or True,
        playback_gate=gate,
    )
    frames = backend.frames(threading.Event())
    assert next(frames).capture_frame_id == 0
    assert list(frames) == []
    assert not gate.is_released()
    clock = _Clock()
    backend = ScenarioVideoCaptureBackend(
        _config(video),
        source_epoch_id="changed",
        clock_ns=clock,
        waiter=clock.wait,
        playback_gate=gate,
    )
    frames = backend.frames(threading.Event())
    next(frames)
    video.write_bytes(video.read_bytes() + b"x")
    with pytest.raises(CaptureContractError, match="changed"):
        next(frames)


class _CaptureProxy:
    def __init__(self, capture):
        self.capture = capture

    def __getattr__(self, name):
        return getattr(self.capture, name)


def test_mjpeg_source_preserves_original_packets_and_pixels_without_reencoding(video, monkeypatch):
    original = cv2.VideoCapture(str(video))
    assert original.set(cv2.CAP_PROP_FORMAT, -1)
    expected = []
    while True:
        ok, packet = original.read()
        if not ok:
            break
        expected.append(packet.tobytes())
    original.release()

    def no_reencoding(*args, **kwargs):
        raise AssertionError("MJPEG source must preserve original JPEG payloads")

    monkeypatch.setattr(cv2, "imencode", no_reencoding)
    frames = list(_backend(video).frames(threading.Event()))
    assert [frame.mjpeg for frame in frames] == expected
    for frame, packet in zip(frames, expected, strict=True):
        assert np.array_equal(
            frame.decode_bgr(),
            cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR),
        )


@pytest.mark.parametrize("bad", ["geometry", "empty", "invalid", "wrong-type"])
def test_raw_mjpeg_rejects_bad_packets_before_emitting_and_releases(video, monkeypatch, bad):
    capture_factory = cv2.VideoCapture
    opened = []

    class BadPacket(_CaptureProxy):
        released = False
        raw = False

        def set(self, key, value):
            accepted = self.capture.set(key, value)
            if key == cv2.CAP_PROP_FORMAT and value == -1 and accepted:
                self.raw = True
            return accepted

        def read(self):
            ok, image = self.capture.read()
            if ok and self.raw:
                if bad == "geometry":
                    encoded_ok, encoded = cv2.imencode(".jpg", np.zeros((24, 34, 3), np.uint8))
                    assert encoded_ok
                    return True, encoded.reshape(1, -1)
                if bad == "empty":
                    return True, np.empty((1, 0), dtype=np.uint8)
                if bad == "wrong-type":
                    return True, np.ones((1, 4), dtype=np.float32)
                return True, np.frombuffer(b"not jpeg", dtype=np.uint8).reshape(1, -1)
            return ok, image

        def release(self):
            self.released = True
            return self.capture.release()

    def open_capture(*args, **kwargs):
        result = BadPacket(capture_factory(*args, **kwargs))
        opened.append(result)
        return result

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    with pytest.raises(CaptureContractError, match="MJPEG|geometry|packet"):
        next(_backend(video).frames(threading.Event()))
    assert len(opened) == 1 and opened[0].released


def test_raw_format_refusal_keeps_decoded_fallback_and_checks_encoding_failure(video, monkeypatch):
    capture_factory = cv2.VideoCapture
    # MJPG decoding may differ by one intensity unit across codec builds.
    # Compare the exact independently decoded/re-encoded stream on this host.
    reference = capture_factory(str(video))
    expected = []
    try:
        while True:
            ok, image = reference.read()
            if not ok:
                break
            encoded_ok, encoded = cv2.imencode(".jpg", image)
            assert encoded_ok
            expected.append(encoded.tobytes())
    finally:
        reference.release()
    attempts = []

    class RefusedRaw(_CaptureProxy):
        def set(self, key, value):
            if key == cv2.CAP_PROP_FORMAT and value == -1:
                attempts.append(value)
                return False
            return self.capture.set(key, value)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: RefusedRaw(capture_factory(*a, **kw)))
    frames = list(_backend(video).frames(threading.Event()))
    assert len(frames) == 3 and attempts == [-1]
    assert [frame.mjpeg for frame in frames] == expected
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(CaptureContractError, match="encod"):
        list(_backend(video).frames(threading.Event()))


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_raw_jpeg_exif_does_not_change_video_orientation_or_propagate_metadata(
    video,
    monkeypatch,
    orientation,
):
    import struct

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :16] = (0, 0, 240)
    image[:, 16:] = (230, 0, 0)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    tiff = b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", 1)
    tiff += struct.pack("<HHI", 274, 3, 1) + struct.pack("<H", orientation) + b"\x00" * 6
    exif = b"Exif\x00\x00" + tiff
    payload = encoded.tobytes()
    payload = payload[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + payload[2:]
    capture_factory = cv2.VideoCapture

    class ExifPacket(_CaptureProxy):
        def read(self):
            ok, image = self.capture.read()
            if ok and self.capture.get(cv2.CAP_PROP_FORMAT) == -1:
                return True, np.frombuffer(payload, dtype=np.uint8).reshape(1, -1)
            return ok, image

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: ExifPacket(capture_factory(*a, **kw)))
    frames = list(_backend(video).frames(threading.Event()))
    assert len(frames) == 3
    for frame in frames:
        assert b"Exif\x00\x00" not in frame.mjpeg
        decoded = frame.decode_bgr()
        assert decoded.shape == (24, 32, 3)
        assert decoded[12, 4, 2] > 200 and decoded[12, 4, 0] < 30
        assert decoded[12, 28, 0] > 200 and decoded[12, 28, 2] < 30


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf"), "20"])
def test_file_preparation_rejects_unbounded_or_invalid_allowance(video, seconds):
    with pytest.raises(ValueError, match="preparation"):
        replace(_config(video), startup_preparation_seconds=seconds)


def test_preparation_timeout_never_opens_video_or_issues_receipt(video, monkeypatch):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    now = [100_000_000_000]
    backend = ScenarioVideoCaptureBackend(
        replace(_config(video), startup_preparation_seconds=20), source_epoch_id="epoch"
    )
    monkeypatch.setattr(scenario, "monotonic_ns", lambda: now[0])
    original_hash = scenario._content_hash

    def slow_hash(path):
        value = original_hash(path)
        now[0] += 21_000_000_000
        return value

    monkeypatch.setattr(scenario, "_content_hash", slow_hash)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: pytest.fail("expired preparation opened"))
    with pytest.raises(CaptureContractError, match="preparation"):
        next(backend.frames(threading.Event()))
    assert backend.preparation_completed_at_ns is None


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_during_full_preparation_hash_does_not_open_video(video, monkeypatch, cancel):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    backend = ScenarioVideoCaptureBackend(
        replace(_config(video), startup_preparation_seconds=20), source_epoch_id="epoch"
    )
    stop = threading.Event()
    original_hash = scenario._content_hash

    def cancelling_hash(path):
        value = original_hash(path)
        backend.cancel() if cancel else stop.set()
        return value

    monkeypatch.setattr(scenario, "_content_hash", cancelling_hash)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: pytest.fail("cancelled preparation opened"))
    assert list(backend.frames(stop)) == []
    assert backend.preparation_completed_at_ns is None


@pytest.mark.parametrize("when", ["before-start", "eof"])
def test_explicit_preparation_keeps_full_hash_even_if_windows_stat_looks_unchanged(
    video, monkeypatch, when
):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    clock = _Clock()
    backend = ScenarioVideoCaptureBackend(
        replace(_config(video), startup_preparation_seconds=20),
        source_epoch_id="epoch",
        clock_ns=clock,
        waiter=clock.wait,
    )
    original_stat = scenario._source_stat(video)
    # On Windows ctime is not a write-generation counter. Freeze all metadata
    # after an actual same-size write/restored mtime; full SHA must still reject.
    monkeypatch.setattr(scenario, "_source_stat", lambda _: original_stat)
    frames = backend.frames(threading.Event())
    if when == "eof":
        for _ in range(3):
            next(frames)
        receipt = backend.preparation_completed_at_ns
    old = video.stat()
    with video.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    os.utime(video, ns=(old.st_atime_ns, old.st_mtime_ns))
    try:
        with pytest.raises(CaptureContractError, match="fingerprint"):
            next(frames)
        if when == "before-start":
            assert backend.preparation_completed_at_ns is None
        else:
            assert backend.preparation_completed_at_ns == receipt
    finally:
        frames.close()


def test_preparation_receipt_uses_real_clock_once_and_eof_cannot_refresh_it(video, monkeypatch):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    clock = _Clock()
    backend = ScenarioVideoCaptureBackend(
        replace(_config(video), startup_preparation_seconds=20),
        source_epoch_id="epoch",
        clock_ns=clock,
        waiter=clock.wait,
    )
    now = [100_000_000_000]
    monkeypatch.setattr(scenario, "monotonic_ns", lambda: now[0])
    frames = backend.frames(threading.Event())
    first = next(frames)
    receipt = backend.preparation_completed_at_ns
    assert receipt == 100_000_000_000
    assert first.captured_at_ns == 2_000_000_000
    now[0] += 90_000_000_000
    assert len(list(frames)) == 2
    assert backend.preparation_completed_at_ns == receipt
