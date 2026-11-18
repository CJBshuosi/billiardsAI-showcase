from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab import mainline1_realtime_ring as ring_module
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import (
    AsyncMjpegSegmentRing,
    MjpegSegmentRing,
    RealtimeRingError,
)


def _jpeg(value: int) -> bytes:
    image = np.full((12, 16, 3), value % 251, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _frame(frame_id: int) -> CaptureFrame:
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=round(frame_id * 1_000_000_000 / 60),
        width=16,
        height=12,
        mjpeg=_jpeg(frame_id),
    )


def _ring(root: Path, *, maximum_bytes: int = 2 * 1024**3) -> MjpegSegmentRing:
    return MjpegSegmentRing(
        root,
        expected_fps=60,
        pre_roll_frames=120,
        post_roll_frames=120,
        maximum_shot_frames=1800,
        maximum_bytes=maximum_bytes,
    )


def test_memory_preroll_avoids_idle_disk_writes_and_discards_incomplete_shot(
    tmp_path: Path,
) -> None:
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=2,
        pre_roll_frames=4,
        post_roll_frames=2,
        maximum_shot_frames=20,
        maximum_bytes=64 * 1024 * 1024,
        persist_idle_segments=False,
    )
    for frame_id in range(10):
        ring.append(_frame(frame_id))

    assert not tuple(tmp_path.glob("*.mjpgseg"))
    assert ring.health.stored_frames == 5

    ring.begin_shot(9)
    ring.append(_frame(10))
    ring.close()

    assert not tuple(tmp_path.glob("*.mjpgseg"))
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["segments"] == []


def test_memory_preroll_materializes_only_a_completed_shot(tmp_path: Path) -> None:
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=2,
        pre_roll_frames=4,
        post_roll_frames=2,
        maximum_shot_frames=20,
        maximum_bytes=64 * 1024 * 1024,
        persist_idle_segments=False,
    )
    for frame_id in range(6):
        ring.append(_frame(frame_id))
    assert not tuple(tmp_path.glob("*.mjpgseg"))

    ring.begin_shot(5)
    ring.append(_frame(6))
    ring.settle_shot(6)
    ring.append(_frame(7))
    ring.append(_frame(8))
    frozen = ring.freeze_shot()

    assert (frozen.start_frame_id, frozen.end_frame_id) == (1, 8)
    assert tuple(frame.capture_frame_id for frame in frozen.frames()) == tuple(range(1, 9))


def test_ring_freezes_contiguous_preroll_shot_and_postroll(tmp_path: Path) -> None:
    ring = _ring(tmp_path)
    for frame_id in range(360):
        ring.append(_frame(frame_id))
        if frame_id == 120:
            ring.begin_shot(frame_id)
        if frame_id == 239:
            ring.settle_shot(frame_id)

    frozen = ring.freeze_shot()

    assert (frozen.start_frame_id, frozen.end_frame_id) == (0, 359)
    assert frozen.frame_count == 360
    assert tuple(frame.capture_frame_id for frame in frozen.frames()) == tuple(range(360))
    assert frozen.segment_sha256
    assert all(len(value) == 64 for value in frozen.segment_sha256.values())


def test_async_ring_writer_does_not_block_capture_during_segment_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=2,
        pre_roll_frames=2,
        post_roll_frames=0,
        maximum_shot_frames=10,
        maximum_bytes=64 * 1024 * 1024,
    )
    original = ring.append

    def slow_append(frame: CaptureFrame) -> None:
        time.sleep(0.05)
        original(frame)

    monkeypatch.setattr(ring, "append", slow_append)
    writer = AsyncMjpegSegmentRing(ring, maximum_pending_frames=10)
    started = time.monotonic()

    writer.append(_frame(0))
    writer.append(_frame(1))

    assert time.monotonic() - started < 0.02
    writer.close(timeout_seconds=2)
    assert writer.health.stored_frames == 2


def test_segment_digest_is_computed_while_writing_without_rereading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=2,
        pre_roll_frames=2,
        post_roll_frames=0,
        maximum_shot_frames=10,
        maximum_bytes=64 * 1024 * 1024,
    )

    def unexpected_reread(_path: Path) -> str:
        raise AssertionError("completed segment was reopened for hashing")

    monkeypatch.setattr(ring_module, "_sha256", unexpected_reread)
    ring.append(_frame(0))
    ring.append(_frame(1))

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    segment = index["segments"][0]
    payload = (tmp_path / segment["name"]).read_bytes()
    assert segment["sha256"] == hashlib.sha256(payload).hexdigest()


def test_idle_ring_retires_segments_older_than_preroll(tmp_path: Path) -> None:
    ring = _ring(tmp_path)
    for frame_id in range(300):
        ring.append(_frame(frame_id))

    ring.close()
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))

    assert index["segments"][0]["first_frame_id"] == 180
    assert index["segments"][-1]["last_frame_id"] == 299


def test_ring_rejects_source_gap_and_epoch_change(tmp_path: Path) -> None:
    ring = _ring(tmp_path)
    ring.append(_frame(0))

    with pytest.raises(RealtimeRingError, match="contiguous"):
        ring.append(_frame(2))
    with pytest.raises(RealtimeRingError, match="degraded"):
        ring.append(_frame(1))


def test_ring_refuses_freeze_before_full_postroll(tmp_path: Path) -> None:
    ring = _ring(tmp_path)
    for frame_id in range(180):
        ring.append(_frame(frame_id))
        if frame_id == 120:
            ring.begin_shot(frame_id)
        if frame_id == 150:
            ring.settle_shot(frame_id)

    with pytest.raises(RealtimeRingError, match="post-roll"):
        ring.freeze_shot()


def test_frozen_interval_detects_segment_corruption(tmp_path: Path) -> None:
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=4,
        pre_roll_frames=2,
        post_roll_frames=2,
        maximum_shot_frames=12,
        maximum_bytes=1024 * 1024,
    )
    for frame_id in range(8):
        ring.append(_frame(frame_id))
        if frame_id == 2:
            ring.begin_shot(frame_id)
        if frame_id == 5:
            ring.settle_shot(frame_id)
    frozen = ring.freeze_shot()
    first_segment = next(iter(frozen.segment_sha256))
    (tmp_path / first_segment).write_bytes(b"corrupt")

    with pytest.raises(RealtimeRingError, match="hash"):
        tuple(frozen.frames())


def test_ring_byte_limit_during_shot_fails_closed(tmp_path: Path) -> None:
    ring = _ring(tmp_path, maximum_bytes=len(_jpeg(0)) * 2)
    ring.append(_frame(0))
    ring.begin_shot(0)

    with pytest.raises(RealtimeRingError, match="byte capacity"):
        for frame_id in range(1, 61):
            ring.append(_frame(frame_id))

    assert ring.health.state == "degraded"


def test_release_retires_only_segments_outside_the_next_preroll(tmp_path: Path) -> None:
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=4,
        pre_roll_frames=2,
        post_roll_frames=2,
        maximum_shot_frames=12,
        maximum_bytes=1024 * 1024,
    )
    for frame_id in range(8):
        ring.append(_frame(frame_id))
        if frame_id == 2:
            ring.begin_shot(frame_id)
        if frame_id == 5:
            ring.settle_shot(frame_id)
    frozen = ring.freeze_shot()

    ring.release_shot()

    assert not (tmp_path / next(iter(frozen.segment_sha256))).exists()
    assert ring.health.stored_frames == 4


@pytest.mark.parametrize("persist_idle", [False, True], ids=["memory", "segments"])
@pytest.mark.parametrize("delay", [3, 4], ids=["exact-budget", "over-budget"])
def test_late_onset_keeps_original_preroll_only_within_explicit_budget(
    tmp_path, persist_idle, delay
):
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=1,
        pre_roll_frames=4,
        post_roll_frames=1,
        maximum_shot_frames=20,
        maximum_bytes=64 * 1024 * 1024,
        persist_idle_segments=persist_idle,
        idle_onset_lookback_frames=3,
    )
    # A long static wait and capture/analysis onset lag must not silently rebase
    # the requested pre-roll. The budget is explicit, not inferred from queue size.
    for frame_id in range(21 + delay):
        ring.append(_frame(frame_id))
    if not persist_idle:
        assert ring.health.stored_frames == 8
        assert not tuple(tmp_path.glob("*.mjpgseg"))
    ring.begin_shot(20)
    ring.settle_shot(20 + delay)
    ring.append(_frame(21 + delay))
    if delay == 4:
        with pytest.raises(RealtimeRingError, match="not contiguous"):
            ring.freeze_shot()
    else:
        frozen = ring.freeze_shot()
        assert (frozen.start_frame_id, frozen.end_frame_id) == (16, 24)
        assert [frame.capture_frame_id for frame in frozen.frames()] == list(range(16, 25))


def test_idle_onset_budget_cannot_exceed_capacity_or_discard_needed_history(tmp_path):
    size = len(_frame(0).mjpeg)
    ring = MjpegSegmentRing(
        tmp_path,
        expected_fps=60,
        pre_roll_frames=4,
        post_roll_frames=1,
        maximum_shot_frames=20,
        maximum_bytes=size * 8,
        persist_idle_segments=False,
        idle_onset_lookback_frames=3,
    )
    # The idle capacity flush must preserve required evidence and fail
    # visibly when all retained history cannot fit; never trim its budget.
    for frame_id in range(8):
        ring.append(replace(_frame(frame_id), mjpeg=_frame(0).mjpeg))
        assert ring.health.stored_payload_bytes <= size * 8
    with pytest.raises(RealtimeRingError, match="byte capacity"):
        ring.append(replace(_frame(8), mjpeg=_frame(0).mjpeg))
    assert tuple(tmp_path.glob("*.mjpgseg"))
    assert ring.health.stored_frames == 8
    assert ring.health.stored_payload_bytes <= size * 8
    assert ring.health.state.value == "degraded"


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_idle_onset_budget_requires_nonnegative_integer(tmp_path, budget):
    with pytest.raises(ValueError, match="configuration"):
        MjpegSegmentRing(
            tmp_path,
            expected_fps=60,
            pre_roll_frames=4,
            post_roll_frames=1,
            maximum_shot_frames=20,
            maximum_bytes=10000,
            idle_onset_lookback_frames=budget,
        )
