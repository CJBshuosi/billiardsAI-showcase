from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from billiards_ai.lab.mainline1_live_spool import (
    CaptureSpool,
    LiveBackpressureError,
    LiveCaptureState,
    LiveSpoolContractError,
)
from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame


def _frame(frame_id: int, timestamp_ms: int | None = None) -> DecodedTrackingFrame:
    return DecodedTrackingFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 17 if timestamp_ms is None else timestamp_ms,
        image=np.full((3, 4, 3), frame_id, dtype=np.uint8),
    )


def test_capacity_exhaustion_degrades_instead_of_dropping_frames(tmp_path: Path) -> None:
    spool = CaptureSpool(tmp_path, maximum_frames=2)
    spool.append(_frame(1))
    spool.append(_frame(2))

    with pytest.raises(LiveBackpressureError):
        spool.append(_frame(3))

    assert spool.health.state is LiveCaptureState.DEGRADED
    assert spool.health.stored_frames == 2
    assert [item.frame_id for item in spool.frames_between(1, 2)] == [1, 2]


def test_spool_rejects_duplicate_or_reversed_frame_order(tmp_path: Path) -> None:
    spool = CaptureSpool(tmp_path, maximum_frames=4)
    spool.append(_frame(2, 20))

    with pytest.raises(LiveSpoolContractError, match="frame_id"):
        spool.append(_frame(2, 21))
    with pytest.raises(LiveSpoolContractError, match="timestamp"):
        spool.append(_frame(3, 19))

    assert spool.health.stored_frames == 1


def test_seal_shot_is_atomic_and_retains_after_frame_for_next_shot(tmp_path: Path) -> None:
    spool = CaptureSpool(tmp_path, maximum_frames=3)
    spool.append(_frame(1))
    spool.append(_frame(2))
    spool.append(_frame(3))

    segment = spool.seal_shot(shot_id="shot-0001", start_frame_id=1, end_frame_id=2, complete=True)

    assert segment.path.name == "shot-0001"
    assert segment.frame_count == 2
    assert segment.complete is True
    assert segment.byte_count > 0
    assert len(segment.sha256) == 64
    manifest = json.loads((segment.path / "manifest.json").read_text())
    assert manifest["frames_sha256"] == segment.sha256
    assert not list(segment.path.parent.glob(".pending-*"))
    assert [item.frame_id for item in spool.frames_between(2, 3)] == [2, 3]

    spool.append(_frame(4))
    assert spool.health.stored_frames == 3


def test_startup_removes_only_pending_directories(tmp_path: Path) -> None:
    pending = tmp_path / ".pending-abandoned"
    pending.mkdir()
    (pending / "partial").write_bytes(b"partial")
    preserved = tmp_path / "shot-existing"
    preserved.mkdir()
    (preserved / "manifest.json").write_text("{}", encoding="utf-8")

    CaptureSpool(tmp_path, maximum_frames=2)

    assert not pending.exists()
    assert preserved.exists()


def test_spool_root_has_one_live_owner_and_can_be_reopened_after_close(
    tmp_path: Path,
) -> None:
    first = CaptureSpool(tmp_path, maximum_frames=2)

    with pytest.raises(LiveSpoolContractError, match="already owned"):
        CaptureSpool(tmp_path, maximum_frames=2)

    first.close()
    recovered = CaptureSpool(tmp_path, maximum_frames=2)
    recovered.close()


def test_startup_recovers_a_lock_left_by_a_dead_process(tmp_path: Path) -> None:
    (tmp_path / ".capture-spool-owner.json").write_text(
        json.dumps({"pid": 999_999_999, "token": "abandoned"}),
        encoding="utf-8",
    )

    spool = CaptureSpool(tmp_path, maximum_frames=2)

    owner = json.loads((tmp_path / ".capture-spool-owner.json").read_text())
    assert owner["pid"] == os.getpid()
    assert owner["token"] != "abandoned"
    spool.close()


def test_startup_quarantines_unpublished_loose_frames_from_a_crashed_owner(
    tmp_path: Path,
) -> None:
    frames = tmp_path / ".live-frames"
    frames.mkdir()
    (frames / "00000000000000000001.npy").write_bytes(b"orphan")
    (frames / ".pending-frame-abandoned.npy").write_bytes(b"partial")

    spool = CaptureSpool(tmp_path, maximum_frames=2)

    assert list(frames.iterdir()) == []
    recovered = next(tmp_path.glob(".recovered-live-frames-*"))
    assert (recovered / "00000000000000000001.npy").read_bytes() == b"orphan"
    spool.close()


def test_incomplete_range_cannot_be_sealed_as_complete(tmp_path: Path) -> None:
    spool = CaptureSpool(tmp_path, maximum_frames=3)
    spool.append(_frame(1))
    spool.append(_frame(3))

    with pytest.raises(LiveSpoolContractError, match="contiguous"):
        spool.seal_shot(shot_id="shot-gap", start_frame_id=1, end_frame_id=3, complete=True)
