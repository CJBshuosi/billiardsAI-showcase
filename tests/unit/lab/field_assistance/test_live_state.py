from __future__ import annotations

import json
from pathlib import Path

import pytest

from billiards_ai.lab.field_assistance.live_state import (
    ReplayClock,
    load_replay_timeline,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _timeline(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": [
                    {"frame_id": 0, "timestamp_ms": 0, "phase": "WAITING"},
                    {"frame_id": 1, "timestamp_ms": 100, "phase": "ROUTE"},
                    {"frame_id": 2, "timestamp_ms": 200, "phase": "MOVING"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_start_pause_reset_and_timeline_projection(tmp_path: Path) -> None:
    now = FakeClock()
    clock = ReplayClock(
        load_replay_timeline(_timeline(tmp_path / "timeline.json"), duration_ms=300),
        monotonic=now,
    )

    clock.start()
    now.advance(0.15)
    running = clock.projection()
    assert running.playing is True
    assert running.position_ms == 150
    assert running.frame.frame_id == 1

    clock.pause()
    now.advance(1.0)
    paused = clock.projection()
    assert paused.playing is False
    assert paused.position_ms == 150

    clock.reset()
    reset = clock.projection()
    assert reset.position_ms == 0
    assert reset.frame.frame_id == 0


def test_end_of_video_clamps_and_stops(tmp_path: Path) -> None:
    now = FakeClock()
    clock = ReplayClock(
        load_replay_timeline(_timeline(tmp_path / "timeline.json"), duration_ms=300),
        monotonic=now,
    )
    clock.start()
    now.advance(0.5)

    projection = clock.projection()

    assert projection.position_ms == 300
    assert projection.playing is False
    assert projection.frame.frame_id == 2


def test_non_monotonic_clock_is_rejected(tmp_path: Path) -> None:
    now = FakeClock()
    clock = ReplayClock(
        load_replay_timeline(_timeline(tmp_path / "timeline.json"), duration_ms=300),
        monotonic=now,
    )
    clock.start()
    now.advance(0.1)
    clock.projection()
    now.advance(-0.2)

    with pytest.raises(ValueError, match="monotonic"):
        clock.projection()


def test_loader_rejects_unsorted_timeline(tmp_path: Path) -> None:
    path = _timeline(tmp_path / "timeline.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frames"] = list(reversed(payload["frames"]))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ordered"):
        load_replay_timeline(path, duration_ms=300)


def test_clock_exposes_the_same_immutable_timeline(tmp_path: Path) -> None:
    timeline = load_replay_timeline(_timeline(tmp_path / "timeline.json"), duration_ms=300)
    clock = ReplayClock(timeline, monotonic=FakeClock())

    assert clock.timeline is timeline


def test_loader_rejects_frame_beyond_media_duration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duration"):
        load_replay_timeline(_timeline(tmp_path / "timeline.json"), duration_ms=150)
