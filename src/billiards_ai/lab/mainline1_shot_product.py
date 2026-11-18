"""Source-neutral canonical Mainline 1 shot product inputs and payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.models import ObservationFrame, ShotLocalResult
from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame
from billiards_ai.lab.shot_local.product import CanonicalShotAnalysis
from billiards_ai.shared.calibration.homography import TableCalibration


class ShotProductConfig(Protocol):
    event: ShotLocalEventConfig


@dataclass(frozen=True, slots=True)
class Mainline1ShotProductPayloads:
    decoded_frames: tuple[DecodedTrackingFrame, ...]
    observations: tuple[ObservationFrame, ...]
    results: tuple[ShotLocalResult, ...]
    analyses: tuple[CanonicalShotAnalysis, ...]
    calibration: TableCalibration
    cushion_distance_mm: int
    fps: float
    payloads: dict[str, object]


def build_mainline1_shot_product(
    *,
    decoded_frames: tuple[DecodedTrackingFrame, ...],
    observations: tuple[ObservationFrame, ...],
    results: tuple[ShotLocalResult, ...],
    analyses: tuple[CanonicalShotAnalysis, ...],
    calibration: TableCalibration,
    config: ShotProductConfig,
    fps: float,
    realtime_perception: Mapping[str, object] | None = None,
) -> Mainline1ShotProductPayloads:
    """Build the exact canonical payload set shared by file and live publication."""

    if not decoded_frames:
        raise ValueError("shot product requires decoded frames")
    if fps <= 0:
        raise ValueError("shot product fps must be positive")
    if len(results) != len(analyses):
        raise ValueError("shot product evidence and analyses must align")

    # Imported lazily while these compatibility helpers move out of the legacy
    # file entry point. This avoids a module import cycle without duplicating
    # any product semantics.
    from billiards_ai.lab.mainline1_shot_local_video import (
        _active_tracks_payload,
        _anchors_payload,
        _canonical_event_timeline_payload,
        _cue_payload,
        _detections_payload,
        _fragment_payload,
        _summary_payload,
        _table_state_payload,
        _window_payload,
        _world_events_payload,
    )

    frames_by_id = {frame.frame_id: frame for frame in observations}
    payloads = {
        "detections.json": _detections_payload(observations),
        "shot-windows.json": _window_payload(results),
        "anchors.json": _anchors_payload(results),
        "cue-track.json": _cue_payload(results),
        "motion-fragments.json": _fragment_payload(results),
        "active-tracks.json": _active_tracks_payload(analyses),
        "events.json": _world_events_payload(analyses),
        "event-timeline.json": _canonical_event_timeline_payload(analyses, frames_by_id),
        "world-events.json": _world_events_payload(analyses),
        "table-state-before.json": _table_state_payload(results, frames_by_id, after=False),
        "table-state-after.json": _table_state_payload(results, frames_by_id, after=True),
        "summary.json": _summary_payload(frame_count=len(decoded_frames), analyses=analyses),
    }
    if realtime_perception is not None:
        payloads["realtime-perception.json"] = dict(realtime_perception)
    return Mainline1ShotProductPayloads(
        decoded_frames=decoded_frames,
        observations=observations,
        results=results,
        analyses=analyses,
        calibration=calibration,
        cushion_distance_mm=config.event.cushion_distance_mm,
        fps=fps,
        payloads=payloads,
    )
