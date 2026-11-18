"""Build renderer-neutral, evidence-linked overlay instructions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_overlay_frames(
    predictions: Sequence[Mapping[str, Any]],
    evidence_frames: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return deterministic overlay instructions while preserving source timestamps."""
    by_frame: dict[int, list[dict[str, Any]]] = {frame_id: [] for frame_id in evidence_frames}
    for shot in predictions:
        for event in shot.get("events", []):
            evidence_ids = event.get("evidence_frame_ids", [])
            if not evidence_ids:
                raise ValueError("displayed events require at least one evidence frame")
            for frame_id in evidence_ids:
                if frame_id not in evidence_frames:
                    raise ValueError(f"event references missing evidence frame {frame_id}")
                by_frame[frame_id].append(dict(event))

    overlay_frames = []
    for frame_id, frame in sorted(evidence_frames.items()):
        overlay_frames.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": frame["timestamp_ms"],
                "labels": list(frame.get("labels", [])),
                "table_boundary": frame.get("table_boundary"),
                "balls": list(frame.get("balls", [])),
                "trajectories": list(frame.get("trajectories", [])),
                "shot_boundary": frame.get("shot_boundary"),
                "events": sorted(by_frame[frame_id], key=lambda event: event["event_id"]),
            }
        )
    return tuple(overlay_frames)
