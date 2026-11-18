"""Strict adapters from cached detector evidence to stable planning snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from billiards_ai.edge.route_assistance.models import SceneObservation
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)

_GROUP_KINDS = {
    BallGroup.CUE: BallKind.CUE,
    BallGroup.EIGHT: BallKind.EIGHT,
    BallGroup.SOLID: BallKind.SOLID,
    BallGroup.STRIPE: BallKind.STRIPE,
    BallGroup.UNKNOWN: BallKind.UNKNOWN,
}


def _cached_observation(row: Any) -> CalibratedBallObservation:
    if not isinstance(row, dict):
        raise ValueError("cached observation row is invalid")
    try:
        center = row["center_mm"]
        box = row["bbox_xyxy_px"]
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("center is invalid")
        if box is not None and (not isinstance(box, list) or len(box) != 4):
            raise ValueError("box is invalid")
        return CalibratedBallObservation(
            detection_id=str(row["detection_id"]),
            x_mm=int(center[0]),
            y_mm=int(center[1]),
            confidence=float(row["detection_confidence"]),
            group=BallGroup(str(row["group"])),
            group_confidence=float(row["group_confidence"]),
            bbox_xyxy_px=None if box is None else tuple(int(value) for value in box),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("cached observation row is invalid") from exc


def load_cached_observations(path: Path) -> tuple[ObservationFrame, ...]:
    """Load the immutable `detections.json` contract produced by shot-local runs."""

    source = path.resolve()
    if path.is_symlink() or not source.is_file():
        raise ValueError("cached observation source is missing or unsafe")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cached observation payload is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("cached observation schema is unsupported")
    rows = payload.get("frames")
    if not isinstance(rows, list):
        raise ValueError("cached observation frames are invalid")

    frames = []
    try:
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("observations"), list):
                raise ValueError("frame row is invalid")
            frames.append(
                ObservationFrame(
                    frame_id=int(row["frame_id"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    observations=tuple(_cached_observation(item) for item in row["observations"]),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("cached observation payload is invalid") from exc

    order = [(frame.frame_id, frame.timestamp_ms) for frame in frames]
    if len({frame.frame_id for frame in frames}) != len(frames) or any(
        current_frame <= previous_frame or current_ms < previous_ms
        for (previous_frame, previous_ms), (current_frame, current_ms) in zip(
            order,
            order[1:],
            strict=False,
        )
    ):
        raise ValueError("cached observation frames must be unique ordered")
    return tuple(frames)


def scene_observation(
    frame: ObservationFrame,
    *,
    visual_motion: bool | None = None,
) -> SceneObservation:
    """Remove detection identity before online motion/stability reduction."""

    return SceneObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        centers_mm=tuple(sorted((item.x_mm, item.y_mm) for item in frame.observations)),
        visual_motion=visual_motion,
    )


def build_snapshot_table_state(
    frame: ObservationFrame,
    *,
    calibration_id: str,
    source_ref: str,
    scene_generation: int,
    radius_mm: float = 28.575,
) -> CanonicalTableState:
    """Build scene-local planning IDs without implying cross-shot identity."""

    if not calibration_id or not source_ref:
        raise ValueError("snapshot calibration and source reference are required")
    if scene_generation < 0:
        raise ValueError("scene generation must be non-negative")
    ordered = sorted(
        frame.observations,
        key=lambda item: (item.group.value, item.x_mm, item.y_mm, item.detection_id),
    )
    balls = tuple(
        CanonicalBall(
            track_id=f"scene-{scene_generation:04d}-ball-{index:02d}",
            identity="cue" if item.group is BallGroup.CUE else "unknown",
            kind=_GROUP_KINDS[item.group],
            identity_confidence=item.group_confidence,
            center=Point(x_mm=item.x_mm, y_mm=item.y_mm),
            radius_mm=radius_mm,
            detection_confidence=item.confidence,
        )
        for index, item in enumerate(ordered, start=1)
    )
    state = CanonicalTableState(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        calibration_id=calibration_id,
        motion_status=MotionStatus.STATIONARY,
        balls=balls,
        source_refs=(source_ref,),
    )
    state.require_plannable()
    return state
