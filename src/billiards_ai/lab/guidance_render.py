"""Device-independent guidance scene projection and local media overlays."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.shared.calibration.homography import TableCalibration, table_to_pixel
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import GuidanceScene, Recommendation

ROUTE_COLORS = ((255, 255, 0), (0, 200, 255), (255, 120, 120))


def build_guidance_scene(recommendation: Recommendation) -> GuidanceScene:
    identity = hashlib.sha256(
        recommendation.model_dump_json(exclude_none=True).encode()
    ).hexdigest()
    return GuidanceScene(
        recommendation_id=identity,
        calibration_id=recommendation.calibration_id,
        routes=recommendation.candidates,
        evidence_refs=recommendation.evidence_refs,
    )


def _point_list(points: tuple[Point, ...]) -> list[list[int]]:
    return [[point.x_mm, point.y_mm] for point in points]


def mobile_scene_payload(scene: GuidanceScene) -> dict[str, object]:
    return {
        "schema_version": scene.schema_version,
        "coordinate_space": scene.coordinate_space,
        "recommendation_id": scene.recommendation_id,
        "calibration_id": scene.calibration_id,
        "evidence_refs": list(scene.evidence_refs),
        "routes": [
            {
                "candidate_id": route.candidate_id,
                "rank": rank,
                "target_track_id": route.target_track_id,
                "target_identity": route.target_identity,
                "target_kind": route.target_kind.value,
                "pocket": route.pocket.value,
                "ghost_ball_center_mm": (
                    [route.ghost_ball_center.x_mm, route.ghost_ball_center.y_mm]
                    if route.ghost_ball_center is not None
                    else None
                ),
                "cue_path_mm": _point_list(route.cue_path),
                "object_path_mm": _point_list(route.object_path),
                "score": route.score,
            }
            for rank, route in enumerate(scene.routes, start=1)
        ],
    }


def _pixel_polyline(points: tuple[Point, ...], calibration: TableCalibration) -> np.ndarray:
    return np.asarray(
        [table_to_pixel(calibration, point.x_mm, point.y_mm) for point in points],
        dtype=np.int32,
    )


def render_scene_frame(
    frame: np.ndarray,
    scene: GuidanceScene,
    calibration: TableCalibration,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    output = frame.copy()
    for rank, route in enumerate(scene.routes):
        color = ROUTE_COLORS[min(rank, len(ROUTE_COLORS) - 1)]
        if len(route.cue_path) >= 2:
            cv2.polylines(
                output,
                [_pixel_polyline(route.cue_path, calibration)],
                False,
                color,
                3,
                cv2.LINE_AA,
            )
        if len(route.object_path) >= 2:
            cv2.polylines(
                output,
                [_pixel_polyline(route.object_path, calibration)],
                False,
                (0, 200, 255),
                3,
                cv2.LINE_AA,
            )
        if route.ghost_ball_center is not None:
            center = table_to_pixel(
                calibration,
                route.ghost_ball_center.x_mm,
                route.ghost_ball_center.y_mm,
            )
            cv2.circle(output, center, 10, color, 2, cv2.LINE_AA)
            cv2.putText(
                output,
                str(rank + 1),
                (center[0] + 12, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def _decoded_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def render_scene_video(
    source: Path,
    destination: Path,
    scene: GuidanceScene,
    calibration: TableCalibration,
) -> Path:
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("video input cannot be decoded")
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    codec = "mp4v" if destination.suffix.lower() == ".mp4" else "MJPG"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError("video output cannot be opened")
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(render_scene_frame(frame, scene, calibration))
            written += 1
    finally:
        capture.release()
        writer.release()
    if written == 0 or _decoded_frame_count(temporary) != written:
        temporary.unlink(missing_ok=True)
        raise ValueError("rendered video failed frame-count verification")
    os.replace(temporary, destination)
    return destination
