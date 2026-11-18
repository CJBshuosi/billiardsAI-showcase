"""Render the one authoritative Mainline 1 overlay from canonical product state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame
from billiards_ai.lab.shot_local.product import (
    CanonicalShotAnalysis,
    canonical_event_key,
    canonical_world_event_rows,
)
from billiards_ai.shared.calibration.homography import TableCalibration, table_to_pixel


@dataclass(frozen=True, slots=True)
class OverlayTrail:
    track_id: str
    points_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class OverlayEventMarker:
    event_key: tuple[object, ...]
    label: str


@dataclass(frozen=True, slots=True)
class OverlayAnnotations:
    shot_id: str | None
    track_labels: tuple[tuple[str, str], ...]
    trails: tuple[OverlayTrail, ...]
    current_events: tuple[OverlayEventMarker, ...]
    event_banners: tuple[str, ...]
    event_log: tuple[str, ...]


def _analysis_for_frame(
    analyses: tuple[CanonicalShotAnalysis, ...], frame_id: int
) -> CanonicalShotAnalysis | None:
    return next(
        (
            analysis
            for analysis in analyses
            if analysis.evidence.window.start_frame_id
            <= frame_id
            <= analysis.evidence.window.end_frame_id
        ),
        None,
    )


def _event_label(row: dict[str, object]) -> str:
    event_type = str(row["event_type"])
    actor = row.get("actor_id") or "unknown"
    layer = f" {row['detection_layer']}" if row.get("detection_layer") else ""
    if event_type == "ball_collision":
        return f"COLLISION{layer} {actor} -> {row.get('target_id') or 'unknown'}"
    if event_type == "cushion":
        return f"CUSHION{layer} {actor} -> {row.get('rail') or 'unknown'}"
    if event_type == "first_contact":
        return f"FIRST CONTACT {actor} -> {row.get('target_id') or 'unknown'}"
    if event_type == "pot":
        return f"POT {actor} -> {row.get('pocket') or 'unknown'}"
    if event_type == "shot_start":
        return f"SHOT START {actor}"
    if event_type == "shot_end":
        return f"SHOT END {actor}"
    return f"{event_type.upper()} {actor}"


def overlay_annotations_for_frame(
    analyses: tuple[CanonicalShotAnalysis, ...],
    *,
    frame_id: int,
    trail_samples: int = 18,
) -> OverlayAnnotations:
    """Project canonical tracks and admitted events onto one source frame."""

    analysis = _analysis_for_frame(analyses, frame_id)
    if analysis is None:
        return OverlayAnnotations(None, (), (), (), (), ())

    labels: list[tuple[str, str]] = []
    trails: list[OverlayTrail] = []
    for track in analysis.active_tracks:
        prior = tuple(sample for sample in track.samples if sample.frame_id <= frame_id)
        current = next((sample for sample in reversed(prior) if sample.frame_id == frame_id), None)
        if current is not None:
            labels.append((track.track_id, current.state))
        if prior:
            trails.append(
                OverlayTrail(
                    track_id=track.track_id,
                    points_mm=tuple(
                        (sample.x_mm, sample.y_mm) for sample in prior[-trail_samples:]
                    ),
                )
            )

    rows = [
        row
        for row in canonical_world_event_rows(analysis)
        if row["status"] == "observed" and row["decision_frame_id"] == frame_id
    ]
    markers = tuple(OverlayEventMarker(canonical_event_key(row), _event_label(row)) for row in rows)
    admitted = tuple(
        _event_label(row)
        for row in canonical_world_event_rows(analysis)
        if row["status"] == "observed" and int(row["decision_frame_id"]) <= frame_id
    )
    return OverlayAnnotations(
        shot_id=analysis.evidence.shot_id,
        track_labels=tuple(labels),
        trails=tuple(trails),
        current_events=markers,
        event_banners=tuple(marker.label for marker in markers),
        event_log=admitted,
    )


def calibration_overlay_geometry(
    calibration: TableCalibration,
    *,
    cushion_distance_mm: float,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Map the audited table boundary, cushion band, and pockets to source pixels."""

    if cushion_distance_mm <= 0:
        raise ValueError("cushion distance must be positive")
    length = float(calibration.table_length_mm)
    width = float(calibration.table_width_mm)
    if cushion_distance_mm * 2 >= min(length, width):
        raise ValueError("cushion distance is too large for calibrated table")

    def mapped(points: tuple[tuple[float, float], ...]) -> tuple[tuple[int, int], ...]:
        return tuple(table_to_pixel(calibration, x_mm, y_mm) for x_mm, y_mm in points)

    return {
        "boundary": mapped(((0, 0), (length, 0), (length, width), (0, width))),
        "cushion_band": mapped(
            (
                (cushion_distance_mm, cushion_distance_mm),
                (length - cushion_distance_mm, cushion_distance_mm),
                (length - cushion_distance_mm, width - cushion_distance_mm),
                (cushion_distance_mm, width - cushion_distance_mm),
            )
        ),
        "pockets": mapped(
            (
                (0, 0),
                (length / 2, 0),
                (length, 0),
                (0, width),
                (length / 2, width),
                (length, width),
            )
        ),
    }


def _draw_calibration(
    image: np.ndarray,
    calibration: TableCalibration,
    cushion_distance_mm: float,
) -> None:
    geometry = calibration_overlay_geometry(
        calibration,
        cushion_distance_mm=cushion_distance_mm,
    )
    cv2.polylines(
        image,
        [np.asarray(geometry["boundary"], dtype=np.int32)],
        True,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.polylines(
        image,
        [np.asarray(geometry["cushion_band"], dtype=np.int32)],
        True,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    for center in geometry["pockets"]:
        cv2.drawMarker(
            image,
            center,
            (0, 255, 255),
            cv2.MARKER_DIAMOND,
            16,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "CALIBRATION: cyan=boundary magenta=cushion-band yellow=pocket",
        (16, max(18, image.shape[0] - 42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def render_canonical_overlay(
    *,
    decoded_frames: tuple[DecodedTrackingFrame, ...],
    analyses: tuple[CanonicalShotAnalysis, ...],
    calibration: TableCalibration,
    cushion_distance_mm: float,
    fps: float,
    output: Path,
) -> None:
    """Render canonical events as H.264 MP4 without fragment identities."""

    height, width = decoded_frames[0].image.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise ValueError("H.264 (avc1) encoder unavailable for canonical overlay")
        for decoded in decoded_frames:
            image = decoded.image.copy()
            _draw_calibration(image, calibration, cushion_distance_mm)
            annotations = overlay_annotations_for_frame(
                analyses,
                frame_id=decoded.frame_id,
            )
            cv2.putText(
                image,
                f"FRAME {decoded.frame_id} TIME {decoded.timestamp_ms} ms "
                f"{annotations.shot_id or 'NO SHOT'}",
                (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            analysis = _analysis_for_frame(analyses, decoded.frame_id)
            if analysis is not None:
                for trail in annotations.trails:
                    points = np.asarray(
                        [table_to_pixel(calibration, x, y) for x, y in trail.points_mm],
                        dtype=np.int32,
                    )
                    if len(points) > 1:
                        cv2.polylines(image, [points], False, (0, 165, 255), 2, cv2.LINE_AA)
                for track in analysis.active_tracks:
                    sample = next(
                        (item for item in track.samples if item.frame_id == decoded.frame_id),
                        None,
                    )
                    if sample is None:
                        continue
                    center = table_to_pixel(calibration, sample.x_mm, sample.y_mm)
                    color = {
                        "observed": (80, 220, 80),
                        "reacquired": (0, 220, 255),
                        "predicted": (160, 160, 160),
                    }[sample.state]
                    cv2.circle(image, center, 9, color, 2, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        f"{track.track_id} {sample.state}",
                        (center[0] + 11, max(16, center[1] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                cue_prior = tuple(
                    sample
                    for sample in analysis.evidence.cue_samples
                    if sample.frame_id <= decoded.frame_id
                    and sample.x_mm is not None
                    and sample.y_mm is not None
                )[-18:]
                if len(cue_prior) > 1:
                    cue_points = np.asarray(
                        [
                            table_to_pixel(calibration, sample.x_mm, sample.y_mm)
                            for sample in cue_prior
                        ],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        image,
                        [cue_points],
                        False,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cue = next(
                    (
                        sample
                        for sample in analysis.evidence.cue_samples
                        if sample.frame_id == decoded.frame_id
                        and sample.x_mm is not None
                        and sample.y_mm is not None
                    ),
                    None,
                )
                if cue is not None:
                    center = table_to_pixel(calibration, cue.x_mm, cue.y_mm)
                    cv2.circle(image, center, 10, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        f"{cue.local_id} {cue.state.value}",
                        (center[0] + 12, max(16, center[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )
            for index, label in enumerate(annotations.event_banners):
                cv2.putText(
                    image,
                    label,
                    (16, 58 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 110, 255),
                    2,
                    cv2.LINE_AA,
                )
            log_x = max(16, width - 620)
            for index, label in enumerate(annotations.event_log[-12:]):
                cv2.putText(
                    image,
                    label,
                    (log_x, 26 + index * 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(image)
    finally:
        writer.release()
