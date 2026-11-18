"""Convert model-independent pixel observations into calibrated tracker inputs."""

from __future__ import annotations

from collections.abc import Mapping

from billiards_ai.lab.vision.contracts import VisionFrameObservation
from billiards_ai.shared.calibration.homography import TableCalibration, pixel_to_table


def to_calibrated_tracker_frame(
    observation: VisionFrameObservation,
    *,
    calibration: TableCalibration,
    identities: Mapping[str, str],
) -> dict[str, object]:
    detections: list[dict[str, object]] = []
    for detection in observation.detections:
        center = pixel_to_table(calibration, detection.center.x, detection.center.y)
        detections.append(
            {
                "center_mm": (center.x_mm, center.y_mm),
                "identity": identities.get(detection.detection_id, "unknown"),
                "confidence": detection.confidence,
                "source_detection_id": detection.detection_id,
            }
        )
    return {
        "frame_id": observation.frame_id,
        "timestamp_ms": observation.timestamp_ms,
        "detections": detections,
    }
