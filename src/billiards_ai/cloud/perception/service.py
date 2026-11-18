"""Calibration-gated conversion from pixel observations to FrameDetections."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from billiards_ai.cloud.calibration import CalibrationService
from billiards_ai.cloud.perception.detector import ClassicCvBallDetector
from billiards_ai.cloud.perception.errors import DetectionInputError
from billiards_ai.shared.constants import BALL_IDS
from billiards_ai.shared.media_timing import pts_to_milliseconds
from billiards_ai.shared.models.calibration import (
    CalibrationLineage,
    CalibrationResolution,
    CameraProfile,
    DistortionParameters,
    TableProfile,
)
from billiards_ai.shared.models.detection import (
    DetectionSource,
    FrameDetection,
    FrameDetections,
    FrameQuality,
    RejectedDetection,
    RejectionReason,
)
from billiards_ai.shared.models.media import FrameManifest, NormalizedMedia


class BallLocatorService:
    """Consume Story 2.4 contracts and require Story 2.3 before all output."""

    def __init__(
        self,
        *,
        calibration_service: CalibrationService,
        detector: ClassicCvBallDetector,
    ):
        self.calibration_service = calibration_service
        self.detector = detector

    @staticmethod
    def _validate_media_contract(
        normalized_media: NormalizedMedia,
        manifest: FrameManifest,
        frame_index: int,
        image_bgr: np.ndarray,
    ) -> None:
        comparable_fields = (
            "input_sha256",
            "processing_config_sha256",
            "tool_versions",
            "equivalence_rule",
        )
        if any(
            getattr(normalized_media, field) != getattr(manifest, field)
            for field in comparable_fields
        ):
            raise DetectionInputError("normalized media contract does not match frame manifest")
        if frame_index < 0 or frame_index >= len(manifest.frames):
            raise DetectionInputError("frame index is outside the manifest")
        expected_shape = (
            normalized_media.resolution.height_px,
            normalized_media.resolution.width_px,
            3,
        )
        if image_bgr.dtype != np.uint8 or image_bgr.shape != expected_shape:
            raise DetectionInputError("frame image does not match normalized media resolution")

    @staticmethod
    def _quality(image_bgr: np.ndarray) -> FrameQuality:
        mean = float(np.mean(image_bgr))
        if mean < 20.0 or mean > 240.0:
            return FrameQuality.UNUSABLE
        if mean < 35.0 or mean > 220.0:
            return FrameQuality.DEGRADED
        return FrameQuality.GOOD

    def locate_frame(
        self,
        *,
        normalized_media: NormalizedMedia,
        manifest: FrameManifest,
        frame_index: int,
        image_bgr: np.ndarray,
        device_id: str,
        camera_profile: CameraProfile,
        resolution: CalibrationResolution,
        table_profile: TableProfile,
        distortion: DistortionParameters,
        at: datetime,
        expected_ball_ids: tuple[int, ...] | None = None,
    ) -> FrameDetections:
        self._validate_media_contract(normalized_media, manifest, frame_index, image_bgr)
        if (
            normalized_media.resolution.width_px != resolution.width_px
            or normalized_media.resolution.height_px != resolution.height_px
        ):
            raise DetectionInputError(
                "normalized media resolution does not match calibration resolution"
            )
        expected_ball_ids = tuple(BALL_IDS) if expected_ball_ids is None else expected_ball_ids
        if tuple(sorted(set(expected_ball_ids))) != expected_ball_ids or any(
            ball_id not in BALL_IDS for ball_id in expected_ball_ids
        ):
            raise DetectionInputError(
                "expected ball IDs must be unique, sorted, and between 0 and 15"
            )

        # This guard runs before CV so even an empty frame cannot bypass ACTIVE calibration.
        calibration = self.calibration_service.require_active(
            device_id=device_id,
            camera_profile=camera_profile,
            resolution=resolution,
            table_profile=table_profile,
            distortion=distortion,
            at=at,
        )

        detections: list[FrameDetection] = []
        rejected: list[RejectedDetection] = []
        source = DetectionSource.CLASSIC_CV_CIRCLE_COLOR
        for observation in self.detector.detect(image_bgr):
            if observation.rejection_reason is not None:
                rejected.append(
                    RejectedDetection(
                        candidates=observation.candidates,
                        reason=observation.rejection_reason,
                        source=source,
                        raw_observation=observation.raw_observation,
                    )
                )
                continue
            point = self.calibration_service.pixel_to_table(
                device_id=device_id,
                camera_profile=camera_profile,
                resolution=resolution,
                table_profile=table_profile,
                distortion=distortion,
                x_px=observation.raw_observation.center_px[0],
                y_px=observation.raw_observation.center_px[1],
                at=at,
            )
            if not point.is_on_table():
                rejected.append(
                    RejectedDetection(
                        candidates=observation.candidates,
                        reason=RejectionReason.OUTSIDE_TABLE,
                        source=source,
                        raw_observation=observation.raw_observation,
                    )
                )
                continue
            detections.append(
                FrameDetection(
                    candidates=observation.candidates,
                    position=point,
                    confidence=observation.confidence,
                    source=source,
                    raw_observation=observation.raw_observation,
                )
            )

        # A candidate list is evidence of an unresolved identity, not proof that
        # any candidate has a usable table position.  Only the unambiguous,
        # high-confidence view is allowed to clear a ball from ``missing``.
        observed_ids = set(
            FrameDetections(
                timestamp_ms=0,
                detections=tuple(detections),
                missing_ball_ids=(),
                frame_quality=self._quality(image_bgr),
                detector_version=self.detector.version,
                calibration=CalibrationLineage.from_record(calibration),
                rejected=tuple(rejected),
            ).high_confidence_positions()
        )
        missing = tuple(ball_id for ball_id in expected_ball_ids if ball_id not in observed_ids)
        current_calibration = self.calibration_service.require_active(
            device_id=device_id,
            camera_profile=camera_profile,
            resolution=resolution,
            table_profile=table_profile,
            distortion=distortion,
            at=at,
        )
        if current_calibration.revision_id != calibration.revision_id:
            raise DetectionInputError("calibration changed while locating the frame")
        return FrameDetections(
            timestamp_ms=pts_to_milliseconds(
                manifest.frames[frame_index].source_pts - manifest.first_source_pts,
                manifest.source_time_base,
            ),
            detections=tuple(detections),
            missing_ball_ids=missing,
            frame_quality=self._quality(image_bgr),
            detector_version=self.detector.version,
            calibration=CalibrationLineage.from_record(current_calibration),
            rejected=tuple(rejected),
        )
