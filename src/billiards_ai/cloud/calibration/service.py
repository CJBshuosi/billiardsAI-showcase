"""Fail-closed calibration lifecycle and distance-production guard."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from billiards_ai.cloud.calibration.errors import (
    CalibrationConfigurationError,
    CalibrationExpiredError,
    CalibrationInvalidError,
    CalibrationNotYetValidError,
    CalibrationStaleError,
)
from billiards_ai.cloud.calibration.repository import CalibrationRepository
from billiards_ai.shared.calibration.homography import (
    CalibrationAccuracyError,
    CalibrationDriftError,
    NotCalibratedError,
    TableCalibration,
    assert_no_drift,
    calibrate,
    pixel_to_table,
    validate_reference_accuracy,
)
from billiards_ai.shared.models.calibration import (
    CalibrationRecord,
    CalibrationReference,
    CalibrationResolution,
    CalibrationStatus,
    CameraProfile,
    DistortionParameters,
    TableProfile,
)
from billiards_ai.shared.models.geometry import Point

logger = logging.getLogger(__name__)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC)


class CalibrationService:
    """Own status transitions and guard every pixel-to-distance conversion."""

    def __init__(self, repository: CalibrationRepository):
        self.repository = repository

    def create_calibration(
        self,
        *,
        device_id: str,
        camera_profile: CameraProfile,
        resolution: CalibrationResolution,
        table_profile: TableProfile,
        distortion: DistortionParameters,
        corners_px: tuple[tuple[float, float], ...],
        source_image_sha256: str,
        references: tuple[CalibrationReference, ...],
        validated_at: datetime,
        valid_until: datetime | None = None,
    ) -> CalibrationRecord:
        validated_at = _utc(validated_at)
        valid_until = None if valid_until is None else _utc(valid_until)
        geometry = calibrate(
            device_id,
            corners_px,
            table_length_mm=table_profile.length_mm,
            table_width_mm=table_profile.width_mm,
            distortion=distortion,
        )
        max_error = validate_reference_accuracy(geometry, references)
        current = self.repository.get_current(device_id)
        record = CalibrationRecord(
            calibration_id=uuid4() if current is None else current.calibration_id,
            revision_id=uuid4(),
            revision_number=1 if current is None else current.revision_number + 1,
            supersedes_revision_id=None if current is None else current.revision_id,
            device_id=device_id,
            camera_profile=camera_profile,
            resolution=resolution,
            table_profile=table_profile,
            distortion=distortion,
            corners_px=corners_px,
            homography=tuple(tuple(float(item) for item in row) for row in geometry.matrix),
            validation_references=references,
            source_image_sha256=source_image_sha256,
            status=CalibrationStatus.ACTIVE,
            validated_at=validated_at,
            valid_until=valid_until,
            max_reference_error_mm=max_error,
            created_at=validated_at,
        )
        return self.repository.append(
            record,
            expected_current_revision_id=None if current is None else current.revision_id,
        )

    @staticmethod
    def _geometry(record: CalibrationRecord) -> TableCalibration:
        return TableCalibration(
            device_id=record.device_id,
            corners_px=list(record.corners_px),
            table_length_mm=record.table_profile.length_mm,
            table_width_mm=record.table_profile.width_mm,
            matrix=[list(row) for row in record.homography],
            calibrated_at=record.validated_at,
            distortion=record.distortion,
        )

    @staticmethod
    def _guard_status(record: CalibrationRecord) -> None:
        if record.status is CalibrationStatus.STALE:
            raise CalibrationStaleError(
                f"设备 {record.device_id} 标定已过期或配置已变化，拒绝产出距离量"
            )
        if record.status is CalibrationStatus.INVALID:
            raise CalibrationInvalidError(f"设备 {record.device_id} 标定无效，拒绝产出距离量")

    def _append_status(
        self,
        current: CalibrationRecord,
        *,
        status: CalibrationStatus,
        reason: str,
        at: datetime,
    ) -> CalibrationRecord:
        payload = current.model_dump()
        payload.update(
            revision_id=uuid4(),
            revision_number=current.revision_number + 1,
            supersedes_revision_id=current.revision_id,
            status=status,
            invalidation_reason=reason,
            created_at=max(_utc(at), current.created_at, current.validated_at),
        )
        revision = CalibrationRecord.model_validate(payload)
        return self.repository.append(
            revision,
            expected_current_revision_id=current.revision_id,
        )

    def require_active(
        self,
        *,
        device_id: str,
        camera_profile: CameraProfile,
        resolution: CalibrationResolution,
        table_profile: TableProfile,
        distortion: DistortionParameters,
        at: datetime,
    ) -> CalibrationRecord:
        at = _utc(at)
        current = self.repository.get_current(device_id)
        if current is None:
            raise NotCalibratedError(f"设备 {device_id} 未完成台面标定，拒绝产出含距离量的结果")
        self._guard_status(current)
        if current.valid_until is not None and at >= current.valid_until:
            self._append_status(
                current,
                status=CalibrationStatus.STALE,
                reason="calibration expired",
                at=at,
            )
            raise CalibrationExpiredError(f"设备 {device_id} 标定已过期，拒绝产出距离量")
        if (
            current.camera_profile != camera_profile
            or current.resolution != resolution
            or current.table_profile != table_profile
            or current.distortion != distortion
        ):
            self._append_status(
                current,
                status=CalibrationStatus.STALE,
                reason="camera, resolution, lens, or table configuration changed",
                at=at,
            )
            raise CalibrationConfigurationError(
                f"设备 {device_id} 当前配置与标定记录不一致，拒绝产出距离量"
            )
        if at < current.validated_at:
            raise CalibrationNotYetValidError(
                f"设备 {device_id} 的帧早于标定验证时间，拒绝产出距离量"
            )
        try:
            validate_reference_accuracy(self._geometry(current), current.validation_references)
        except CalibrationAccuracyError as exc:
            self._append_status(
                current,
                status=CalibrationStatus.INVALID,
                reason="persisted independent reference evidence failed verification",
                at=at,
            )
            raise CalibrationInvalidError(
                f"设备 {device_id} 标定证据无法复核，拒绝产出距离量"
            ) from exc
        return current

    def pixel_to_table(
        self,
        *,
        device_id: str,
        camera_profile: CameraProfile,
        resolution: CalibrationResolution,
        table_profile: TableProfile,
        distortion: DistortionParameters,
        x_px: float,
        y_px: float,
        at: datetime,
    ) -> Point:
        record = self.require_active(
            device_id=device_id,
            camera_profile=camera_profile,
            resolution=resolution,
            table_profile=table_profile,
            distortion=distortion,
            at=at,
        )
        converted = pixel_to_table(self._geometry(record), x_px, y_px)
        latest = self.repository.get_current(device_id)
        if (
            latest is None
            or latest.revision_id != record.revision_id
            or latest.status is not CalibrationStatus.ACTIVE
        ):
            raise CalibrationInvalidError(
                f"设备 {device_id} 标定在坐标转换期间已变更，拒绝返回过期距离量"
            )
        return converted

    def invalidate(self, device_id: str, *, reason: str, at: datetime) -> CalibrationRecord:
        current = self.repository.get_current(device_id)
        if current is None:
            raise NotCalibratedError(f"设备 {device_id} 没有可失效的台面标定")
        if current.status is CalibrationStatus.INVALID:
            return current
        return self._append_status(
            current,
            status=CalibrationStatus.INVALID,
            reason=reason,
            at=at,
        )

    def check_drift(
        self,
        device_id: str,
        current_corners_px: tuple[tuple[float, float], ...],
        *,
        threshold_px: float = 15.0,
        checked_at: datetime,
    ) -> None:
        current = self.repository.get_current(device_id)
        if current is None:
            raise NotCalibratedError(f"设备 {device_id} 未完成台面标定，无法检查漂移")
        self._guard_status(current)
        geometry = self._geometry(current)
        try:
            assert_no_drift(geometry, list(current_corners_px), threshold_px)
        except CalibrationDriftError as exc:
            self._append_status(
                current,
                status=CalibrationStatus.INVALID,
                reason=str(exc),
                at=checked_at,
            )
            logger.warning(
                "calibration drift detected device_id=%s revision_id=%s reason=%s",
                device_id,
                current.revision_id,
                exc,
            )
            raise
