"""Executable AD-13/18 contracts for versioned table calibration."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError


def _model_api():
    from billiards_ai.shared.models.calibration import (
        CalibrationRecord,
        CalibrationReference,
        CalibrationResolution,
        CalibrationStatus,
        CameraProfile,
        DistortionModel,
        DistortionParameters,
        TableProfile,
    )

    return {
        "CalibrationReference": CalibrationReference,
        "CalibrationRecord": CalibrationRecord,
        "CalibrationResolution": CalibrationResolution,
        "CalibrationStatus": CalibrationStatus,
        "CameraProfile": CameraProfile,
        "DistortionModel": DistortionModel,
        "DistortionParameters": DistortionParameters,
        "TableProfile": TableProfile,
    }


def _record_values(api: dict, **overrides):
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    values = {
        "calibration_id": uuid4(),
        "revision_id": uuid4(),
        "revision_number": 1,
        "device_id": "device-7",
        "camera_profile": api["CameraProfile"](
            camera_profile_id="overhead-v1",
            position_fingerprint="a" * 64,
            lens_fingerprint="b" * 64,
        ),
        "resolution": api["CalibrationResolution"](width_px=2540, height_px=1270),
        "table_profile": api["TableProfile"](
            table_profile_id="chinese-eight-ball-v1",
            length_mm=2540,
            width_mm=1270,
            configuration_fingerprint="c" * 64,
        ),
        "distortion": api["DistortionParameters"](
            model=api["DistortionModel"].NONE,
            verified=True,
        ),
        "corners_px": ((0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)),
        "homography": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "validation_references": (
            api["CalibrationReference"](pixel=(1000.0, 500.0), table={"x_mm": 1000, "y_mm": 500}),
        ),
        "source_image_sha256": "d" * 64,
        "status": api["CalibrationStatus"].ACTIVE,
        "validated_at": now,
        "max_reference_error_mm": 20.0,
        "created_at": now,
    }
    values.update(overrides)
    return values


def test_calibration_record_is_strict_frozen_and_complete():
    api = _model_api()
    record = api["CalibrationRecord"](**_record_values(api))

    assert record.schema_version == "1.0"
    assert record.status is api["CalibrationStatus"].ACTIVE
    assert record.max_reference_error_mm == 20.0
    assert record.camera_profile.camera_profile_id == "overhead-v1"
    assert record.resolution.width_px == 2540
    assert record.table_profile.table_profile_id == "chinese-eight-ball-v1"
    assert record.source_image_sha256 == "d" * 64

    with pytest.raises(ValidationError):
        api["CalibrationRecord"].model_validate(record.model_dump() | {"pixel_leak": 12})
    with pytest.raises(ValidationError):
        record.device_id = "mutated"


def test_revision_status_and_distortion_contracts_fail_closed():
    api = _model_api()

    with pytest.raises(ValidationError, match="invalidation reason"):
        api["CalibrationRecord"](**_record_values(api, status=api["CalibrationStatus"].STALE))

    unverified = api["DistortionParameters"](
        model=api["DistortionModel"].OPENCV,
        verified=False,
        camera_matrix=((1000.0, 0.0, 500.0), (0.0, 1000.0, 500.0), (0.0, 0.0, 1.0)),
        coefficients=(0.2, 0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValidationError, match="verified distortion"):
        api["CalibrationRecord"](**_record_values(api, distortion=unverified))


def test_later_revision_must_identify_the_revision_it_supersedes():
    api = _model_api()

    with pytest.raises(ValidationError, match="superseded revision"):
        api["CalibrationRecord"](**_record_values(api, revision_number=2))


def test_active_record_requires_strict_scalars_non_singular_matrix_and_references():
    api = _model_api()

    with pytest.raises(ValidationError):
        api["CalibrationRecord"](**_record_values(api, revision_number="1"))
    with pytest.raises(ValidationError, match="singular"):
        api["CalibrationRecord"](
            **_record_values(
                api,
                homography=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            )
        )
    with pytest.raises(ValidationError, match="independent reference"):
        api["CalibrationRecord"](**_record_values(api, validation_references=()))
