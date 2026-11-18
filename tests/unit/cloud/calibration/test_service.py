"""Lifecycle and fail-closed tests for the calibration service."""

from datetime import UTC, datetime, timedelta

import pytest


def _api():
    from billiards_ai.cloud.calibration import (
        CalibrationAccuracyError,
        CalibrationConfigurationError,
        CalibrationDriftError,
        CalibrationExpiredError,
        CalibrationInvalidError,
        CalibrationNotYetValidError,
        CalibrationService,
        JsonCalibrationRepository,
        NotCalibratedError,
    )
    from billiards_ai.shared.models import Point
    from billiards_ai.shared.models.calibration import (
        CalibrationReference,
        CalibrationResolution,
        CalibrationStatus,
        CameraProfile,
        DistortionModel,
        DistortionParameters,
        TableProfile,
    )

    return {
        "CalibrationAccuracyError": CalibrationAccuracyError,
        "CalibrationConfigurationError": CalibrationConfigurationError,
        "CalibrationDriftError": CalibrationDriftError,
        "CalibrationExpiredError": CalibrationExpiredError,
        "CalibrationInvalidError": CalibrationInvalidError,
        "CalibrationNotYetValidError": CalibrationNotYetValidError,
        "CalibrationService": CalibrationService,
        "JsonCalibrationRepository": JsonCalibrationRepository,
        "NotCalibratedError": NotCalibratedError,
        "Point": Point,
        "CalibrationReference": CalibrationReference,
        "CalibrationResolution": CalibrationResolution,
        "CalibrationStatus": CalibrationStatus,
        "CameraProfile": CameraProfile,
        "DistortionModel": DistortionModel,
        "DistortionParameters": DistortionParameters,
        "TableProfile": TableProfile,
    }


def _context(api: dict):
    return {
        "camera_profile": api["CameraProfile"](
            camera_profile_id="overhead-v1",
            position_fingerprint="a" * 64,
            lens_fingerprint="b" * 64,
        ),
        "resolution": api["CalibrationResolution"](width_px=2540, height_px=1270),
        "table_profile": api["TableProfile"](
            table_profile_id="table-v1",
            length_mm=2540,
            width_mm=1270,
            configuration_fingerprint="c" * 64,
        ),
        "distortion": api["DistortionParameters"](
            model=api["DistortionModel"].NONE,
            verified=True,
        ),
    }


def _create(service, api: dict, *, reference_error_mm: int = 0, valid_until=None):
    context = _context(api)
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    record = service.create_calibration(
        device_id="device-7",
        **context,
        corners_px=((0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)),
        source_image_sha256="d" * 64,
        references=(
            api["CalibrationReference"](
                pixel=(1000.0, 500.0),
                table=api["Point"](x_mm=1000 + reference_error_mm, y_mm=500),
            ),
        ),
        validated_at=now,
        valid_until=valid_until,
    )
    return record, context, now


def test_exactly_twenty_millimetres_is_active_and_twenty_one_is_rejected(tmp_path):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)

    record, _, _ = _create(service, api, reference_error_mm=20)

    assert record.status is api["CalibrationStatus"].ACTIVE
    assert record.max_reference_error_mm == 20.0

    other = api["CalibrationService"](
        api["JsonCalibrationRepository"](tmp_path / "other" / "store.json")
    )
    with pytest.raises(api["CalibrationAccuracyError"], match="21"):
        _create(other, api, reference_error_mm=21)
    assert other.repository.get_current("device-7") is None


def test_empty_independent_reference_set_is_typed_accuracy_failure(tmp_path):
    api = _api()
    service = api["CalibrationService"](
        api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    )
    context = _context(api)
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)

    with pytest.raises(api["CalibrationAccuracyError"], match="independent reference"):
        service.create_calibration(
            device_id="device-7",
            **context,
            corners_px=(
                (0.0, 0.0),
                (2540.0, 0.0),
                (2540.0, 1270.0),
                (0.0, 1270.0),
            ),
            source_image_sha256="d" * 64,
            references=(),
            validated_at=now,
        )


def test_missing_calibration_never_returns_a_distance(tmp_path):
    api = _api()
    service = api["CalibrationService"](
        api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    )

    with pytest.raises(api["NotCalibratedError"]):
        service.pixel_to_table(
            device_id="missing",
            **_context(api),
            x_px=100.0,
            y_px=100.0,
            at=datetime(2026, 7, 28, 12, tzinfo=UTC),
        )


def test_expiry_marks_current_revision_stale_before_rejecting(tmp_path):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    expiry = datetime(2026, 7, 28, 13, tzinfo=UTC)
    _, context, _ = _create(service, api, valid_until=expiry)

    with pytest.raises(api["CalibrationExpiredError"]):
        service.pixel_to_table(
            device_id="device-7",
            **context,
            x_px=100.0,
            y_px=100.0,
            at=expiry + timedelta(microseconds=1),
        )

    current = repository.get_current("device-7")
    assert current.status is api["CalibrationStatus"].STALE
    assert current.revision_number == 2
    assert "expired" in current.invalidation_reason


@pytest.mark.parametrize(
    "changed_field", ["camera_profile", "resolution", "table_profile", "distortion"]
)
def test_configuration_change_marks_old_record_stale(tmp_path, changed_field):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / changed_field / "calibration.json")
    service = api["CalibrationService"](repository)
    _, context, now = _create(service, api)
    changed = dict(context)
    if changed_field == "camera_profile":
        changed[changed_field] = context[changed_field].model_copy(
            update={"position_fingerprint": "e" * 64}
        )
    elif changed_field == "resolution":
        changed[changed_field] = context[changed_field].model_copy(update={"width_px": 1920})
    elif changed_field == "table_profile":
        changed[changed_field] = context[changed_field].model_copy(
            update={"configuration_fingerprint": "f" * 64}
        )
    else:
        changed[changed_field] = context[changed_field].model_copy(update={"verified": False})

    with pytest.raises(api["CalibrationConfigurationError"]):
        service.pixel_to_table(device_id="device-7", **changed, x_px=100.0, y_px=100.0, at=now)

    assert repository.get_current("device-7").status is api["CalibrationStatus"].STALE


def test_explicit_invalidation_blocks_distance_output(tmp_path):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    _, context, now = _create(service, api)

    service.invalidate("device-7", reason="manual inspection failed", at=now)

    with pytest.raises(api["CalibrationInvalidError"]):
        service.pixel_to_table(device_id="device-7", **context, x_px=100.0, y_px=100.0, at=now)


def test_drift_appends_invalid_revision_warns_and_raises(tmp_path, caplog):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    _, _, now = _create(service, api)

    with caplog.at_level("WARNING"), pytest.raises(api["CalibrationDriftError"]):
        service.check_drift(
            "device-7",
            ((16.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)),
            threshold_px=15.0,
            checked_at=now,
        )

    current = repository.get_current("device-7")
    assert current.status is api["CalibrationStatus"].INVALID
    assert current.revision_number == 2
    assert "device-7" in caplog.text
    assert "drift" in caplog.text


def test_nonfinite_drift_evidence_invalidates_before_rejecting(tmp_path):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    _, _, now = _create(service, api)

    with pytest.raises(api["CalibrationDriftError"], match="finite"):
        service.check_drift(
            "device-7",
            ((float("nan"), 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)),
            checked_at=now,
        )

    assert repository.get_current("device-7").status is api["CalibrationStatus"].INVALID


def test_out_of_order_configuration_event_still_records_stale_revision(tmp_path):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    _, context, now = _create(service, api)
    changed = dict(context)
    changed["resolution"] = context["resolution"].model_copy(update={"width_px": 1920})

    with pytest.raises(api["CalibrationConfigurationError"]):
        service.pixel_to_table(
            device_id="device-7",
            **changed,
            x_px=100.0,
            y_px=100.0,
            at=now - timedelta(seconds=1),
        )

    current = repository.get_current("device-7")
    assert current.status is api["CalibrationStatus"].STALE
    assert current.created_at >= current.validated_at


def test_frames_before_validation_are_rejected(tmp_path):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    _, context, now = _create(service, api)

    with pytest.raises(api["CalibrationNotYetValidError"]):
        service.pixel_to_table(
            device_id="device-7",
            **context,
            x_px=100.0,
            y_px=100.0,
            at=now - timedelta(microseconds=1),
        )


def test_conversion_rechecks_revision_when_it_is_invalidated_mid_operation(tmp_path, monkeypatch):
    api = _api()
    repository = api["JsonCalibrationRepository"](tmp_path / "calibration" / "store.json")
    service = api["CalibrationService"](repository)
    _, context, now = _create(service, api)
    original_geometry = service._geometry
    calls = 0

    def invalidate_before_conversion(record):
        nonlocal calls
        calls += 1
        if calls == 2:
            service.invalidate("device-7", reason="concurrent drift", at=now)
        return original_geometry(record)

    monkeypatch.setattr(service, "_geometry", invalidate_before_conversion)

    with pytest.raises(api["CalibrationInvalidError"], match="转换期间"):
        service.pixel_to_table(device_id="device-7", **context, x_px=100.0, y_px=100.0, at=now)
