"""Filesystem integration tests for versioned calibration records."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest


def _api():
    from billiards_ai.cloud.calibration import (
        CalibrationConflictError,
        CalibrationRepositoryError,
        JsonCalibrationRepository,
    )
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
        "CalibrationConflictError": CalibrationConflictError,
        "CalibrationRepositoryError": CalibrationRepositoryError,
        "JsonCalibrationRepository": JsonCalibrationRepository,
        "CalibrationRecord": CalibrationRecord,
        "CalibrationReference": CalibrationReference,
        "CalibrationResolution": CalibrationResolution,
        "CalibrationStatus": CalibrationStatus,
        "CameraProfile": CameraProfile,
        "DistortionModel": DistortionModel,
        "DistortionParameters": DistortionParameters,
        "TableProfile": TableProfile,
    }


def _record(api: dict, *, previous=None, revision_id=None):
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    revision_number = 1 if previous is None else previous.revision_number + 1
    created_at = now if previous is None else previous.created_at + timedelta(seconds=1)
    return api["CalibrationRecord"](
        calibration_id=uuid4() if previous is None else previous.calibration_id,
        revision_id=revision_id or uuid4(),
        revision_number=revision_number,
        supersedes_revision_id=None if previous is None else previous.revision_id,
        device_id="device-7",
        camera_profile=api["CameraProfile"](
            camera_profile_id="overhead-v1",
            position_fingerprint="a" * 64,
            lens_fingerprint="b" * 64,
        ),
        resolution=api["CalibrationResolution"](width_px=2540, height_px=1270),
        table_profile=api["TableProfile"](
            table_profile_id="table-v1",
            length_mm=2540,
            width_mm=1270,
            configuration_fingerprint="c" * 64,
        ),
        distortion=api["DistortionParameters"](
            model=api["DistortionModel"].NONE,
            verified=True,
        ),
        corners_px=((0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)),
        homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        validation_references=(
            api["CalibrationReference"](pixel=(1000.0, 500.0), table={"x_mm": 1000, "y_mm": 500}),
        ),
        source_image_sha256="d" * 64,
        status=api["CalibrationStatus"].ACTIVE,
        validated_at=now,
        max_reference_error_mm=0.0,
        created_at=created_at,
    )


def test_round_trip_is_private_versioned_and_strict(tmp_path):
    api = _api()
    store = tmp_path / "calibration" / "store.json"
    repository = api["JsonCalibrationRepository"](store)
    first = _record(api)

    repository.append(first, expected_current_revision_id=None)

    assert repository.path == store.resolve()
    assert store.parent.stat().st_mode & 0o777 == 0o700
    assert store.stat().st_mode & 0o777 == 0o600
    assert repository.get_current("device-7") == first
    assert repository.get_history("device-7") == (first,)
    payload = json.loads(store.read_text())
    assert payload["schema_version"] == "1.0"

    payload["devices"]["device-7"][0]["unexpected"] = True
    store.write_text(json.dumps(payload))
    store.chmod(0o600)
    with pytest.raises(api["CalibrationRepositoryError"]):
        repository.get_current("device-7")


def test_corrupt_unsafe_and_unknown_schema_stores_fail_closed(tmp_path):
    api = _api()
    store = tmp_path / "calibration" / "store.json"
    store.parent.mkdir(mode=0o700)
    store.write_text("not-json")
    store.chmod(0o600)
    with pytest.raises(api["CalibrationRepositoryError"]):
        api["JsonCalibrationRepository"](store).get_history("device-7")

    store.write_text('{"schema_version":"999","revision":0,"devices":{}}')
    store.chmod(0o600)
    with pytest.raises(api["CalibrationRepositoryError"]):
        api["JsonCalibrationRepository"](store).get_history("device-7")

    store.write_text('{"schema_version":"1.0","revision":0,"devices":{}}')
    store.chmod(0o644)
    with pytest.raises(api["CalibrationRepositoryError"]):
        api["JsonCalibrationRepository"](store).get_history("device-7")


def test_atomic_replace_failure_preserves_the_previous_history(tmp_path, monkeypatch):
    api = _api()
    from billiards_ai.cloud.calibration import repository as repository_module

    store = tmp_path / "calibration" / "store.json"
    repository = api["JsonCalibrationRepository"](store)
    first = _record(api)
    repository.append(first, expected_current_revision_id=None)
    original = store.read_bytes()
    second = _record(api, previous=first)

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(repository_module.os, "replace", fail_replace)
    with pytest.raises(api["CalibrationRepositoryError"]):
        repository.append(second, expected_current_revision_id=first.revision_id)

    assert store.read_bytes() == original
    assert not list(store.parent.glob(f".{store.name}.*"))


def test_concurrent_compare_and_swap_leaves_one_contiguous_revision(tmp_path):
    api = _api()
    store = tmp_path / "calibration" / "store.json"
    repository = api["JsonCalibrationRepository"](store)
    first = _record(api)
    repository.append(first, expected_current_revision_id=None)
    candidates = [_record(api, previous=first), _record(api, previous=first)]

    def append(candidate):
        try:
            return repository.append(
                candidate, expected_current_revision_id=first.revision_id
            ).revision_id
        except api["CalibrationConflictError"]:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(append, candidates))

    assert sum(result is not None for result in results) == 1
    history = repository.get_history("device-7")
    assert [item.revision_number for item in history] == [1, 2]
    assert history[1].supersedes_revision_id == history[0].revision_id


def test_symlink_store_is_rejected(tmp_path):
    api = _api()
    directory = tmp_path / "calibration"
    directory.mkdir(mode=0o700)
    target = directory / "target.json"
    target.write_text('{"schema_version":"1.0","revision":0,"devices":{}}')
    target.chmod(0o600)
    link = directory / "store.json"
    link.symlink_to(target)

    with pytest.raises(api["CalibrationRepositoryError"]):
        api["JsonCalibrationRepository"](link).get_current("device-7")


def test_dangling_symlink_store_is_rejected(tmp_path):
    api = _api()
    directory = tmp_path / "calibration"
    directory.mkdir(mode=0o700)
    link = directory / "store.json"
    link.symlink_to(directory / "missing.json")

    with pytest.raises(api["CalibrationRepositoryError"]):
        api["JsonCalibrationRepository"](link).get_current("device-7")
