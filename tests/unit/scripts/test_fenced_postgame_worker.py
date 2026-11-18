"""Cancelled captured work must not initialize runtime or model resources."""

import json
from pathlib import Path

import pytest

from scripts.algorithm import run_mainline1_shot_worker as worker
from tests.unit.lab.test_mainline1_recording_media_worker import _legacy_scene


def test_completed_unselected_session_exits_before_resource_or_model_loading(
    tmp_path, monkeypatch, capsys
):
    records, jobs, source, root, encoder = _legacy_scene(tmp_path)
    manifest = records.get_postgame_manifest(source.session_id)
    records.complete_postgame_review("legacy-artifact", source.session_id, manifest.manifest_sha256)

    def forbidden(*args, **kwargs):
        pytest.fail("fenced session initialized worker resource or model loading")

    monkeypatch.setattr(worker, "_resource_gate", forbidden)
    monkeypatch.setattr(worker, "load_mainline1_shot_local_config", forbidden)
    result = worker.main(
        [
            "--state-db",
            str(records.path),
            "--jobs-db",
            str(tmp_path / "jobs.sqlite3"),
            "--artifact-root",
            str(root / "postgame" / source.session_id),
            "--session-id",
            source.session_id,
            "--worker-id",
            "fenced-worker",
            "--purpose",
            "analysis",
            "--once",
            "--capture-setup",
            str(tmp_path / "not-loaded-setup.json"),
            "--calibration-root",
            str(tmp_path / "not-loaded-calibration"),
            "--config",
            str(tmp_path / "not-loaded-config.yaml"),
            "--detector-training-run",
            str(tmp_path / "not-loaded-detector"),
            "--identity-training-run",
            str(tmp_path / "not-loaded-identity"),
            "--dinov2-assets",
            str(tmp_path / "not-loaded-dinov2"),
            "--fps",
            "60",
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize("damage", [None, "config", "scope"])
def test_fenced_managed_entry_keeps_context_checks_without_reading_model_assets(
    tmp_path,
    monkeypatch,
    capsys,
    damage,
):
    from tests.unit.edge.route_assistance.test_original_owner_cleanup import _scene
    from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
        SCOPE,
        SESSION,
        task,
    )

    records, jobs, source, root, jobs_root, jobs_db, output = _scene(tmp_path)
    records.accept_owner_delete_task(task(records), managed_scope=SCOPE)
    record = records.get_managed_resource_spec_for_session(SCOPE, SESSION)
    spec = record.spec
    Path(spec.paths["ring_root"]).mkdir(parents=True, exist_ok=True)

    def forbidden(*args, **kwargs):
        pytest.fail("fenced managed work initialized resources or model loading")

    monkeypatch.setattr(worker, "_resource_gate", forbidden)
    arguments = [
        "--state-db",
        str(records.path),
        "--jobs-db",
        str(jobs_db),
        "--artifact-root",
        str(root / "postgame" / SESSION),
        "--session-id",
        SESSION,
        "--worker-id",
        "fenced-managed",
        "--purpose",
        "analysis",
        "--once",
        "--managed-environment-id",
        SCOPE.environment_id,
        "--managed-device-id",
        SCOPE.device_id if damage != "scope" else "foreign-device",
        "--managed-table-id",
        SCOPE.table_id,
        "--managed-open-request-id",
        record.open_request_id,
        "--managed-resource-spec-sha256",
        record.spec_sha256,
        "--fps",
        str(spec.fps),
        "--device",
        spec.device,
        "--batch-size",
        str(spec.batch_size),
    ]
    for option, name in (
        ("--capture-setup", "capture_setup"),
        ("--calibration-root", "calibration_root"),
        ("--config", "config"),
        ("--detector-training-run", "detector_training_run"),
        ("--identity-training-run", "identity_training_run"),
        ("--dinov2-assets", "dinov2_assets"),
    ):
        arguments += [
            option,
            str(tmp_path / "substituted-config")
            if damage == "config" and name == "config"
            else spec.paths[name],
        ]
    result = worker.main(arguments)
    if damage is None:
        assert result == 0
        assert json.loads(capsys.readouterr().out) == []
    else:
        assert result == 2
