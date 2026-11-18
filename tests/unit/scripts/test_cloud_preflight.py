from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.cloud_preflight import create_job_evidence, run_preflight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, dict]:
    derived_manifest = root / "derived-dataset.yaml"
    derived_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "revision": "derived-r1",
                "assets": [
                    {
                        "relative_path": "frames/session-1/frame-0001.jpg",
                        "sha256": "1" * 64,
                        "bytes": 1024,
                        "is_real_person_source": True,
                        "approved_uses": ["cloud_training"],
                        "table_crop_only": True,
                        "audio_removed": True,
                        "contains_user_identity": False,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assets = root / "assets.yaml"
    assets.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "assets": [
                    {
                        "id": "rfdetr-package-1.9.0",
                        "version": "1.9.0@fixture",
                        "sha256": "2" * 64,
                        "license": "Apache-2.0",
                        "commercial_status": "allowed",
                        "approval_status": "approved",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    policy = root / "experiment-policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "cloud_budget_usd": 100,
                "cloud_budget_status": "approved",
                "preferred_region": "eu",
                "automatic_shutdown_minutes": 30,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    train_config = root / "train.yaml"
    train_config.write_text("seed: 20260801\n", encoding="utf-8")
    output_dir = root / "collected"
    payload = {
        "schema_version": "1.0",
        "job_id": "rfdetr-training-proposal-r1",
        "provider": {
            "name": "provider-fixture",
            "terms_url": "https://example.invalid/terms",
            "dpa_url": "https://example.invalid/dpa",
            "approval_status": "approved",
        },
        "region": {
            "id": "EU-FIXTURE-1",
            "jurisdiction": "EU",
            "approval_status": "approved",
        },
        "instance": {
            "gpu": "A6000-48GB",
            "secure_cloud": True,
            "approval_status": "approved",
        },
        "image": {
            "reference": "registry.invalid/billiards-train@sha256:" + "3" * 64,
            "approval_status": "approved",
        },
        "budget": {
            "maximum_usd": 100,
            "estimated_hours": 10,
            "hourly_usd": 2,
        },
        "policy": {
            "path": policy.name,
            "sha256": _sha256(policy),
        },
        "dataset": {
            "manifest_path": derived_manifest.name,
            "manifest_sha256": _sha256(derived_manifest),
            "upload_bytes": 1024,
        },
        "asset_manifest": {
            "path": assets.name,
            "sha256": _sha256(assets),
            "required_asset_ids": ["rfdetr-package-1.9.0"],
        },
        "training_configs": [{"path": train_config.name, "sha256": _sha256(train_config)}],
        "shutdown": {
            "enabled": True,
            "idle_minutes": 30,
            "hard_deadline_minutes": 720,
        },
        "deletion": {
            "approval_status": "approved",
            "delete_instance": True,
            "delete_attached_volume": True,
            "verification_required": True,
        },
        "collection": {
            "local_output_dir": output_dir.name,
            "expected_output_bytes": 2048,
            "free_space_multiplier": 2,
        },
    }
    spec = root / "cloud-job.yaml"
    spec.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return spec, payload


def _rewrite(spec: Path, payload: dict) -> None:
    spec.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_approved_offline_proposal_passes_without_contacting_cloud(tmp_path: Path) -> None:
    spec, _ = _write_fixture(tmp_path)

    result = run_preflight(spec)

    assert result.passed is True
    assert result.blockers == ()
    assert result.upload_bytes == 1024
    assert result.projected_cost_usd == 20


def test_unapproved_budget_is_rejected(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    policy = tmp_path / payload["policy"]["path"]
    policy_payload = yaml.safe_load(policy.read_text(encoding="utf-8"))
    policy_payload["cloud_budget_status"] = "pending"
    policy.write_text(yaml.safe_dump(policy_payload, sort_keys=True), encoding="utf-8")
    payload["policy"]["sha256"] = _sha256(policy)
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert "budget_not_approved" in result.blockers


def test_non_eu_region_is_rejected(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    payload["region"]["jurisdiction"] = "US"
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert "region_not_eu" in result.blockers


def test_real_data_without_cloud_training_purpose_is_rejected(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    manifest = tmp_path / payload["dataset"]["manifest_path"]
    manifest_payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    manifest_payload["assets"][0]["approved_uses"] = ["local_training"]
    manifest.write_text(yaml.safe_dump(manifest_payload, sort_keys=True), encoding="utf-8")
    payload["dataset"]["manifest_sha256"] = _sha256(manifest)
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert "real_asset_missing_cloud_training_approval" in result.blockers


@pytest.mark.parametrize(
    "reference",
    ["registry.invalid/billiards-train:latest", "registry.invalid/billiards-train:1.0"],
)
def test_mutable_container_reference_is_rejected(tmp_path: Path, reference: str) -> None:
    spec, payload = _write_fixture(tmp_path)
    payload["image"]["reference"] = reference
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert "container_image_not_digest_pinned" in result.blockers


def test_absent_shutdown_timer_is_rejected(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    payload["shutdown"]["enabled"] = False
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert "automatic_shutdown_not_enforced" in result.blockers


def test_output_without_local_free_space_headroom_is_rejected(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    payload["collection"]["expected_output_bytes"] = 10**18
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert "insufficient_local_collection_space" in result.blockers


@pytest.mark.parametrize(
    ("field", "unsafe_value", "blocker"),
    [
        ("table_crop_only", False, "dataset_not_table_cropped"),
        ("audio_removed", False, "dataset_audio_not_removed"),
        ("contains_user_identity", True, "dataset_contains_user_identity"),
    ],
)
def test_privacy_unsafe_derived_asset_is_rejected(
    tmp_path: Path, field: str, unsafe_value: bool, blocker: str
) -> None:
    spec, payload = _write_fixture(tmp_path)
    manifest = tmp_path / payload["dataset"]["manifest_path"]
    manifest_payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    manifest_payload["assets"][0][field] = unsafe_value
    manifest.write_text(yaml.safe_dump(manifest_payload, sort_keys=True), encoding="utf-8")
    payload["dataset"]["manifest_sha256"] = _sha256(manifest)
    _rewrite(spec, payload)

    result = run_preflight(spec)

    assert blocker in result.blockers


def test_passing_preflight_creates_content_addressed_immutable_evidence(tmp_path: Path) -> None:
    spec, _ = _write_fixture(tmp_path)
    evidence_root = tmp_path / "evidence"

    first = create_job_evidence(spec, evidence_root)
    second = create_job_evidence(spec, evidence_root)

    assert first == second
    assert len(first.name) == 64
    record = json.loads((first / "preflight.json").read_text(encoding="utf-8"))
    assert record["passed"] is True
    assert record["upload_bytes"] == 1024
    assert not (first.stat().st_mode & stat.S_IWUSR)
    assert not ((first / "cloud-job.yaml").stat().st_mode & stat.S_IWUSR)


def test_failing_preflight_cannot_create_job_evidence(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    payload["region"]["jurisdiction"] = "US"
    _rewrite(spec, payload)

    with pytest.raises(ValueError, match="preflight failed"):
        create_job_evidence(spec, tmp_path / "evidence")

    assert not (tmp_path / "evidence").exists()


def test_existing_job_evidence_is_reverified_before_reuse(tmp_path: Path) -> None:
    spec, _ = _write_fixture(tmp_path)
    evidence = create_job_evidence(spec, tmp_path / "evidence")
    copied_config = evidence / "inputs/training-config-000.yaml"
    copied_config.chmod(0o644)
    copied_config.write_text("seed: 999\n", encoding="utf-8")

    with pytest.raises(ValueError, match="existing job evidence is corrupt"):
        create_job_evidence(spec, tmp_path / "evidence")


def test_cli_exits_nonzero_and_reports_blockers_without_cloud_side_effects(tmp_path: Path) -> None:
    spec, payload = _write_fixture(tmp_path)
    payload["shutdown"]["enabled"] = False
    _rewrite(spec, payload)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/cloud_preflight.py",
            "--spec",
            str(spec),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "automatic_shutdown_not_enforced" in json.loads(completed.stdout)["blockers"]
