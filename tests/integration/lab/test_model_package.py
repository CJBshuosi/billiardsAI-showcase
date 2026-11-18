from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.package import (
    ReleasePackagingError,
    claim_locked_test_opening,
    package_release,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release_fixture(root: Path) -> tuple[Path, Path, Path, dict]:
    source = root / "source"
    source.mkdir()
    files = {
        "detector_checkpoint": ("detector.pth", "detector-segmenter/best.pth"),
        "detector_onnx": ("detector.onnx", "detector-segmenter/model.onnx"),
        "identity_checkpoint": ("identity.pth", "identity/best.pth"),
        "identity_onnx": ("identity.onnx", "identity/ball-identity.onnx"),
        "finalist_config": ("finalist.yaml", "configs/finalist.yaml"),
        "tracker_config": ("tracker.yaml", "tracker/tracker.yaml"),
        "event_config": ("events.yaml", "events/event-rules.yaml"),
        "model_card": ("model-card.md", "model-card.md"),
        "benchmark_report": ("benchmark-report.html", "benchmark-report.html"),
    }
    artifacts = []
    for role, (source_name, package_path) in files.items():
        path = source / source_name
        path.write_text(f"fixture:{role}\n", encoding="utf-8")
        artifacts.append(
            {
                "role": role,
                "source_path": source_name,
                "package_path": package_path,
                "sha256": _sha256(path),
            }
        )
    claim = source / "locked-test-claim.json"
    claim.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generation_id": "brain-generation-0001",
                "dataset_revision": "benchmark-v1-r1",
                "authorization_id": "boyu-approval-0001",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(
        {
            "role": "locked_test_claim",
            "source_path": claim.name,
            "package_path": "locked-test/claim.json",
            "sha256": _sha256(claim),
        }
    )
    metrics = source / "locked-test-metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generation_id": "brain-generation-0001",
                "dataset_revision": "benchmark-v1-r1",
                "result": "PASS",
                "all_hard_gates_passed": True,
                "thresholds_sha256": "4" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(
        {
            "role": "locked_test_metrics",
            "source_path": metrics.name,
            "package_path": "locked-test/metrics.json",
            "sha256": _sha256(metrics),
        }
    )
    notice = source / "apache-2.0.txt"
    notice.write_text("Apache License 2.0 fixture\n", encoding="utf-8")
    payload = {
        "schema_version": "1.0",
        "release_id": "billiards-brain-v1",
        "generation_id": "brain-generation-0001",
        "generation_status": "frozen_before_locked_test",
        "result": "PASS",
        "dataset_revision": "benchmark-v1-r1",
        "dataset_manifest_sha256": "1" * 64,
        "training_seeds": [7, 17, 29],
        "code_commit": "a" * 40,
        "dependency_lock_sha256": "2" * 64,
        "class_map": {"cue": 0, "object_ball": 1},
        "calibration_assumptions": ["table coordinates are millimetres"],
        "supported_events": ["first_contact", "cushion", "pot", "cue_ball_scratch"],
        "unsupported_events": ["all_other_foul_rules"],
        "runtime_requirements": ["Python 3.13", "ONNX Runtime CPU or CUDA"],
        "asset_ids": ["rfdetr-package-1.9.0"],
        "artifacts": artifacts,
        "license_notices": [
            {
                "source_path": notice.name,
                "package_path": "LICENSES/apache-2.0.txt",
                "sha256": _sha256(notice),
            }
        ],
        "locked_test": {
            "generation_id": "brain-generation-0001",
            "dataset_revision": "benchmark-v1-r1",
            "open_count": 1,
            "all_hard_gates_passed": True,
            "metrics_sha256": _sha256(metrics),
            "thresholds_sha256": "4" * 64,
            "authorization_id": "boyu-approval-0001",
            "claim_sha256": _sha256(claim),
        },
    }
    config = root / "release.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    assets = root / "assets.yaml"
    assets.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "assets": [
                    {
                        "id": "rfdetr-package-1.9.0",
                        "version": "1.9.0@fixture",
                        "license": "Apache-2.0",
                        "sha256": "5" * 64,
                        "commercial_status": "allowed",
                        "approval_status": "approved",
                    },
                    {
                        "id": "excluded-candidate",
                        "sha256": "",
                        "commercial_status": "excluded",
                        "approval_status": "rejected",
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source, config, assets, payload


@pytest.mark.parametrize(
    "missing_field",
    [
        "license_notices",
        "class_map",
        "calibration_assumptions",
        "supported_events",
        "unsupported_events",
        "dataset_revision",
        "training_seeds",
        "runtime_requirements",
        "locked_test",
    ],
)
def test_package_fails_closed_when_release_evidence_is_missing(
    tmp_path: Path, missing_field: str
) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    payload.pop(missing_field)
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="release configuration is incomplete"):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


def test_package_rejects_missing_model_hash(tmp_path: Path) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    payload["artifacts"][0].pop("sha256")
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="release configuration is incomplete"):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


@pytest.mark.parametrize(
    ("locked_update", "message"),
    [
        ({"open_count": 2}, "exactly once"),
        ({"all_hard_gates_passed": False}, "hard gates"),
        ({"generation_id": "other-generation"}, "generation"),
    ],
)
def test_package_rejects_invalid_locked_test_evidence(
    tmp_path: Path, locked_update: dict, message: str
) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    payload["locked_test"].update(locked_update)
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match=message):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


def test_locked_test_opening_can_be_claimed_exactly_once(tmp_path: Path) -> None:
    ledger = tmp_path / "locked-test-ledger"

    claim = claim_locked_test_opening(
        ledger_root=ledger,
        generation_id="brain-generation-0001",
        dataset_revision="benchmark-v1-r1",
        authorization_id="boyu-approval-0001",
    )

    assert claim.is_file()
    with pytest.raises(ReleasePackagingError, match="already been opened"):
        claim_locked_test_opening(
            ledger_root=ledger,
            generation_id="brain-generation-0001",
            dataset_revision="benchmark-v1-r1",
            authorization_id="boyu-approval-0002",
        )


def test_package_rejects_claim_that_does_not_match_locked_test(tmp_path: Path) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    claim = source / "locked-test-claim.json"
    claim.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generation_id": "different-generation",
                "dataset_revision": "benchmark-v1-r1",
                "authorization_id": "boyu-approval-0001",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    claim_hash = _sha256(claim)
    next(artifact for artifact in payload["artifacts"] if artifact["role"] == "locked_test_claim")[
        "sha256"
    ] = claim_hash
    payload["locked_test"]["claim_sha256"] = claim_hash
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="claim generation"):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


def test_package_checks_only_referenced_assets_and_copies_verified_artifacts(
    tmp_path: Path,
) -> None:
    source, config, assets, _ = _write_release_fixture(tmp_path)

    packaged = package_release(
        source_root=source,
        output_root=tmp_path / "models",
        release_config=config,
        asset_manifest=assets,
    )

    assert packaged == tmp_path / "models" / "billiards-brain-v1"
    assert (packaged / "detector-segmenter/model.onnx").read_text() == "fixture:detector_onnx\n"
    assert (packaged / "LICENSES/apache-2.0.txt").is_file()
    manifest = json.loads((packaged / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_id"] == "billiards-brain-v1"
    assert manifest["locked_test"]["open_count"] == 1


def test_package_rejects_unapproved_referenced_asset(tmp_path: Path) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    payload["asset_ids"] = ["excluded-candidate"]
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="not approved for commercial packaging"):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


@pytest.mark.parametrize("missing_metadata", ["version", "license"])
def test_package_rejects_referenced_asset_without_commercial_metadata(
    tmp_path: Path, missing_metadata: str
) -> None:
    source, config, assets, _ = _write_release_fixture(tmp_path)
    asset_payload = yaml.safe_load(assets.read_text(encoding="utf-8"))
    asset_payload["assets"][0].pop(missing_metadata)
    assets.write_text(yaml.safe_dump(asset_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="not approved for commercial packaging"):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


def test_package_rejects_locked_metrics_that_do_not_match_generation(tmp_path: Path) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    metrics = source / "locked-test-metrics.json"
    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    metrics_payload["generation_id"] = "different-generation"
    metrics.write_text(
        json.dumps(metrics_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    metrics_hash = _sha256(metrics)
    next(
        artifact for artifact in payload["artifacts"] if artifact["role"] == "locked_test_metrics"
    )["sha256"] = metrics_hash
    payload["locked_test"]["metrics_sha256"] = metrics_hash
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="metrics generation"):
        package_release(
            source_root=source,
            output_root=tmp_path / "models",
            release_config=config,
            asset_manifest=assets,
        )


def test_package_rejects_artifact_hash_mismatch_and_existing_release(tmp_path: Path) -> None:
    source, config, assets, payload = _write_release_fixture(tmp_path)
    output = tmp_path / "models"
    payload_with_bad_hash = deepcopy(payload)
    payload_with_bad_hash["artifacts"][0]["sha256"] = "f" * 64
    config.write_text(yaml.safe_dump(payload_with_bad_hash, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReleasePackagingError, match="hash mismatch"):
        package_release(
            source_root=source,
            output_root=output,
            release_config=config,
            asset_manifest=assets,
        )

    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    package_release(
        source_root=source,
        output_root=output,
        release_config=config,
        asset_manifest=assets,
    )
    with pytest.raises(ReleasePackagingError, match="already exists"):
        package_release(
            source_root=source,
            output_root=output,
            release_config=config,
            asset_manifest=assets,
        )
