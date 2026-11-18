from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path("scripts/algorithm/run_reviewed_expansion_frozen_benchmark.py")
WINDOWS_SCRIPT = Path("scripts/algorithm/windows/run-reviewed-expansion-frozen-benchmark.ps1")


def _module():
    assert SCRIPT.is_file(), "reviewed expansion benchmark launcher is missing"
    spec = importlib.util.spec_from_file_location("run_reviewed_expansion_frozen_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_discovery_binds_exact_dataset_training_and_prior_frozen_population(
    tmp_path: Path,
) -> None:
    module = _module()
    project = tmp_path / "project"
    artifacts = tmp_path / "artifacts"
    documents = tmp_path / "Documents"
    dataset_id = "a" * 64
    bundle_id = "b" * 64
    evaluation_id = "c" * 64
    run_hash = "d" * 64

    weight = project / "rf-detr-seg-s-ft.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"base-weight")
    dataset = artifacts / "bootstrap-datasets" / dataset_id
    _write_json(dataset / "manifest.json", {"artifact_id": dataset_id})
    training_run = (
        artifacts
        / "bootstrap-training-runs"
        / dataset_id
        / "rfdetr-seg-small-bootstrap-short-r1"
        / run_hash
    )
    checkpoint = training_run / "checkpoint_best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"expanded-checkpoint")
    _write_json(
        training_run / "training-manifest.json",
        {
            "candidate_id": "rfdetr-seg-small-bootstrap-short-r1",
            "dataset_id": dataset_id,
            "run_hash": run_hash,
            "artifacts": {"checkpoint_best.pth": _sha256(checkpoint)},
        },
    )
    decoy = (
        artifacts
        / "bootstrap-training-runs"
        / ("e" * 64)
        / "rfdetr-seg-small-bootstrap-short-r1"
        / ("f" * 64)
    )
    _write_json(
        decoy / "training-manifest.json",
        {"dataset_id": "e" * 64, "candidate_id": "rfdetr-seg-small-bootstrap-short-r1"},
    )

    _write_json(
        artifacts / "frozen-detector-benchmarks" / ("1" * 64) / "report.json",
        {
            "identity": {
                "kind": "rfdetr_frozen_three_way_benchmark",
                "gates": {"passed": True},
                "provenance": {
                    "input_bundle_id": bundle_id,
                    "evaluation_manifest_id": evaluation_id,
                },
            }
        },
    )
    bundle = documents / "billiardsAI-git-sam2-pilot-input" / bundle_id
    _write_json(bundle / "inventory.json", {"bundle_id": bundle_id})
    control_root = artifacts / "sam2-real-video-pilot" / "fixed-control"
    evaluation = control_root / "evaluation-manifests" / evaluation_id / "evaluation-manifest.json"
    _write_json(evaluation, {"manifest_id": evaluation_id})
    materialized = control_root / "imported-inputs" / bundle_id / "annotation-export"
    _write_json(materialized / "annotations.json", {})
    _write_json(materialized / "summary.json", {})
    stale_duplicate = (
        artifacts
        / "stale-pilot"
        / "fixed-control"
        / "evaluation-manifests"
        / evaluation_id
        / "evaluation-manifest.json"
    )
    _write_json(stale_duplicate, {"manifest_id": evaluation_id})
    duplicate_materialized = (
        stale_duplicate.parents[2] / "imported-inputs" / bundle_id / "annotation-export"
    )
    _write_json(duplicate_materialized / "annotations.json", {})
    _write_json(duplicate_materialized / "summary.json", {})

    discovered = module.discover_benchmark_inputs(
        project_root=project,
        artifact_root=artifacts,
        documents_root=documents,
        dataset_id=dataset_id,
    )

    assert discovered.input_bundle == bundle
    assert discovered.evaluation_manifest == evaluation
    assert discovered.bootstrap_dataset == dataset
    assert discovered.training_run == training_run
    assert discovered.base_weight == weight
    assert discovered.training_manifest_sha256 == _sha256(training_run / "training-manifest.json")
    assert discovered.trained_checkpoint_sha256 == _sha256(checkpoint)
    assert discovered.base_weight_sha256 == _sha256(weight)


def test_discovery_rejects_ambiguous_frozen_population(tmp_path: Path) -> None:
    module = _module()
    artifacts = tmp_path / "artifacts"
    for index, bundle_id in enumerate(("b" * 64, "c" * 64)):
        _write_json(
            artifacts / "frozen-detector-benchmarks" / (str(index + 1) * 64) / "report.json",
            {
                "identity": {
                    "kind": "rfdetr_frozen_three_way_benchmark",
                    "gates": {"passed": True},
                    "provenance": {
                        "input_bundle_id": bundle_id,
                        "evaluation_manifest_id": "d" * 64,
                    },
                }
            },
        )

    try:
        module.discover_benchmark_inputs(
            project_root=tmp_path / "project",
            artifact_root=artifacts,
            documents_root=tmp_path / "Documents",
            dataset_id="a" * 64,
        )
    except ValueError as exc:
        assert str(exc) == "prior frozen benchmark population is ambiguous"
    else:
        raise AssertionError("ambiguous frozen population was accepted")


def test_windows_launcher_requires_explicit_dataset_id_and_forwards_fixed_roots() -> None:
    assert WINDOWS_SCRIPT.is_file(), "Windows reviewed expansion benchmark launcher is missing"
    source = WINDOWS_SCRIPT.read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]\n    [string]$DatasetId" in source
    assert "run_reviewed_expansion_frozen_benchmark.py" in source
    assert '"--dataset-id", $DatasetId' in source
    assert '"--artifact-root", $resolvedArtifacts' in source
    assert '"--documents-root", $resolvedDocuments' in source
    assert "Sort-Object LastWriteTime" not in source
