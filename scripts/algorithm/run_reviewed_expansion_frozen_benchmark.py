#!/usr/bin/env python3
"""Discover and run the frozen benchmark for one reviewed expansion dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_CANDIDATE_ID = "rfdetr-seg-small-bootstrap-short-r1"


@dataclass(frozen=True)
class DiscoveredBenchmarkInputs:
    input_bundle: Path
    evaluation_manifest: Path
    bootstrap_dataset: Path
    training_run: Path
    base_weight: Path
    training_manifest_sha256: str
    trained_checkpoint_sha256: str
    base_weight_sha256: str


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_id(value: object) -> str | None:
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return None


def _frozen_population_ids(artifact_root: Path) -> tuple[str, str]:
    pairs: set[tuple[str, str]] = set()
    reports_root = artifact_root / "frozen-detector-benchmarks"
    for report_path in reports_root.glob("*/report.json"):
        try:
            report = _json_object(report_path, label="prior frozen benchmark report")
            identity = report["identity"]
            if not isinstance(identity, dict):
                continue
            provenance = identity["provenance"]
            gates = identity["gates"]
            if (
                identity.get("kind") != "rfdetr_frozen_three_way_benchmark"
                or not isinstance(provenance, dict)
                or not isinstance(gates, dict)
                or gates.get("passed") is not True
            ):
                continue
            bundle_id = _sha256_id(provenance.get("input_bundle_id"))
            evaluation_id = _sha256_id(provenance.get("evaluation_manifest_id"))
            if bundle_id is not None and evaluation_id is not None:
                pairs.add((bundle_id, evaluation_id))
        except (KeyError, ValueError):
            continue
    if not pairs:
        raise ValueError("prior passed frozen benchmark population was not found")
    if len(pairs) != 1:
        raise ValueError("prior frozen benchmark population is ambiguous")
    return next(iter(pairs))


def _only(paths: list[Path], *, missing: str, ambiguous: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError(missing)
    if len(unique) != 1:
        raise ValueError(ambiguous)
    return unique[0]


def _one_identity_copy(paths: list[Path], *, missing: str, ambiguous: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError(missing)
    identities = [_json_object(path, label="frozen evaluation manifest") for path in unique]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(ambiguous)
    return unique[0]


def discover_benchmark_inputs(
    *,
    project_root: Path,
    artifact_root: Path,
    documents_root: Path,
    dataset_id: str,
) -> DiscoveredBenchmarkInputs:
    """Bind one expanded dataset to its exact training and prior frozen population."""

    if _sha256_id(dataset_id) is None:
        raise ValueError("reviewed expansion dataset id is invalid")
    bundle_id, evaluation_id = _frozen_population_ids(artifact_root)

    bundle_candidates = []
    direct_bundle = documents_root / bundle_id
    if (direct_bundle / "inventory.json").is_file():
        bundle_candidates.append(direct_bundle)
    for parent in documents_root.glob("billiardsAI*"):
        candidate = parent / bundle_id
        if parent.is_dir() and (candidate / "inventory.json").is_file():
            bundle_candidates.append(candidate)
    input_bundle = _only(
        bundle_candidates,
        missing="frozen input bundle was not found",
        ambiguous="frozen input bundle is ambiguous",
    )
    inventory = _json_object(input_bundle / "inventory.json", label="frozen input bundle")
    inventory_id = inventory.get("bundle_id")
    if inventory_id is not None and inventory_id != bundle_id:
        raise ValueError("frozen input bundle identity changed")

    evaluation_candidates = []
    for path in artifact_root.rglob("evaluation-manifest.json"):
        if path.parent.name != evaluation_id or path.parent.parent.name != "evaluation-manifests":
            continue
        try:
            control_root = path.parents[2]
            materialized = control_root / "imported-inputs" / bundle_id / "annotation-export"
            if (
                _json_object(path, label="frozen evaluation manifest").get("manifest_id")
                == evaluation_id
                and (materialized / "annotations.json").is_file()
                and (materialized / "summary.json").is_file()
            ):
                evaluation_candidates.append(path)
        except (IndexError, ValueError):
            continue
    evaluation_manifest = _one_identity_copy(
        evaluation_candidates,
        missing="frozen evaluation manifest was not found",
        ambiguous="frozen evaluation manifest is ambiguous",
    )

    bootstrap_dataset = artifact_root / "bootstrap-datasets" / dataset_id
    dataset_manifest = _json_object(
        bootstrap_dataset / "manifest.json", label="reviewed expansion dataset"
    )
    if dataset_manifest.get("artifact_id") != dataset_id:
        raise ValueError("reviewed expansion dataset identity changed")

    candidate_root = artifact_root / "bootstrap-training-runs" / dataset_id / _CANDIDATE_ID
    training_candidates = []
    for manifest_path in candidate_root.glob("*/training-manifest.json"):
        try:
            manifest = _json_object(manifest_path, label="reviewed expansion training manifest")
            checkpoint = manifest_path.parent / "checkpoint_best.pth"
            artifacts = manifest.get("artifacts")
            if (
                manifest.get("dataset_id") == dataset_id
                and manifest.get("candidate_id") == _CANDIDATE_ID
                and manifest.get("run_hash") == manifest_path.parent.name
                and isinstance(artifacts, dict)
                and checkpoint.is_file()
                and artifacts.get("checkpoint_best.pth") == _sha256(checkpoint)
            ):
                training_candidates.append(manifest_path.parent)
        except (OSError, ValueError):
            continue
    training_run = _only(
        training_candidates,
        missing="matching reviewed expansion training run was not found",
        ambiguous="matching reviewed expansion training run is ambiguous",
    )

    base_weight = project_root / "rf-detr-seg-s-ft.pth"
    if not base_weight.is_file():
        raise ValueError("approved RF-DETR base weight was not found")
    training_manifest = training_run / "training-manifest.json"
    checkpoint = training_run / "checkpoint_best.pth"
    return DiscoveredBenchmarkInputs(
        input_bundle=input_bundle,
        evaluation_manifest=evaluation_manifest,
        bootstrap_dataset=bootstrap_dataset.resolve(),
        training_run=training_run,
        base_weight=base_weight.resolve(),
        training_manifest_sha256=_sha256(training_manifest),
        trained_checkpoint_sha256=_sha256(checkpoint),
        base_weight_sha256=_sha256(base_weight),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--documents-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        discovered = discover_benchmark_inputs(
            project_root=arguments.project_root,
            artifact_root=arguments.artifact_root,
            documents_root=arguments.documents_root,
            dataset_id=arguments.dataset_id,
        )
        from rfdetr import RFDETRSegSmall, from_checkpoint

        from billiards_ai.lab.adapters.classic_cv import ClassicDetectorBackend
        from billiards_ai.lab.training.rfdetr_frozen_benchmark import (
            FrozenBenchmarkInputs,
            run_frozen_benchmark,
        )

        run = run_frozen_benchmark(
            FrozenBenchmarkInputs(
                input_bundle=discovered.input_bundle,
                evaluation_manifest=discovered.evaluation_manifest,
                bootstrap_dataset=discovered.bootstrap_dataset,
                training_run=discovered.training_run,
                base_weight=discovered.base_weight,
                expected_base_weight_sha256=discovered.base_weight_sha256,
                expected_training_manifest_sha256=discovered.training_manifest_sha256,
                expected_trained_checkpoint_sha256=discovered.trained_checkpoint_sha256,
                output_root=arguments.artifact_root,
                trained_model_factory=lambda path: from_checkpoint(str(path)),
                coco_model_factory=lambda path: RFDETRSegSmall(pretrain_weights=str(path)),
                classic_backend_factory=ClassicDetectorBackend,
            )
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"reviewed expansion frozen benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": run.artifact_id,
                "exit_code": 0,
                "output": str(run.path),
                "status": "PASS",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
