"""Artifact-first exact ball-identity training contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import yaml

from billiards_ai.lab.training.identity_review import load_reviewed_identity_dataset
from billiards_ai.lab.training.identity_train import evaluate_identity_gates, verify_export_parity


class IdentityTrainingError(RuntimeError):
    """Identity training inputs, outputs, gates, or export parity are invalid."""


class IdentityTrainingBackend(Protocol):
    name: str
    version: str

    def train(
        self,
        *,
        train_records: tuple[object, ...],
        valid_records: tuple[object, ...],
        group_class_map: tuple[str, ...],
        exact_class_map: tuple[str, ...],
        checkpoint_path: Path,
        metrics_path: Path,
    ) -> None: ...

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None: ...

    def parity_logits(
        self, *, checkpoint_path: Path, onnx_path: Path
    ) -> tuple[dict[str, Sequence[float]], dict[str, Sequence[float]]]: ...


@dataclass(frozen=True)
class IdentityTrainingRun:
    output: Path
    run_id: str
    status: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], str, tuple[str, ...], tuple[str, ...]]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise IdentityTrainingError("identity training config is invalid") from exc
    if not isinstance(config, dict) or config.get("schema_version") != "2.0":
        raise IdentityTrainingError("identity training config is invalid")
    classes = config.get("classes")
    groups = config.get("groups")
    expected_exact = ("cue", *(str(number) for number in range(1, 16)), "unknown")
    expected_groups = ("cue", "solid", "eight", "stripe", "unknown")
    if not isinstance(classes, list) or tuple(classes) != expected_exact:
        raise IdentityTrainingError("identity training class map is invalid")
    if not isinstance(groups, list) or tuple(groups) != expected_groups:
        raise IdentityTrainingError("identity training group map is invalid")
    if config.get("export") != {"format": "onnx", "logits_absolute_tolerance": 0.0001}:
        raise IdentityTrainingError("identity export contract is invalid")
    return (
        config,
        hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        expected_groups,
        expected_exact,
    )


def _load_metrics(path: Path) -> dict[str, float | int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IdentityTrainingError("identity training metrics are invalid") from exc
    expected = {
        "group_macro_f1",
        "cue_f1",
        "eight_f1",
        "solid_f1",
        "stripe_f1",
        "forced_exact_without_evidence",
        "exact_labeled_count",
        "exact_macro_f1",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise IdentityTrainingError("identity training metrics are invalid")
    try:
        return {
            "group_macro_f1": float(payload["group_macro_f1"]),
            "cue_f1": float(payload["cue_f1"]),
            "eight_f1": float(payload["eight_f1"]),
            "solid_f1": float(payload["solid_f1"]),
            "stripe_f1": float(payload["stripe_f1"]),
            "forced_exact_without_evidence": int(payload["forced_exact_without_evidence"]),
            "exact_labeled_count": int(payload["exact_labeled_count"]),
            "exact_macro_f1": float(payload["exact_macro_f1"]),
        }
    except (TypeError, ValueError) as exc:
        raise IdentityTrainingError("identity training metrics are invalid") from exc


def _make_immutable(root: Path) -> None:
    if os.name == "nt":
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def run_identity_training(
    *,
    candidate_artifact: Path,
    verdict_artifact: Path,
    config_path: Path,
    output_root: Path,
    backend: IdentityTrainingBackend,
) -> IdentityTrainingRun:
    """Train and publish one identity checkpoint only after frozen gates and ONNX parity pass."""

    train = load_reviewed_identity_dataset(
        candidate_artifact=candidate_artifact,
        verdict_artifact=verdict_artifact,
        split="train",
    )
    valid = load_reviewed_identity_dataset(
        candidate_artifact=candidate_artifact,
        verdict_artifact=verdict_artifact,
        split="valid",
    )
    config, config_hash, group_class_map, exact_class_map = _load_config(config_path)
    identity = {
        "schema_version": "2.0",
        "kind": "ball_identity_training_run",
        "candidate_artifact_id": candidate_artifact.resolve().name,
        "verdict_artifact_id": verdict_artifact.resolve().name,
        "config_hash": config_hash,
        "backend": {"name": backend.name, "version": backend.version},
        "class_maps": {
            "group": list(group_class_map),
            "exact": list(exact_class_map),
        },
        "dataset_labels": {
            "train_exact_labeled_count": train.exact_labeled_count,
            "valid_exact_labeled_count": valid.exact_labeled_count,
        },
        "seed": config.get("seed"),
    }
    run_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / run_id
    if output.exists():
        manifest = output / "manifest.json"
        if not manifest.is_file():
            raise IdentityTrainingError("existing identity training run is invalid")
        return IdentityTrainingRun(output=output, run_id=run_id, status="PASS")

    temporary = output_root / f".pending-{uuid4().hex}"
    try:
        temporary.mkdir()
        checkpoint = temporary / "checkpoint.pt"
        metrics_path = temporary / "metrics.json"
        onnx_path = temporary / "model.onnx"
        backend.train(
            train_records=tuple(train.records),
            valid_records=tuple(valid.records),
            group_class_map=group_class_map,
            exact_class_map=exact_class_map,
            checkpoint_path=checkpoint,
            metrics_path=metrics_path,
        )
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise IdentityTrainingError("identity backend did not produce required artifacts")
        metrics = _load_metrics(metrics_path)
        try:
            gate = evaluate_identity_gates(
                group_macro_f1=float(metrics["group_macro_f1"]),
                cue_f1=float(metrics["cue_f1"]),
                eight_f1=float(metrics["eight_f1"]),
                solid_f1=float(metrics["solid_f1"]),
                stripe_f1=float(metrics["stripe_f1"]),
                forced_exact_without_evidence=int(metrics["forced_exact_without_evidence"]),
            )
        except ValueError as exc:
            raise IdentityTrainingError("identity training metrics are invalid") from exc
        if not gate.passed:
            raise IdentityTrainingError(
                "frozen identity gates failed: " + ", ".join(gate.violations)
            )
        backend.export_onnx(checkpoint_path=checkpoint, output_path=onnx_path)
        if not onnx_path.is_file():
            raise IdentityTrainingError("identity backend did not export ONNX")
        pytorch_raw, onnx_raw = backend.parity_logits(
            checkpoint_path=checkpoint,
            onnx_path=onnx_path,
        )
        try:
            verify_export_parity(
                pytorch_logits={
                    head: tuple(float(value) for value in values)
                    for head, values in pytorch_raw.items()
                },
                onnx_logits={
                    head: tuple(float(value) for value in values)
                    for head, values in onnx_raw.items()
                },
                pytorch_class_maps={
                    "group": group_class_map,
                    "exact": exact_class_map,
                },
                onnx_class_maps={
                    "group": group_class_map,
                    "exact": exact_class_map,
                },
                absolute_tolerance=0.0001,
            )
        except ValueError as exc:
            raise IdentityTrainingError("identity checkpoint/ONNX parity failed") from exc
        inventory = {
            name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in {
                "checkpoint.pt": checkpoint,
                "metrics.json": metrics_path,
                "model.onnx": onnx_path,
            }.items()
        }
        manifest = {
            "run_id": run_id,
            **identity,
            "gate_status": "PASS",
            "metrics": metrics,
            "inventory": inventory,
        }
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
        _make_immutable(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return IdentityTrainingRun(output=output, run_id=run_id, status="PASS")
