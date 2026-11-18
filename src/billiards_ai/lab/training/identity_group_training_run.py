"""Artifact-first development short training for the group-only identity head."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from billiards_ai.lab.training.identity_review import load_reviewed_identity_dataset
from billiards_ai.lab.training.identity_train import evaluate_identity_gates, verify_export_parity
from billiards_ai.lab.training.identity_training_run import _load_config


class GroupShortTrainingBackend(Protocol):
    name: str
    version: str

    def train(self, **kwargs: object) -> None: ...

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None: ...

    def parity_logits(self, **kwargs: object) -> tuple[object, object]: ...


@dataclass(frozen=True)
class GroupShortTrainingRun:
    output: Path
    run_id: str
    status: str
    gate_status: str


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


def _make_immutable(root: Path) -> None:
    if os.name == "nt":
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _metrics(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("group training metrics are invalid") from exc
    required = {
        "trained_heads",
        "exact_head",
        "group_macro_f1",
        "cue_f1",
        "eight_f1",
        "solid_f1",
        "stripe_f1",
        "forced_exact_without_evidence",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("group training metrics are invalid")
    if payload["trained_heads"] != ["group"] or payload["exact_head"] != {
        "status": "deferred",
        "trained": False,
    }:
        raise ValueError("group training unexpectedly produced an exact head")
    return payload


def run_group_short_training(
    *,
    candidate_artifact: Path,
    verdict_artifact: Path,
    correction_verdict_artifact: Path | None = None,
    config_path: Path,
    output_root: Path,
    backend: GroupShortTrainingBackend,
) -> GroupShortTrainingRun:
    """Run local group training and retain diagnostics even when frozen gates fail."""

    train = load_reviewed_identity_dataset(
        candidate_artifact=candidate_artifact,
        verdict_artifact=verdict_artifact,
        correction_verdict_artifact=correction_verdict_artifact,
        split="train",
    )
    valid = load_reviewed_identity_dataset(
        candidate_artifact=candidate_artifact,
        verdict_artifact=verdict_artifact,
        correction_verdict_artifact=correction_verdict_artifact,
        split="valid",
    )
    config, config_hash, group_class_map, _ = _load_config(config_path)
    identity = {
        "schema_version": "1.0",
        "kind": "ball_group_development_short_training",
        "candidate_artifact_id": candidate_artifact.resolve().name,
        "verdict_artifact_id": verdict_artifact.resolve().name,
        "correction_verdict_artifact_id": (
            correction_verdict_artifact.resolve().name
            if correction_verdict_artifact is not None
            else None
        ),
        "config_hash": config_hash,
        "backend": {"name": backend.name, "version": backend.version},
        "backend_configuration": getattr(backend, "configuration", None),
        "feature_head": "group",
        "exact_head": {"status": "deferred", "trained": False},
        "group_class_map": list(group_class_map),
        "seed": config.get("seed"),
        "scientific_use": "development_short_train_not_locked_accuracy",
    }
    run_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / run_id
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        return GroupShortTrainingRun(
            output=output,
            run_id=run_id,
            status="PASS",
            gate_status=str(manifest["gate_status"]),
        )

    temporary = output_root / f".pending-{uuid4().hex}"
    try:
        temporary.mkdir()
        checkpoint = temporary / "group-head.npz"
        metrics_path = temporary / "metrics.json"
        onnx_path = temporary / "group-head.onnx"
        backend.train(
            train_records=tuple(train.records),
            valid_records=tuple(valid.records),
            group_class_map=group_class_map,
            exact_class_map=(),
            checkpoint_path=checkpoint,
            metrics_path=metrics_path,
        )
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise ValueError("group backend did not produce required artifacts")
        metrics = _metrics(metrics_path)
        gate = evaluate_identity_gates(
            group_macro_f1=float(metrics["group_macro_f1"]),
            cue_f1=float(metrics["cue_f1"]),
            eight_f1=float(metrics["eight_f1"]),
            solid_f1=float(metrics["solid_f1"]),
            stripe_f1=float(metrics["stripe_f1"]),
            forced_exact_without_evidence=int(metrics["forced_exact_without_evidence"]),
        )
        backend.export_onnx(checkpoint_path=checkpoint, output_path=onnx_path)
        if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
            raise ValueError("group backend did not export ONNX")
        checkpoint_logits, onnx_logits = backend.parity_logits(
            checkpoint_path=checkpoint, onnx_path=onnx_path
        )
        parity_class_maps = getattr(backend, "parity_class_maps", None)
        if parity_class_maps is None:
            parity_class_map = tuple(getattr(backend, "parity_class_map", group_class_map))
            parity_class_maps = {"group": parity_class_map}
        verify_export_parity(
            pytorch_logits={
                head: tuple(float(value) for value in values)
                for head, values in checkpoint_logits.items()  # type: ignore[union-attr]
            },
            onnx_logits={
                head: tuple(float(value) for value in values)
                for head, values in onnx_logits.items()  # type: ignore[union-attr]
            },
            pytorch_class_maps=parity_class_maps,
            onnx_class_maps=parity_class_maps,
        )
        inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (checkpoint, metrics_path, onnx_path)
        }
        gate_status = "PASS" if gate.passed else "FAIL"
        manifest = {
            "run_id": run_id,
            **identity,
            "status": "PASS",
            "gate_status": gate_status,
            "gate_violations": list(gate.violations),
            "release_eligible": False,
            "trained_heads": ["group"],
            "metrics": metrics,
            "inventory": inventory,
        }
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
        _make_immutable(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return GroupShortTrainingRun(
        output=output, run_id=run_id, status="PASS", gate_status=gate_status
    )
