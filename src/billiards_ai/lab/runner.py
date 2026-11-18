"""Atomic, content-addressed artifact runs for local algorithm experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from billiards_ai.lab.adapters.base import Renderer
from billiards_ai.lab.render import build_overlay_frames
from billiards_ai.lab.report import build_report_html


class RunInputs(BaseModel):
    """Normalized inputs owned by the workbench, never by candidate adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    manifest: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    failure_records: tuple[Any, ...]
    evidence_frames: dict[int, dict[str, Any]]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def create_artifact_run(root: Path, inputs: RunInputs, renderer: Renderer) -> Path:
    """Create one immutable run without exposing partial or overwritten artifacts."""
    payload = inputs.model_dump(mode="json")
    payload["renderer"] = {"name": renderer.name, "version": renderer.version}
    run_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    candidate_root = root / inputs.dataset_revision / inputs.candidate_id
    run_dir = candidate_root / run_hash
    if run_dir.exists():
        return run_dir

    candidate_root.mkdir(parents=True, exist_ok=True)
    temporary = candidate_root / f".tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        overlay_frames = build_overlay_frames(inputs.predictions, inputs.evidence_frames)
        manifest = dict(inputs.manifest)
        manifest.update(
            {
                "dataset_revision": inputs.dataset_revision,
                "candidate_id": inputs.candidate_id,
                "run_hash": run_hash,
                "renderer": payload["renderer"],
            }
        )
        _write_json(temporary / "manifest.json", manifest)
        with (temporary / "predictions.jsonl").open("wb") as predictions_file:
            for prediction in inputs.predictions:
                predictions_file.write(_canonical_bytes(prediction) + b"\n")
        _write_json(temporary / "metrics.json", inputs.metrics)
        (temporary / "report.html").write_text(
            build_report_html(inputs.predictions, inputs.metrics), encoding="utf-8"
        )
        failures_dir = temporary / "failures"
        failures_dir.mkdir()
        for index, failure in enumerate(inputs.failure_records):
            _write_json(failures_dir / f"failure-{index:05d}.json", failure)
        renderer.render(
            output_path=temporary / "overlay.mp4",
            overlay_frames=overlay_frames,
        )
        _make_immutable(temporary)
        os.replace(temporary, run_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return run_dir
