"""Atomic, content-addressed artifacts for propagation-window results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from billiards_ai.lab.propagation.contracts import PropagationResult


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(_canonical_bytes(row) + b"\n")


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def write_propagation_run(
    *,
    root: Path,
    result: PropagationResult,
    manifest: dict[str, object],
) -> Path:
    result_payload = result.model_dump(mode="json")
    identity_payload = {"manifest": manifest, "result": result_payload}
    content_hash = hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()
    run_directory = root / content_hash
    if run_directory.exists():
        return run_directory

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        stored_manifest = dict(manifest)
        stored_manifest.update(
            {
                "backend": {
                    "id": result.backend_id,
                    "version": result.backend_version,
                },
                "content_hash": content_hash,
                "window": result.window.model_dump(mode="json"),
            }
        )
        _write_json(temporary / "manifest.json", stored_manifest)
        prompts = sorted(
            (item.model_dump(mode="json") for item in result.prompts),
            key=lambda item: (item["frame_index"], item["instance_id"], item["source_id"]),
        )
        observations = sorted(
            (item.model_dump(mode="json") for item in result.observations),
            key=lambda item: (item["frame_index"], item["instance_id"]),
        )
        failures = sorted(
            (item.model_dump(mode="json") for item in result.failures),
            key=lambda item: (
                -1 if item["frame_index"] is None else item["frame_index"],
                item["code"],
            ),
        )
        _write_jsonl(temporary / "prompts.jsonl", prompts)
        _write_jsonl(temporary / "observations.jsonl", observations)
        _write_jsonl(temporary / "failures.jsonl", failures)
        _make_immutable(temporary)
        os.replace(temporary, run_directory)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            for path in temporary.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
            shutil.rmtree(temporary)
    return run_directory
