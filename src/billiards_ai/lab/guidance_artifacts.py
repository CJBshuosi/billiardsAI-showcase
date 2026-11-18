"""Content-addressed, immutable local guidance-run publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _safe_output_name(name: str) -> None:
    path = Path(name)
    if not name or path.name != name or path.is_absolute() or name in {"manifest.json", ".", ".."}:
        raise ValueError("artifact output requires a safe filename")


def publish_guidance_run(
    output_root: Path,
    *,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object | bytes],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".guidance-", dir=output_root))
    try:
        inventory = []
        for name, value in sorted(outputs.items()):
            _safe_output_name(name)
            payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
            (staging / name).write_bytes(payload)
            inventory.append(
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        identity = {"schema_version": "1.0", "inputs": dict(inputs), "inventory": inventory}
        artifact_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        manifest = {**identity, "artifact_id": artifact_id}
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        destination = output_root / artifact_id
        if destination.exists():
            existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            if existing != manifest:
                raise RuntimeError("content-addressed guidance artifact identity collision")
            return destination
        os.replace(staging, destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
