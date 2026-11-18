"""Approval-gated RF-DETR adapter with local weight integrity checks."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml


class RfDetrBackend(Protocol):
    def predict(self, image: object) -> Sequence[Mapping[str, Any]]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_by_id(manifest: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    assets = [asset for asset in manifest.get("assets", []) if asset.get("id") == asset_id]
    if len(assets) != 1:
        raise ValueError(f"asset manifest must contain exactly one {asset_id}")
    return assets[0]


def _require_commercial_approval(asset: Mapping[str, Any]) -> None:
    if asset.get("approval_status") != "approved":
        raise PermissionError(f"asset {asset.get('id')} is not approved")
    if asset.get("commercial_status") != "allowed":
        raise PermissionError(f"asset {asset.get('id')} is not commercially allowed")
    if asset.get("license") != "Apache-2.0":
        raise PermissionError(f"asset {asset.get('id')} is not Apache-2.0")
    digest = asset.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PermissionError(f"asset {asset.get('id')} lacks an approved sha256")


def _verify_file(asset: Mapping[str, Any], path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != asset.get("expected_bytes"):
        raise ValueError(f"RF-DETR {label} byte size mismatch")
    actual_digest = _sha256(path)
    if actual_digest != asset["sha256"]:
        raise ValueError(f"RF-DETR {label} sha256 mismatch")
    return actual_digest


class RfDetrAdapter:
    """Loads the heavy backend only after governance and file checks pass."""

    def __init__(self, backend: RfDetrBackend, *, weight_sha256: str) -> None:
        self._backend = backend
        self.weight_sha256 = weight_sha256

    @classmethod
    def from_approved_assets(
        cls,
        *,
        asset_manifest: Path,
        source_asset_id: str,
        source_path: Path,
        weight_asset_id: str,
        weight_path: Path,
        backend_factory: Callable[[], RfDetrBackend],
    ) -> RfDetrAdapter:
        manifest = yaml.safe_load(asset_manifest.read_text(encoding="utf-8"))
        source = _asset_by_id(manifest, source_asset_id)
        weight = _asset_by_id(manifest, weight_asset_id)
        _require_commercial_approval(source)
        _require_commercial_approval(weight)
        _verify_file(source, source_path, "package")
        weight_digest = _verify_file(weight, weight_path, "weight")
        return cls(backend_factory(), weight_sha256=weight_digest)

    def detect(self, frames: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        observations: list[dict[str, Any]] = []
        for frame in frames:
            for detection in self._backend.predict(frame["image"]):
                observations.append(
                    {
                        "frame_id": frame["frame_id"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "mask": detection["mask"],
                        "center_px": list(detection["center_px"]),
                        "confidence": detection["confidence"],
                        "class_id": detection["class_id"],
                        "supported": True,
                        "weight_sha256": self.weight_sha256,
                    }
                )
        return tuple(observations)
