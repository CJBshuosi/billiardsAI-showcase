"""Fail-closed freezing and loading for algorithm datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from billiards_ai.lab.contracts import DatasetManifest, DatasetPurpose, VideoAsset


class DatasetIntegrityError(RuntimeError):
    """A frozen dataset is missing, changed, unauthorized, or inconsistent."""


class GateUseAuthorizer(Protocol):
    def allows(self, asset: VideoAsset, purpose: DatasetPurpose) -> bool:
        """Return whether the current gate permits this exact asset and purpose."""


def _canonical_bytes(manifest: DatasetManifest) -> bytes:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{payload}\n".encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(f"{manifest_path.suffix}.sha256")


def _verify_asset_files(manifest: DatasetManifest, manifest_path: Path) -> None:
    root = manifest_path.parent.resolve()
    for asset in manifest.assets:
        asset_path = (root / asset.relative_path).resolve()
        if not asset_path.is_relative_to(root):
            raise DatasetIntegrityError(f"asset path escapes dataset root: {asset.asset_id}")
        if not asset_path.is_file():
            raise DatasetIntegrityError(f"asset is missing: {asset.asset_id}")
        if _sha256_file(asset_path) != asset.sha256:
            raise DatasetIntegrityError(f"asset hash mismatch: {asset.asset_id}")


def freeze_dataset(source: Path, output: Path) -> DatasetManifest:
    """Validate a draft manifest and write an immutable canonical revision."""

    if output.exists() or _sidecar_path(output).exists():
        raise DatasetIntegrityError("frozen dataset output already exists")
    try:
        draft = DatasetManifest.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise DatasetIntegrityError("draft dataset manifest is invalid") from exc
    frozen = draft.model_copy(update={"locked": True})
    _verify_asset_files(frozen, output)

    payload = _canonical_bytes(frozen)
    digest = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    sidecar = _sidecar_path(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    output.chmod(0o444)
    sidecar.chmod(0o444)
    return frozen


def load_frozen_dataset(
    path: Path,
    *,
    required_use: DatasetPurpose | None = None,
    authorizer: GateUseAuthorizer | None = None,
) -> DatasetManifest:
    """Load only when manifest, media, grouping, and purpose checks all pass."""

    sidecar = _sidecar_path(path)
    try:
        payload = path.read_bytes()
        expected_hash, expected_name = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError("frozen manifest or checksum sidecar is unavailable") from exc
    if expected_name != path.name or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise DatasetIntegrityError("frozen manifest hash mismatch")
    try:
        manifest = DatasetManifest.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise DatasetIntegrityError("frozen manifest contract is invalid") from exc
    if not manifest.locked:
        raise DatasetIntegrityError("dataset manifest is not locked")
    _verify_asset_files(manifest, path)

    if required_use is not None:
        for asset in manifest.assets:
            if required_use not in asset.approved_uses:
                raise DatasetIntegrityError(f"purpose is not authorized: {asset.asset_id}")
            if asset.is_real_person:
                if authorizer is None or not authorizer.allows(asset, required_use):
                    raise DatasetIntegrityError(f"purpose is not authorized: {asset.asset_id}")
    return manifest
