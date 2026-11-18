import hashlib
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.adapters.rfdetr import RfDetrAdapter


def _manifest(
    path: Path,
    package: Path,
    weight: Path,
    *,
    status: str,
    digest: str,
) -> Path:
    payload = {
        "schema_version": "1.0",
        "assets": [
            {
                "id": "rfdetr-source-1.9.0",
                "kind": "source",
                "source": "https://github.com/roboflow/rf-detr",
                "version": "1.9.0@a700efc",
                "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                "license": "Apache-2.0",
                "expected_bytes": package.stat().st_size,
                "commercial_status": "allowed",
                "approval_status": status,
            },
            {
                "id": "rfdetr-seg-small-1.9.0",
                "kind": "model_weight",
                "source": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
                "version": "1.9.0",
                "sha256": digest,
                "license": "Apache-2.0",
                "expected_bytes": weight.stat().st_size,
                "commercial_status": "allowed",
                "approval_status": status,
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_rfdetr_refuses_unapproved_weight_before_backend_import(tmp_path: Path) -> None:
    package = tmp_path / "rfdetr.whl"
    package.write_bytes(b"synthetic-package")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"synthetic-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    manifest = _manifest(tmp_path / "assets.yaml", package, weight, status="pending", digest=digest)
    imported = False

    def backend_factory() -> object:
        nonlocal imported
        imported = True
        return object()

    with pytest.raises(PermissionError, match="not approved"):
        RfDetrAdapter.from_approved_assets(
            asset_manifest=manifest,
            source_asset_id="rfdetr-source-1.9.0",
            source_path=package,
            weight_asset_id="rfdetr-seg-small-1.9.0",
            weight_path=weight,
            backend_factory=backend_factory,
        )
    assert imported is False


def test_rfdetr_refuses_hash_mismatch(tmp_path: Path) -> None:
    package = tmp_path / "rfdetr.whl"
    package.write_bytes(b"synthetic-package")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"tampered")
    manifest = _manifest(
        tmp_path / "assets.yaml", package, weight, status="approved", digest="b" * 64
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        RfDetrAdapter.from_approved_assets(
            asset_manifest=manifest,
            source_asset_id="rfdetr-source-1.9.0",
            source_path=package,
            weight_asset_id="rfdetr-seg-small-1.9.0",
            weight_path=weight,
            backend_factory=object,
        )


def test_rfdetr_refuses_package_hash_mismatch_before_import(tmp_path: Path) -> None:
    package = tmp_path / "rfdetr.whl"
    package.write_bytes(b"approved-package")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"synthetic-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    manifest = _manifest(
        tmp_path / "assets.yaml", package, weight, status="approved", digest=digest
    )
    package.write_bytes(b"tampered-package")

    with pytest.raises(ValueError, match="package sha256 mismatch"):
        RfDetrAdapter.from_approved_assets(
            asset_manifest=manifest,
            source_asset_id="rfdetr-source-1.9.0",
            source_path=package,
            weight_asset_id="rfdetr-seg-small-1.9.0",
            weight_path=weight,
            backend_factory=object,
        )
