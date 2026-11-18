from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from billiards_ai.lab.vision.package import (
    VisionPackageError,
    canonical_package_id,
    load_vision_package,
)


def _canonical_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode()


def _base_payload(checkpoint: bytes) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "task": "ball_instance_segmentation",
        "family": "rfdetr",
        "model_version": "1.9.0",
        "model_class": "RFDETRSegSmall",
        "artifacts": [
            {
                "role": "checkpoint",
                "relative_path": "checkpoint_best.pth",
                "byte_count": len(checkpoint),
                "sha256": hashlib.sha256(checkpoint).hexdigest(),
            }
        ],
        "preprocess": {
            "color_space": "RGB",
            "input_width_px": 384,
            "input_height_px": 384,
        },
        "postprocess": {"confidence_threshold": 0.25, "mask_threshold": 0.5},
        "class_map": {"ball": 0},
        "runtime_requirements": ["rfdetr==1.9.0"],
        "training_run_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "base_weight_sha256": "3" * 64,
        "benchmark_artifact_ids": ["frozen-benchmark-r1"],
        "supported_capabilities": ["ball_mask", "ball_bbox", "ball_center"],
        "unsupported_capabilities": ["ball_identity_1_15", "tracking", "events"],
        "asset_status_ref": "internal-rnd-only",
    }


def _write_package(
    tmp_path: Path,
    *,
    mutation: str | None = None,
    payload_mutation: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    package_root = tmp_path / "vision-package"
    package_root.mkdir()
    checkpoint = b"checkpoint"
    checkpoint_path = package_root / "checkpoint_best.pth"
    checkpoint_path.write_bytes(checkpoint)
    payload = _base_payload(checkpoint)
    if mutation == "declared_size":
        payload["artifacts"][0]["byte_count"] = 99  # type: ignore[index]
    if mutation == "declared_hash":
        payload["artifacts"][0]["sha256"] = "0" * 64  # type: ignore[index]
    if payload_mutation:
        payload.update(payload_mutation)
    payload["package_id"] = canonical_package_id(payload)
    (package_root / "manifest.json").write_bytes(_canonical_bytes(payload))
    if mutation == "checkpoint_bytes":
        checkpoint_path.write_bytes(b"tampered")
    return package_root, payload


def test_loader_verifies_a_content_addressed_package(tmp_path: Path) -> None:
    package_root, expected = _write_package(tmp_path)

    verified = load_vision_package(package_root)

    assert verified.identity.package_id == expected["package_id"]
    assert verified.checkpoint.read_bytes() == b"checkpoint"
    assert verified.manifest.postprocess.confidence_threshold == 0.25


@pytest.mark.parametrize("mutation", ["checkpoint_bytes", "declared_size", "declared_hash"])
def test_loader_rejects_tampered_artifacts(tmp_path: Path, mutation: str) -> None:
    package_root, _ = _write_package(tmp_path, mutation=mutation)

    with pytest.raises(VisionPackageError, match="mismatch"):
        load_vision_package(package_root)


def test_loader_rejects_manifest_identity_change(tmp_path: Path) -> None:
    package_root, _ = _write_package(tmp_path)
    payload = json.loads((package_root / "manifest.json").read_text())
    payload["postprocess"]["confidence_threshold"] = 0.4
    (package_root / "manifest.json").write_bytes(_canonical_bytes(payload))

    with pytest.raises(VisionPackageError, match="package identity"):
        load_vision_package(package_root)


@pytest.mark.parametrize(
    "relative_path",
    ["/tmp/model.pth", "../model.pth", "nested/../../model.pth"],
)
def test_loader_rejects_unconfined_artifact_paths(tmp_path: Path, relative_path: str) -> None:
    payload = _base_payload(b"checkpoint")
    payload["artifacts"][0]["relative_path"] = relative_path  # type: ignore[index]
    package_root, _ = _write_package(tmp_path, payload_mutation={"artifacts": payload["artifacts"]})

    with pytest.raises(VisionPackageError, match="relative and confined"):
        load_vision_package(package_root)


def test_loader_rejects_symlinked_artifact(tmp_path: Path) -> None:
    package_root, _ = _write_package(tmp_path)
    checkpoint = package_root / "checkpoint_best.pth"
    outside = tmp_path / "outside.pth"
    outside.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    os.symlink(outside, checkpoint)

    with pytest.raises(VisionPackageError, match="symlink"):
        load_vision_package(package_root)


def test_loader_requires_exactly_one_checkpoint(tmp_path: Path) -> None:
    artifact = _base_payload(b"checkpoint")["artifacts"][0]  # type: ignore[index]
    package_root, _ = _write_package(
        tmp_path,
        payload_mutation={"artifacts": [{**artifact, "role": "onnx"}]},
    )

    with pytest.raises(VisionPackageError, match="exactly one checkpoint"):
        load_vision_package(package_root)


def test_loader_rejects_duplicate_roles(tmp_path: Path) -> None:
    artifact = _base_payload(b"checkpoint")["artifacts"][0]  # type: ignore[index]
    package_root, _ = _write_package(
        tmp_path,
        payload_mutation={"artifacts": [artifact, artifact]},
    )

    with pytest.raises(VisionPackageError, match="unique"):
        load_vision_package(package_root)


@pytest.mark.parametrize(
    "payload_mutation,error",
    [
        ({"class_map": {"ball": 1}}, "class map"),
        ({"unexpected": True}, "invalid manifest"),
    ],
)
def test_loader_rejects_invalid_manifest_contract(
    tmp_path: Path,
    payload_mutation: dict[str, object],
    error: str,
) -> None:
    package_root, _ = _write_package(tmp_path, payload_mutation=payload_mutation)

    with pytest.raises(VisionPackageError, match=error):
        load_vision_package(package_root)
