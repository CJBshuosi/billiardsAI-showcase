from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.training.identity_crop_export import (
    IdentityCropExportError,
    export_identity_crop_candidates,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_dataset(
    tmp_path: Path, *, leak_asset: bool = False, multiple_polygons: bool = False
) -> Path:
    pending = tmp_path / "pending"
    inventory_paths: list[Path] = []
    for split, asset_id, colour in (
        ("train", "video-train", (0, 0, 255)),
        ("valid", "video-train" if leak_asset else "video-valid", (255, 0, 0)),
    ):
        split_dir = pending / split
        split_dir.mkdir(parents=True)
        image_name = f"{split}.png"
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[:] = (0, 120, 0)
        cv2.circle(image, (40, 35), 8, colour, -1)
        encoded, payload = cv2.imencode(".png", image)
        assert encoded
        (split_dir / image_name).write_bytes(payload.tobytes())
        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": image_name,
                    "width": 100,
                    "height": 80,
                    "source_asset_id": asset_id,
                    "source_frame_index": 15,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": (
                        [
                            [32.0, 27.0, 48.0, 27.0, 48.0, 43.0, 32.0, 43.0],
                            [60.0, 60.0, 62.0, 60.0, 62.0, 62.0, 60.0, 62.0],
                        ]
                        if multiple_polygons
                        else [[32.0, 27.0, 48.0, 27.0, 48.0, 43.0, 32.0, 43.0]]
                    ),
                    "bbox": [32.0, 27.0, 16.0, 16.0],
                    "area": 256,
                    "iscrowd": 0,
                    "instance_id": f"ball-{split}",
                }
            ],
            "categories": [{"id": 1, "name": "ball", "supercategory": "billiards"}],
        }
        annotation_path = split_dir / "_annotations.coco.json"
        annotation_path.write_bytes(_canonical_bytes(coco))
        inventory_paths.extend((split_dir / image_name, annotation_path))
    inventory = [
        {
            "path": path.relative_to(pending).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(inventory_paths)
    ]
    identity = {
        "schema_version": "1.0",
        "kind": "sam2_rfdetr_bootstrap",
        "source_provenance": {
            "evaluation_manifest_hash": "c" * 64,
            "input_bundle_id": "b" * 64,
            "raw_run_hash": "a" * 64,
            "verdict_artifact_hash": "d" * 64,
        },
        "policy": {
            "benchmark_frames_excluded": True,
            "frame_complete_labels_required": True,
            "qa_flags_allowed": False,
            "scientific_use": "development_bootstrap_not_independent_accuracy",
        },
        "frame_stride": 15,
        "split_assets": {
            "train": ["video-train"],
            "valid": ["video-train" if leak_asset else "video-valid"],
        },
        "rejection_counts": {},
        "image_count": 2,
        "annotation_count": 2,
        "inventory": inventory,
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    dataset = tmp_path / artifact_id
    pending.rename(dataset)
    (dataset / "manifest.json").write_bytes(
        _canonical_bytes({"artifact_id": artifact_id, **identity})
    )
    if os.name != "nt":
        for path in sorted(dataset.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        dataset.chmod(0o555)
    return dataset


def test_export_builds_content_addressed_unreviewed_square_crops(tmp_path: Path) -> None:
    dataset = _bootstrap_dataset(tmp_path)

    result = export_identity_crop_candidates(
        bootstrap_dataset=dataset,
        output_root=tmp_path / "outputs",
        crop_size_px=64,
        context_scale=2.0,
    )

    manifest = json.loads((result.path / "manifest.json").read_text(encoding="utf-8"))
    assert result.crop_count == 2
    assert manifest["artifact_id"] == result.artifact_id == result.path.name
    assert manifest["source_dataset_id"] == dataset.name
    assert {item["source_split"] for item in manifest["crops"]} == {"train", "valid"}
    assert {item["review_status"] for item in manifest["crops"]} == {"unreviewed"}
    assert all(item["exact_identity"] is None for item in manifest["crops"])
    for item in manifest["crops"]:
        crop = cv2.imread(str(result.path / item["relative_path"]), cv2.IMREAD_COLOR)
        assert crop is not None
        assert crop.shape == (64, 64, 3)
        assert _sha256(result.path / item["relative_path"]) == item["sha256"]


def test_export_is_deterministic_and_reuses_the_verified_artifact(tmp_path: Path) -> None:
    dataset = _bootstrap_dataset(tmp_path)
    first = export_identity_crop_candidates(
        bootstrap_dataset=dataset,
        output_root=tmp_path / "outputs",
        crop_size_px=64,
        context_scale=2.0,
    )
    second = export_identity_crop_candidates(
        bootstrap_dataset=dataset,
        output_root=tmp_path / "outputs",
        crop_size_px=64,
        context_scale=2.0,
    )

    assert second == first


def test_export_uses_largest_polygon_when_a_mask_has_disconnected_contours(
    tmp_path: Path,
) -> None:
    dataset = _bootstrap_dataset(tmp_path, multiple_polygons=True)

    result = export_identity_crop_candidates(
        bootstrap_dataset=dataset,
        output_root=tmp_path / "outputs",
        crop_size_px=64,
        context_scale=2.0,
    )

    manifest = json.loads((result.path / "manifest.json").read_text(encoding="utf-8"))
    assert result.crop_count == 2
    assert {item["detector_mask_polygon_count"] for item in manifest["crops"]} == {2}
    assert {
        tuple(tuple(point) for point in item["detector_mask_polygon_px"])
        for item in manifest["crops"]
    } == {((32.0, 27.0), (48.0, 27.0), (48.0, 43.0), (32.0, 43.0))}


def test_export_rejects_an_asset_shared_by_train_and_valid(tmp_path: Path) -> None:
    dataset = _bootstrap_dataset(tmp_path, leak_asset=True)

    with pytest.raises(IdentityCropExportError, match="video-disjoint"):
        export_identity_crop_candidates(
            bootstrap_dataset=dataset,
            output_root=tmp_path / "outputs",
            crop_size_px=64,
            context_scale=2.0,
        )
