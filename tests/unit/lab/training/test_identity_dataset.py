import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from billiards_ai.lab.training.identity_dataset import (
    IdentityCrop,
    IdentityDatasetError,
    build_identity_dataset,
)


def _crop(
    root: Path,
    *,
    crop_id: str,
    group: str,
    identity: str | None,
    group_split: str,
    evidence: str | None = None,
) -> IdentityCrop:
    path = root / f"{crop_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"synthetic-{crop_id}".encode())
    return IdentityCrop(
        crop_id=crop_id,
        relative_path=f"{crop_id}.png",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_asset_id=f"asset-{group_split}",
        source_frame_id=f"frame-{crop_id}",
        source_frame_sha256="f" * 64,
        calibration_revision="cal-r1",
        calibration_sha256="c" * 64,
        detector_bbox_xywh=(10.0, 10.0, 20.0, 20.0),
        detector_mask_polygon_px=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0)),
        group_split=group_split,
        group=group,
        exact_identity=identity,
        exact_evidence=evidence,
        visibility="occluded" if identity is None else "visible",
        orientation="number_hidden" if identity is None else "number_visible",
    )


def test_crop_contract_requires_full_detector_and_calibration_provenance() -> None:
    with pytest.raises(ValidationError):
        IdentityCrop.model_validate(
            {
                "crop_id": "crop-a",
                "relative_path": "crop-a.png",
                "sha256": "a" * 64,
                "source_asset_id": "asset-a",
                "source_frame_id": "frame-a",
                "group_split": "train",
                "group": "eight",
                "exact_identity": "8",
                "exact_evidence": "group_intrinsic",
                "visibility": "visible",
                "orientation": "number_visible",
            }
        )


def test_external_identity_crop_can_explicitly_omit_table_calibration(tmp_path: Path) -> None:
    source = tmp_path / "source"
    path = source / "external-crop.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"external-identity-crop")

    crop = IdentityCrop(
        crop_id="external-crop",
        relative_path=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_asset_id="external-video",
        source_frame_id="external-video:15",
        source_frame_sha256="f" * 64,
        calibration_revision=None,
        calibration_sha256=None,
        detector_bbox_xywh=(10.0, 10.0, 20.0, 20.0),
        detector_mask_polygon_px=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0)),
        group_split="external-video:train",
        group="eight",
        exact_identity="8",
        exact_evidence="group_intrinsic",
        visibility="visible",
        orientation="number_visible",
    )

    assert build_identity_dataset(crops=(crop,), source_root=source, split="train").records == (
        crop,
    )


def test_identity_dataset_balances_groups_and_masks_missing_exact_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    crops = (
        _crop(
            source,
            crop_id="solid-a",
            group="solid",
            identity=None,
            group_split="train",
        ),
        _crop(
            source,
            crop_id="solid-b",
            group="solid",
            identity="4",
            evidence="number_visible",
            group_split="train",
        ),
        _crop(
            source,
            crop_id="stripe-a",
            group="stripe",
            identity=None,
            group_split="train",
        ),
    )

    dataset = build_identity_dataset(crops=crops, source_root=source, split="train")

    assert dataset.group_class_weights == {"solid": 0.5, "stripe": 1.0}
    assert dataset.exact_class_weights == {"4": 1.0}
    assert dataset.exact_labeled_count == 1


def test_identity_dataset_rejects_group_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    crops = (
        _crop(
            source,
            crop_id="train-a",
            group="solid",
            identity="1",
            evidence="known_set_mapping",
            group_split="session-a:train",
        ),
        _crop(
            source,
            crop_id="dev-a",
            group="solid",
            identity="1",
            evidence="known_set_mapping",
            group_split="session-a:dev",
        ),
    )

    with pytest.raises(IdentityDatasetError, match="group leakage"):
        build_identity_dataset(crops=crops, source_root=source, split="train")
