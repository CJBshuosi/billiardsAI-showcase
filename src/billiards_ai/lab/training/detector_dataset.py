"""Deterministic COCO instance-segmentation export for approved training frames."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from billiards_ai.lab.contracts import (
    CalibrationReference,
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    VideoAsset,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SafeId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", min_length=1),
]


class DetectorTrainingDataError(RuntimeError):
    """The requested derived training dataset is unsafe or inconsistent."""


class DetectorAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    polygon_px: tuple[tuple[float, float], ...] = Field(min_length=3)
    bbox_xywh: tuple[float, float, float, float]
    visibility: Literal["visible", "partial", "occluded", "unknown"]
    difficult: bool


class DetectorFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: SafeId
    source_asset_id: SafeId
    relative_path: str
    sha256: Sha256
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    table_cropped: bool
    audio_free: bool
    annotations: tuple[DetectorAnnotation, ...] = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("frame path must be relative and confined")
        return value


@dataclass(frozen=True)
class DetectorDatasetExport:
    output: Path
    data_hash: str
    image_count: int
    annotation_count: int


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_group_isolation(manifest: DatasetManifest) -> None:
    asset_by_id = {asset.asset_id: asset for asset in manifest.assets}
    training_ids = set(manifest.splits.get(DatasetSplit.TRAIN, ()))
    other_ids = {
        asset_id
        for split, asset_ids in manifest.splits.items()
        if split is not DatasetSplit.TRAIN
        for asset_id in asset_ids
    }
    group_fields = (
        ("person", "person_group_id"),
        ("session", "session_group_id"),
        ("table", "table_group_id"),
        ("camera", "camera_group_id"),
    )
    for label, field in group_fields:
        training_groups = {getattr(asset_by_id[asset_id], field) for asset_id in training_ids}
        other_groups = {getattr(asset_by_id[asset_id], field) for asset_id in other_ids}
        if training_groups & other_groups:
            raise DetectorTrainingDataError(f"{label} leakage across training and evaluation")


def _source_path(source_root: Path, relative_path: str) -> Path:
    root = source_root.resolve()
    unresolved = source_root / relative_path
    if unresolved.is_symlink():
        raise DetectorTrainingDataError("training frame cannot be a symbolic link")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise DetectorTrainingDataError("training frame is missing or escapes source root")
    return resolved


def _validate_geometry(frame: DetectorFrame) -> None:
    for annotation in frame.annotations:
        x, y, width, height = annotation.bbox_xywh
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise DetectorTrainingDataError("annotation bounding box is invalid")
        if x + width > frame.width or y + height > frame.height:
            raise DetectorTrainingDataError("annotation bounding box exceeds image")
        if any(
            x_px < 0 or y_px < 0 or x_px > frame.width or y_px > frame.height
            for x_px, y_px in annotation.polygon_px
        ):
            raise DetectorTrainingDataError("segmentation polygon exceeds image")


def _immutable_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(payload, checksum)
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def _synthetic_rgb_png(*, width: int = 64, height: int = 64) -> bytes:
    """Build a deterministic RGB PNG with a single ball-like square."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            if 24 <= x < 40 and 24 <= y < 40:
                rows.extend((245, 245, 245))
            else:
                rows.extend((18, 92, 52))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def export_detector_dataset(
    *,
    manifest: DatasetManifest,
    frames: tuple[DetectorFrame, ...],
    source_root: Path,
    output: Path,
    include_rfdetr_layout: bool = False,
) -> DetectorDatasetExport:
    """Export an immutable training-only COCO dataset without augmentation."""
    if output.exists():
        raise DetectorTrainingDataError("derived dataset output already exists")
    if not manifest.locked:
        raise DetectorTrainingDataError("source dataset must be frozen")
    if not frames:
        raise DetectorTrainingDataError("at least one training frame is required")
    _assert_group_isolation(manifest)

    asset_by_id = {asset.asset_id: asset for asset in manifest.assets}
    training_ids = set(manifest.splits.get(DatasetSplit.TRAIN, ()))
    if not training_ids:
        raise DetectorTrainingDataError("source dataset has no training split")
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise DetectorTrainingDataError("training frame IDs must be unique")

    checked: list[tuple[DetectorFrame, Path]] = []
    for frame in frames:
        if frame.source_asset_id not in training_ids:
            raise DetectorTrainingDataError("frame source is not in the training split")
        asset = asset_by_id[frame.source_asset_id]
        if DatasetPurpose.LOCAL_TRAINING not in asset.approved_uses:
            raise DetectorTrainingDataError("frame source lacks local-training approval")
        if not frame.table_cropped or not frame.audio_free:
            raise DetectorTrainingDataError("training frames must be table-cropped and audio-free")
        _validate_geometry(frame)
        path = _source_path(source_root, frame.relative_path)
        if _sha256_file(path) != frame.sha256:
            raise DetectorTrainingDataError("training frame sha256 mismatch")
        checked.append((frame, path))

    ordered = sorted(checked, key=lambda item: item[0].frame_id)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    source_assets = []
    for asset_id in sorted({frame.source_asset_id for frame, _ in ordered}):
        asset = asset_by_id[asset_id]
        source_assets.append(
            {
                "asset_id": asset.asset_id,
                "sha256": asset.sha256,
                "gate_revision_id": str(asset.gate_revision_id)
                if asset.gate_revision_id is not None
                else None,
                "calibration_revision": asset.calibration.revision,
                "calibration_sha256": asset.calibration.sha256,
            }
        )

    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    try:
        (temporary / "images").mkdir(parents=True)
        (temporary / "annotations").mkdir()
        annotation_id = 1
        for image_id, (frame, source) in enumerate(ordered, start=1):
            suffix = Path(frame.relative_path).suffix.lower() or ".bin"
            file_name = f"images/{frame.frame_id}{suffix}"
            shutil.copyfile(source, temporary / file_name)
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": frame.width,
                    "height": frame.height,
                    "sha256": frame.sha256,
                    "source_asset_id": frame.source_asset_id,
                }
            )
            for annotation in frame.annotations:
                polygon = [coordinate for point in annotation.polygon_px for coordinate in point]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": [polygon],
                        "bbox": list(annotation.bbox_xywh),
                        "area": annotation.bbox_xywh[2] * annotation.bbox_xywh[3],
                        "iscrowd": 0,
                        "visibility": annotation.visibility,
                        "difficult": annotation.difficult,
                    }
                )
                annotation_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "ball", "supercategory": "billiards"}],
        }
        rfdetr_layout_hashes: dict[str, str] | None = None
        if include_rfdetr_layout:
            rfdetr_images = [
                {
                    "id": image["id"],
                    "file_name": Path(str(image["file_name"])).name,
                    "width": image["width"],
                    "height": image["height"],
                }
                for image in images
            ]
            rfdetr_annotations = [
                {
                    key: annotation[key]
                    for key in (
                        "id",
                        "image_id",
                        "category_id",
                        "segmentation",
                        "bbox",
                        "area",
                        "iscrowd",
                    )
                }
                for annotation in annotations
            ]
            rfdetr_coco = {
                "images": rfdetr_images,
                "annotations": rfdetr_annotations,
                "categories": coco["categories"],
            }
            rfdetr_annotation_bytes = _canonical_bytes(rfdetr_coco)
            rfdetr_layout_hashes = {}
            for split in ("train", "valid"):
                split_dir = temporary / split
                split_dir.mkdir()
                for frame, source in ordered:
                    suffix = Path(frame.relative_path).suffix.lower() or ".bin"
                    relative_path = f"{split}/{frame.frame_id}{suffix}"
                    shutil.copyfile(source, temporary / relative_path)
                    rfdetr_layout_hashes[relative_path] = frame.sha256
                relative_annotations = f"{split}/_annotations.coco.json"
                (temporary / relative_annotations).write_bytes(rfdetr_annotation_bytes)
                rfdetr_layout_hashes[relative_annotations] = hashlib.sha256(
                    rfdetr_annotation_bytes
                ).hexdigest()
        derived_base = {
            "schema_version": "1.0",
            "source_dataset_revision": manifest.revision,
            "purpose": DatasetPurpose.LOCAL_TRAINING.value,
            "table_cropped": True,
            "audio_free": True,
            "source_assets": source_assets,
            "frame_hashes": [frame.sha256 for frame, _ in ordered],
            "augmentation_applied": False,
        }
        if rfdetr_layout_hashes is not None:
            derived_base["rfdetr_layout_hashes"] = dict(sorted(rfdetr_layout_hashes.items()))
        data_hash = hashlib.sha256(
            _canonical_bytes({"coco": coco, "derived_manifest": derived_base})
        ).hexdigest()
        derived = {**derived_base, "data_hash": data_hash}
        (temporary / "annotations" / "instances_train.json").write_bytes(_canonical_bytes(coco))
        (temporary / "derived-manifest.json").write_bytes(_canonical_bytes(derived))
        _immutable_tree(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return DetectorDatasetExport(
        output=output,
        data_hash=data_hash,
        image_count=len(images),
        annotation_count=len(annotations),
    )


def create_synthetic_rfdetr_fixture(output: Path) -> DetectorDatasetExport:
    """Create one deterministic, decodable, synthetic-only RF-DETR COCO fixture."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rfdetr-synthetic-", dir=output.parent) as source_name:
        source_root = Path(source_name)
        image = source_root / "frame.png"
        image.write_bytes(_synthetic_rgb_png())
        image_sha256 = _sha256_file(image)
        asset = VideoAsset(
            asset_id="synthetic-rfdetr-a",
            relative_path="synthetic-rfdetr-a.bin",
            sha256="a" * 64,
            is_real_person=False,
            person_group_id="synthetic-person-a",
            session_group_id="synthetic-session-a",
            table_group_id="synthetic-table-a",
            camera_group_id="synthetic-camera-a",
            calibration=CalibrationReference(revision="synthetic-cal-r1", sha256="c" * 64),
            approved_uses=(DatasetPurpose.LOCAL_TRAINING,),
        )
        manifest = DatasetManifest(
            revision="synthetic-rfdetr-r1",
            locked=True,
            assets=(asset,),
            splits={DatasetSplit.TRAIN: (asset.asset_id,)},
        )
        frame = DetectorFrame(
            frame_id="frame-1",
            source_asset_id=asset.asset_id,
            relative_path="frame.png",
            sha256=image_sha256,
            width=64,
            height=64,
            table_cropped=True,
            audio_free=True,
            annotations=(
                DetectorAnnotation(
                    polygon_px=((24, 24), (40, 24), (40, 40), (24, 40)),
                    bbox_xywh=(24, 24, 16, 16),
                    visibility="visible",
                    difficult=False,
                ),
            ),
        )
        return export_detector_dataset(
            manifest=manifest,
            frames=(frame,),
            source_root=source_root,
            output=output,
            include_rfdetr_layout=True,
        )
