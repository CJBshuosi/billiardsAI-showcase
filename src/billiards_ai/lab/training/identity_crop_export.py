"""Deterministic high-resolution ball crops for human identity review."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

import cv2

from billiards_ai.lab.mainline1_identity_preprocessing import (
    IdentityCropExportError as IdentityCropExportError,
)
from billiards_ai.lab.mainline1_identity_preprocessing import (
    square_identity_crop as _square_crop,
)
from billiards_ai.lab.training.rfdetr_bootstrap_train import (
    DetectorTrainingError,
    verify_bootstrap_training_dataset,
)


@dataclass(frozen=True)
class IdentityCropCandidateExport:
    path: Path
    artifact_id: str
    crop_count: int


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, reference: object) -> Path:
    if not isinstance(reference, str):
        raise IdentityCropExportError("COCO image path is invalid")
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or reference in {"", "."}:
        raise IdentityCropExportError("COCO image path is unsafe")
    unresolved = root.joinpath(*relative.parts)
    resolved = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or not resolved.is_relative_to(root.resolve())
        or not resolved.is_file()
    ):
        raise IdentityCropExportError("COCO image is missing or unsafe")
    return resolved


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IdentityCropExportError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise IdentityCropExportError(f"{label} is invalid")
    return value


def _normalise_polygon(
    annotation: dict[str, object],
) -> tuple[list[list[float]], int]:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or not segmentation:
        raise IdentityCropExportError("identity crop requires detector polygons")
    polygons: list[list[list[float]]] = []
    for polygon in segmentation:
        if (
            not isinstance(polygon, list)
            or len(polygon) < 6
            or len(polygon) % 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in polygon
            )
        ):
            raise IdentityCropExportError("identity crop detector polygon is invalid")
        points = [
            [float(polygon[index]), float(polygon[index + 1])]
            for index in range(0, len(polygon), 2)
        ]
        if any(not math.isfinite(value) or value < 0 for point in points for value in point):
            raise IdentityCropExportError("identity crop detector polygon is invalid")
        polygons.append(points)

    def area(points: list[list[float]]) -> float:
        return (
            abs(
                sum(
                    points[index][0] * points[(index + 1) % len(points)][1]
                    - points[(index + 1) % len(points)][0] * points[index][1]
                    for index in range(len(points))
                )
            )
            / 2
        )

    largest = max(
        polygons,
        key=lambda points: (area(points), tuple(value for point in points for value in point)),
    )
    return largest, len(polygons)


def _normalise_box(annotation: dict[str, object]) -> tuple[float, float, float, float]:
    bbox = annotation.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
    ):
        raise IdentityCropExportError("identity crop detector box is invalid")
    x, y, width, height = (float(value) for value in bbox)
    if (
        any(not math.isfinite(value) for value in (x, y, width, height))
        or min(x, y) < 0
        or min(width, height) <= 0
    ):
        raise IdentityCropExportError("identity crop detector box is invalid")
    return x, y, width, height


def _make_immutable(root: Path) -> None:
    if os.name == "nt":
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _verify_published(path: Path) -> IdentityCropCandidateExport:
    manifest = _load_json(path / "manifest.json", label="identity crop manifest")
    artifact_id = manifest.get("artifact_id")
    crops = manifest.get("crops")
    if (
        path.is_symlink()
        or not path.is_dir()
        or not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or path.name != artifact_id
        or not isinstance(crops, list)
        or not crops
    ):
        raise IdentityCropExportError("identity crop artifact is invalid")
    identity = {key: value for key, value in manifest.items() if key != "artifact_id"}
    if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != artifact_id:
        raise IdentityCropExportError("identity crop artifact identity changed")
    expected = {"manifest.json"}
    for item in crops:
        if not isinstance(item, dict):
            raise IdentityCropExportError("identity crop record is invalid")
        relative_path = item.get("relative_path")
        crop = _safe_file(path, relative_path)
        expected.add(str(relative_path))
        if crop.stat().st_size != item.get("byte_count") or _sha256(crop) != item.get("sha256"):
            raise IdentityCropExportError("identity crop payload changed")
    actual = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
    if actual != expected:
        raise IdentityCropExportError("identity crop artifact inventory changed")
    return IdentityCropCandidateExport(path=path, artifact_id=artifact_id, crop_count=len(crops))


def export_identity_crop_candidates(
    *,
    bootstrap_dataset: Path,
    output_root: Path,
    crop_size_px: int = 224,
    context_scale: float = 2.0,
) -> IdentityCropCandidateExport:
    """Create an immutable, unlabeled crop review artifact from reviewed ball masks."""

    if crop_size_px < 32 or crop_size_px > 1024:
        raise IdentityCropExportError("identity crop size must be between 32 and 1024 pixels")
    if not math.isfinite(context_scale) or not 1 <= context_scale <= 4:
        raise IdentityCropExportError("identity crop context scale must be between 1 and 4")
    try:
        evidence = verify_bootstrap_training_dataset(bootstrap_dataset)
    except DetectorTrainingError as exc:
        raise IdentityCropExportError(str(exc)) from exc
    source_manifest = _load_json(
        bootstrap_dataset / "manifest.json", label="bootstrap dataset manifest"
    )
    records: list[dict[str, object]] = []
    encoded_crops: dict[str, bytes] = {}
    for split in ("train", "valid"):
        coco = _load_json(
            bootstrap_dataset / split / "_annotations.coco.json",
            label=f"{split} COCO annotations",
        )
        images = coco.get("images")
        annotations = coco.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise IdentityCropExportError("bootstrap COCO payload is invalid")
        images_by_id = {
            image.get("id"): image
            for image in images
            if isinstance(image, dict) and isinstance(image.get("id"), int)
        }
        for annotation in sorted(
            annotations,
            key=lambda item: int(item.get("id", -1)) if isinstance(item, dict) else -1,
        ):
            if not isinstance(annotation, dict) or annotation.get("image_id") not in images_by_id:
                raise IdentityCropExportError("bootstrap COCO annotation is invalid")
            image_record = images_by_id[annotation["image_id"]]
            source = _safe_file(bootstrap_dataset / split, image_record.get("file_name"))
            decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if decoded is None:
                raise IdentityCropExportError("bootstrap source image cannot be decoded")
            if decoded.shape[:2] != (image_record.get("height"), image_record.get("width")):
                raise IdentityCropExportError("bootstrap source image geometry changed")
            bbox = _normalise_box(annotation)
            polygon, polygon_count = _normalise_polygon(annotation)
            crop = _square_crop(
                decoded,
                bbox,
                context_scale=context_scale,
                output_size=crop_size_px,
            )
            encoded, payload = cv2.imencode(".png", crop)
            if not encoded:
                raise IdentityCropExportError("identity crop encoding failed")
            source_hash = _sha256(source)
            record_identity = {
                "source_dataset_id": evidence.dataset_id,
                "source_split": split,
                "source_asset_id": image_record.get("source_asset_id"),
                "source_frame_index": image_record.get("source_frame_index"),
                "source_image_sha256": source_hash,
                "source_annotation_id": annotation.get("id"),
                "source_instance_id": annotation.get("instance_id"),
                "detector_bbox_xywh": list(bbox),
                "detector_mask_polygon_px": polygon,
                "detector_mask_polygon_count": polygon_count,
            }
            crop_id = hashlib.sha256(_canonical_bytes(record_identity)).hexdigest()
            relative_path = f"crops/{crop_id}.png"
            crop_bytes = payload.tobytes()
            encoded_crops[relative_path] = crop_bytes
            records.append(
                {
                    "crop_id": crop_id,
                    "relative_path": relative_path,
                    "byte_count": len(crop_bytes),
                    "sha256": hashlib.sha256(crop_bytes).hexdigest(),
                    **record_identity,
                    "review_status": "unreviewed",
                    "exact_identity": None,
                    "visibility": None,
                    "orientation": None,
                }
            )
    if not records:
        raise IdentityCropExportError("bootstrap dataset produced no identity crops")
    records.sort(key=lambda item: str(item["crop_id"]))
    identity = {
        "schema_version": "1.0",
        "kind": "ball_identity_crop_candidates",
        "source_dataset_id": evidence.dataset_id,
        "source_dataset_manifest_sha256": _sha256(bootstrap_dataset / "manifest.json"),
        "crop_size_px": crop_size_px,
        "context_scale": context_scale,
        "split_assets": source_manifest["split_assets"],
        "crop_count": len(records),
        "crops": records,
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / artifact_id
    if target.exists():
        return _verify_published(target)
    temporary = output_root / f".pending-{uuid4().hex}"
    try:
        (temporary / "crops").mkdir(parents=True)
        for relative_path, payload in encoded_crops.items():
            temporary.joinpath(*PurePosixPath(relative_path).parts).write_bytes(payload)
        (temporary / "manifest.json").write_bytes(
            _canonical_bytes({"artifact_id": artifact_id, **identity})
        )
        _make_immutable(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _verify_published(target)
