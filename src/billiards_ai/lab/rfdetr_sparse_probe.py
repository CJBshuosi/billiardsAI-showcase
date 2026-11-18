"""Verified RF-DETR COCO sports-ball probe over a frozen review-frame pack."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import cv2
import numpy as np
from pydantic import Field

from billiards_ai.lab.contracts import FrozenContract, Sha256

SPORTS_BALL_COCO_CLASS_ID = 37
SPARSE_PROBE_REVISION = "1.1"


class PredictionCollection(Protocol):
    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    mask: np.ndarray | None

    def __len__(self) -> int: ...


class SparseModel(Protocol):
    def predict(self, image: np.ndarray, **kwargs: object) -> PredictionCollection: ...


@dataclass(frozen=True)
class SparseProbeInputs:
    review_root: Path
    weight_path: Path
    expected_weight_sha256: str
    output_root: Path
    threshold: float = 0.1
    model_factory: Callable[[Path], SparseModel] = field(compare=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if len(self.expected_weight_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.expected_weight_sha256
        ):
            raise ValueError("expected weight sha256 must be lowercase hexadecimal")
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if self.model_factory is None:
            raise ValueError("model factory is required")
        if self.output_root.resolve().is_relative_to(self.review_root.resolve()):
            raise ValueError("sparse probe output must be outside the review root")


class SparseProbeRun(FrozenContract):
    run_id: Sha256
    path: Path
    frame_count: int = Field(ge=0)
    all_class_detection_count: int = Field(ge=0)
    sports_ball_detection_count: int = Field(ge=0)
    accuracy_claim: Literal["none"] = "none"
    reused: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _existing(path: Path) -> SparseProbeRun | None:
    summary = path / "summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return SparseProbeRun(
        run_id=payload["run_id"],
        path=path,
        frame_count=payload["frame_count"],
        all_class_detection_count=payload["all_class_detection_count"],
        sports_ball_detection_count=payload["sports_ball_detection_count"],
        reused=True,
    )


def run_sparse_probe(inputs: SparseProbeInputs) -> SparseProbeRun:
    """Run one verified model instance across all review JPGs and save mask evidence."""

    actual_weight_sha256 = _sha256(inputs.weight_path)
    if actual_weight_sha256 != inputs.expected_weight_sha256:
        raise ValueError("RF-DETR weight sha256 mismatch")
    images = tuple(sorted(inputs.review_root.rglob("*.jpg")))
    image_evidence = [
        {
            "path": path.relative_to(inputs.review_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in images
    ]
    run_id = _canonical_digest(
        {
            "schema_version": "1.0",
            "probe_revision": SPARSE_PROBE_REVISION,
            "weight_sha256": actual_weight_sha256,
            "threshold": inputs.threshold,
            "sports_ball_class_id": SPORTS_BALL_COCO_CLASS_ID,
            "images": image_evidence,
        }
    )
    final_path = inputs.output_root / f"run-{run_id}"
    existing = _existing(final_path)
    if existing is not None:
        return existing

    inputs.output_root.mkdir(parents=True, exist_ok=True)
    temporary_path = inputs.output_root / f"run-{run_id}.tmp-{uuid4().hex}"
    temporary_path.mkdir()
    mask_directory = temporary_path / "masks"
    mask_directory.mkdir()
    overlay_directory = temporary_path / "overlays"
    overlay_directory.mkdir()
    all_count = 0
    sports_count = 0
    try:
        model = inputs.model_factory(inputs.weight_path)
        with (temporary_path / "detections.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for image_path, image_record in zip(images, image_evidence, strict=True):
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError(f"review frame cannot be decoded: {image_path}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                detections = model.predict(
                    image_rgb,
                    threshold=inputs.threshold,
                    include_source_image=False,
                )
                all_count += len(detections)
                overlay = image_bgr.copy()
                image_sports_count = 0
                for index in range(len(detections)):
                    class_id = int(detections.class_id[index])
                    if class_id != SPORTS_BALL_COCO_CLASS_ID:
                        continue
                    box = [float(value) for value in detections.xyxy[index]]
                    detection_id = _canonical_digest(
                        {
                            "image_sha256": image_record["sha256"],
                            "index": index,
                            "box": box,
                            "confidence": float(detections.confidence[index]),
                        }
                    )
                    mask_path = None
                    if detections.mask is not None:
                        mask_path = f"masks/{detection_id}.png"
                        mask = np.where(detections.mask[index], 255, 0).astype(np.uint8)
                        if not cv2.imwrite(str(temporary_path / mask_path), mask):
                            raise ValueError(f"failed to write mask for {detection_id}")
                    row: dict[str, Any] = {
                        "detection_id": detection_id,
                        "source_frame": image_record,
                        "class_id": class_id,
                        "class_name": "sports ball",
                        "confidence": float(detections.confidence[index]),
                        "box_xyxy_px": box,
                        "center_px": [
                            (box[0] + box[2]) / 2,
                            (box[1] + box[3]) / 2,
                        ],
                        "mask_path": mask_path,
                        "teacher_id": "rfdetr-seg-small-coco",
                        "teacher_weight_sha256": actual_weight_sha256,
                        "provenance": "pseudo",
                        "review_status": "unreviewed",
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    sports_count += 1
                    image_sports_count += 1
                    cv2.rectangle(
                        overlay,
                        (round(box[0]), round(box[1])),
                        (round(box[2]), round(box[3])),
                        (0, 0, 255),
                        1,
                    )
                    cv2.putText(
                        overlay,
                        f"ball {float(detections.confidence[index]):.2f}",
                        (round(box[0]), max(10, round(box[1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                if image_sports_count:
                    overlay_name = f"{image_record['sha256']}.jpg"
                    if not cv2.imwrite(str(overlay_directory / overlay_name), overlay):
                        raise ValueError(f"failed to write overlay for {image_path}")
        run = SparseProbeRun(
            run_id=run_id,
            path=final_path,
            frame_count=len(images),
            all_class_detection_count=all_count,
            sports_ball_detection_count=sports_count,
        )
        _write_json(
            temporary_path / "summary.json",
            {
                **run.model_dump(mode="json", exclude={"path", "reused"}),
                "weight_sha256": actual_weight_sha256,
                "threshold": inputs.threshold,
            },
        )
        os.replace(temporary_path, final_path)
        return run
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
