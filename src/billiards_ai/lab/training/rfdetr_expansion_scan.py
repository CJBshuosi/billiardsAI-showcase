"""Run a trained RF-DETR over expansion frames without opening locked-test videos."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

import cv2
import numpy as np
from pydantic import Field

from billiards_ai.lab.contracts import FrozenContract, Sha256
from billiards_ai.lab.training.external_video_split import load_external_video_split

_FRAME_PATTERN = re.compile(
    r"^shot-(?P<shot>\d+)-(?P<role>start|middle|end)-frame-(?P<frame>\d+)\.jpg$"
)
_SCAN_REVISION = "1.0"


class ExpansionScanError(RuntimeError):
    """The trained expansion scan is unsafe, incomplete, or identity-inconsistent."""


class PredictionCollection(Protocol):
    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    mask: np.ndarray | None

    def __len__(self) -> int: ...


class ExpansionModel(Protocol):
    def predict(self, image: np.ndarray, **kwargs: object) -> PredictionCollection: ...


@dataclass(frozen=True)
class ExpansionScanInputs:
    split_manifest: Path
    review_root: Path
    checkpoint: Path
    expected_checkpoint_sha256: str
    output_root: Path
    model_factory: Callable[[Path], ExpansionModel] = field(compare=False, repr=False)
    threshold: float = 0.1
    low_confidence_below: float = 0.6
    review_limit: int = 200

    def __post_init__(self) -> None:
        if len(self.expected_checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.expected_checkpoint_sha256
        ):
            raise ValueError("checkpoint sha256 must be lowercase hexadecimal")
        if not 0 < self.threshold <= self.low_confidence_below <= 1:
            raise ValueError("expansion scan confidence thresholds are invalid")
        if self.review_limit < 0:
            raise ValueError("expansion scan review limit must be non-negative")
        if self.output_root.resolve().is_relative_to(self.review_root.resolve()):
            raise ValueError("expansion scan output must be outside the review root")


class ExpansionScanRun(FrozenContract):
    run_id: Sha256
    path: Path
    frame_count: int = Field(ge=0)
    detection_count: int = Field(ge=0)
    hard_review_count: int = Field(ge=0)
    locked_test_opened: Literal[False] = False
    accuracy_claim: Literal["none"] = "none"
    reused: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _frame_identity(review_root: Path, path: Path) -> dict[str, object]:
    try:
        relative = path.relative_to(review_root)
    except ValueError as exc:
        raise ExpansionScanError("expansion frame escapes the review root") from exc
    parts = relative.parts
    if len(parts) != 4 or parts[0] != "videos" or parts[2] != "review-frames":
        raise ExpansionScanError("expansion review frame path is invalid")
    match = _FRAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ExpansionScanError("expansion review frame name is invalid")
    return {
        "asset_id": parts[1],
        "shot_index": int(match.group("shot")),
        "role": match.group("role"),
        "frame_index": int(match.group("frame")),
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _hard_reason(
    *, detections: list[dict[str, object]], width: int, height: int, low_below: float
) -> tuple[str, int]:
    if not detections:
        return "no_detection", 0
    if len(detections) > 16:
        return "count_above_16", 1
    confidences = [float(item["confidence"]) for item in detections]
    if min(confidences) < low_below:
        return "low_confidence", 2
    for detection in detections:
        left, top, right, bottom = detection["box_xyxy_px"]  # type: ignore[misc]
        box_width = float(right) - float(left)
        box_height = float(bottom) - float(top)
        if (
            min(box_width, box_height) <= 0
            or max(box_width, box_height) / min(box_width, box_height) > 2
            or max(box_width, box_height) > min(width, height) * 0.1
        ):
            return "geometry_outlier", 3
    return "easy_sample", 4


def _existing(path: Path) -> ExpansionScanRun | None:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return ExpansionScanRun(
            run_id=summary["run_id"],
            path=path,
            frame_count=summary["frame_count"],
            detection_count=summary["detection_count"],
            hard_review_count=summary["hard_review_count"],
            reused=True,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ExpansionScanError("existing expansion scan is invalid") from exc


def run_expansion_scan(inputs: ExpansionScanInputs) -> ExpansionScanRun:
    """Scan all supplied train/dev frames and publish a bounded hard-review queue."""

    split = load_external_video_split(inputs.split_manifest)
    if split.locked_test_opened:
        raise ExpansionScanError("expansion scan cannot use an opened locked-test split")
    if not inputs.checkpoint.is_file():
        raise ExpansionScanError("trained checkpoint is missing")
    checkpoint_sha256 = _sha256_file(inputs.checkpoint)
    if checkpoint_sha256 != inputs.expected_checkpoint_sha256:
        raise ExpansionScanError("trained checkpoint hash mismatch")
    images = tuple(sorted(inputs.review_root.rglob("*.jpg")))
    if not images:
        raise ExpansionScanError("expansion review pack contains no frames")
    evidence = [_frame_identity(inputs.review_root, path) for path in images]
    authorized = set(split.scan_asset_ids)
    if any(item["asset_id"] not in authorized for item in evidence):
        raise ExpansionScanError("review frame asset is not authorized for scanning")

    run_id = _canonical_digest(
        {
            "schema_version": "1.0",
            "scan_revision": _SCAN_REVISION,
            "split_artifact_id": split.artifact_id,
            "checkpoint_sha256": checkpoint_sha256,
            "threshold": inputs.threshold,
            "low_confidence_below": inputs.low_confidence_below,
            "images": evidence,
        }
    )
    final_path = inputs.output_root / f"run-{run_id}"
    existing = _existing(final_path)
    if existing is not None:
        return existing
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    temporary = inputs.output_root / f"run-{run_id}.tmp-{uuid4().hex}"
    temporary.mkdir()
    (temporary / "masks").mkdir()
    (temporary / "overlays").mkdir()
    detections_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    try:
        model = inputs.model_factory(inputs.checkpoint)
        for image_path, frame in zip(images, evidence, strict=True):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ExpansionScanError("expansion review frame cannot be decoded")
            height, width = image_bgr.shape[:2]
            predictions = model.predict(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                threshold=inputs.threshold,
                include_source_image=False,
            )
            frame_detections: list[dict[str, object]] = []
            overlay = image_bgr.copy()
            for index in range(len(predictions)):
                if int(predictions.class_id[index]) != 0:
                    continue
                box = [float(value) for value in predictions.xyxy[index]]
                detection_id = _canonical_digest(
                    {
                        "frame_sha256": frame["sha256"],
                        "index": index,
                        "box": box,
                        "confidence": float(predictions.confidence[index]),
                    }
                )
                mask_path: str | None = None
                if predictions.mask is not None:
                    mask = np.where(predictions.mask[index], 255, 0).astype(np.uint8)
                    if mask.shape != (height, width):
                        raise ExpansionScanError("expansion prediction mask shape changed")
                    mask_path = f"masks/{detection_id}.png"
                    if not cv2.imwrite(str(temporary / mask_path), mask):
                        raise ExpansionScanError("expansion prediction mask cannot be written")
                row = {
                    "detection_id": detection_id,
                    "asset_id": frame["asset_id"],
                    "shot_index": frame["shot_index"],
                    "role": frame["role"],
                    "frame_index": frame["frame_index"],
                    "source_frame": frame,
                    "confidence": float(predictions.confidence[index]),
                    "box_xyxy_px": box,
                    "mask_path": mask_path,
                    "class_id": 0,
                    "class_name": "ball",
                    "provenance": "pseudo",
                    "review_status": "unreviewed",
                }
                detections_rows.append(row)
                frame_detections.append(row)
                cv2.rectangle(
                    overlay,
                    (round(box[0]), round(box[1])),
                    (round(box[2]), round(box[3])),
                    (0, 0, 255),
                    1,
                )
            reason, priority = _hard_reason(
                detections=frame_detections,
                width=width,
                height=height,
                low_below=inputs.low_confidence_below,
            )
            overlay_name = f"{frame['sha256']}.jpg"
            if not cv2.imwrite(str(temporary / "overlays" / overlay_name), overlay):
                raise ExpansionScanError("expansion overlay cannot be written")
            frame_row = {
                **frame,
                "detection_count": len(frame_detections),
                "reason": reason,
                "priority": priority,
                "overlay_path": f"overlays/{overlay_name}",
            }
            frame_rows.append(frame_row)
            review_rows.append(frame_row)
        ordered_review = sorted(
            review_rows,
            key=lambda item: (
                int(item["priority"]),
                str(item["asset_id"]),
                int(item["frame_index"]),
            ),
        )[: inputs.review_limit]
        _write_jsonl(temporary / "detections.jsonl", detections_rows)
        _write_jsonl(temporary / "frame-results.jsonl", frame_rows)
        _write_jsonl(temporary / "hard-review.jsonl", ordered_review)
        summary = {
            "schema_version": "1.0",
            "run_id": run_id,
            "split_artifact_id": split.artifact_id,
            "checkpoint_sha256": checkpoint_sha256,
            "frame_count": len(frame_rows),
            "detection_count": len(detections_rows),
            "hard_review_count": len(ordered_review),
            "locked_test_opened": False,
            "accuracy_claim": "none",
        }
        _write_json(temporary / "summary.json", summary)
        os.replace(temporary, final_path)
        return ExpansionScanRun(
            run_id=run_id,
            path=final_path,
            frame_count=len(frame_rows),
            detection_count=len(detections_rows),
            hard_review_count=len(ordered_review),
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
