from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.training.external_video_split import build_external_video_split
from billiards_ai.lab.training.rfdetr_expansion_scan import (
    ExpansionScanError,
    ExpansionScanInputs,
    run_expansion_scan,
)


class _Detections:
    def __init__(self, *, empty: bool = False) -> None:
        self.xyxy = np.empty((0, 4)) if empty else np.array([[4.0, 4.0, 12.0, 12.0]])
        self.confidence = np.empty((0,)) if empty else np.array([0.4])
        self.class_id = np.empty((0,), dtype=int) if empty else np.array([0])
        self.mask = None if empty else np.array([np.eye(16, dtype=bool)])

    def __len__(self) -> int:
        return len(self.confidence)


class _Model:
    def predict(self, image: np.ndarray, **kwargs: object) -> _Detections:
        return _Detections(empty=bool(image[0, 0, 0]))


def _split() -> object:
    files = [
        {
            "path": f"videos/video{index}.mp4",
            "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "media_kind": "video",
        }
        for index in range(6)
    ]
    return build_external_video_split(
        {
            "dataset_id": "fixture",
            "source_repository": "ATA-space/Det-SAM2",
            "source_commit": "fixed",
            "files": files,
        },
        descriptor_sha256="a" * 64,
        previously_used_asset_ids=("video0",),
        seed=7,
        train_count=4,
        dev_count=1,
        locked_test_count=1,
    )


def _write_split(path: Path, manifest: object) -> None:
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")  # type: ignore[attr-defined]


def _frame(root: Path, asset_id: str, *, value: int) -> None:
    directory = root / "videos" / asset_id / "review-frames"
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(directory / "shot-000-start-frame-000002.jpg"), image)


def test_scan_publishes_detections_and_ranked_hard_queue(tmp_path: Path) -> None:
    manifest = _split()
    split_path = tmp_path / "split.json"
    _write_split(split_path, manifest)
    review = tmp_path / "review"
    first, second = manifest.scan_asset_ids[:2]  # type: ignore[attr-defined]
    _frame(review, first, value=0)
    _frame(review, second, value=255)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    run = run_expansion_scan(
        ExpansionScanInputs(
            split_manifest=split_path,
            review_root=review,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=checkpoint_sha256,
            output_root=tmp_path / "output",
            model_factory=lambda _: _Model(),
            review_limit=10,
        )
    )

    summary = json.loads((run.path / "summary.json").read_text(encoding="utf-8"))
    queue = [
        json.loads(line)
        for line in (run.path / "hard-review.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["frame_count"] == 2
    assert summary["detection_count"] == 1
    assert summary["hard_review_count"] == 2
    assert {item["reason"] for item in queue} == {"no_detection", "low_confidence"}
    assert (run.path / "detections.jsonl").is_file()
    assert len(tuple((run.path / "masks").glob("*.png"))) == 1


def test_scan_rejects_frame_from_locked_test(tmp_path: Path) -> None:
    manifest = _split()
    split_path = tmp_path / "split.json"
    _write_split(split_path, manifest)
    review = tmp_path / "review"
    _frame(review, manifest.splits["locked-test"][0], value=0)  # type: ignore[attr-defined]
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ExpansionScanError, match="not authorized for scanning"):
        run_expansion_scan(
            ExpansionScanInputs(
                split_manifest=split_path,
                review_root=review,
                checkpoint=checkpoint,
                expected_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
                output_root=tmp_path / "output",
                model_factory=lambda _: _Model(),
            )
        )
