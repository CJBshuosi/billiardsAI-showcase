from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.rfdetr_sparse_probe import SparseProbeInputs, run_sparse_probe


class FakeDetections:
    xyxy = np.asarray([[10.0, 12.0, 20.0, 22.0], [1.0, 2.0, 8.0, 9.0]])
    confidence = np.asarray([0.8, 0.9])
    class_id = np.asarray([37, 1])
    mask = np.asarray(
        [
            np.pad(np.ones((10, 10), dtype=bool), ((12, 42), (10, 76))),
            np.zeros((64, 96), dtype=bool),
        ]
    )

    def __len__(self) -> int:
        return 2


class FakeModel:
    def predict(self, image: np.ndarray, **kwargs: object) -> FakeDetections:
        assert image.shape == (64, 96, 3)
        assert kwargs["threshold"] == 0.1
        return FakeDetections()


def test_sparse_probe_preserves_only_unreviewed_sports_ball_evidence(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    frame = review_root / "videos" / "video1" / "review-frames" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    assert cv2.imwrite(str(frame), np.zeros((64, 96, 3), dtype=np.uint8))
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"verified fixture weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()

    first = run_sparse_probe(
        SparseProbeInputs(
            review_root=review_root,
            weight_path=weight,
            expected_weight_sha256=digest,
            output_root=tmp_path / "output",
            threshold=0.1,
            model_factory=lambda _: FakeModel(),
        )
    )
    second = run_sparse_probe(
        SparseProbeInputs(
            review_root=review_root,
            weight_path=weight,
            expected_weight_sha256=digest,
            output_root=tmp_path / "output",
            threshold=0.1,
            model_factory=lambda _: FakeModel(),
        )
    )

    assert first.path == second.path
    assert second.reused is True
    assert first.frame_count == 1
    assert first.sports_ball_detection_count == 1
    row = (first.path / "detections.jsonl").read_text().splitlines()
    assert len(row) == 1
    assert '"provenance": "pseudo"' in row[0]
    assert '"review_status": "unreviewed"' in row[0]
    assert '"accuracy_claim": "none"' in (first.path / "summary.json").read_text()
    assert len(list((first.path / "masks").glob("*.png"))) == 1
    assert len(list((first.path / "overlays").glob("*.jpg"))) == 1
