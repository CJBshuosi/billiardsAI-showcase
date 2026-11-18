from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from billiards_ai.lab.training.identity_group_backend import (
    Dinov2GroupTrainingBackend,
)


class FakeFeatures:
    version = "fake-dinov2"

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray:
        vectors = []
        for crop in crops:
            group_index = int(crop.stem.split("-")[0])
            vector = np.zeros(5, dtype=np.float32)
            vector[group_index] = 1.0
            vectors.append(vector)
        return np.stack(vectors)


class FakeOnnxAdapter:
    def export(self, *, weight: np.ndarray, bias: np.ndarray, output_path: Path) -> None:
        output_path.write_bytes(b"group-head-onnx")

    def logits(self, *, weight: np.ndarray, bias: np.ndarray, features: np.ndarray) -> np.ndarray:
        return features @ weight.T + bias


def _records(root: Path, *, copies: int) -> tuple[object, ...]:
    groups = ("cue", "solid", "eight", "stripe", "unknown")
    records = []
    for group_index, group in enumerate(groups):
        for copy in range(copies):
            relative_path = f"{group_index}-{copy}.png"
            (root / relative_path).write_bytes(b"crop")
            records.append(
                SimpleNamespace(
                    crop_id=f"{group}-{copy}",
                    group=group,
                    relative_path=relative_path,
                )
            )
    return tuple(records)


def test_real_group_backend_trains_only_group_head_and_exports_diagnostics(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "crops"
    train_root.mkdir()
    backend = Dinov2GroupTrainingBackend(
        candidate_root=train_root,
        feature_backend=FakeFeatures(),
        onnx_adapter=FakeOnnxAdapter(),
        epochs=160,
        learning_rate=0.1,
        seed=7,
    )
    checkpoint = tmp_path / "checkpoint.npz"
    metrics = tmp_path / "metrics.json"

    backend.train(
        train_records=_records(train_root, copies=3),
        valid_records=_records(train_root, copies=1),
        group_class_map=("cue", "solid", "eight", "stripe", "unknown"),
        exact_class_map=(),
        checkpoint_path=checkpoint,
        metrics_path=metrics,
    )
    backend.export_onnx(checkpoint_path=checkpoint, output_path=tmp_path / "model.onnx")
    checkpoint_logits, onnx_logits = backend.parity_logits(
        checkpoint_path=checkpoint,
        onnx_path=tmp_path / "model.onnx",
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["trained_heads"] == ["group"]
    assert payload["exact_head"] == {"status": "deferred", "trained": False}
    assert payload["group_macro_f1"] >= 0.99
    assert payload["confusion_matrix"]["labels"] == [
        "cue",
        "solid",
        "eight",
        "stripe",
        "unknown",
    ]
    assert set(checkpoint_logits) == {"group"}
    assert checkpoint_logits == onnx_logits


def test_group_backend_rejects_requested_exact_head(tmp_path: Path) -> None:
    backend = Dinov2GroupTrainingBackend(
        candidate_root=tmp_path,
        feature_backend=FakeFeatures(),
        onnx_adapter=FakeOnnxAdapter(),
    )

    try:
        backend.train(
            train_records=(),
            valid_records=(),
            group_class_map=("cue", "solid", "eight", "stripe", "unknown"),
            exact_class_map=("cue", "1"),
            checkpoint_path=tmp_path / "checkpoint.npz",
            metrics_path=tmp_path / "metrics.json",
        )
    except ValueError as exc:
        assert "exact head is deferred" in str(exc)
    else:
        raise AssertionError("group-only backend accepted an exact head")
