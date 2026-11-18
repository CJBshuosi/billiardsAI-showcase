"""Real frozen-DINOv2 training backend for the mandatory ball-group head."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from billiards_ai.lab.mainline1_identity_preprocessing import (
    normalise_identity_features as _normalise,
)


class IdentityFeatureBackend(Protocol):
    version: str

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray: ...


class LinearHeadOnnxAdapter(Protocol):
    def export(self, *, weight: np.ndarray, bias: np.ndarray, output_path: Path) -> None: ...

    def logits(
        self, *, weight: np.ndarray, bias: np.ndarray, features: np.ndarray
    ) -> np.ndarray: ...


class OnnxRuntimeLinearHeadAdapter:
    """Export and execute a single linear classifier without network access."""

    def export(self, *, weight: np.ndarray, bias: np.ndarray, output_path: Path) -> None:
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError as exc:
            raise RuntimeError("approved ONNX runtime dependencies are unavailable") from exc
        feature_count = int(weight.shape[1])
        class_count = int(weight.shape[0])
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Gemm",
                    ["features", "weight", "bias"],
                    ["group_logits"],
                    transB=1,
                )
            ],
            "billiardsAI_group_identity_head",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
            [helper.make_tensor_value_info("group_logits", TensorProto.FLOAT, [None, class_count])],
            [
                numpy_helper.from_array(weight.astype(np.float32), name="weight"),
                numpy_helper.from_array(bias.astype(np.float32), name="bias"),
            ],
        )
        model = helper.make_model(
            graph,
            producer_name="billiardsAI",
            opset_imports=[helper.make_opsetid("", 17)],
        )
        onnx.checker.check_model(model)
        output_path.write_bytes(model.SerializeToString())

    def logits(self, *, weight: np.ndarray, bias: np.ndarray, features: np.ndarray) -> np.ndarray:
        del weight, bias
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError("approved ONNX Runtime dependency is unavailable") from exc
        if not hasattr(self, "_onnx_path"):
            raise RuntimeError("ONNX model path is unavailable")
        session = onnxruntime.InferenceSession(
            str(self._onnx_path), providers=["CPUExecutionProvider"]
        )
        return np.asarray(
            session.run(["group_logits"], {"features": features.astype(np.float32)})[0]
        )

    def bind(self, path: Path) -> None:
        self._onnx_path = path.resolve()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _metrics(
    *, truth: np.ndarray, predicted: np.ndarray, class_map: tuple[str, ...]
) -> dict[str, object]:
    matrix = np.zeros((len(class_map), len(class_map)), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        matrix[int(expected), int(actual)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, label in enumerate(class_map):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    return {
        "group_macro_f1": float(np.mean(f1_values)),
        "cue_f1": per_class["cue"]["f1"],
        "eight_f1": per_class["eight"]["f1"],
        "solid_f1": per_class["solid"]["f1"],
        "stripe_f1": per_class["stripe"]["f1"],
        "unknown_f1": per_class["unknown"]["f1"],
        "per_class": per_class,
        "confusion_matrix": {"labels": list(class_map), "values": matrix.tolist()},
    }


class Dinov2GroupTrainingBackend:
    """Train a deterministic balanced linear head over frozen DINOv2 CLS features."""

    name = "dinov2-frozen-group-linear"
    version = "1.0"

    def __init__(
        self,
        *,
        candidate_root: Path,
        feature_backend: IdentityFeatureBackend,
        onnx_adapter: LinearHeadOnnxAdapter | None = None,
        epochs: int = 300,
        learning_rate: float = 0.03,
        weight_decay: float = 1e-4,
        seed: int = 20260801,
    ) -> None:
        if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("group training hyperparameters are invalid")
        self.candidate_root = candidate_root.resolve()
        self.feature_backend = feature_backend
        self.onnx_adapter = onnx_adapter or OnnxRuntimeLinearHeadAdapter()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self._parity_features: np.ndarray | None = None

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "feature_backend": self.feature_backend.version,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "weight_decay": self.weight_decay,
        }

    def _paths_and_labels(
        self, records: tuple[object, ...], class_map: tuple[str, ...]
    ) -> tuple[tuple[Path, ...], np.ndarray]:
        if not records:
            raise ValueError("group training split is empty")
        indexes = {label: index for index, label in enumerate(class_map)}
        paths = []
        labels = []
        for record in records:
            relative_path = getattr(record, "relative_path", None)
            group = getattr(record, "group", None)
            if not isinstance(relative_path, str) or group not in indexes:
                raise ValueError("group training record is invalid")
            path = (self.candidate_root / relative_path).resolve()
            if not path.is_relative_to(self.candidate_root) or not path.is_file():
                raise ValueError("group training crop is missing or unsafe")
            paths.append(path)
            labels.append(indexes[group])
        return tuple(paths), np.asarray(labels, dtype=np.int64)

    def train(
        self,
        *,
        train_records: tuple[object, ...],
        valid_records: tuple[object, ...],
        group_class_map: tuple[str, ...],
        exact_class_map: tuple[str, ...],
        checkpoint_path: Path,
        metrics_path: Path,
    ) -> None:
        expected_groups = ("cue", "solid", "eight", "stripe", "unknown")
        if group_class_map != expected_groups:
            raise ValueError("group class map is invalid")
        if exact_class_map:
            raise ValueError("exact head is deferred for this training backend")
        train_paths, train_labels = self._paths_and_labels(train_records, group_class_map)
        valid_paths, valid_labels = self._paths_and_labels(valid_records, group_class_map)
        train_features = _normalise(self.feature_backend.embed(train_paths))
        valid_features = _normalise(self.feature_backend.embed(valid_paths))
        if len(train_features) != len(train_labels) or len(valid_features) != len(valid_labels):
            raise ValueError("DINOv2 feature count differs from crop count")

        rng = np.random.default_rng(self.seed)
        class_count = len(group_class_map)
        feature_count = train_features.shape[1]
        weight = rng.normal(0, 0.01, (class_count, feature_count)).astype(np.float32)
        bias = np.zeros(class_count, dtype=np.float32)
        counts = np.bincount(train_labels, minlength=class_count).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError("group training split does not cover every class")
        class_weights = counts.sum() / (class_count * counts)
        sample_weights = class_weights[train_labels]
        sample_weights /= sample_weights.mean()

        for _ in range(self.epochs):
            probabilities = _softmax(train_features @ weight.T + bias)
            probabilities[np.arange(len(train_labels)), train_labels] -= 1.0
            gradient = probabilities * sample_weights[:, None] / len(train_labels)
            weight_gradient = gradient.T @ train_features + self.weight_decay * weight
            bias_gradient = gradient.sum(axis=0)
            weight -= self.learning_rate * weight_gradient.astype(np.float32)
            bias -= self.learning_rate * bias_gradient.astype(np.float32)

        predicted = np.argmax(valid_features @ weight.T + bias, axis=1)
        payload = {
            "schema_version": "1.0",
            "trained_heads": ["group"],
            "exact_head": {"status": "deferred", "trained": False},
            "forced_exact_without_evidence": 0,
            "feature_backend": self.feature_backend.version,
            "train_count": len(train_records),
            "valid_count": len(valid_records),
            **_metrics(truth=valid_labels, predicted=predicted, class_map=group_class_map),
        }
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        self._parity_features = valid_features[:1].copy()
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                weight=weight,
                bias=bias,
                class_map=np.asarray(group_class_map),
                parity_features=self._parity_features,
            )

    @staticmethod
    def _checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            return (
                np.asarray(payload["weight"], dtype=np.float32),
                np.asarray(payload["bias"], dtype=np.float32),
                np.asarray(payload["parity_features"], dtype=np.float32),
            )

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        weight, bias, _ = self._checkpoint(checkpoint_path)
        self.onnx_adapter.export(weight=weight, bias=bias, output_path=output_path)
        bind = getattr(self.onnx_adapter, "bind", None)
        if callable(bind):
            bind(output_path)

    def parity_logits(
        self, *, checkpoint_path: Path, onnx_path: Path
    ) -> tuple[dict[str, Sequence[float]], dict[str, Sequence[float]]]:
        weight, bias, features = self._checkpoint(checkpoint_path)
        bind = getattr(self.onnx_adapter, "bind", None)
        if callable(bind):
            bind(onnx_path)
        checkpoint_logits = features @ weight.T + bias
        onnx_logits = self.onnx_adapter.logits(weight=weight, bias=bias, features=features)
        return (
            {"group": tuple(float(value) for value in checkpoint_logits[0])},
            {"group": tuple(float(value) for value in onnx_logits[0])},
        )
