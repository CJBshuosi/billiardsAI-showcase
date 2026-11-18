from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from billiards_ai.lab.training.identity_hierarchical_backend import (
    Dinov2HierarchicalGroupTrainingBackend,
    OnnxRuntimeHierarchicalAdapter,
    PillowIdentityEvidenceDegrader,
    calibrate_known_probability_min,
    combine_hierarchical_probabilities,
    predict_groups_with_rejection,
)


class FakeFeatures:
    version = "fake-dinov2"

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray:
        vectors = []
        for crop in crops:
            group_index = int(crop.stem.split("-")[0])
            vector = np.zeros(4, dtype=np.float32)
            if group_index < 4:
                vector[group_index] = 1.0
            vectors.append(vector)
        return np.stack(vectors)


class FakeOnnxAdapter:
    def export(self, *, parameters: dict[str, np.ndarray], output_path: Path) -> None:
        assert parameters["special_w2"].shape[1] == 3
        assert parameters["pattern_w2"].shape[1] == 2
        assert parameters["known_w2"].shape[1] == 2
        output_path.write_bytes(b"hierarchical-onnx")

    def probabilities(
        self, *, parameters: dict[str, np.ndarray], features: np.ndarray
    ) -> np.ndarray:
        special_hidden = np.maximum(
            features @ parameters["special_w1"] + parameters["special_b1"], 0
        )
        pattern_hidden = np.maximum(
            features @ parameters["pattern_w1"] + parameters["pattern_b1"], 0
        )
        return combine_hierarchical_probabilities(
            special_logits=special_hidden @ parameters["special_w2"] + parameters["special_b2"],
            pattern_logits=pattern_hidden @ parameters["pattern_w2"] + parameters["pattern_b2"],
        )

    def known_probabilities(
        self, *, parameters: dict[str, np.ndarray], features: np.ndarray
    ) -> np.ndarray:
        hidden = np.maximum(features @ parameters["known_w1"] + parameters["known_b1"], 0)
        logits = hidden @ parameters["known_w2"] + parameters["known_b2"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities[:, 1:2] / probabilities.sum(axis=1, keepdims=True)


class FakeEvidence:
    version = "fake-evidence"
    feature_names = ("white_ratio", "black_ratio")

    def extract(self, crops: tuple[Path, ...]) -> np.ndarray:
        return np.zeros((len(crops), 2), dtype=np.float32)


class FakeAugmenter:
    def augment(
        self, *, crops: tuple[Path, ...], variant: int, output_root: Path
    ) -> tuple[Path, ...]:
        augmented = []
        for crop in crops:
            target = output_root / f"{crop.stem}--aug-{variant}.png"
            target.write_bytes(crop.read_bytes())
            augmented.append(target)
        return tuple(augmented)


class FakeDegrader:
    def degrade(self, *, crops: tuple[Path, ...], output_root: Path) -> tuple[Path, ...]:
        output_root.mkdir(parents=True, exist_ok=True)
        degraded = []
        for crop in crops:
            target = output_root / f"4-{crop.stem}--degraded.png"
            target.write_bytes(crop.read_bytes())
            degraded.append(target)
        return tuple(degraded)


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
                    source_asset_id=f"asset-{copy}",
                )
            )
    return tuple(records)


def test_hierarchy_routes_object_probability_through_solid_stripe_head() -> None:
    special_logits = np.asarray(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
            [0.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )
    pattern_logits = np.asarray(
        [
            [0.0, 0.0],
            [8.0, 0.0],
            [0.0, 0.0],
            [0.0, 8.0],
        ],
        dtype=np.float32,
    )

    probabilities = combine_hierarchical_probabilities(
        special_logits=special_logits,
        pattern_logits=pattern_logits,
    )

    assert probabilities.shape == (4, 4)
    assert np.argmax(probabilities, axis=1).tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_low_confidence_or_small_margin_becomes_unknown_not_a_trained_class() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.04, 0.03, 0.03],
            [0.40, 0.35, 0.15, 0.10],
            [0.48, 0.47, 0.03, 0.02],
        ],
        dtype=np.float32,
    )

    predicted = predict_groups_with_rejection(
        probabilities,
        confidence_min=0.50,
        margin_min=0.10,
    )

    assert predicted.tolist() == [0, 4, 4]


def test_independent_knownness_head_can_reject_an_otherwise_confident_group() -> None:
    probabilities = np.asarray(
        [[0.95, 0.02, 0.02, 0.01], [0.95, 0.02, 0.02, 0.01]],
        dtype=np.float32,
    )

    predicted = predict_groups_with_rejection(
        probabilities,
        confidence_min=0.50,
        margin_min=0.10,
        known_probabilities=np.asarray([0.90, 0.20], dtype=np.float32),
        known_probability_min=0.55,
    )

    assert predicted.tolist() == [0, 4]


def test_train_only_calibration_selects_threshold_that_rejects_unknown() -> None:
    probabilities = np.asarray(
        [
            [0.95, 0.02, 0.02, 0.01],
            [0.02, 0.95, 0.02, 0.01],
            [0.95, 0.02, 0.02, 0.01],
            [0.02, 0.95, 0.02, 0.01],
        ],
        dtype=np.float32,
    )

    threshold, summary = calibrate_known_probability_min(
        probabilities=probabilities,
        known_probabilities=np.asarray([0.90, 0.80, 0.40, 0.20], dtype=np.float32),
        truth=np.asarray([0, 1, 4, 4], dtype=np.int64),
        configured_min=0.30,
        confidence_min=0.50,
        margin_min=0.10,
    )

    assert threshold == 0.45
    assert summary["source"] == "train_asset_holdout"
    assert summary["unknown_support"] == 2
    assert summary["selected_unknown_f1"] == 1.0


def test_train_only_calibration_falls_back_with_only_one_unknown() -> None:
    threshold, summary = calibrate_known_probability_min(
        probabilities=np.asarray(
            [[0.95, 0.02, 0.02, 0.01], [0.95, 0.02, 0.02, 0.01]],
            dtype=np.float32,
        ),
        known_probabilities=np.asarray([0.90, 0.20], dtype=np.float32),
        truth=np.asarray([0, 4], dtype=np.int64),
        configured_min=0.55,
        confidence_min=0.50,
        margin_min=0.10,
    )

    assert threshold == 0.55
    assert summary["source"] == "configured_fallback"
    assert summary["unknown_support"] == 1


def test_evidence_degrader_creates_occluded_blurred_crop(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    crop = tmp_path / "crop.png"
    image = Image.new("RGB", (64, 64), (20, 120, 20))
    ImageDraw.Draw(image).ellipse((16, 16, 48, 48), fill=(240, 240, 240))
    image.save(crop)

    output = PillowIdentityEvidenceDegrader().degrade(
        crops=(crop,), output_root=tmp_path / "degraded"
    )

    assert len(output) == 1
    assert output[0].is_file()
    assert output[0].read_bytes() != crop.read_bytes()


def test_hierarchical_backend_trains_known_groups_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    crop_root.mkdir()
    backend = Dinov2HierarchicalGroupTrainingBackend(
        candidate_root=crop_root,
        feature_backend=FakeFeatures(),
        evidence_backend=FakeEvidence(),
        onnx_adapter=FakeOnnxAdapter(),
        augmenter=FakeAugmenter(),
        evidence_degrader=FakeDegrader(),
        hidden_size=12,
        epochs=400,
        learning_rate=0.03,
        augmentation_count=1,
        known_probability_min=0.55,
        hard_negative_focal_gamma=2.0,
        object_hard_negative_weight=1.5,
        confidence_min=0.50,
        margin_min=0.10,
        seed=7,
    )
    checkpoint = tmp_path / "checkpoint.npz"
    metrics = tmp_path / "metrics.json"

    backend.train(
        train_records=_records(crop_root, copies=4),
        valid_records=_records(crop_root, copies=1),
        group_class_map=("cue", "solid", "eight", "stripe", "unknown"),
        exact_class_map=(),
        checkpoint_path=checkpoint,
        metrics_path=metrics,
    )
    backend.export_onnx(checkpoint_path=checkpoint, output_path=tmp_path / "model.onnx")
    native, exported = backend.parity_logits(
        checkpoint_path=checkpoint,
        onnx_path=tmp_path / "model.onnx",
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["architecture"] == "hierarchical_fused_evidence_mlp"
    assert payload["schema_version"] == "5.0"
    assert payload["visual_evidence_policy"] == {
        "backend": "fake-evidence",
        "feature_names": ["white_ratio", "black_ratio"],
        "weight": 0.5,
    }
    assert payload["unknown_policy"]["trained_class"] is False
    assert payload["unknown_policy"]["trained_rejection_head"] is True
    assert payload["unknown_policy"]["head"] == "binary_knownness_mlp"
    assert 0.30 <= payload["unknown_policy"]["known_probability_min"] <= 0.90
    assert payload["unknown_policy"]["calibration"]["source"] == "train_asset_holdout"
    assert payload["unknown_policy"]["calibration"]["unknown_support"] >= 2
    assert payload["augmented_train_count"] == 32
    assert payload["unknown_rejection_train_count"] > 40
    assert payload["insufficient_evidence_train_count"] > 0
    assert payload["hard_negative_policy"]["source"] == "train_only_online_focal"
    assert payload["hard_negative_policy"]["focal_gamma"] == 2.0
    assert payload["hard_negative_policy"]["object_weight"] == 1.5
    assert payload["group_macro_f1"] >= 0.99
    assert payload["unknown_f1"] >= 0.99
    assert backend.parity_class_maps == {
        "group": ("cue", "solid", "eight", "stripe"),
        "known": ("known",),
    }
    assert native == exported

    predictions = backend.predict_records(
        checkpoint_path=checkpoint,
        records=_records(crop_root, copies=1),
    )
    assert len(predictions) == 5
    assert {item["truth"] for item in predictions} == {
        "cue",
        "solid",
        "eight",
        "stripe",
        "unknown",
    }
    assert all(len(item["probabilities"]) == 4 for item in predictions)
    assert all(0 <= item["known_probability"] <= 1 for item in predictions)
    assert predictions[-1]["predicted"] == "unknown"


def test_hierarchical_backend_uses_real_onnx_adapter_by_default(tmp_path: Path) -> None:
    backend = Dinov2HierarchicalGroupTrainingBackend(
        candidate_root=tmp_path,
        feature_backend=FakeFeatures(),
        evidence_backend=FakeEvidence(),
    )

    assert isinstance(backend.onnx_adapter, OnnxRuntimeHierarchicalAdapter)


def test_checkpoint_exposes_only_mlp_parameters_to_onnx_export(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    parameter_names = {
        f"{prefix}_{suffix}"
        for prefix in ("special", "pattern", "known")
        for suffix in ("w1", "b1", "w2", "b2")
    }
    with checkpoint.open("wb") as target:
        np.savez_compressed(
            target,
            **{name: np.ones((1,), dtype=np.float32) for name in parameter_names},
            known_probability_min=np.asarray([0.55], dtype=np.float32),
            confidence_min=np.asarray([0.55], dtype=np.float32),
            margin_min=np.asarray([0.10], dtype=np.float32),
            parity_features=np.ones((1, 4), dtype=np.float32),
        )

    parameters, _ = Dinov2HierarchicalGroupTrainingBackend._checkpoint(checkpoint)

    assert set(parameters) == parameter_names


def test_frozen_validation_features_do_not_change_trained_parameters(tmp_path: Path) -> None:
    crop_root = tmp_path / "crops"
    crop_root.mkdir()
    train_records = _records(crop_root, copies=3)
    valid_records = _records(crop_root, copies=1)
    shifted_valid = []
    for record in valid_records:
        shifted_path = f"{record.relative_path[:-4]}-shifted.png"
        (crop_root / shifted_path).write_bytes(b"shifted")
        shifted_valid.append(
            SimpleNamespace(
                crop_id=f"{record.crop_id}-shifted",
                group=record.group,
                relative_path=shifted_path,
            )
        )

    class ShiftAwareFeatures(FakeFeatures):
        def embed(self, crops: tuple[Path, ...]) -> np.ndarray:
            features = super().embed(crops)
            for index, crop in enumerate(crops):
                if "shifted" in crop.stem:
                    features[index] = np.roll(features[index], 1)
            return features

    backend = Dinov2HierarchicalGroupTrainingBackend(
        candidate_root=crop_root,
        feature_backend=ShiftAwareFeatures(),
        evidence_backend=FakeEvidence(),
        onnx_adapter=FakeOnnxAdapter(),
        augmenter=FakeAugmenter(),
        evidence_degrader=FakeDegrader(),
        hidden_size=8,
        epochs=80,
        learning_rate=0.03,
        augmentation_count=0,
        seed=19,
    )
    checkpoints = (tmp_path / "original.npz", tmp_path / "shifted.npz")
    for checkpoint, validation in zip(
        checkpoints, (valid_records, tuple(shifted_valid)), strict=True
    ):
        backend.train(
            train_records=train_records,
            valid_records=validation,
            group_class_map=("cue", "solid", "eight", "stripe", "unknown"),
            exact_class_map=(),
            checkpoint_path=checkpoint,
            metrics_path=checkpoint.with_suffix(".json"),
        )

    with (
        np.load(checkpoints[0], allow_pickle=False) as original,
        np.load(checkpoints[1], allow_pickle=False) as shifted,
    ):
        parameter_names = [
            name for name in original.files if name.startswith(("special_", "pattern_", "known_"))
        ]
        assert parameter_names
        for name in parameter_names:
            np.testing.assert_array_equal(original[name], shifted[name])
