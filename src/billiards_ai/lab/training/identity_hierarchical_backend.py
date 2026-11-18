"""Hierarchical group classification over frozen visual features."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import numpy as np

from billiards_ai.lab.training.identity_group_backend import _metrics, _normalise
from billiards_ai.lab.training.identity_visual_evidence import (
    OpenCvIdentityVisualEvidenceBackend,
)


class IdentityFeatureBackend(Protocol):
    version: str

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray: ...


class IdentityVisualEvidenceBackend(Protocol):
    version: str
    feature_names: tuple[str, ...]

    def extract(self, crops: tuple[Path, ...]) -> np.ndarray: ...


class IdentityCropAugmenter(Protocol):
    def augment(
        self, *, crops: tuple[Path, ...], variant: int, output_root: Path
    ) -> tuple[Path, ...]: ...


class IdentityEvidenceDegrader(Protocol):
    def degrade(self, *, crops: tuple[Path, ...], output_root: Path) -> tuple[Path, ...]: ...


class PillowIdentityCropAugmenter:
    """Create deterministic lighting, blur, and rotation variants."""

    def augment(
        self, *, crops: tuple[Path, ...], variant: int, output_root: Path
    ) -> tuple[Path, ...]:
        try:
            from PIL import Image, ImageEnhance, ImageFilter
        except ImportError as exc:
            raise RuntimeError("approved Pillow dependency is unavailable") from exc
        augmented = []
        for index, path in enumerate(crops):
            with Image.open(path) as source:
                image = source.convert("RGB")
                brightness = 0.78 if variant % 2 == 0 else 1.22
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(1.12)
                if variant % 2:
                    image = image.filter(ImageFilter.GaussianBlur(radius=0.55))
                angle = (-6 if index % 2 else 6) if variant >= 1 else 0
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
                target = output_root / f"{path.stem}--aug-{variant}.png"
                image.save(target, format="PNG")
                augmented.append(target)
        return tuple(augmented)


class PillowIdentityEvidenceDegrader:
    """Create deterministic evidence-poor crops for the knownness head only."""

    def degrade(self, *, crops: tuple[Path, ...], output_root: Path) -> tuple[Path, ...]:
        try:
            from PIL import Image, ImageDraw, ImageFilter, ImageStat
        except ImportError as exc:
            raise RuntimeError("approved Pillow dependency is unavailable") from exc
        output_root.mkdir(parents=True, exist_ok=True)
        degraded = []
        for crop in crops:
            with Image.open(crop) as source:
                image = source.convert("RGB").filter(ImageFilter.GaussianBlur(radius=3.0))
                mean = tuple(int(value) for value in ImageStat.Stat(image).mean)
                width, height = image.size
                draw = ImageDraw.Draw(image)
                draw.rectangle(
                    (int(width * 0.22), 0, int(width * 0.78), height),
                    fill=mean,
                )
                target = output_root / f"{crop.stem}--evidence-poor.png"
                image.save(target, format="PNG")
                degraded.append(target)
        return tuple(degraded)


class HierarchicalOnnxAdapter(Protocol):
    def export(self, *, parameters: dict[str, np.ndarray], output_path: Path) -> None: ...

    def probabilities(
        self, *, parameters: dict[str, np.ndarray], features: np.ndarray
    ) -> np.ndarray: ...

    def known_probabilities(
        self, *, parameters: dict[str, np.ndarray], features: np.ndarray
    ) -> np.ndarray: ...


class OnnxRuntimeHierarchicalAdapter:
    """Export hierarchical group heads plus an independent knownness head."""

    def export(self, *, parameters: dict[str, np.ndarray], output_path: Path) -> None:
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError as exc:
            raise RuntimeError("approved ONNX runtime dependencies are unavailable") from exc
        feature_count = int(parameters["special_w1"].shape[0])
        nodes = []
        for prefix in ("special", "pattern", "known"):
            nodes.extend(
                (
                    helper.make_node("MatMul", ["features", f"{prefix}_w1"], [f"{prefix}_linear1"]),
                    helper.make_node(
                        "Add", [f"{prefix}_linear1", f"{prefix}_b1"], [f"{prefix}_pre"]
                    ),
                    helper.make_node("Relu", [f"{prefix}_pre"], [f"{prefix}_hidden"]),
                    helper.make_node(
                        "MatMul",
                        [f"{prefix}_hidden", f"{prefix}_w2"],
                        [f"{prefix}_linear2"],
                    ),
                    helper.make_node(
                        "Add", [f"{prefix}_linear2", f"{prefix}_b2"], [f"{prefix}_logits"]
                    ),
                    helper.make_node(
                        "Softmax", [f"{prefix}_logits"], [f"{prefix}_probabilities"], axis=1
                    ),
                )
            )
        nodes.extend(
            (
                helper.make_node(
                    "Split",
                    ["special_probabilities"],
                    ["cue_probability", "object_probability", "eight_probability"],
                    axis=1,
                ),
                helper.make_node(
                    "Split",
                    ["pattern_probabilities"],
                    ["solid_pattern_probability", "stripe_pattern_probability"],
                    axis=1,
                ),
                helper.make_node(
                    "Mul",
                    ["object_probability", "solid_pattern_probability"],
                    ["solid_probability"],
                ),
                helper.make_node(
                    "Mul",
                    ["object_probability", "stripe_pattern_probability"],
                    ["stripe_probability"],
                ),
                helper.make_node(
                    "Concat",
                    [
                        "cue_probability",
                        "solid_probability",
                        "eight_probability",
                        "stripe_probability",
                    ],
                    ["group_probabilities"],
                    axis=1,
                ),
                helper.make_node(
                    "Split",
                    ["known_probabilities"],
                    ["unknown_probability", "known_probability"],
                    axis=1,
                ),
            )
        )
        graph = helper.make_graph(
            nodes,
            "billiardsAI_hierarchical_group_identity",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
            [
                helper.make_tensor_value_info("group_probabilities", TensorProto.FLOAT, [None, 4]),
                helper.make_tensor_value_info("known_probability", TensorProto.FLOAT, [None, 1]),
            ],
            [
                numpy_helper.from_array(value.astype(np.float32), name=name)
                for name, value in sorted(parameters.items())
            ],
        )
        model = helper.make_model(
            graph,
            producer_name="billiardsAI",
            opset_imports=[helper.make_opsetid("", 17)],
        )
        onnx.checker.check_model(model)
        output_path.write_bytes(model.SerializeToString())

    def bind(self, path: Path) -> None:
        self._onnx_path = path.resolve()

    def probabilities(
        self, *, parameters: dict[str, np.ndarray], features: np.ndarray
    ) -> np.ndarray:
        del parameters
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
            session.run(["group_probabilities"], {"features": features.astype(np.float32)})[0]
        )

    def known_probabilities(
        self, *, parameters: dict[str, np.ndarray], features: np.ndarray
    ) -> np.ndarray:
        del parameters
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
            session.run(["known_probability"], {"features": features.astype(np.float32)})[0]
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("hierarchical logits are invalid")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def combine_hierarchical_probabilities(
    *, special_logits: np.ndarray, pattern_logits: np.ndarray
) -> np.ndarray:
    """Return cue/solid/eight/stripe probabilities from two trained heads."""

    if (
        special_logits.ndim != 2
        or pattern_logits.ndim != 2
        or special_logits.shape != (pattern_logits.shape[0], 3)
        or pattern_logits.shape[1] != 2
    ):
        raise ValueError("hierarchical head shapes are invalid")
    special = _softmax(np.asarray(special_logits, dtype=np.float32))
    pattern = _softmax(np.asarray(pattern_logits, dtype=np.float32))
    return np.stack(
        (
            special[:, 0],
            special[:, 1] * pattern[:, 0],
            special[:, 2],
            special[:, 1] * pattern[:, 1],
        ),
        axis=1,
    )


def predict_groups_with_rejection(
    probabilities: np.ndarray,
    *,
    confidence_min: float,
    margin_min: float,
    known_probabilities: np.ndarray | None = None,
    known_probability_min: float = 0.0,
) -> np.ndarray:
    """Map four known groups to unknown index 4 when evidence is weak."""

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 4
        or not 0 <= confidence_min <= 1
        or not 0 <= margin_min <= 1
        or not 0 <= known_probability_min <= 1
        or (
            known_probabilities is not None
            and np.asarray(known_probabilities).reshape(-1).shape != (len(probabilities),)
        )
    ):
        raise ValueError("group rejection inputs are invalid")
    ordered = np.sort(probabilities, axis=1)
    maximum = ordered[:, -1]
    margin = maximum - ordered[:, -2]
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    rejected = (maximum < confidence_min) | (margin < margin_min)
    if known_probabilities is not None:
        rejected |= np.asarray(known_probabilities).reshape(-1) < known_probability_min
    predicted[rejected] = 4
    return predicted


def calibrate_known_probability_min(
    *,
    probabilities: np.ndarray,
    known_probabilities: np.ndarray,
    truth: np.ndarray,
    configured_min: float,
    confidence_min: float,
    margin_min: float,
    min_unknown_support: int = 2,
) -> tuple[float, dict[str, object]]:
    """Select a knownness threshold from a training-asset holdout only."""

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 4
        or known_probabilities.reshape(-1).shape != (len(probabilities),)
        or truth.shape != (len(probabilities),)
        or not 0 <= configured_min <= 1
        or min_unknown_support < 2
    ):
        raise ValueError("knownness calibration inputs are invalid")
    unknown_support = int(np.sum(truth == 4))
    known_support = int(np.sum(truth != 4))
    if unknown_support < min_unknown_support or known_support < 1:
        return configured_min, {
            "source": "configured_fallback",
            "known_support": known_support,
            "unknown_support": unknown_support,
            "selected_unknown_f1": 0.0,
        }
    candidates = sorted(
        {round(float(value), 2) for value in np.arange(0.30, 0.901, 0.05)}
        | {round(configured_min, 2)}
    )
    scored = []
    class_map = ("cue", "solid", "eight", "stripe", "unknown")
    for threshold in candidates:
        predicted = predict_groups_with_rejection(
            probabilities,
            confidence_min=confidence_min,
            margin_min=margin_min,
            known_probabilities=known_probabilities,
            known_probability_min=threshold,
        )
        metrics = _metrics(truth=truth, predicted=predicted, class_map=class_map)
        all_macro = float(np.mean([metrics["per_class"][label]["f1"] for label in class_map]))
        unknown_f1 = float(metrics["per_class"]["unknown"]["f1"])
        known_macro = float(
            np.mean([metrics["per_class"][label]["f1"] for label in class_map[:-1]])
        )
        scored.append(
            (
                all_macro,
                unknown_f1,
                known_macro,
                -abs(threshold - configured_min),
                -threshold,
                threshold,
            )
        )
    selected = max(scored)
    return selected[-1], {
        "source": "train_asset_holdout",
        "known_support": known_support,
        "unknown_support": unknown_support,
        "selected_macro_f1": selected[0],
        "selected_unknown_f1": selected[1],
        "candidate_count": len(candidates),
    }


def _forward_mlp(
    features: np.ndarray, parameters: dict[str, np.ndarray], prefix: str
) -> np.ndarray:
    hidden = np.maximum(features @ parameters[f"{prefix}_w1"] + parameters[f"{prefix}_b1"], 0)
    return hidden @ parameters[f"{prefix}_w2"] + parameters[f"{prefix}_b2"]


def _train_mlp(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    prefix: str,
    rejection_features: np.ndarray | None = None,
    rejection_weight: float = 0.0,
    class_multipliers: np.ndarray | None = None,
    focal_gamma: float = 0.0,
) -> dict[str, np.ndarray]:
    if len(features) != len(labels) or len(features) < 1:
        raise ValueError("hierarchical training split is invalid")
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("hierarchical training split does not cover every class")
    rng = np.random.default_rng(seed)
    feature_count = features.shape[1]
    parameters = {
        f"{prefix}_w1": rng.normal(
            0, np.sqrt(2.0 / feature_count), (feature_count, hidden_size)
        ).astype(np.float32),
        f"{prefix}_b1": np.zeros(hidden_size, dtype=np.float32),
        f"{prefix}_w2": rng.normal(
            0, np.sqrt(2.0 / hidden_size), (hidden_size, class_count)
        ).astype(np.float32),
        f"{prefix}_b2": np.zeros(class_count, dtype=np.float32),
    }
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    if (
        rejection_weight < 0
        or focal_gamma < 0
        or (
            rejection_features is not None
            and (rejection_features.ndim != 2 or rejection_features.shape[1] != features.shape[1])
        )
    ):
        raise ValueError("hierarchical rejection training inputs are invalid")
    if class_multipliers is None:
        class_multipliers = np.ones(class_count, dtype=np.float32)
    if class_multipliers.shape != (class_count,) or np.any(class_multipliers <= 0):
        raise ValueError("hierarchical class multipliers are invalid")
    class_weights = counts.sum() / (class_count * counts)
    class_weights *= class_multipliers
    sample_weights = class_weights[labels]
    sample_weights /= sample_weights.mean()
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    for epoch in range(1, epochs + 1):
        hidden_pre = features @ parameters[f"{prefix}_w1"] + parameters[f"{prefix}_b1"]
        hidden = np.maximum(hidden_pre, 0)
        logits = hidden @ parameters[f"{prefix}_w2"] + parameters[f"{prefix}_b2"]
        probabilities = _softmax(logits)
        focal_weights = (1.0 - probabilities[np.arange(len(labels)), labels]) ** focal_gamma
        gradient_logits = probabilities
        gradient_logits[np.arange(len(labels)), labels] -= 1.0
        effective_weights = sample_weights * focal_weights
        effective_weights /= max(float(effective_weights.mean()), 1e-8)
        gradient_logits *= effective_weights[:, None] / len(labels)
        gradients = {
            f"{prefix}_w2": hidden.T @ gradient_logits + weight_decay * parameters[f"{prefix}_w2"],
            f"{prefix}_b2": gradient_logits.sum(axis=0),
        }
        gradient_hidden = gradient_logits @ parameters[f"{prefix}_w2"].T
        gradient_hidden[hidden_pre <= 0] = 0
        gradients[f"{prefix}_w1"] = (
            features.T @ gradient_hidden + weight_decay * parameters[f"{prefix}_w1"]
        )
        gradients[f"{prefix}_b1"] = gradient_hidden.sum(axis=0)
        if rejection_features is not None and len(rejection_features) and rejection_weight:
            rejection_pre = (
                rejection_features @ parameters[f"{prefix}_w1"] + parameters[f"{prefix}_b1"]
            )
            rejection_hidden = np.maximum(rejection_pre, 0)
            rejection_logits = (
                rejection_hidden @ parameters[f"{prefix}_w2"] + parameters[f"{prefix}_b2"]
            )
            rejection_gradient = _softmax(rejection_logits) - (1.0 / class_count)
            rejection_gradient *= rejection_weight / len(rejection_features)
            gradients[f"{prefix}_w2"] += rejection_hidden.T @ rejection_gradient
            gradients[f"{prefix}_b2"] += rejection_gradient.sum(axis=0)
            rejection_hidden_gradient = rejection_gradient @ parameters[f"{prefix}_w2"].T
            rejection_hidden_gradient[rejection_pre <= 0] = 0
            gradients[f"{prefix}_w1"] += rejection_features.T @ rejection_hidden_gradient
            gradients[f"{prefix}_b1"] += rejection_hidden_gradient.sum(axis=0)
        for name in sorted(parameters):
            gradient = gradients[name].astype(np.float32)
            first_moment[name] = beta1 * first_moment[name] + (1 - beta1) * gradient
            second_moment[name] = beta2 * second_moment[name] + (1 - beta2) * gradient**2
            corrected_first = first_moment[name] / (1 - beta1**epoch)
            corrected_second = second_moment[name] / (1 - beta2**epoch)
            parameters[name] -= (
                learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
            )
    return parameters


class Dinov2HierarchicalGroupTrainingBackend:
    """Train group hierarchy plus a separate known-vs-unknown rejection head."""

    name = "dinov2-frozen-hierarchical-group-mlp"
    version = "5.0"
    parity_class_map = ("cue", "solid", "eight", "stripe")
    parity_class_maps = {
        "group": ("cue", "solid", "eight", "stripe"),
        "known": ("known",),
    }

    def __init__(
        self,
        *,
        candidate_root: Path,
        feature_backend: IdentityFeatureBackend,
        evidence_backend: IdentityVisualEvidenceBackend | None = None,
        onnx_adapter: HierarchicalOnnxAdapter | None = None,
        augmenter: IdentityCropAugmenter | None = None,
        evidence_degrader: IdentityEvidenceDegrader | None = None,
        hidden_size: int = 64,
        epochs: int = 500,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        augmentation_count: int = 2,
        confidence_min: float = 0.55,
        margin_min: float = 0.10,
        known_probability_min: float = 0.55,
        hard_negative_focal_gamma: float = 2.0,
        object_hard_negative_weight: float = 1.5,
        calibration_min_unknown_support: int = 2,
        visual_evidence_weight: float = 0.5,
        seed: int = 20260801,
    ) -> None:
        if (
            hidden_size < 2
            or epochs < 1
            or learning_rate <= 0
            or weight_decay < 0
            or augmentation_count < 0
            or augmentation_count > 4
            or not 0 <= confidence_min <= 1
            or not 0 <= margin_min <= 1
            or not 0 <= known_probability_min <= 1
            or hard_negative_focal_gamma < 0
            or object_hard_negative_weight < 1
            or calibration_min_unknown_support < 2
            or visual_evidence_weight <= 0
        ):
            raise ValueError("hierarchical training configuration is invalid")
        self.candidate_root = candidate_root.resolve()
        self.feature_backend = feature_backend
        self.evidence_backend = evidence_backend or OpenCvIdentityVisualEvidenceBackend()
        self.onnx_adapter = onnx_adapter or OnnxRuntimeHierarchicalAdapter()
        self.augmenter = augmenter or PillowIdentityCropAugmenter()
        self.evidence_degrader = evidence_degrader or PillowIdentityEvidenceDegrader()
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.augmentation_count = augmentation_count
        self.confidence_min = confidence_min
        self.margin_min = margin_min
        self.known_probability_min = known_probability_min
        self.hard_negative_focal_gamma = hard_negative_focal_gamma
        self.object_hard_negative_weight = object_hard_negative_weight
        self.calibration_min_unknown_support = calibration_min_unknown_support
        self.visual_evidence_weight = visual_evidence_weight
        self.seed = seed

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "architecture": "hierarchical_fused_evidence_mlp",
            "augmentation_count": self.augmentation_count,
            "confidence_min": self.confidence_min,
            "epochs": self.epochs,
            "feature_backend": self.feature_backend.version,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "margin_min": self.margin_min,
            "known_probability_min": self.known_probability_min,
            "hard_negative_focal_gamma": self.hard_negative_focal_gamma,
            "object_hard_negative_weight": self.object_hard_negative_weight,
            "calibration_min_unknown_support": self.calibration_min_unknown_support,
            "visual_evidence_backend": self.evidence_backend.version,
            "visual_evidence_features": list(self.evidence_backend.feature_names),
            "visual_evidence_weight": self.visual_evidence_weight,
            "seed": self.seed,
            "weight_decay": self.weight_decay,
        }

    def _paths_and_groups(
        self, records: tuple[object, ...]
    ) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        if not records:
            raise ValueError("hierarchical training split is empty")
        paths = []
        groups = []
        allowed = {"cue", "solid", "eight", "stripe", "unknown"}
        for record in records:
            relative_path = getattr(record, "relative_path", None)
            group = getattr(record, "group", None)
            if not isinstance(relative_path, str) or group not in allowed:
                raise ValueError("hierarchical training record is invalid")
            path = (self.candidate_root / relative_path).resolve()
            if not path.is_relative_to(self.candidate_root) or not path.is_file():
                raise ValueError("hierarchical training crop is missing or unsafe")
            paths.append(path)
            groups.append(group)
        return tuple(paths), tuple(groups)

    def _embed(self, paths: tuple[Path, ...]) -> np.ndarray:
        semantic = _normalise(self.feature_backend.embed(paths))
        evidence = np.asarray(self.evidence_backend.extract(paths), dtype=np.float32)
        if (
            evidence.ndim != 2
            or evidence.shape != (len(paths), len(self.evidence_backend.feature_names))
            or not np.all(np.isfinite(evidence))
            or np.any(evidence < 0)
            or np.any(evidence > 1)
        ):
            raise ValueError("identity visual evidence is invalid")
        centred_evidence = (evidence - 0.5) * self.visual_evidence_weight
        return _normalise(np.concatenate((semantic, centred_evidence), axis=1))

    def _training_features(
        self, paths: tuple[Path, ...], groups: tuple[str, ...]
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int]:
        known = [
            (path, group) for path, group in zip(paths, groups, strict=True) if group != "unknown"
        ]
        known_paths = tuple(item[0] for item in known)
        known_groups = tuple(item[1] for item in known)
        unknown_paths = tuple(
            path for path, group in zip(paths, groups, strict=True) if group == "unknown"
        )
        if not known_paths or not unknown_paths:
            raise ValueError("hierarchical training requires known and unknown crops")
        features = [self._embed(known_paths)]
        rejection_features = [self._embed(unknown_paths)]
        labels = list(known_groups)
        if self.augmentation_count:
            with TemporaryDirectory(prefix="billiards-identity-augment-") as directory:
                temporary = Path(directory)
                for variant in range(self.augmentation_count):
                    augmented_paths = self.augmenter.augment(
                        crops=known_paths,
                        variant=variant,
                        output_root=temporary,
                    )
                    if len(augmented_paths) != len(known_paths):
                        raise ValueError("identity augmentation output count is invalid")
                    features.append(self._embed(augmented_paths))
                    labels.extend(known_groups)
                    augmented_unknown = self.augmenter.augment(
                        crops=unknown_paths,
                        variant=variant,
                        output_root=temporary,
                    )
                    if len(augmented_unknown) != len(unknown_paths):
                        raise ValueError("identity augmentation output count is invalid")
                    rejection_features.append(self._embed(augmented_unknown))
                degraded = self.evidence_degrader.degrade(
                    crops=known_paths,
                    output_root=temporary / "evidence-poor",
                )
                if len(degraded) != len(known_paths):
                    raise ValueError("identity evidence degradation output count is invalid")
                rejection_features.append(self._embed(degraded))
                degraded_count = len(degraded)
        else:
            with TemporaryDirectory(prefix="billiards-identity-degrade-") as directory:
                degraded = self.evidence_degrader.degrade(
                    crops=known_paths,
                    output_root=Path(directory),
                )
                if len(degraded) != len(known_paths):
                    raise ValueError("identity evidence degradation output count is invalid")
                rejection_features.append(self._embed(degraded))
                degraded_count = len(degraded)
        return (
            np.concatenate(features, axis=0),
            tuple(labels),
            np.concatenate(rejection_features, axis=0),
            degraded_count,
        )

    def _fit_parameters(
        self, paths: tuple[Path, ...], groups: tuple[str, ...]
    ) -> tuple[dict[str, np.ndarray], int, int, int]:
        train_features, augmented_groups, unknown_features, degraded_count = (
            self._training_features(paths, groups)
        )
        special_labels = np.asarray(
            [0 if group == "cue" else 2 if group == "eight" else 1 for group in augmented_groups],
            dtype=np.int64,
        )
        pattern_mask = np.asarray(
            [group in {"solid", "stripe"} for group in augmented_groups], dtype=bool
        )
        pattern_labels = np.asarray(
            [0 if group == "solid" else 1 for group in np.asarray(augmented_groups)[pattern_mask]],
            dtype=np.int64,
        )
        parameters = _train_mlp(
            features=train_features,
            labels=special_labels,
            class_count=3,
            hidden_size=self.hidden_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
            prefix="special",
            class_multipliers=np.asarray(
                [1.0, self.object_hard_negative_weight, 1.0], dtype=np.float32
            ),
            focal_gamma=self.hard_negative_focal_gamma,
        )
        parameters.update(
            _train_mlp(
                features=train_features[pattern_mask],
                labels=pattern_labels,
                class_count=2,
                hidden_size=self.hidden_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                seed=self.seed + 1,
                prefix="pattern",
            )
        )
        knownness_features = np.concatenate((train_features, unknown_features), axis=0)
        knownness_labels = np.concatenate(
            (
                np.ones(len(train_features), dtype=np.int64),
                np.zeros(len(unknown_features), dtype=np.int64),
            )
        )
        parameters.update(
            _train_mlp(
                features=knownness_features,
                labels=knownness_labels,
                class_count=2,
                hidden_size=self.hidden_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                seed=self.seed + 2,
                prefix="known",
            )
        )
        return parameters, len(augmented_groups), len(knownness_features), degraded_count

    def _calibration_partition(
        self, records: tuple[object, ...]
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        by_asset: dict[str, list[object]] = {}
        for record in records:
            asset = getattr(record, "source_asset_id", getattr(record, "crop_id", ""))
            by_asset.setdefault(str(asset), []).append(record)
        required = {"cue", "solid", "eight", "stripe", "unknown"}
        assets = tuple(sorted(by_asset))
        unknown_assets = tuple(
            asset
            for asset in assets
            if any(getattr(record, "group", None) == "unknown" for record in by_asset[asset])
        )

        def partition(held: set[str]) -> tuple[tuple[object, ...], tuple[object, ...]] | None:
            calibration = tuple(record for asset in sorted(held) for record in by_asset[asset])
            fit = tuple(
                record
                for record in records
                if str(getattr(record, "source_asset_id", getattr(record, "crop_id", "")))
                not in held
            )
            fit_groups = {getattr(record, "group", None) for record in fit}
            calibration_groups = [getattr(record, "group", None) for record in calibration]
            unknown_count = calibration_groups.count("unknown")
            if (
                required.issubset(fit_groups)
                and unknown_count >= self.calibration_min_unknown_support
                and len(calibration_groups) - unknown_count >= 1
            ):
                return fit, calibration
            return None

        max_unknown_assets = min(self.calibration_min_unknown_support, len(unknown_assets))
        for asset_count in range(1, max_unknown_assets + 1):
            for held_unknown_assets in combinations(unknown_assets, asset_count):
                held = set(held_unknown_assets)
                candidate = partition(held)
                if candidate is not None:
                    return candidate
                if all(
                    getattr(record, "group", None) == "unknown"
                    for asset in held
                    for record in by_asset[asset]
                ):
                    for known_asset in assets:
                        if known_asset in held or not any(
                            getattr(record, "group", None) != "unknown"
                            for record in by_asset[known_asset]
                        ):
                            continue
                        candidate = partition(held | {known_asset})
                        if candidate is not None:
                            return candidate
        return records, ()

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
        if group_class_map != ("cue", "solid", "eight", "stripe", "unknown"):
            raise ValueError("hierarchical group class map is invalid")
        if exact_class_map:
            raise ValueError("exact head is deferred for this training backend")
        train_paths, train_groups = self._paths_and_groups(train_records)
        valid_paths, valid_groups = self._paths_and_groups(valid_records)
        fit_records, calibration_records = self._calibration_partition(train_records)
        calibration_summary: dict[str, object]
        calibrated_known_probability_min = self.known_probability_min
        if calibration_records:
            fit_paths, fit_groups = self._paths_and_groups(fit_records)
            provisional, _, _, _ = self._fit_parameters(fit_paths, fit_groups)
            calibration_paths, calibration_groups = self._paths_and_groups(calibration_records)
            calibration_features = self._embed(calibration_paths)
            calibration_probabilities = combine_hierarchical_probabilities(
                special_logits=_forward_mlp(calibration_features, provisional, "special"),
                pattern_logits=_forward_mlp(calibration_features, provisional, "pattern"),
            )
            calibration_known = _softmax(_forward_mlp(calibration_features, provisional, "known"))[
                :, 1
            ]
            label_indexes = {label: index for index, label in enumerate(group_class_map)}
            calibration_truth = np.asarray(
                [label_indexes[group] for group in calibration_groups], dtype=np.int64
            )
            calibrated_known_probability_min, calibration_summary = calibrate_known_probability_min(
                probabilities=calibration_probabilities,
                known_probabilities=calibration_known,
                truth=calibration_truth,
                configured_min=self.known_probability_min,
                confidence_min=self.confidence_min,
                margin_min=self.margin_min,
                min_unknown_support=self.calibration_min_unknown_support,
            )
            calibration_summary["asset_count"] = len(
                {getattr(record, "source_asset_id", None) for record in calibration_records}
            )
        else:
            calibration_summary = {
                "source": "configured_fallback",
                "known_support": 0,
                "unknown_support": 0,
                "selected_unknown_f1": 0.0,
                "asset_count": 0,
            }
        parameters, augmented_train_count, unknown_rejection_train_count, degraded_count = (
            self._fit_parameters(train_paths, train_groups)
        )
        valid_features = self._embed(valid_paths)
        probabilities = combine_hierarchical_probabilities(
            special_logits=_forward_mlp(valid_features, parameters, "special"),
            pattern_logits=_forward_mlp(valid_features, parameters, "pattern"),
        )
        known_probabilities = _softmax(_forward_mlp(valid_features, parameters, "known"))[:, 1]
        predicted = predict_groups_with_rejection(
            probabilities,
            confidence_min=self.confidence_min,
            margin_min=self.margin_min,
            known_probabilities=known_probabilities,
            known_probability_min=calibrated_known_probability_min,
        )
        label_indexes = {label: index for index, label in enumerate(group_class_map)}
        truth = np.asarray([label_indexes[group] for group in valid_groups], dtype=np.int64)
        metrics = _metrics(truth=truth, predicted=predicted, class_map=group_class_map)
        known_f1 = [metrics["per_class"][label]["f1"] for label in self.parity_class_map]
        metrics["group_macro_f1"] = float(np.mean(known_f1))
        payload = {
            "schema_version": "5.0",
            "architecture": "hierarchical_fused_evidence_mlp",
            "trained_heads": ["group"],
            "exact_head": {"status": "deferred", "trained": False},
            "unknown_policy": {
                "trained_class": False,
                "trained_rejection_head": True,
                "head": "binary_knownness_mlp",
                "configured_known_probability_min": self.known_probability_min,
                "known_probability_min": calibrated_known_probability_min,
                "confidence_min": self.confidence_min,
                "margin_min": self.margin_min,
                "calibration": calibration_summary,
            },
            "hard_negative_policy": {
                "source": "train_only_online_focal",
                "focal_gamma": self.hard_negative_focal_gamma,
                "object_weight": self.object_hard_negative_weight,
                "validation_feedback_used": False,
            },
            "visual_evidence_policy": {
                "backend": self.evidence_backend.version,
                "feature_names": list(self.evidence_backend.feature_names),
                "weight": self.visual_evidence_weight,
            },
            "forced_exact_without_evidence": 0,
            "feature_backend": self.feature_backend.version,
            "train_count": len(train_records),
            "known_train_count": len([group for group in train_groups if group != "unknown"]),
            "augmented_train_count": augmented_train_count,
            "unknown_rejection_train_count": unknown_rejection_train_count,
            "insufficient_evidence_train_count": degraded_count,
            "valid_count": len(valid_records),
            **metrics,
        }
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                **parameters,
                confidence_min=np.asarray([self.confidence_min], dtype=np.float32),
                margin_min=np.asarray([self.margin_min], dtype=np.float32),
                known_probability_min=np.asarray(
                    [calibrated_known_probability_min], dtype=np.float32
                ),
                parity_features=valid_features[:1],
            )

    @staticmethod
    def _checkpoint(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            parameter_names = {
                f"{prefix}_{suffix}"
                for prefix in ("special", "pattern", "known")
                for suffix in ("w1", "b1", "w2", "b2")
            }
            parameters = {
                name: np.asarray(payload[name], dtype=np.float32)
                for name in payload.files
                if name in parameter_names
            }
            features = np.asarray(payload["parity_features"], dtype=np.float32)
        return parameters, features

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        if self.onnx_adapter is None:
            raise RuntimeError("hierarchical ONNX adapter is unavailable")
        parameters, _ = self._checkpoint(checkpoint_path)
        self.onnx_adapter.export(parameters=parameters, output_path=output_path)
        bind = getattr(self.onnx_adapter, "bind", None)
        if callable(bind):
            bind(output_path)

    def predict_records(
        self, *, checkpoint_path: Path, records: tuple[object, ...]
    ) -> tuple[dict[str, object], ...]:
        """Score reviewed crops with a frozen hierarchical checkpoint."""

        paths, groups = self._paths_and_groups(records)
        features = self._embed(paths)
        parameters, _ = self._checkpoint(checkpoint_path)
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            confidence_min = float(np.asarray(payload["confidence_min"])[0])
            margin_min = float(np.asarray(payload["margin_min"])[0])
            known_probability_min = (
                float(np.asarray(payload["known_probability_min"])[0])
                if "known_probability_min" in payload.files
                else 0.0
            )
        probabilities = combine_hierarchical_probabilities(
            special_logits=_forward_mlp(features, parameters, "special"),
            pattern_logits=_forward_mlp(features, parameters, "pattern"),
        )
        known_probabilities = (
            _softmax(_forward_mlp(features, parameters, "known"))[:, 1]
            if "known_w1" in parameters
            else None
        )
        predicted = predict_groups_with_rejection(
            probabilities,
            confidence_min=confidence_min,
            margin_min=margin_min,
            known_probabilities=known_probabilities,
            known_probability_min=known_probability_min,
        )
        labels = (*self.parity_class_map, "unknown")
        output = []
        known_values = (
            known_probabilities
            if known_probabilities is not None
            else np.ones(len(probabilities), dtype=np.float32)
        )
        for record, truth, prediction, row, known_probability in zip(
            records, groups, predicted, probabilities, known_values, strict=True
        ):
            ordered = np.sort(row)
            output.append(
                {
                    "crop_id": record.crop_id,
                    "truth": truth,
                    "predicted": labels[int(prediction)],
                    "probabilities": [float(value) for value in row],
                    "confidence": float(ordered[-1]),
                    "margin": float(ordered[-1] - ordered[-2]),
                    "known_probability": float(known_probability),
                }
            )
        return tuple(output)

    def parity_logits(
        self, *, checkpoint_path: Path, onnx_path: Path
    ) -> tuple[dict[str, Sequence[float]], dict[str, Sequence[float]]]:
        if self.onnx_adapter is None:
            raise RuntimeError("hierarchical ONNX adapter is unavailable")
        parameters, features = self._checkpoint(checkpoint_path)
        native = combine_hierarchical_probabilities(
            special_logits=_forward_mlp(features, parameters, "special"),
            pattern_logits=_forward_mlp(features, parameters, "pattern"),
        )
        bind = getattr(self.onnx_adapter, "bind", None)
        if callable(bind):
            bind(onnx_path)
        exported = self.onnx_adapter.probabilities(parameters=parameters, features=features)
        native_known = _softmax(_forward_mlp(features, parameters, "known"))[:, 1:2]
        exported_known = self.onnx_adapter.known_probabilities(
            parameters=parameters, features=features
        )
        return (
            {
                "group": tuple(float(value) for value in native[0]),
                "known": tuple(float(value) for value in native_known[0]),
            },
            {
                "group": tuple(float(value) for value in exported[0]),
                "known": tuple(float(value) for value in exported_known[0]),
            },
        )
