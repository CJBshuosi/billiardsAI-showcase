from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from billiards_ai.lab.mainline1_runtime import (
    OnnxGroupIdentityClassifier,
    RfdetrDetectionsAdapter,
    load_development_group_classifier,
    load_development_vision_backend,
)
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
)


class _Detections:
    xyxy = np.asarray(((1, 2, 5, 8), (10, 3, 15, 9)), dtype=np.float32)
    confidence = np.asarray((0.91, 0.82), dtype=np.float32)
    class_id = np.asarray((0, 0), dtype=np.int64)

    def __len__(self) -> int:
        return 2


class _PrivateDetector:
    def predict(self, image, *, threshold, include_source_image):
        assert image[0, 0].tolist() == [30, 20, 10]
        assert threshold == 0.25
        assert include_source_image is False
        return _Detections()


class _Features:
    version = "test-features"

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray:
        assert len(crops) == 2 and all(path.is_file() for path in crops)
        return np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)


class _Evidence:
    version = "test-evidence"
    feature_names = ("white", "pattern")

    def extract(self, crops: tuple[Path, ...]) -> np.ndarray:
        assert len(crops) == 2
        return np.asarray(((1.0, 0.5), (0.5, 1.0)), dtype=np.float32)


class _Session:
    def run(self, names, inputs):
        assert names == ["group_probabilities", "known_probability"]
        assert inputs["features"].shape == (2, 4)
        return (
            np.asarray(((0.90, 0.05, 0.03, 0.02), (0.05, 0.10, 0.05, 0.80))),
            np.asarray(((0.95,), (0.20,))),
        )


def _detection(detection_id: str, left: float) -> BallInstanceObservation:
    return BallInstanceObservation(
        detection_id=detection_id,
        bbox=PixelBox(left=left, top=4, right=left + 4, bottom=8),
        center=PixelPoint(x=left + 2, y=6),
        confidence=0.9,
        class_label="ball",
    )


def test_rfdetr_private_detections_are_normalized_at_the_existing_boundary() -> None:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)

    rows = RfdetrDetectionsAdapter(_PrivateDetector()).predict(image, threshold=0.25)

    assert rows == [
        {"bbox_xyxy": [1.0, 2.0, 5.0, 8.0], "confidence": rows[0]["confidence"], "class_id": 0},
        {
            "bbox_xyxy": [10.0, 3.0, 15.0, 9.0],
            "confidence": rows[1]["confidence"],
            "class_id": 0,
        },
    ]
    assert np.allclose([row["confidence"] for row in rows], [0.91, 0.82])


def test_group_runtime_uses_existing_rejection_policy_and_never_forces_exact_identity() -> None:
    classifier = OnnxGroupIdentityClassifier(
        feature_backend=_Features(),
        evidence_backend=_Evidence(),
        session=_Session(),
        confidence_min=0.55,
        margin_min=0.10,
        known_probability_min=0.30,
        visual_evidence_weight=0.5,
    )

    rows = classifier.classify(
        image=np.zeros((16, 24, 3), dtype=np.uint8),
        detections=(_detection("d1", 2), _detection("d2", 12)),
    )

    assert rows["d1"]["group"] == "cue"
    assert rows["d2"]["group"] == "unknown"
    assert rows["d1"]["exact_identity"] is None
    assert rows["d2"]["exact_identity"] is None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_runtime_loaders_verify_existing_training_artifacts(
    tmp_path: Path,
) -> None:
    detector_run = tmp_path / ("a" * 64)
    detector_run.mkdir()
    checkpoint = detector_run / "checkpoint_best.pth"
    checkpoint.write_bytes(b"detector")
    detector_manifest = {
        "schema_version": "1.0",
        "run_hash": detector_run.name,
        "dataset_id": "dataset-r1",
        "config_hash": "b" * 64,
        "candidate_id": "rfdetr-seg-small-bootstrap-short-r1",
        "model_class": "RFDETRSegSmall",
        "backend": {"weight_sha256": "d" * 64},
        "artifacts": {"checkpoint_best.pth": _sha256(checkpoint)},
    }
    (detector_run / "training-manifest.json").write_text(json.dumps(detector_manifest))

    identity_run = tmp_path / ("c" * 64)
    identity_run.mkdir()
    onnx = identity_run / "group-head.onnx"
    onnx.write_bytes(b"identity")
    metrics = identity_run / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "unknown_policy": {
                    "confidence_min": 0.55,
                    "margin_min": 0.10,
                    "known_probability_min": 0.30,
                },
                "visual_evidence_policy": {"weight": 0.5},
            }
        )
    )
    identity_manifest = {
        "schema_version": "1.0",
        "run_id": identity_run.name,
        "status": "PASS",
        "gate_status": "FAIL",
        "backend_configuration": {"architecture": "hierarchical_fused_evidence_mlp"},
        "inventory": {
            "group-head.onnx": {"sha256": _sha256(onnx)},
            "metrics.json": {"sha256": _sha256(metrics)},
        },
    }
    (identity_run / "manifest.json").write_text(json.dumps(identity_manifest))

    backend = load_development_vision_backend(
        detector_run,
        model_factory=lambda path: _PrivateDetector(),
    )
    classifier = load_development_group_classifier(
        identity_run,
        feature_backend=_Features(),
        evidence_backend=_Evidence(),
        session_factory=lambda path: _Session(),
    )

    assert backend.package_identity.package_id == detector_run.name
    assert backend._package.manifest.base_weight_sha256 == "d" * 64
    assert isinstance(classifier, OnnxGroupIdentityClassifier)
