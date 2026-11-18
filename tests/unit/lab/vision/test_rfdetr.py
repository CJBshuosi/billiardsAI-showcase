from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from billiards_ai.lab.vision.contracts import PixelPoint, make_vision_frame_input
from billiards_ai.lab.vision.package import (
    VerifiedVisionPackage,
    VisionArtifact,
    VisionModelPackageManifest,
    VisionPostprocess,
    VisionPreprocess,
)
from billiards_ai.lab.vision.rfdetr import RfdetrVisionBackend, VisionInferenceError


class _FakeRfdetr:
    def __init__(self, predictions: list[object]) -> None:
        self.predictions = predictions
        self.thresholds: list[float] = []

    def predict(self, image: np.ndarray, *, threshold: float) -> list[object]:
        assert image.flags.writeable is False
        self.thresholds.append(threshold)
        return self.predictions


class _FakeBatchRfdetr:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, images: object, *, threshold: float) -> object:
        self.calls += 1
        assert isinstance(images, list)
        assert threshold == 0.25
        return [
            [{"bbox_xyxy": [index + 1, 2, index + 5, 6], "confidence": 0.9, "class_id": 0}]
            for index, _ in enumerate(images)
        ]


def _package(tmp_path: Path) -> VerifiedVisionPackage:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    manifest = VisionModelPackageManifest(
        package_id="a" * 64,
        task="ball_instance_segmentation",
        family="rfdetr",
        model_version="1.9.0",
        model_class="RFDETRSegSmall",
        artifacts=(
            VisionArtifact(
                role="checkpoint",
                relative_path="checkpoint.pth",
                byte_count=10,
                sha256="d" * 64,
            ),
        ),
        preprocess=VisionPreprocess(
            color_space="RGB",
            input_width_px=384,
            input_height_px=384,
        ),
        postprocess=VisionPostprocess(confidence_threshold=0.25, mask_threshold=0.5),
        class_map={"ball": 0},
        runtime_requirements=("rfdetr==1.9.0",),
        training_run_sha256="1" * 64,
        code_sha256="2" * 64,
        base_weight_sha256="3" * 64,
        asset_status_ref="internal-rnd-only",
    )
    from billiards_ai.lab.vision.contracts import VisionPackageIdentity

    return VerifiedVisionPackage(
        root=tmp_path,
        manifest_path=manifest_path,
        identity=VisionPackageIdentity(package_id="a" * 64, manifest_sha256="b" * 64),
        manifest=manifest,
        checkpoint=checkpoint,
    )


def _frame():
    return make_vision_frame_input(
        np.zeros((24, 32, 3), dtype=np.uint8),
        frame_id=7,
        timestamp_ms=233,
    )


def test_explicit_execution_preparation_discards_private_outputs_then_keeps_real_inference(
    tmp_path,
):
    calls = []

    class Model:
        def predict(self, image, *, threshold):
            calls.append((image.shape, threshold))
            assert not image.flags.writeable
            if image.shape == (1080, 1920, 3):
                assert not np.any(image)
                return object()  # Invalid as public detections; it must be discarded.
            return [{"bbox_xyxy": [2, 3, 12, 13], "confidence": 0.9, "class_id": 0}]

    backend = RfdetrVisionBackend(_package(tmp_path), model_factory=lambda _: Model())
    backend.prepare_execution(width_px=1920, height_px=1080)
    backend.prepare_execution(width_px=1920, height_px=1080)
    result = backend.infer(_frame())
    assert result.frame_id == 7 and len(result.detections) == 1
    assert calls == [((1080, 1920, 3), 0.25), ((24, 32, 3), 0.25)]


def test_execution_preparation_preserves_original_failure_without_retry(tmp_path):
    calls = []
    failure = RuntimeError("controlled model execution preparation failure")

    class Model:
        def predict(self, *args, **kwargs):
            calls.append(1)
            raise failure

    backend = RfdetrVisionBackend(_package(tmp_path), model_factory=lambda _: Model())
    with pytest.raises(RuntimeError) as raised:
        backend.prepare_execution(width_px=1920, height_px=1080)
    assert raised.value is failure and calls == [1]


def test_rfdetr_backend_normalizes_private_predictions(tmp_path: Path) -> None:
    model = _FakeRfdetr([{"bbox_xyxy": [2, 3, 12, 13], "confidence": 0.9, "class_id": 0}])
    backend = RfdetrVisionBackend(_package(tmp_path), model_factory=lambda _: model)

    result = backend.infer(_frame())

    assert result.package == _package(tmp_path).identity
    assert result.detections[0].detection_id == "frame-000007-detection-0000"
    assert result.detections[0].center == PixelPoint(x=7, y=8)
    assert result.threshold == 0.25
    assert model.thresholds == [0.25]


def test_rfdetr_backend_preserves_a_valid_zero_detection_frame(tmp_path: Path) -> None:
    result = RfdetrVisionBackend(_package(tmp_path), lambda _: _FakeRfdetr([])).infer(_frame())

    assert result.detections == ()


def test_rfdetr_batch_calls_model_once_and_preserves_source_identity(tmp_path: Path) -> None:
    model = _FakeBatchRfdetr()
    backend = RfdetrVisionBackend(_package(tmp_path), lambda _: model)
    first = _frame()
    second = make_vision_frame_input(
        np.zeros((24, 32, 3), dtype=np.uint8),
        frame_id=8,
        timestamp_ms=266,
    )

    rows = backend.infer_batch((first, second))

    assert model.calls == 1
    assert [row.frame_id for row in rows] == [7, 8]
    assert [row.detections[0].detection_id for row in rows] == [
        "frame-000007-detection-0000",
        "frame-000008-detection-0000",
    ]


@pytest.mark.parametrize(
    "prediction",
    [
        {"bbox_xyxy": [1, 2, 3]},
        {"bbox_xyxy": [1, 2, 3, 4], "confidence": float("nan"), "class_id": 0},
        {"bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9, "class_id": 1},
    ],
)
def test_rfdetr_backend_rejects_malformed_predictions(
    tmp_path: Path,
    prediction: dict[str, object],
) -> None:
    with pytest.raises(VisionInferenceError):
        RfdetrVisionBackend(_package(tmp_path), lambda _: _FakeRfdetr([prediction])).infer(_frame())


def test_backend_rejects_non_rfdetr_package(tmp_path: Path) -> None:
    package = _package(tmp_path)
    changed = package.model_copy(
        update={"manifest": package.manifest.model_copy(update={"family": "other"})}
    )

    with pytest.raises(VisionInferenceError, match="family"):
        RfdetrVisionBackend(changed, lambda _: _FakeRfdetr([]))
