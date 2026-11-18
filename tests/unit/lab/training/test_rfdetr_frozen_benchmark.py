from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import billiards_ai.lab.training.rfdetr_frozen_benchmark as benchmark_module
from billiards_ai.lab.propagation.pilot_bundle import PilotInputBundle
from billiards_ai.lab.propagation.pilot_data import EvaluationCircle
from billiards_ai.lab.training.rfdetr_frozen_benchmark import (
    BenchmarkExecution,
    BenchmarkMethodMetrics,
    FrozenBenchmarkInputs,
    adapt_classic_predictions,
    adapt_rfdetr_predictions,
    compute_benchmark_gates,
    decode_sparse_video_frames,
    evaluate_method,
    publish_benchmark,
    resolve_materialized_annotation_export,
    run_model_factories_on_frames,
    run_models_on_frames,
    snapshot_locked_file,
    verify_frozen_evaluation_population,
    verify_no_frozen_frame_leakage,
    verify_output_root_isolated,
    verify_rebuilt_evaluation,
    verify_rfdetr_runtime,
    verify_training_artifact_hashes,
    verify_training_manifest_identity,
)


class _Detections:
    def __init__(
        self,
        *,
        boxes: tuple[tuple[float, float, float, float], ...] = (
            (5.0, 5.0, 15.0, 15.0),
            (25.0, 5.0, 35.0, 15.0),
            (45.0, 5.0, 55.0, 15.0),
        ),
        confidences: tuple[float, ...] = (0.9, 0.24, 0.8),
        class_ids: tuple[int, ...] = (0, 0, 37),
    ) -> None:
        self.xyxy = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
        self.confidence = np.asarray(confidences, dtype=np.float32)
        self.class_id = np.asarray(class_ids, dtype=np.int64)
        self.mask = None

    def __len__(self) -> int:
        return len(self.xyxy)


def _circle(annotation_id: str, x_px: float) -> EvaluationCircle:
    return EvaluationCircle(
        annotation_id=annotation_id,
        video_id="video1",
        shot_index=0,
        source_frame_index=10,
        source_frame_sha256="a" * 64,
        center_x_px=x_px,
        center_y_px=10.0,
        radius_px=5.0,
        role="middle",
    )


class _Model:
    def __init__(self, detections: _Detections) -> None:
        self._detections = detections

    def predict(self, image: np.ndarray, **kwargs: object) -> _Detections:
        if image.shape != (40, 60, 3):
            raise AssertionError("model did not receive the exact frozen RGB frame")
        if kwargs != {"threshold": 0.25, "include_source_image": False}:
            raise AssertionError("model inference contract changed")
        return self._detections


class _Classic:
    def detect(self, image: object) -> tuple[dict[str, object], ...]:
        if np.asarray(image).shape != (40, 60, 3):
            raise AssertionError("classic CV did not receive the exact frozen BGR frame")
        return (
            {
                "center_px": (10.0, 10.0),
                "radius_px": 5.0,
                "confidence": 0.9,
                "mask_polygon_px": ((5.0, 10.0), (10.0, 5.0), (15.0, 10.0)),
                "rejection_reason": None,
            },
        )


def _metrics(*, precision: float, recall: float, f1: float) -> BenchmarkMethodMetrics:
    return BenchmarkMethodMetrics(
        method="trained_rfdetr",
        frame_count=50,
        ground_truth_count=510,
        prediction_count=500,
        true_positive_count=480,
        false_negative_count=30,
        false_positive_count=20,
        precision=precision,
        recall=recall,
        f1=f1,
        median_normalized_center_error=0.25,
        p90_normalized_center_error=0.75,
    )


def test_rfdetr_adapter_enforces_the_frozen_threshold_and_class_id() -> None:
    observations = adapt_rfdetr_predictions(
        asset_id="video1",
        frame_index=10,
        detections=_Detections(),
        accepted_class_id=0,
        method="trained_rfdetr",
    )

    assert len(observations) == 1
    assert observations[0].center_px == (10.0, 10.0)
    assert observations[0].confidence == pytest.approx(0.9)


def test_classic_adapter_drops_abstentions_and_preserves_contour_geometry() -> None:
    observations = adapt_classic_predictions(
        asset_id="video1",
        frame_index=10,
        detections=(
            {
                "center_px": (10.0, 10.0),
                "radius_px": 5.0,
                "confidence": 0.9,
                "mask_polygon_px": ((5.0, 10.0), (10.0, 5.0), (15.0, 10.0)),
                "rejection_reason": None,
            },
            {
                "center_px": (30.0, 10.0),
                "radius_px": 5.0,
                "confidence": 0.4,
                "mask_polygon_px": ((25.0, 10.0), (30.0, 5.0), (35.0, 10.0)),
                "rejection_reason": "low_confidence",
            },
        ),
    )

    assert len(observations) == 1
    assert observations[0].box_xyxy_px == (5.0, 5.0, 15.0, 15.0)
    assert observations[0].area_px == 25


def test_method_evaluation_uses_the_existing_one_radius_assignment() -> None:
    circles = (_circle("gt-a", 10.0), _circle("gt-b", 30.0), _circle("gt-c", 50.0))
    predictions = adapt_classic_predictions(
        asset_id="video1",
        frame_index=10,
        detections=tuple(
            {
                "center_px": center,
                "radius_px": 5.0,
                "confidence": 0.9,
                "mask_polygon_px": (
                    (center[0] - 5, center[1]),
                    (center[0], center[1] - 5),
                    (center[0] + 5, center[1]),
                ),
                "rejection_reason": None,
            }
            for center in ((10.0, 10.0), (34.0, 10.0), (100.0, 10.0))
        ),
    )

    result = evaluate_method(
        method="classic_cv",
        circles=circles,
        predictions={(("video1", 10)): predictions},
    )

    assert result.true_positive_count == 2
    assert result.false_negative_count == 1
    assert result.false_positive_count == 1
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.median_normalized_center_error == pytest.approx(0.4)
    assert result.p90_normalized_center_error == pytest.approx(0.8)


def test_benchmark_gate_requires_absolute_quality_and_strict_baseline_lift() -> None:
    trained = _metrics(precision=0.90, recall=0.88, f1=0.89)
    base = _metrics(precision=0.80, recall=0.82, f1=0.81).model_copy(
        update={"method": "coco_rfdetr"}
    )
    classic = _metrics(precision=0.75, recall=0.78, f1=0.76).model_copy(
        update={"method": "classic_cv"}
    )

    gates = compute_benchmark_gates(trained=trained, coco=base, classic=classic)

    assert gates.passed is True
    assert gates.precision.passed is True
    assert gates.recall.passed is True
    assert gates.f1.passed is True
    assert gates.beats_coco_f1.passed is True
    assert gates.beats_classic_f1.passed is True


def test_benchmark_gate_rejects_a_tie_with_either_baseline() -> None:
    trained = _metrics(precision=0.90, recall=0.88, f1=0.89)
    base = _metrics(precision=0.80, recall=0.82, f1=0.89).model_copy(
        update={"method": "coco_rfdetr"}
    )
    classic = _metrics(precision=0.75, recall=0.78, f1=0.76).model_copy(
        update={"method": "classic_cv"}
    )

    gates = compute_benchmark_gates(trained=trained, coco=base, classic=classic)

    assert gates.passed is False
    assert gates.beats_coco_f1.passed is False


def test_benchmark_gate_represents_missing_center_errors_as_strict_json_null() -> None:
    trained = _metrics(precision=0.0, recall=0.0, f1=0.0).model_copy(
        update={
            "prediction_count": 0,
            "true_positive_count": 0,
            "false_negative_count": 510,
            "false_positive_count": 0,
            "median_normalized_center_error": None,
            "p90_normalized_center_error": None,
        }
    )
    base = trained.model_copy(update={"method": "coco_rfdetr"})
    classic = trained.model_copy(update={"method": "classic_cv"})

    gates = compute_benchmark_gates(trained=trained, coco=base, classic=classic)

    assert gates.median_center_error.value is None
    assert gates.median_center_error.passed is False
    assert gates.p90_center_error.value is None
    json.dumps(gates.model_dump(mode="json"), allow_nan=False)


def test_bootstrap_dataset_rejects_any_frozen_frame_leakage(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "valid"):
        directory = dataset / split
        directory.mkdir(parents=True)
        (directory / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "frame.jpg",
                            "source_asset_id": "video1" if split == "train" else "video2",
                            "source_frame_index": 10 if split == "train" else 20,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="leaks a frozen evaluation frame"):
        verify_no_frozen_frame_leakage(
            dataset,
            frozen_frames=frozenset({("video1", 10)}),
        )


def test_bootstrap_overlap_check_rejects_missing_frame_provenance(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "valid"):
        directory = dataset / split
        directory.mkdir(parents=True)
        (directory / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "frame.jpg",
                            "source_asset_id": "video1",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="frame provenance"):
        verify_no_frozen_frame_leakage(dataset, frozen_frames=frozenset())


def test_training_manifest_rejects_a_non_seg_small_checkpoint_identity() -> None:
    dataset_id = "a" * 64
    config_hash = "b" * 64
    backend = {
        "name": "rfdetr",
        "version": "1.9.0",
        "model_class": "RFDETRSegMedium",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "weight_sha256": "c" * 64,
        "accelerator": "cuda",
        "precision": "16-mixed",
    }
    run_hash = hashlib.sha256(
        json.dumps(
            {"backend": backend, "config_hash": config_hash, "dataset_id": dataset_id},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "run_hash": run_hash,
        "dataset_id": dataset_id,
        "config_hash": config_hash,
        "candidate_id": "rfdetr-seg-small-bootstrap-short-r1",
        "model_class": "RFDETRSegMedium",
        "seed": 20260801,
        "synthetic_only": False,
        "scientific_use": "development_short_train_not_locked_accuracy",
        "backend": backend,
        "checkpoint_reloaded": True,
        "onnx_exported": True,
    }

    with pytest.raises(ValueError, match="training identity"):
        verify_training_manifest_identity(
            manifest,
            dataset_id=dataset_id,
            approved_config_hash=config_hash,
        )


def test_training_manifest_accepts_and_recomputes_the_approved_identity() -> None:
    dataset_id = "a" * 64
    config_hash = "b" * 64
    weight_sha256 = "c" * 64
    backend = {
        "name": "rfdetr",
        "version": "1.9.0",
        "model_class": "RFDETRSegSmall",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "weight_sha256": weight_sha256,
        "accelerator": "cuda",
        "precision": "16-mixed",
    }
    run_hash = hashlib.sha256(
        (
            json.dumps(
                {"backend": backend, "config_hash": config_hash, "dataset_id": dataset_id},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "run_hash": run_hash,
        "dataset_id": dataset_id,
        "config_hash": config_hash,
        "candidate_id": "rfdetr-seg-small-bootstrap-short-r1",
        "model_class": "RFDETRSegSmall",
        "seed": 20260801,
        "synthetic_only": False,
        "scientific_use": "development_short_train_not_locked_accuracy",
        "backend": backend,
        "checkpoint_reloaded": True,
        "onnx_exported": True,
    }

    assert (
        verify_training_manifest_identity(
            manifest,
            dataset_id=dataset_id,
            approved_config_hash=config_hash,
        )
        == weight_sha256
    )


def test_training_artifact_requires_independently_locked_manifest_and_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    training_run = tmp_path / "training"
    training_run.mkdir()
    manifest = training_run / "training-manifest.json"
    checkpoint = training_run / "checkpoint_best.pth"
    manifest.write_bytes(b'{"run_hash":"approved"}')
    checkpoint.write_bytes(b"approved checkpoint")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert verify_training_artifact_hashes(
        training_run,
        expected_manifest_sha256=manifest_sha256,
        expected_checkpoint_sha256=checkpoint_sha256,
    ) == (manifest_sha256, checkpoint_sha256)

    checkpoint.write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="locked hash"):
        verify_training_artifact_hashes(
            training_run,
            expected_manifest_sha256=manifest_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
        )


def test_locked_model_snapshot_is_immune_to_later_source_replacement(tmp_path: Path) -> None:
    source = tmp_path / "approved.pth"
    snapshot = tmp_path / "private" / "approved.pth"
    approved_bytes = b"approved checkpoint bytes"
    source.write_bytes(approved_bytes)
    expected_sha256 = hashlib.sha256(approved_bytes).hexdigest()

    assert (
        snapshot_locked_file(
            source,
            destination=snapshot,
            expected_sha256=expected_sha256,
        )
        == expected_sha256
    )
    source.write_bytes(b"replacement after verification")

    assert snapshot.read_bytes() == approved_bytes
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected_sha256


def test_runtime_rejects_any_rfdetr_version_other_than_1_9() -> None:
    with pytest.raises(ValueError, match="rfdetr==1.9.0"):
        verify_rfdetr_runtime(
            version_resolver=lambda name: "2.0.0",
            module_loader=lambda name: object(),
        )


def test_output_root_cannot_live_inside_a_frozen_input(tmp_path: Path) -> None:
    input_bundle = tmp_path / "input"
    dataset = tmp_path / "dataset"
    training = tmp_path / "training"
    for path in (input_bundle, dataset, training):
        path.mkdir()
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    weight = tmp_path / "base.pth"
    weight.write_bytes(b"weight")
    inputs = FrozenBenchmarkInputs(
        input_bundle=input_bundle,
        evaluation_manifest=evaluation,
        bootstrap_dataset=dataset,
        training_run=training,
        base_weight=weight,
        expected_base_weight_sha256="a" * 64,
        expected_training_manifest_sha256="b" * 64,
        expected_trained_checkpoint_sha256="c" * 64,
        output_root=training / "benchmark-output",
        trained_model_factory=lambda path: _Model(_Detections()),
        coco_model_factory=lambda path: _Model(_Detections()),
        classic_backend_factory=_Classic,
    )

    with pytest.raises(ValueError, match="overlaps a frozen input"):
        verify_output_root_isolated(inputs)


@pytest.mark.parametrize("protected_input", ["evaluation", "base_weight"])
def test_output_root_cannot_write_inside_any_frozen_file_artifact_tree(
    tmp_path: Path, protected_input: str
) -> None:
    input_bundle = tmp_path / "bundle" / "input"
    dataset = tmp_path / "dataset" / "bootstrap"
    training = tmp_path / "training" / "run"
    evaluation = tmp_path / "evaluation-artifact" / "evaluation-manifest.json"
    weight = tmp_path / "weight-artifact" / "base.pth"
    for directory in (
        input_bundle,
        dataset,
        training,
        evaluation.parent,
        weight.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    evaluation.write_text("{}", encoding="utf-8")
    weight.write_bytes(b"weight")
    protected_tree = evaluation.parent if protected_input == "evaluation" else weight.parent
    inputs = FrozenBenchmarkInputs(
        input_bundle=input_bundle,
        evaluation_manifest=evaluation,
        bootstrap_dataset=dataset,
        training_run=training,
        base_weight=weight,
        expected_base_weight_sha256="a" * 64,
        expected_training_manifest_sha256="b" * 64,
        expected_trained_checkpoint_sha256="c" * 64,
        output_root=protected_tree / "benchmark-output",
        trained_model_factory=lambda path: _Model(_Detections()),
        coco_model_factory=lambda path: _Model(_Detections()),
        classic_backend_factory=_Classic,
    )

    with pytest.raises(ValueError, match="overlaps a frozen input"):
        verify_output_root_isolated(inputs)


def test_frozen_population_rejects_49_frames_even_with_valid_circles() -> None:
    circles = tuple(
        _circle(f"gt-{index}", 10.0).model_copy(
            update={
                "video_id": f"video{index % 10}",
                "shot_index": index // 2,
                "source_frame_index": index,
                "role": "middle" if index % 2 == 0 else "end",
            }
        )
        for index in range(49)
    )

    with pytest.raises(ValueError, match="exactly 50 held-out frames"):
        verify_frozen_evaluation_population(circles)


def test_supplied_evaluation_must_exactly_match_bundle_reconstruction() -> None:
    circle = _circle("gt-a", 10.0)
    supplied = object.__new__(type("Manifest", (), {}))
    rebuilt = object.__new__(type("Manifest", (), {}))
    supplied.model_dump = lambda **kwargs: {  # type: ignore[attr-defined]
        "manifest_id": "a" * 64,
        "circles": [circle.model_dump(mode="json")],
    }
    rebuilt.model_dump = lambda **kwargs: {  # type: ignore[attr-defined]
        "manifest_id": "b" * 64,
        "circles": [circle.model_copy(update={"center_x_px": 11.0}).model_dump(mode="json")],
    }

    with pytest.raises(ValueError, match="does not match the verified input bundle"):
        verify_rebuilt_evaluation(supplied, rebuilt)  # type: ignore[arg-type]


def test_benchmark_rebuild_uses_prepare_bundle_materialized_review_images(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / ("a" * 64)
    raw_export = bundle_root / "annotation-export"
    raw_export.mkdir(parents=True)
    annotations = {
        "images": [
            {
                "file_name": "nested/review.jpg",
                "source_video": "video1",
                "source_frame_index": 10,
            }
        ]
    }
    (raw_export / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    (raw_export / "summary.json").write_text('{"image_count":1}', encoding="utf-8")
    control_root = tmp_path / "control"
    materialized = control_root / "imported-inputs" / bundle_root.name / "annotation-export"
    (materialized / "images" / "nested").mkdir(parents=True)
    (materialized / "annotations.json").write_bytes((raw_export / "annotations.json").read_bytes())
    (materialized / "summary.json").write_bytes((raw_export / "summary.json").read_bytes())
    (materialized / "images" / "nested" / "review.jpg").write_bytes(b"review-jpeg")
    evaluation = control_root / "evaluation-manifests" / ("b" * 64) / "evaluation-manifest.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text("{}", encoding="utf-8")
    bundle = PilotInputBundle(bundle_id=bundle_root.name, path=bundle_root, inventory=())

    assert (
        resolve_materialized_annotation_export(
            bundle=bundle,
            evaluation_manifest=evaluation,
        )
        == materialized.resolve()
    )

    (materialized / "images" / "nested" / "review.jpg").unlink()
    with pytest.raises(ValueError, match="materialized annotation export"):
        resolve_materialized_annotation_export(
            bundle=bundle,
            evaluation_manifest=evaluation,
        )


def test_three_models_run_on_identical_frames_and_publish_visual_evidence(tmp_path: Path) -> None:
    key = ("video1", 10)
    frames = {key: np.full((40, 60, 3), (60, 130, 70), dtype=np.uint8)}
    circles = (_circle("gt-a", 10.0),)
    trained = _Model(
        _Detections(boxes=((5.0, 5.0, 15.0, 15.0),), confidences=(0.9,), class_ids=(0,))
    )
    coco = _Model(
        _Detections(boxes=((40.0, 5.0, 50.0, 15.0),), confidences=(0.9,), class_ids=(37,))
    )

    execution = run_models_on_frames(
        frames=frames,
        circles=circles,
        trained_model=trained,
        coco_model=coco,
        classic_backend=_Classic(),
    )
    published = publish_benchmark(
        execution=execution,
        frames=frames,
        circles=circles,
        provenance={
            "input_bundle_id": "a" * 64,
            "evaluation_manifest_id": "b" * 64,
            "bootstrap_dataset_id": "c" * 64,
            "training_run_hash": "d" * 64,
            "training_manifest_sha256": "0" * 64,
            "trained_checkpoint_sha256": "e" * 64,
            "coco_weight_sha256": "f" * 64,
        },
        output_root=tmp_path,
    )

    assert execution.metrics["trained_rfdetr"].f1 == 1.0
    assert execution.metrics["coco_rfdetr"].f1 == 0.0
    assert execution.metrics["classic_cv"].f1 == 1.0
    assert published.reused is False
    assert (published.path / "index.html").is_file()
    assert (published.path / "report.json").is_file()
    assert (published.path / "predictions.json").is_file()
    assert (published.path / "failures.json").is_file()
    assert len(tuple((published.path / "overlays").glob("*.jpg"))) == 3
    failures = json.loads((published.path / "failures.json").read_text(encoding="utf-8"))
    report = json.loads((published.path / "report.json").read_text(encoding="utf-8"))
    assert report["identity"]["model_contracts"] == {
        "classic_cv": {
            "confidence_abstain_below": 0.5,
            "foreground_distance": 28.0,
            "maximum_radius_px": 40.0,
            "minimum_circularity": 0.82,
            "minimum_radius_px": 10.0,
            "version": "opencv-ball-locator/1.0",
        },
        "coco_rfdetr": {"accepted_class_id": 37, "model": "RFDETRSegSmall"},
        "rfdetr_version": "1.9.0",
        "trained_rfdetr": {"accepted_class_id": 0, "model": "RFDETRSegSmall"},
    }
    assert failures["trained_rfdetr"] == []
    assert failures["coco_rfdetr"] == [
        {
            "asset_id": "video1",
            "false_positive_ids": [execution.predictions["coco_rfdetr"][key][0].instance_id],
            "frame_index": 10,
            "missed_ground_truth_ids": ["gt-a"],
        }
    ]
    html = (published.path / "index.html").read_text(encoding="utf-8")
    assert "trained RF-DETR" in html
    assert "COCO RF-DETR" in html
    assert "classic CV" in html
    assert "Overall gate: FAIL" in html
    assert "Median center error" in html
    assert "False positives" in html
    assert "beats_classic_f1" in html
    assert "&gt;" in html
    assert "0.8500" in html

    reused = publish_benchmark(
        execution=execution,
        frames=frames,
        circles=circles,
        provenance={
            "input_bundle_id": "a" * 64,
            "evaluation_manifest_id": "b" * 64,
            "bootstrap_dataset_id": "c" * 64,
            "training_run_hash": "d" * 64,
            "training_manifest_sha256": "0" * 64,
            "trained_checkpoint_sha256": "e" * 64,
            "coco_weight_sha256": "f" * 64,
        },
        output_root=tmp_path,
    )
    assert reused.path == published.path
    assert reused.reused is True


def test_runtime_releases_trained_model_before_loading_coco_model() -> None:
    events: list[str] = []
    frames = {("video1", 10): np.zeros((40, 60, 3), dtype=np.uint8)}
    circles = (_circle("gt-a", 10.0),)

    class TrackedModel(_Model):
        def __init__(self, name: str, class_id: int) -> None:
            self.name = name
            super().__init__(
                _Detections(
                    boxes=((5.0, 5.0, 15.0, 15.0),),
                    confidences=(0.9,),
                    class_ids=(class_id,),
                )
            )

        def __del__(self) -> None:
            events.append(f"released:{self.name}")

    def trained_factory() -> TrackedModel:
        events.append("loaded:trained")
        return TrackedModel("trained", 0)

    def coco_factory() -> TrackedModel:
        if events != ["loaded:trained", "released:trained"]:
            raise AssertionError("COCO model loaded before trained model release")
        events.append("loaded:coco")
        return TrackedModel("coco", 37)

    execution = run_model_factories_on_frames(
        frames=frames,
        circles=circles,
        trained_model_factory=trained_factory,
        coco_model_factory=coco_factory,
        classic_backend_factory=_Classic,
    )

    assert execution.metrics["trained_rfdetr"].f1 == 1.0
    assert execution.metrics["coco_rfdetr"].f1 == 1.0
    assert events == [
        "loaded:trained",
        "released:trained",
        "loaded:coco",
        "released:coco",
    ]


def test_sparse_decoder_keeps_only_requested_indices_without_writing_intermediate_frames(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"stable-video-evidence")

    class Capture:
        def __init__(self) -> None:
            self.index = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, value: float) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 6.0
            if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 4.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 6.0
            return 0.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index == 6:
                return False, None
            frame = np.full((4, 6, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = Capture()
    decoded = decode_sparse_video_frames(
        video_path=video,
        wanted_indices=frozenset({1, 5}),
        capture_factory=lambda path: capture,
    )

    assert set(decoded) == {1, 5}
    assert int(decoded[1][0, 0, 0]) == 1
    assert int(decoded[5][0, 0, 0]) == 5
    assert capture.index == 6
    assert capture.released is True
    assert tuple(tmp_path.iterdir()) == (video,)


def test_publication_rejects_a_modified_overlay(tmp_path: Path) -> None:
    key = ("video1", 10)
    frames = {key: np.zeros((40, 60, 3), dtype=np.uint8)}
    circles = (_circle("gt-a", 10.0),)
    predictions = adapt_classic_predictions(
        asset_id="video1",
        frame_index=10,
        detections=_Classic().detect(frames[key]),
    )
    metric = evaluate_method(method="classic_cv", circles=circles, predictions={key: predictions})
    execution = BenchmarkExecution(
        predictions={
            "trained_rfdetr": {key: predictions},
            "coco_rfdetr": {key: predictions},
            "classic_cv": {key: predictions},
        },
        metrics={
            "trained_rfdetr": metric.model_copy(update={"method": "trained_rfdetr"}),
            "coco_rfdetr": metric.model_copy(update={"method": "coco_rfdetr"}),
            "classic_cv": metric,
        },
        gates=compute_benchmark_gates(
            trained=metric.model_copy(update={"method": "trained_rfdetr"}),
            coco=metric.model_copy(update={"method": "coco_rfdetr"}),
            classic=metric,
        ),
    )
    provenance = {
        "input_bundle_id": "a" * 64,
        "evaluation_manifest_id": "b" * 64,
        "bootstrap_dataset_id": "c" * 64,
        "training_run_hash": "d" * 64,
        "training_manifest_sha256": "0" * 64,
        "trained_checkpoint_sha256": "e" * 64,
        "coco_weight_sha256": "f" * 64,
    }
    published = publish_benchmark(
        execution=execution,
        frames=frames,
        circles=circles,
        provenance=provenance,
        output_root=tmp_path,
    )
    overlay = next((published.path / "overlays").glob("*.jpg"))
    overlay.chmod(0o600)
    overlay.write_bytes(b"changed")
    overlay.chmod(0o444)
    manifest_path = published.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = overlay.relative_to(published.path).as_posix()
    item = next(entry for entry in manifest["inventory"] if entry["path"] == relative)
    item["bytes"] = len(b"changed")
    item["sha256"] = hashlib.sha256(b"changed").hexdigest()
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)

    with pytest.raises(ValueError, match="benchmark publication changed"):
        publish_benchmark(
            execution=execution,
            frames=frames,
            circles=circles,
            provenance=provenance,
            output_root=tmp_path,
        )


def test_failed_final_validation_never_leaves_a_formal_artifact_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = ("video1", 10)
    frames = {key: np.zeros((40, 60, 3), dtype=np.uint8)}
    circles = (_circle("gt-a", 10.0),)
    predictions = adapt_classic_predictions(
        asset_id="video1",
        frame_index=10,
        detections=_Classic().detect(frames[key]),
    )
    metric = evaluate_method(method="classic_cv", circles=circles, predictions={key: predictions})
    execution = BenchmarkExecution(
        predictions={
            "trained_rfdetr": {key: predictions},
            "coco_rfdetr": {key: predictions},
            "classic_cv": {key: predictions},
        },
        metrics={
            "trained_rfdetr": metric.model_copy(update={"method": "trained_rfdetr"}),
            "coco_rfdetr": metric.model_copy(update={"method": "coco_rfdetr"}),
            "classic_cv": metric,
        },
        gates=compute_benchmark_gates(
            trained=metric.model_copy(update={"method": "trained_rfdetr"}),
            coco=metric.model_copy(update={"method": "coco_rfdetr"}),
            classic=metric,
        ),
    )

    def reject_publication(*args: object, **kwargs: object) -> None:
        raise ValueError("forced final verification failure")

    monkeypatch.setattr(benchmark_module, "_verify_publication", reject_publication)
    with pytest.raises(ValueError, match="forced final verification failure"):
        publish_benchmark(
            execution=execution,
            frames=frames,
            circles=circles,
            provenance={
                "input_bundle_id": "a" * 64,
                "evaluation_manifest_id": "b" * 64,
                "bootstrap_dataset_id": "c" * 64,
                "training_run_hash": "d" * 64,
                "training_manifest_sha256": "0" * 64,
                "trained_checkpoint_sha256": "e" * 64,
                "coco_weight_sha256": "f" * 64,
            },
            output_root=tmp_path,
        )

    publication_root = tmp_path / "frozen-detector-benchmarks"
    assert not publication_root.exists() or tuple(publication_root.iterdir()) == ()
