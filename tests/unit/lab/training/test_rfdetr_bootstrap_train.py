from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.training.rfdetr_bootstrap_train import (
    BootstrapShortTrainingConfig,
    run_bootstrap_short_training,
)
from billiards_ai.lab.training.rfdetr_train import DetectorTrainingError


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_dataset(tmp_path: Path) -> Path:
    pending = tmp_path / "pending"
    for split, asset_id in (("train", "video-train"), ("valid", "video-valid")):
        split_dir = pending / split
        split_dir.mkdir(parents=True)
        image_name = f"{asset_id}.jpg"
        (split_dir / image_name).write_bytes(f"jpeg:{asset_id}".encode())
        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": image_name,
                    "width": 1920,
                    "height": 1080,
                    "source_asset_id": asset_id,
                    "source_frame_index": 15,
                }
            ],
            "annotations": [
                {
                    "id": 1 if split == "train" else 2,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0]],
                    "bbox": [10.0, 10.0, 10.0, 10.0],
                    "area": 100,
                    "iscrowd": 0,
                    "instance_id": f"ball-{asset_id}",
                }
            ],
            "categories": [{"id": 1, "name": "ball", "supercategory": "billiards"}],
        }
        (split_dir / "_annotations.coco.json").write_bytes(_canonical_bytes(coco))
    inventory = [
        {
            "path": path.relative_to(pending).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(pending.rglob("*"))
        if path.is_file()
    ]
    identity = {
        "schema_version": "1.0",
        "kind": "sam2_rfdetr_bootstrap",
        "source_provenance": {
            "evaluation_manifest_hash": "c" * 64,
            "input_bundle_id": "b" * 64,
            "raw_run_hash": "a" * 64,
            "verdict_artifact_hash": "d" * 64,
        },
        "policy": {
            "benchmark_frames_excluded": True,
            "frame_complete_labels_required": True,
            "qa_flags_allowed": False,
            "scientific_use": "development_bootstrap_not_independent_accuracy",
        },
        "frame_stride": 15,
        "split_assets": {"train": ["video-train"], "valid": ["video-valid"]},
        "rejection_counts": {"qa_flagged": 1},
        "image_count": 2,
        "annotation_count": 2,
        "inventory": inventory,
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    dataset = tmp_path / artifact_id
    pending.rename(dataset)
    (dataset / "manifest.json").write_bytes(
        _canonical_bytes({"artifact_id": artifact_id, **identity})
    )
    if os.name != "nt":
        for path in sorted(dataset.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        dataset.chmod(0o555)
    return dataset


class _BootstrapBackend:
    name = "bootstrap-test-backend"
    version = "1.9.0"

    def identity_evidence(self) -> dict[str, str]:
        return {
            "accelerator": "cuda",
            "model_class": "RFDETRSegSmall",
            "precision": "16-mixed",
            "weight_asset_id": "rfdetr-seg-small-1.9.0",
            "weight_sha256": "e" * 64,
        }

    def train_bootstrap_short(
        self,
        *,
        dataset_dir: Path,
        config: dict[str, object],
        checkpoint_path: Path,
        log_path: Path,
        metrics_path: Path,
        upstream_metrics_path: Path,
    ) -> None:
        assert (dataset_dir / "train" / "_annotations.coco.json").is_file()
        assert config["epochs"] == 12
        assert config["batch_size"] == 1
        assert config["grad_accum_steps"] == 4
        assert config["resolution"] == [384, 384]
        assert config["synthetic_only"] is False
        checkpoint_path.write_bytes(b"best-checkpoint")
        log_path.write_text(
            '{"epoch":1,"loss":2.5}\n{"epoch":12,"loss":0.75}\n',
            encoding="utf-8",
        )
        metrics_path.write_bytes(
            _canonical_bytes(
                {
                    "before_finetune": {"val/segm_mAP_50": 0.05},
                    "after_short_train": {"val/segm_mAP_50": 0.35},
                    "delta": {"val/segm_mAP_50": 0.3},
                }
            )
        )
        upstream_metrics_path.write_text(
            "epoch,train/loss_epoch,val/segm_mAP_50\n11,0.75,0.35\n",
            encoding="utf-8",
        )

    def reload(self, checkpoint_path: Path) -> None:
        assert checkpoint_path.read_bytes() == b"best-checkpoint"

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"onnx:" + checkpoint_path.read_bytes())


def test_repository_config_locks_the_approved_short_training_shape() -> None:
    config = BootstrapShortTrainingConfig.model_validate(
        yaml.safe_load(
            Path("configs/algorithm/training/rfdetr-seg-small-bootstrap-short.yaml").read_text(
                encoding="utf-8"
            )
        )
    )

    assert config.model_class == "RFDETRSegSmall"
    assert config.resolution == (384, 384)
    assert config.batch_size == 1
    assert config.grad_accum_steps == 4
    assert config.epochs == 12
    assert config.accelerator == "cuda"
    assert config.amp is True
    assert config.synthetic_only is False


def test_short_training_publishes_reproducible_comparison_and_model_artifacts(
    tmp_path: Path,
) -> None:
    dataset = _bootstrap_dataset(tmp_path)

    first = run_bootstrap_short_training(
        dataset_dir=dataset,
        config_path=Path("configs/algorithm/training/rfdetr-seg-small-bootstrap-short.yaml"),
        runs_root=tmp_path / "runs-a",
        backend=_BootstrapBackend(),
    )
    second = run_bootstrap_short_training(
        dataset_dir=dataset,
        config_path=Path("configs/algorithm/training/rfdetr-seg-small-bootstrap-short.yaml"),
        runs_root=tmp_path / "runs-b",
        backend=_BootstrapBackend(),
    )

    assert first.dataset_id == dataset.name
    assert first.config_hash == second.config_hash
    assert first.run_hash == second.run_hash
    assert (first.output / "checkpoint_best.pth").read_bytes() == b"best-checkpoint"
    assert (first.output / "model.onnx").is_file()
    assert (first.output / "metrics.csv").is_file()
    metrics = json.loads((first.output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["before_finetune"]["val/segm_mAP_50"] == 0.05
    assert metrics["after_short_train"]["val/segm_mAP_50"] == 0.35
    assert metrics["delta"]["val/segm_mAP_50"] == 0.3
    manifest = json.loads((first.output / "training-manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == dataset.name
    assert manifest["dataset_counts"] == {
        "annotations": 2,
        "images": 2,
        "train_images": 1,
        "valid_images": 1,
    }
    assert manifest["scientific_use"] == "development_short_train_not_locked_accuracy"
    assert manifest["synthetic_only"] is False
    assert set(manifest["artifacts"]) == {
        "checkpoint_best.pth",
        "metrics.csv",
        "metrics.json",
        "model.onnx",
        "training.jsonl",
    }


def test_short_training_rejects_tampered_bootstrap_before_backend_runs(
    tmp_path: Path,
) -> None:
    dataset = _bootstrap_dataset(tmp_path)
    image = dataset / "train" / "video-train.jpg"
    image.chmod(0o644)
    image.write_bytes(b"tampered")

    with pytest.raises(DetectorTrainingError, match="bootstrap dataset payload changed"):
        run_bootstrap_short_training(
            dataset_dir=dataset,
            config_path=Path("configs/algorithm/training/rfdetr-seg-small-bootstrap-short.yaml"),
            runs_root=tmp_path / "runs",
            backend=_BootstrapBackend(),
        )


def test_short_training_refuses_a_tampered_cached_model_artifact(tmp_path: Path) -> None:
    dataset = _bootstrap_dataset(tmp_path)
    arguments = {
        "dataset_dir": dataset,
        "config_path": Path("configs/algorithm/training/rfdetr-seg-small-bootstrap-short.yaml"),
        "runs_root": tmp_path / "runs",
        "backend": _BootstrapBackend(),
    }
    first = run_bootstrap_short_training(**arguments)
    checkpoint = first.output / "checkpoint_best.pth"
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(DetectorTrainingError, match="cached bootstrap training artifact changed"):
        run_bootstrap_short_training(**arguments)


def test_short_training_rejects_overlapping_video_splits(tmp_path: Path) -> None:
    dataset = _bootstrap_dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_assets"]["valid"] = ["video-train"]
    manifest_path.write_bytes(_canonical_bytes(manifest))
    if os.name != "nt":
        manifest_path.chmod(0o444)

    with pytest.raises(DetectorTrainingError, match="video-disjoint"):
        run_bootstrap_short_training(
            dataset_dir=dataset,
            config_path=Path("configs/algorithm/training/rfdetr-seg-small-bootstrap-short.yaml"),
            runs_root=tmp_path / "runs",
            backend=_BootstrapBackend(),
        )
