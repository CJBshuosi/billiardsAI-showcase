import hashlib
import json
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.training.detector_dataset import create_synthetic_rfdetr_fixture
from billiards_ai.lab.training.rfdetr_train import (
    DetectorTrainingError,
    SmokeTrainingConfig,
    run_detector_training_smoke,
)


class SyntheticTrainingBackend:
    name = "synthetic-rfdetr"
    version = "1.0"

    def train(
        self,
        *,
        dataset_dir: Path,
        config: dict,
        checkpoint_path: Path,
        log_path: Path,
    ) -> None:
        assert (dataset_dir / "annotations" / "instances_train.json").is_file()
        checkpoint_path.write_bytes(f"checkpoint:{config['seed']}".encode())
        log_path.write_text(
            json.dumps({"epoch": 1, "loss": 0.25, "synthetic": True}) + "\n",
            encoding="utf-8",
        )

    def reload(self, checkpoint_path: Path) -> None:
        assert checkpoint_path.read_bytes().startswith(b"checkpoint:")

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"onnx:" + checkpoint_path.read_bytes())


def _dataset(tmp_path: Path) -> Path:
    return create_synthetic_rfdetr_fixture(tmp_path / "dataset").output


def test_synthetic_smoke_records_reproducible_hashes_and_artifacts(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    config = {
        "schema_version": "1.0",
        "candidate_id": "rfdetr-seg-small-smoke-r1",
        "model_class": "RFDETRSegSmall",
        "seed": 20260801,
        "batch_size": 1,
        "epochs": 3,
        "synthetic_only": True,
        "augmentation": {
            "horizontal_flip": False,
            "vertical_flip": False,
            "calibration_transform_required": True,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    first = run_detector_training_smoke(
        dataset_dir=dataset,
        config_path=config_path,
        runs_root=tmp_path / "runs-a",
        backend=SyntheticTrainingBackend(),
    )
    second = run_detector_training_smoke(
        dataset_dir=dataset,
        config_path=config_path,
        runs_root=tmp_path / "runs-b",
        backend=SyntheticTrainingBackend(),
    )

    assert first.data_hash == second.data_hash
    assert first.config_hash == second.config_hash
    assert first.run_hash == second.run_hash
    assert (first.output / "checkpoint.pt").is_file()
    assert (first.output / "model.onnx").is_file()
    assert (first.output / "training.jsonl").is_file()
    manifest = json.loads((first.output / "training-manifest.json").read_text())
    assert manifest["checkpoint_reloaded"] is True
    assert manifest["onnx_exported"] is True
    assert manifest["synthetic_only"] is True


def test_repository_smoke_config_is_executable_and_geometry_safe(tmp_path: Path) -> None:
    config_path = Path("configs/algorithm/training/rfdetr-smoke.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    run = run_detector_training_smoke(
        dataset_dir=_dataset(tmp_path),
        config_path=config_path,
        runs_root=tmp_path / "runs",
        backend=SyntheticTrainingBackend(),
    )

    assert run.output.is_dir()
    assert config["augmentation"]["horizontal_flip"] is False
    assert config["augmentation"]["vertical_flip"] is False
    assert config["augmentation"]["calibration_transform_required"] is True


def test_repository_real_smoke_config_is_exactly_one_batch() -> None:
    config = SmokeTrainingConfig.model_validate(
        yaml.safe_load(
            Path("configs/algorithm/training/rfdetr-single-batch-smoke.yaml").read_text(
                encoding="utf-8"
            )
        )
    )

    assert config.batch_size == 1
    assert config.epochs == 1
    assert config.max_train_batches == 1
    assert config.seed == 20260801


def test_training_contract_supports_commercial_seg_medium_candidate() -> None:
    config = SmokeTrainingConfig.model_validate(
        {
            "candidate_id": "rfdetr-seg-medium-short-r1",
            "model_class": "RFDETRSegMedium",
            "seed": 20260801,
            "batch_size": 1,
            "epochs": 1,
            "synthetic_only": True,
            "augmentation": {
                "horizontal_flip": False,
                "vertical_flip": False,
                "calibration_transform_required": True,
            },
        }
    )

    assert config.model_class == "RFDETRSegMedium"


@pytest.mark.parametrize("loss", ["NaN", "Infinity", "-Infinity", "not-a-number", True])
def test_smoke_rejects_non_finite_or_non_numeric_loss(tmp_path: Path, loss: object) -> None:
    class InvalidLossBackend(SyntheticTrainingBackend):
        def train(
            self,
            *,
            dataset_dir: Path,
            config: dict,
            checkpoint_path: Path,
            log_path: Path,
        ) -> None:
            super().train(
                dataset_dir=dataset_dir,
                config=config,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
            )
            log_path.write_text(
                json.dumps({"epoch": 1, "loss": loss}) + "\n",
                encoding="utf-8",
            )

    config = {
        "candidate_id": "invalid-loss-smoke-r1",
        "model_class": "RFDETRSegSmall",
        "seed": 20260801,
        "batch_size": 1,
        "epochs": 1,
        "synthetic_only": True,
        "augmentation": {
            "horizontal_flip": False,
            "vertical_flip": False,
            "calibration_transform_required": True,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(DetectorTrainingError, match="finite numeric loss"):
        run_detector_training_smoke(
            dataset_dir=_dataset(tmp_path),
            config_path=config_path,
            runs_root=tmp_path / "runs",
            backend=InvalidLossBackend(),
        )


def test_single_batch_limit_is_part_of_the_config_hash(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    base = {
        "candidate_id": "single-batch-hash-r1",
        "model_class": "RFDETRSegSmall",
        "seed": 20260801,
        "batch_size": 1,
        "epochs": 1,
        "synthetic_only": True,
        "augmentation": {
            "horizontal_flip": False,
            "vertical_flip": False,
            "calibration_transform_required": True,
        },
    }
    implicit = tmp_path / "implicit.yaml"
    explicit = tmp_path / "explicit.yaml"
    implicit.write_text(yaml.safe_dump(base), encoding="utf-8")
    explicit.write_text(yaml.safe_dump({**base, "max_train_batches": 1}), encoding="utf-8")

    first = run_detector_training_smoke(
        dataset_dir=dataset,
        config_path=implicit,
        runs_root=tmp_path / "runs-a",
        backend=SyntheticTrainingBackend(),
    )
    second = run_detector_training_smoke(
        dataset_dir=dataset,
        config_path=explicit,
        runs_root=tmp_path / "runs-b",
        backend=SyntheticTrainingBackend(),
    )

    assert first.config_hash == second.config_hash


def test_backend_weight_identity_separates_content_addressed_runs(tmp_path: Path) -> None:
    class WeightBoundBackend(SyntheticTrainingBackend):
        def __init__(self, weight_sha256: str) -> None:
            self._weight_sha256 = weight_sha256

        def identity_evidence(self) -> dict[str, str]:
            return {
                "model_class": "RFDETRSegSmall",
                "weight_asset_id": "rfdetr-seg-small-1.9.0",
                "weight_sha256": self._weight_sha256,
            }

    dataset = _dataset(tmp_path)
    config_path = Path("configs/algorithm/training/rfdetr-single-batch-smoke.yaml")
    first = run_detector_training_smoke(
        dataset_dir=dataset,
        config_path=config_path,
        runs_root=tmp_path / "runs",
        backend=WeightBoundBackend("a" * 64),
    )
    second = run_detector_training_smoke(
        dataset_dir=dataset,
        config_path=config_path,
        runs_root=tmp_path / "runs",
        backend=WeightBoundBackend("b" * 64),
    )

    assert first.run_hash != second.run_hash
    assert first.output != second.output
    first_manifest = json.loads((first.output / "training-manifest.json").read_text())
    second_manifest = json.loads((second.output / "training-manifest.json").read_text())
    assert first_manifest["backend"]["weight_sha256"] == "a" * 64
    assert second_manifest["backend"]["weight_sha256"] == "b" * 64


def test_smoke_rejects_tampered_rfdetr_projection_before_cache_lookup(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    projected_image = dataset / "train" / "frame-1.png"
    projected_image.chmod(0o644)
    projected_image.write_bytes(b"tampered")

    with pytest.raises(DetectorTrainingError, match="RF-DETR layout hash mismatch"):
        run_detector_training_smoke(
            dataset_dir=dataset,
            config_path=Path("configs/algorithm/training/rfdetr-single-batch-smoke.yaml"),
            runs_root=tmp_path / "runs",
            backend=SyntheticTrainingBackend(),
        )


def test_smoke_rejects_recomputed_manifest_with_incomplete_layout_evidence(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    derived_path = dataset / "derived-manifest.json"
    derived_path.chmod(0o644)
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived.pop("data_hash")
    derived["rfdetr_layout_hashes"].pop("train/frame-1.png")
    coco = json.loads(
        (dataset / "annotations" / "instances_train.json").read_text(encoding="utf-8")
    )

    def canonical_bytes(value: object) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    derived["data_hash"] = hashlib.sha256(
        canonical_bytes({"coco": coco, "derived_manifest": derived})
    ).hexdigest()
    derived_path.write_bytes(canonical_bytes(derived))

    with pytest.raises(DetectorTrainingError, match="layout hash evidence is incomplete"):
        run_detector_training_smoke(
            dataset_dir=dataset,
            config_path=Path("configs/algorithm/training/rfdetr-single-batch-smoke.yaml"),
            runs_root=tmp_path / "runs",
            backend=SyntheticTrainingBackend(),
        )
