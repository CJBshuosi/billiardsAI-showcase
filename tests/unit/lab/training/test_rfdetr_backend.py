from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.training.detector_dataset import create_synthetic_rfdetr_fixture
from billiards_ai.lab.training.rfdetr_backend import (
    DetectorOutOfMemoryError,
    RfDetrTrainingBackend,
)
from billiards_ai.lab.training.rfdetr_train import DetectorTrainingError


def _config() -> dict[str, object]:
    return {
        "candidate_id": "rfdetr-seg-small-smoke-r1",
        "model_class": "RFDETRSegSmall",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "seed": 20260801,
        "resolution": [384, 384],
        "batch_size": 1,
        "epochs": 1,
        "max_train_batches": 1,
        "synthetic_only": True,
        "augmentation_backend": "albumentations",
        "augmentation": {
            "horizontal_flip": False,
            "vertical_flip": False,
            "calibration_transform_required": True,
        },
    }


def _bootstrap_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "candidate_id": "rfdetr-seg-small-bootstrap-short-r1",
        "model_class": "RFDETRSegSmall",
        "package_asset_id": "rfdetr-package-1.9.0",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "seed": 20260801,
        "resolution": [384, 384],
        "batch_size": 1,
        "grad_accum_steps": 4,
        "epochs": 12,
        "num_workers": 0,
        "accelerator": "cuda",
        "amp": True,
        "synthetic_only": False,
        "dataset_kind": "sam2_rfdetr_bootstrap",
        "training_purpose": "local_bootstrap_short",
        "augmentation_backend": "torchvision",
        "geometry_augmentation": False,
        "use_ema": False,
        "early_stopping": False,
        "baseline_validation": True,
        "export_onnx": True,
        "scientific_use": "development_short_train_not_locked_accuracy",
    }


def _weight(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "rf-detr-seg-s-ft.pth"
    path.write_bytes(b"approved synthetic test weight")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeModel:
    def __init__(self, events: list[tuple[str, object]], **kwargs: object) -> None:
        events.append(("construct", kwargs))
        self.model_config = SimpleNamespace(model_name=None)

    def get_train_config(self, **kwargs: object) -> SimpleNamespace:
        events = kwargs.pop("_events") if "_events" in kwargs else None
        if events is not None:
            events.append(("train_config", kwargs))
        return SimpleNamespace(**kwargs)


class _FakeCudaOutOfMemoryError(RuntimeError):
    pass


def _fake_runtime(
    events: list[tuple[str, object]], *, failure_stage: str | None = None
) -> tuple[object, object]:
    class FakeSegSmall(_FakeModel):
        def __init__(self, **kwargs: object) -> None:
            if failure_stage == "model_construction":
                raise _FakeCudaOutOfMemoryError("construction OOM")
            super().__init__(events, **kwargs)

        def get_train_config(self, **kwargs: object) -> SimpleNamespace:
            events.append(("train_config", kwargs))
            return SimpleNamespace(**kwargs)

    class ReloadedModel:
        def export(self, **kwargs: object) -> Path:
            if failure_stage == "onnx_export":
                raise _FakeCudaOutOfMemoryError("export OOM")
            events.append(("export", kwargs))
            output = Path(str(kwargs["output_dir"])) / "model.onnx"
            output.write_bytes(b"real-fake-onnx")
            return output

    class FakeApi:
        RFDETRSegSmall = FakeSegSmall

        @staticmethod
        def from_checkpoint(path: str) -> ReloadedModel:
            if failure_stage == "checkpoint_reload":
                raise _FakeCudaOutOfMemoryError("reload OOM")
            events.append(("reload", path))
            return ReloadedModel()

    class FakeModelModule:
        def __init__(self, model_config: object, train_config: object) -> None:
            events.append(("model_module", (model_config, train_config)))

    class FakeDataModule:
        def __init__(self, model_config: object, train_config: object) -> None:
            events.append(("data_module", (model_config, train_config)))

    class FakeTrainer:
        def __init__(self, train_config: SimpleNamespace) -> None:
            self._train_config = train_config
            self._validation_count = 0

        def validate(
            self,
            module: object,
            *args: object,
            datamodule: object | None = None,
        ) -> list[dict[str, float]]:
            self._validation_count += 1
            events.append(
                (
                    "validate",
                    {
                        "datamodule": datamodule,
                        "module": module,
                        "positional_count": len(args),
                    },
                )
            )
            value = 0.05 if self._validation_count == 1 else 0.35
            return [{"val/segm_mAP_50": value, "val/segm_mAP_50:95": value / 2}]

        def fit(
            self,
            module: object,
            *args: object,
            datamodule: object | None = None,
        ) -> None:
            if failure_stage == "training":
                raise _FakeCudaOutOfMemoryError("training OOM")
            selected_datamodule = datamodule if datamodule is not None else args[0]
            events.append(("fit", (module, selected_datamodule)))
            events.append(
                (
                    "fit_binding",
                    {
                        "keyword": datamodule is not None,
                        "positional_count": len(args),
                    },
                )
            )
            output = Path(self._train_config.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "checkpoint_best_total.pth").write_bytes(b"trained-checkpoint")
            (output / "metrics.csv").write_text(
                "epoch,step,train/loss_epoch\n0,0,0.375\n",
                encoding="utf-8",
            )

    class FakeTrainingApi:
        RFDETRModelModule = FakeModelModule
        RFDETRDataModule = FakeDataModule

        @staticmethod
        def seed_everything(seed: int, *, workers: bool) -> None:
            events.append(("seed", (seed, workers)))

        @staticmethod
        def build_trainer(
            train_config: SimpleNamespace, model_config: object, **kwargs: object
        ) -> FakeTrainer:
            events.append(("build_trainer", kwargs))
            return FakeTrainer(train_config)

    return FakeApi(), FakeTrainingApi()


def _fake_torch_runtime(*, available: bool = True) -> object:
    class FakeCuda:
        OutOfMemoryError = _FakeCudaOutOfMemoryError

        @staticmethod
        def is_available() -> bool:
            return available

        @staticmethod
        def device_count() -> int:
            return 1 if available else 0

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA GeForce GTX 1650 Ti"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (7, 5)

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            assert index == 0
            return (3_000_000_000, 4_000_000_000)

    return SimpleNamespace(
        __version__="2.13.0+cu126",
        version=SimpleNamespace(cuda="12.6"),
        cuda=FakeCuda(),
    )


def test_backend_is_lazy_and_rejects_weight_hash_before_import(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    imports: list[str] = []
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256="0" * 64,
        module_loader=lambda name: imports.append(name),
        version_resolver=lambda _name: "1.9.0",
    )

    assert imports == []
    with pytest.raises(DetectorTrainingError, match="weight SHA-256 mismatch"):
        backend.train(
            dataset_dir=tmp_path,
            config=_config(),
            checkpoint_path=tmp_path / "checkpoint.pt",
            log_path=tmp_path / "training.jsonl",
        )

    assert digest != "0" * 64
    assert imports == []


def test_backend_identity_evidence_binds_model_and_weight(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=lambda _name: pytest.fail("identity must not import the runtime"),
        version_resolver=lambda _name: "1.9.0",
    )

    assert backend.identity_evidence() == {
        "model_class": "RFDETRSegSmall",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "weight_sha256": digest,
    }


def test_backend_rejects_unexpected_distribution_version_before_import(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    imports: list[str] = []
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=lambda name: imports.append(name),
        version_resolver=lambda _name: "1.8.1",
    )

    with pytest.raises(DetectorTrainingError, match="requires rfdetr==1.9.0; found 1.8.1"):
        backend.train(
            dataset_dir=tmp_path,
            config=_config(),
            checkpoint_path=tmp_path / "checkpoint.pt",
            log_path=tmp_path / "training.jsonl",
        )

    assert imports == []


def test_backend_fails_closed_when_rfdetr_runtime_is_unavailable(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)

    def missing(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'rfdetr'", name="rfdetr")

    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=missing,
        version_resolver=lambda _name: "1.9.0",
    )

    with pytest.raises(DetectorTrainingError, match="RF-DETR training runtime is unavailable"):
        backend.train(
            dataset_dir=tmp_path,
            config=_config(),
            checkpoint_path=tmp_path / "checkpoint.pt",
            log_path=tmp_path / "training.jsonl",
        )


def test_backend_runs_exactly_one_training_batch_and_standardizes_loss(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    events: list[tuple[str, object]] = []
    api, training_api = _fake_runtime(events)
    modules = {"rfdetr": api, "rfdetr.training": training_api}
    dataset = create_synthetic_rfdetr_fixture(tmp_path / "dataset").output
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=modules.__getitem__,
        version_resolver=lambda _name: "1.9.0",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    log = tmp_path / "training.jsonl"

    backend.train(
        dataset_dir=dataset,
        config=_config(),
        checkpoint_path=checkpoint,
        log_path=log,
    )
    backend.reload(checkpoint)
    backend.export_onnx(checkpoint_path=checkpoint, output_path=tmp_path / "model.onnx")

    build_kwargs = next(value for name, value in events if name == "build_trainer")
    construct_kwargs = next(value for name, value in events if name == "construct")
    assert "device" not in construct_kwargs
    assert "amp" not in construct_kwargs
    assert build_kwargs == {
        "accelerator": "cpu",
        "devices": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "num_sanity_val_steps": 0,
    }
    train_config = next(value for name, value in events if name == "train_config")
    assert train_config["batch_size"] == 1
    assert train_config["epochs"] == 1
    assert train_config["grad_accum_steps"] == 1
    assert train_config["aug_config"] == {}
    assert train_config["scale_jitter"] is False
    assert checkpoint.read_bytes() == b"trained-checkpoint"
    assert json.loads(log.read_text(encoding="utf-8")) == {
        "batch": 1,
        "epoch": 1,
        "loss": 0.375,
        "upstream_metric": "train/loss_epoch",
    }
    assert (tmp_path / "model.onnx").read_bytes() == b"real-fake-onnx"


def test_real_backend_refuses_a_multi_batch_or_multi_epoch_config(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=lambda _name: pytest.fail("runtime should not be imported"),
        version_resolver=lambda _name: "1.9.0",
    )
    config = _config()
    config["epochs"] = 2

    with pytest.raises(DetectorTrainingError, match="exactly one training batch"):
        backend.train(
            dataset_dir=tmp_path,
            config=config,
            checkpoint_path=tmp_path / "checkpoint.pt",
            log_path=tmp_path / "training.jsonl",
        )


def test_backend_rejects_wrong_seed_before_loading_runtime(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=lambda _name: pytest.fail("runtime must not load for wrong seed"),
        version_resolver=lambda _name: pytest.fail("version must not resolve for wrong seed"),
    )
    config = _config()
    config["seed"] = 9

    with pytest.raises(DetectorTrainingError, match="seed 20260801"):
        backend.train(
            dataset_dir=tmp_path,
            config=config,
            checkpoint_path=tmp_path / "checkpoint.pt",
            log_path=tmp_path / "training.jsonl",
        )


def test_cuda_backend_selects_gpu_mixed_precision_and_reports_device(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    events: list[tuple[str, object]] = []
    api, training_api = _fake_runtime(events)
    modules = {
        "rfdetr": api,
        "rfdetr.training": training_api,
        "torch": _fake_torch_runtime(),
    }
    dataset = create_synthetic_rfdetr_fixture(tmp_path / "dataset").output
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        accelerator="cuda",
        amp=True,
        module_loader=modules.__getitem__,
        version_resolver=lambda _name: "1.9.0",
    )

    backend.validate_runtime()
    backend.train(
        dataset_dir=dataset,
        config=_config(),
        checkpoint_path=tmp_path / "checkpoint.pt",
        log_path=tmp_path / "training.jsonl",
    )

    build_kwargs = next(value for name, value in events if name == "build_trainer")
    construct_kwargs = next(value for name, value in events if name == "construct")
    assert construct_kwargs["device"] == "cuda"
    assert construct_kwargs["amp"] is True
    assert build_kwargs["accelerator"] == "gpu"
    assert build_kwargs["precision"] == "16-mixed"
    assert backend.runtime_evidence() == {
        "accelerator": "cuda",
        "amp": True,
        "cuda_available": True,
        "cuda_runtime": "12.6",
        "device_capability": "7.5",
        "device_count": 1,
        "device_name": "NVIDIA GeForce GTX 1650 Ti",
        "memory_free_bytes": 3_000_000_000,
        "memory_total_bytes": 4_000_000_000,
        "torch_version": "2.13.0+cu126",
    }
    assert backend.identity_evidence()["accelerator"] == "cuda"
    assert backend.identity_evidence()["precision"] == "16-mixed"


def test_cuda_backend_runs_fixed_bootstrap_short_train_with_before_after_validation(
    tmp_path: Path,
) -> None:
    weight, digest = _weight(tmp_path)
    events: list[tuple[str, object]] = []
    api, training_api = _fake_runtime(events)
    modules = {
        "rfdetr": api,
        "rfdetr.training": training_api,
        "torch": _fake_torch_runtime(),
    }
    dataset = tmp_path / "bootstrap"
    for split in ("train", "valid"):
        split_dir = dataset / split
        split_dir.mkdir(parents=True)
        (split_dir / "_annotations.coco.json").write_text("{}", encoding="utf-8")
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        accelerator="cuda",
        amp=True,
        module_loader=modules.__getitem__,
        version_resolver=lambda _name: "1.9.0",
    )
    checkpoint = tmp_path / "checkpoint_best.pth"
    log = tmp_path / "training.jsonl"
    metrics = tmp_path / "metrics.json"
    upstream_metrics = tmp_path / "metrics.csv"

    backend.train_bootstrap_short(
        dataset_dir=dataset,
        config=_bootstrap_config(),
        checkpoint_path=checkpoint,
        log_path=log,
        metrics_path=metrics,
        upstream_metrics_path=upstream_metrics,
    )

    train_config = next(value for name, value in events if name == "train_config")
    build_kwargs = next(value for name, value in events if name == "build_trainer")
    assert train_config["batch_size"] == 1
    assert train_config["grad_accum_steps"] == 4
    assert train_config["epochs"] == 12
    assert train_config["num_workers"] == 0
    assert train_config["use_ema"] is False
    assert train_config["multi_scale"] is False
    assert train_config["aug_config"] == {}
    assert build_kwargs == {
        "accelerator": "gpu",
        "devices": 1,
        "num_sanity_val_steps": 0,
        "precision": "16-mixed",
    }
    assert [name for name, _value in events].count("validate") == 2
    assert [value["positional_count"] for name, value in events if name == "validate"] == [0, 0]
    assert next(value for name, value in events if name == "fit_binding") == {
        "keyword": True,
        "positional_count": 0,
    }
    assert checkpoint.read_bytes() == b"trained-checkpoint"
    assert upstream_metrics.read_text(encoding="utf-8").startswith("epoch,step")
    assert json.loads(log.read_text(encoding="utf-8")) == {
        "epoch": 1,
        "loss": 0.375,
        "upstream_metric": "train/loss_epoch",
    }
    assert json.loads(metrics.read_text(encoding="utf-8")) == {
        "after_short_train": {
            "val/segm_mAP_50": 0.35,
            "val/segm_mAP_50:95": 0.175,
        },
        "before_finetune": {
            "val/segm_mAP_50": 0.05,
            "val/segm_mAP_50:95": 0.025,
        },
        "delta": {
            "val/segm_mAP_50": 0.3,
            "val/segm_mAP_50:95": 0.15,
        },
    }


def test_bootstrap_backend_refuses_to_weaken_the_fixed_short_train(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        accelerator="cuda",
        amp=True,
        module_loader=lambda _name: pytest.fail("runtime must not load"),
        version_resolver=lambda _name: pytest.fail("version must not resolve"),
    )
    config = _bootstrap_config()
    config["epochs"] = 13

    with pytest.raises(DetectorTrainingError, match="fixed 12-epoch contract"):
        backend.train_bootstrap_short(
            dataset_dir=tmp_path,
            config=config,
            checkpoint_path=tmp_path / "checkpoint.pth",
            log_path=tmp_path / "training.jsonl",
            metrics_path=tmp_path / "metrics.json",
            upstream_metrics_path=tmp_path / "metrics.csv",
        )


def test_cuda_backend_fails_closed_when_cuda_is_unavailable(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    api, training_api = _fake_runtime([])
    modules = {
        "rfdetr": api,
        "rfdetr.training": training_api,
        "torch": _fake_torch_runtime(available=False),
    }
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        accelerator="cuda",
        amp=True,
        module_loader=modules.__getitem__,
        version_resolver=lambda _name: "1.9.0",
    )

    with pytest.raises(DetectorTrainingError, match="CUDA is unavailable"):
        backend.validate_runtime()


def test_cpu_backend_keeps_existing_identity_evidence(tmp_path: Path) -> None:
    weight, digest = _weight(tmp_path)
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        module_loader=lambda _name: pytest.fail("identity must not import the runtime"),
        version_resolver=lambda _name: "1.9.0",
    )

    assert backend.identity_evidence() == {
        "model_class": "RFDETRSegSmall",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "weight_sha256": digest,
    }


@pytest.mark.parametrize(
    "failure_stage",
    ["model_construction", "training", "checkpoint_reload", "onnx_export"],
)
def test_cuda_oom_remains_typed_across_every_backend_stage(
    tmp_path: Path, failure_stage: str
) -> None:
    weight, digest = _weight(tmp_path)
    events: list[tuple[str, object]] = []
    api, training_api = _fake_runtime(events, failure_stage=failure_stage)
    modules = {
        "rfdetr": api,
        "rfdetr.training": training_api,
        "torch": _fake_torch_runtime(),
    }
    dataset = create_synthetic_rfdetr_fixture(tmp_path / "dataset").output
    backend = RfDetrTrainingBackend(
        weight_path=weight,
        expected_weight_sha256=digest,
        accelerator="cuda",
        amp=True,
        module_loader=modules.__getitem__,
        version_resolver=lambda _name: "1.9.0",
    )
    checkpoint = tmp_path / "checkpoint.pt"

    with pytest.raises(DetectorOutOfMemoryError) as captured:
        backend.train(
            dataset_dir=dataset,
            config=_config(),
            checkpoint_path=checkpoint,
            log_path=tmp_path / "training.jsonl",
        )
        backend.reload(checkpoint)
        backend.export_onnx(checkpoint_path=checkpoint, output_path=tmp_path / "model.onnx")

    assert captured.value.stage == failure_stage
