"""Lazy, fail-closed RF-DETR 1.9.0 backend for one-batch synthetic smoke runs."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from billiards_ai.lab.training.rfdetr_train import DetectorTrainingError

_EXPECTED_VERSION = "1.9.0"
_SUPPORTED_MODEL_CLASSES = frozenset(
    {
        "RFDETRSegSmall",
        "RFDETRSegMedium",
        "RFDETRSegLarge",
        "RFDETRSegXLarge",
    }
)


class DetectorOutOfMemoryError(DetectorTrainingError):
    """CUDA ran out of memory at a named RF-DETR lifecycle stage."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"RF-DETR CUDA out of memory during {stage}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class RfDetrTrainingBackend:
    """Adapt RF-DETR's exported Lightning components to the smoke backend contract."""

    name = "rfdetr"

    def __init__(
        self,
        *,
        weight_path: Path,
        expected_weight_sha256: str,
        expected_version: str = _EXPECTED_VERSION,
        model_class: str = "RFDETRSegSmall",
        weight_asset_id: str = "rfdetr-seg-small-1.9.0",
        accelerator: str = "cpu",
        amp: bool = False,
        module_loader: Callable[[str], Any] = importlib.import_module,
        version_resolver: Callable[[str], str] = importlib.metadata.version,
    ) -> None:
        if len(expected_weight_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_weight_sha256
        ):
            raise ValueError("expected_weight_sha256 must be 64 lowercase hexadecimal characters")
        if expected_version != _EXPECTED_VERSION:
            raise ValueError(f"only RF-DETR {_EXPECTED_VERSION} is supported")
        if model_class not in _SUPPORTED_MODEL_CLASSES:
            raise ValueError("unsupported RF-DETR segmentation model class")
        if not weight_asset_id or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in weight_asset_id
        ):
            raise ValueError("weight_asset_id is invalid")
        if accelerator not in {"cpu", "cuda"}:
            raise ValueError("accelerator must be cpu or cuda")
        if amp and accelerator != "cuda":
            raise ValueError("AMP is supported only with the CUDA accelerator")
        self.weight_path = weight_path
        self.expected_weight_sha256 = expected_weight_sha256
        self.version = expected_version
        self.model_class = model_class
        self.weight_asset_id = weight_asset_id
        self.accelerator = accelerator
        self.amp = amp
        self._module_loader = module_loader
        self._version_resolver = version_resolver
        self._api: Any | None = None
        self._training_api: Any | None = None
        self._reloaded_model: Any | None = None
        self._reloaded_checkpoint: Path | None = None
        self._resolution: tuple[int, int] | None = None
        self._runtime_evidence: dict[str, Any] | None = None
        self._cuda_oom_error_type: type[BaseException] | None = None

    def _validate_config(self, config: dict[str, Any]) -> tuple[int, int]:
        if config.get("seed") != 20260801:
            raise DetectorTrainingError("RF-DETR smoke requires seed 20260801")
        if config.get("synthetic_only") is not True:
            raise DetectorTrainingError("real RF-DETR smoke backend accepts synthetic-only data")
        if config.get("model_class") != self.model_class:
            raise DetectorTrainingError("RF-DETR weight/model class binding mismatch")
        if config.get("weight_asset_id") != self.weight_asset_id:
            raise DetectorTrainingError("RF-DETR weight asset binding mismatch")
        if (
            config.get("batch_size") != 1
            or config.get("epochs") != 1
            or config.get("max_train_batches", 1) != 1
        ):
            raise DetectorTrainingError("RF-DETR smoke requires exactly one training batch")
        resolution = config.get("resolution")
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in resolution)
            or resolution[0] <= 0
            or resolution[0] != resolution[1]
        ):
            raise DetectorTrainingError("RF-DETR smoke requires a positive square resolution")
        return int(resolution[0]), int(resolution[1])

    def _validate_bootstrap_config(self, config: dict[str, Any]) -> tuple[int, int]:
        expected = {
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
        if any(config.get(key) != value for key, value in expected.items()):
            raise DetectorTrainingError("RF-DETR bootstrap requires the fixed 12-epoch contract")
        if (
            self.model_class != "RFDETRSegSmall"
            or self.weight_asset_id != expected["weight_asset_id"]
        ):
            raise DetectorTrainingError("RF-DETR bootstrap model/weight binding mismatch")
        if self.accelerator != "cuda" or not self.amp:
            raise DetectorTrainingError("RF-DETR bootstrap requires CUDA 16-bit AMP")
        return (384, 384)

    def _load_runtime(self) -> tuple[Any, Any]:
        if self._api is not None and self._training_api is not None:
            return self._api, self._training_api
        try:
            actual_sha256 = _sha256_file(self.weight_path)
        except OSError as exc:
            raise DetectorTrainingError("approved RF-DETR weight is unavailable") from exc
        if actual_sha256 != self.expected_weight_sha256:
            raise DetectorTrainingError("RF-DETR weight SHA-256 mismatch")
        try:
            actual_version = self._version_resolver("rfdetr")
        except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError) as exc:
            raise DetectorTrainingError(
                "RF-DETR training runtime is unavailable; install the approved 1.9.0 runtime "
                "and train/onnx extras in an isolated environment"
            ) from exc
        if actual_version != self.version:
            raise DetectorTrainingError(
                f"RF-DETR smoke requires rfdetr=={self.version}; found {actual_version}"
            )
        try:
            api = self._module_loader("rfdetr")
            training_api = self._module_loader("rfdetr.training")
        except (ImportError, ModuleNotFoundError) as exc:
            raise DetectorTrainingError(
                "RF-DETR training runtime is unavailable; install the approved 1.9.0 runtime "
                "and train/onnx extras in an isolated environment"
            ) from exc
        required_api = (self.model_class, "from_checkpoint")
        required_training_api = (
            "RFDETRModelModule",
            "RFDETRDataModule",
            "build_trainer",
            "seed_everything",
        )
        if any(not hasattr(api, name) for name in required_api) or any(
            not hasattr(training_api, name) for name in required_training_api
        ):
            raise DetectorTrainingError("RF-DETR 1.9.0 runtime lacks the required training API")
        self._api = api
        self._training_api = training_api
        return api, training_api

    def validate_runtime(self) -> None:
        """Validate approved weights, exact version, imports, and the required API surface."""
        self._load_runtime()
        self.runtime_evidence()

    def runtime_evidence(self) -> dict[str, Any]:
        """Return device evidence for the selected training runtime."""
        if self._runtime_evidence is not None:
            return dict(self._runtime_evidence)
        if self.accelerator == "cpu":
            self._runtime_evidence = {"accelerator": "cpu", "amp": False}
            return dict(self._runtime_evidence)
        try:
            torch = self._module_loader("torch")
            cuda = torch.cuda
            self._cuda_oom_error_type = getattr(cuda, "OutOfMemoryError", None)
            cuda_available = bool(cuda.is_available())
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            raise DetectorTrainingError("PyTorch CUDA runtime is unavailable") from exc
        if not cuda_available or int(cuda.device_count()) < 1:
            raise DetectorTrainingError("CUDA is unavailable for the RF-DETR smoke")
        capability = cuda.get_device_capability(0)
        memory_free, memory_total = cuda.mem_get_info(0)
        self._runtime_evidence = {
            "accelerator": "cuda",
            "amp": self.amp,
            "cuda_available": True,
            "cuda_runtime": str(torch.version.cuda),
            "device_capability": f"{capability[0]}.{capability[1]}",
            "device_count": int(cuda.device_count()),
            "device_name": str(cuda.get_device_name(0)),
            "memory_free_bytes": int(memory_free),
            "memory_total_bytes": int(memory_total),
            "torch_version": str(torch.__version__),
        }
        return dict(self._runtime_evidence)

    def _raise_stage_error(self, stage: str, exc: Exception, message: str) -> None:
        if self._cuda_oom_error_type is not None and isinstance(exc, self._cuda_oom_error_type):
            raise DetectorOutOfMemoryError(stage) from exc
        raise DetectorTrainingError(message) from exc

    def identity_evidence(self) -> dict[str, str]:
        """Return immutable model inputs that must participate in the run hash."""
        evidence = {
            "model_class": self.model_class,
            "weight_asset_id": self.weight_asset_id,
            "weight_sha256": self.expected_weight_sha256,
        }
        if self.accelerator == "cuda":
            evidence.update(
                {
                    "accelerator": "cuda",
                    "precision": "16-mixed" if self.amp else "32-true",
                }
            )
        return evidence

    @staticmethod
    def _read_finite_loss(metrics_path: Path) -> tuple[float, str, int]:
        try:
            with metrics_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        except OSError as exc:
            raise DetectorTrainingError("RF-DETR did not produce metrics.csv") from exc
        for metric in ("train/loss_epoch", "train/loss", "loss"):
            for row in reversed(rows):
                value = row.get(metric)
                if value in (None, ""):
                    continue
                try:
                    loss = float(value)
                except ValueError:
                    continue
                if math.isfinite(loss):
                    try:
                        epoch = int(float(row.get("epoch") or 0)) + 1
                    except ValueError:
                        epoch = 1
                    return loss, metric, epoch
        raise DetectorTrainingError("RF-DETR metrics lack finite training loss evidence")

    @staticmethod
    def _finite_validation_metrics(result: object) -> dict[str, float]:
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
            raise DetectorTrainingError("RF-DETR validation metrics are invalid")
        normalized: dict[str, float] = {}
        for key, raw_value in result[0].items():
            if not isinstance(key, str):
                continue
            value = raw_value
            item = getattr(value, "item", None)
            if callable(item):
                try:
                    value = item()
                except (RuntimeError, TypeError, ValueError):
                    continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                normalized[key] = numeric
        if not normalized:
            raise DetectorTrainingError("RF-DETR validation lacks finite metric evidence")
        return dict(sorted(normalized.items()))

    def train(
        self,
        *,
        dataset_dir: Path,
        config: dict[str, Any],
        checkpoint_path: Path,
        log_path: Path,
    ) -> None:
        resolution = self._validate_config(config)
        api, training_api = self._load_runtime()
        self.runtime_evidence()
        upstream_accelerator = "gpu" if self.accelerator == "cuda" else "cpu"
        try:
            derived = json.loads(
                (dataset_dir / "derived-manifest.json").read_text(encoding="utf-8")
            )
            layout_hashes = derived["rfdetr_layout_hashes"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DetectorTrainingError(
                "RF-DETR synthetic COCO layout lacks hash evidence"
            ) from exc
        if not isinstance(layout_hashes, dict) or not layout_hashes:
            raise DetectorTrainingError("RF-DETR synthetic COCO layout lacks hash evidence")
        for split in ("train", "valid"):
            if not (dataset_dir / split / "_annotations.coco.json").is_file():
                raise DetectorTrainingError(
                    "RF-DETR synthetic COCO train/valid layout is incomplete"
                )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="rfdetr-upstream-", dir=checkpoint_path.parent
            ) as output_name:
                output_dir = Path(output_name)
                model_type = getattr(api, self.model_class)
                model_options: dict[str, Any] = {
                    "pretrain_weights": str(self.weight_path.resolve()),
                    "resolution": resolution[0],
                    "num_classes": 1,
                }
                if self.accelerator == "cuda":
                    model_options.update({"device": "cuda", "amp": self.amp})
                try:
                    model = model_type(**model_options)
                except Exception as exc:
                    self._raise_stage_error(
                        "model_construction",
                        exc,
                        "RF-DETR model construction failed",
                    )
                model.model_config.model_name = self.model_class
                try:
                    train_config = model.get_train_config(
                        dataset_dir=str(dataset_dir.resolve()),
                        output_dir=str(output_dir),
                        batch_size=1,
                        grad_accum_steps=1,
                        epochs=1,
                        checkpoint_interval=1,
                        seed=int(config["seed"]),
                        accelerator=upstream_accelerator,
                        devices=1,
                        num_workers=0,
                        use_ema=False,
                        early_stopping=False,
                        tensorboard=False,
                        wandb=False,
                        mlflow=False,
                        progress_bar=None,
                        multi_scale=False,
                        expanded_scales=False,
                        aug_config={},
                        scale_jitter=False,
                        augmentation_backend=config.get("augmentation_backend") or "torchvision",
                        save_dataset_grids=False,
                        run_test=False,
                    )
                    training_api.seed_everything(int(config["seed"]), workers=True)
                    module = training_api.RFDETRModelModule(model.model_config, train_config)
                    datamodule = training_api.RFDETRDataModule(model.model_config, train_config)
                    trainer_options: dict[str, Any] = {
                        "accelerator": upstream_accelerator,
                        "devices": 1,
                        "limit_train_batches": 1,
                        "limit_val_batches": 1,
                        "num_sanity_val_steps": 0,
                    }
                    if self.amp:
                        trainer_options["precision"] = "16-mixed"
                    trainer = training_api.build_trainer(
                        train_config,
                        model.model_config,
                        **trainer_options,
                    )
                    trainer.fit(module, datamodule)
                except DetectorTrainingError:
                    raise
                except Exception as exc:
                    self._raise_stage_error(
                        "training",
                        exc,
                        "RF-DETR one-batch training failed",
                    )
                upstream_checkpoint = output_dir / "checkpoint_best_total.pth"
                if not upstream_checkpoint.is_file():
                    raise DetectorTrainingError("RF-DETR did not produce checkpoint_best_total.pth")
                loss, metric, epoch = self._read_finite_loss(output_dir / "metrics.csv")
                shutil.copyfile(upstream_checkpoint, checkpoint_path)
                log_path.write_text(
                    json.dumps(
                        {
                            "batch": 1,
                            "epoch": epoch,
                            "loss": loss,
                            "upstream_metric": metric,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except DetectorTrainingError:
            raise
        except Exception as exc:
            raise DetectorTrainingError("RF-DETR one-batch training failed") from exc
        self._resolution = resolution

    def train_bootstrap_short(
        self,
        *,
        dataset_dir: Path,
        config: dict[str, Any],
        checkpoint_path: Path,
        log_path: Path,
        metrics_path: Path,
        upstream_metrics_path: Path,
    ) -> None:
        """Run the separately locked 12-epoch reviewed-label bootstrap experiment."""

        resolution = self._validate_bootstrap_config(config)
        api, training_api = self._load_runtime()
        self.runtime_evidence()
        for split in ("train", "valid"):
            if not (dataset_dir / split / "_annotations.coco.json").is_file():
                raise DetectorTrainingError(
                    "RF-DETR bootstrap COCO train/valid layout is incomplete"
                )
        for path in (checkpoint_path, log_path, metrics_path, upstream_metrics_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="rfdetr-bootstrap-upstream-", dir=checkpoint_path.parent
            ) as output_name:
                output_dir = Path(output_name)
                model_type = getattr(api, self.model_class)
                try:
                    model = model_type(
                        pretrain_weights=str(self.weight_path.resolve()),
                        resolution=resolution[0],
                        num_classes=1,
                        device="cuda",
                        amp=True,
                    )
                except Exception as exc:
                    self._raise_stage_error(
                        "model_construction",
                        exc,
                        "RF-DETR bootstrap model construction failed",
                    )
                model.model_config.model_name = self.model_class
                try:
                    train_config = model.get_train_config(
                        dataset_dir=str(dataset_dir.resolve()),
                        output_dir=str(output_dir),
                        batch_size=1,
                        grad_accum_steps=4,
                        epochs=12,
                        checkpoint_interval=1,
                        seed=20260801,
                        accelerator="gpu",
                        devices=1,
                        num_workers=0,
                        use_ema=False,
                        early_stopping=False,
                        tensorboard=False,
                        wandb=False,
                        mlflow=False,
                        progress_bar=None,
                        multi_scale=False,
                        expanded_scales=False,
                        aug_config={},
                        scale_jitter=False,
                        augmentation_backend="torchvision",
                        save_dataset_grids=False,
                        run_test=False,
                    )
                    training_api.seed_everything(20260801, workers=True)
                    module = training_api.RFDETRModelModule(model.model_config, train_config)
                    datamodule = training_api.RFDETRDataModule(model.model_config, train_config)
                    trainer = training_api.build_trainer(
                        train_config,
                        model.model_config,
                        accelerator="gpu",
                        devices=1,
                        num_sanity_val_steps=0,
                        precision="16-mixed",
                    )
                    before = self._finite_validation_metrics(
                        trainer.validate(module, datamodule=datamodule)
                    )
                    trainer.fit(module, datamodule=datamodule)
                    after = self._finite_validation_metrics(
                        trainer.validate(module, datamodule=datamodule)
                    )
                except DetectorTrainingError:
                    raise
                except Exception as exc:
                    self._raise_stage_error(
                        "training",
                        exc,
                        "RF-DETR bootstrap short training failed",
                    )
                upstream_checkpoint = output_dir / "checkpoint_best_total.pth"
                raw_metrics = output_dir / "metrics.csv"
                if not upstream_checkpoint.is_file():
                    raise DetectorTrainingError("RF-DETR did not produce checkpoint_best_total.pth")
                loss, metric, epoch = self._read_finite_loss(raw_metrics)
                shared = sorted(set(before) & set(after))
                if not shared:
                    raise DetectorTrainingError(
                        "RF-DETR before/after validation metrics do not overlap"
                    )
                comparison = {
                    "before_finetune": {key: before[key] for key in shared},
                    "after_short_train": {key: after[key] for key in shared},
                    "delta": {key: round(after[key] - before[key], 12) for key in shared},
                }
                shutil.copyfile(upstream_checkpoint, checkpoint_path)
                shutil.copyfile(raw_metrics, upstream_metrics_path)
                log_path.write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "upstream_metric": metric,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                metrics_path.write_text(
                    json.dumps(comparison, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
        except DetectorTrainingError:
            raise
        except Exception as exc:
            raise DetectorTrainingError("RF-DETR bootstrap short training failed") from exc
        self._resolution = resolution

    def reload(self, checkpoint_path: Path) -> None:
        api, _ = self._load_runtime()
        if not checkpoint_path.is_file():
            raise DetectorTrainingError("RF-DETR checkpoint is unavailable for reload")
        try:
            self._reloaded_model = api.from_checkpoint(str(checkpoint_path.resolve()))
        except Exception as exc:
            self._raise_stage_error(
                "checkpoint_reload",
                exc,
                "RF-DETR checkpoint reload failed",
            )
        self._reloaded_checkpoint = checkpoint_path.resolve()

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        if (
            self._reloaded_model is None
            or self._reloaded_checkpoint != checkpoint_path.resolve()
            or self._resolution is None
        ):
            raise DetectorTrainingError("RF-DETR checkpoint must be reloaded before ONNX export")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="rfdetr-export-", dir=output_path.parent
            ) as export_name:
                exported = self._reloaded_model.export(
                    output_dir=export_name,
                    shape=self._resolution,
                    batch_size=1,
                    dynamic_batch=False,
                    format="onnx",
                    verbose=False,
                    fp16=False,
                )
                exported_path = Path(exported)
                if not exported_path.is_file():
                    raise DetectorTrainingError("RF-DETR ONNX export path is missing")
                shutil.copyfile(exported_path, output_path)
        except DetectorTrainingError:
            raise
        except Exception as exc:
            self._raise_stage_error(
                "onnx_export",
                exc,
                "RF-DETR ONNX export failed",
            )
