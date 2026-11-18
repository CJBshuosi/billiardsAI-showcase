from __future__ import annotations

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from billiards_ai.lab.raytheon_handoff import FrozenInputs
from billiards_ai.lab.training.rfdetr_bootstrap_train import BootstrapTrainingRun

_SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "train_rfdetr_bootstrap_short.py"
_SPEC = spec_from_file_location("train_rfdetr_bootstrap_short_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("RF-DETR bootstrap short training script cannot be loaded")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


class _CliBackend:
    name = "cli-bootstrap-backend"
    version = "1.9.0"
    created_with: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).created_with = kwargs

    def validate_runtime(self) -> None:
        return None

    def runtime_evidence(self) -> dict[str, object]:
        return {
            "accelerator": "cuda",
            "amp": True,
            "device_name": "NVIDIA GeForce GTX 1650 Ti",
        }


def _weight(tmp_path: Path) -> Path:
    path = tmp_path / "rf-detr-seg-s-ft.pth"
    path.write_bytes(b"approved-weight")
    return path


def _frozen_loader(_project_root: Path, weight: Path) -> FrozenInputs:
    return FrozenInputs(
        project_root=Path.cwd(),
        config_path=Path("configs/algorithm/training/rfdetr-single-batch-smoke.yaml"),
        config_sha256="a" * 64,
        weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        weight_asset_id="rfdetr-seg-small-1.9.0",
        seed=20260801,
        batch_size=1,
        model_class="RFDETRSegSmall",
        packages={},
    )


def test_cli_locks_cuda_amp_and_the_repository_short_config(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "dataset-id"
    dataset.mkdir()
    runs_root = tmp_path / "runs"
    weight = _weight(tmp_path)

    def runner(**kwargs: object) -> BootstrapTrainingRun:
        assert kwargs["dataset_dir"] == dataset
        assert kwargs["runs_root"] == runs_root
        assert Path(kwargs["config_path"]).name == "rfdetr-seg-small-bootstrap-short.yaml"
        assert isinstance(kwargs["backend"], _CliBackend)
        output = tmp_path / "result"
        output.mkdir()
        return BootstrapTrainingRun(
            output=output,
            dataset_id="b" * 64,
            config_hash="c" * 64,
            run_hash="d" * 64,
        )

    exit_code = main(
        ["--dataset", str(dataset), "--runs-root", str(runs_root), "--weight", str(weight)],
        backend_factory=_CliBackend,
        frozen_inputs_loader=_frozen_loader,
        runner=runner,
    )

    assert exit_code == 0
    assert _CliBackend.created_with == {
        "weight_path": weight,
        "expected_weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "expected_version": "1.9.0",
        "model_class": "RFDETRSegSmall",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "accelerator": "cuda",
        "amp": True,
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "config_hash": "c" * 64,
        "dataset_id": "b" * 64,
        "exit_code": 0,
        "output": str(tmp_path / "result"),
        "run_hash": "d" * 64,
        "runtime": {
            "accelerator": "cuda",
            "amp": True,
            "device_name": "NVIDIA GeForce GTX 1650 Ti",
        },
        "schema_version": "1.0",
        "scientific_use": "development_short_train_not_locked_accuracy",
        "status": "PASS",
        "synthetic_only": False,
    }


def test_cli_does_not_offer_training_shape_overrides(tmp_path: Path) -> None:
    weight = _weight(tmp_path)

    try:
        main(
            [
                "--dataset",
                str(tmp_path / "dataset"),
                "--runs-root",
                str(tmp_path / "runs"),
                "--weight",
                str(weight),
                "--epochs",
                "50",
            ],
            backend_factory=_CliBackend,
            frozen_inputs_loader=_frozen_loader,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("the fixed short-train CLI accepted an epochs override")
