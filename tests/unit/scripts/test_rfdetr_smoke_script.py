from __future__ import annotations

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from billiards_ai.lab.raytheon_handoff import (
    FrozenInputs,
    HandoffContractError,
)
from billiards_ai.lab.training.rfdetr_backend import DetectorOutOfMemoryError
from billiards_ai.lab.training.rfdetr_train import DetectorTrainingError

_SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "rfdetr_smoke.py"
_SPEC = spec_from_file_location("rfdetr_smoke_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("RF-DETR smoke script cannot be loaded")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


class _CliBackend:
    name = "cli-contract-backend"
    version = "1.9.0"

    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs

    def validate_runtime(self) -> None:
        return None

    def train(
        self,
        *,
        dataset_dir: Path,
        config: dict,
        checkpoint_path: Path,
        log_path: Path,
    ) -> None:
        assert (dataset_dir / "train" / "_annotations.coco.json").is_file()
        assert config["seed"] == 20260801
        assert config["max_train_batches"] == 1
        checkpoint_path.write_bytes(b"checkpoint")
        log_path.write_text('{"epoch":1,"loss":0.5}\n', encoding="utf-8")

    def reload(self, checkpoint_path: Path) -> None:
        assert checkpoint_path.is_file()

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"onnx")


def _weight(tmp_path: Path) -> Path:
    path = tmp_path / "weight.pth"
    path.write_bytes(b"weight")
    return path


def _frozen_loader(_project_root: Path, weight: Path) -> FrozenInputs:
    config = Path("configs/algorithm/training/rfdetr-single-batch-smoke.yaml").resolve()
    return FrozenInputs(
        project_root=Path.cwd(),
        config_path=config,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        weight_asset_id="rfdetr-seg-small-1.9.0",
        seed=20260801,
        batch_size=1,
        model_class="RFDETRSegSmall",
        packages={},
    )


def _arguments(tmp_path: Path, weight: Path, *, cuda: bool = False) -> list[str]:
    result = [
        "--workspace",
        str(tmp_path / "workspace"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--weight",
        str(weight),
    ]
    if cuda:
        result.extend(("--accelerator", "cuda", "--amp"))
    return result


def test_cli_uses_frozen_inputs_and_existing_smoke_orchestrator(tmp_path: Path, capsys) -> None:
    weight = _weight(tmp_path)

    exit_code = main(
        _arguments(tmp_path, weight),
        backend_factory=_CliBackend,
        frozen_inputs_loader=_frozen_loader,
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS"
    assert summary["exit_code"] == 0
    assert summary["seed"] == 20260801
    assert summary["synthetic_only"] is True
    assert summary["real_backend"] is True
    assert "runtime" not in summary
    assert Path(summary["output"]).is_dir()


def test_cli_keeps_backend_logs_out_of_the_json_summary(tmp_path: Path, capsys) -> None:
    class ChattyBackend(_CliBackend):
        def validate_runtime(self) -> None:
            print("upstream runtime log")

        def train(self, **kwargs: object) -> None:
            print("upstream training log")
            super().train(**kwargs)

    weight = _weight(tmp_path)

    exit_code = main(
        _arguments(tmp_path, weight),
        backend_factory=ChattyBackend,
        frozen_inputs_loader=_frozen_loader,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "PASS"
    assert "upstream runtime log" in captured.err
    assert "upstream training log" in captured.err


def test_cli_rejects_removed_hash_and_config_override_arguments(tmp_path: Path) -> None:
    weight = _weight(tmp_path)

    with pytest.raises(SystemExit) as captured:
        main(
            _arguments(tmp_path, weight) + ["--weight-sha256", "0" * 64, "--config", "other.yaml"],
            backend_factory=_CliBackend,
            frozen_inputs_loader=_frozen_loader,
        )

    assert captured.value.code == 2


def test_cli_rejects_frozen_input_mismatch_before_backend_or_fixture(
    tmp_path: Path, capsys
) -> None:
    weight = _weight(tmp_path)
    backend_called = False

    def backend_factory(**_kwargs: object) -> _CliBackend:
        nonlocal backend_called
        backend_called = True
        return _CliBackend()

    def mismatch(_project_root: Path, _weight: Path) -> FrozenInputs:
        raise HandoffContractError("smoke config SHA-256 mismatch")

    exit_code = main(
        _arguments(tmp_path, weight),
        backend_factory=backend_factory,
        frozen_inputs_loader=mismatch,
    )

    assert exit_code == 2
    assert backend_called is False
    assert not (tmp_path / "workspace").exists()
    assert capsys.readouterr().out == ""


def test_cli_reports_runtime_failure_without_claiming_pass(tmp_path: Path, capsys) -> None:
    class MissingRuntimeBackend(_CliBackend):
        def validate_runtime(self) -> None:
            raise DetectorTrainingError("RF-DETR training runtime is unavailable")

    weight = _weight(tmp_path)
    exit_code = main(
        _arguments(tmp_path, weight),
        backend_factory=MissingRuntimeBackend,
        frozen_inputs_loader=_frozen_loader,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "runtime is unavailable" in captured.err


def test_cli_preserves_explicit_typed_oom_result_and_exit_21(tmp_path: Path, capsys) -> None:
    class OomBackend(_CliBackend):
        def train(self, **_kwargs: object) -> None:
            raise DetectorOutOfMemoryError("checkpoint_reload")

    weight = _weight(tmp_path)
    exit_code = main(
        _arguments(tmp_path, weight, cuda=True),
        backend_factory=OomBackend,
        frozen_inputs_loader=_frozen_loader,
    )

    assert exit_code == 21
    assert json.loads(capsys.readouterr().out) == {
        "detail": "RF-DETR CUDA out of memory during checkpoint_reload",
        "exit_code": 21,
        "schema_version": "1.0",
        "seed": 20260801,
        "stage": "checkpoint_reload",
        "status": "OOM",
        "synthetic_only": True,
    }


def test_cli_passes_cuda_amp_and_emits_runtime_evidence(tmp_path: Path, capsys) -> None:
    seen_arguments: dict[str, object] = {}

    class CudaBackend(_CliBackend):
        def __init__(self, **kwargs: object) -> None:
            seen_arguments.update(kwargs)
            super().__init__(**kwargs)

        def runtime_evidence(self) -> dict[str, object]:
            return {"accelerator": "cuda", "amp": True, "cuda_available": True}

    weight = _weight(tmp_path)
    assert (
        main(
            _arguments(tmp_path, weight, cuda=True),
            backend_factory=CudaBackend,
            frozen_inputs_loader=_frozen_loader,
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["runtime"]["accelerator"] == "cuda"
    assert summary["runtime"]["amp"] is True
    assert (
        seen_arguments["expected_weight_sha256"] == hashlib.sha256(weight.read_bytes()).hexdigest()
    )
    assert seen_arguments["accelerator"] == "cuda"
    assert seen_arguments["amp"] is True
