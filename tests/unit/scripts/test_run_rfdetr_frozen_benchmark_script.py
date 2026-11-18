from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from billiards_ai.lab.training.rfdetr_frozen_benchmark import FrozenBenchmarkRun

_SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_rfdetr_frozen_benchmark.py"
_SPEC = spec_from_file_location("run_rfdetr_frozen_benchmark_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("RF-DETR frozen benchmark script cannot be loaded")
script = module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_cli_forwards_every_frozen_input_and_prints_one_pass_result(
    tmp_path: Path, capsys: object
) -> None:
    captured: dict[str, object] = {}

    def runner(inputs: object) -> FrozenBenchmarkRun:
        captured["inputs"] = inputs
        output = tmp_path / "runs" / ("9" * 64)
        output.mkdir(parents=True)
        return FrozenBenchmarkRun(path=output, artifact_id="9" * 64)

    exit_code = script.main(
        [
            "--input-bundle",
            str(tmp_path / "input"),
            "--evaluation-manifest",
            str(tmp_path / "evaluation.json"),
            "--bootstrap-dataset",
            str(tmp_path / "dataset"),
            "--training-run",
            str(tmp_path / "training"),
            "--training-manifest-sha256",
            "b" * 64,
            "--trained-checkpoint-sha256",
            "c" * 64,
            "--base-weight",
            str(tmp_path / "base.pth"),
            "--base-weight-sha256",
            "a" * 64,
            "--output-root",
            str(tmp_path / "artifacts"),
        ],
        runner=runner,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload == {
        "artifact_id": "9" * 64,
        "exit_code": 0,
        "output": str(tmp_path / "runs" / ("9" * 64)),
        "status": "PASS",
    }
    inputs = captured["inputs"]
    assert inputs.input_bundle == tmp_path / "input"  # type: ignore[attr-defined]
    assert inputs.evaluation_manifest == tmp_path / "evaluation.json"  # type: ignore[attr-defined]
    assert inputs.bootstrap_dataset == tmp_path / "dataset"  # type: ignore[attr-defined]
    assert inputs.training_run == tmp_path / "training"  # type: ignore[attr-defined]
    assert inputs.base_weight == tmp_path / "base.pth"  # type: ignore[attr-defined]
    assert inputs.expected_base_weight_sha256 == "a" * 64  # type: ignore[attr-defined]
    assert inputs.expected_training_manifest_sha256 == "b" * 64  # type: ignore[attr-defined]
    assert inputs.expected_trained_checkpoint_sha256 == "c" * 64  # type: ignore[attr-defined]


def test_cli_fails_closed_without_a_partial_success_payload(tmp_path: Path, capsys: object) -> None:
    def runner(inputs: object) -> FrozenBenchmarkRun:
        del inputs
        raise RuntimeError("CUDA inference failed")

    exit_code = script.main(
        [
            "--input-bundle",
            str(tmp_path / "input"),
            "--evaluation-manifest",
            str(tmp_path / "evaluation.json"),
            "--bootstrap-dataset",
            str(tmp_path / "dataset"),
            "--training-run",
            str(tmp_path / "training"),
            "--training-manifest-sha256",
            "b" * 64,
            "--trained-checkpoint-sha256",
            "c" * 64,
            "--base-weight",
            str(tmp_path / "base.pth"),
            "--base-weight-sha256",
            "a" * 64,
            "--output-root",
            str(tmp_path / "artifacts"),
        ],
        runner=runner,
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert captured.out == ""
    assert "frozen benchmark failed: CUDA inference failed" in captured.err
