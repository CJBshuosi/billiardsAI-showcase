from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "train_identity_group_short.py"
_SPEC = spec_from_file_location("train_identity_group_short_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("group identity short training script cannot be loaded")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


class FakeFeatures:
    created_with: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).created_with = kwargs


class FakeBackend:
    created_with: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).created_with = kwargs


def test_cli_locks_group_only_cuda_short_training(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "candidate"
    verdict = tmp_path / "verdict"
    assets = tmp_path / "assets"
    output_root = tmp_path / "runs"
    correction = tmp_path / "correction"
    for path in (candidate, verdict, correction, assets):
        path.mkdir()

    def runner(**kwargs: object) -> object:
        assert kwargs["candidate_artifact"] == candidate
        assert kwargs["verdict_artifact"] == verdict
        assert kwargs["correction_verdict_artifact"] == correction
        assert kwargs["output_root"] == output_root
        assert Path(kwargs["config_path"]).name == "identity.yaml"
        return SimpleNamespace(
            output=tmp_path / "result",
            run_id="a" * 64,
            status="PASS",
            gate_status="FAIL",
        )

    exit_code = main(
        [
            "--candidate-artifact",
            str(candidate),
            "--verdict-artifact",
            str(verdict),
            "--correction-verdict-artifact",
            str(correction),
            "--asset-root",
            str(assets),
            "--output-root",
            str(output_root),
        ],
        feature_factory=FakeFeatures,
        backend_factory=FakeBackend,
        runner=runner,
    )

    assert exit_code == 0
    assert FakeFeatures.created_with == {
        "asset_root": assets,
        "device": "cuda",
        "batch_size": 8,
    }
    assert FakeBackend.created_with["candidate_root"] == candidate
    assert FakeBackend.created_with["hidden_size"] == 64
    assert FakeBackend.created_with["epochs"] == 500
    assert FakeBackend.created_with["learning_rate"] == 0.01
    assert FakeBackend.created_with["weight_decay"] == 0.0001
    assert FakeBackend.created_with["augmentation_count"] == 2
    assert FakeBackend.created_with["confidence_min"] == 0.55
    assert FakeBackend.created_with["margin_min"] == 0.10
    assert FakeBackend.created_with["known_probability_min"] == 0.55
    assert FakeBackend.created_with["calibration_min_unknown_support"] == 2
    assert FakeBackend.created_with["visual_evidence_weight"] == 0.5
    assert FakeBackend.created_with["hard_negative_focal_gamma"] == 2.0
    assert FakeBackend.created_with["object_hard_negative_weight"] == 1.5
    assert FakeBackend.created_with["seed"] == 20260801
    summary = json.loads(capsys.readouterr().out)
    assert summary["trained_heads"] == ["group"]
    assert summary["exact_head"] == {"status": "deferred", "trained": False}
    assert summary["gate_status"] == "FAIL"
    assert summary["scientific_use"] == "development_short_train_not_locked_accuracy"
