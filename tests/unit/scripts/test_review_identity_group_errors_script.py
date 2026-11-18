from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "review_identity_group_errors.py"
_SPEC = spec_from_file_location("review_identity_group_errors_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("group error review script cannot be loaded")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_group_error_review_cli_forwards_frozen_inputs(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "candidate"
    verdict = tmp_path / "verdict"
    correction = tmp_path / "correction"
    training = tmp_path / "training"
    assets = tmp_path / "assets"
    output = tmp_path / "review"
    for path in (candidate, verdict, correction, training, assets):
        path.mkdir()
    captured: dict[str, object] = {}

    def runner(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    exit_code = _MODULE.main(
        [
            "--candidate-artifact",
            str(candidate),
            "--verdict-artifact",
            str(verdict),
            "--correction-verdict-artifact",
            str(correction),
            "--training-run",
            str(training),
            "--asset-root",
            str(assets),
            "--output-root",
            str(output),
            "--port",
            "8767",
        ],
        runner=runner,
    )

    assert exit_code == 0
    assert captured["candidate_artifact"] == candidate
    assert captured["verdict_artifact"] == verdict
    assert captured["correction_verdict_artifact"] == correction
    assert captured["training_run"] == training
    assert captured["asset_root"] == assets
    assert captured["output_root"] == output
    assert captured["port"] == 8767
    assert captured["open_browser"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "STOPPED"
