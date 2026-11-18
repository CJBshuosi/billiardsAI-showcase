from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "evaluate_shot_local_timeline.py"
    spec = importlib.util.spec_from_file_location("evaluate_shot_local_timeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_auditable_metrics(tmp_path: Path, capsys) -> None:
    timeline = tmp_path / "timeline.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "evaluation.json"
    payload = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [{"event_type": "shot_start", "status": "admitted", "timestamp_ms": 100}],
            }
        ]
    }
    timeline.write_text(json.dumps(payload))
    truth.write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "shot-0001",
                        "events": [{"event_type": "shot_start", "timestamp_ms": 110}],
                    }
                ]
            }
        )
    )

    assert (
        _module().main(
            ["--timeline", str(timeline), "--truth", str(truth), "--output", str(output)]
        )
        == 0
    )
    assert json.loads(output.read_text())["overall"]["tp"] == 1
    assert json.loads(capsys.readouterr().out)["tp"] == 1
