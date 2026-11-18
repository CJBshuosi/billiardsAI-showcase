from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.shared.calibration.homography import calibrate


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_field_route_replay.py"
    spec = importlib.util.spec_from_file_location("run_field_route_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_uses_verified_mainline1_artifact(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _module()
    seen: dict[str, object] = {}
    calibration = calibrate(
        "synthetic-camera",
        ((0, 180), (320, 180), (320, 0), (0, 0)),
    )
    inputs = SimpleNamespace(
        artifact_id="a" * 64,
        video_path=tmp_path / "input.avi",
        observations_path=tmp_path / "artifact" / "detections.json",
        shot_windows_path=tmp_path / "artifact" / "shot-windows.json",
        events_path=tmp_path / "artifact" / "world-events.json",
        calibration=calibration,
    )

    def run_replay(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            artifact_id="b" * 64,
            frame_count=8,
            guidance_generation_count=2,
            path=tmp_path / "runs" / ("b" * 64),
            shot_count=1,
        )

    monkeypatch.setattr(module, "load_mainline1_replay_inputs", lambda **_: inputs)
    monkeypatch.setattr(module, "run_field_route_replay", run_replay)

    exit_code = module.main(
        [
            "--video",
            str(inputs.video_path),
            "--mainline1-artifact",
            str(tmp_path / "artifact"),
            "--context-group",
            "solid",
            "--output-root",
            str(tmp_path / "runs"),
        ]
    )

    assert exit_code == 0
    assert seen["video"] == inputs.video_path
    assert seen["observations_path"] == inputs.observations_path
    assert seen["shot_windows_path"] == inputs.shot_windows_path
    assert seen["events_path"] == inputs.events_path
    assert seen["calibration"] == calibration
    assert seen["source_mainline1_artifact_id"] == "a" * 64
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


@pytest.mark.parametrize("forbidden", ["--calibration", "--shot-windows"])
def test_cli_rejects_explicit_replay_inputs_in_artifact_mode(
    tmp_path: Path,
    forbidden: str,
) -> None:
    module = _module()

    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "--video",
                str(tmp_path / "input.avi"),
                "--mainline1-artifact",
                str(tmp_path / "artifact"),
                forbidden,
                str(tmp_path / "explicit.json"),
                "--context-group",
                "solid",
                "--output-root",
                str(tmp_path / "runs"),
            ]
        )

    assert raised.value.code == 64


def test_cli_requires_one_replay_input_mode(tmp_path: Path) -> None:
    module = _module()

    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "--video",
                str(tmp_path / "input.avi"),
                "--context-group",
                "solid",
                "--output-root",
                str(tmp_path / "runs"),
            ]
        )

    assert raised.value.code == 64
