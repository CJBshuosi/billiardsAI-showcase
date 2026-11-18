from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _module():
    scripts = Path(__file__).parents[3] / "scripts" / "algorithm"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "reobserve_mainline1_shot_local_artifact",
            scripts / "reobserve_mainline1_shot_local_artifact.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_cached_reobserver_cli_passes_existing_artifact_without_model_arguments(
    tmp_path: Path, capsys
) -> None:
    module = _module()
    seen = {}

    class Artifact:
        artifact_id = "a" * 64
        frame_count = 43
        shot_count = 2
        path = tmp_path / "result"

    def runner(**kwargs):
        seen.update(kwargs)
        return Artifact()

    exit_code = module.main(
        [
            "--video",
            str(tmp_path / "video43.mp4"),
            "--source-artifact",
            str(tmp_path / "source"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-root",
            str(tmp_path / "output"),
        ],
        config_loader=lambda path: ("config", path),
        runner=runner,
    )

    assert exit_code == 0
    assert seen["source_artifact"] == tmp_path / "source"
    assert json.loads(capsys.readouterr().out)["frame_count"] == 43
