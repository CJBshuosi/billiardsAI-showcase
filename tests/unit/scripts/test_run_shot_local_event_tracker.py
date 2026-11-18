from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_shot_local_event_tracker.py"
    spec = importlib.util.spec_from_file_location("run_shot_local_event_tracker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_legacy_tracker_delegates_to_cached_mainline_publisher(tmp_path: Path, capsys) -> None:
    module = _module()
    calls = []

    class Artifact:
        path = tmp_path / "unified" / ("a" * 64)
        artifact_id = "a" * 64
        frame_count = 221
        shot_count = 1

    def runner(**kwargs):
        calls.append(kwargs)
        return Artifact()

    source = tmp_path / "source"
    config = tmp_path / "config.yaml"
    exit_code = module.main(
        [
            "--video",
            str(tmp_path / "video.mp4"),
            "--source-artifact",
            str(source),
            "--shot-local-config",
            str(config),
            "--output-root",
            str(tmp_path / "unified"),
        ],
        config_loader=lambda path: ("config", path),
        runner=runner,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        {
            "video": tmp_path / "video.mp4",
            "source_artifact": source,
            "config": ("config", config),
            "output_root": tmp_path / "unified",
        }
    ]
    assert "deprecated" in captured.err.lower()
    assert "shot-local-unified-product-v1" in captured.out


def test_legacy_tracker_has_no_second_algorithm_or_renderer() -> None:
    source = Path("scripts/algorithm/run_shot_local_event_tracker.py").read_text()

    assert "def _render" not in source
    assert "def _active_config" not in source
    assert "def _world_event_rows" not in source
    assert '"kind": "shot_local_event_tracker"' not in source
