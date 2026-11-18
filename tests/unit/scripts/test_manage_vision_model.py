from __future__ import annotations

import json

from scripts.algorithm.manage_vision_model import main
from tests.unit.lab.vision.test_registry import _setup


def test_cli_inspect_prints_verified_active_identity(tmp_path, capsys) -> None:
    registry, active, incumbent, _ = _setup(tmp_path)

    exit_code = main(["inspect", "--registry", str(registry), "--active", str(active)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["package_id"] == incumbent
