"""A managed config digest must label exactly the bytes which were parsed."""

import hashlib
from pathlib import Path

import pytest

from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config


def test_verified_config_parses_the_same_read_it_hashes(tmp_path):
    path = tmp_path / "config.yaml"
    original = Path("configs/algorithm/shot-local.yaml").read_bytes()
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    config = load_mainline1_shot_local_config(path, expected_sha256=digest)
    assert config.window.motion_confirm_frames == 2
    path.write_bytes(original + b"\n# new bytes")
    with pytest.raises(ValueError):
        load_mainline1_shot_local_config(path, expected_sha256=digest)
