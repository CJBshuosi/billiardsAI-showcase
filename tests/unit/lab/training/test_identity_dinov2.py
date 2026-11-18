from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from billiards_ai.lab import mainline1_identity_backends as identity_dinov2
from billiards_ai.lab.training.identity_dinov2 import (
    Dinov2AssetError,
    verify_dinov2_assets,
)


def test_asset_verifier_requires_exact_three_file_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {}
    for name, payload in {
        "model.safetensors": b"weights",
        "config.json": b"{}",
        "preprocessor_config.json": b"{}\n",
    }.items():
        (tmp_path / name).write_bytes(payload)
        expected[name] = (len(payload), hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(identity_dinov2, "EXPECTED_ASSETS", expected)

    verified = verify_dinov2_assets(tmp_path)

    assert tuple(verified) == (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    )


def test_asset_verifier_rejects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {}
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        payload = name.encode()
        (tmp_path / name).write_bytes(payload)
        expected[name] = (len(payload), hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(identity_dinov2, "EXPECTED_ASSETS", expected)
    (tmp_path / "config.json").write_bytes(b"changed")

    with pytest.raises(Dinov2AssetError, match="identity changed"):
        verify_dinov2_assets(tmp_path)
