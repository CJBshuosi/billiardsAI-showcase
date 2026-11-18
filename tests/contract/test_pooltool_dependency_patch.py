"""Auditable dependency-only patch contract for pooltool 0.6.0."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPECTED_WHEEL_SHA256 = "c47a46aeeeac6eea645e5c1610211ce0f0ae350e4b77909b8dd55b66e77b3bd5"


def test_pooltool_version_and_dependency_metadata_patch_are_exact():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.12,<3.14"
    assert "pooltool-billiards==0.6.0" in pyproject["project"]["dependencies"]
    assert pyproject["tool"]["uv"]["override-dependencies"] == ["panda3d>=1.10.13,<1.11"]


def test_lock_keeps_the_verified_upstream_wheel_hash():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert 'name = "pooltool-billiards"' in lock
    assert 'version = "0.6.0"' in lock
    assert f"sha256:{EXPECTED_WHEEL_SHA256}" in lock


def test_patch_provenance_records_source_license_hash_and_scope():
    record = (ROOT / "third_party" / "pooltool-billiards-0.6.0.md").read_text(encoding="utf-8")

    for evidence in (
        "pooltool-billiards==0.6.0",
        "Apache-2.0",
        EXPECTED_WHEEL_SHA256,
        "panda3d==1.11.0.dev3702",
        "panda3d>=1.10.13,<1.11",
        "dependency metadata only",
        "No pooltool physics algorithm or source file is modified",
        "https://files.pythonhosted.org/",
    ):
        assert evidence in record
