"""Windows physics preparation is download-only and fixed to reviewed wheels."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

MANIFEST = Path("scripts/development/assets/s1-physics-windows.yaml")


def tool():
    assert Path("scripts/development/s1_physics_environment.py").is_file()
    return importlib.import_module("scripts.development.s1_physics_environment")


def test_windows_wheel_manifest_matches_the_current_project_lock_exactly():
    import tomllib

    manifest = json.loads(MANIFEST.read_bytes())
    raw = Path("uv.lock").read_bytes()
    packages = {p["name"]: p for p in tomllib.loads(raw.decode())["package"]}
    expected = set()

    def visit(name):
        if name in expected:
            return
        expected.add(name)
        for dependency in packages[name].get("dependencies", []):
            visit(dependency["name"])

    visit("pooltool-billiards")
    assert manifest["source_uv_lock_sha256"] == hashlib.sha256(raw).hexdigest()
    assert {w["name"] for w in manifest["wheels"]} == expected
    assert manifest["payload_count"] == 24
    assert manifest["payload_bytes"] == 202686001
    for wheel in manifest["wheels"]:
        package = packages[wheel["name"]]
        assert wheel["version"] == package["version"]
        assert any(
            w["url"] == wheel["url"]
            and w["size"] == wheel["bytes"]
            and w["hash"] == "sha256:" + wheel["sha256"]
            for w in package["wheels"]
        )
    pool = next(w for w in manifest["wheels"] if w["name"] == "pooltool-billiards")
    assert pool["sha256"] == "c47a46aeeeac6eea645e5c1610211ce0f0ae350e4b77909b8dd55b66e77b3bd5"
    assert next(w["version"] for w in manifest["wheels"] if w["name"] == "panda3d") == "1.10.16"
    assert tool().load_manifest(MANIFEST) == manifest


@pytest.fixture
def small_manifest(tmp_path):
    data = b"fixed test download; never installed"
    wheel = dict(
        name="fixture",
        version="1",
        filename="fixture-1-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/test/fixture-1-py3-none-any.whl",
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    document = dict(
        contract="s1-physics-windows-wheels/v1",
        target="CPython312-win_amd64",
        source_uv_lock_sha256="a" * 64,
        metadata_override="test-only",
        payload_count=1,
        payload_bytes=len(data),
        wheels=[wheel],
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(json.dumps(document))
    return path, data, document


def test_download_does_not_install_and_only_publishes_verified_bytes(tmp_path, small_manifest):
    path, data, document = small_manifest
    module = tool()
    output = tmp_path / "wheelhouse"
    calls = []

    def fetch(url, destination, maximum):
        calls.append(url)
        assert maximum == len(data)
        destination.write_bytes(data)

    receipt = module.download(path, output, stream=fetch)
    assert calls == [document["wheels"][0]["url"]]
    assert receipt["installs"] == 0 and receipt["imports"] == 0
    assert (output / document["wheels"][0]["filename"]).read_bytes() == data
    assert (output / "wheels-verified.json").is_file()
    with pytest.raises(FileExistsError):
        module.download(path, output, stream=fetch)
    assert len(calls) == 1


def test_mismatched_download_cannot_publish_a_verified_receipt(tmp_path, small_manifest):
    path, _, _ = small_manifest
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="wheel"):
        tool().download(path, output, stream=lambda u, p, n: p.write_bytes(b"changed"))
    assert not (output / "wheels-verified.json").exists()
    assert not list(output.glob("*.whl"))


@pytest.mark.parametrize(
    "fault", ["url", "filename", "total", "size", "hash", "duplicate", "target"]
)
def test_invalid_wheel_manifest_rejects_before_network(tmp_path, small_manifest, fault):
    path, _, document = small_manifest
    wheel = document["wheels"][0]
    if fault == "url":
        wheel["url"] = "https://untrusted.example/test.whl"
    elif fault == "filename":
        wheel["filename"] = "../escape.whl"
    elif fault == "total":
        document["payload_bytes"] += 1
    elif fault == "size":
        wheel["bytes"] = 10**400
    elif fault == "hash":
        wheel["sha256"] = "x" * 64
    elif fault == "duplicate":
        document["wheels"].append(wheel.copy())
    elif fault == "target":
        document["target"] = "other"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        tool().load_manifest(path)


def test_manifest_changed_during_download_cannot_sign_the_original_wheels(tmp_path, small_manifest):
    path, data, document = small_manifest
    output = tmp_path / "changed-manifest"

    def mutate_manifest(url, destination, maximum):
        document["metadata_override"] = "changed after original bytes were selected"
        path.write_text(json.dumps(document))
        destination.write_bytes(data)

    with pytest.raises(ValueError, match="manifest changed"):
        tool().download(path, output, stream=mutate_manifest)
    assert not (output / "wheels-verified.json").exists()
