import hashlib
import importlib
import importlib.util
import json
import zipfile

import pytest


def builder():
    name = "scripts.algorithm.package_recording_capture_diagnostic"
    assert importlib.util.find_spec(name) is not None, "Missing diagnostic packet builder"
    return importlib.import_module(name)


def test_changed_base_zip_is_refused_without_creating_stage(tmp_path):
    package = builder()
    base = tmp_path / "base.zip"
    base.write_bytes(b"changed")
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="ZIP changed"):
        package.build_stage(stage, base, "a" * 40)
    assert not stage.exists()


def test_new_packet_keeps_base_product_bytes_and_substitutes_only_new_diagnostic_roots(
    tmp_path, monkeypatch
):
    package = builder()
    values = {
        "entry.py": b"old entry must not run",
        "prepare.ps1": b"old prepare",
        "source/example.py": b"unchanged product",
        "source/SOURCE-LOCK.json": b"original source lock",
        "input/input.avi": b"fixed synthetic data",
    }
    pins = {n: dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n, b in values.items()}
    manifest = dict(
        members=pins,
        source_commit="b" * 40,
        input_sha256="c" * 64,
        source_lock_sha256=pins["source/SOURCE-LOCK.json"]["sha256"],
        dependency_site_packages="unchanged dependencies",
    )
    raw = json.dumps(manifest).encode()
    base = tmp_path / "base.zip"
    with zipfile.ZipFile(base, "w") as z:
        for name, data in values.items():
            z.writestr(name, data)
        z.writestr("packet-manifest.json", raw)
    monkeypatch.setattr(package, "BASE_ZIP_SHA", hashlib.sha256(base.read_bytes()).hexdigest())
    monkeypatch.setattr(package, "BASE_MANIFEST_SHA", hashlib.sha256(raw).hexdigest())
    source = {
        package.PROBE: b"new diagnostic",
        package.PREPARE: b"new prepare",
        package.BUILDER: b"new builder",
    }
    monkeypatch.setattr(package, "_source", lambda commit: (source, {}))
    stage = tmp_path / "stage"
    digest = package.build_stage(stage, base, "a" * 40)
    packet = stage / "packet"
    assert (packet / "source/example.py").read_bytes() == b"unchanged product"
    assert (packet / "input/input.avi").read_bytes() == b"fixed synthetic data"
    assert (packet / "entry.py").read_bytes() == b"new diagnostic"
    assert (packet / "prepare.ps1").read_bytes() == b"new prepare"
    value = json.loads((packet / "packet-manifest.json").read_bytes())
    assert value["purpose"] == "capture-throughput-diagnostic/v1"
    assert value["source_commit"] == "b" * 40 and value["diagnostic_source_commit"] == "a" * 40
    assert not value["native_execution_authorized"] and not value["product_pass"]
    archive = package.freeze(stage, tmp_path / "frozen", digest)
    with zipfile.ZipFile(archive) as z:
        assert z.read("entry.py") == b"new diagnostic"
        assert z.read("prepare.ps1") == b"new prepare"
        assert z.read("source/example.py") == b"unchanged product"
    assert archive.name == "recording-capture-diagnostic.zip"
