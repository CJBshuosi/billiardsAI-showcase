import copy
import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError

import pytest

MODULE = "billiards_ai.edge.route_assistance.runtime_resources"
PROFILE = "cbsa-chinese-pool-2017-observable-v1"
SHA = "a" * 64


def api():
    return importlib.import_module(MODULE)


def payload():
    root = "/original/台球"
    paths = {
        name: f"{root}/{name}"
        for name in (
            "artifact_root",
            "jobs_root",
            "ring_root",
            "capture_setup",
            "calibration_root",
            "config",
            "detector_training_run",
            "identity_training_run",
            "dinov2_assets",
        )
    }
    paths["detector_training_run"] += "/" + SHA
    paths["identity_training_run"] += "/" + SHA
    calibration = paths["calibration_root"] + "/" + SHA
    locations = {
        "capture_setup": paths["capture_setup"],
        "shot_config": paths["config"],
        "calibration_manifest": calibration + "/manifest.json",
        "calibration_inventory:calibration.json": calibration + "/calibration.json",
        "detector_manifest": paths["detector_training_run"] + "/training-manifest.json",
        "detector_checkpoint": paths["detector_training_run"] + "/checkpoint_best.pth",
        "identity_manifest": paths["identity_training_run"] + "/manifest.json",
        "identity_metrics": paths["identity_training_run"] + "/metrics.json",
        "identity_onnx": paths["identity_training_run"] + "/group-head.onnx",
        **{
            "dinov2:" + n: paths["dinov2_assets"] + "/" + n
            for n in ("model.safetensors", "config.json", "preprocessor_config.json")
        },
    }
    return dict(
        schema_version="1.0",
        contract="managed_runtime_resources_v1",
        artifact_id="usb-field-" + hashlib.sha256(paths["artifact_root"].encode()).hexdigest(),
        rule_profile_id=PROFILE,
        paths=paths,
        files=[dict(role=k, path=v, byte_size=0, sha256=SHA) for k, v in locations.items()],
        capture_setup_id="setup-a",
        setup_fingerprint=SHA,
        calibration_artifact_id=SHA,
        calibration_artifact_path=calibration,
        calibration_revision=1,
        calibration_manifest_sha256=SHA,
        model_package_id=SHA,
        model_manifest_sha256=SHA,
        identity_training_package_id=SHA,
        identity_training_package_sha256=SHA,
        dinov2_version="facebook/dinov2-small@pinned",
        dinov2_inventory_sha256=api().runtime_inventory_sha256(
            {
                n: {"bytes": 0, "sha256": SHA}
                for n in ("model.safetensors", "config.json", "preprocessor_config.json")
            }
        ),
        device="cpu",
        batch_size=32,
        fps=60,
    )


def test_contract_exists():
    assert api().RuntimeResourceFile


def test_canonical_deeply_immutable_and_detached():
    raw = payload()
    spec = api().ManagedRuntimeResourceSpec.from_payload(raw)
    original = spec.canonical_json
    raw["paths"]["config"] = "/other"
    raw["files"][0]["sha256"] = "b" * 64
    detached = spec.as_payload()
    detached["paths"]["config"] = "/elsewhere"
    assert spec.canonical_json == original
    assert spec.config_sha256 == SHA
    assert spec.spec_sha256 == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert "台球" in original
    assert json.loads(original)["files"] == sorted(
        spec.as_payload()["files"], key=lambda v: v["role"]
    )
    assert api().ManagedRuntimeResourceSpec.from_payload(json.loads(original)) == spec
    with pytest.raises(TypeError):
        spec.paths["config"] = "/bad"
    with pytest.raises(FrozenInstanceError):
        spec.device = "cuda"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(extra=True),
        lambda p: p.pop("fps"),
        lambda p: p.update(fps=True),
        lambda p: p.update(batch_size=True),
        lambda p: p.update(batch_size=257),
        lambda p: p.update(calibration_revision=True),
        lambda p: p.update(model_package_id="wrong"),
        lambda p: p.update(artifact_id="other"),
        lambda p: p["paths"].update(extra="/extra"),
        lambda p: p["files"].append(copy.deepcopy(p["files"][0])),
        lambda p: p["files"][0].update(sha256="A" * 64),
        lambda p: p["files"][0].update(byte_size=True),
        lambda p: p["files"][0].update(byte_size=-1),
        lambda p: p["files"][0].update(path="/wrong"),
        lambda p: p.update(calibration_manifest_sha256="b" * 64),
        lambda p: p.update(dinov2_inventory_sha256="b" * 64),
    ],
)
def test_strict_and_contradictory_metadata_rejected(change):
    raw = payload()
    change(raw)
    with pytest.raises(ValueError):
        api().ManagedRuntimeResourceSpec.from_payload(raw)


@pytest.mark.parametrize(
    "path", ["relative", "/a/../b", "/a/./b", "/a//b", "/bad\ud800", "C:relative", "C:\\a\\..\\b"]
)
def test_invalid_paths_rejected_without_io(path):
    with pytest.raises(ValueError):
        api().RuntimeResourceFile("empty", path, 0, SHA)


@pytest.mark.parametrize(
    "path", ["/original/file", "C:\\original\\file", "\\\\server\\share\\file"]
)
def test_native_path_text_and_empty_inventory_are_preserved(path):
    record = api().RuntimeResourceFile.from_payload(
        dict(role="empty", path=path, byte_size=0, sha256=SHA)
    )
    assert record.path == path
    assert record.as_payload()["byte_size"] == 0


def test_utf8_and_one_mib_limit():
    raw = payload()
    raw["capture_setup_id"] = "x" * 1048576
    with pytest.raises(ValueError):
        api().ManagedRuntimeResourceSpec.from_payload(raw)
    raw = payload()
    raw["capture_setup_id"] = "\ud800"
    with pytest.raises(ValueError):
        api().ManagedRuntimeResourceSpec.from_payload(raw)


def test_strict_records_and_epoch_binding():
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope

    spec = api().ManagedRuntimeResourceSpec.from_payload(payload())
    scope = ManagedSessionScope("dev", "device", "table")
    uid = "12345678-1234-4234-8234-123456789abc"
    record = api().ManagedResourceSpecRecord(
        scope, uid, spec, spec.spec_sha256, "2026-11-01T00:00:00Z"
    )
    assert api().ManagedResourceSpecRecord.from_payload(record.as_payload()) == record
    epoch = api().ManagedCaptureEpochRecord(
        scope, uid, "managed-" + uid, uid, "epoch-a", spec.spec_sha256, "2026-11-01T00:00:00Z"
    )
    assert api().ManagedCaptureEpochRecord.from_payload(epoch.as_payload()) == epoch
    for cls, raw in [(type(record), record.as_payload()), (type(epoch), epoch.as_payload())]:
        raw["extra"] = "forbidden"
        with pytest.raises(ValueError):
            cls.from_payload(raw)
    raw = record.as_payload()
    raw["spec_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        type(record).from_payload(raw)
    raw = epoch.as_payload()
    raw["session_id"] = "managed-wrong"
    with pytest.raises(ValueError):
        type(epoch).from_payload(raw)


def test_contradictory_file_identity_rejected():
    raw = payload()
    raw["paths"]["config"] = raw["paths"]["capture_setup"]
    for f in raw["files"]:
        if f["role"] == "shot_config":
            f.update(path=raw["paths"]["config"], sha256="b" * 64)
    with pytest.raises(ValueError):
        api().ManagedRuntimeResourceSpec.from_payload(raw)


def test_inventory_hash_has_no_newline_and_no_paths():
    inventory = {"config.json": {"bytes": 0, "sha256": SHA}}
    expected = hashlib.sha256(
        ("{" + '"config.json":{"bytes":0,"sha256":"' + SHA + '"}}').encode()
    ).hexdigest()
    assert api().runtime_inventory_sha256(inventory) == expected


def test_entire_windows_spec_decodes_without_native_io(monkeypatch):
    from pathlib import Path

    raw = payload()

    def windows(p):
        return "C:" + p.replace("/", "\\")

    raw["paths"] = {k: windows(v) for k, v in raw["paths"].items()}
    raw["calibration_artifact_path"] = windows(raw["calibration_artifact_path"])
    raw["files"] = [{**f, "path": windows(f["path"])} for f in raw["files"]]
    raw["artifact_id"] = (
        "usb-field-" + hashlib.sha256(raw["paths"]["artifact_root"].encode()).hexdigest()
    )

    def forbidden(*args, **kwargs):
        pytest.fail("pure decoding touched the filesystem")

    monkeypatch.setattr(Path, "stat", forbidden)
    spec = api().ManagedRuntimeResourceSpec.from_payload(raw)
    assert spec.paths == raw["paths"]


def test_one_mib_boundary_is_utf8_bytes_not_characters():
    raw = payload()
    raw["capture_setup_id"] = "x"
    base = len(api().ManagedRuntimeResourceSpec.from_payload(raw).canonical_json.encode())
    raw["capture_setup_id"] = "x" * (1048576 - base + 1)
    spec = api().ManagedRuntimeResourceSpec.from_payload(raw)
    assert len(spec.canonical_json.encode()) == 1048576
    raw["capture_setup_id"] += "台"
    with pytest.raises(ValueError):
        api().ManagedRuntimeResourceSpec.from_payload(raw)


def test_resource_module_imports_without_lab_or_runtime_dependencies():
    import subprocess
    import sys

    source = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.startswith(('billiards_ai.lab', 'torch', 'cv2', 'boto', 'onnxruntime')):
        raise AssertionError('forbidden pure-contract import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from billiards_ai.edge.route_assistance.runtime_resources import RuntimeResourceFile
RuntimeResourceFile('empty', '/safe/file', 0, 'a' * 64)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
