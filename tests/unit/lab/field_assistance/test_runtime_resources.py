import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab import mainline1_identity_backends as identity_dinov2
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import CaptureSetup, capture_setup_fingerprint


def test_pure_metadata_inspection_api_exists():
    resources = importlib.import_module("billiards_ai.lab.field_assistance.runtime_resources")
    assert callable(resources.inspect_managed_runtime_resources)
    assert callable(resources.verify_managed_runtime_resources)


def test_development_inspection_does_not_require_model_factories():
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    assert callable(runtime.inspect_development_vision_package)
    assert callable(runtime.inspect_development_group_assets)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def assets(tmp_path, monkeypatch):
    # Tiny synthetic assets exercise integrity mechanics, never approved model validity.
    root = tmp_path.resolve()
    paths = {
        n: root / n
        for n in ("artifact_root", "jobs_root", "ring_root", "calibration_root", "dinov2_assets")
    }
    for p in paths.values():
        p.mkdir()
    setup = CaptureSetup.model_validate(
        dict(
            capture_setup_id="setup-a",
            table_id="table-a",
            camera_device_id="camera-a",
            lens_id="lens",
            mount_revision="1",
            distortion_revision="1",
            source_kind="file",
            geometry=dict(width=420, height=220, crop=[0, 0, 420, 220]),
        )
    )
    paths["capture_setup"] = root / "capture-setup.json"
    paths["capture_setup"].write_text(setup.model_dump_json())
    paths["config"] = root / "shot.yaml"
    paths["config"].write_bytes(Path("configs/algorithm/shot-local.yaml").read_bytes())
    paths["detector_training_run"] = root / ("d" * 64)
    paths["identity_training_run"] = root / ("e" * 64)
    detector = paths["detector_training_run"]
    detector.mkdir()
    (detector / "checkpoint_best.pth").write_bytes(b"checkpoint")
    write_json(
        detector / "training-manifest.json",
        dict(
            run_hash=detector.name,
            model_class="RFDETRSegSmall",
            backend={"weight_sha256": "b" * 64},
            artifacts={"checkpoint_best.pth": sha(detector / "checkpoint_best.pth")},
        ),
    )
    group = paths["identity_training_run"]
    group.mkdir()
    (group / "group-head.onnx").write_bytes(b"onnx")
    write_json(
        group / "metrics.json",
        dict(
            unknown_policy=dict(confidence_min=0.5, margin_min=0.1, known_probability_min=0.3),
            visual_evidence_policy={"weight": 0.5},
        ),
    )
    write_json(
        group / "manifest.json",
        dict(
            run_id=group.name,
            status="PASS",
            backend_configuration={"architecture": "hierarchical_fused_evidence_mlp"},
            inventory={n: {"sha256": sha(group / n)} for n in ("group-head.onnx", "metrics.json")},
        ),
    )
    expected = {}
    for n in identity_dinov2.EXPECTED_ASSETS:
        p = paths["dinov2_assets"] / n
        p.write_bytes(n.encode())
        expected[n] = (p.stat().st_size, sha(p))
    monkeypatch.setattr(identity_dinov2, "EXPECTED_ASSETS", expected)

    def calibration(letter):
        aid = letter * 64
        directory = paths["calibration_root"] / aid
        directory.mkdir()
        (directory / "calibration.json").write_bytes(b"{}")
        (directory / "empty.bin").write_bytes(b"")
        write_json(
            directory / "manifest.json",
            dict(
                artifact_id=aid,
                capture_setup_id=setup.capture_setup_id,
                capture_setup_fingerprint=capture_setup_fingerprint(setup),
                source_video_sha256="c" * 64,
                inventory={
                    n: {"bytes": (directory / n).stat().st_size, "sha256": sha(directory / n)}
                    for n in ("calibration.json", "empty.bin")
                },
            ),
        )
        return register_calibration(
            paths["calibration_root"], SimpleNamespace(path=directory, artifact_id=aid), setup
        )

    first = calibration("a")
    return SimpleNamespace(
        **paths,
        device="cpu",
        batch_size=32,
        fps=60,
        token="must-not-persist",
        calibration=calibration,
        first=first,
    )


def inspect(assets):
    return importlib.import_module(
        "billiards_ai.lab.field_assistance.runtime_resources"
    ).inspect_managed_runtime_resources(
        assets, rule_profile_id="cbsa-chinese-pool-2017-observable-v1"
    )


def test_real_tiny_assets_are_inspected_without_heavy_io(assets, monkeypatch):
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    import socket

    import cv2

    def forbidden(*args, **kwargs):
        pytest.fail("metadata inspection performed heavy I/O")

    monkeypatch.setattr(runtime, "RfdetrVisionBackend", forbidden)
    monkeypatch.setattr(runtime, "OnnxGroupIdentityClassifier", forbidden)
    monkeypatch.setattr(identity_dinov2.Dinov2IdentityFeatureBackend, "__init__", forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    spec = inspect(assets)
    assert spec.config_sha256 == sha(assets.config)
    assert spec.model_manifest_sha256 == sha(
        assets.detector_training_run / "training-manifest.json"
    )
    assert spec.identity_training_package_sha256 == sha(
        assets.identity_training_run / "manifest.json"
    )
    assert "must-not-persist" not in spec.canonical_json
    assert any(f.byte_size == 0 for f in spec.files)
    resources = importlib.import_module("billiards_ai.lab.field_assistance.runtime_resources")
    resources.verify_managed_runtime_resources(spec, purpose="analysis")
    resources.verify_managed_runtime_resources(spec, purpose="capture")


def test_exact_superseded_calibration_is_retained_without_registry_write(assets):
    spec = inspect(assets)
    assets.calibration("b")
    registry = assets.calibration_root / "registry.json"
    before = registry.read_bytes()
    module = importlib.import_module("billiards_ai.lab.mainline1_calibration_registry")
    selected = module.resolve_calibration_record(
        assets.calibration_root,
        setup_fingerprint=spec.setup_fingerprint,
        artifact_id=spec.calibration_artifact_id,
        revision=1,
    )
    assert selected.status == "superseded"
    assert selected.artifact_id == spec.calibration_artifact_id
    assert (
        module.resolve_calibration_record(
            assets.calibration_root,
            setup_fingerprint=spec.setup_fingerprint,
            artifact_id="f" * 64,
            revision=1,
        )
        is None
    )
    importlib.import_module(
        "billiards_ai.lab.field_assistance.runtime_resources"
    ).verify_managed_runtime_resources(spec, purpose="analysis")
    assert registry.read_bytes() == before


@pytest.mark.parametrize(
    "which", ["config", "capture_setup", "detector", "identity", "dinov2", "calibration"]
)
def test_pinned_asset_drift_is_rejected(assets, which):
    spec = inspect(assets)
    target = dict(
        config=assets.config,
        capture_setup=assets.capture_setup,
        detector=assets.detector_training_run / "checkpoint_best.pth",
        identity=assets.identity_training_run / "metrics.json",
        dinov2=assets.dinov2_assets / "config.json",
        calibration=Path(spec.calibration_artifact_path) / "calibration.json",
    )[which]
    target.write_bytes(b"changed")
    with pytest.raises((ValueError, RuntimeError)):
        importlib.import_module(
            "billiards_ai.lab.field_assistance.runtime_resources"
        ).verify_managed_runtime_resources(spec, purpose="analysis")


def test_cleanup_needs_original_roots_only(assets):
    spec = inspect(assets)
    import shutil

    for root in (
        assets.detector_training_run,
        assets.identity_training_run,
        assets.dinov2_assets,
        assets.calibration_root,
    ):
        shutil.rmtree(root)
    assets.config.unlink()
    resources = importlib.import_module("billiards_ai.lab.field_assistance.runtime_resources")
    resources.verify_managed_runtime_resources(spec, purpose="cleanup")
    with pytest.raises(ValueError):
        resources.verify_managed_runtime_resources(spec, purpose="foreign")
    assets.jobs_root.rmdir()
    with pytest.raises(ValueError):
        resources.verify_managed_runtime_resources(spec, purpose="cleanup")


def test_symlink_payload_and_private_archive_rejected(assets, tmp_path):
    original = assets.detector_training_run / "checkpoint_best.pth"
    moved = tmp_path / "moved"
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(ValueError):
        inspect(assets)
    original.unlink()
    original.write_bytes(moved.read_bytes())
    assets.artifact_root = tmp_path / ".project-archive" / "never-inspect"
    with pytest.raises(ValueError):
        inspect(assets)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested\\escape", "nested/../escape"])
def test_historical_calibration_inventory_is_confined(assets, name):
    spec = inspect(assets)
    manifest_path = Path(spec.calibration_artifact_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["inventory"][name] = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    write_json(manifest_path, manifest)
    module = importlib.import_module("billiards_ai.lab.mainline1_calibration_registry")
    with pytest.raises(ValueError):
        module.resolve_calibration_record(
            assets.calibration_root,
            setup_fingerprint=spec.setup_fingerprint,
            artifact_id=spec.calibration_artifact_id,
            revision=1,
        )


def test_direct_inspectors_are_immutable_and_validate_policy(assets):
    from dataclasses import FrozenInstanceError

    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    package = runtime.inspect_development_vision_package(assets.detector_training_run)
    group = runtime.inspect_development_group_assets(assets.identity_training_run)
    assert package.checkpoint.name == "checkpoint_best.pth"
    assert group.manifest_sha256 == sha(group.manifest_path)
    assert group.confidence_min == 0.5
    with pytest.raises(FrozenInstanceError):
        group.confidence_min = 0.6
    metrics = json.loads(group.metrics_path.read_bytes())
    metrics["visual_evidence_policy"]["weight"] = -1
    write_json(group.metrics_path, metrics)
    manifest = json.loads(group.manifest_path.read_bytes())
    manifest["inventory"]["metrics.json"]["sha256"] = sha(group.metrics_path)
    write_json(group.manifest_path, manifest)
    with pytest.raises(ValueError, match="policy"):
        runtime.inspect_development_group_assets(assets.identity_training_run)


def test_hash_rejects_source_replaced_while_reading(assets, monkeypatch):
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    path = assets.config
    original_open = Path.open

    class ReplacingReader:
        def __init__(self, source):
            self.source = source
            self.replaced = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def fileno(self):
            return self.source.fileno()

        def read(self, size):
            value = self.source.read(size)
            if not self.replaced:
                replacement = path.with_name("replacement")
                replacement.write_bytes(b"replacement")
                replacement.replace(path)
                self.replaced = True
            return value

    def replacing_open(candidate, *args, **kwargs):
        source = original_open(candidate, *args, **kwargs)
        return ReplacingReader(source) if candidate == path and args == ("rb",) else source

    monkeypatch.setattr(Path, "open", replacing_open)
    with pytest.raises(ValueError, match="changed"):
        runtime._sha256(path)


def test_inspection_never_reads_credentials_or_imports_provider_runtimes(assets, monkeypatch):
    import builtins
    import os

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith(
            ("torch", "transformers", "onnxruntime", "boto3", "supabase", "google.auth")
        ):
            pytest.fail("inspection imported a model or credential/provider runtime")
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("inspection read environment/credentials")

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(os, "getenv", forbidden)
    assert inspect(assets).device == "cpu"


@pytest.mark.parametrize("file", ["training-manifest.json", "checkpoint_best.pth"])
def test_development_detector_rejects_symlinked_files(assets, tmp_path, file):
    path = assets.detector_training_run / file
    moved = tmp_path / "payload"
    path.rename(moved)
    path.symlink_to(moved)
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    with pytest.raises(ValueError):
        runtime.inspect_development_vision_package(assets.detector_training_run)


@pytest.mark.parametrize("file", ["manifest.json", "metrics.json", "group-head.onnx"])
def test_development_group_rejects_symlinked_files(assets, tmp_path, file):
    path = assets.identity_training_run / file
    moved = tmp_path / "payload"
    path.rename(moved)
    path.symlink_to(moved)
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    with pytest.raises(ValueError):
        runtime.inspect_development_group_assets(assets.identity_training_run)


def test_detector_checkpoint_drift_between_integrity_reads_is_rejected(assets, monkeypatch):
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    checkpoint = assets.detector_training_run / "checkpoint_best.pth"
    original = runtime._sha256
    changed = False

    def replacing_hash(path):
        nonlocal changed
        digest = original(path)
        if path == checkpoint and not changed:
            checkpoint.write_bytes(b"changed-after-first-check")
            changed = True
        return digest

    monkeypatch.setattr(runtime, "_sha256", replacing_hash)
    with pytest.raises(ValueError, match="changed"):
        runtime.inspect_development_vision_package(assets.detector_training_run)


def test_verification_rejects_archive_metadata_before_any_filesystem_access(assets, monkeypatch):
    from billiards_ai.edge.route_assistance.runtime_resources import ManagedRuntimeResourceSpec
    from billiards_ai.lab.field_assistance.runtime_resources import verify_managed_runtime_resources

    spec = inspect(assets)
    raw = spec.as_payload()
    previous = raw["paths"]["calibration_root"]
    replacement = previous + "/.project-archive"
    raw["paths"]["calibration_root"] = replacement
    raw["calibration_artifact_path"] = raw["calibration_artifact_path"].replace(
        previous, replacement
    )
    for file in raw["files"]:
        if file["role"].startswith("calibration_"):
            file["path"] = file["path"].replace(previous, replacement)
    metadata = ManagedRuntimeResourceSpec.from_payload(raw)
    original_stat = Path.stat
    original_lstat = Path.lstat

    def guarded_stat(path, *args, **kwargs):
        if ".project-archive" in path.parts:
            raise AssertionError("archive metadata reached filesystem stat")
        return original_stat(path, *args, **kwargs)

    def guarded_lstat(path, *args, **kwargs):
        if ".project-archive" in path.parts:
            raise AssertionError("archive metadata reached filesystem lstat")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "lstat", guarded_lstat)
    with pytest.raises(ValueError, match="archive"):
        verify_managed_runtime_resources(metadata, purpose="analysis")


def test_exact_calibration_does_not_substitute_or_accept_manifest_drift(assets):
    from billiards_ai.lab.mainline1_calibration_registry import resolve_calibration_record

    spec = inspect(assets)
    assets.calibration("b")
    arguments = dict(
        setup_fingerprint=spec.setup_fingerprint,
        artifact_id=spec.calibration_artifact_id,
        revision=1,
    )
    assert (
        resolve_calibration_record(assets.calibration_root, **{**arguments, "revision": 2}) is None
    )
    assert (
        resolve_calibration_record(
            assets.calibration_root, **{**arguments, "setup_fingerprint": "f" * 64}
        )
        is None
    )
    manifest_path = Path(spec.calibration_artifact_path) / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_calibration_record(assets.calibration_root, **arguments)


def test_exact_calibration_rejects_ambiguous_retained_revision(assets):
    from billiards_ai.lab.mainline1_calibration_registry import resolve_calibration_record

    spec = inspect(assets)
    path = assets.calibration_root / "registry.json"
    raw = json.loads(path.read_bytes())
    raw["records"].append(raw["records"][0])
    write_json(path, raw)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_calibration_record(
            assets.calibration_root,
            setup_fingerprint=spec.setup_fingerprint,
            artifact_id=spec.calibration_artifact_id,
            revision=spec.calibration_revision,
        )


def test_existing_capture_cli_namespaces_use_fixed_60_fps(assets):
    # Both current capture entrypoints get 60 fps from RealtimeCaptureConfig, not argparse.
    del assets.fps
    assert inspect(assets).fps == 60


@pytest.mark.parametrize("resolution", ["active", "exact", "managed"])
def test_calibration_archive_inventory_rejected_before_filesystem_checks(
    assets, monkeypatch, resolution
):
    from billiards_ai.lab.mainline1_calibration_registry import (
        resolve_active_calibration,
        resolve_calibration_record,
    )

    fingerprint = capture_setup_fingerprint(
        CaptureSetup.model_validate_json(assets.capture_setup.read_bytes())
    )
    manifest_path = assets.calibration_root / assets.first.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    # Only synthetic manifest text: no archive directory or payload is created or inspected.
    manifest["inventory"][".project-archive/fixture"] = {
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    write_json(manifest_path, manifest)
    archive_checks = []
    for method in ("is_symlink", "is_file", "is_dir", "stat", "lstat", "open", "resolve"):
        original = getattr(Path, method)

        def guarded(path, *args, _original=original, _method=method, **kwargs):
            if ".project-archive" in path.parts:
                archive_checks.append(_method)
                raise AssertionError("archive inventory reached filesystem " + _method)
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, guarded)
    with pytest.raises(ValueError, match="unsafe"):
        if resolution == "active":
            resolve_active_calibration(assets.calibration_root, fingerprint)
        elif resolution == "exact":
            resolve_calibration_record(
                assets.calibration_root,
                setup_fingerprint=fingerprint,
                artifact_id=assets.first.artifact_id,
                revision=assets.first.revision,
            )
        else:
            inspect(assets)
    assert archive_checks == []


@pytest.mark.parametrize("name", ["artifact_root", "jobs_root", "ring_root"])
@pytest.mark.parametrize("location", ["existing_file", "missing_child_under_file"])
def test_output_roots_reject_existing_files_and_nondirectory_ancestors(assets, name, location):
    root = getattr(assets, name)
    root.rmdir()
    root.write_bytes(b"ordinary file, not an output directory")
    if location == "missing_child_under_file":
        setattr(assets, name, root / "not-created" / "child")
    with pytest.raises(ValueError, match="directory"):
        inspect(assets)


def test_absent_output_directories_remain_valid_and_are_not_created(assets):
    for name in ("artifact_root", "jobs_root", "ring_root"):
        root = getattr(assets, name)
        root.rmdir()
        setattr(assets, name, root / "missing-parent" / "future-output")
    spec = inspect(assets)
    for name in ("artifact_root", "jobs_root", "ring_root"):
        assert spec.paths[name] == str(getattr(assets, name))
        assert not getattr(assets, name).exists()
        assert not getattr(assets, name).parent.exists()
