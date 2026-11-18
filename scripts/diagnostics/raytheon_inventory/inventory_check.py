"""Read-only Raytheon release and model-asset inventory.

The diagnostic deliberately exports hashes and a small metadata allow-list only.
It never copies model/config payloads and never opens a state database writable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import stat
import sys
from pathlib import Path

PACKET_VERSION = "raytheon-asset-inventory-20260905-v1"
BASELINE = "9a144955377b289ce47de71769d1305dd8585de4"
METADATA_KEYS = (
    "calibration_id",
    "capture_setup_id",
    "created_at_utc",
    "family",
    "gate_status",
    "kind",
    "model_class",
    "pipeline_version",
    "revision",
    "run_hash",
    "run_id",
    "schema_version",
    "source",
    "status",
    "task",
    "use_scope",
)
PRUNED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-rfdetr",
    "__pycache__",
    "datasets",
    "node_modules",
    "recordings",
    "videos",
}
PACKAGE_NAMES = (
    "numpy",
    "onnxruntime",
    "opencv-python",
    "opencv-python-headless",
    "rfdetr",
    "torch",
    "torchvision",
    "transformers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _safe_metadata(payload) -> dict:
    if not isinstance(payload, dict):
        return {}
    result = {}
    generic = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]{0,127}$")
    source = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    timestamp = re.compile(r"^[0-9T:.+Z-]{10,40}$")
    for key in METADATA_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            allowed = (
                timestamp.fullmatch(value)
                if key == "created_at_utc"
                else (source.fullmatch(value) if key == "source" else generic.fullmatch(value))
            )
            if allowed:
                result[key] = value
        elif isinstance(value, (int, float, bool)) and not isinstance(value, complex):
            result[key] = value
    return result


def _is_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            return True
        # Python 3.11 has no Path.is_junction(). Windows exposes all junctions
        # and other reparse points through st_file_attributes.
        if getattr(metadata, "st_file_attributes", 0) & 0x400:
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _path_chain_is_safe(path: Path) -> bool:
    """Reject a link/reparse point anywhere between the filesystem root and path."""
    try:
        absolute = Path(os.path.abspath(path))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            if _is_link(current):
                return False
            if not current.exists():
                break
        return True
    except (OSError, ValueError):
        return False


def _classify(path: Path, sibling_file_names) -> str | None:
    name = path.name.casefold()
    parts = [part.casefold() for part in path.parts]
    parent_names = {str(child).casefold() for child in sibling_file_names}
    if name == "training-manifest.json":
        return "detector_manifest"
    if name == "checkpoint_best.pth":
        return "detector_checkpoint"
    if name == "group-head.onnx":
        return "identity_head"
    if name == "manifest.json" and "group-head.onnx" in parent_names:
        return "identity_manifest"
    if name == "model.safetensors":
        return "dinov2_weights"
    if name == "manifest.json" and "model.safetensors" in parent_names:
        return "dinov2_manifest"
    if name in {"config.json", "preprocessor_config.json"} and "model.safetensors" in parent_names:
        return "dinov2_config"
    if name == "shot-local.yaml":
        return "shot_local_config"
    joined_parts = "/".join(parts)
    if path.suffix.casefold() == ".json" and "capture" in joined_parts and "setup" in joined_parts:
        return "capture_setup"
    if path.suffix.casefold() == ".json" and any("calibration" in part for part in parts):
        return "calibration_json"
    return None


def _manifest_hashes(payload) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    result = {}
    for container_name in ("artifacts", "inventory"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            for name, value in container.items():
                if isinstance(value, str):
                    result[str(name)] = value.casefold()
                elif isinstance(value, dict) and isinstance(value.get("sha256"), str):
                    result[str(name)] = value["sha256"].casefold()
        elif isinstance(container, list):
            for item in container:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("name"), str)
                    and isinstance(item.get("sha256"), str)
                ):
                    result[item["name"]] = item["sha256"].casefold()
    payloads = payload.get("payloads")
    if isinstance(payloads, list):
        for item in payloads:
            if (
                isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("sha256"), str)
            ):
                result[item["name"]] = item["sha256"].casefold()
    return result


def _relation(kind: str, manifest_path: Path, manifest_payload) -> dict:
    allowed_payloads = {
        "detector_run": {"checkpoint_best.pth"},
        "identity_run": {"group-head.onnx", "metrics.json"},
        "dinov2_assets": {"model.safetensors", "config.json", "preprocessor_config.json"},
    }[kind]
    expected = {
        name: value
        for name, value in _manifest_hashes(manifest_payload).items()
        if name in allowed_payloads
    }
    checked = []
    missing = []
    mismatched = []
    for name, expected_hash in sorted(expected.items()):
        # A manifest cannot make the scanner escape its own asset directory.
        payload = manifest_path.parent / name
        if not _path_chain_is_safe(payload) or not payload.is_file() or _is_link(payload):
            missing.append(name)
            continue
        actual_hash = sha256(payload)
        checked.append({"name": name, "sha256": actual_hash})
        if actual_hash.casefold() != expected_hash:
            mismatched.append(name)
    if not isinstance(manifest_payload, dict):
        status = "INVALID_MANIFEST"
    elif not expected or missing:
        status = "INCOMPLETE"
    elif mismatched:
        status = "HASH_MISMATCH"
    else:
        status = "MANIFEST_HASHES_MATCH"
    return {
        "kind": kind,
        "root": str(manifest_path.parent),
        "manifest_sha256": sha256(manifest_path),
        "status": status,
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
    }


def scan_roots(roots, *, maximum_candidates=500, maximum_entries=200_000):
    """Inventory known runtime assets without following links or exporting payloads."""
    candidates = []
    skipped_links = []
    root_status = {}
    relations = []
    entries_seen = 0
    relation_manifests = []
    for label, raw_root in roots.items():
        root = Path(raw_root)
        if not _path_chain_is_safe(root) or not root.is_dir() or _is_link(root):
            root_status[label] = {"path": str(root), "status": "MISSING_OR_UNSAFE"}
            continue
        root_status[label] = {"path": str(root), "status": "PRESENT"}
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(directory)
            kept = []
            for name in sorted(directory_names):
                child = current / name
                if _is_link(child):
                    skipped_links.append(str(child))
                elif name.casefold() not in PRUNED_DIRECTORY_NAMES:
                    kept.append(name)
            directory_names[:] = kept
            for name in sorted(file_names):
                entries_seen += 1
                if entries_seen > maximum_entries:
                    raise ValueError("entry limit exceeded while scanning assets")
                path = current / name
                if _is_link(path):
                    skipped_links.append(str(path))
                    continue
                kind = _classify(path, file_names)
                if kind is None:
                    continue
                if len(candidates) >= maximum_candidates:
                    raise ValueError("candidate limit exceeded while scanning assets")
                item = {
                    "kind": kind,
                    "root_label": label,
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                if path.suffix.casefold() == ".json":
                    payload = _load_json(path)
                    item["metadata"] = _safe_metadata(payload)
                    if kind in {"detector_manifest", "identity_manifest", "dinov2_manifest"}:
                        relation_manifests.append((kind, path, payload))
                candidates.append(item)
    relation_kinds = {
        "detector_manifest": "detector_run",
        "identity_manifest": "identity_run",
        "dinov2_manifest": "dinov2_assets",
    }
    for kind, path, payload in relation_manifests:
        relations.append(_relation(relation_kinds[kind], path, payload))
    return {
        "schema_version": "1.0",
        "read_only": True,
        "roots": root_status,
        "candidates": candidates,
        "relations": relations,
        "skipped_links": sorted(set(skipped_links)),
        "entries_seen": entries_seen,
        "limits": {"maximum_candidates": maximum_candidates, "maximum_entries": maximum_entries},
    }


def python_environment() -> dict:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def prepare_output_directory(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    unexpected = [item.name for item in path.iterdir() if item.name != "packet-manifest.json"]
    if unexpected:
        raise ValueError("output directory is not empty")


def _safe_local_path_value(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        return None
    if any(character in value for character in ("\r", "\n", "\0")):
        return None
    has_unsafe_character = any(character in value[2:] for character in (":", "*", "?"))
    if value.startswith(("\\\\", "//")) or has_unsafe_character:
        return None
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return value
    try:
        return value if Path(value).is_absolute() else None
    except (OSError, ValueError):
        return None


def sanitize_windows_audit(audit: dict) -> dict:
    """Remove untrusted current.json values that are not absolute local paths."""
    result = dict(audit)
    current = audit.get("current")
    result["current"] = (
        {
            key: _safe_local_path_value(current.get(key))
            for key in ("current_release", "previous_release", "artifact", "python_path")
        }
        if isinstance(current, dict)
        else {}
    )
    return result


def inspect_installation_safe(audit: dict, camera_check) -> dict:
    """Hash installation sources and inspect stores only after full path-chain checks."""
    result = {"roots": {}, "state_databases": []}
    roots = {
        "project": audit.get("project_root"),
        "installed_release": audit.get("current", {}).get("current_release"),
    }
    for label, raw_root in roots.items():
        if not raw_root:
            continue
        root = Path(raw_root)
        if not _path_chain_is_safe(root) or not root.is_dir():
            result["roots"][label] = {"path": str(root), "status": "MISSING_OR_UNSAFE"}
            continue
        hashes = {}
        for name in camera_check.SOURCE_FILES:
            path = root / name
            hashes[name] = (
                sha256(path)
                if _path_chain_is_safe(path) and path.is_file() and not _is_link(path)
                else None
            )
        result["roots"][label] = {"path": str(root), "status": "PRESENT_SAFE", "sha256": hashes}
    database_paths = {audit.get("default_state_db")}
    for process in audit.get("processes", []):
        if isinstance(process, dict):
            arguments = process.get("safe_arguments", {})
            if isinstance(arguments, dict):
                database_paths.add(arguments.get("state-db") or arguments.get("statedb"))
    for raw_path in sorted(value for value in database_paths if value):
        safe_value = _safe_local_path_value(raw_path)
        if safe_value is None:
            result["state_databases"].append({"path": None, "status": "UNSAFE_PATH_REJECTED"})
            continue
        path = Path(safe_value)
        if not _path_chain_is_safe(path) or _is_link(path):
            result["state_databases"].append({"path": str(path), "status": "MISSING_OR_UNSAFE"})
        elif not path.is_file():
            result["state_databases"].append({"path": str(path), "status": "MISSING"})
        else:
            result["state_databases"].append(camera_check.inspect_state_db(path))
    return result


def _load_camera_check():
    path = Path(__file__).resolve().parent / "camera_check.py"
    spec = importlib.util.spec_from_file_location("inventory_camera_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--algorithm-root", type=Path, required=True)
    parser.add_argument("--approved-assets-root", type=Path, required=True)
    args = parser.parse_args()
    # run.ps1 creates the directory and places the authenticated packet manifest
    # in it before Python starts. Refuse to overwrite any other pre-existing file.
    prepare_output_directory(args.output)
    camera_check = _load_camera_check()
    errors = []
    audit_inputs = (args.project_root, args.runtime_root, args.runtime_root / "current.json")
    try:
        if not all(_path_chain_is_safe(path) for path in audit_inputs):
            raise ValueError("unsafe audit input path")
        audit = sanitize_windows_audit(
            camera_check.collect_windows_audit(args.project_root, args.runtime_root)
        )
    except Exception as error:  # Preserve a useful return package when one query fails.
        audit = {"status": "UNAVAILABLE", "reason": type(error).__name__}
        errors.append("WINDOWS_AUDIT_UNAVAILABLE")
    camera_check.write_json(args.output / "windows-audit.json", audit)
    try:
        installation = inspect_installation_safe(audit, camera_check)
    except Exception as error:
        installation = {"status": "UNAVAILABLE", "reason": type(error).__name__}
        errors.append("INSTALLATION_INSPECTION_UNAVAILABLE")
    camera_check.write_json(args.output / "installation.json", installation)
    current_release = audit.get("current", {}).get("current_release")
    roots = {
        "project": args.project_root,
        "algorithm_artifacts": args.algorithm_root,
        "approved_assets": args.approved_assets_root,
    }
    if current_release:
        roots["current_release"] = current_release
    try:
        inventory = scan_roots(roots)
    except Exception as error:
        inventory = {"status": "FAILED_CLOSED", "reason": type(error).__name__}
        errors.append("ASSET_SCAN_FAILED_CLOSED")
    camera_check.write_json(args.output / "asset-inventory.json", inventory)
    camera_check.write_json(args.output / "python-environment.json", python_environment())
    packet_manifest = Path(__file__).resolve().parent / "packet-manifest.json"
    if packet_manifest.is_file():
        (args.output / "packet-manifest.json").write_bytes(packet_manifest.read_bytes())
    summary = {
        "schema_version": "1.0",
        "packet_version": PACKET_VERSION,
        "baseline_commit": BASELINE,
        "status": "COLLECTED_REVIEW_REQUIRED" if not errors else "PARTIAL_REVIEW_REQUIRED",
        "read_only": True,
        "errors": errors,
        "candidate_count": len(inventory.get("candidates", [])),
        "relation_count": len(inventory.get("relations", [])),
        "note": "Inventory evidence only; it does not make the system production-ready.",
    }
    target = camera_check.pack_result(args.output, summary)
    print(json.dumps({"result_zip": str(target), "sha256": sha256(target)}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
