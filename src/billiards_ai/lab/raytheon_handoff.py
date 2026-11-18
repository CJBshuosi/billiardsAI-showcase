"""Cross-platform contracts for the Raytheon Windows RF-DETR handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Final
from uuid import uuid4

REGISTRY_RELATIVE_PATH = Path("configs/algorithm/windows/raytheon-rfdetr-smoke.json")
EXPECTED_REGISTRY_SHA256 = "ea34598892f7913ff0a4e467cd37ba375c0e52c68919e9a57a105d10f2da67b9"
TEXT_HASH_MODE = "sha256-normalized-utf8-lf"
CAPTURE_HELPER_FAILURE = 125
CAPTURE_MANIFEST_SCHEMA = "1.0"
CAPTURE_MANIFEST_HASH_SENTINEL = "0" * 64
MAX_CAPTURE_MANIFEST_BYTES = 1024 * 1024
MAX_CAPTURE_STREAM_BYTES: Final = 4 * 1024 * 1024
MAX_CAPTURE_COMBINED_BYTES: Final = 6 * 1024 * 1024
CAPTURE_READ_CHUNK_BYTES: Final = 64 * 1024
CAPTURE_TERMINATE_TIMEOUT_SECONDS: Final = 1.0
CAPTURE_PIPE_DRAIN_TIMEOUT_SECONDS: Final = 2.0
ModuleLoader = Callable[[str], Any]
VersionGetter = Callable[[str], str]


class HandoffContractError(ValueError):
    """A handoff input or artifact violates the frozen contract."""


@dataclass(frozen=True)
class FrozenInputs:
    project_root: Path
    config_path: Path
    config_sha256: str
    weight_sha256: str
    weight_asset_id: str
    seed: int
    batch_size: int
    model_class: str
    packages: Mapping[str, str]


def interpreter_probe() -> dict[str, Any]:
    """Return interpreter identity without importing any third-party package."""
    return {
        "base_prefix": sys.base_prefix,
        "executable": sys.executable,
        "implementation": sys.implementation.name,
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "pointer_bits": struct.calcsize("P") * 8,
        "prefix": sys.prefix,
    }


def collect_runtime_diagnostics(
    *,
    module_loader: ModuleLoader | None = None,
    version_getter: VersionGetter | None = None,
) -> dict[str, Any]:
    """Collect installed Torch/RF-DETR diagnostics only when explicitly invoked."""
    if module_loader is None:
        module_loader = importlib.import_module
    if version_getter is None:
        from importlib.metadata import version

        version_getter = version
    torch = module_loader("torch")
    cuda = torch.cuda
    payload: dict[str, Any] = {
        "python": sys.version,
        "implementation": sys.implementation.name,
        "pointer_bits": struct.calcsize("P") * 8,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": version_getter("torchvision"),
        "rfdetr": version_getter("rfdetr"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(cuda.is_available()),
        "device_count": int(cuda.device_count()),
    }
    if payload["cuda_available"] and payload["device_count"]:
        payload.update(
            {
                "device_name": cuda.get_device_name(0),
                "device_capability": ".".join(map(str, cuda.get_device_capability(0))),
                "memory_bytes": int(cuda.get_device_properties(0).total_memory),
            }
        )
    return payload


def collect_cuda_preflight(
    seed: int,
    *,
    module_loader: ModuleLoader | None = None,
) -> tuple[dict[str, Any], int]:
    """Collect the frozen CUDA preflight payload and its typed native exit."""
    if module_loader is None:
        module_loader = importlib.import_module
    torch = module_loader("torch")
    cuda = torch.cuda
    available = bool(cuda.is_available() and cuda.device_count())
    versions_ok = torch.__version__ == "2.13.0+cu126" and torch.version.cuda == "12.6"
    exit_code = 0 if available and versions_ok else (20 if not available else 22)
    status = (
        "PASS" if exit_code == 0 else ("CUDA_UNAVAILABLE" if exit_code == 20 else "OTHER_FAILURE")
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": status,
        "exit_code": exit_code,
        "detail": "CUDA preflight",
        "seed": seed,
        "synthetic_only": True,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(cuda.is_available()),
        "device_count": int(cuda.device_count()),
    }
    if available:
        free, total = cuda.mem_get_info(0)
        payload.update(
            {
                "device_name": cuda.get_device_name(0),
                "device_capability": ".".join(map(str, cuda.get_device_capability(0))),
                "memory_free_bytes": int(free),
                "memory_total_bytes": int(total),
            }
        )
    return payload, exit_code


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _capture_artifact_directory(path: Path) -> Path:
    if _is_reparse_like(path) or not path.is_dir():
        raise HandoffContractError(
            "capture artifact directory must be an existing non-link directory"
        )
    return path.resolve()


def _capture_member_path(
    artifact_directory: Path,
    path: Path,
    label: str,
    *,
    must_exist: bool,
) -> Path:
    root = _capture_artifact_directory(artifact_directory)
    absolute = path.absolute()
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise HandoffContractError(f"{label} parent is invalid") from exc
    if parent != root or absolute.name in {"", ".", ".."}:
        raise HandoffContractError(f"{label} must be a direct artifact child")
    if _is_reparse_like(absolute):
        raise HandoffContractError(f"{label} must not be a link or reparse point")
    if must_exist:
        return _require_regular_file(absolute, label)
    if absolute.exists():
        raise HandoffContractError(f"{label} already exists")
    return absolute


def _exclusive_binary_writer(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _capture_stream_record(path: Path) -> dict[str, Any]:
    return {
        "byte_count": path.stat().st_size,
        "path": path.name,
        "sha256": _sha256_file(path),
    }


class _CaptureAccounting:
    """Atomically reserve bounded raw-byte evidence across two reader threads."""

    def __init__(self) -> None:
        self.failure_event = threading.Event()
        self._lock = threading.Lock()
        self._observed = {"stderr": 0, "stdout": 0}
        self._reserved = {"stderr": 0, "stdout": 0}
        self._overflowed = False
        self._reader_errors: list[dict[str, str]] = []

    def reserve(self, stream_name: str, observed_bytes: int) -> int:
        """Account for one read and return the prefix length safe to persist."""
        with self._lock:
            self._observed[stream_name] += observed_bytes
            stream_remaining = MAX_CAPTURE_STREAM_BYTES - self._reserved[stream_name]
            combined_reserved = sum(self._reserved.values())
            combined_remaining = MAX_CAPTURE_COMBINED_BYTES - combined_reserved
            permitted = min(observed_bytes, stream_remaining, combined_remaining)
            self._reserved[stream_name] += permitted
            if permitted != observed_bytes:
                self._overflowed = True
                self.failure_event.set()
            return permitted

    def record_reader_error(self, stream_name: str, exc: BaseException) -> None:
        with self._lock:
            self._reader_errors.append(
                {
                    "kind": type(exc).__name__,
                    "message": str(exc),
                    "stream": stream_name,
                }
            )
            self.failure_event.set()

    def snapshot(self) -> tuple[bool, dict[str, int], list[dict[str, str]]]:
        with self._lock:
            observed = {
                "combined_bytes": sum(self._observed.values()),
                "stderr_bytes": self._observed["stderr"],
                "stdout_bytes": self._observed["stdout"],
            }
            return self._overflowed, observed, list(self._reader_errors)


def _capture_pipe(
    stream_name: str,
    source: BinaryIO,
    destination: BinaryIO,
    accounting: _CaptureAccounting,
) -> None:
    try:
        while chunk := source.read(CAPTURE_READ_CHUNK_BYTES):
            permitted = accounting.reserve(stream_name, len(chunk))
            if permitted:
                written = destination.write(chunk[:permitted])
                if written != permitted:
                    raise OSError(f"short capture write: expected {permitted}, wrote {written}")
    except Exception as exc:
        accounting.record_reader_error(stream_name, exc)


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_captured_process_tree(
    process: subprocess.Popen[bytes],
) -> dict[str, Any]:
    """Stop the capture target and descendants, then return manifest evidence."""
    evidence: dict[str, Any] = {
        "attempted": True,
        "graceful_method": None,
        "graceful_sent": False,
        "kill_method": None,
        "kill_sent": False,
        "target_exited": False,
        "target_return_code": None,
    }
    errors: list[str] = []
    deadline = time.monotonic() + CAPTURE_TERMINATE_TIMEOUT_SECONDS

    if os.name == "nt":
        evidence["graceful_method"] = "windows_ctrl_break_process_group"
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                evidence["graceful_sent"] = True
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(f"CTRL_BREAK_EVENT: {type(exc).__name__}: {exc}")
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if process.poll() is None:
            evidence["kill_method"] = "windows_taskkill_tree"
            try:
                taskkill = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=CAPTURE_TERMINATE_TIMEOUT_SECONDS,
                    shell=False,
                )
                evidence["kill_sent"] = taskkill.returncode == 0
                if taskkill.returncode != 0:
                    errors.append(f"taskkill returned {taskkill.returncode}")
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"taskkill: {type(exc).__name__}: {exc}")
        if process.poll() is None:
            evidence["kill_method"] = "windows_process_kill_fallback"
            try:
                process.kill()
                evidence["kill_sent"] = True
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"process.kill: {type(exc).__name__}: {exc}")
    else:
        evidence["graceful_method"] = "posix_process_group_sigterm"
        try:
            os.killpg(process.pid, signal.SIGTERM)
            evidence["graceful_sent"] = True
        except ProcessLookupError:
            pass
        except (OSError, ValueError) as exc:
            errors.append(f"SIGTERM: {type(exc).__name__}: {exc}")
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_exists(process.pid):
                break
            time.sleep(0.01)
        if _process_group_exists(process.pid):
            evidence["kill_method"] = "posix_process_group_sigkill"
            try:
                os.killpg(process.pid, signal.SIGKILL)
                evidence["kill_sent"] = True
            except ProcessLookupError:
                pass
            except (OSError, ValueError) as exc:
                errors.append(f"SIGKILL: {type(exc).__name__}: {exc}")

    try:
        process.wait(timeout=CAPTURE_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        errors.append(f"target wait: {type(exc).__name__}: {exc}")
    evidence["target_exited"] = process.poll() is not None
    evidence["target_return_code"] = process.returncode
    if errors:
        evidence["errors"] = errors
    return evidence


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


def _capture_manifest(
    *,
    argv: Sequence[str],
    capture_status: str,
    target_exit_code: int | None,
    started_at_utc: str,
    ended_at_utc: str,
    stdout_path: Path,
    stderr_path: Path,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "argv": list(argv),
        "capture_status": capture_status,
        "ended_at_utc": ended_at_utc,
        "manifest_sha256": CAPTURE_MANIFEST_HASH_SENTINEL,
        "schema_version": CAPTURE_MANIFEST_SCHEMA,
        "started_at_utc": started_at_utc,
        "stderr": _capture_stream_record(stderr_path),
        "stdout": _capture_stream_record(stdout_path),
        "target_exit_code": target_exit_code,
    }
    if failure is not None:
        manifest["failure"] = dict(failure)
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    return manifest


def run_captured_process(
    artifact_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    manifest_path: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    """Run argv without a shell and persist bounded, exact raw-byte evidence."""
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise HandoffContractError("captured argv must contain non-empty strings")
    root = _capture_artifact_directory(artifact_directory)
    stdout = _capture_member_path(root, stdout_path, "capture stdout", must_exist=False)
    stderr = _capture_member_path(root, stderr_path, "capture stderr", must_exist=False)
    manifest = _capture_member_path(
        root,
        manifest_path,
        "capture manifest",
        must_exist=False,
    )
    if len({stdout, stderr, manifest}) != 3:
        raise HandoffContractError("capture output paths must be distinct")

    started_at_utc = _utc_timestamp()
    capture_status = "CAPTURED"
    target_exit_code: int | None = None
    failure: dict[str, Any] | None = None
    with (
        _exclusive_binary_writer(stdout) as stdout_file,
        _exclusive_binary_writer(stderr) as stderr_file,
    ):
        try:
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_options["start_new_session"] = True
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                **popen_options,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            capture_status = "SPAWN_FAILED"
            failure = {
                "kind": type(exc).__name__,
                "message": str(exc),
            }
        else:
            accounting = _CaptureAccounting()
            assert process.stdout is not None
            assert process.stderr is not None
            readers = [
                threading.Thread(
                    name="raytheon-capture-stdout",
                    target=_capture_pipe,
                    args=("stdout", process.stdout, stdout_file, accounting),
                    daemon=True,
                ),
                threading.Thread(
                    name="raytheon-capture-stderr",
                    target=_capture_pipe,
                    args=("stderr", process.stderr, stderr_file, accounting),
                    daemon=True,
                ),
            ]
            started_readers: list[threading.Thread] = []
            termination: dict[str, Any] | None = None
            try:
                for reader in readers:
                    reader.start()
                    started_readers.append(reader)
                target_exit_observed_at: float | None = None
                while True:
                    if accounting.failure_event.is_set():
                        break
                    if process.poll() is not None:
                        if not any(reader.is_alive() for reader in started_readers):
                            break
                        if target_exit_observed_at is None:
                            target_exit_observed_at = time.monotonic()
                        elif (
                            time.monotonic() - target_exit_observed_at
                            >= CAPTURE_PIPE_DRAIN_TIMEOUT_SECONDS
                        ):
                            accounting.record_reader_error(
                                "capture",
                                TimeoutError("capture pipes did not close after target exit"),
                            )
                            break
                    accounting.failure_event.wait(0.01)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                accounting.record_reader_error("capture", exc)
            finally:
                if accounting.failure_event.is_set() or process.poll() is None:
                    if accounting.failure_event.is_set():
                        termination = _terminate_captured_process_tree(process)
                for reader in started_readers:
                    reader.join(CAPTURE_PIPE_DRAIN_TIMEOUT_SECONDS)
                if any(reader.is_alive() for reader in started_readers):
                    _close_process_pipes(process)
                    for reader in started_readers:
                        reader.join(CAPTURE_TERMINATE_TIMEOUT_SECONDS)
                _close_process_pipes(process)

            target_exit_code = process.poll()
            overflowed, observed, reader_errors = accounting.snapshot()
            if any(reader.is_alive() for reader in started_readers):
                reader_errors.append(
                    {
                        "kind": "TimeoutError",
                        "message": "capture reader thread did not stop",
                        "stream": "capture",
                    }
                )
            stdout_file.flush()
            stderr_file.flush()
            if overflowed:
                capture_status = "CAPTURE_FAILED"
                failure = {
                    "kind": "output_limit_exceeded",
                    "limits": {
                        "combined_bytes": MAX_CAPTURE_COMBINED_BYTES,
                        "per_stream_bytes": MAX_CAPTURE_STREAM_BYTES,
                    },
                    "message": "raw child output exceeded immutable capture limits",
                    "observed": observed,
                    "termination": termination,
                    "written": {
                        "combined_bytes": stdout_file.tell() + stderr_file.tell(),
                        "stderr_bytes": stderr_file.tell(),
                        "stdout_bytes": stdout_file.tell(),
                    },
                }
            elif reader_errors:
                capture_status = "CAPTURE_FAILED"
                failure = {
                    "kind": "reader_exception",
                    "message": "raw child output capture failed",
                    "reader_errors": reader_errors,
                    "termination": termination,
                }
    ended_at_utc = _utc_timestamp()
    payload = _capture_manifest(
        argv=argv,
        capture_status=capture_status,
        target_exit_code=target_exit_code,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        stdout_path=stdout,
        stderr_path=stderr,
        failure=failure,
    )
    with _exclusive_binary_writer(manifest) as manifest_file:
        manifest_file.write(_canonical_json_bytes(payload))
    return payload


def read_capture_manifest(
    artifact_directory: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Validate a canonical capture manifest and its exact stream evidence."""
    root = _capture_artifact_directory(artifact_directory)
    manifest_file = _capture_member_path(
        root,
        manifest_path,
        "capture manifest",
        must_exist=True,
    )
    try:
        with manifest_file.open("rb") as source:
            raw = source.read(MAX_CAPTURE_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise HandoffContractError("capture manifest could not be read") from exc
    if len(raw) > MAX_CAPTURE_MANIFEST_BYTES:
        raise HandoffContractError("capture manifest exceeds size limit")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise HandoffContractError("capture manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or raw != _canonical_json_bytes(manifest):
        raise HandoffContractError("capture manifest is not canonical")
    claimed_hash = manifest.get("manifest_sha256")
    unsigned = {**manifest, "manifest_sha256": CAPTURE_MANIFEST_HASH_SENTINEL}
    if (
        not isinstance(claimed_hash, str)
        or claimed_hash != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise HandoffContractError("capture manifest SHA-256 mismatch")
    if (
        manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA
        or manifest.get("capture_status") != "CAPTURED"
        or not isinstance(manifest.get("argv"), list)
        or not manifest["argv"]
        or any(not isinstance(value, str) or not value for value in manifest["argv"])
        or not isinstance(manifest.get("target_exit_code"), int)
        or not isinstance(manifest.get("started_at_utc"), str)
        or not isinstance(manifest.get("ended_at_utc"), str)
    ):
        raise HandoffContractError("capture manifest contract is invalid")

    streams: list[Path] = []
    for label in ("stdout", "stderr"):
        record = manifest.get(label)
        if not isinstance(record, dict) or set(record) != {"byte_count", "path", "sha256"}:
            raise HandoffContractError(f"capture {label} record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise HandoffContractError(f"capture {label} path is invalid")
        stream = _capture_member_path(
            root,
            root / relative,
            f"capture {label}",
            must_exist=True,
        )
        if (
            not isinstance(record.get("byte_count"), int)
            or record["byte_count"] < 0
            or stream.stat().st_size != record["byte_count"]
            or not isinstance(record.get("sha256"), str)
            or _sha256_file(stream) != record["sha256"]
        ):
            raise HandoffContractError(f"capture {label} evidence mismatch")
        streams.append(stream)
    return manifest, streams[0], streams[1]


def _sha256_normalized_utf8_lf(path: Path) -> str:
    """Hash UTF-8 text after converting CRLF to LF; reject other CR bytes."""
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HandoffContractError("frozen text must be valid UTF-8") from exc
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise HandoffContractError("frozen text contains unsupported CR line endings")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_reparse_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _require_regular_file(path: Path, label: str) -> Path:
    if _is_reparse_like(path) or not path.is_file():
        raise HandoffContractError(f"{label} must be a regular non-link file")
    return path


def _load_registry(project_root: Path) -> tuple[Path, dict[str, Any]]:
    root = project_root.resolve()
    _require_regular_file(root / "pyproject.toml", "project marker")
    registry_path = _require_regular_file(root / REGISTRY_RELATIVE_PATH, "handoff registry")
    if _sha256_normalized_utf8_lf(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise HandoffContractError("handoff registry SHA-256 mismatch")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise HandoffContractError("handoff registry is invalid") from exc
    if registry.get("schema_version") != "1.0" or registry.get("text_hash_mode") != TEXT_HASH_MODE:
        raise HandoffContractError("handoff registry schema is invalid")
    return root, registry


def verify_frozen_inputs(
    project_root: Path,
    weight_path: Path,
    *,
    after_verification: Callable[[], None] | None = None,
) -> FrozenInputs:
    """Verify the immutable registry, config, weight, and smoke identity."""
    root, registry = _load_registry(project_root)
    try:
        weight = registry["weight"]
        smoke = registry["smoke"]
        packages = registry["packages"]
    except (KeyError, TypeError) as exc:
        raise HandoffContractError("handoff registry is invalid") from exc
    config_path = (root / smoke["config_relative_path"]).resolve()
    if not config_path.is_relative_to(root):
        raise HandoffContractError("smoke config escapes project root")
    _require_regular_file(config_path, "smoke config")
    if _sha256_normalized_utf8_lf(config_path) != smoke["config_sha256"]:
        raise HandoffContractError("smoke config SHA-256 mismatch")
    resolved_weight = _require_regular_file(weight_path.resolve(), "Seg-Small weight")
    if _sha256_file(resolved_weight) != weight["sha256"]:
        raise HandoffContractError("Seg-Small weight SHA-256 mismatch")
    if (
        smoke.get("seed") != 20260801
        or smoke.get("batch_size") != 1
        or smoke.get("model_class") != "RFDETRSegSmall"
        or weight.get("asset_id") != "rfdetr-seg-small-1.9.0"
    ):
        raise HandoffContractError("frozen smoke identity is invalid")
    if after_verification is not None:
        after_verification()
    return FrozenInputs(
        project_root=root,
        config_path=config_path,
        config_sha256=str(smoke["config_sha256"]),
        weight_sha256=str(weight["sha256"]),
        weight_asset_id=str(weight["asset_id"]),
        seed=int(smoke["seed"]),
        batch_size=int(smoke["batch_size"]),
        model_class=str(smoke["model_class"]),
        packages=dict(packages),
    )


def classify_smoke_result(
    *,
    native_exit_code: int,
    summary: Mapping[str, Any],
    artifact_directory: Path,
    expected_seed: int,
) -> dict[str, Any]:
    """Map a native exit plus explicit JSON payload to the typed result contract."""
    mappings = {
        0: "PASS",
        20: "CUDA_UNAVAILABLE",
        21: "OOM",
    }
    expected_status = mappings.get(native_exit_code)
    valid = (
        expected_status is not None
        and summary.get("status") == expected_status
        and summary.get("exit_code") == native_exit_code
        and summary.get("seed") == expected_seed
        and summary.get("synthetic_only") is True
    )
    if valid and expected_status == "PASS":
        runtime = summary.get("runtime")
        valid = (
            isinstance(runtime, Mapping)
            and runtime.get("accelerator") == "cuda"
            and runtime.get("cuda_available") is True
            and runtime.get("amp") is True
        )
    status = expected_status if valid else "OTHER_FAILURE"
    exit_code = native_exit_code if valid else 22
    if valid and status == "PASS":
        detail: str | None = None
    else:
        detail = str(summary.get("detail") or "typed smoke result validation failed")
    result = {
        "schema_version": "1.0",
        "status": status,
        "exit_code": exit_code,
        "detail": detail,
        "synthetic_only": True,
        "real_video_count": 0,
        "accuracy_claim": "none_synthetic_smoke_only",
        "seed": expected_seed,
        "batch_size": 1,
        "model_class": "RFDETRSegSmall",
        "artifact_directory": str(artifact_directory.resolve()),
    }
    if valid and isinstance(summary.get("stage"), str):
        result["stage"] = summary["stage"]
    return result


def create_unique_artifact_directory(
    root: Path,
    kind: str,
    *,
    now: datetime | None = None,
    token: str | None = None,
) -> Path:
    """Create a collision-proof empty artifact directory without reuse."""
    if not kind or any(character not in "abcdefghijklmnopqrstuvwxyz-" for character in kind):
        raise HandoffContractError("artifact kind is invalid")
    selected_now = (now or datetime.now(UTC)).astimezone(UTC)
    selected_token = token or uuid4().hex
    if len(selected_token) != 32 or any(
        character not in "0123456789abcdef" for character in selected_token
    ):
        raise HandoffContractError("artifact token is invalid")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = selected_now.strftime("%Y%m%dT%H%M%S.%fZ")
    target = root / f"{kind}-{timestamp}-{selected_token}"
    target.mkdir()
    return target


def create_project_artifact_directory(
    project_root: Path,
    artifacts_root: Path,
    kind: str,
) -> Path:
    """Create evidence outside the repository so Git can never capture it."""
    project = project_root.resolve()
    _require_regular_file(project / "pyproject.toml", "project marker")
    root = artifacts_root.resolve()
    if root == project or root.is_relative_to(project):
        raise HandoffContractError("artifacts root must be outside project root")
    return create_unique_artifact_directory(root, kind)


def validate_venv_identity(requested_venv: Path, probe: Mapping[str, Any]) -> None:
    """Require an isolated 64-bit CPython 3.12 rooted at the requested venv."""
    requested = requested_venv.resolve()
    try:
        prefix = Path(str(probe["prefix"])).resolve()
        base_prefix = Path(str(probe["base_prefix"])).resolve()
        executable = Path(str(probe["executable"])).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise HandoffContractError("interpreter probe is incomplete") from exc
    if probe.get("implementation") != "cpython" or (probe.get("major"), probe.get("minor")) != (
        3,
        12,
    ):
        raise HandoffContractError("venv must use CPython 3.12")
    if probe.get("pointer_bits") != 64:
        raise HandoffContractError("venv must use 64-bit Python")
    if prefix != requested:
        raise HandoffContractError("venv sys.prefix does not match requested .venv-rfdetr")
    if base_prefix == prefix:
        raise HandoffContractError("venv sys.base_prefix must differ from sys.prefix")
    if not executable.is_relative_to(prefix):
        raise HandoffContractError("venv executable is outside requested .venv-rfdetr")


def validate_uninstall_target(
    project_root: Path,
    requested_venv: Path,
    probe: Mapping[str, Any],
) -> None:
    """Require the exact repository-owned non-link venv before deletion."""
    project = project_root.resolve()
    venv = requested_venv.resolve()
    _require_regular_file(project / "pyproject.toml", "project marker")
    if requested_venv.name != ".venv-rfdetr" or venv.parent != project:
        raise HandoffContractError("uninstall target must be project .venv-rfdetr")
    if _is_reparse_like(requested_venv) or not requested_venv.is_dir():
        raise HandoffContractError("uninstall target must be a non-link directory")
    _require_regular_file(requested_venv / "pyvenv.cfg", "venv pyvenv.cfg")
    validate_venv_identity(requested_venv, probe)


def _require_selected_run(artifacts_root: Path, selected: Path, prefix: str) -> Path:
    root = artifacts_root.resolve()
    resolved = selected.resolve()
    if (
        selected.is_symlink()
        or resolved.parent != root
        or not resolved.name.startswith(prefix)
        or not resolved.is_dir()
    ):
        raise HandoffContractError(f"selected {prefix.rstrip('-')} directory is invalid")
    return resolved


def create_evidence_bundle(
    artifacts_root: Path,
    install_directory: Path,
    smoke_directory: Path,
    *,
    now: datetime | None = None,
    token: str | None = None,
) -> Path:
    """Copy only the selected current install/smoke top-level evidence."""
    install = _require_selected_run(artifacts_root, install_directory, "install-")
    smoke = _require_selected_run(artifacts_root, smoke_directory, "smoke-")
    result_path = _require_regular_file(smoke / "result.json", "current smoke result")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise HandoffContractError("current smoke result is invalid") from exc
    if result.get("status") not in {"PASS", "CUDA_UNAVAILABLE", "OOM", "OTHER_FAILURE"}:
        raise HandoffContractError("current smoke result status is invalid")
    bundle = create_unique_artifact_directory(
        artifacts_root,
        "bundle",
        now=now,
        token=token,
    )
    allowed_suffixes = {".json", ".txt", ".csv"}
    for label, source in (("install", install), ("smoke", smoke)):
        destination = bundle / label
        destination.mkdir()
        for item in source.iterdir():
            if item.is_file() and not item.is_symlink() and item.suffix.lower() in allowed_suffixes:
                shutil.copyfile(item, destination / item.name)
    return bundle


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError) as exc:
        raise HandoffContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise HandoffContractError(f"{label} must be a JSON object")
    return value


def _read_diagnostic_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        with path.open("rb") as source:
            payload = source.read(4097)
    except OSError as exc:
        return f"stderr unavailable: {exc}"
    truncated = len(payload) > 4096
    text = payload[:4096].decode("utf-8-sig", errors="replace").replace("\x00", "")
    cleaned = text.strip()
    if not cleaned:
        return None
    return cleaned + (" [truncated]" if truncated else "")


def _classification_summary(
    native_exit_code: int,
    summary_path: Path,
    stderr_path: Path | None,
) -> dict[str, Any]:
    summary_error: str | None = None
    try:
        summary = _read_json_file(summary_path, "smoke summary")
    except HandoffContractError as exc:
        summary = {}
        summary_error = str(exc)
    if native_exit_code in {0, 20, 21} and summary_error is None:
        return summary
    details = [f"native exit code {native_exit_code}"]
    summary_detail = summary.get("detail")
    if isinstance(summary_detail, str) and summary_detail.strip():
        details.append(summary_detail.strip())
    if summary_error is not None:
        details.append(summary_error)
    stderr_detail = _read_diagnostic_text(stderr_path)
    if stderr_detail is not None:
        details.append(f"stderr: {stderr_detail}")
    return {"detail": "; ".join(details)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("interpreter-probe")
    subparsers.add_parser("runtime-diagnostics")

    cuda = subparsers.add_parser("cuda-preflight")
    cuda.add_argument("--seed", type=int, required=True)

    echo = subparsers.add_parser("echo-argument")
    echo.add_argument("--value", required=True)

    native_exit = subparsers.add_parser("exit-with-code")
    native_exit.add_argument("--code", type=int, choices=(7,), required=True)

    subparsers.add_parser("native-warning-probe")

    capture = subparsers.add_parser("run-captured")
    capture.add_argument("--artifact-directory", type=Path, required=True)
    capture.add_argument("--stdout", type=Path, required=True)
    capture.add_argument("--stderr", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("target_argv", nargs=argparse.REMAINDER)

    capture_exit = subparsers.add_parser("read-capture-exit")
    capture_exit.add_argument("--artifact-directory", type=Path, required=True)
    capture_exit.add_argument("--manifest", type=Path, required=True)

    create = subparsers.add_parser("create-artifact")
    create.add_argument("--project-root", type=Path, required=True)
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--kind", choices=("install", "smoke", "bundle", "selftest"), required=True)

    classify = subparsers.add_parser("classify-smoke")
    classify.add_argument("--project-root", type=Path, required=True)
    classification_source = classify.add_mutually_exclusive_group(required=True)
    classification_source.add_argument("--native-exit-code", type=int)
    classification_source.add_argument("--capture-manifest", type=Path)
    classify.add_argument("--summary", type=Path)
    classify.add_argument("--stderr", type=Path)
    classify.add_argument("--artifact-directory", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--project-root", type=Path, required=True)
    verify.add_argument("--weight", type=Path, required=True)

    plan = subparsers.add_parser("package-plan")
    plan.add_argument("--project-root", type=Path, required=True)

    validate = subparsers.add_parser("validate-venv")
    validate.add_argument("--venv", type=Path, required=True)
    validate.add_argument("--probe", type=Path, required=True)

    uninstall = subparsers.add_parser("validate-uninstall")
    uninstall.add_argument("--project-root", type=Path, required=True)
    uninstall.add_argument("--venv", type=Path, required=True)
    uninstall.add_argument("--probe", type=Path, required=True)

    bundle = subparsers.add_parser("bundle-evidence")
    bundle.add_argument("--artifacts-root", type=Path, required=True)
    bundle.add_argument("--install-directory", type=Path, required=True)
    bundle.add_argument("--smoke-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute handoff decisions without package installation or external access."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "interpreter-probe":
            print(_canonical_json(interpreter_probe()))
            return 0
        if args.command == "runtime-diagnostics":
            print(_canonical_json(collect_runtime_diagnostics()))
            return 0
        if args.command == "cuda-preflight":
            payload, exit_code = collect_cuda_preflight(args.seed)
            print(_canonical_json(payload))
            return exit_code
        if args.command == "echo-argument":
            print(args.value)
            return 0
        if args.command == "exit-with-code":
            return int(args.code)
        if args.command == "native-warning-probe":
            print(
                _canonical_json(
                    {
                        "exit_code": 0,
                        "runtime": {
                            "accelerator": "cuda",
                            "amp": True,
                            "cuda_available": True,
                        },
                        "schema_version": "1.0",
                        "seed": 20260801,
                        "status": "PASS",
                        "synthetic_only": True,
                    }
                )
            )
            print("nonfatal native stderr warning", file=sys.stderr)
            return 0
        if args.command == "run-captured":
            target_argv = list(args.target_argv)
            if target_argv and target_argv[0] == "--":
                target_argv.pop(0)
            try:
                manifest = run_captured_process(
                    args.artifact_directory,
                    args.stdout,
                    args.stderr,
                    args.manifest,
                    target_argv,
                )
            except (HandoffContractError, OSError) as exc:
                print(f"Raytheon capture failed: {exc}", file=sys.stderr)
                return CAPTURE_HELPER_FAILURE
            if manifest["capture_status"] == "CAPTURED":
                return 0
            failure_kind = manifest.get("failure", {}).get("kind", "unknown")
            print(f"Raytheon capture failed: {failure_kind}", file=sys.stderr)
            return CAPTURE_HELPER_FAILURE
        if args.command == "read-capture-exit":
            try:
                manifest, _, _ = read_capture_manifest(
                    args.artifact_directory,
                    args.manifest,
                )
            except (HandoffContractError, OSError) as exc:
                print(f"Raytheon capture failed: {exc}", file=sys.stderr)
                return CAPTURE_HELPER_FAILURE
            print(manifest["target_exit_code"])
            return 0
        if args.command == "create-artifact":
            print(
                create_project_artifact_directory(
                    args.project_root,
                    args.root,
                    args.kind,
                )
            )
            return 0
        if args.command == "classify-smoke":
            if args.capture_manifest is not None:
                try:
                    manifest, summary_path, stderr_path = read_capture_manifest(
                        args.artifact_directory,
                        args.capture_manifest,
                    )
                except (HandoffContractError, OSError) as exc:
                    print(f"Raytheon capture failed: {exc}", file=sys.stderr)
                    return CAPTURE_HELPER_FAILURE
                native_exit_code = int(manifest["target_exit_code"])
            else:
                if args.summary is None:
                    raise HandoffContractError(
                        "classification summary is required with native exit code"
                    )
                native_exit_code = int(args.native_exit_code)
                summary_path = args.summary
                stderr_path = args.stderr
            summary = _classification_summary(
                native_exit_code,
                summary_path,
                stderr_path,
            )
            expected_seed = 20260801
            classified_native_exit = native_exit_code
            try:
                _, registry = _load_registry(args.project_root)
                registry_seed = int(registry["smoke"]["seed"])
                if registry_seed != expected_seed:
                    raise HandoffContractError("handoff registry seed is invalid")
            except (HandoffContractError, KeyError, TypeError, ValueError) as exc:
                detail = str(summary.get("detail") or "native smoke failure")
                summary = {"detail": f"{detail}; registry validation failed: {exc}"}
                classified_native_exit = 22
            result = classify_smoke_result(
                native_exit_code=classified_native_exit,
                summary=summary,
                artifact_directory=args.artifact_directory,
                expected_seed=expected_seed,
            )
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return int(result["exit_code"])
        if args.command == "verify-inputs":
            frozen = verify_frozen_inputs(args.project_root, args.weight)
            print(
                json.dumps(
                    {
                        "config_path": str(frozen.config_path),
                        "config_sha256": frozen.config_sha256,
                        "weight_sha256": frozen.weight_sha256,
                        "weight_asset_id": frozen.weight_asset_id,
                        "seed": frozen.seed,
                        "batch_size": frozen.batch_size,
                        "model_class": frozen.model_class,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "package-plan":
            _, registry = _load_registry(args.project_root)
            print(json.dumps(registry["packages"], sort_keys=True))
            return 0
        if args.command == "validate-venv":
            validate_venv_identity(args.venv, _read_json_file(args.probe, "venv probe"))
            return 0
        if args.command == "validate-uninstall":
            validate_uninstall_target(
                args.project_root,
                args.venv,
                _read_json_file(args.probe, "venv probe"),
            )
            return 0
        if args.command == "bundle-evidence":
            print(
                create_evidence_bundle(
                    args.artifacts_root,
                    args.install_directory,
                    args.smoke_directory,
                )
            )
            return 0
    except (HandoffContractError, FileExistsError, OSError, KeyError) as exc:
        print(f"Raytheon handoff failed: {exc}", file=sys.stderr)
        return 2
    raise RuntimeError("unreachable handoff command")


if __name__ == "__main__":
    raise SystemExit(main())
