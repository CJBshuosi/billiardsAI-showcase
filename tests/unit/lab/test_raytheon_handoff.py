from __future__ import annotations

import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

HELPER = Path.cwd() / "src/billiards_ai/lab/raytheon_handoff.py"
CAPTURE_HELPER_FAILURE = 125


def _api():
    try:
        from billiards_ai.lab import raytheon_handoff
    except ImportError as exc:  # expected in the initial TDD red run
        pytest.fail(f"Raytheon handoff module is missing: {exc}")
    return raytheon_handoff


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _run_captured(
    artifact_directory: Path,
    argv: list[str],
) -> tuple[subprocess.CompletedProcess[bytes], Path, Path, Path]:
    stdout_path = artifact_directory / "target.stdout.bin"
    stderr_path = artifact_directory / "target.stderr.bin"
    manifest_path = artifact_directory / "capture-manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "run-captured",
            "--artifact-directory",
            str(artifact_directory),
            "--stdout",
            str(stdout_path),
            "--stderr",
            str(stderr_path),
            "--manifest",
            str(manifest_path),
            "--",
            *argv,
        ],
        check=False,
        capture_output=True,
    )
    return completed, stdout_path, stderr_path, manifest_path


def _canonical_manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _project(tmp_path: Path) -> tuple[Path, Path, str]:
    project = tmp_path / "project"
    config = project / "configs" / "algorithm" / "training" / "rfdetr-single-batch-smoke.yaml"
    registry = project / "configs" / "algorithm" / "windows" / "raytheon-rfdetr-smoke.json"
    config.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    config.write_text("seed: 20260801\nbatch_size: 1\n", encoding="utf-8")
    weight = tmp_path / "rf-detr-seg-s-ft.pth"
    weight.write_bytes(b"verified fixture weight")
    registry.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "request_id": "windows-cuda-rfdetr-smoke-r1",
                "text_hash_mode": "sha256-normalized-utf8-lf",
                "weight": {
                    "asset_id": "rfdetr-seg-small-1.9.0",
                    "sha256": _sha256(weight),
                },
                "smoke": {
                    "config_relative_path": (
                        "configs/algorithm/training/rfdetr-single-batch-smoke.yaml"
                    ),
                    "config_sha256": _sha256(config),
                    "seed": 20260801,
                    "batch_size": 1,
                    "model_class": "RFDETRSegSmall",
                },
                "packages": {
                    "pypi_index": "https://pypi.org/simple",
                    "pytorch_index": "https://download.pytorch.org/whl/cu126",
                    "torch": "torch==2.13.0+cu126",
                    "torchvision": "torchvision==0.28.0+cu126",
                    "rfdetr": "rfdetr[train,onnx]==1.9.0",
                },
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return project, weight, _sha256(registry)


def test_frozen_inputs_verify_registry_config_weight_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    project, weight, registry_sha256 = _project(tmp_path)
    monkeypatch.setattr(api, "EXPECTED_REGISTRY_SHA256", registry_sha256)

    verified = api.verify_frozen_inputs(project, weight)

    assert verified.weight_sha256 == _sha256(weight)
    assert verified.config_path.name == "rfdetr-single-batch-smoke.yaml"
    assert verified.seed == 20260801
    assert verified.batch_size == 1


def test_file_backed_interpreter_probe_prints_exact_canonical_json() -> None:
    helper = Path.cwd() / "src/billiards_ai/lab/raytheon_handoff.py"

    completed = subprocess.run(
        [sys.executable, str(helper), "interpreter-probe"],
        check=False,
        capture_output=True,
        text=True,
    )

    expected = {
        "base_prefix": sys.base_prefix,
        "executable": sys.executable,
        "implementation": sys.implementation.name,
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "pointer_bits": struct.calcsize("P") * 8,
        "prefix": sys.prefix,
    }
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert (
        completed.stdout
        == json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_runtime_diagnostics_supports_injected_torch_and_metadata() -> None:
    api = _api()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Fixture GPU"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (7, 5)

        @staticmethod
        def get_device_properties(index: int):
            assert index == 0
            return type("Properties", (), {"total_memory": 4_294_967_296})()

    fake_torch = type(
        "FakeTorch",
        (),
        {
            "__version__": "2.13.0+cu126",
            "cuda": FakeCuda(),
            "version": type("Version", (), {"cuda": "12.6"})(),
        },
    )()
    versions = {"torchvision": "0.28.0+cu126", "rfdetr": "1.9.0"}

    diagnostics = api.collect_runtime_diagnostics(
        module_loader=lambda name: fake_torch,
        version_getter=versions.__getitem__,
    )

    assert diagnostics["torch"] == "2.13.0+cu126"
    assert diagnostics["torchvision"] == "0.28.0+cu126"
    assert diagnostics["rfdetr"] == "1.9.0"
    assert diagnostics["cuda_runtime"] == "12.6"
    assert diagnostics["cuda_available"] is True
    assert diagnostics["device_count"] == 1
    assert diagnostics["device_name"] == "Fixture GPU"
    assert diagnostics["device_capability"] == "7.5"
    assert diagnostics["memory_bytes"] == 4_294_967_296


def test_cuda_preflight_and_self_test_commands_are_file_backed(capsys) -> None:
    api = _api()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    fake_torch = type(
        "FakeTorch",
        (),
        {
            "__version__": "2.13.0+cu126",
            "cuda": FakeCuda(),
            "version": type("Version", (), {"cuda": "12.6"})(),
        },
    )()

    payload, exit_code = api.collect_cuda_preflight(
        20260801,
        module_loader=lambda name: fake_torch,
    )

    assert exit_code == 20
    assert payload["status"] == "CUDA_UNAVAILABLE"
    assert payload["seed"] == 20260801
    assert api.main(["echo-argument", "--value", "path with spaces"]) == 0
    assert capsys.readouterr().out == "path with spaces\n"
    assert api.main(["exit-with-code", "--code", "7"]) == 7


def test_native_warning_probe_emits_typed_pass_stdout_warning_stderr_and_exit_zero(
    capsys,
) -> None:
    api = _api()

    exit_code = api.main(["native-warning-probe"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary == {
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
    assert captured.err == "nonfatal native stderr warning\n"


def test_run_captured_round_trips_unicode_and_space_argv_as_raw_utf8(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    value = "path with spaces / \u5e26 \u7a7a\u683c \u7684\u53c2\u6570"

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(HELPER), "echo-argument", "--value", value],
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""
    assert stdout_path.read_bytes() == (value + "\n").encode("utf-8")
    assert stderr_path.read_bytes() == b""
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["argv"][-1] == value
    assert manifest["target_exit_code"] == 0


def test_run_captured_preserves_warning_bytes_and_manifest_hashes_for_pass(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(HELPER), "native-warning-probe"],
    )

    assert completed.returncode == 0
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    assert json.loads(stdout_bytes)["status"] == "PASS"
    assert stderr_bytes == b"nonfatal native stderr warning\n"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["schema_version"] == "1.0"
    assert manifest["capture_status"] == "CAPTURED"
    assert manifest["target_exit_code"] == 0
    assert manifest["stdout"] == {
        "byte_count": len(stdout_bytes),
        "path": stdout_path.name,
        "sha256": hashlib.sha256(stdout_bytes).hexdigest(),
    }
    assert manifest["stderr"] == {
        "byte_count": len(stderr_bytes),
        "path": stderr_path.name,
        "sha256": hashlib.sha256(stderr_bytes).hexdigest(),
    }
    unsigned = {**manifest, "manifest_sha256": "0" * 64}
    assert (
        manifest["manifest_sha256"]
        == hashlib.sha256(_canonical_manifest_bytes(unsigned)).hexdigest()
    )
    assert manifest_path.read_bytes() == _canonical_manifest_bytes(manifest)

    result_path = artifact / "result.json"
    classified = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "classify-smoke",
            "--project-root",
            str(Path.cwd()),
            "--capture-manifest",
            str(manifest_path),
            "--artifact-directory",
            str(artifact),
            "--output",
            str(result_path),
        ],
        check=False,
        capture_output=True,
    )
    assert classified.returncode == 0
    assert json.loads(result_path.read_bytes())["status"] == "PASS"


@pytest.mark.parametrize(
    ("status", "target_exit"),
    [("CUDA_UNAVAILABLE", 20), ("OOM", 21)],
)
def test_captured_typed_failure_retains_exit_and_status(
    tmp_path: Path,
    status: str,
    target_exit: int,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = {
        "exit_code": target_exit,
        "schema_version": "1.0",
        "seed": 20260801,
        "status": status,
        "synthetic_only": True,
    }
    probe = tmp_path / "typed_failure_probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.stdout.write({(json.dumps(payload, sort_keys=True) + chr(10))!r})\n"
        f"raise SystemExit({target_exit})\n",
        encoding="ascii",
    )
    captured, _, _, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )
    assert captured.returncode == 0
    result_path = artifact / "result.json"

    classified = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "classify-smoke",
            "--project-root",
            str(Path.cwd()),
            "--capture-manifest",
            str(manifest_path),
            "--artifact-directory",
            str(artifact),
            "--output",
            str(result_path),
        ],
        check=False,
        capture_output=True,
    )

    assert classified.returncode == target_exit
    result = json.loads(result_path.read_bytes())
    assert result["exit_code"] == target_exit
    assert result["status"] == status


@pytest.mark.parametrize(
    ("stdout_bytes", "target_exit"),
    [(b"", 7), (b"not-json\n", 0)],
)
def test_captured_missing_or_malformed_summary_fails_closed_with_result(
    tmp_path: Path,
    stdout_bytes: bytes,
    target_exit: int,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    probe = tmp_path / "invalid_summary_probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.stdout.buffer.write({stdout_bytes!r})\n"
        "sys.stderr.buffer.write(b'bounded diagnostic warning\\n')\n"
        f"raise SystemExit({target_exit})\n",
        encoding="ascii",
    )
    captured, _, _, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )
    assert captured.returncode == 0
    result_path = artifact / "result.json"

    classified = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "classify-smoke",
            "--project-root",
            str(Path.cwd()),
            "--capture-manifest",
            str(manifest_path),
            "--artifact-directory",
            str(artifact),
            "--output",
            str(result_path),
        ],
        check=False,
        capture_output=True,
    )

    assert classified.returncode == 22
    result = json.loads(result_path.read_bytes())
    assert result["exit_code"] == 22
    assert result["status"] == "OTHER_FAILURE"
    assert "bounded diagnostic warning" in result["detail"]


def test_run_captured_keeps_target_nonzero_as_manifest_data(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    completed, _, _, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(HELPER), "exit-with-code", "--code", "7"],
    )

    assert completed.returncode == 0
    assert json.loads(manifest_path.read_bytes())["target_exit_code"] == 7
    read_exit = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "read-capture-exit",
            "--artifact-directory",
            str(artifact),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
    )
    assert read_exit.returncode == 0
    assert read_exit.stdout == b"7\n"
    assert read_exit.stderr == b""


def test_run_captured_streams_large_output_without_text_conversion(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    probe = tmp_path / "large_output_probe.py"
    probe.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'\\x00\\xff' * 1048576)\n"
        "sys.stderr.buffer.write(b'warning\\r\\n' * 65536)\n",
        encoding="ascii",
    )

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )

    assert completed.returncode == 0
    assert stdout_path.stat().st_size == 2_097_152
    assert stdout_path.read_bytes()[:4] == b"\x00\xff\x00\xff"
    assert stderr_path.stat().st_size == len(b"warning\r\n") * 65_536
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["stdout"]["byte_count"] == 2_097_152
    assert manifest["stderr"]["byte_count"] == len(b"warning\r\n") * 65_536


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_run_captured_caps_a_single_stream_and_records_limit_failure(
    tmp_path: Path,
    stream_name: str,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    probe = tmp_path / f"{stream_name}_overflow_probe.py"
    stream = "sys.stdout" if stream_name == "stdout" else "sys.stderr"
    probe.write_text(
        f"import sys\n{stream}.buffer.write(b'x' * 4194305)\n{stream}.buffer.flush()\n",
        encoding="ascii",
    )

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )

    assert completed.returncode == CAPTURE_HELPER_FAILURE
    assert completed.stdout == b""
    assert b"output_limit_exceeded" in completed.stderr
    selected_path = stdout_path if stream_name == "stdout" else stderr_path
    other_path = stderr_path if stream_name == "stdout" else stdout_path
    assert selected_path.stat().st_size == 4_194_304
    assert other_path.stat().st_size == 0
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["capture_status"] == "CAPTURE_FAILED"
    assert manifest["failure"]["kind"] == "output_limit_exceeded"
    assert manifest["failure"]["limits"] == {
        "combined_bytes": 6_291_456,
        "per_stream_bytes": 4_194_304,
    }
    assert manifest["failure"]["observed"][f"{stream_name}_bytes"] >= 4_194_305
    assert manifest["failure"]["written"][f"{stream_name}_bytes"] == 4_194_304
    assert manifest["failure"]["written"]["combined_bytes"] == 4_194_304
    assert manifest["failure"]["termination"]["attempted"] is True
    assert manifest["failure"]["termination"]["target_exited"] is True
    unsigned = {**manifest, "manifest_sha256": "0" * 64}
    assert (
        manifest["manifest_sha256"]
        == hashlib.sha256(_canonical_manifest_bytes(unsigned)).hexdigest()
    )
    assert manifest_path.read_bytes() == _canonical_manifest_bytes(manifest)


def test_run_captured_caps_combined_output_before_either_stream_limit(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    probe = tmp_path / "combined_overflow_probe.py"
    probe.write_text(
        "import os\n"
        "import threading\n"
        "def emit(descriptor, byte):\n"
        "    remaining = 3145729\n"
        "    while remaining:\n"
        "        size = min(65536, remaining)\n"
        "        os.write(descriptor, byte * size)\n"
        "        remaining -= size\n"
        "threads = [\n"
        "    threading.Thread(target=emit, args=(1, b'o')),\n"
        "    threading.Thread(target=emit, args=(2, b'e')),\n"
        "]\n"
        "for thread in threads:\n"
        "    thread.start()\n"
        "for thread in threads:\n"
        "    thread.join()\n",
        encoding="ascii",
    )

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )

    assert completed.returncode == CAPTURE_HELPER_FAILURE
    assert stdout_path.stat().st_size <= 4_194_304
    assert stderr_path.stat().st_size <= 4_194_304
    assert stdout_path.stat().st_size + stderr_path.stat().st_size == 6_291_456
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["failure"]["kind"] == "output_limit_exceeded"
    assert manifest["failure"]["observed"]["combined_bytes"] >= 6_291_457
    assert manifest["failure"]["written"] == {
        "combined_bytes": 6_291_456,
        "stderr_bytes": stderr_path.stat().st_size,
        "stdout_bytes": stdout_path.stat().st_size,
    }


def test_run_captured_accepts_exact_stream_and_combined_boundaries(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    probe = tmp_path / "exact_boundary_probe.py"
    probe.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'o' * 4194304)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'e' * 2097152)\n"
        "sys.stderr.buffer.flush()\n",
        encoding="ascii",
    )

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )

    assert completed.returncode == 0
    assert stdout_path.read_bytes() == b"o" * 4_194_304
    assert stderr_path.read_bytes() == b"e" * 2_097_152
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["capture_status"] == "CAPTURED"
    assert manifest["stdout"]["byte_count"] == 4_194_304
    assert manifest["stderr"]["byte_count"] == 2_097_152


def test_run_captured_drains_simultaneous_stdout_and_stderr_without_deadlock(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    probe = tmp_path / "simultaneous_output_probe.py"
    probe.write_text(
        "import os\n"
        "import threading\n"
        "def emit(descriptor, byte):\n"
        "    for _ in range(32):\n"
        "        os.write(descriptor, byte * 65536)\n"
        "threads = [\n"
        "    threading.Thread(target=emit, args=(1, b'o')),\n"
        "    threading.Thread(target=emit, args=(2, b'e')),\n"
        "]\n"
        "for thread in threads:\n"
        "    thread.start()\n"
        "for thread in threads:\n"
        "    thread.join()\n",
        encoding="ascii",
    )

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(probe)],
    )

    assert completed.returncode == 0
    assert stdout_path.read_bytes() == b"o" * 2_097_152
    assert stderr_path.read_bytes() == b"e" * 2_097_152
    assert json.loads(manifest_path.read_bytes())["target_exit_code"] == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_run_captured_force_kills_overflowing_target_tree_after_timeout(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    heartbeat = tmp_path / "child-heartbeat.bin"
    child_pid_path = tmp_path / "child.pid"
    child_probe = tmp_path / "stubborn_child.py"
    child_probe.write_text(
        "import signal\n"
        "import time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"path = Path({str(heartbeat)!r})\n"
        "with path.open('ab', buffering=0) as sink:\n"
        "    while True:\n"
        "        sink.write(b'x')\n"
        "        time.sleep(0.02)\n",
        encoding="ascii",
    )
    parent_probe = tmp_path / "stubborn_parent.py"
    parent_probe.write_text(
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"child = subprocess.Popen([sys.executable, {str(child_probe)!r}])\n"
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding='ascii')\n"
        f"heartbeat = Path({str(heartbeat)!r})\n"
        "while not heartbeat.exists():\n"
        "    time.sleep(0.01)\n"
        "sys.stdout.buffer.write(b'x' * 4194305)\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(1.5)\n",
        encoding="ascii",
    )

    completed, _, _, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(parent_probe)],
    )
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    try:
        heartbeat_size = heartbeat.stat().st_size
        time.sleep(0.15)
        assert completed.returncode == CAPTURE_HELPER_FAILURE
        assert heartbeat.stat().st_size == heartbeat_size
        termination = json.loads(manifest_path.read_bytes())["failure"]["termination"]
        assert termination["graceful_sent"] is True
        assert termination["kill_sent"] is True
        assert termination["target_exited"] is True
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_run_captured_turns_reader_thread_exception_into_failure_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    stdout_path = artifact / "target.stdout.bin"
    stderr_path = artifact / "target.stderr.bin"
    manifest_path = artifact / "capture-manifest.json"
    probe = tmp_path / "sleep_probe.py"
    probe.write_text("import time\ntime.sleep(10)\n", encoding="ascii")
    original_capture_pipe = api._capture_pipe

    def capture_pipe(stream_name, source, destination, accounting) -> None:
        if stream_name == "stdout":
            accounting.record_reader_error("stdout", OSError("injected reader failure"))
            return
        original_capture_pipe(stream_name, source, destination, accounting)

    monkeypatch.setattr(api, "_capture_pipe", capture_pipe)

    manifest = api.run_captured_process(
        artifact,
        stdout_path,
        stderr_path,
        manifest_path,
        [sys.executable, str(probe)],
    )

    assert manifest["capture_status"] == "CAPTURE_FAILED"
    assert manifest["failure"]["kind"] == "reader_exception"
    assert manifest["failure"]["reader_errors"] == [
        {
            "kind": "OSError",
            "message": "injected reader failure",
            "stream": "stdout",
        }
    ]
    assert manifest["failure"]["termination"]["attempted"] is True
    assert manifest["failure"]["termination"]["target_exited"] is True
    assert manifest_path.read_bytes() == _canonical_manifest_bytes(manifest)


def test_run_captured_records_spawn_failure_as_distinct_helper_failure(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    completed, stdout_path, stderr_path, manifest_path = _run_captured(
        artifact,
        [str(tmp_path / "missing executable")],
    )

    assert completed.returncode == CAPTURE_HELPER_FAILURE
    assert stdout_path.read_bytes() == b""
    assert stderr_path.read_bytes() == b""
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["capture_status"] == "SPAWN_FAILED"
    assert manifest["target_exit_code"] is None


def test_run_captured_rejects_outside_and_link_capture_targets(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    outside = tmp_path / "outside.bin"
    stderr_path = artifact / "target.stderr.bin"
    manifest_path = artifact / "capture-manifest.json"
    outside_attempt = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "run-captured",
            "--artifact-directory",
            str(artifact),
            "--stdout",
            str(outside),
            "--stderr",
            str(stderr_path),
            "--manifest",
            str(manifest_path),
            "--",
            sys.executable,
            str(HELPER),
            "echo-argument",
            "--value",
            "must not run",
        ],
        check=False,
        capture_output=True,
    )
    assert outside_attempt.returncode == CAPTURE_HELPER_FAILURE
    assert not outside.exists()
    assert not stderr_path.exists()
    assert not manifest_path.exists()

    link_target = tmp_path / "link-target.bin"
    link_target.write_bytes(b"unchanged")
    link_path = artifact / "target.stdout.bin"
    os.symlink(link_target, link_path)
    link_attempt, _, _, _ = _run_captured(
        artifact,
        [sys.executable, str(HELPER), "echo-argument", "--value", "must not run"],
    )
    assert link_attempt.returncode == CAPTURE_HELPER_FAILURE
    assert link_target.read_bytes() == b"unchanged"


def test_read_capture_exit_rejects_manifest_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    completed, _, _, manifest_path = _run_captured(
        artifact,
        [sys.executable, str(HELPER), "exit-with-code", "--code", "7"],
    )
    assert completed.returncode == 0
    manifest = json.loads(manifest_path.read_bytes())
    manifest["target_exit_code"] = 0
    manifest_path.write_bytes(_canonical_manifest_bytes(manifest))

    read_exit = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "read-capture-exit",
            "--artifact-directory",
            str(artifact),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
    )

    assert read_exit.returncode == CAPTURE_HELPER_FAILURE
    assert read_exit.stdout == b""
    assert b"manifest SHA-256 mismatch" in read_exit.stderr

    classified = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "classify-smoke",
            "--project-root",
            str(Path.cwd()),
            "--capture-manifest",
            str(manifest_path),
            "--artifact-directory",
            str(artifact),
            "--output",
            str(artifact / "result.json"),
        ],
        check=False,
        capture_output=True,
    )
    assert classified.returncode == CAPTURE_HELPER_FAILURE
    assert not (artifact / "result.json").exists()


def test_run_captured_never_expands_shell_metacharacters(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (tmp_path / "would-expand.txt").write_text("fixture", encoding="ascii")
    literal = "*; $(touch should-not-exist)"

    completed, stdout_path, _, _ = _run_captured(
        artifact,
        [sys.executable, str(HELPER), "echo-argument", "--value", literal],
    )

    assert completed.returncode == 0
    assert stdout_path.read_bytes() == (literal + "\n").encode("utf-8")
    assert not (Path.cwd() / "should-not-exist").exists()


def test_frozen_inputs_reject_tampered_config_before_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    project, weight, registry_sha256 = _project(tmp_path)
    monkeypatch.setattr(api, "EXPECTED_REGISTRY_SHA256", registry_sha256)
    config = project / "configs/algorithm/training/rfdetr-single-batch-smoke.yaml"
    config.write_text("seed: 9\nbatch_size: 1\n", encoding="utf-8")
    called = False

    def callback() -> None:
        nonlocal called
        called = True

    with pytest.raises(api.HandoffContractError, match="config SHA-256 mismatch"):
        api.verify_frozen_inputs(
            project,
            weight,
            after_verification=callback,
        )

    assert called is False


def test_frozen_inputs_accept_crlf_checkout_of_frozen_text_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    project, weight, registry_sha256 = _project(tmp_path)
    config = project / "configs/algorithm/training/rfdetr-single-batch-smoke.yaml"
    registry = project / "configs/algorithm/windows/raytheon-rfdetr-smoke.json"
    monkeypatch.setattr(api, "EXPECTED_REGISTRY_SHA256", registry_sha256)
    config.write_bytes(config.read_bytes().replace(b"\n", b"\r\n"))
    registry.write_bytes(registry.read_bytes().replace(b"\n", b"\r\n"))

    verified = api.verify_frozen_inputs(project, weight)

    assert verified.config_sha256 == _normalized_text_sha256(config)


def test_frozen_inputs_reject_semantic_tamper_after_line_ending_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    project, weight, registry_sha256 = _project(tmp_path)
    config = project / "configs/algorithm/training/rfdetr-single-batch-smoke.yaml"
    monkeypatch.setattr(api, "EXPECTED_REGISTRY_SHA256", registry_sha256)
    config.write_text("seed: 20260802\r\nbatch_size: 1\r\n", encoding="utf-8")

    with pytest.raises(api.HandoffContractError, match="config SHA-256 mismatch"):
        api.verify_frozen_inputs(project, weight)


def test_explicit_oom_exit_and_payload_map_to_typed_result(tmp_path: Path) -> None:
    api = _api()
    summary = {
        "schema_version": "1.0",
        "status": "OOM",
        "exit_code": 21,
        "stage": "onnx_export",
        "seed": 20260801,
        "synthetic_only": True,
    }

    result = api.classify_smoke_result(
        native_exit_code=21,
        summary=summary,
        artifact_directory=tmp_path,
        expected_seed=20260801,
    )

    assert result["status"] == "OOM"
    assert result["exit_code"] == 21
    assert result["stage"] == "onnx_export"
    assert result["seed"] == 20260801


def test_oom_exit_without_explicit_oom_payload_fails_closed(tmp_path: Path) -> None:
    api = _api()

    result = api.classify_smoke_result(
        native_exit_code=21,
        summary={"stderr": "CUDA out of memory"},
        artifact_directory=tmp_path,
        expected_seed=20260801,
    )

    assert result["status"] == "OTHER_FAILURE"
    assert result["exit_code"] == 22


def test_pass_without_verified_cuda_runtime_evidence_fails_closed(tmp_path: Path) -> None:
    api = _api()

    result = api.classify_smoke_result(
        native_exit_code=0,
        summary={
            "status": "PASS",
            "exit_code": 0,
            "seed": 20260801,
            "synthetic_only": True,
            "runtime": {"accelerator": "cpu", "cuda_available": False, "amp": False},
        },
        artifact_directory=tmp_path,
        expected_seed=20260801,
    )

    assert result["status"] == "OTHER_FAILURE"
    assert result["exit_code"] == 22


def test_valid_pass_has_empty_success_detail(tmp_path: Path) -> None:
    api = _api()

    result = api.classify_smoke_result(
        native_exit_code=0,
        summary={
            "status": "PASS",
            "exit_code": 0,
            "seed": 20260801,
            "synthetic_only": True,
            "runtime": {"accelerator": "cuda", "cuda_available": True, "amp": True},
        },
        artifact_directory=tmp_path,
        expected_seed=20260801,
    )

    assert result == {
        "schema_version": "1.0",
        "status": "PASS",
        "exit_code": 0,
        "detail": None,
        "synthetic_only": True,
        "real_video_count": 0,
        "accuracy_claim": "none_synthetic_smoke_only",
        "seed": 20260801,
        "batch_size": 1,
        "model_class": "RFDETRSegSmall",
        "artifact_directory": str(tmp_path.resolve()),
    }


def test_unique_artifact_directories_never_reuse_existing_content(tmp_path: Path) -> None:
    api = _api()
    now = datetime(2026, 8, 2, 12, 34, 56, 123456, tzinfo=UTC)

    first = api.create_unique_artifact_directory(
        tmp_path,
        "smoke",
        now=now,
        token="a" * 32,
    )
    (first / "result.json").write_text('{"status":"PASS"}', encoding="utf-8")
    second = api.create_unique_artifact_directory(
        tmp_path,
        "smoke",
        now=now,
        token="b" * 32,
    )

    assert first != second
    assert list(second.iterdir()) == []
    with pytest.raises(FileExistsError):
        api.create_unique_artifact_directory(
            tmp_path,
            "smoke",
            now=now,
            token="a" * 32,
        )


def test_venv_identity_requires_requested_prefix_and_isolation(tmp_path: Path) -> None:
    api = _api()
    venv = tmp_path / "project" / ".venv-rfdetr"
    executable = venv / "Scripts" / "python.exe"
    probe = {
        "implementation": "cpython",
        "major": 3,
        "minor": 12,
        "pointer_bits": 64,
        "prefix": str(venv),
        "base_prefix": str(tmp_path / "Python312"),
        "executable": str(executable),
    }

    api.validate_venv_identity(venv, probe)

    probe["prefix"] = str(tmp_path / "different-venv")
    with pytest.raises(api.HandoffContractError, match="sys.prefix"):
        api.validate_venv_identity(venv, probe)


def test_bundle_contains_only_selected_current_run_evidence(tmp_path: Path) -> None:
    api = _api()
    artifacts = tmp_path / "artifacts"
    install = artifacts / "install-current"
    smoke = artifacts / "smoke-current"
    old = artifacts / "smoke-old"
    for directory in (install, smoke, old):
        directory.mkdir(parents=True)
    (install / "pip-check.txt").write_text("No broken requirements", encoding="utf-8")
    (smoke / "result.json").write_text('{"status":"OOM"}', encoding="utf-8")
    (smoke / "smoke.stderr.txt").write_text("typed failure", encoding="utf-8")
    (old / "result.json").write_text('{"status":"PASS"}', encoding="utf-8")

    bundle = api.create_evidence_bundle(
        artifacts,
        install,
        smoke,
        now=datetime(2026, 8, 2, tzinfo=UTC),
        token="c" * 32,
    )

    assert (bundle / "install" / "pip-check.txt").is_file()
    assert json.loads((bundle / "smoke" / "result.json").read_text())["status"] == "OOM"
    assert "PASS" not in "".join(
        path.read_text(encoding="utf-8") for path in bundle.rglob("*") if path.is_file()
    )


def test_uninstall_target_requires_project_marker_cfg_and_exact_venv_identity(
    tmp_path: Path,
) -> None:
    api = _api()
    project = tmp_path / "project"
    venv = project / ".venv-rfdetr"
    executable = venv / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    (project / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = C:\\Python312", encoding="utf-8")
    probe = {
        "implementation": "cpython",
        "major": 3,
        "minor": 12,
        "pointer_bits": 64,
        "prefix": str(venv),
        "base_prefix": str(tmp_path / "Python312"),
        "executable": str(executable),
    }

    api.validate_uninstall_target(project, venv, probe)

    (venv / "pyvenv.cfg").unlink()
    with pytest.raises(api.HandoffContractError, match="pyvenv.cfg"):
        api.validate_uninstall_target(project, venv, probe)


def test_cli_executes_unique_path_and_explicit_result_logic(tmp_path: Path, capsys) -> None:
    api = _api()
    artifacts = tmp_path / "artifacts"

    assert (
        api.main(
            [
                "create-artifact",
                "--project-root",
                str(Path.cwd()),
                "--root",
                str(artifacts),
                "--kind",
                "smoke",
            ]
        )
        == 0
    )
    created = Path(capsys.readouterr().out.strip())
    assert created.is_dir()
    summary = tmp_path / "summary.json"
    output = tmp_path / "result.json"
    summary.write_text(
        json.dumps(
            {
                "status": "OOM",
                "exit_code": 21,
                "stage": "training",
                "seed": 20260801,
                "synthetic_only": True,
            }
        ),
        encoding="utf-8",
    )

    assert (
        api.main(
            [
                "classify-smoke",
                "--project-root",
                str(Path.cwd()),
                "--native-exit-code",
                "21",
                "--summary",
                str(summary),
                "--artifact-directory",
                str(created),
                "--output",
                str(output),
            ]
        )
        == 21
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "OOM"


def test_cli_rejects_artifact_root_inside_project(tmp_path: Path) -> None:
    api = _api()
    project, _, _ = _project(tmp_path)
    artifacts = project / "artifacts"

    assert (
        api.main(
            [
                "create-artifact",
                "--project-root",
                str(project),
                "--root",
                str(artifacts),
                "--kind",
                "smoke",
            ]
        )
        == 2
    )
    assert not artifacts.exists()


@pytest.mark.parametrize("summary_state", ["empty", "malformed", "missing"])
def test_cli_native_failure_always_writes_other_failure_result(
    tmp_path: Path, summary_state: str
) -> None:
    api = _api()
    summary = tmp_path / "smoke-summary.json"
    stderr = tmp_path / "smoke.stderr.txt"
    output = tmp_path / "result.json"
    if summary_state == "empty":
        summary.write_text("", encoding="utf-8")
    elif summary_state == "malformed":
        summary.write_text("{not-json", encoding="utf-8")
    stderr.write_text("backend validation failed", encoding="utf-8")

    exit_code = api.main(
        [
            "classify-smoke",
            "--project-root",
            str(Path.cwd()),
            "--native-exit-code",
            "2",
            "--summary",
            str(summary),
            "--stderr",
            str(stderr),
            "--artifact-directory",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 22
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "OTHER_FAILURE"
    assert result["exit_code"] == 22
    assert "backend validation failed" in result["detail"]


def test_cli_native_failure_writes_result_even_when_registry_is_invalid(
    tmp_path: Path,
) -> None:
    api = _api()
    project, _, _ = _project(tmp_path)
    summary = tmp_path / "missing-summary.json"
    output = tmp_path / "result.json"

    exit_code = api.main(
        [
            "classify-smoke",
            "--project-root",
            str(project),
            "--native-exit-code",
            "2",
            "--summary",
            str(summary),
            "--artifact-directory",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 22
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "OTHER_FAILURE"
    assert "registry" in result["detail"]
