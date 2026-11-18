from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _artifact(root: Path) -> Path:
    root.mkdir()
    media = root / "presentation.mp4"
    writer = cv2.VideoWriter(
        str(media),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 48),
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    timeline = root / "timeline.json"
    timeline.write_bytes(
        _canonical(
            {
                "schema_version": "1.0",
                "frames": [
                    {
                        "frame_id": 0,
                        "timestamp_ms": 0,
                        "phase": "stable",
                        "guidance_visible": True,
                    }
                ],
            }
        )
    )
    inventory = {}
    for path in (media, timeline):
        content = path.read_bytes()
        inventory[path.name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (root / "manifest.json").write_bytes(
        _canonical(
            {
                "kind": "field_route_file_replay",
                "artifact_id": "shell-fixture",
                "inventory": inventory,
            }
        )
    )
    return root


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    return environment


def test_shell_cli_check_publishes_screen_and_secret_tablet_urls(tmp_path: Path) -> None:
    state_db = tmp_path / "state" / "field.sqlite3"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/serve_field_route_shell.py",
            "--artifact",
            str(_artifact(tmp_path / "artifact")),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--display-host",
            "10.0.0.20",
            "--control-token",
            "fixed-secret",
            "--state-db",
            str(state_db),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "READY"
    assert result["screen_url"].startswith("http://10.0.0.20:")
    assert result["screen_url"].endswith("/screen")
    assert result["tablet_url"].endswith("/tablet?token=fixed-secret")
    assert result["control_token"] == "fixed-secret"
    assert result["state_db"] == str(state_db.resolve())
    assert result["session_id"].startswith("session-")
    assert state_db.is_file()


def test_shell_cli_rejects_unverified_artifact(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "artifact")
    (artifact / "presentation.mp4").write_bytes(b"tampered")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/serve_field_route_shell.py",
            "--artifact",
            str(artifact),
            "--state-db",
            str(tmp_path / "field.sqlite3"),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "field route shell failed" in completed.stderr


def test_shell_cli_returns_usage_exit_for_invalid_port(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/serve_field_route_shell.py",
            "--artifact",
            str(_artifact(tmp_path / "artifact")),
            "--port",
            "70000",
            "--state-db",
            str(tmp_path / "field.sqlite3"),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 64
    assert "arguments invalid" in completed.stderr


def test_windows_shell_wrapper_is_policy_neutral_and_forwards_exit_code() -> None:
    wrapper = (REPOSITORY_ROOT / "scripts/algorithm/windows/serve-field-route-shell.ps1").read_text(
        encoding="utf-8"
    )

    assert "ExecutionPolicy" not in wrapper
    assert "Resolve-Path -LiteralPath" in wrapper
    assert '"--host", $HostAddress' in wrapper
    assert "--state-db" in wrapper
    assert "--join-qr-asset" in wrapper
    assert "--sync-base-url" in wrapper
    assert "--device-id" in wrapper
    assert "--credential-target" in wrapper
    assert "$PythonPath" in wrapper
    assert "CredentialSecret" not in wrapper
    assert "$env:LOCALAPPDATA" in wrapper
    assert "billiardsAI\\field-route" in wrapper
    assert "$LASTEXITCODE" in wrapper


def test_windows_shell_wrapper_prefers_the_deployed_release_source() -> None:
    wrapper = (REPOSITORY_ROOT / "scripts/algorithm/windows/serve-field-route-shell.ps1").read_text(
        encoding="utf-8"
    )

    source_assignment = '$SourceRoot = Join-Path $ResolvedProjectRoot "src"'
    python_path_assignment = "$env:PYTHONPATH = $SourceRoot"

    assert source_assignment in wrapper
    assert python_path_assignment in wrapper
    assert wrapper.index(source_assignment) < wrapper.index(python_path_assignment)
    invocation = "& $Python @PythonPrefixArguments $Runner @RunnerArguments"
    assert wrapper.index(python_path_assignment) < wrapper.index(invocation)
