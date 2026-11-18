from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from billiards_ai.deployment.windows_acl import grant_artifact_read_execute


def test_grant_artifact_read_execute_targets_only_the_requested_identity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    observed: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    grant_artifact_read_execute(
        artifact,
        identity=r"BOYU\oor20",
        runner=runner,
    )

    assert observed == [
        [
            "icacls.exe",
            str(artifact.resolve()),
            "/grant:r",
            r"BOYU\oor20:(OI)(CI)RX",
            "/T",
            "/Q",
        ]
    ]


def test_grant_artifact_read_execute_fails_closed_when_icacls_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 5, stdout="", stderr="access denied")

    with pytest.raises(RuntimeError, match="access denied"):
        grant_artifact_read_execute(
            artifact,
            identity=r"BOYU\oor20",
            runner=runner,
        )
