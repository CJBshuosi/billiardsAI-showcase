from __future__ import annotations

import re
import subprocess
from pathlib import Path


def test_miniprogram_tree_contains_no_local_cloud_or_device_secrets() -> None:
    tracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "apps/miniprogram",
        ],
        text=True,
    ).splitlines()
    assert not any(path.endswith("env.local.js") for path in tracked)
    assert not any(path.endswith("project.private.config.json") for path in tracked)
    for filename in tracked:
        content = Path(filename).read_text(encoding="utf-8", errors="ignore")
        assert "REPLACE_WITH_REAL_SECRET" not in content
        assert "wx-openid-" not in content
        assert re.search(r"AKID[A-Za-z0-9]{32,}", content) is None


def test_documented_powershell_commands_are_single_physical_lines() -> None:
    for path in Path("docs/operations").glob("miniprogram-*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        in_powershell = False
        command_lines = 0
        for line in lines:
            if line.strip() == "```powershell":
                in_powershell = True
                command_lines = 0
            elif in_powershell and line.strip() == "```":
                assert command_lines == 1, f"PowerShell block must be one line: {path}"
                in_powershell = False
            elif in_powershell and line.strip():
                command_lines += 1


def test_miniprogram_deployment_is_mac_first_and_raytheon_sync_is_separate() -> None:
    deployment = Path("docs/operations/miniprogram-development-deployment.md").read_text(
        encoding="utf-8"
    )
    raytheon = Path("docs/operations/miniprogram-raytheon-sync.md").read_text(encoding="utf-8")

    assert "macOS" in deployment
    assert "apps/miniprogram" in deployment
    assert "C:\\Users" not in deployment
    assert "微信开发者工具" not in raytheon
    assert "SQLite outbox" in raytheon
