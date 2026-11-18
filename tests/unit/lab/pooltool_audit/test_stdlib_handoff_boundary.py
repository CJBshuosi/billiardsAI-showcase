import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_handoff_imports_with_site_packages_disabled() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/algorithm/pooltool_packet_handoff.py"),
            "--packet",
            str(ROOT / "missing-packet.json"),
            "--packet-sha256",
            "a" * 64,
            "--wheelhouse",
            str(ROOT / "missing-wheelhouse"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "pooltool packet cannot be read" in result.stderr
    assert "pydantic" not in result.stderr
