import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parents[2] / "fixtures" / "guidance" / "table-state.json"
SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_mainline1_guidance.py"


def test_cli_runs_synthetic_geometry_only_fixture(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--table-state",
            str(FIXTURE),
            "--context-group",
            "solid",
            "--physics",
            "off",
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "PASS"
    assert summary["recommendation_status"] == "geometry_only"
    assert summary["candidate_count"] == 3
    assert summary["physics_scored_count"] == 0
    assert summary["backend_versions"] == []
    artifact = Path(summary["output"])
    assert (artifact / "recommendation.json").is_file()
    assert (artifact / "mobile-scene.json").is_file()


def test_cli_rejects_mutually_exclusive_context_and_intent(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--table-state",
            str(FIXTURE),
            "--context-group",
            "solid",
            "--target-track-id",
            "solid",
            "--pocket",
            "top_right",
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
