from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDED_CONTROLLED_PYTHON = "/Users/developer/Documents/VScodeProject/billiardsAI/.venv/bin/python"
ARTIFACT_ROOT = (
    PROJECT_ROOT / "_bmad-output" / "implementation-artifacts" / "feasibility" / "story-2.1"
)
EXPECTED_COMMITS = {
    "Interactive-Pool": "479508d7504a9b540165482cf477220c84c7cd65",
    "pool-telemetry-web": "83ec155add823bebb3a31795461aba57e9078a97",
    "CueDetat": "240a9d5f693f4712c670e280102495267ffeb6ef",
    "DEADPOOL_Project": "2e974916fb888f212e57d5baae3ec316d71825dd",
    "AISnooker": "4a2df91bbe446fb4a2021317d66997f7f0f6e97f",
}
ALLOWED_STATUSES = {
    "RUNNABLE",
    "REJECTED_LICENSE",
    "REPRODUCTION_FAILED",
    "NOT_APPLICABLE",
}
REQUIRED_ARTIFACTS = {
    "adapter-entrypoints.md",
    "candidate-register.json",
    "candidate-register.md",
    "license-review.md",
    "reproduce.sh",
    "reproduction-log.md",
    "reproduction-results.json",
}


def _load_json(name: str) -> dict:
    path = ARTIFACT_ROOT / name
    assert path.is_file(), f"missing Story 2.1 artifact: {path.relative_to(PROJECT_ROOT)}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_artifact_reference_exists(reference: str) -> None:
    relative_path = reference.split("#", maxsplit=1)[0]
    target = ARTIFACT_ROOT / relative_path
    assert target.is_file(), f"artifact reference does not resolve: {reference}"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_story_2_1_artifact_bundle_is_complete_and_contains_no_forbidden_payloads() -> None:
    assert ARTIFACT_ROOT.is_dir(), "Story 2.1 feasibility artifact directory is missing"
    files = {path.name for path in ARTIFACT_ROOT.iterdir() if path.is_file()}
    assert REQUIRED_ARTIFACTS <= files

    allowed_suffixes = {".json", ".md", ".sh"}
    forbidden_suffixes = {
        ".avi",
        ".env",
        ".key",
        ".mkv",
        ".mov",
        ".mp4",
        ".onnx",
        ".pem",
        ".pt",
        ".tflite",
        ".weights",
    }
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() in allowed_suffixes
        assert path.suffix.lower() not in forbidden_suffixes
        assert path.stat().st_size < 1_000_000

    combined_text = "\n".join(
        path.read_text(encoding="utf-8") for path in ARTIFACT_ROOT.rglob("*") if path.is_file()
    )
    forbidden_secret_patterns = (
        r"AIza[0-9A-Za-z_-]{20,}",
        r"sk-ant-[0-9A-Za-z_-]{20,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    for pattern in forbidden_secret_patterns:
        assert re.search(pattern, combined_text) is None


def test_candidate_register_records_fixed_legal_and_reproduction_evidence() -> None:
    register = _load_json("candidate-register.json")
    assert register["schema_version"] == "1.0"
    assert register["story_id"] == "2.1"
    assert register["product_dependency_selected"] is False
    assert register["decision"] == "DEFERRED_TO_STORY_2_2"
    assert register["timebox"]["limit_developer_days"] == 2
    assert register["real_video_used"] is False
    assert register["patches_applied"] is False
    assert register["adapter_contract"] == {
        "status": "DRAFT_ONLY",
        "ref": "adapter-entrypoints.md",
        "imports_upstream": False,
        "wraps_upstream": False,
        "verified_capabilities": [],
    }

    candidates = {candidate["name"]: candidate for candidate in register["candidates"]}
    assert set(candidates) == set(EXPECTED_COMMITS)
    required_fields = {
        "name",
        "repository_url",
        "commit",
        "checkout",
        "maintenance",
        "license",
        "runtime_environment",
        "pretrained_models",
        "camera_geometry",
        "supported_outputs",
        "tests_or_demo",
        "known_limitations",
        "status",
        "product_dependency",
        "command_refs",
        "log_summary_ref",
        "result_ref",
        "adapter_ref",
    }
    for name, expected_commit in EXPECTED_COMMITS.items():
        candidate = candidates[name]
        assert required_fields <= set(candidate)
        assert candidate["repository_url"].startswith("https://github.com/")
        assert re.fullmatch(r"[0-9a-f]{40}", candidate["commit"])
        assert candidate["commit"] == expected_commit
        assert candidate["checkout"]["detached"] is True
        assert candidate["checkout"]["patches_applied"] is False
        assert candidate["maintenance"]["last_commit_at"]
        assert candidate["license"]["status"] in {"MIT_CONFIRMED", "NO_LICENSE_FOUND"}
        assert candidate["status"] in ALLOWED_STATUSES
        assert candidate["product_dependency"] is False
        assert candidate["command_refs"]
        for reference in candidate["command_refs"]:
            _assert_artifact_reference_exists(reference)
        _assert_artifact_reference_exists(candidate["log_summary_ref"])
        _assert_artifact_reference_exists(candidate["result_ref"])
        if candidate["status"] == "RUNNABLE":
            assert candidate["adapter_ref"]
            _assert_artifact_reference_exists(candidate["adapter_ref"])
        else:
            assert candidate["adapter_ref"] is None

    for name in ("Interactive-Pool", "pool-telemetry-web"):
        runtime = candidates[name]["runtime_environment"]
        assert runtime["python_bin"] == RECORDED_CONTROLLED_PYTHON
        assert f"PYTHON_BIN={RECORDED_CONTROLLED_PYTHON}" in runtime["script_rerun_command"]

    assert candidates["DEADPOOL_Project"]["status"] == "REJECTED_LICENSE"
    assert candidates["AISnooker"]["status"] == "REJECTED_LICENSE"
    assert candidates["DEADPOOL_Project"]["license"]["status"] == "NO_LICENSE_FOUND"
    assert candidates["AISnooker"]["license"]["status"] == "NO_LICENSE_FOUND"
    assessed = [
        candidate for candidate in candidates.values() if candidate["status"] != "REJECTED_LICENSE"
    ]
    assert 2 <= len(assessed) <= 4


def test_reproduction_results_preserve_commands_environment_exit_codes_and_failures() -> None:
    results_document = _load_json("reproduction-results.json")
    assert results_document["schema_version"] == "1.0"
    assert results_document["story_id"] == "2.1"
    assert results_document["real_video_used"] is False
    assert results_document["models_downloaded"] is False
    assert results_document["network_services_started"] is False
    assert results_document["environment"]["platform"]
    assert results_document["environment"]["architecture"]
    assert results_document["environment"]["git_version"]
    assert results_document["environment"]["python_versions"]
    assert results_document["environment"]["tool_versions"]
    assert results_document["environment"]["controlled_python"] == {
        "path": RECORDED_CONTROLLED_PYTHON,
        "version": "3.12.13",
        "script_default": "$REPOSITORY_ROOT/.venv/bin/python",
        "mismatch_exit_code": 69,
    }

    results = {result["name"]: result for result in results_document["results"]}
    assert set(results) == set(EXPECTED_COMMITS)
    for name, expected_commit in EXPECTED_COMMITS.items():
        result = results[name]
        assert result["commit"] == expected_commit
        assert result["status"] in ALLOWED_STATUSES
        assert result["checked_at"].endswith("Z")
        assert result["commands"]
        assert result["summary"]
        assert result["limitations"]
        for command in result["commands"]:
            assert command["id"]
            assert command["command"]
            assert isinstance(command["executed"], bool)
            assert command["log_summary"]
            if command["executed"]:
                assert isinstance(command["exit_code"], int)
            else:
                assert command["exit_code"] is None
                assert command["not_run_reason"]
        if result["status"] == "RUNNABLE":
            assert any(
                command["executed"] and command["exit_code"] == 0 for command in result["commands"]
            )
            assert result["rerun_command"]
            assert result["adapter_ref"] == "adapter-entrypoints.md"
        else:
            assert result["failure_evidence"]
            assert result["adapter_ref"] is None

    for name in ("Interactive-Pool", "pool-telemetry-web"):
        assert f"PYTHON_BIN={RECORDED_CONTROLLED_PYTHON}" in results[name]["rerun_command"]


def test_reproduction_script_rejects_a_workdir_inside_the_repository(tmp_path: Path) -> None:
    script = ARTIFACT_ROOT / "reproduce.sh"
    assert script.is_file(), "reproduce.sh is missing"

    help_result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "FEASIBILITY_WORKDIR" in help_result.stdout

    rejected = subprocess.run(
        ["bash", str(script), "--validate-workdir"],
        cwd=PROJECT_ROOT,
        env={"FEASIBILITY_WORKDIR": str(ARTIFACT_ROOT / "forbidden-clone")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 64
    assert "outside the repository" in rejected.stderr
    assert not (ARTIFACT_ROOT / "forbidden-clone").exists()

    accepted = subprocess.run(
        ["bash", str(script), "--validate-workdir"],
        cwd=PROJECT_ROOT,
        env={"FEASIBILITY_WORKDIR": str(tmp_path / "external-checkouts")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0
    assert not (tmp_path / "external-checkouts").exists()


def test_reproduction_script_uses_configured_python_instead_of_path_python3(
    tmp_path: Path,
) -> None:
    script = ARTIFACT_ROOT / "reproduce.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    controlled_marker = tmp_path / "controlled-python-invoked"
    system_marker = tmp_path / "system-python-invoked"
    controlled_python = fake_bin / "controlled-python"

    _write_executable(
        controlled_python,
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ] && '
        "printf '%s' \"${2:-}\" | grep -q 'platform.python_version'; then\n"
        "  printf '3.12.13\\n'\n"
        "  exit 0\n"
        "fi\n"
        f"printf 'invoked\\n' > '{controlled_marker}'\n"
        "exit 23\n",
    )
    _write_executable(
        fake_bin / "python3",
        f"#!/bin/sh\nprintf 'invoked\\n' > '{system_marker}'\nexit 97\n",
    )
    _write_executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        'if [ "${1:-}" = "clone" ]; then\n'
        '  mkdir -p "$4"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "${1:-}" = "-C" ]; then\n'
        '  case "${3:-}" in\n'
        "    checkout) exit 0 ;;\n"
        "    rev-parse) printf '479508d7504a9b540165482cf477220c84c7cd65\\n'; exit 0 ;;\n"
        "    status) exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "exit 91\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "FEASIBILITY_WORKDIR": str(tmp_path / "external-checkouts"),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PYTHON_BIN": str(controlled_python),
        }
    )
    result = subprocess.run(
        ["bash", str(script), "--candidate", "interactive-pool"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    assert controlled_marker.is_file()
    assert not system_marker.exists()


def test_reproduction_script_default_never_falls_back_to_system_python3(
    tmp_path: Path,
) -> None:
    script = ARTIFACT_ROOT / "reproduce.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    system_marker = tmp_path / "system-python-invoked"
    _write_executable(
        fake_bin / "python3",
        f"#!/bin/sh\nprintf 'invoked\\n' > '{system_marker}'\nexit 97\n",
    )
    env = os.environ.copy()
    env["FEASIBILITY_WORKDIR"] = str(tmp_path / "external-checkouts")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("PYTHON_BIN", None)

    result = subprocess.run(
        ["bash", str(script), "--validate-python"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode in {0, 69}
    assert not system_marker.exists()
    if result.returncode == 0:
        assert ".venv/bin/python" in result.stdout
        assert "Python 3.12.13" in result.stdout
    else:
        assert "controlled Python unavailable" in result.stderr

    explicit_env = env | {"PYTHON_BIN": sys.executable}
    explicit = subprocess.run(
        ["bash", str(script), "--validate-python"],
        cwd=PROJECT_ROOT,
        env=explicit_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert explicit.returncode == 0
    assert f"PYTHON_BIN={sys.executable}" in explicit.stdout
    assert "Python 3.12.13" in explicit.stdout
