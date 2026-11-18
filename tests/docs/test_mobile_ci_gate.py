"""Executable repository contract for the mobile pull-request gate."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "mobile-pr-gate.yml"


def test_mobile_ci_dependency_resolution_fails_closed_on_lockfile_drift() -> None:
    """CI must enforce and recheck the committed mobile dependency graph."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["mobile"]["steps"]
    mobile_commands = [
        step.get("run", "") for step in steps if step.get("working-directory") == "apps/mobile"
    ]

    enforced_get = "flutter pub get --enforce-lockfile"
    lockfile_diff = "git diff --exit-code -- pubspec.lock"
    fail_closed_commands = [
        command
        for command in mobile_commands
        if enforced_get in command and lockfile_diff in command
    ]

    assert fail_closed_commands, (
        "mobile CI must enforce and recheck the committed lockfile in one step"
    )
    assert fail_closed_commands[0].index(enforced_get) < fail_closed_commands[0].index(
        lockfile_diff
    ), "mobile CI must check for lockfile drift after dependency resolution"


def test_mobile_ci_builds_android_then_ios_after_tests_for_prs_to_dev() -> None:
    """A2.1 acceptance requires both platform build gates after mobile tests."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["pull_request"]["branches"] == ["dev"]
    assert "apps/mobile/**" in triggers["pull_request"]["paths"]

    steps = workflow["jobs"]["mobile"]["steps"]
    commands = [step.get("run", "") for step in steps]
    test_index = next(index for index, command in enumerate(commands) if command == "flutter test")
    apk_index = next(
        index for index, command in enumerate(commands) if command == "flutter build apk --debug"
    )
    ios_index = next(
        index
        for index, command in enumerate(commands)
        if command == "flutter build ios --simulator --debug"
    )

    assert test_index < apk_index < ios_index
    assert all("--no-codesign" not in command for command in commands)
