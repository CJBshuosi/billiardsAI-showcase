from __future__ import annotations

import copy
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPO = "CJBshuosi/billiardsAI"
CORE = ("test", "lint", "security", "audit", "candidate")


def test_cli_help_works_without_product_imports(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/ci/final_candidate_gate.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "check-pr" in result.stdout


@pytest.fixture
def gate():
    assert (ROOT / "scripts/ci/final_candidate_gate.py").is_file(), "candidate gate is missing"
    return importlib.import_module("scripts.ci.final_candidate_gate")


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def history(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True)
    git(work, "config", "user.name", "candidate-test")
    git(work, "config", "user.email", "candidate@example.test")
    git(work, "config", "commit.gpgsign", "false")
    git(work, "switch", "-c", "dev")
    (work / "value").write_text("base")
    git(work, "add", "value")
    git(work, "commit", "-qm", "base")
    base = git(work, "rev-parse", "HEAD")
    git(work, "push", "-q", "origin", "HEAD:dev")
    git(work, "switch", "-c", "codex/feature")
    (work / "value").write_text("validated feature")
    git(work, "commit", "-qam", "feature")
    head = git(work, "rev-parse", "HEAD")
    return work, remote, base, head


def pr(base, head, *, merged=False, merge_sha=None):
    return {
        "number": 22,
        "state": "closed" if merged else "open",
        "merged": merged,
        "draft": False,
        "merge_commit_sha": merge_sha,
        "head": {"sha": head, "repo": {"full_name": REPO}},
        "base": {"sha": base, "ref": "dev", "repo": {"full_name": REPO}},
    }


def run(head, workflow="pr-gate.yml", *, run_id=123):
    return {
        "id": run_id,
        "run_attempt": 1,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_sha": head,
        "path": ".github/workflows/" + workflow,
        "repository": {"full_name": REPO},
        "head_repository": {"full_name": REPO},
    }


def jobs(names=CORE):
    return [{"name": n, "status": "completed", "conclusion": "success"} for n in names]


def evidence(head, paths, target):
    return {"pr-gate.yml": 123}


@pytest.mark.parametrize(
    "field,value",
    [
        ("conclusion", "failure"),
        ("status", "in_progress"),
        ("event", "push"),
        ("head_sha", "0" * 40),
        ("path", ".github/workflows/other.yml"),
    ],
)
def test_failed_or_foreign_run_is_never_reused(gate, field, value):
    data = run("a" * 40)
    data[field] = value
    with pytest.raises(gate.GateError):
        gate.validate_run(data, jobs(), "a" * 40, "pr-gate.yml", CORE)


@pytest.mark.parametrize("conclusion", ["skipped", "cancelled", "failure", None])
def test_every_required_job_must_really_succeed(gate, conclusion):
    data = jobs()
    data[0]["conclusion"] = conclusion
    with pytest.raises(gate.GateError):
        gate.validate_run(run("a" * 40), data, "a" * 40, "pr-gate.yml", CORE)


def test_missing_duplicate_or_foreign_repository_proof_fails(gate):
    for data in (jobs()[:-1], jobs() + [jobs()[0]]):
        with pytest.raises(gate.GateError):
            gate.validate_run(run("a" * 40), data, "a" * 40, "pr-gate.yml", CORE)
    data = run("a" * 40)
    data["head_repository"]["full_name"] = "other/fork"
    with pytest.raises(gate.GateError):
        gate.validate_run(data, jobs(), "a" * 40, "pr-gate.yml", CORE)


@pytest.mark.parametrize(
    "paths,target,want",
    [
        ([], "dev", {"pr-gate.yml"}),
        (
            ["apps/miniprogram/pages/clips/index.js"],
            "dev",
            {"pr-gate.yml", "miniprogram-pr-gate.yml"},
        ),
        (["apps/mobile/lib/main.dart"], "dev", {"pr-gate.yml", "mobile-pr-gate.yml"}),
        (["apps/mobile/lib/main.dart"], "main", {"pr-gate.yml"}),
        (
            [".github/workflows/miniprogram-pr-gate.yml"],
            "dev",
            {"pr-gate.yml", "miniprogram-pr-gate.yml"},
        ),
    ],
)
def test_required_checks_follow_actual_client_workflow_scope(gate, paths, target, want):
    assert set(gate.required_workflows(paths, target)) == want


def test_dry_run_validates_real_tree_without_publishing(gate, history):
    work, remote, base, head = history
    receipt = gate.verify_candidate(gate.Repository(work), pr(base, head), head, base, evidence)
    assert receipt["head"] == head
    assert receipt["tree"] == git(work, "rev-parse", head + "^{tree}")
    assert git(remote, "rev-parse", "dev") == base


@pytest.mark.parametrize("mutation", ["head", "base", "draft", "fork", "dirty"])
def test_changed_candidate_is_rejected_before_remote_write(gate, history, mutation):
    work, remote, base, head = history
    data = pr(base, head)
    if mutation == "head":
        data["head"]["sha"] = base
    elif mutation == "base":
        data["base"]["sha"] = "0" * 40
    elif mutation == "draft":
        data["draft"] = True
    elif mutation == "fork":
        data["head"]["repo"]["full_name"] = "other/fork"
    else:
        (work / "untracked").write_text("not committed")
    with pytest.raises(gate.GateError):
        gate.verify_candidate(gate.Repository(work), data, head, base, evidence)
    assert git(remote, "rev-parse", "dev") == base


def test_successful_publication_is_exact_fast_forward(gate, history):
    work, remote, base, head = history
    repo = gate.Repository(work)
    receipt = gate.verify_candidate(repo, pr(base, head), head, base, evidence)
    gate.publish_candidate(repo, receipt)
    assert git(remote, "rev-parse", "dev") == head
    assert git(remote, "show", "dev:value") == "validated feature"
    assert git(remote, "merge-base", "--is-ancestor", base, head) == ""


def test_concurrent_advance_cannot_be_merged_into_old_green(gate, history):
    work, remote, base, head = history
    repo = gate.Repository(work)
    receipt = gate.verify_candidate(repo, pr(base, head), head, base, evidence)
    git(work, "switch", "-c", "concurrent", base)
    (work / "other").write_text("other contribution")
    git(work, "add", "other")
    git(work, "commit", "-qm", "concurrent")
    other = git(work, "rev-parse", "HEAD")
    git(work, "push", "-q", "origin", "HEAD:dev")
    git(work, "switch", "codex/feature")
    with pytest.raises(gate.GateError):
        gate.publish_candidate(repo, receipt)
    assert git(remote, "rev-parse", "dev") == other
    assert git(remote, "show", "dev:other") == "other contribution"


def test_server_race_after_precheck_is_rejected_by_normal_git(gate, history, monkeypatch):
    work, remote, base, head = history
    repo = gate.Repository(work)
    receipt = gate.verify_candidate(repo, pr(base, head), head, base, evidence)
    git(work, "switch", "-c", "racing", base)
    (work / "other").write_text("racing contribution")
    git(work, "add", "other")
    git(work, "commit", "-qm", "racing")
    other = git(work, "rev-parse", "HEAD")
    git(work, "switch", "codex/feature")
    original = repo.git

    def race(*args):
        if args and args[0] == "push":
            git(work, "push", "-q", "origin", other + ":dev")
        return original(*args)

    monkeypatch.setattr(repo, "git", race)
    with pytest.raises(gate.GateError):
        gate.publish_candidate(repo, receipt)
    assert git(remote, "rev-parse", "dev") == other


def test_pr_checkout_requires_successful_needs_and_exact_head_tree(gate, history):
    work, _, base, head = history
    repo = gate.Repository(work)
    needs = {n: {"result": "success"} for n in CORE[:-1]}
    assert gate.validate_checkout(repo, pr(base, head), head, needs)["tree"] == git(
        work, "rev-parse", head + "^{tree}"
    )
    bad = copy.deepcopy(needs)
    bad["test"]["result"] = "skipped"
    with pytest.raises(gate.GateError):
        gate.validate_checkout(repo, pr(base, head), head, bad)
    with pytest.raises(gate.GateError):
        gate.validate_checkout(repo, pr(base, head), base, needs)


def test_push_reuses_successful_proof_even_for_different_merge_sha(gate, history):
    work, _, base, head = history
    git(work, "switch", "dev")
    git(work, "merge", "--no-ff", "-qm", "reviewed merge", head)
    merged = git(work, "rev-parse", "HEAD")
    result = gate.verify_publication(
        gate.Repository(work),
        pr(base, head, merged=True, merge_sha=merged),
        base,
        merged,
        "dev",
        evidence,
    )
    assert result["mode"] == "reused_premerge"
    assert result["head"] == head
    assert result["published_commit"] == merged
    assert result["full_rerun"] is False


def test_unverified_push_or_different_tree_is_not_green(gate, history):
    work, _, base, head = history
    data = pr(base, head, merged=True, merge_sha=head)
    with pytest.raises(gate.GateError):
        gate.verify_publication(gate.Repository(work), pr(base, head), base, head, "dev", evidence)
    (work / "value").write_text("unverified change")
    git(work, "commit", "-qam", "unverified")
    pushed = git(work, "rev-parse", "HEAD")
    data["merge_commit_sha"] = pushed
    with pytest.raises(gate.GateError):
        gate.verify_publication(gate.Repository(work), data, base, pushed, "dev", evidence)


def test_cli_is_read_only_by_default_and_has_no_force_switch(gate):
    arguments = gate.parser().parse_args(
        ["merge", "--pr", "22", "--expected-head", "a" * 40, "--expected-base", "b" * 40]
    )
    assert arguments.execute is False
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/final_candidate_gate.py"), "merge", "--force"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--force" not in gate.parser().format_help()


def test_latest_failed_attempt_cannot_borrow_older_green(gate):
    head = "a" * 40
    old = run(head, run_id=100)
    latest = run(head, run_id=101)
    latest["conclusion"] = "failure"

    class API:
        def get(self, path):
            if "/runs?" in path:
                return {"workflow_runs": [old, latest]}
            return {"jobs": jobs(), "total_count": 5}

    with pytest.raises(gate.GateError):
        gate.Checks(API()).verify(head, [], "dev")


def test_missing_api_evidence_never_produces_reused_result(gate):
    class API:
        def get(self, path):
            return {"workflow_runs": []}

    with pytest.raises(gate.GateError):
        gate.Checks(API()).verify("a" * 40, [], "dev")


def test_merge_cli_ignores_only_harmless_pr_metadata_updates(gate, history, monkeypatch):
    work, remote, base, head = history
    repo = gate.Repository(work)
    monkeypatch.setattr(repo, "check_origin", lambda: None)
    monkeypatch.setattr(gate, "Repository", lambda root: repo)
    initial = pr(base, head)
    current = initial | {"title": "clarified description", "updated_at": "later"}
    responses = iter([initial, current])

    class API:
        def pr(self, number):
            return next(responses)

    monkeypatch.setattr(gate, "API", API)
    monkeypatch.setattr(gate.Checks, "verify", lambda self, h, p, t: evidence(h, p, t))
    assert gate.main(["merge", "--pr", "22", "--expected-head", head, "--expected-base", base]) == 0
    assert git(remote, "rev-parse", "dev") == base


@pytest.mark.parametrize("change", ["head", "base", "closed", "target"])
def test_current_pr_must_still_identify_open_validated_candidate(gate, history, change):
    _, _, base, head = history
    current = pr(base, head)
    if change in ("base", "head"):
        current[change]["sha"] = "0" * 40
    elif change == "closed":
        current["state"] = "closed"
    else:
        current["base"]["ref"] = "main"
    with pytest.raises(gate.GateError):
        gate.validate_current_pr(current, {"base": base, "head": head, "target": "dev"})


def test_origin_push_destination_must_also_be_trusted(gate, history):
    work, remote, _, _ = history
    git(work, "remote", "set-url", "origin", "https://github.com/" + REPO + ".git")
    repo = gate.Repository(work)
    repo.check_origin()
    git(work, "config", "remote.origin.pushurl", str(remote))
    with pytest.raises(gate.GateError):
        repo.check_origin()


def test_missing_client_gate_blocks_even_with_successful_core(gate):
    class API:
        def get(self, path):
            if "miniprogram-pr-gate.yml/runs?" in path:
                return {"workflow_runs": []}
            if "/runs?" in path:
                return {"workflow_runs": [run("a" * 40)]}
            return {"jobs": jobs(), "total_count": 5}

    with pytest.raises(gate.GateError):
        gate.Checks(API()).verify("a" * 40, ["apps/miniprogram/page.js"], "dev")


def test_cancelled_client_gate_is_not_replaced_by_core_green(gate):
    class API:
        def get(self, path):
            if "miniprogram-pr-gate.yml/runs?" in path:
                client = run("a" * 40, "miniprogram-pr-gate.yml", run_id=456)
                client["conclusion"] = "cancelled"
                return {"workflow_runs": [client]}
            if "/runs?" in path:
                return {"workflow_runs": [run("a" * 40)]}
            names = ("miniprogram",) if "/456/" in path else CORE
            return {"jobs": jobs(names), "total_count": len(names)}

    with pytest.raises(gate.GateError):
        gate.Checks(API()).verify("a" * 40, ["apps/miniprogram/page.js"], "dev")


@pytest.mark.parametrize("replacement", ["rerun", "new_run"])
def test_workflow_changed_during_evidence_reads_is_rejected(gate, replacement):
    original = run("a" * 40)
    newer = copy.deepcopy(original)
    if replacement == "rerun":
        newer.update(run_attempt=2, status="queued", conclusion=None)
    else:
        newer.update(id=124, conclusion="failure")

    class API:
        inventories = 0

        def get(self, path):
            if "/runs?" in path:
                self.inventories += 1
                return {"workflow_runs": [original if self.inventories == 1 else newer]}
            return {"jobs": jobs(), "total_count": 5}

    with pytest.raises(gate.GateError):
        gate.Checks(API()).verify("a" * 40, [], "dev")


def test_receipt_identifies_run_attempt_and_uses_immutable_job_endpoint(gate):
    requests = []

    class API:
        def get(self, path):
            requests.append(path)
            if "/runs?" in path:
                return {"workflow_runs": [run("a" * 40)]}
            return {"jobs": jobs(), "total_count": 5}

    result = gate.Checks(API()).verify("a" * 40, [], "dev")
    assert result == {"pr-gate.yml": {"run_id": 123, "run_attempt": 1}}
    assert "actions/runs/123/attempts/1/jobs?per_page=100" in requests


def test_contained_concurrent_advance_still_publishes_only_validated_tree(
    gate, history, monkeypatch
):
    work, remote, base, middle = history
    (work / "extra").write_text("also validated")
    git(work, "add", "extra")
    git(work, "commit", "-qm", "final validated candidate")
    head = git(work, "rev-parse", "HEAD")
    repo = gate.Repository(work)
    receipt = gate.verify_candidate(repo, pr(base, head), head, base, evidence)
    original = repo.git

    def race(*args):
        if args and args[0] == "push":
            git(work, "push", "-q", "origin", middle + ":dev")
        return original(*args)

    monkeypatch.setattr(repo, "git", race)
    result = gate.publish_candidate(repo, receipt)
    assert result["published"] is True
    assert git(remote, "rev-parse", "dev") == head
    assert git(remote, "rev-parse", "dev^{tree}") == receipt["tree"]
