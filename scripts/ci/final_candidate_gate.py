"""Verify a PR's final tree; publish only that commit by ordinary fast-forward.

This is a software-evidence guard, not a substitute for independent review,
product/platform acceptance or explicit release/deployment authorization.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "CJBshuosi/billiardsAI"
CORE = ("test", "lint", "security", "audit", "candidate")
TARGETS = {"dev", "main"}


class GateError(ValueError):
    """Evidence is missing, stale, failed or inconsistent; do not publish."""


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), "invalid SHA")
    return value


def trusted(value):
    return (
        isinstance(value, dict) and value.get("full_name", "").casefold() == REPOSITORY.casefold()
    )


class Repository:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def git(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, text=True, timeout=45
        )
        if result.returncode:
            raise GateError(f"git {args[0]} rejected; inspect state before retrying")
        return result.stdout.strip()

    def check_origin(self):
        urls = (
            self.git("remote", "get-url", "--all", "origin"),
            self.git("remote", "get-url", "--push", "--all", "origin"),
        )
        require(
            all(
                url
                in {
                    "https://github.com/" + REPOSITORY,
                    "https://github.com/" + REPOSITORY + ".git",
                    "git@github.com:" + REPOSITORY + ".git",
                }
                for url in urls
            ),
            "origin fetch and push must both identify only the trusted project repository",
        )

    def ancestor(self, older, newer):
        result = subprocess.run(
            ["git", "-C", str(self.root), "merge-base", "--is-ancestor", sha(older), sha(newer)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        require(result.returncode in (0, 1), "cannot establish commit ancestry")
        return result.returncode == 0

    def tree(self, commit):
        return sha(self.git("rev-parse", sha(commit) + "^{tree}"))

    def remote_head(self, target):
        require(target in TARGETS, "unsupported target")
        rows = self.git("ls-remote", "--exit-code", "origin", "refs/heads/" + target).splitlines()
        require(len(rows) == 1, "target branch identity missing")
        return sha(rows[0].split()[0])

    def changed(self, base, head):
        return self.git("diff", "--name-only", sha(base), sha(head)).splitlines()


class API:
    def get(self, path):
        result = subprocess.run(
            ["gh", "api", "repos/" + REPOSITORY + "/" + path],
            capture_output=True,
            text=True,
            timeout=45,
        )
        require(result.returncode == 0, "GitHub evidence read failed")
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise GateError("GitHub evidence is not JSON") from exc

    def pr(self, number):
        require(type(number) is int and number > 0, "invalid PR number")
        return self.get(f"pulls/{number}")


def required_workflows(paths, target):
    require(target in TARGETS, "unsupported target")
    result = {"pr-gate.yml": CORE}
    if any(
        fnmatch.fnmatchcase(p, "apps/miniprogram/**")
        or p == ".github/workflows/miniprogram-pr-gate.yml"
        for p in paths
    ):
        result["miniprogram-pr-gate.yml"] = ("miniprogram",)
    mobile = (
        "apps/mobile/**",
        "scripts/bootstrap_flutter.sh",
        "tests/docs/test_mobile_toolchain_pin.py",
        ".github/workflows/mobile-pr-gate.yml",
    )
    if target == "dev" and any(
        fnmatch.fnmatchcase(p, pattern) for p in paths for pattern in mobile
    ):
        result["mobile-pr-gate.yml"] = ("mobile",)
    return result


def validate_run(run, jobs, head, workflow, required):
    require(isinstance(run, dict), "invalid workflow evidence")
    require(
        run.get("event") == "pull_request"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and run.get("head_sha") == sha(head)
        and run.get("path") == ".github/workflows/" + workflow
        and trusted(run.get("repository"))
        and trusted(run.get("head_repository")),
        "workflow is not successful trusted PR evidence for this head",
    )
    require(isinstance(jobs, list), "invalid job evidence")
    for name in required:
        matches = [j for j in jobs if isinstance(j, dict) and j.get("name") == name]
        require(
            len(matches) == 1
            and matches[0].get("status") == "completed"
            and matches[0].get("conclusion") == "success",
            f"required job {name} did not actually succeed",
        )
    require(type(run.get("id")) is int and run["id"] > 0, "invalid workflow run id")
    require(type(run.get("run_attempt")) is int and run["run_attempt"] > 0, "invalid run attempt")
    return {"run_id": run["id"], "run_attempt": run["run_attempt"]}


class Checks:
    def __init__(self, api, *, wait_seconds=0):
        self.api = api
        self.wait_seconds = wait_seconds

    def latest(self, head, workflow):
        response = self.api.get(
            f"actions/workflows/{workflow}/runs?event=pull_request&head_sha={sha(head)}&per_page=100"
        )
        runs = response.get("workflow_runs", [])
        require(isinstance(runs, list) and runs, f"missing {workflow} evidence")
        require(
            all(isinstance(r, dict) and type(r.get("id")) is int for r in runs),
            "invalid run inventory",
        )
        return max(runs, key=lambda r: r["id"])

    def _verify(self, head, paths, target, *, include_core):
        proofs = {}
        validated = {}
        deadline = time.monotonic() + self.wait_seconds
        for workflow, names in required_workflows(paths, target).items():
            if not include_core and workflow == "pr-gate.yml":
                continue
            while True:
                latest = self.latest(head, workflow)
                if latest.get("status") not in ("queued", "in_progress", "waiting", "pending"):
                    break
                require(time.monotonic() < deadline, f"{workflow} has not completed")
                time.sleep(5)
            attempt = latest.get("run_attempt")
            require(type(attempt) is int and attempt > 0, "invalid run attempt")
            data = self.api.get(f"actions/runs/{latest['id']}/attempts/{attempt}/jobs?per_page=100")
            jobs = data.get("jobs", [])
            require(data.get("total_count") == len(jobs), "incomplete job inventory")
            proofs[workflow] = validate_run(latest, jobs, head, workflow, names)
            validated[workflow] = (jobs, names)
        # A rerun keeps its run ID, so bind the execution attempt as well. Read
        # inventories again after all applicable workflows, not just each one.
        for workflow, (jobs, names) in validated.items():
            current = validate_run(self.latest(head, workflow), jobs, head, workflow, names)
            require(current == proofs[workflow], "workflow changed while evidence was read")
        return proofs

    def verify(self, head, paths, target):
        return self._verify(head, paths, target, include_core=True)

    def optional(self, head, paths, target):
        return self._verify(head, paths, target, include_core=False)


def pr_identity(pr):
    require(isinstance(pr, dict) and isinstance(pr.get("head"), dict), "invalid PR evidence")
    head, base = pr["head"], pr.get("base", {})
    require(trusted(head.get("repo")) and trusted(base.get("repo")), "foreign PR repository")
    require(base.get("ref") in TARGETS and pr.get("draft") is False, "invalid PR target or draft")
    return sha(base.get("sha")), sha(head.get("sha")), base["ref"]


def validate_checkout(repo, pr, tested_commit, needs):
    base, head, target = pr_identity(pr)
    require(
        isinstance(needs, dict)
        and set(needs) == set(CORE[:-1])
        and all(isinstance(n, dict) and n.get("result") == "success" for n in needs.values()),
        "all full software gates must succeed before candidate acceptance",
    )
    require(repo.git("rev-parse", "HEAD") == sha(tested_commit), "checkout differs from event SHA")
    require(repo.ancestor(base, head), "PR must include the latest target baseline")
    require(repo.tree(tested_commit) == repo.tree(head), "tested merge tree differs from PR head")
    return {"base": base, "head": head, "target": target, "tree": repo.tree(head)}


def validate_current_pr(pr, receipt):
    base, head, target = pr_identity(pr)
    require(pr.get("state") == "open" and pr.get("merged") is False, "PR is no longer open")
    require(
        (base, head, target) == (receipt["base"], receipt["head"], receipt["target"]),
        "PR candidate or baseline changed while checking evidence",
    )


def verify_candidate(repo, pr, expected_head, expected_base, evidence):
    base, head, target = pr_identity(pr)
    require(pr.get("state") == "open" and pr.get("merged") is False, "PR is not open")
    require(head == sha(expected_head) and base == sha(expected_base), "PR candidate changed")
    require(repo.git("rev-parse", "HEAD") == head, "local HEAD differs from candidate")
    require(not repo.git("status", "--porcelain", "--untracked-files=all"), "checkout is not clean")
    require(repo.remote_head(target) == base, "target baseline changed")
    repo.git("fetch", "--no-tags", "origin", target)
    require(repo.git("rev-parse", "FETCH_HEAD") == base, "target changed during fetch")
    require(repo.ancestor(base, head), "update branch to latest target before validation")
    require(base != head, "candidate has no new commit")
    checks = evidence(head, repo.changed(base, head), target)
    require(bool(checks), "no software evidence")
    return {
        "mode": "verified_candidate",
        "base": base,
        "head": head,
        "target": target,
        "tree": repo.tree(head),
        "checks": checks,
        "published": False,
        "product_acceptance_inferred": False,
        "release_authorization_inferred": False,
    }


def publish_candidate(repo, receipt):
    head, base, target = sha(receipt["head"]), sha(receipt["base"]), receipt["target"]
    require(repo.git("rev-parse", "HEAD") == head, "local head changed before publication")
    require(not repo.git("status", "--porcelain", "--untracked-files=all"), "checkout changed")
    require(repo.remote_head(target) == base, "target advanced; revalidate before publishing")
    require(repo.ancestor(base, head) and repo.tree(head) == receipt["tree"], "candidate changed")
    # Normal Git fast-forward only. An unrelated concurrent advance is rejected
    # by the server, never recombined with our already-validated tree.
    repo.git("push", "--no-follow-tags", "origin", head + ":refs/heads/" + target)
    require(
        repo.remote_head(target) == head, "target changed after publication; inspect, do not retry"
    )
    return receipt | {"published": True}


def verify_publication(repo, pr, before, after, target, evidence):
    _, head, pr_target = pr_identity(pr)
    sha(before)
    sha(after)
    require(
        pr.get("state") == "closed"
        and pr.get("merged") is True
        and pr.get("merge_commit_sha") == after
        and target == pr_target,
        "push has no matching merged PR",
    )
    require(repo.git("rev-parse", "HEAD") == after, "push checkout differs from actual commit")
    require(repo.ancestor(before, head) and repo.ancestor(head, after), "unverified push ancestry")
    require(repo.tree(after) == repo.tree(head), "published tree differs from validated head")
    if after != head:
        require(
            repo.git("show", "-s", "--format=%P", after).split() == [before, head],
            "unexpected merge parents",
        )
    checks = evidence(head, repo.changed(before, head), target)
    require(bool(checks), "missing premerge evidence")
    return {
        "mode": "reused_premerge",
        "base": before,
        "head": head,
        "published_commit": after,
        "tree": repo.tree(after),
        "checks": checks,
        "full_rerun": False,
        "product_acceptance_inferred": False,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("check-pr", "check-push"):
        commands.add_parser(name)
    merge = commands.add_parser("merge")
    merge.add_argument("--pr", type=int, required=True)
    merge.add_argument("--expected-head", required=True)
    merge.add_argument("--expected-base", required=True)
    merge.add_argument("--execute", action="store_true")
    merge.add_argument("--release-approved", action="store_true")
    return p


def _pushed_pr(api, after, target):
    deadline = time.monotonic() + 60
    while True:
        summaries = api.get(f"commits/{sha(after)}/pulls?per_page=100")
        require(isinstance(summaries, list), "invalid commit PR inventory")
        matches = []
        for summary in summaries:
            if summary.get("base", {}).get("ref") != target:
                continue
            pr = api.pr(summary.get("number"))
            if pr.get("merged") is True and pr.get("merge_commit_sha") == after:
                matches.append(pr)
        require(len(matches) <= 1, "ambiguous merged PR identity")
        if matches:
            return matches[0]
        require(time.monotonic() < deadline, "no merged PR proof for this push")
        time.sleep(5)


def main(argv=None):
    args = parser().parse_args(argv)
    repo = Repository(Path(__file__).resolve().parents[2])
    try:
        repo.check_origin()
        api = API()
        if args.command == "merge":
            pr = api.pr(args.pr)
            result = verify_candidate(
                repo, pr, args.expected_head, args.expected_base, Checks(api).verify
            )
            # Re-read mutable PR metadata after the evidence requests.
            validate_current_pr(api.pr(args.pr), result)
            if args.execute:
                require(
                    result["target"] != "main" or args.release_approved,
                    "main publication requires explicit release decision and acknowledgement",
                )
                result = publish_candidate(repo, result)
        else:
            expected_event = "pull_request" if args.command == "check-pr" else "push"
            require(os.environ.get("GITHUB_EVENT_NAME") == expected_event, "wrong workflow event")
            require(
                os.environ.get("GITHUB_REPOSITORY", "").casefold() == REPOSITORY.casefold(),
                "wrong workflow repository",
            )
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            commit = sha(os.environ["GITHUB_SHA"])
            if args.command == "check-pr":
                pr = event["pull_request"]
                result = validate_checkout(
                    repo, pr, commit, json.loads(os.environ["CANDIDATE_NEEDS"])
                )
                checks = Checks(api, wait_seconds=1800).optional(
                    result["head"], repo.changed(result["base"], result["head"]), result["target"]
                )
                validate_current_pr(api.pr(event["number"]), result)
                repo.git("fetch", "--no-tags", "origin", result["target"])
                latest = sha(repo.git("rev-parse", "FETCH_HEAD"))
                require(
                    repo.ancestor(latest, result["head"]), "target advanced outside validated tree"
                )
                result.update(mode="full_premerge", checked_base=latest, optional_checks=checks)
            else:
                target = os.environ.get("GITHUB_REF", "").removeprefix("refs/heads/")
                require(target in TARGETS and event.get("after") == commit, "wrong push identity")
                result = verify_publication(
                    repo,
                    _pushed_pr(api, commit, target),
                    sha(event["before"]),
                    commit,
                    target,
                    Checks(api).verify,
                )
        print(json.dumps(result, sort_keys=True))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with Path(summary).open("a") as stream:
                stream.write("\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
        return 0
    except (GateError, KeyError, TypeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"candidate gate rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
