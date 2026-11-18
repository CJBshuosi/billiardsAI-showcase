"""Portable loop regressions; subprocess seams do not prove Windows Job behavior."""

import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from scripts.development import blackbox_report as report
from scripts.development import blackbox_source as source
from scripts.development import blackbox_validation as validation


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def owned_result(code=0, **overrides):
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "reason": "completed" if code == 0 else "worker-failed",
        "worker_exit_code": code,
        "cleanup_confirmed": True,
        **overrides,
    }


def child_result(root, phase="worker", code=0, **overrides):
    value = {
        "contract": "blackbox-dev-child/v1",
        "run_id": root.name,
        "phase": phase,
        "exit_code": code,
        "failure_code": None,
        "failure_stage": phase.upper(),
        "identity": None,
        **overrides,
    }
    report.atomic_json(root / f"{phase}-result.json", value)
    return value


def prepare_run(root):
    root.mkdir(exist_ok=True)
    (root / "export").mkdir()
    (root / "local-logs").mkdir()
    return root


def machine_config(root):
    config = {"contract": "blackbox-dev-machine/v1"}
    for name in (
        "runtime",
        "dependencies",
        "fixtures",
        "codec",
        "ownership",
        "runs",
        "git_executable",
        "bootstrap_python",
    ):
        config[name] = str(root / name)
    return config


@pytest.fixture
def committed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Synthetic Test")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    for name, data in {
        "src/package/__init__.py": b"",
        "src/package/code.py": b"value = 1\n",
        "scripts/development/entry.py": b"pass\n",
        "configs/algorithm/shot-local.yaml": b"fixed: true\n",
        ".superpowers/private.txt": b"PRIVATE",
        "clips/private.mp4": b"PRIVATE",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "synthetic fixture")
    return repo, git(repo, "rev-parse", "HEAD").decode().strip()


def commit_snapshot(path):
    git(path, "init", "-q")
    git(path, "config", "user.name", "Synthetic Test")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "commit.gpgsign", "false")
    git(path, "add", ".")
    git(path, "commit", "-qm", "code-only transport")
    return git(path, "rev-parse", "HEAD").decode().strip()


def test_actual_git_export_keeps_committed_bytes_and_excludes_private_history(committed, tmp_path):
    repo, commit = committed
    (repo / "src/package/code.py").write_bytes(b"uncommitted change\n")
    target = tmp_path / "snapshot"
    lock = source.export(repo, commit, target)
    assert source.verify(target) == lock
    assert (target / "src/package/code.py").read_bytes() == b"value = 1\n"
    assert (target / "src/package/__init__.py").stat().st_size == 0
    assert not any((target / name).exists() for name in (".superpowers", "clips", ".git"))
    assert lock["commit"] == commit
    assert (target / ".gitattributes").read_bytes() == b"* -text\n"


@pytest.mark.parametrize("mutation", ["changed", "extra", "symlink"])
def test_snapshot_rejects_tampering(committed, tmp_path, mutation):
    repo, commit = committed
    target = tmp_path / "snapshot"
    source.export(repo, commit, target)
    if mutation == "changed":
        (target / "src/package/code.py").write_bytes(b"different")
    elif mutation == "extra":
        (target / "extra.txt").write_bytes(b"extra")
    else:
        (target / "linked").symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError):
        source.verify(target)


def test_sync_checkout_freezes_exact_transport_and_rejects_dirty_checkout(committed, tmp_path):
    repo, commit = committed
    sync = tmp_path / "sync"
    source.export(repo, commit, sync)
    transport = commit_snapshot(sync)
    lock = validation.freeze_checkout(sync, transport, tmp_path / "frozen")
    assert lock["commit"] == commit
    with pytest.raises(validation.ValidationFailure, match="TRANSPORT_COMMIT_MISMATCH"):
        validation.freeze_checkout(sync, "f" * 40, tmp_path / "wrong")
    (sync / "untracked").write_text("private")
    with pytest.raises(validation.ValidationFailure, match="SOURCE_CHECKOUT_DIRTY"):
        validation.freeze_checkout(sync, transport, tmp_path / "dirty")


def test_real_git_autocrlf_checkout_preserves_locked_source_bytes(committed, tmp_path):
    repo, commit = committed
    sync = tmp_path / "sync"
    expected = source.export(repo, commit, sync)
    transport = commit_snapshot(sync)
    checkout = tmp_path / "windows-line-ending-checkout"
    subprocess.check_call(
        ["git", "clone", "-q", "-c", "core.autocrlf=true", "--no-local", str(sync), str(checkout)]
    )
    assert git(checkout, "config", "--get", "core.autocrlf").strip() == b"true"
    assert source.verify(checkout, checkout=True) == expected
    assert validation.freeze_checkout(checkout, transport, tmp_path / "frozen") == expected


def test_dependency_snapshot_imports_only_pinned_members_and_uses_copied_bytes(tmp_path):
    origin = tmp_path / "original-dependencies"
    origin.mkdir()
    listed = origin / "blackbox_pinned_probe.py"
    listed.write_text("VALUE = 1\n")
    (origin / "blackbox_unlisted_probe.py").write_text("raise AssertionError('UNREVIEWED')\n")
    root = tmp_path / "run"
    root.mkdir()
    validation.copy_pinned_tree(origin, root / "dependencies", {listed.name: source.pin(listed)})
    listed.write_text("VALUE = 2\n")
    script = (
        "import sys,importlib.util; "
        f"sys.path.insert(0, {str(root / 'dependencies')!r}); "
        "import blackbox_pinned_probe as probe; assert probe.VALUE == 1; "
        "assert importlib.util.find_spec('blackbox_unlisted_probe') is None"
    )
    subprocess.check_call([sys.executable, "-I", "-B", "-c", script], cwd=root)
    runtime = tmp_path / "original-runtime"
    runtime.mkdir()
    (runtime / "python.exe").write_bytes(b"pinned-runtime")
    validation.child_runtime(
        root,
        {
            "runtime": str(runtime),
            "runtime_pins": {"python.exe": source.pin(runtime / "python.exe")},
            "dependencies": str(origin),
        },
    )
    search_path = (root / "runtime/python312._pth").read_text().splitlines()
    assert str(root / "dependencies") in search_path
    assert str(origin) not in search_path
    assert not (root / "dependencies/blackbox_unlisted_probe.py").exists()


@pytest.mark.parametrize("name", ["inject.pth", "inject._pth", "sitecustomize.py", "module.pyc"])
def test_dependency_snapshot_rejects_startup_injection_members(tmp_path, name):
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / name).write_bytes(b"unreviewed")
    with pytest.raises(validation.ValidationFailure, match="DEPENDENCY_MEMBER_INVALID"):
        validation.copy_pinned_tree(origin, tmp_path / "target", {name: source.pin(origin / name)})


@pytest.mark.parametrize(
    "key",
    [
        "runtime",
        "dependencies",
        "fixtures",
        "codec",
        "ownership",
        "runs",
        "git_executable",
        "bootstrap_python",
    ],
)
def test_original_relative_machine_path_is_rejected_before_normalization(tmp_path, key):
    config = machine_config(tmp_path)
    config[key] = "."
    with pytest.raises(validation.ValidationFailure, match="CONFIG_ABSOLUTE_PATH_REQUIRED") as exc:
        validation.normalize_config(config)
    assert exc.value.stage == "CONFIG" and exc.value.exit_code == 2


def test_normalize_config_performs_no_filesystem_access(tmp_path, monkeypatch):
    config = machine_config(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("configuration normalization must remain lexical before owned setup")

    with monkeypatch.context() as patch:
        patch.setattr(source, "plain", forbidden)
        patch.setattr(Path, "stat", forbidden)
        normalized = validation.normalize_config(config)
    assert normalized == config and normalized is not config


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\dependencies",
        "//server/share/dependencies",
        r"\\?\C:\dependencies",
        r"\\.\C:\dependencies",
    ],
)
def test_normalize_config_rejects_unc_and_device_paths_lexically(monkeypatch, path):
    keys = (
        "runtime",
        "dependencies",
        "fixtures",
        "codec",
        "ownership",
        "runs",
        "git_executable",
        "bootstrap_python",
    )
    config = {"contract": "blackbox-dev-machine/v1", **{key: f"C:/local/{key}" for key in keys}}
    config["dependencies"] = path
    # Exercise Windows lexical semantics on this host without touching Windows or any share.
    monkeypatch.setattr(validation, "Path", PureWindowsPath)
    monkeypatch.setattr(validation, "os", SimpleNamespace(name="nt"))
    with pytest.raises(validation.ValidationFailure, match="CONFIG_ABSOLUTE_PATH_REQUIRED") as exc:
        validation.normalize_config(config)
    assert exc.value.stage == "CONFIG" and exc.value.exit_code == 2


def test_read_json_rejects_oversize_before_open_or_hash(tmp_path, monkeypatch):
    path = tmp_path / "oversize.json"
    with path.open("wb") as stream:
        stream.truncate(2 * 1024**2 + 1)
    original_open = Path.open

    def forbid_oversize_read(self, *args, **kwargs):
        if self == path:
            pytest.fail("oversize JSON must be rejected from stat before opening")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_oversize_read)
    monkeypatch.setattr(source, "pin", lambda *args: pytest.fail("JSON precheck must not hash"))
    with pytest.raises(validation.ValidationFailure, match="JSON_SIZE_BUDGET"):
        validation.read_json(path)


def test_monitor_exports_only_safe_progress_and_preserves_partial_lines(tmp_path):
    prepare_run(tmp_path)
    summary = {"run_id": "test", "status": "RUNNING"}
    monitor = validation.Monitor(tmp_path, summary)
    stdout = tmp_path / "local-logs/worker-stdout.txt"
    first = {
        "stage": "background",
        "available_count": 1,
        "processing": {"status": "COMPLETE", "token": "SECRET"},
        "url": "SECRET",
    }
    stdout.write_bytes(json.dumps(first).encode() + b'\n{"stage":')
    monitor.observe()
    with stdout.open("ab") as stream:
        stream.write(b'"verified_media","count":2,"private":"SECRET"}\n')
    monitor.observe()
    exported = (tmp_path / "export/events.jsonl").read_text()
    assert "SECRET" not in exported
    assert summary["last_observed"]["count"] == 2
    assert summary["last_observed"]["phase"] == "worker"
    report.publish(tmp_path / "export", summary)
    manifest = json.loads((tmp_path / "export/evidence-manifest.json").read_bytes())
    assert manifest["members"]["summary.json"] == source.pin(tmp_path / "export/summary.json")


@pytest.mark.parametrize("reason", ["completed", "descendants-after-parent-exit"])
@pytest.mark.parametrize("terminal_record_failed", [False, True])
def test_controller_failure_with_zero_worker_and_confirmed_cleanup_never_passes(
    reason, terminal_record_failed
):
    summary = {"profile": "Preflight"}
    flags = {"terminal_record_failed": True} if terminal_record_failed else {}
    validation.finish(
        summary,
        owned_result(status="FAIL", reason=reason, **flags),
        result={"exit_code": 0},
    )
    assert summary["status"] == "ERROR" and summary["exit_code"] == 5
    assert summary["failure_code"] == "OWNERSHIP_FAILED"


@pytest.mark.parametrize(
    "failure",
    [
        {"cleanup_confirmed": False},
        {"terminal_record_failed": True},
        {"job_close_error": "PRIVATE"},
    ],
)
def test_owned_cleanup_and_journal_failures_take_priority_over_timeouts(failure):
    summary = {"profile": "Preflight"}
    validation.finish(
        summary, owned_result(reason="deadline", **failure), observed_failure="PREFLIGHT_TIMEOUT"
    )
    assert (summary["exit_code"], summary["failure_code"]) == (5, "OWNERSHIP_FAILED")


@pytest.mark.parametrize(
    "owned,status,code",
    [
        (owned_result(reason="deadline"), "TIMEOUT", 3),
        (owned_result(exception_type="KeyboardInterrupt"), "CANCELLED", 4),
        (owned_result(1), "FAILED", 1),
        (owned_result(cleanup_confirmed=False), "ERROR", 5),
    ],
)
def test_timeout_cancel_failure_never_fabricate_product_or_browser_pass(owned, status, code):
    result = validation.finish(
        {"profile": "Fullgame", "source_commit": "a" * 40, "browser_status": "NOT_RUN"}, owned
    )
    assert result["status"] == status and result["exit_code"] == code
    assert result["product_status"] == "REPORT_UNAVAILABLE"
    assert result["browser_status"] == "NOT_RUN"


@pytest.mark.parametrize("media,expected", [(None, None), ([], 0)])
def test_unknown_media_count_is_null_and_observed_empty_list_is_zero(media, expected):
    product = {"source_commit": "a" * 40, "source_sha256": validation.SOURCE_SHA}
    if media is not None:
        product["media"] = media
    summary = {"profile": "Fullgame", "source_commit": "a" * 40}
    validation.finish(
        summary,
        owned_result(1),
        product,
        result={"exit_code": 1, "failure_code": "PRODUCT_FAILED", "failure_stage": "FULLGAME"},
    )
    assert summary["product_evidence"]["clip_count"] is expected
    assert summary["product_status"] == "FAIL"


def test_valid_product_timeout_is_exit_three_and_wrong_identity_is_error():
    product = {
        "source_commit": "a" * 40,
        "source_sha256": validation.SOURCE_SHA,
        "failure": {"type": "TimeoutError"},
    }
    summary = {"profile": "Fullgame", "source_commit": "a" * 40}
    validation.finish(
        summary,
        owned_result(1),
        product,
        result={"exit_code": 1, "failure_code": "PRODUCT_FAILED", "failure_stage": "FULLGAME"},
    )
    assert (summary["status"], summary["exit_code"], summary["failure_code"]) == (
        "TIMEOUT",
        3,
        "PRODUCT_TIMEOUT",
    )
    product["source_commit"] = "b" * 40
    validation.finish(summary, owned_result(1), product, result={"exit_code": 1})
    assert summary["exit_code"] == 5
    assert summary["failure_code"] == "REPORT_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    "owned,status,code,failure_code",
    [
        (owned_result(None, reason="deadline"), "TIMEOUT", 3, "WORKER_TIMEOUT"),
        (
            owned_result(None, reason="exception", exception_type="KeyboardInterrupt"),
            "CANCELLED",
            4,
            "CANCELLED",
        ),
        (owned_result(cleanup_confirmed=False), "ERROR", 5, "OWNERSHIP_FAILED"),
    ],
)
def test_wrong_identity_report_does_not_replace_terminal_cause(owned, status, code, failure_code):
    product = {"source_commit": "b" * 40, "source_sha256": validation.SOURCE_SHA}
    summary = {"profile": "Fullgame", "source_commit": "a" * 40}
    validation.finish(summary, owned, product)
    assert (summary["status"], summary["exit_code"], summary["failure_code"]) == (
        status,
        code,
        failure_code,
    )
    assert summary["product_report_state"] == "IDENTITY_MISMATCH"
    assert summary["product_status"] == "REPORT_UNAVAILABLE"


@pytest.mark.parametrize(
    "failure,stage",
    [
        ("DEPENDENCY_IMPORT_FAILED", "DEPENDENCIES"),
        ("INPUT_IDENTITY_MISMATCH", "INPUT"),
        ("H264_ENCODER_UNAVAILABLE", "CODEC"),
        ("RUNTIME_IDENTITY_MISMATCH", "RUNTIME"),
    ],
)
def test_typed_child_preconditions_produce_exit_two_and_bounded_result(
    tmp_path, monkeypatch, failure, stage
):
    root = prepare_run(tmp_path)
    report.atomic_json(root / "context.json", {"config": {}, "profile": "Preflight"})
    monkeypatch.setattr(
        validation.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(("BLACKBOX-RELEASE " + root.name + "\n").encode())),
    )

    def fail(*args):
        raise validation.ValidationFailure(failure, stage)

    monkeypatch.setattr(validation, "worker", fail)
    assert validation.child_main(root, "worker") == 2
    result = validation.read_result(root, "worker", owned_result(2))
    summary = {"profile": "Preflight"}
    validation.finish(summary, owned_result(2), result=result)
    assert summary["exit_code"] == 2 and summary["status"] == "PRECONDITION_FAILED"
    assert summary["failure_code"] == failure and summary["failure_stage"] == stage


def test_child_fullgame_timeout_uses_identity_bound_report(tmp_path, monkeypatch):
    root = prepare_run(tmp_path)
    report.atomic_json(root / "context.json", {"config": {}, "profile": "Fullgame"})
    report.atomic_json(root / "setup-result.json", {"identity": {"source_commit": "a" * 40}})
    (root / "recording").mkdir()
    report.atomic_json(
        root / "recording/report.json",
        {
            "source_commit": "a" * 40,
            "source_sha256": validation.SOURCE_SHA,
            "failure": {"type": "TimeoutError", "reason": "PRIVATE diagnostic"},
        },
    )
    monkeypatch.setattr(
        validation.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(("BLACKBOX-RELEASE " + root.name + "\n").encode())),
    )
    monkeypatch.setattr(validation, "worker", lambda *args: 1)
    assert validation.child_main(root, "worker") == 3
    result = validation.read_result(root, "worker", owned_result(3))
    assert result["failure_code"] == "PRODUCT_TIMEOUT"
    assert "PRIVATE" not in (root / "worker-result.json").read_text()


@pytest.mark.parametrize(
    "key,value",
    [
        ("failure_code", "TOKEN_SECRET"),
        ("failure_stage", "PRIVATE_STAGE"),
        ("failure_code", "https://private.invalid/token"),
        ("exit_code", True),
        ("run_id", "another-run"),
    ],
)
def test_child_result_rejects_unknown_failure_fields_and_identity(tmp_path, key, value):
    result = child_result(
        tmp_path, code=2, failure_code="INPUT_IDENTITY_MISMATCH", failure_stage="INPUT"
    )
    result[key] = value
    report.atomic_json(tmp_path / "worker-result.json", result)
    with pytest.raises(validation.ValidationFailure):
        validation.read_result(tmp_path, "worker", owned_result(2))


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "commit",
        "environment",
        "bool_size",
        "huge_size",
        "extra_pin",
        "worker_identity",
    ],
)
def test_setup_identity_rejects_unbounded_fields_and_invalid_pins(tmp_path, mutation):
    identity = {
        "source_commit": "a" * 40,
        "environment_id": "b" * 64,
        "source_lock": {"bytes": 1, "sha256": "c" * 64},
    }
    if mutation == "extra":
        identity["private"] = "PRIVATE"
    elif mutation == "missing":
        del identity["environment_id"]
    elif mutation == "commit":
        identity["source_commit"] = "moving-branch"
    elif mutation == "environment":
        identity["environment_id"] = "PRIVATE"
    elif mutation == "bool_size":
        identity["source_lock"]["bytes"] = True
    elif mutation == "huge_size":
        identity["source_lock"]["bytes"] = 2 * 1024**2 + 1
    elif mutation == "extra_pin":
        identity["source_lock"]["private"] = "PRIVATE"
    phase = "worker" if mutation == "worker_identity" else "setup"
    child_result(tmp_path, phase=phase, identity=identity)
    with pytest.raises(validation.ValidationFailure) as exc:
        validation.read_result(tmp_path, phase, owned_result())
    assert exc.value.exit_code == 5


def test_child_result_rejects_extra_fields_and_success_with_failure_code(tmp_path):
    child_result(tmp_path, private="PRIVATE")
    with pytest.raises(validation.ValidationFailure):
        validation.read_result(tmp_path, "worker", owned_result())
    child_result(tmp_path, failure_code="PRODUCT_FAILED")
    with pytest.raises(validation.ValidationFailure):
        validation.read_result(tmp_path, "worker", owned_result())


@pytest.mark.parametrize(
    "phase,expected", [("setup", "SETUP_TIMEOUT"), ("worker", "PREFLIGHT_TIMEOUT")]
)
def test_phase_timeout_is_sticky_and_cleanup_poll_does_not_throw_twice(tmp_path, phase, expected):
    prepare_run(tmp_path)
    clock = [0.0]
    monitor = validation.Monitor(
        tmp_path, {"run_id": "timeout"}, phase=phase, clock=lambda: clock[0]
    )
    clock[0] = 61
    with pytest.raises(RuntimeError, match=expected):
        monitor.observe()
    monitor.observe()
    assert monitor.summary["failure_code"] == expected


def test_log_budget_does_not_lose_failure_classification(tmp_path):
    prepare_run(tmp_path)
    monitor = validation.Monitor(tmp_path, {"run_id": "budget"})
    with (tmp_path / "local-logs/worker-stderr.txt").open("wb") as stream:
        stream.truncate(validation.LOG_LIMIT + 1)
    with pytest.raises(RuntimeError, match="LOG_BUDGET"):
        monitor.observe()
    monitor.observe()
    assert monitor.failure_code == "LOG_BUDGET"


def test_unknown_events_and_nonfinite_counters_are_not_exported():
    assert report.event({"stage": "SECRET", "token": "SECRET"}) is None
    assert report.event({"stage": "background", "available_count": float("nan")}) == {
        "stage": "background"
    }


@pytest.mark.parametrize(
    "phase,mode",
    [
        ("worker", "success"),
        ("worker", "failure"),
        ("worker", "timeout"),
        ("setup", "timeout"),
    ],
)
def test_execute_phase_real_child_via_portable_process_seam(tmp_path, monkeypatch, phase, mode):
    """Uses a real child and release pipe; no Windows Job API is exercised here."""
    root = prepare_run(tmp_path)
    script = root / "seam-child.py"
    code = 2 if mode == "failure" else 0
    script.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "phase=sys.argv[2]; root=Path(sys.argv[3])\n"
        "assert sys.stdin.buffer.readline() == ('BLACKBOX-RELEASE '+root.name+'\\n').encode()\n"
        "print(json.dumps({'stage':'preflight','decoded_frames':2,'token':'SECRET'}),flush=True)\n"
        + (
            "time.sleep(10)\n"
            if mode == "timeout"
            else "value="
            + repr(
                {
                    "contract": "blackbox-dev-child/v1",
                    "phase": phase,
                    "exit_code": code,
                    "failure_code": "INPUT_IDENTITY_MISMATCH" if code else None,
                    "failure_stage": "INPUT" if code else phase.upper(),
                    "identity": None,
                }
            )
            + "\nvalue['run_id']=root.name\n"
            "(root/(phase+'-result.json')).write_text(json.dumps(value))\n"
            f"raise SystemExit({code})\n"
        )
    )
    actual_popen = subprocess.Popen
    launched, deadlines = [], []

    def spawn(command, **kwargs):
        launched.append(command)
        return actual_popen([sys.executable, "-I", "-B", str(script), *command[-3:]], **kwargs)

    def own(api, spawn, persist, metadata, release, **kwargs):
        deadlines.append(kwargs["deadline"])
        process = spawn()
        persist("released", {"pid": process.pid})
        process.stdin.write(release)
        process.stdin.close()
        reason = "completed"
        try:
            process.wait(timeout=0.3 if mode == "timeout" else 5)
        except subprocess.TimeoutExpired:
            reason = "deadline"
            process.kill()  # Exact test child handle; actual implementation uses its Job.
            process.wait(timeout=5)
        return owned_result(
            process.returncode,
            reason=reason
            if reason == "deadline"
            else "completed"
            if process.returncode == 0
            else "worker-failed",
        )

    monkeypatch.setattr(validation.subprocess, "Popen", spawn)
    monkeypatch.setattr(validation, "child_environment", lambda *a, **k: dict(os.environ))
    summary = {"profile": "Preflight", "run_id": root.name}
    validation.execute_phase(
        root, {}, summary, SimpleNamespace(WindowsAPI=lambda: None, run_owned=own), phase
    )
    assert summary["exit_code"] == {"success": 0, "failure": 2, "timeout": 3}[mode]
    assert summary["owned_cleanup"] is True
    assert deadlines == [60]
    assert launched[0][-3:] == ["--child", phase, str(root)]
    assert summary["last_observed"]["decoded_frames"] == 2
    assert "SECRET" not in (root / "export/events.jsonl").read_text()


def test_final_observe_first_timeout_keeps_exit_three(tmp_path, monkeypatch):
    root = prepare_run(tmp_path)
    child_result(root)
    clock = iter([0.0, 61.0])
    monitor_type = validation.Monitor
    monkeypatch.setattr(
        validation, "Monitor", lambda *a, **k: monitor_type(*a, **k, clock=lambda: next(clock))
    )
    ownership = SimpleNamespace(WindowsAPI=lambda: None, run_owned=lambda *a, **k: owned_result())
    summary = {"profile": "Preflight", "run_id": root.name}
    validation.execute_phase(root, {}, summary, ownership, "worker")
    assert (summary["exit_code"], summary["status"], summary["failure_code"]) == (
        3,
        "TIMEOUT",
        "PREFLIGHT_TIMEOUT",
    )


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("deadline", 3),
        ("monitor", 3),
        ("cancel", 4),
        ("cleanup", 5),
        ("ordinary", 5),
    ],
)
def test_partial_report_during_termination_preserves_terminal_cause(
    tmp_path, monkeypatch, failure, expected
):
    root = prepare_run(tmp_path)
    (root / "recording").mkdir()
    (root / "recording/report.json").write_text('{"source_commit":')
    # Result can be present before a deadline while worker_exit_code is still unknown.
    child_result(root)
    if failure == "monitor":
        clock = iter([0.0, 61.0])
        monitor_type = validation.Monitor
        monkeypatch.setattr(
            validation, "Monitor", lambda *a, **k: monitor_type(*a, **k, clock=lambda: next(clock))
        )

    def own(*args, **kwargs):
        if failure == "monitor":
            try:
                kwargs["sleep"](0)
            except RuntimeError:
                return owned_result(None, reason="exception", exception_type="RuntimeError")
            pytest.fail("active monitor should have enforced the preflight deadline")
        if failure == "deadline":
            return owned_result(None, reason="deadline")
        if failure == "cancel":
            return owned_result(None, reason="exception", exception_type="KeyboardInterrupt")
        if failure == "cleanup":
            return owned_result(None, reason="deadline", cleanup_confirmed=False)
        return owned_result()

    summary = {"profile": "Fullgame", "run_id": root.name, "source_commit": "a" * 40}
    validation.execute_phase(
        root, {}, summary, SimpleNamespace(WindowsAPI=lambda: None, run_owned=own), "worker"
    )
    assert summary["exit_code"] == expected
    assert summary["product_report_state"] == "INVALID"
    assert (
        summary["failure_code"]
        == {
            "deadline": "WORKER_TIMEOUT",
            "monitor": "PREFLIGHT_TIMEOUT",
            "cancel": "CANCELLED",
            "cleanup": "OWNERSHIP_FAILED",
            "ordinary": "REPORT_INVALID",
        }[failure]
    )


@pytest.mark.parametrize(
    "error,expected,code",
    [
        (TimeoutError("PRIVATE"), "PRODUCT_TIMEOUT", 3),
        (RuntimeError("PRIVATE"), "FULLGAME_EXECUTION_FAILED", 5),
    ],
)
def test_worker_fullgame_exception_is_not_misclassified_as_precondition(
    tmp_path, monkeypatch, error, expected, code
):
    import cv2
    import numpy as np

    from billiards_ai.lab import mainline1_recording_media as media
    from scripts.algorithm import run_recording_first_fullgame as fullgame

    monkeypatch.setattr(source, "verify", lambda *a, **k: {})
    monkeypatch.setattr(cv2, "__file__", str(tmp_path / "dependencies/cv2/__init__.py"))
    monkeypatch.setattr(np, "__file__", str(tmp_path / "dependencies/numpy/__init__.py"))
    monkeypatch.setattr(fullgame, "__file__", str(tmp_path / "source/fullgame.py"))
    monkeypatch.setattr(media, "__file__", str(tmp_path / "source/media.py"))
    monkeypatch.setattr(fullgame, "_validate_inputs", lambda *a: None)
    monkeypatch.setattr(validation, "media_preflight", lambda *a: None)

    def fail(*args):
        raise error

    monkeypatch.setattr(fullgame, "main", fail)
    with pytest.raises(validation.ValidationFailure) as exc:
        validation.worker(tmp_path, {"profile": "Fullgame", "config": {"fixtures": str(tmp_path)}})
    assert (exc.value.code, exc.value.stage, exc.value.exit_code) == (expected, "FULLGAME", code)


def test_source_oversize_blob_rejected_before_blob_contents_are_requested(
    committed, tmp_path, monkeypatch
):
    repo, _ = committed
    large = repo / "src/package/oversize.py"
    with large.open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)
    git(repo, "add", "src/package/oversize.py")
    git(repo, "commit", "-qm", "oversize synthetic source")
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    oid = git(repo, "rev-parse", "HEAD:src/package/oversize.py").decode().strip()
    actual_git = source.git

    def bounded_git(repo, *args, **kwargs):
        if args == ("cat-file", "blob", oid):
            pytest.fail("oversize source bytes must not be read into the export process")
        return actual_git(repo, *args, **kwargs)

    monkeypatch.setattr(source, "git", bounded_git)
    with pytest.raises(ValueError, match="budget"):
        source.export(repo, commit, tmp_path / "snapshot")


def test_setup_child_contains_source_freeze_dependency_codec_and_runtime_copy(
    tmp_path, monkeypatch
):
    calls = []
    config = {
        "git_executable": "verified-git",
        "dependencies": "dependency-origin",
        "dependency_pins": {"member.py": {}},
        "codec": "codec-origin",
    }
    monkeypatch.setattr(validation, "check_config", lambda value: calls.append("config") or value)
    monkeypatch.setattr(
        validation,
        "freeze_checkout",
        lambda *a, **k: calls.append(("freeze", k)) or {"commit": "a" * 40},
    )
    monkeypatch.setattr(
        validation,
        "copy_pinned_tree",
        lambda origin, target, pins: calls.append(("copy", origin, target.name)),
    )
    monkeypatch.setattr(validation, "child_runtime", lambda *a: calls.append("runtime"))
    monkeypatch.setattr(source, "pin", lambda *a: {"bytes": 1, "sha256": "b" * 64})
    monkeypatch.setattr(
        validation.shutil, "disk_usage", lambda *a: SimpleNamespace(free=100 * 1024**3)
    )
    config["runtime_pins"] = {"python.exe": {"bytes": 1, "sha256": "c" * 64}}
    identity = validation.setup_worker(
        tmp_path,
        {
            "config": config,
            "repo": str(tmp_path),
            "commit": "d" * 40,
        },
    )
    assert calls == [
        "config",
        ("freeze", {"git_executable": "verified-git", "check_remote": True}),
        ("copy", "dependency-origin", "dependencies"),
        ("copy", "codec-origin", "codec"),
        "runtime",
    ]
    assert identity["source_commit"] == "a" * 40
    assert len(identity["environment_id"]) == 64


def test_failed_final_publication_still_releases_owned_lock(tmp_path, monkeypatch):
    root = prepare_run(tmp_path / "run")
    lock = tmp_path / "validation.lock"
    lock.write_text(root.name)
    monkeypatch.setattr(report, "publish", lambda *a: (_ for _ in ()).throw(OSError("PRIVATE")))
    summary = {"run_id": root.name, "status": "PASS", "exit_code": 0}
    validation.finalize(root, summary, lock, True, validation.time.monotonic())
    assert not lock.exists()
    assert summary["exit_code"] == 5 and summary["failure_code"] == "REPORT_PUBLICATION_FAILED"
    durable = json.loads((root / "export/summary.json").read_bytes())
    assert durable["loop_status"] == "REPORT_FAILED"
    assert "PRIVATE" not in json.dumps(durable)


@pytest.mark.parametrize("mode", ["success", "setup_failure", "worker_failure", "busy"])
def test_run_orchestrator_records_identity_timestamps_and_unlocks(tmp_path, monkeypatch, mode):
    config = machine_config(tmp_path)
    runs = Path(config["runs"])
    runs.mkdir()
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(config))
    if mode == "busy":
        (runs / "validation.lock").write_text("existing-run")
    calls = []

    def execute(root, config, summary, ownership, phase):
        calls.append(phase)
        if mode == "setup_failure" and phase == "setup":
            raise validation.ValidationFailure("SOURCE_LOCK_INVALID", "SOURCE")
        code = 2 if mode == "worker_failure" and phase == "worker" else 0
        result = child_result(
            root,
            phase,
            code,
            failure_code="INPUT_IDENTITY_MISMATCH" if code else None,
            failure_stage="INPUT" if code else phase.upper(),
        )
        (root / f"local-logs/{phase}-stdout.txt").write_text("PRIVATE raw diagnostic")
        report.atomic_json(root / f"{phase}-ownership.json", owned_result(code))
        validation.finish(summary, owned_result(code), result=result, phase=phase)
        if phase == "setup":
            result["identity"] = {
                "source_commit": "a" * 40,
                "source_lock": {"bytes": 1, "sha256": "c" * 64},
                "environment_id": "d" * 64,
            }
        elif not code:
            summary["last_observed"] = {"stage": "preflight", "decoded_frames": 2}
        return result

    monkeypatch.setattr(validation.sys, "platform", "win32")
    monkeypatch.setattr(validation, "load_ownership", lambda *a: object())
    monkeypatch.setattr(validation, "execute_phase", execute)
    result = validation.run(tmp_path, "b" * 40, path, "Preflight")
    root = next(item for item in runs.iterdir() if item.is_dir())
    summary = json.loads((root / "export/summary.json").read_bytes())
    assert result == {"success": 0, "setup_failure": 2, "worker_failure": 2, "busy": 6}[mode]
    assert summary["exit_code"] == result
    assert summary["wrapper_commit"] == summary["transport_commit"] == "b" * 40
    assert summary["input_sha256"] == validation.SOURCE_SHA
    assert datetime.fromisoformat(summary["ended_at_utc"]) >= datetime.fromisoformat(
        summary["started_at_utc"]
    )
    assert summary["elapsed_seconds"] >= 0 and summary["loop_status"] == "COMPLETE"
    assert summary["browser_status"] == "NOT_RUN"
    if mode == "success":
        assert summary["source_commit"] == "a" * 40 and summary["input_verified"] is True
        assert calls == ["setup", "worker"]
    elif mode == "setup_failure":
        assert calls == ["setup"] and summary["failure_code"] == "SOURCE_LOCK_INVALID"
    if mode == "busy":
        assert (runs / "validation.lock").read_text() == "existing-run" and calls == []
    else:
        assert not (runs / "validation.lock").exists()
    manifest = json.loads((root / "export/evidence-manifest.json").read_bytes())
    assert manifest["members"]["summary.json"] == source.pin(root / "export/summary.json")
    assert manifest["local_artifacts"]["recording/report.json"] == "MISSING"
    if calls and mode != "setup_failure":
        assert manifest["local_artifacts"]["local-logs/setup-stdout.txt"] == "PRESENT"
    assert "PRIVATE" not in (root / "export/summary.json").read_text()
    assert not (root / "export/local-logs").exists()


def test_real_input_codec_and_product_fsync_preflight(tmp_path):
    import cv2
    import numpy as np

    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    video = cv2.VideoWriter(
        str(fixture / "input.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 60, (1920, 1080)
    )
    assert video.isOpened()
    for _ in range(2):
        video.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    video.release()
    root = tmp_path / "run"
    root.mkdir()
    validation.media_preflight(root, fixture)
    assert (root / "encoder-preflight.mp4").stat().st_size > 0


@pytest.mark.parametrize("elapsed", [601, 901, 961, 1000000])
def test_until_complete_monitor_has_no_product_elapsed_deadline(tmp_path, elapsed):
    prepare_run(tmp_path)
    now = [0.0]
    summary = {"run_id": tmp_path.name, "profile": "FullgameComplete"}
    monitor = validation.Monitor(tmp_path, summary, clock=lambda: now[0])
    (tmp_path / "local-logs/worker-stdout.txt").write_text(
        '{"stage":"preflight"}\n{"stage":"capture_stopped"}\n'
    )
    monitor.observe()
    now[0] = elapsed
    monitor.observe()
    assert monitor.failure_code is None


def test_until_complete_cancel_marker_is_run_bound_and_sticky(tmp_path):
    prepare_run(tmp_path)
    monitor = validation.Monitor(tmp_path, {"run_id": tmp_path.name, "profile": "FullgameComplete"})
    (tmp_path / "cancel.json").write_text(json.dumps({"run_id": "another"}))
    monitor.observe()
    assert monitor.failure_code is None
    (tmp_path / "cancel.json").write_text(json.dumps({"run_id": tmp_path.name}))
    with pytest.raises(KeyboardInterrupt):
        monitor.observe()
    assert monitor.failure_code == "CANCELLED"
    monitor.observe()


def test_until_complete_parent_keeps_only_a_short_pre_end_capture_watchdog(tmp_path):
    prepare_run(tmp_path)
    now = [0.0]
    monitor = validation.Monitor(
        tmp_path, {"run_id": tmp_path.name, "profile": "FullgameComplete"}, clock=lambda: now[0]
    )
    (tmp_path / "local-logs/worker-stdout.txt").write_text('{"stage":"preflight"}\n')
    monitor.observe()
    now[0] = 391
    with pytest.raises(RuntimeError, match="CAPTURE_TIMEOUT"):
        monitor.observe()


def test_export_includes_only_the_pinned_approved_felt_asset(committed, tmp_path):
    repo, _ = committed
    asset = "scripts/development/assets/s1-felt.png"
    target = repo / asset
    target.parent.mkdir(parents=True)
    target.write_bytes(Path(asset).read_bytes())
    (target.parent / "private-photo.png").write_bytes(b"PRIVATE")
    git(repo, "add", "scripts/development/assets")
    git(repo, "commit", "-qm", "approved input artwork and excluded image")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    snapshot = tmp_path / "snapshot"
    lock = source.export(repo, head, snapshot)
    assert (snapshot / asset).is_file(), "approved input artwork must survive source export"
    assert lock["files"][asset] == {
        "bytes": 2775905,
        "sha256": "5b87d7f16d8a83c0f5bdc9c670577af02adadea47d9c1313a981472ddefac0f3",
    }
    assert not (snapshot / "scripts/development/assets/private-photo.png").exists()
    assert source.verify(snapshot) == lock


def test_export_rejects_changed_content_at_the_named_artwork_path(committed, tmp_path):
    repo, _ = committed
    asset = repo / "scripts/development/assets/s1-felt.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"PRIVATE replacement")
    git(repo, "add", "scripts/development/assets")
    git(repo, "commit", "-qm", "invalid artwork content")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="asset"):
        source.export(repo, head, tmp_path / "snapshot")
