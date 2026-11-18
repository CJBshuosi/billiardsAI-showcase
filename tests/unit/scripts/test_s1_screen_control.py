"""Windows owner transport only; these tests never prove Windows execution."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def tool():
    assert Path("scripts/development/s1_screen_control.py").is_file()
    return importlib.import_module("scripts.development.s1_screen_control")


def test_owner_budget_separates_cold_preparation_from_unchanged_hot_request():
    from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotConfig

    config = SnapshotConfig()
    budgets = tool().execution_budgets()
    assert budgets["cold_preparation_seconds"] == config.preparation_seconds == 300
    assert budgets["hot_request_seconds"] == config.task_seconds == 8
    assert budgets["case_seconds"] == 240
    assert budgets["owner_seconds"] == budgets["cold_preparation_seconds"] + 240
    assert budgets["cleanup_seconds"] == 10


def test_owned_child_uses_fixed_screen_entry_and_exact_release_id(tmp_path, monkeypatch):
    args = SimpleNamespace(
        **{
            k: tmp_path / k
            for k in (
                "worker_python",
                "input_root",
                "config",
                "browser_cli",
                "browser_exe",
                "ffmpeg",
            )
        },
        mode="recommendation",
    )
    command = tool().child_command(args, tmp_path / "case", "a" * 32)
    assert command[:3] == [str(args.worker_python), "-I", "-B"]
    assert command[3].endswith("/scripts/development/s1_screen_acceptance.py")
    assert command[-2:] == ["--owned-run-id", "a" * 32]
    assert "--allow-dirty-development" not in command
    assert command[command.index("--capture-seconds") + 1] == "240"
    from scripts.development import s1_screen_acceptance as screen

    received = []
    monkeypatch.setattr(screen, "await_owned_release", received.append)
    monkeypatch.setattr(screen, "run", lambda parsed: received.append(parsed) or 0)
    monkeypatch.setattr(sys, "argv", command[3:])
    assert screen.main() == 0
    assert received[0] == "a" * 32
    assert received[1].capture_seconds == 240


@pytest.mark.parametrize("profile_imports", [False, True])
def test_explicit_import_profile_reaches_real_isolated_spawn_and_native_trace(
    tmp_path, profile_imports
):
    args = SimpleNamespace(
        **{
            k: tmp_path / k
            for k in ("input_root", "config", "browser_cli", "browser_exe", "ffmpeg")
        },
        worker_python=sys.executable,
        mode="recommendation",
        profile_imports=profile_imports,
    )
    command = tool().child_command(args, tmp_path / "case", "a" * 32)
    entry = next(i for i, part in enumerate(command) if part.endswith("s1_screen_acceptance.py"))
    program = tmp_path / "native_import_profile.py"
    destination = tmp_path / "child.json"
    program.write_text(
        "import json,multiprocessing,os,pathlib,sys\n"
        "def child():\n"
        " import fractions\n"
        f" sys.path.insert(0,{str(Path.cwd())!r})\n"
        " from scripts.development.s1_screen_acceptance import trace_event\n"
        " path=pathlib.Path(sys.argv[1])\n"
        " record=dict(pid=os.getpid(),enabled=bool(sys._xoptions.get('importtime')))\n"
        " path.write_text(json.dumps(record),encoding='utf-8')\n"
        " trace_event(path.with_suffix('.jsonl'),'factory_enter')\n"
        "if __name__=='__main__':\n"
        " process=multiprocessing.get_context('spawn').Process(target=child)\n"
        " process.start();process.join(5)\n"
        " assert process.exitcode==0\n"
        " print(json.dumps(dict(parent_pid=os.getpid())))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        command[:entry] + [str(program), str(destination)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    native = json.loads(destination.read_bytes())
    assert native["enabled"] is profile_imports
    assert native["pid"] != json.loads(result.stdout)["parent_pid"]
    trace = json.loads(destination.with_suffix(".jsonl").read_bytes())
    assert trace["pid"] == native["pid"] and trace["python_import_time"] is profile_imports
    assert (
        any("import time:" in line and "fractions" in line for line in result.stderr.splitlines())
        is profile_imports
    )


@pytest.mark.parametrize(
    "change", ["deadline", "cleanup", "exit", "terminal", "source", "run", "childcleanup"]
)
def test_parent_rejects_unowned_partial_or_wrong_child_result(change):
    module = tool()
    owned = dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0)
    result = dict(
        contract="s1-online-screen-acceptance/v1",
        mode="recommendation",
        case_pass=False,
        online_pass=True,
        media_complete=False,
        cleanup_confirmed=True,
        owned_run_id="a" * 32,
        source=dict(kind="locked", producer="b" * 40, files={"x": {"sha256": "c" * 64}}),
    )
    if change == "deadline":
        owned["status"] = "DEADLINE"
    if change == "cleanup":
        owned["cleanup_confirmed"] = False
    if change == "exit":
        owned["worker_exit_code"] = 2
    if change == "terminal":
        owned["terminal_record_failed"] = True
    if change == "source":
        result["source"]["producer"] = "d" * 40
    if change == "run":
        result["owned_run_id"] = "e" * 32
    if change == "childcleanup":
        result["cleanup_confirmed"] = False
    assert not module.accept_report(
        owned,
        result,
        run_id="a" * 32,
        mode="recommendation",
        lock=dict(commit="b" * 40, files={"x": {"sha256": "c" * 64}}),
    )


def test_parent_accepts_only_matching_completed_owned_online_stage():
    module = tool()
    owned = dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0)
    files = {"x": {"sha256": "c" * 64}}
    result = dict(
        contract="s1-online-screen-acceptance/v1",
        mode="recommendation",
        case_pass=False,
        online_pass=True,
        media_complete=False,
        cleanup_confirmed=True,
        owned_run_id="a" * 32,
        source=dict(kind="locked", producer="b" * 40, files=files),
    )
    assert module.accept_report(
        owned,
        result,
        run_id="a" * 32,
        mode="recommendation",
        lock=dict(commit="b" * 40, files=files),
    )


@pytest.mark.parametrize("fault", [None, "cleanup", "source", "online_pin", "raw_pin", "stage"])
def test_media_gate_requires_matching_owned_prerequisites(fault):
    owned = dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0)
    source = dict(kind="locked", producer="b" * 40, files={})
    pin = dict(bytes=10, sha256="c" * 64)
    raw_pin = dict(bytes=20, sha256="d" * 64)
    result = dict(
        contract="s1-media-acceptance/v1",
        stage="export",
        stage_pass=True,
        owned_run_id="a" * 32,
        online_run_id="e" * 32,
        source=source,
        online_report_pin=pin,
        raw_report_pin=raw_pin,
    )
    if fault == "cleanup":
        owned["cleanup_confirmed"] = False
    if fault == "source":
        result["source"] = {}
    if fault == "online_pin":
        result["online_report_pin"] = {}
    if fault == "raw_pin":
        result["raw_report_pin"] = {}
    if fault == "stage":
        result["stage_pass"] = False
    assert tool().accept_media_report(
        owned,
        result,
        run_id="a" * 32,
        online_run_id="e" * 32,
        stage="export",
        source=source,
        online_report_pin=pin,
        raw_report_pin=raw_pin,
    ) is (fault is None)


def test_offline_command_is_a_separate_isolated_media_entry(tmp_path):
    args = SimpleNamespace(worker_python=Path(sys.executable))
    command = tool().media_command(
        args,
        tmp_path / "case",
        tmp_path / "media/result",
        "export",
        "a" * 32,
        tmp_path / "raw/local-report.json",
    )
    assert command[:3] == [sys.executable, "-I", "-B"]
    assert command[3].endswith("s1_media_acceptance.py")
    assert "s1_screen_acceptance.py" not in command[3]
    assert "--raw-report" in command


def test_split_verification_command_binds_capture_source_and_encode_report(tmp_path):
    args = SimpleNamespace(
        worker_python=Path(sys.executable), capture_source_root=tmp_path / "capture-source"
    )
    command = tool().media_command(
        args,
        tmp_path / "case",
        tmp_path / "verify/result",
        "verify",
        "4" * 32,
        tmp_path / "raw/result/local-report.json",
        tmp_path / "encode/result/local-report.json",
    )
    assert command[command.index("--capture-source-root") + 1] == str(args.capture_source_root)
    assert command[command.index("--encoded-report") + 1] == str(
        tmp_path / "encode/result/local-report.json"
    )


@pytest.mark.parametrize("fault", [None, "encode_pin", "capture_source", "verified"])
def test_verify_gate_requires_the_exact_encode_and_capture_identity(fault):
    source = dict(kind="locked", producer="b" * 40, files={})
    capture = dict(kind="locked", producer="c" * 40, files={})
    pin = dict(bytes=10, sha256="d" * 64)
    encoded = dict(bytes=20, sha256="e" * 64)
    result = dict(
        contract="s1-media-acceptance/v1",
        stage="verify",
        stage_pass=True,
        owned_run_id="4" * 32,
        online_run_id="1" * 32,
        source=source,
        capture_source=capture,
        online_report_pin=pin,
        raw_report_pin=pin,
        encoded_report_pin=encoded,
        verification_complete=True,
    )
    if fault == "encode_pin":
        result["encoded_report_pin"] = pin
    if fault == "capture_source":
        result["capture_source"] = source
    if fault == "verified":
        result["verification_complete"] = False
    assert tool().accept_media_report(
        dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0),
        result,
        run_id="4" * 32,
        online_run_id="1" * 32,
        stage="verify",
        source=source,
        capture_source=capture,
        online_report_pin=pin,
        raw_report_pin=pin,
        encoded_report_pin=encoded,
    ) is (fault is None)


def test_codec_requires_the_existing_pinned_dll_and_license(tmp_path):
    from scripts.algorithm.recording_windows_entry import CODEC

    module = tool()
    (tmp_path / CODEC["filename"]).write_bytes(b"untrusted DLL")
    (tmp_path / "BINARY_LICENSE.txt").write_bytes(b"untrusted license")
    with pytest.raises(ValueError, match="codec"):
        module.codec_environment(tmp_path)


def test_codec_environment_uses_one_existing_authoritative_inventory(tmp_path, monkeypatch):
    from scripts.algorithm.recording_windows_entry import CODEC

    module = tool()

    def pin(path):
        prefix = "license_" if path.name == "BINARY_LICENSE.txt" else ""
        return dict(bytes=CODEC[prefix + "bytes"], sha256=CODEC[prefix + "sha256"])

    monkeypatch.setattr(module.sources, "pin", pin)
    assert module.codec_environment(tmp_path) == {
        "OPENH264_LIBRARY": str(tmp_path / CODEC["filename"])
    }


@pytest.fixture
def resumable_case(tmp_path, monkeypatch):
    from scripts.development import blackbox_source as sources
    from scripts.development import s1_media_acceptance as media
    from scripts.development.blackbox_report import atomic_json

    capture_root, current = tmp_path / "capture-source", tmp_path / "execution-source"
    locks = []
    for root, producer in ((capture_root, "c" * 40), (current, "e" * 40)):
        root.mkdir()
        (root / ".gitattributes").write_bytes(sources.ATTRIBUTES)
        (root / "src").mkdir()
        (root / "src/runtime.py").write_text("UNCHANGED = True\n")
        offline = root / "scripts/development/s1_media_acceptance.py"
        offline.parent.mkdir(parents=True)
        offline.write_text(producer + "\n")
        lock = dict(
            contract=sources.CONTRACT,
            commit=producer,
            files={
                str(p.relative_to(root)): sources.pin(p) for p in root.rglob("*") if p.is_file()
            },
        )
        atomic_json(root / "SOURCE-LOCK.json", lock)
        locks.append(lock)
    monkeypatch.setattr(media, "ROOT", current)
    case = tmp_path / "original/case"
    case.mkdir(parents=True)
    online_id, raw_id = "1" * 32, "2" * 32
    capture = dict(kind="locked", producer=locks[0]["commit"], files=locks[0]["files"])
    online = dict(
        contract="s1-online-screen-acceptance/v1",
        mode="recommendation",
        owned_run_id=online_id,
        source=capture,
        case_pass=False,
        online_pass=True,
        media_complete=False,
        cleanup_confirmed=True,
        input_manifest=dict(frame_count=3),
    )
    atomic_json(case / "local-report.json", online)
    atomic_json(
        case.parent / "ownership.json",
        dict(run_id=online_id, status="PASS", worker_exit_code=0, cleanup_confirmed=True),
    )
    assets = dict(codec_dll=dict(bytes=10, sha256="a" * 64))
    atomic_json(
        case.parent / "context.json",
        dict(pins=assets, candidate=locks[0]["commit"], mode="recommendation"),
    )
    atomic_json(
        case.parent / "summary.json", dict(case_pass=False, online_pass=True, media_complete=False)
    )
    raw_path = tmp_path / "original/media-raw/result/local-report.json"
    raw_path.parent.mkdir(parents=True)
    atomic_json(
        raw_path,
        dict(
            contract="s1-media-acceptance/v1",
            stage="raw",
            stage_pass=True,
            owned_run_id=raw_id,
            online_run_id=online_id,
            source=capture,
            online_report_pin=sources.pin(case / "local-report.json"),
            frame_count=3,
            raw_recording=dict(decoded_frames=3),
        ),
    )
    atomic_json(
        raw_path.parent.parent / "ownership.json",
        dict(run_id=raw_id, status="PASS", worker_exit_code=0, cleanup_confirmed=True),
    )
    args = SimpleNamespace(
        resume_case_root=case,
        capture_source_root=capture_root,
        raw_report=raw_path,
        mode="recommendation",
    )
    return args, locks[1], assets, current


def test_resume_reads_successful_online_and_raw_without_modifying_failed_original(resumable_case):
    from scripts.development import blackbox_source as sources

    args, lock, assets, _ = resumable_case
    root = args.resume_case_root.parent
    before = {p: sources.pin(p) for p in root.rglob("*") if p.is_file()}
    result = tool().resume_evidence(args, lock, assets)
    assert result["online"]["source"]["producer"] == "c" * 40
    assert result["applicability"]["source"]["producer"] == "e" * 40
    assert result["raw"]["raw_recording"]["decoded_frames"] == 3
    assert before == {p: sources.pin(p) for p in root.rglob("*") if p.is_file()}
    assert json.loads((root / "summary.json").read_bytes())["case_pass"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "assets",
        "online_owner",
        "online_failed",
        "raw_failed",
        "raw_owner",
        "raw_pin",
        "raw_coverage",
        "runtime",
    ],
)
def test_resume_refuses_changed_or_failed_prerequisites(resumable_case, fault):
    from scripts.development import blackbox_source as sources
    from scripts.development.blackbox_report import atomic_json

    args, lock, assets, current = resumable_case
    path = None
    if fault == "assets":
        assets = {}
    elif fault == "runtime":
        changed = current / "src/runtime.py"
        changed.write_text("UNCHANGED = False\n")
        lock["files"]["src/runtime.py"] = sources.pin(changed)
        atomic_json(current / "SOURCE-LOCK.json", lock)
    else:
        path = (
            args.resume_case_root.parent / "ownership.json"
            if fault == "online_owner"
            else args.resume_case_root / "local-report.json"
            if fault == "online_failed"
            else args.raw_report.parent.parent / "ownership.json"
            if fault == "raw_owner"
            else args.raw_report
        )
        row = json.loads(path.read_bytes())
        if fault in {"online_owner", "raw_owner"}:
            row["cleanup_confirmed"] = False
        elif fault == "online_failed":
            row["online_pass"] = False
        elif fault == "raw_failed":
            row["stage_pass"] = False
        elif fault == "raw_pin":
            row["online_report_pin"] = {}
        else:
            row["raw_recording"]["decoded_frames"] = 2
        atomic_json(path, row)
    with pytest.raises(ValueError):
        tool().resume_evidence(args, lock, assets)


@pytest.mark.parametrize("fault", [None, "encode", "verify_cleanup", "old_report_changed"])
def test_resumed_controller_runs_only_offline_phases_and_stops_on_failure(
    resumable_case, tmp_path, monkeypatch, fault
):
    import os

    from scripts.development import blackbox_source as sources
    from scripts.development.blackbox_report import atomic_json

    module = tool()
    args, lock, _, current = resumable_case
    for name in (
        "input_root",
        "config",
        "worker_python",
        "ownership",
        "browser_cli",
        "browser_exe",
        "ffmpeg",
        "codec_root",
    ):
        path = tmp_path / name
        if name in {"input_root", "codec_root"}:
            path.mkdir()
        else:
            path.write_text(name)
        setattr(args, name, path)
    for name in ("codec.dll", "BINARY_LICENSE.txt"):
        (args.codec_root / name).write_text(name)
    assets = {
        name: sources.pin(getattr(args, name))
        for name in ("worker_python", "ownership", "browser_cli", "browser_exe", "ffmpeg", "config")
    }
    assets.update(
        codec_dll=sources.pin(args.codec_root / "codec.dll"),
        codec_license=sources.pin(args.codec_root / "BINARY_LICENSE.txt"),
    )
    context = args.resume_case_root.parent / "context.json"
    old_context = json.loads(context.read_bytes())
    old_context["pins"] = assets
    atomic_json(context, old_context)
    online_path = args.resume_case_root / "local-report.json"
    online = json.loads(online_path.read_bytes())
    online.update(
        screen={}, media=[], guidance_assessment={}, display_assessment={}, postgame_assessment={}
    )
    atomic_json(online_path, online)
    raw = json.loads(args.raw_report.read_bytes())
    raw["online_report_pin"] = sources.pin(online_path)
    atomic_json(args.raw_report, raw)
    original_summary = args.resume_case_root.parent / "summary.json"
    before = original_summary.read_bytes()
    args.output_root = tmp_path / "resumed-case"
    args.candidate = lock["commit"]
    args.source_lock_sha256 = sources.pin(current / "SOURCE-LOCK.json")["sha256"]
    monkeypatch.setattr(module, "ROOT", current)
    monkeypatch.setattr(
        module,
        "os",
        SimpleNamespace(
            name="nt", environ={"SystemRoot": str(tmp_path)}, pathsep=os.pathsep, path=os.path
        ),
    )
    monkeypatch.setattr(module, "load_ownership", lambda path: object())
    monkeypatch.setattr(
        module, "codec_environment", lambda path: dict(OPENH264_LIBRARY=str(path / "codec.dll"))
    )
    phases = []

    def transport(owner, arguments, root, environment, command, run_id, purpose, seconds):
        stage = command[command.index("--stage") + 1]
        phases.append(stage)
        assert seconds == 180
        assert command[3].endswith("s1_media_acceptance.py")
        result = dict(
            contract="s1-media-acceptance/v1",
            stage=stage,
            stage_pass=not (fault == "encode" and stage == "encode"),
            owned_run_id=run_id,
            online_run_id=online["owned_run_id"],
            source=dict(kind="locked", producer=lock["commit"], files=lock["files"]),
            capture_source=online["source"],
            online_report_pin=sources.pin(online_path),
            raw_report_pin=sources.pin(args.raw_report),
        )
        if stage == "verify":
            result.update(
                encoded_report_pin=sources.pin(
                    args.output_root / "media-encode/result/local-report.json"
                ),
                verification_complete=True,
                independent_original={},
            )
            if fault == "old_report_changed":
                original_summary.write_text("{}")
        output = root / "result"
        output.mkdir()
        atomic_json(output / "local-report.json", result)
        return dict(
            status="FAIL" if fault == "encode" else "PASS",
            worker_exit_code=1 if fault == "encode" else 0,
            cleanup_confirmed=not (fault == "verify_cleanup" and stage == "verify"),
        )

    monkeypatch.setattr(module, "run_owned_phase", transport)
    assert module.control(args) == (0 if fault is None else 2)
    summary = json.loads((args.output_root / "summary.json").read_bytes())
    assert summary["case_pass"] is (fault is None)
    assert phases == (["encode"] if fault == "encode" else ["encode", "verify"])
    assert summary["reused_online"] is True
    if fault != "old_report_changed":
        assert original_summary.read_bytes() == before
