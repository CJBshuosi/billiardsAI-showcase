"""One bounded Windows screen case under the existing pinned Job owner.

No installation, retry, generic command forwarding or positive result injection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.development import blackbox_source as sources  # noqa: E402
from scripts.development.blackbox_report import atomic_json  # noqa: E402
from scripts.development.blackbox_validation import (  # noqa: E402
    ValidationFailure,
    load_ownership,
    read_json,
)


def execution_budgets():
    # Explicit operation protection, not a product latency requirement. The
    # original hot-case/media allowance follows the separately bounded cold stage.
    return dict(
        cold_preparation_seconds=300,
        hot_request_seconds=8,
        case_seconds=240,
        owner_seconds=300 + 240,
        cleanup_seconds=10,
        offline_media_seconds=180,
    )


def child_command(args, output, run_id):
    command = [str(args.worker_python), "-I", "-B"]
    if getattr(args, "profile_imports", False):
        command += ["-X", "importtime"]
    command.append(str(Path(__file__).with_name("s1_screen_acceptance.py")))
    for name, value in (
        ("input-root", args.input_root),
        ("output-root", output),
        ("config", args.config),
        ("browser-cli", args.browser_cli),
        ("browser-exe", args.browser_exe),
        ("ffmpeg", args.ffmpeg),
        ("mode", args.mode),
        ("capture-seconds", execution_budgets()["case_seconds"]),
        ("owned-run-id", run_id),
    ):
        command += ["--" + name, str(value)]
    return command


def accept_report(owned, result, *, run_id, mode, lock):
    return (
        owned_complete(owned)
        and isinstance(result, dict)
        and result.get("contract") == "s1-online-screen-acceptance/v1"
        and result.get("owned_run_id") == run_id
        and result.get("mode") == mode
        and result.get("case_pass") is False
        and result.get("online_pass") is True
        and result.get("media_complete") is False
        and result.get("cleanup_confirmed") is True
        and result.get("source")
        == dict(kind="locked", producer=lock["commit"], files=lock["files"])
    )


def owned_complete(owned):
    return (
        owned.get("status") == "PASS"
        and owned.get("cleanup_confirmed") is True
        and owned.get("worker_exit_code") == 0
        and not owned.get("terminal_record_failed")
        and not owned.get("job_close_error")
    )


def media_command(args, case, output, stage, run_id, raw_report=None, encoded_report=None):
    command = [
        str(args.worker_python),
        "-I",
        "-B",
        str(Path(__file__).with_name("s1_media_acceptance.py")),
        "--case-root",
        str(case),
        "--output-root",
        str(output),
        "--stage",
        stage,
        "--owned-run-id",
        run_id,
    ]
    if raw_report is not None:
        command += ["--raw-report", str(raw_report)]
    if getattr(args, "capture_source_root", None) is not None:
        command += ["--capture-source-root", str(args.capture_source_root)]
    if encoded_report is not None:
        command += ["--encoded-report", str(encoded_report)]
    return command


def accept_media_report(
    owned,
    result,
    *,
    run_id,
    online_run_id,
    stage,
    source,
    online_report_pin,
    raw_report_pin=None,
    capture_source=None,
    encoded_report_pin=None,
):
    return (
        owned_complete(owned)
        and isinstance(result, dict)
        and result.get("contract") == "s1-media-acceptance/v1"
        and result.get("stage_pass") is True
        and result.get("stage") == stage
        and result.get("owned_run_id") == run_id
        and result.get("online_run_id") == online_run_id
        and result.get("source") == source
        and result.get("online_report_pin") == online_report_pin
        and (capture_source is None or result.get("capture_source") == capture_source)
        and (
            stage not in {"export", "encode", "verify"}
            or result.get("raw_report_pin") == raw_report_pin
        )
        and (
            stage != "verify"
            or (
                result.get("encoded_report_pin") == encoded_report_pin
                and encoded_report_pin is not None
                and result.get("verification_complete") is True
            )
        )
    )


def resume_evidence(args, lock, asset_pins):
    """Admit immutable online/raw evidence; never update the failed old summary."""
    from scripts.development.s1_media_acceptance import source_applicability

    case = sources.plain(args.resume_case_root)
    capture_root = sources.plain(args.capture_source_root)
    applicability = source_applicability(capture_root)
    if applicability["source"] != dict(kind="locked", producer=lock["commit"], files=lock["files"]):
        raise ValueError("resume execution source conflicts")
    pins = {}

    def read(path):
        path = sources.plain(path)
        before = sources.pin(path)
        value = read_json(path, limit=4 * 1024**2)
        if sources.pin(path) != before:
            raise ValueError("resume prerequisite changed while reading")
        pins[path] = before
        return value

    online = read(case / "local-report.json")
    owned = read(case.parent / "ownership.json")
    context = read(case.parent / "context.json")
    capture = applicability["capture_source"]
    capture_lock = dict(commit=capture["producer"], files=capture["files"])
    if (
        context.get("pins") != asset_pins
        or context.get("candidate") != capture["producer"]
        or context.get("mode") != args.mode
        or owned.get("run_id") != online.get("owned_run_id")
        or not accept_report(
            owned, online, run_id=online.get("owned_run_id"), mode=args.mode, lock=capture_lock
        )
    ):
        raise ValueError("resume online owner, environment or source conflicts")
    raw_path = sources.plain(args.raw_report)
    raw, raw_owner = read(raw_path), read(raw_path.parent.parent / "ownership.json")
    if (
        raw_owner.get("run_id") != raw.get("owned_run_id")
        or raw.get("owned_run_id") == online.get("owned_run_id")
        or not accept_media_report(
            raw_owner,
            raw,
            run_id=raw.get("owned_run_id"),
            online_run_id=online.get("owned_run_id"),
            stage="raw",
            source=capture,
            online_report_pin=pins[case / "local-report.json"],
        )
        or raw.get("frame_count") != online.get("input_manifest", {}).get("frame_count")
        or raw.get("raw_recording", {}).get("decoded_frames") != raw.get("frame_count")
    ):
        raise ValueError("resume raw audit or owner conflicts")
    # Retain the original failure identity without treating it as a successful phase.
    read(case.parent / "summary.json")
    return dict(
        case=case,
        online=online,
        owned=owned,
        raw=raw,
        raw_path=raw_path,
        pins=pins,
        applicability=applicability,
    )


def run_owned_phase(owner, args, root, environment, command, run_id, purpose, seconds):
    temporary = root / "tmp"
    if not temporary.exists():
        temporary.mkdir()
    env = dict(environment, TEMP=str(temporary), TMP=str(temporary))
    logs = [root / "local-stdout.txt", root / "local-stderr.txt"]

    def sleep(duration):
        time.sleep(duration)
        if any(p.exists() and p.stat().st_size > 4 * 1024**2 for p in logs):
            raise RuntimeError("owned stage log budget exceeded")

    with logs[0].open("xb") as stdout, logs[1].open("xb") as stderr:
        return owner.run_owned(
            owner.WindowsAPI(),
            lambda: subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env=env,
                cwd=root,
                close_fds=True,
            ),
            lambda stage, state: atomic_json(root / "ownership.json", dict(stage=stage, **state)),
            dict(run_id=run_id, purpose=purpose),
            ("S1-SCREEN-RELEASE " + run_id + "\n").encode(),
            deadline=seconds,
            cleanup_seconds=10,
            sleep=sleep,
        )


def codec_environment(root):
    from scripts.algorithm.recording_windows_entry import CODEC

    root = sources.plain(root)
    for filename, prefix in ((CODEC["filename"], ""), ("BINARY_LICENSE.txt", "license_")):
        if sources.pin(root / filename) != {
            "bytes": CODEC[prefix + "bytes"],
            "sha256": CODEC[prefix + "sha256"],
        }:
            raise ValueError("existing codec or license identity mismatch")
    return {"OPENH264_LIBRARY": str(root / CODEC["filename"])}


def control(args):
    if os.name != "nt":
        raise ValueError("screen controller requires Windows and the existing pinned owner")
    for name in (
        "input_root",
        "output_root",
        "config",
        "worker_python",
        "ownership",
        "browser_cli",
        "browser_exe",
        "ffmpeg",
        "codec_root",
    ):
        setattr(args, name, sources.plain(getattr(args, name)))
    lock = sources.verify(ROOT)
    if (
        lock["commit"] != args.candidate
        or sources.pin(ROOT / "SOURCE-LOCK.json")["sha256"] != args.source_lock_sha256
    ):
        raise ValueError("fixed candidate or source lock mismatch")
    resume = getattr(args, "resume_case_root", None) is not None
    if (
        any(getattr(args, name, None) is not None for name in ("capture_source_root", "raw_report"))
        != resume
    ):
        raise ValueError(
            "resume requires the original case, capture source and raw report together"
        )
    if resume and any(
        getattr(args, name, None) is None for name in ("capture_source_root", "raw_report")
    ):
        raise ValueError("resume capture source or raw report missing")
    codec = codec_environment(args.codec_root)
    assets = {
        name: getattr(args, name)
        for name in ("worker_python", "ownership", "browser_cli", "browser_exe", "ffmpeg", "config")
    }
    assets.update(
        codec_dll=Path(codec["OPENH264_LIBRARY"]),
        codec_license=args.codec_root / "BINARY_LICENSE.txt",
    )
    pins = {name: sources.pin(path) for name, path in assets.items()}
    owner = load_ownership(args.ownership)
    root = sources.plain(args.output_root)
    reused = resume_evidence(args, lock, pins) if resume else None
    if reused and (
        root.is_relative_to(reused["case"].parent) or reused["case"].parent.is_relative_to(root)
    ):
        raise ValueError("resume output must be separate from the original case")
    root.mkdir()
    (root / "tmp").mkdir()
    run_id = uuid4().hex
    budgets = execution_budgets()
    atomic_json(
        root / "context.json",
        dict(
            run_id=run_id,
            candidate=args.candidate,
            mode=args.mode,
            pins=pins,
            profile_imports=bool(getattr(args, "profile_imports", False)),
            execution_budgets=budgets,
        ),
    )
    # Keep required native profile/font locations; never pass auth or tool state.
    environment = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
        if key in os.environ
    }
    environment.update(
        **codec,
        TEMP=str(root / "tmp"),
        TMP=str(root / "tmp"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        NUMBA_CACHE_DIR=str(root / "tmp/numba"),
        MPLCONFIGDIR=str(root / "tmp/mpl"),
        PATH=os.pathsep.join(
            (str(args.ffmpeg.parent), str(Path(os.environ["SystemRoot"]) / "System32"))
        ),
    )
    if reused:
        owned, result, case = reused["owned"], reused["online"], reused["case"]
        run_id = result["owned_run_id"]
    else:
        case = root / "case"
        owned = run_owned_phase(
            owner,
            args,
            root,
            environment,
            child_command(args, case, run_id),
            run_id,
            "S1_REAL_SCREEN_CASE",
            budgets["owner_seconds"],
        )
        try:
            result = read_json(case / "local-report.json", limit=4 * 1024**2)
        except (ValueError, OSError, ValidationFailure):
            result = {}
    try:
        unchanged = sources.verify(ROOT) == lock and all(
            sources.pin(assets[name]) == identity for name, identity in pins.items()
        )
    except (ValueError, OSError):
        unchanged = False
    online_pass = unchanged and accept_report(
        owned,
        result,
        run_id=run_id,
        mode=args.mode,
        lock=dict(commit=result["source"]["producer"], files=result["source"]["files"])
        if reused
        else lock,
    )
    media_stages, media_reports = [], {}
    evidence = dict(
        case_root=str(case), capture_source_root=str(args.capture_source_root if reused else ROOT)
    )
    if online_pass:
        online_pin = sources.pin(case / "local-report.json")
        raw_report = reused["raw_path"] if reused else None
        raw_pin = sources.pin(raw_report) if raw_report else None
        encoded_report, encoded_pin = None, None
        if reused:
            raw = reused["raw"]
            media_reports["raw"] = raw
            media_stages.append(
                dict(
                    stage="raw",
                    run_id=raw["owned_run_id"],
                    stage_pass=True,
                    ownership_status="PASS",
                    owned_cleanup=True,
                    worker_exit_code=0,
                    report_pin=raw_pin,
                    reused=True,
                )
            )
            evidence["raw_report"] = str(raw_report)
        for stage in ("encode", "verify") if reused else ("raw", "encode", "verify"):
            phase_root = root / ("media-" + stage)
            phase_root.mkdir()
            phase_id = uuid4().hex
            atomic_json(
                root / "controller-progress.json",
                dict(
                    phase=stage,
                    online_pass=True,
                    case_pass=False,
                    online_run_id=run_id,
                    media_stages=media_stages,
                ),
            )
            phase_owned = run_owned_phase(
                owner,
                args,
                phase_root,
                environment,
                media_command(
                    args,
                    case,
                    phase_root / "result",
                    stage,
                    phase_id,
                    raw_report if stage != "raw" else None,
                    encoded_report if stage == "verify" else None,
                ),
                phase_id,
                "S1_OFFLINE_MEDIA_" + stage.upper(),
                budgets["offline_media_seconds"],
            )
            path = phase_root / "result/local-report.json"
            try:
                phase_result = read_json(path, limit=4 * 1024**2)
                report_pin = sources.pin(path)
            except (ValueError, OSError, ValidationFailure):
                phase_result, report_pin = {}, None
            accepted = accept_media_report(
                phase_owned,
                phase_result,
                run_id=phase_id,
                online_run_id=run_id,
                stage=stage,
                source=dict(kind="locked", producer=lock["commit"], files=lock["files"]),
                capture_source=result["source"],
                online_report_pin=online_pin,
                raw_report_pin=raw_pin,
                encoded_report_pin=encoded_pin,
            )
            media_stages.append(
                dict(
                    stage=stage,
                    run_id=phase_id,
                    stage_pass=accepted,
                    ownership_status=phase_owned.get("status"),
                    owned_cleanup=phase_owned.get("cleanup_confirmed") is True,
                    worker_exit_code=phase_owned.get("worker_exit_code"),
                    report_pin=report_pin,
                    failure=phase_result.get("failure"),
                )
            )
            if not accepted:
                break
            media_reports[stage] = phase_result
            evidence[stage + "_report"] = str(path)
            if stage == "raw":
                raw_report, raw_pin = path, report_pin
            elif stage == "encode":
                encoded_report, encoded_pin = path, report_pin
    try:
        unchanged = (
            unchanged
            and sources.verify(ROOT) == lock
            and all(sources.pin(assets[name]) == identity for name, identity in pins.items())
            and (not reused or resume_evidence(args, lock, pins) == reused)
        )
    except (ValueError, OSError):
        unchanged = False
    passed = (
        online_pass
        and unchanged
        and len(media_stages) == 3
        and all(stage["stage_pass"] for stage in media_stages)
    )
    summary = dict(
        contract="s1-owned-screen-case/v1",
        run_id=run_id,
        candidate=args.candidate,
        capture_candidate=result.get("source", {}).get("producer"),
        evidence=evidence,
        reused_online=bool(reused),
        source_applicability={
            key: value
            for key, value in (
                reused["applicability"] if reused else media_reports.get("verify", {})
            ).items()
            if key in {"execution_source_lock_pin", "capture_source_lock_pin", "changed_files"}
        },
        mode=args.mode,
        case_pass=passed,
        online_pass=online_pass,
        media_complete=passed,
        media_stages=media_stages,
        owned_cleanup=owned.get("cleanup_confirmed") is True
        and all(stage["owned_cleanup"] for stage in media_stages),
        ownership_status=owned.get("status"),
        worker_exit_code=owned.get("worker_exit_code"),
        identities_unchanged=unchanged,
        full_goal_complete=False,
        field_accuracy="OPEN",
        profile_imports=bool(getattr(args, "profile_imports", False)),
        execution_budgets=budgets,
    )
    if passed:
        summary.update(
            screen=result["screen"],
            independent_original=media_reports["verify"]["independent_original"],
            raw_recording=media_reports["raw"]["raw_recording"],
            media=result["media"],
            guidance_assessment=result["guidance_assessment"],
            display_assessment=result.get("display_assessment"),
            postgame_assessment=result.get("postgame_assessment"),
        )
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root",
        "output-root",
        "config",
        "worker-python",
        "ownership",
        "browser-cli",
        "browser-exe",
        "ffmpeg",
        "codec-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--mode", choices=("recommendation", "planner-failure"), required=True)
    parser.add_argument("--profile-imports", action="store_true")
    for name in ("resume-case-root", "capture-source-root", "raw-report"):
        parser.add_argument("--" + name, type=Path)
    return control(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
