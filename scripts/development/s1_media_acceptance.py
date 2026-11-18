"""Read-only offline audits of a completed, owned S1 online case.

The original case and database are never modified. Each media phase runs under
its own 180-second parent Job: raw/verify have 160 cooperative seconds and encode
has 170, covering the measured encode work without extending the parent budget.
Software results here do not assert model, display or real-table acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.development import blackbox_source as sources  # noqa: E402
from scripts.development.blackbox_report import atomic_json  # noqa: E402

REPORT_LIMIT = 4 * 1024**2
WORK_SECONDS = 160
ENCODE_WORK_SECONDS = 170
OFFLINE_SOURCE_PATHS = frozenset(
    {
        "scripts/development/s1_media_acceptance.py",
        "scripts/development/s1_screen_control.py",
    }
)


def _stage(diagnostics, name):
    return nullcontext() if diagnostics is None else diagnostics.stage(name)


def _deadline_check(continue_check=None, *, seconds=WORK_SECONDS):
    deadline = time.monotonic() + seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("independent media stage budget exhausted")
        if continue_check is not None:
            continue_check()

    return check


class _ObservedWriter:
    """Observe release while delegating all original codec operations."""

    def __init__(self, writer, diagnostics):
        self.writer, self.diagnostics = writer, diagnostics

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def release(self):
        attempted = False
        try:
            with _stage(self.diagnostics, "original_writer_release"):
                attempted = True
                return self.writer.release()
        finally:
            # Even a diagnostic entry failure must release the real encoder.
            if not attempted:
                self.writer.release()


def encode_original(frozen, output, frame_count, *, continue_check=None, diagnostics=None):
    """Encode/finalize only; the separate verify stage must decode the output."""
    import cv2

    from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval
    from billiards_ai.lab.mainline1_recording_media import (
        _encode_interval,
        _hash_media_file,
        _measure_timing,
    )

    path = sources.plain(frozen["raw_manifest_path"])
    raw = json.loads(path.read_bytes())
    interval = FrozenCaptureInterval(
        root=path.parent,
        source_epoch_id=frozen["source_epoch_id"],
        width=1920,
        height=1080,
        start_frame_id=0,
        end_frame_id=frame_count - 1,
        frame_count=frame_count,
        segment_sha256={segment["name"]: segment["sha256"] for segment in raw["segments"]},
    )
    check = _deadline_check(continue_check, seconds=ENCODE_WORK_SECONDS)
    with _stage(diagnostics, "original_timing"):
        timing = _measure_timing(interval, check)
    destination = Path(output) / "independent-original.mp4"
    if destination.exists():
        raise FileExistsError("independent original requires a fresh destination")

    def writer_factory(*args):
        writer = cv2.VideoWriter(*args)
        return writer if diagnostics is None else _ObservedWriter(writer, diagnostics)

    with _stage(diagnostics, "original_encode"):
        _encode_interval(
            interval,
            destination,
            timing["fps"],
            continue_check=check,
            writer_factory=writer_factory,
        )
    with _stage(diagnostics, "original_fsync"):
        check()
        # Windows _commit/fsync requires a writable descriptor. Do not truncate.
        with destination.open("r+b") as stream:
            os.fsync(stream.fileno())
        check()
    observed = _hash_media_file(destination, continue_check=check)
    return dict(
        bytes=observed["byte_size"],
        sha256=observed["content_sha256"],
        fps=timing["fps"],
        frame_count=frame_count,
    )


def _verify_original(destination, encoded, check, diagnostics):
    from billiards_ai.lab.mainline1_recording_media import _verify_mp4

    with _stage(diagnostics, "original_verify"):
        return _verify_mp4(
            destination,
            expected_frames=encoded["frame_count"],
            width=1920,
            height=1080,
            fps=encoded["fps"],
            continue_check=check,
        )


def export_original(frozen, output, frame_count, *, continue_check=None, diagnostics=None):
    """Compatibility entry; owned complete cases use separate encode/verify jobs."""
    check = _deadline_check(continue_check)
    encoded = encode_original(
        frozen, output, frame_count, continue_check=check, diagnostics=diagnostics
    )
    return _verify_original(Path(output) / "independent-original.mp4", encoded, check, diagnostics)


def _read_json(path, pins, *, limit=REPORT_LIMIT):
    path = sources.plain(path)
    before = sources.pin(path)
    if before["bytes"] > limit:
        raise ValueError("media input JSON exceeds budget")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or sources.pin(path) != before:
        raise ValueError("media input changed while reading")
    pins[path] = before
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("media input JSON must be an object")
    return value


def _source_identity(root=None):
    lock = sources.verify(ROOT if root is None else root)
    return dict(kind="locked", producer=lock["commit"], files=lock["files"])


def source_applicability(capture_source_root=None):
    """Reuse capture evidence only across the two explicit offline CLI changes."""
    capture_root = sources.plain(ROOT if capture_source_root is None else capture_source_root)
    execution_root = sources.plain(ROOT)
    execution, capture = _source_identity(execution_root), _source_identity(capture_root)
    if execution["files"].keys() != capture["files"].keys():
        raise ValueError("capture/execution source inventory differs")
    changed = sorted(
        name for name in execution["files"] if execution["files"][name] != capture["files"][name]
    )
    if not set(changed) <= OFFLINE_SOURCE_PATHS:
        raise ValueError("source applicability exceeds the two offline CLI paths")
    return dict(
        source=execution,
        capture_source=capture,
        execution_source_lock_pin=sources.pin(execution_root / "SOURCE-LOCK.json"),
        capture_source_lock_pin=sources.pin(capture_root / "SOURCE-LOCK.json"),
        changed_files=changed,
    )


def _require_owner(path, run_id, pins):
    owner = _read_json(path, pins)
    if (
        owner.get("status") != "PASS"
        or owner.get("cleanup_confirmed") is not True
        or type(owner.get("worker_exit_code")) is not int
        or owner["worker_exit_code"] != 0
        or owner.get("run_id") != run_id
        or owner.get("terminal_record_failed")
        or owner.get("job_close_error")
    ):
        raise ValueError("media prerequisite owner did not finish successfully")


def _validate_frozen(frozen, case_root, frame_count, pins, check):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.edge.route_assistance.records import recording_end_from_payload

    check()
    payload = dict(frozen)
    digest = payload.pop("manifest_sha256")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != digest:
        raise ValueError("frozen recording end identity changed")
    payload = recording_end_from_payload(payload)
    path = sources.plain(payload["raw_manifest_path"])
    if (
        not path.is_relative_to(case_root)
        or payload["contract"] != "recording-end/v1"
        or payload["ended_at_utc"] is None
        or payload["media_status"] != "RECORDED"
        or payload["first_frame_id"] != 0
        or payload["last_frame_id"] != frame_count - 1
        or (payload["width"], payload["height"]) != (1920, 1080)
        or payload["cutoff"] is None
        or payload["cutoff"]["frame_id"] != frame_count - 1
    ):
        raise ValueError("frozen recording identity, cutoff or path conflicts")
    raw = _read_json(path, pins, limit=8 * 1024**2)
    if pins[path]["sha256"] != payload["raw_manifest_sha256"]:
        raise ValueError("raw manifest changed")
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments or len(segments) > frame_count:
        raise ValueError("raw segment coverage invalid")
    for segment in segments:
        name = segment["name"]
        sources.safe_name(name)
        if Path(name).name != name:
            raise ValueError("raw segment path escaped recording")
        sources.plain(path.parent / name)
    # Existing pure validator checks epoch/session, cutoff time, contiguous
    # receipts and durable file identity, without opening or updating any DB.
    SQLiteFieldRecordStore._validate_recording_raw(
        payload, deadline=time.monotonic() + WORK_SECONDS
    )
    if sum(segment["frame_count"] for segment in segments) != frame_count:
        raise ValueError("raw segment frame count conflicts")
    check()
    return pins[path]


def _verify_raw(frozen, case_root, frame_count, check):
    from scripts.algorithm import run_recording_first_fullgame as existing

    original = existing.FrozenCaptureInterval

    class CheckedInterval:
        def __init__(self, *args, **kwargs):
            self.interval = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.interval, name)

        def frames(self):
            for frame in self.interval.frames(before_segment=check):
                check()
                yield frame

    # One isolated offline process owns this helper; patch restores even when
    # native decode, source validation or the cooperative deadline raises.
    with patch.object(existing, "FrozenCaptureInterval", CheckedInterval):
        return existing.verify_raw_recording(frozen, case_root, frame_count=frame_count)


def run_stage(
    case_root,
    output_root,
    stage,
    owned_run_id,
    raw_report=None,
    *,
    capture_source_root=None,
    encoded_report=None,
    continue_check=None,
    diagnostics=None,
):
    """Write a fresh local receipt; prerequisite failures never rescue old cases."""
    case_root, output = sources.plain(case_root), sources.plain(output_root)
    if output.is_relative_to(case_root) or case_root.is_relative_to(output):
        raise ValueError("offline output must be separate from the original case")
    output.mkdir()
    report = dict(
        contract="s1-media-acceptance/v1",
        stage=stage,
        stage_pass=False,
        owned_run_id=owned_run_id,
        online_run_id=None,
        source=None,
        online_report_pin=None,
        raw_manifest_pin=None,
        frame_count=None,
    )
    pins = {}
    check = _deadline_check(
        continue_check, seconds=ENCODE_WORK_SECONDS if stage == "encode" else WORK_SECONDS
    )
    try:
        if stage not in {"raw", "export", "encode", "verify"} or not re.fullmatch(
            r"[0-9a-f]{32}", owned_run_id
        ):
            raise ValueError("invalid owned media stage")
        with _stage(diagnostics, "media_inputs"):
            check()
            applicability = source_applicability(capture_source_root)
            report.update(applicability)
            online_path = case_root / "local-report.json"
            online = _read_json(online_path, pins)
            report["online_report_pin"] = pins[online_path]
            online_id = online.get("owned_run_id")
            if not isinstance(online_id, str) or not re.fullmatch(r"[0-9a-f]{32}", online_id):
                raise ValueError("invalid online owner identity")
            report["online_run_id"] = online_id
            if (
                online.get("contract") != "s1-online-screen-acceptance/v1"
                or online.get("online_pass") is not True
                or online.get("case_pass") is not False
                or online.get("media_complete") is not False
                or online.get("cleanup_confirmed") is not True
                or online.get("source") != report["capture_source"]
                or online_id == owned_run_id
            ):
                raise ValueError("online case is not a qualified media prerequisite")
            _require_owner(case_root.parent / "ownership.json", online_id, pins)
            manifest = online["input_manifest"]
            frames = manifest["frame_count"]
            if (
                type(frames) is not int
                or not 2 <= frames <= 12_000
                or type(manifest["fps"]) not in {int, float}
                or manifest["fps"] != 60
                or manifest["case"] not in {"inside", "outside"}
                or not isinstance(manifest["intervals"], list)
            ):
                raise ValueError("online input frame inventory invalid")
            report["frame_count"] = frames
            frozen = online["frozen"]
            report["raw_manifest_pin"] = _validate_frozen(frozen, case_root, frames, pins, check)
            if stage in {"export", "encode", "verify"}:
                if raw_report is None:
                    raise ValueError("media encoding/verification requires a successful raw report")
                raw_path = sources.plain(raw_report)
                raw = _read_json(raw_path, pins)
                report["raw_report_pin"] = pins[raw_path]
                if (
                    raw.get("contract") != report["contract"]
                    or raw.get("stage") != "raw"
                    or raw.get("stage_pass") is not True
                    or raw.get("online_run_id") != online_id
                    or raw.get("source") != report["capture_source"]
                    or raw.get("online_report_pin") != report["online_report_pin"]
                    or raw.get("raw_manifest_pin") != report["raw_manifest_pin"]
                    or raw.get("frame_count") != frames
                    or raw.get("raw_recording", {}).get("decoded_frames") != frames
                    or not re.fullmatch(r"[0-9a-f]{32}", str(raw.get("owned_run_id")))
                    or raw["owned_run_id"] in {online_id, owned_run_id}
                ):
                    raise ValueError("raw prerequisite identity or success conflicts")
                _require_owner(raw_path.parent.parent / "ownership.json", raw["owned_run_id"], pins)
            elif raw_report is not None:
                raise ValueError("raw stage cannot consume a raw report")
            if stage == "verify":
                if encoded_report is None:
                    raise ValueError("verify requires a successful encoded report")
                encoded_path = sources.plain(encoded_report)
                encoded = _read_json(encoded_path, pins)
                report["encoded_report_pin"] = pins[encoded_path]
                if (
                    encoded.get("contract") != report["contract"]
                    or encoded.get("stage") != "encode"
                    or encoded.get("stage_pass") is not True
                    or encoded.get("verification_complete") is not False
                    or any(encoded.get(key) != value for key, value in applicability.items())
                    or encoded.get("online_run_id") != online_id
                    or encoded.get("online_report_pin") != report["online_report_pin"]
                    or encoded.get("raw_manifest_pin") != report["raw_manifest_pin"]
                    or encoded.get("raw_report_pin") != report["raw_report_pin"]
                    or encoded.get("frame_count") != frames
                    or not re.fullmatch(r"[0-9a-f]{32}", str(encoded.get("owned_run_id")))
                    or encoded["owned_run_id"] in {online_id, raw["owned_run_id"], owned_run_id}
                    or encoded.get("failure")
                ):
                    raise ValueError("encoded prerequisite source, identity or success conflicts")
                _require_owner(
                    encoded_path.parent.parent / "ownership.json", encoded["owned_run_id"], pins
                )
                info = encoded.get("encoded_original")
                if (
                    not isinstance(info, dict)
                    or set(info) != {"bytes", "sha256", "fps", "frame_count"}
                    or type(info["bytes"]) is not int
                    or info["bytes"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", str(info["sha256"]))
                    or type(info["frame_count"]) is not int
                    or info["frame_count"] != frames
                    or type(info["fps"]) not in {int, float}
                    or not math.isfinite(info["fps"])
                    or not 1 <= info["fps"] <= 240
                ):
                    raise ValueError("encoded original inventory invalid")
                video = sources.plain(encoded_path.parent / "independent-original.mp4")
                pins[video] = sources.pin(video)
                if pins[video] != {key: info[key] for key in ("bytes", "sha256")}:
                    raise ValueError("encoded original file changed")
                report["encoded_original"] = info
            elif encoded_report is not None:
                raise ValueError("only verify can consume an encoded report")
        if stage == "raw":
            with _stage(diagnostics, "raw_validation"):
                report["raw_recording"] = _verify_raw(frozen, case_root, frames, check)
                if report["raw_recording"]["decoded_frames"] != frames:
                    raise ValueError("raw decode frame coverage conflicts")
        elif stage == "encode":
            report["verification_complete"] = False
            report["encoded_original"] = encode_original(
                frozen,
                output,
                frames,
                continue_check=check,
                diagnostics=diagnostics,
            )
        elif stage == "verify":
            report["independent_original"] = _verify_original(video, info, check, diagnostics)
            report["verification_complete"] = True
        else:
            report["independent_original"] = export_original(
                frozen,
                output,
                frames,
                continue_check=check,
                diagnostics=diagnostics,
            )
            report["verification_complete"] = True
        with _stage(diagnostics, "media_source_after"):
            check()
            if source_applicability(capture_source_root) != applicability:
                raise ValueError("media source changed during verification")
            for path, identity in pins.items():
                check()
                if sources.pin(path) != identity:
                    raise ValueError("media prerequisite changed during verification")
            _validate_frozen(frozen, case_root, frames, pins, check)
        report["stage_pass"] = True
    except Exception as exc:
        report["failure"] = dict(type=type(exc).__name__, message=str(exc))
    atomic_json(output / "local-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("raw", "export", "encode", "verify"), required=True)
    parser.add_argument("--owned-run-id", required=True)
    parser.add_argument("--raw-report", type=Path)
    parser.add_argument("--capture-source-root", type=Path)
    parser.add_argument("--encoded-report", type=Path)
    args = parser.parse_args()
    from scripts.development.s1_screen_acceptance import await_owned_release

    await_owned_release(args.owned_run_id)
    from scripts.development.s1_probe_diagnostics import ProbeDiagnostics

    # The report directory is created by run_stage; diagnostic receipts may be
    # written only after that fresh output has been admitted.
    with ProbeDiagnostics(args.output_root, args.owned_run_id) as diagnostics:
        report = run_stage(
            args.case_root,
            args.output_root,
            args.stage,
            args.owned_run_id,
            args.raw_report,
            capture_source_root=args.capture_source_root,
            encoded_report=args.encoded_report,
            diagnostics=diagnostics,
        )
    return 0 if report["stage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
