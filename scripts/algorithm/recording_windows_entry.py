"""One-shot R4 Windows entry; product imports wait for owned-Job release."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.algorithm.recording_packet import pin_file, plain, verify_packet, write_json

WORK_SECONDS = 900
CLEANUP_SECONDS = 10
BROWSER_WINDOW_SECONDS = 90
LOG_LIMIT = 4 * 1024**2
REPORT_LIMIT = 256 * 1024
CODEC = dict(
    filename="openh264-2.5.0-win64.dll",
    bytes=986712,
    sha256="bf91c488638b32448f7d89611f6bb9c232dce82975d7d2827c78f78fc2a7fb2b",
    license_bytes=4612,
    license_sha256="bd9f363c5ea11ef723d0304cddacb5273c43c0e1194097c7a045d05273635418",
)
DOCUMENTS = Path("C:/Users/developer/Documents")
_ROOT_NAME = r"billiardsAI-recording-r4-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"


def checked_root(root, manifest_sha):
    root = plain(root)
    if os.name != "nt" or root.parent != DOCUMENTS or not re.fullmatch(_ROOT_NAME, root.name):
        raise ValueError("fresh expected Windows Documents GUID root required")
    prepared_path = plain(root / "PREPARED.json")
    if prepared_path.stat().st_size > 16 * 1024:
        raise ValueError("prepared record too large")
    prepared = json.loads(prepared_path.read_bytes())
    if prepared.get("manifest_sha256") != manifest_sha:
        raise ValueError("prepared packet identity mismatch")
    packet = root / "packet"
    if Path(__file__).resolve() != packet / "entry.py":
        raise ValueError("only the exact packet entry may execute")
    return root, verify_packet(packet, manifest_sha)


def _codec_environment(root):
    """Opt in only to a separately delivered, reviewed codec for this prepared packet."""
    selection_path = plain(root / "codec-enabled.json")
    if not selection_path.exists():
        return {}
    try:
        if selection_path.stat().st_size > 16384:
            raise ValueError("codec selection exceeds limit")
        selection = json.loads(selection_path.read_bytes())
        prepared_path = plain(root / "PREPARED.json")
        if prepared_path.stat().st_size > 16384:
            raise ValueError("prepared codec binding exceeds limit")
        prepared = json.loads(prepared_path.read_bytes())
        if (
            not isinstance(selection, dict)
            or not isinstance(prepared, dict)
            or set(selection) != {"contract", "enabled", "packet_manifest_sha256"}
            or selection["contract"] != "recording-codec-selection/v1"
            or type(selection["enabled"]) is not bool
            or not re.fullmatch("[0-9a-f]{64}", str(selection["packet_manifest_sha256"]))
            or selection["packet_manifest_sha256"] != prepared.get("manifest_sha256")
        ):
            raise ValueError("codec selection does not bind this prepared packet")
        if not selection["enabled"]:
            return {}
        dll = plain(root / "codec" / CODEC["filename"])
        license_path = plain(root / "codec/BINARY_LICENSE.txt")
        if pin_file(dll) != {"bytes": CODEC["bytes"], "sha256": CODEC["sha256"]} or (
            pin_file(license_path)
            != {"bytes": CODEC["license_bytes"], "sha256": CODEC["license_sha256"]}
        ):
            raise ValueError("reviewed external codec or license changed")
    except (OSError, TypeError) as exc:
        raise ValueError("reviewed external codec is missing or unsafe") from exc
    return {"OPENH264_LIBRARY": str(dll)}


def _codec_preflight(root):
    """Acceptance-only two-frame check inside the existing owned worker budget."""
    selected = _codec_environment(root)
    if not selected:
        return None
    if os.environ.get("OPENH264_LIBRARY") != selected["OPENH264_LIBRARY"]:
        raise ValueError("codec worker environment differs from prepared selection")
    import cv2
    import numpy as np

    from billiards_ai.lab.mainline1_recording_media import _fsync_file

    probe = plain(root / "codec/encoder-preflight.mp4")
    result = dict(
        contract="recording-codec-preflight/v1",
        status="FAIL",
        failure_code="H264_ENCODER_UNAVAILABLE",
        decoded_frames=0,
    )
    writer = capture = None
    created = False
    try:
        with probe.open("xb"):
            created = True
        writer = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*"avc1"), 60, (64, 64))
        if writer.isOpened():
            for _ in range(2):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            writer = None
            result["failure_code"] = "OUTPUT_FLUSH_FAILED"
            _fsync_file(probe)
            result["failure_code"] = "H264_READBACK_FAILED"
            capture = cv2.VideoCapture(str(probe))
            while result["decoded_frames"] < 3:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.shape != (64, 64, 3):
                    break
                result["decoded_frames"] += 1
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).lower()
            if result["decoded_frames"] == 2 and codec in {"avc1", "h264", "x264"}:
                result.update(status="PASS", failure_code=None)
    except (OSError, ValueError, cv2.error):
        if result["failure_code"] != "OUTPUT_FLUSH_FAILED":
            result["failure_code"] = "CODEC_PREFLIGHT_FAILED"
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if created:
            probe.unlink(missing_ok=True)
    write_json(plain(root / "codec/preflight.json"), result)
    return result["status"] == "PASS"


def _include_codec_summary(root, result):
    path = plain(root / "codec/preflight.json")
    if not path.exists():
        return
    try:
        if path.stat().st_size > 16384:
            raise ValueError("codec result exceeds limit")
        data = json.loads(path.read_bytes())
        if (
            not isinstance(data, dict)
            or not {"contract", "status", "failure_code", "decoded_frames"} <= data.keys()
            or data.get("contract") != "recording-codec-preflight/v1"
            or data.get("status") not in {"PASS", "FAIL"}
            or type(data.get("decoded_frames")) is not int
            or not 0 <= data["decoded_frames"] <= 3
            or data.get("failure_code")
            not in {
                None,
                "H264_ENCODER_UNAVAILABLE",
                "H264_READBACK_FAILED",
                "CODEC_PREFLIGHT_FAILED",
                "OUTPUT_FLUSH_FAILED",
            }
            or (
                data["status"] == "PASS"
                and (data["decoded_frames"] != 2 or data["failure_code"] is not None)
            )
        ):
            raise ValueError("invalid codec result")
        result["codec_preflight"] = {
            key: data[key] for key in ("status", "failure_code", "decoded_frames")
        }
        if data["status"] == "FAIL":
            result.update(product_pass=False, failure_type="CODEC_PREFLIGHT_FAILED")
    except (OSError, ValueError, TypeError, AttributeError):
        result.update(product_pass=False, failure_type="CODEC_PREFLIGHT_RESULT_INVALID")


def child_environment(root):
    system_root = os.environ.get("SYSTEMROOT")
    if not system_root:
        raise ValueError("SYSTEMROOT is required")
    profile = root / "profile"
    paths = dict(
        TEMP=root / "temp",
        TMP=root / "temp",
        USERPROFILE=profile,
        APPDATA=profile / "AppData/Roaming",
        LOCALAPPDATA=profile / "AppData/Local",
    )
    for path in paths.values():
        plain(path).mkdir(parents=True, exist_ok=True)
    return {
        **_codec_environment(root),
        **{name: str(path) for name, path in paths.items()},
        "SYSTEMROOT": system_root,
        "WINDIR": system_root,
        "OS": "Windows_NT",
        "PATH": os.pathsep.join(
            (str(root / "packet/runtime"), str(Path(system_root) / "System32"))
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class _LogBudget:
    def __init__(self, paths, limit):
        self.paths, self.limit, self.cleaning = paths, limit, False

    def exceeded(self):
        return any(path.stat().st_size > self.limit for path in self.paths)

    def sleep(self, seconds):
        if not self.cleaning and self.exceeded():
            raise RuntimeError("owned log budget exceeded")
        time.sleep(seconds)


class _CleanupAPI:
    def __init__(self, api, budget):
        self.api, self.budget = api, budget

    def __getattr__(self, name):
        return getattr(self.api, name)

    def terminate(self, job):
        self.budget.cleaning = True
        return self.api.terminate(job)


def summarize(report, ownership, manifest):
    """Only a bounded synthetic-result whitelist may be returned to the Mac."""
    result = dict(
        contract="recording-first-windows-summary/v1",
        product_pass=False,
        model_pass=False,
        field_pass=False,
        browser_pass=False,
        source_commit=manifest["source_commit"],
        input_sha256=manifest["input_sha256"],
        cleanup_confirmed=ownership.get("cleanup_confirmed") is True,
        ownership_status=ownership.get("status"),
        worker_exit_code=ownership.get("worker_exit_code"),
        native_execution_authorization_inferred=False,
    )
    if not isinstance(report, dict):
        result["failure_type"] = "REPORT_UNAVAILABLE"
        return result
    if report.get("source_commit") != manifest["source_commit"] or (
        report.get("source_sha256") != manifest["input_sha256"]
    ):
        result["failure_type"] = "REPORT_IDENTITY_MISMATCH"
        return result

    def number(value):
        if type(value) is int:
            return value if 0 <= value < 2**63 else None
        return value if type(value) is float and math.isfinite(value) else None

    for key in (
        "emitted_frames",
        "capture_seconds",
        "capture_stop_seconds",
        "processing_seconds",
        "encoding_seconds",
        "raw_verification_seconds",
        "elapsed_seconds",
        "sampled_queue_high_water",
        "sampled_detection_backlog_high_water",
        "sampled_writer_high_water",
        "peak_rss_bytes",
        "output_bytes",
        "analysis_jobs",
        "saved_outbox",
    ):
        if key in report:
            result[key] = number(report[key])
    result["windows_run"] = report.get("windows_run") is True
    media = report.get("media")
    result["clip_count"] = len(media) if isinstance(media, list) else 0
    raw = report.get("raw_recording")
    result["raw_recording"] = (
        {
            key: number(raw.get(key))
            for key in (
                "decoded_frames",
                "segments",
                "first_frame_id",
                "last_frame_id",
                "width",
                "height",
                "segment_bytes",
            )
        }
        if isinstance(raw, dict)
        else {}
    )
    for name in ("capture_health", "consumer_health"):
        health = report.get(name)
        result[name] = (
            {
                key: number(health.get(key))
                for key in (
                    "received_frames",
                    "measured_fps",
                    "maximum_gap_ms",
                    "timestamp_reversals",
                    "unaccounted_frame_gaps",
                    "accounted_frame_gaps",
                    "decode_failures",
                    "ring_write_failures",
                    "control_drifts",
                )
            }
            if isinstance(health, dict)
            else {}
        )
        result[name]["reason_present"] = (
            bool(health.get("reason")) if isinstance(health, dict) else True
        )
    differences = report.get("differences")
    result["differences"] = (
        [
            text
            if isinstance(text, str)
            and re.fullmatch(r"(?:shot_count:\d+!=\d+|(?:shot_\d+|shot-\d+|None):[a-z_]+)", text)
            else "INVALID_DIFFERENCE"
            for text in differences[:64]
        ]
        if isinstance(differences, list)
        else ["DIFFERENCES_UNAVAILABLE"]
    )
    failure = report.get("failure")
    failure_type = failure.get("type") if isinstance(failure, dict) else None
    result["failure_type"] = (
        failure_type
        if isinstance(failure_type, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,79}", failure_type)
        else None
    )
    review = report.get("review")
    clips = review.get("clips") if isinstance(review, dict) else None
    result["media_errors"] = []
    if isinstance(clips, list):
        for clip in clips[:28]:
            if not isinstance(clip, dict) or not clip.get("error_code"):
                continue
            shot, code = clip.get("shot_id"), clip["error_code"]
            if not isinstance(shot, str) or not re.fullmatch(r"shot-\d{4,8}", shot):
                shot = "UNKNOWN_SHOT"
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code):
                code = "INVALID_MEDIA_ERROR"
            result["media_errors"].append(dict(shot_id=shot, error_code=code))
    shot_ids = (
        {
            item.get("shot_id")
            for item in media
            if isinstance(item, dict) and isinstance(item.get("shot_id"), str)
        }
        if isinstance(media, list)
        else set()
    )
    result["product_pass"] = bool(
        ownership.get("status") == "PASS"
        and result["cleanup_confirmed"]
        and report.get("cleanup_confirmed") is True
        and report.get("product_pass") is True
        and report.get("windows_run") is True
        and not report.get("failure")
        and result.get("emitted_frames") == 15880
        and result["clip_count"] == 28
        and shot_ids == {f"shot-{i:04d}" for i in range(1, 29)}
        and result["capture_health"].get("received_frames") == 15880
        and result["consumer_health"].get("received_frames") == 15880
        and not result["capture_health"]["reason_present"]
        and not result["consumer_health"]["reason_present"]
        and result["raw_recording"].get("decoded_frames") == 15880
        and result["raw_recording"].get("first_frame_id") == 0
        and result["raw_recording"].get("last_frame_id") == 15879
        and result.get("analysis_jobs") == result.get("saved_outbox") == 0
        and not result["differences"]
        and not result["media_errors"]
    )
    return result


def launch(root, manifest_sha, manifest):
    # ownership.py is an unchanged, manifest-pinned reviewed predecessor module.
    from ownership import WindowsAPI, run_owned

    write_json(
        root / "once.json", dict(contract="recording-r4-once/v1", manifest_sha256=manifest_sha)
    )
    output = root / "output"
    output.mkdir()
    log_paths = (root / "worker-stdout.txt", root / "worker-stderr.txt")
    environment = child_environment(root)
    runtime = root / "packet/runtime/python.exe"
    command = [
        str(runtime),
        "-I",
        "-B",
        str(root / "packet/entry.py"),
        "worker",
        str(root),
        manifest_sha,
    ]

    def persist(stage, state):
        write_json(output / "ownership.json", dict(stage=stage, **state), replace=True)

    budget = _LogBudget(log_paths, LOG_LIMIT)

    with log_paths[0].open("xb") as stdout, log_paths[1].open("xb") as stderr:

        def spawn():
            return subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                cwd=root / "packet",
                close_fds=True,
            )

        ownership = run_owned(
            _CleanupAPI(WindowsAPI(), budget),
            spawn,
            persist,
            dict(
                contract="recording-r4-owned-run/v1",
                manifest_sha256=manifest_sha,
                work_seconds=WORK_SECONDS,
                cleanup_seconds=CLEANUP_SECONDS,
            ),
            ("R4-RELEASE " + manifest_sha + "\n").encode("ascii"),
            deadline=WORK_SECONDS,
            cleanup_seconds=CLEANUP_SECONDS,
            sleep=budget.sleep,
        )
    report_path = plain(root / "recording/report.json")
    report = None
    if report_path.is_file() and report_path.stat().st_size <= REPORT_LIMIT:
        try:
            report = json.loads(report_path.read_bytes())
        except (ValueError, OSError):
            pass
    result = summarize(report, ownership, manifest)
    _include_codec_summary(root, result)
    try:
        if budget.exceeded():
            result.update(product_pass=False, failure_type="LOG_BUDGET_EXCEEDED")
    except OSError:
        result.update(product_pass=False, failure_type="LOG_BUDGET_UNAVAILABLE")
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["product_pass"] else 1


def worker(root, manifest_sha):
    if _codec_preflight(root) is False:
        return 1
    from scripts.algorithm import run_recording_first_fullgame

    return run_recording_first_fullgame.main(
        [
            "--input-root",
            str(root / "packet/input"),
            "--output-root",
            str(root / "recording"),
            "--capture-seconds",
            "360",
            "--encoding-seconds",
            "600",
            "--browser-seconds",
            str(BROWSER_WINDOW_SECONDS),
        ]
    )


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3 or args[0] not in {"launch", "worker"}:
        raise ValueError("explicit role, prepared root and manifest SHA required")
    if args[0] == "worker":
        expected = ("R4-RELEASE " + args[2] + "\n").encode("ascii")
        if sys.stdin.buffer.readline(200) != expected:
            raise ValueError("worker did not receive exact owned release")
    root, manifest = checked_root(Path(args[1]), args[2])
    return launch(root, args[2], manifest) if args[0] == "launch" else worker(root, args[2])


if __name__ == "__main__":
    raise SystemExit(main())
