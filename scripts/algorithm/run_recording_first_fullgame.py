#!/usr/bin/env python3
"""Local synthetic recording acceptance; never a cloud or natural-terminal episode."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import secrets
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope  # noqa: E402
from billiards_ai.lab.field_assistance import server as service  # noqa: E402
from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval  # noqa: E402
from billiards_ai.lab.mainline1_recording_media import (  # noqa: E402
    verify_recording_playback_artifact,
)
from billiards_ai.lab.mainline1_scenario_capture import ScenarioVideoCaptureBackend  # noqa: E402
from scripts.algorithm import serve_mainline1_blackbox_field as blackbox  # noqa: E402

SOURCE_SHA256 = "6a234abb4d9df0efea2c23f21bfe138025e5bd8c291e33229af9e48ed37a9453"
FRAME_COUNT = 15880
_INPUT_PINS = {
    "input.avi": SOURCE_SHA256,
    "source/configs/algorithm/shot-local.yaml": (
        "91a956cee485fa5160b1f7f6cb352fd8b989216ad621b4aa9cf247920deda673"
    ),
    "calibration/capture-setup.json": (
        "5b522ff6b0c922c59b5e1bf338b374bd0a4ac7baae67ebb612fb98fe14352164"
    ),
    "build-evidence/oracle-interval-binding.json": (
        "5173539cc8dfce0eca2bf89f9e51c90c8da968134c5b8d70c368f8888bcb0de0"
    ),
}
_CALIBRATION = (
    "calibration/calibration-registry/"
    "6bb0db76adc0123e8f04d6bb9a3a38aa515a04dedab6ad435fdcc2cc9071cf73"
)
_INPUT_PINS.update(
    {
        "build-evidence/actual-video-review.json": (
            "741bd2e27ca95fec220f74440672d1215fcb86de89b34cafd81c1a8161fc18a3"
        ),
        "build-evidence/calibration-compatibility.json": (
            "d96d46457c196db2a67e48b8bb9e0535e2f7d01f14e33a829885f993f9f7e27b"
        ),
        f"{_CALIBRATION}/calibration.json": (
            "3ad626439971ab1a70fa02da8a057bc3ab145c00592b76a2a882d073e2a0c2aa"
        ),
        f"{_CALIBRATION}/manifest.json": (
            "7d2222180e7c2c2c2d6d00b1f976a1149f56756b4b0b6ede08fabd9294bf46f7"
        ),
        f"{_CALIBRATION}/reference-frame.jpg": (
            "f56a70e4756153fcb5072a3d04c7c47a43bbbead7a7726a6187f7a976fa14753"
        ),
        "calibration/calibration-registry/registry.json": (
            "65ee11e5ebe9e17339c7287708055d39edafad123beaff52aafbf6c98837f718"
        ),
        "input-oracle.json": "a385fb9d90c5425e8bdce1ef074bac3e84155ac6f2d0cbfaace5c19d1c2c2915",
        "physics-manifest.json": "466279b9981227bf2645080970adac8d22068d927f382197c7dc2b974e6d9305",
    }
)


def _plain(path):
    path = Path(path).absolute()
    if any(part.casefold() == ".project-archive" or part == ".." for part in path.parts) or any(
        p.is_symlink() or p.is_junction() for p in (path, *path.parents)
    ):
        raise ValueError("unsafe acceptance path")
    return path


def _validate_inputs(root):
    root = _plain(root)
    report = json.loads((root / "input-preflight.json").read_bytes())
    if report.get("contract") != "recording-first-r4-input-preflight/v1":
        raise ValueError("input preflight contract mismatch")
    if set(report["selected_files"]) != set(_INPUT_PINS) or any(
        report["selected_files"][name]["sha256"] != digest for name, digest in _INPUT_PINS.items()
    ):
        raise ValueError("reviewed input/config/comparator identity changed")
    for name, expected in report["selected_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("input inventory path escape")
        path = _plain(root / name)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError("input inventory changed")
    if report["selected_files"]["input.avi"]["sha256"] != SOURCE_SHA256:
        raise ValueError("only the reviewed synthetic whole-game input is admitted")
    return root, report


def _source_revision():
    """Bind portable source to its reviewed inventory without requiring Git."""
    lock_path = _ROOT / "SOURCE-LOCK.json"
    if not lock_path.exists():
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    lock = json.loads(_plain(lock_path).read_bytes())
    revision = lock.get("commit")
    if (
        lock.get("contract") != "recording-first-source-lock/v1"
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
        or not isinstance(lock.get("files"), dict)
    ):
        raise ValueError("frozen source identity invalid")
    files = set()
    for path in _ROOT.rglob("*"):
        _plain(path)
        if path.is_file() and path != lock_path:
            files.add(path.relative_to(_ROOT).as_posix())
    if files != set(lock["files"]):
        raise ValueError("frozen source inventory changed")
    for name, pin in lock["files"].items():
        path = _plain(_ROOT / name)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("frozen source path escape")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != pin["bytes"] or digest != pin["sha256"]:
            raise ValueError("frozen source content changed")
    return revision


def _hold_for_browser(server, report, output, seconds):
    """Expose an optional bounded local browser window; never assert browser success."""
    if seconds <= 0 or not report["product_pass"]:
        return
    ready = dict(
        contract="recording-first-browser-ready/v1",
        url=f"http://127.0.0.1:{server.server_address[1]}/screen",
        session_id=report["session_id"],
        postgame_manifest_sha256=report["review"]["postgame_manifest_sha256"],
    )
    with (output / "browser-ready.json").open("x", encoding="utf-8") as stream:
        json.dump(ready, stream)
        stream.write("\n")
    report["browser_window_seconds"] = seconds
    print(json.dumps(dict(stage="browser_ready", **ready)), flush=True)
    deadline = time.monotonic() + seconds
    while (remaining := deadline - time.monotonic()) > 0:
        threading.Event().wait(min(1, remaining))


def compare_intervals(evidence, expected):
    """External comparison only; no expected value is passed into the runtime."""
    differences = []
    if len(evidence) != len(expected):
        differences.append(f"shot_count:{len(evidence)}!={len(expected)}")
    matches = [[] for _ in expected]
    for shot in evidence:
        boundary = shot.get("boundary") or (shot.get("media") or {}).get("boundary")
        if not boundary:
            differences.append(f"{shot.get('shot_id')}:boundary_missing")
            continue
        hits = [
            i
            for i, truth in enumerate(expected)
            if (
                boundary["start_frame_id"] <= truth["last_motion_frame"]
                and boundary["end_frame_id"] >= truth["motion_start_frame"]
            )
        ]
        if not hits:
            differences.append(f"{shot.get('shot_id')}:extra_boundary")
        elif len(hits) > 1:
            differences.append(f"{shot.get('shot_id')}:merged_intervals")
        else:
            matches[hits[0]].append(shot)
    for index, (shots, truth) in enumerate(zip(matches, expected, strict=True), 1):
        if len(shots) != 1:
            differences.append(f"shot_{index}:missing" if not shots else f"shot_{index}:split")
            continue
        shot = shots[0]
        media = shot.get("media") or {}
        if shot.get("media_status") != "RECORDED":
            differences.append(f"shot_{index}:not_recorded")
        elif not (
            media["start_frame_id"] <= truth["motion_start_frame"]
            and media["end_frame_id"] >= truth["last_motion_frame"]
        ):
            differences.append(f"shot_{index}:motion_not_contained")
    return differences


def _request(server, path, *, payload=None, token=None, timeout_seconds=30):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("acceptance HTTP budget is invalid")
    budget = min(30, timeout_seconds)
    deadline = time.monotonic() + budget
    connection = http.client.HTTPConnection(*server.server_address, timeout=budget)
    transport = timer = None
    try:
        connection.connect()
        transport = connection.sock
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("acceptance HTTP budget exhausted")
        transport.settimeout(remaining)

        def expire():
            # HTTP/1.0 can transfer socket ownership to the response. Retain the
            # original transport so split headers/body cannot renew the budget.
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        headers = {} if token is None else {"X-Control-Token": token}
        body = None if payload is None else json.dumps(payload)
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request("GET" if payload is None else "POST", path, body, headers)
        with connection.getresponse() as response:
            raw = response.read(2 * 1024**2 + 1)
            if time.monotonic() >= deadline:
                raise TimeoutError("acceptance HTTP budget exhausted")
            if response.status != 200 or len(raw) > 2 * 1024**2:
                raise RuntimeError(f"acceptance HTTP {response.status}: {raw[:300]!r}")
            return json.loads(raw)
    except Exception as exc:
        if time.monotonic() >= deadline:
            raise TimeoutError("acceptance HTTP budget exhausted") from exc
        raise
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        connection.close()
        if transport is not None:
            transport.close()


def _capture_sample(state, runtime, *, started):
    """Read checkpoint first so the later committed watermark cannot trail it."""
    journal = runtime._runtime._recording_processing
    checkpoint = journal.state["checkpoint"]
    processed = None if checkpoint is None else checkpoint["last_frame_id"]
    progress = runtime.recording_read_progress
    if progress is None or progress.session_id != state.session_id:
        raise RuntimeError("whole input observer requires original disk recording")
    first = runtime._recording_store.first_recorded_frame_id
    committed = progress.committed_frame_id
    backlog = (
        0 if committed is None else committed - (processed if processed is not None else first - 1)
    )
    if backlog < 0:
        raise RuntimeError("processing checkpoint exceeds committed recording")
    frame = runtime.current_capture_frame
    health = runtime.recording_writer_health
    failure = runtime.failure_reason or runtime.recording_failure_reason or health.reason
    if failure:
        raise RuntimeError(failure)
    return dict(
        seconds=round(time.monotonic() - started, 3),
        frame=None if frame is None else frame.capture_frame_id,
        committed_frame_id=committed,
        delivered_frame_id=progress.last_delivered_frame_id,
        processed_frame_id=processed,
        detection_backlog_frames=backlog,
        buffered_frames=health.buffered_frames,
        pending_chunks=health.pending_chunks,
        shots=len(state.record_store.list_shots(state.session_id)),
        peak_rss_bytes=_rss(),
    )


def _wait_for_background_completion(server, state, receipt, *, deadline, report, progress=None):
    """Observe the original stop through detection and media; never supplies facts."""
    session = state.session_id
    if (
        receipt.get("contract") != "recording-capture-end/v1"
        or receipt.get("capture_status") != "STOPPED"
        or receipt.get("session_id") != session
        or receipt.get("source_epoch_id") != state.projection_source.source_epoch_id
    ):
        raise ValueError("capture stop receipt identity mismatch")
    started = time.monotonic()
    final_seen = None
    last_sample = float("-inf")
    last_signature = None
    last_change = started
    last_checkpoint = started
    last_emitted = None
    final_sources_checked = False
    samples = report.setdefault("processing_samples", [])
    while True:
        observed = time.monotonic()
        if deadline is not None and observed >= deadline:
            raise TimeoutError("whole input background processing/encoding budget exhausted")
        frozen = state.record_store.get_recording_end_manifest(session)
        current = state.record_store.get_recording_capture_end(session)
        if current is None or any(
            current[key] != receipt[key]
            for key in (
                "session_id",
                "source_epoch_id",
                "cutoff",
                "accepted_at_utc",
            )
        ):
            raise ValueError("original capture stop receipt changed")
        if current["tail_status"] == "FAILED" or current["processing_status"] == "FAILED":
            raise RuntimeError(f"original background processing failed: {current['failure_code']}")
        remaining = 30 if deadline is None else min(30, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("whole input background processing/encoding budget exhausted")
        # Both budget modes observe progress from metadata. Admit the full HTTP
        # media projection only after the original inventory and encoding finish.
        if frozen is not None and not final_sources_checked:
            if any(
                state.record_store.get_recording_evidence(session, shot["shot_id"]).get(
                    "media_status"
                )
                == "FAILED"
                for shot in frozen["shots"]
            ):
                raise RuntimeError("terminal original recording interval failure")
            final_sources_checked = True
        jobs = state.media_job_store.session_media_progress(session)
        if jobs["failed_count"]:
            raise RuntimeError("terminal original media encoding failure")
        ready = (
            frozen is not None
            and current["processing_status"] == "COMPLETE"
            and jobs["generated_count"] >= len(frozen["shots"])
        )
        if ready:
            review = _request(server, "/api/postgame-review", timeout_seconds=remaining)
        else:
            review = dict(
                review_session_id=session,
                phase="RECORDING_PROCESSING",
                processing=state._recording_processing_progress(current),
                discovered_count=len(jobs["items"]),
                expected_count=None,
                available_count=jobs["generated_count"],
                clips=[],
                encoding_count=jobs["encoding_count"],
                retrying_count=jobs["retrying_count"],
            )
        observed = time.monotonic()
        if deadline is not None and observed >= deadline:
            raise TimeoutError("whole input background processing/encoding budget exhausted")
        if review.get("review_session_id") != session:
            raise ValueError("whole input review identity changed")
        if observed - last_sample >= 2:
            sample = dict(
                seconds=round(observed - started, 3),
                phase=review["phase"],
                processing=review.get("processing"),
                discovered_count=review.get("discovered_count"),
                expected_count=review.get("expected_count"),
                available_count=review["available_count"],
                encoding_count=review.get("encoding_count"),
                retrying_count=review.get("retrying_count"),
            )
            samples.append(sample)
            del samples[:-128]
            signature = (
                review["phase"],
                review.get("processing", {}).get("status"),
                review.get("available_count"),
                review.get("expected_count"),
            )
            changed = (
                signature,
                (review.get("processing") or {}).get("processed_frame_id"),
                review.get("discovered_count"),
            )
            if changed != last_signature:
                last_change, last_signature = observed, changed
            if signature != last_emitted:
                print(json.dumps(dict(stage="background", **sample)), flush=True)
                last_emitted = signature
            if observed - started >= 600 and "background_checkpoint_600" not in report:
                report["background_checkpoint_600"] = dict(sample)
                print(json.dumps(dict(stage="checkpoint_600", **sample)), flush=True)
            if observed - last_change >= 300 and observed - last_checkpoint >= 300:
                report["background_diagnostic"] = dict(
                    code="NO_PROGRESS_CHECKPOINT",
                    seconds=sample["seconds"],
                    idle_seconds=round(observed - last_change, 3),
                    processing=sample["processing"],
                    available_count=sample["available_count"],
                )
                last_checkpoint = observed
            if progress is not None:
                progress.heartbeat(sample)
            last_sample = observed
        if frozen is not None:
            if (
                frozen["cutoff"] != receipt["cutoff"]
                or frozen["source_epoch_id"] != receipt["source_epoch_id"]
                or current["tail_status"] != "SAVED"
                or current["processing_status"] != "COMPLETE"
            ):
                raise ValueError("final recording does not prove original cutoff completion")
            if final_seen is None:
                final_seen = observed
                report["processing_seconds"] = observed - started
            if (
                review["phase"] == "RECORDING_ENDED"
                and review.get("processing", {}).get("status") == "COMPLETE"
                and type(review["expected_count"]) is int
                and review["expected_count"] == len(frozen["shots"]) == len(review["clips"])
                and all(clip["media_url"] or clip["terminal_failure"] for clip in review["clips"])
            ):
                report["encoding_seconds"] = observed - final_seen
                return frozen, review
        time.sleep(0.25 if deadline is None else min(0.25, max(0, deadline - time.monotonic())))


def _setup_players(state):
    commands = [
        FieldCommandEnvelope(
            "r4-red", state.session_id, "r4-local-red", "claim_role", {"role": "red"}, 0, 1
        ),
        FieldCommandEnvelope(
            "r4-blue", state.session_id, "r4-local-blue", "claim_role", {"role": "blue"}, 1, 2
        ),
        FieldCommandEnvelope(
            "r4-breaker",
            state.session_id,
            "r4-local-red",
            "set_breaker",
            {
                "role": "red",
                "mode": "assistance",
                "format": "single",
                "race_target": None,
                "ranking_intent": False,
                "settings_revision": 1,
            },
            2,
            3,
        ),
    ]

    class LocalCommands:
        results = []

        def claim_field_command(self):
            return commands.pop(0) if commands else None

        def complete_field_command(self, result):
            self.results.append(result)

    gateway = LocalCommands()
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )
    for _ in range(3):
        if not worker.claim_and_apply_once(raise_errors=True):
            raise RuntimeError("local setup command was not applied")
    if any(result.outcome != "applied" for result in gateway.results):
        raise RuntimeError("local setup command rejected")
    return [asdict(result) for result in gateway.results]


def _media_hash(server, url):
    connection = http.client.HTTPConnection(*server.server_address, timeout=30)
    try:
        connection.request("GET", url)
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Content-Type") != "video/mp4":
            raise RuntimeError("media HTTP admission failed")
        digest, size = hashlib.sha256(), 0
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size
    finally:
        connection.close()


def _rss():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def verify_raw_recording(frozen, output, *, frame_count=FRAME_COUNT, geometry=(1920, 1080)):
    """Read every durable original frame, independently of playable shot clips."""
    path = _plain(frozen["raw_manifest_path"])
    if not path.is_relative_to(output):
        raise ValueError("raw recording escaped this episode")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen["raw_manifest_sha256"]:
        raise ValueError("raw manifest changed")
    manifest = json.loads(raw)
    segments = manifest["segments"]
    if (
        manifest["session_id"] != frozen["session_id"]
        or manifest["source_epoch_id"] != frozen["source_epoch_id"]
        or (manifest["width"], manifest["height"]) != geometry
        or frozen["first_frame_id"] != 0
        or frozen["last_frame_id"] != frame_count - 1
        or not segments
        or segments[0]["first_frame_id"] != 0
        or segments[-1]["last_frame_id"] != frame_count - 1
        or sum(segment["frame_count"] for segment in segments) != frame_count
    ):
        raise ValueError("raw recording identity or coverage mismatch")
    for segment in segments:
        if Path(segment["name"]).name != segment["name"]:
            raise ValueError("raw segment path escape")
        _plain(path.parent / segment["name"])
    interval = FrozenCaptureInterval(
        root=path.parent,
        source_epoch_id=manifest["source_epoch_id"],
        width=geometry[0],
        height=geometry[1],
        start_frame_id=0,
        end_frame_id=frame_count - 1,
        frame_count=frame_count,
        segment_sha256={s["name"]: s["sha256"] for s in segments},
    )
    decoded, last_timestamp = 0, None
    for frame in interval.frames():
        if last_timestamp is not None and frame.captured_at_ns < last_timestamp:
            raise ValueError("raw frame timestamp reversed")
        frame.decode_bgr()
        decoded += 1
        last_timestamp = frame.captured_at_ns
    return dict(
        decoded_frames=decoded,
        segments=len(segments),
        first_frame_id=0,
        last_frame_id=frame_count - 1,
        width=geometry[0],
        height=geometry[1],
        segment_bytes=sum((path.parent / s["name"]).stat().st_size for s in segments),
    )


def run(args):
    source_root, preflight = _validate_inputs(args.input_root)
    output = _plain(args.output_root)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("acceptance requires a fresh output directory")
    for budget in (args.capture_seconds, args.encoding_seconds):
        if not math.isfinite(budget) or not 1 <= budget <= 900:
            raise ValueError("acceptance budget outside 1..900 seconds")
    browser_seconds = getattr(args, "browser_seconds", 0)
    if not math.isfinite(browser_seconds) or not 0 <= browser_seconds <= 300:
        raise ValueError("browser inspection budget outside 0..300 seconds")
    source_revision = _source_revision()
    output.mkdir()
    for name in ("ring", "jobs", "artifacts"):
        (output / name).mkdir()
    report = dict(
        contract="recording-first-fullgame-run/v2",
        product_pass=False,
        model_pass=False,
        field_pass=False,
        windows_run=sys.platform == "win32",
        source_sha256=SOURCE_SHA256,
        source_preflight=preflight,
        source_commit=source_revision,
        cleanup_confirmed=False,
        samples=[],
    )
    report["source_files"] = {
        name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/algorithm/run_recording_first_fullgame.py",
            "src/billiards_ai/lab/mainline1_field_runtime.py",
            "src/billiards_ai/lab/mainline1_session_recording.py",
            "src/billiards_ai/lab/field_assistance/live_source.py",
            "src/billiards_ai/lab/field_assistance/visual_motion.py",
            "src/billiards_ai/lab/mainline1_product_realtime_runtime.py",
            "src/billiards_ai/lab/mainline1_live_window.py",
            "src/billiards_ai/lab/mainline1_realtime_ring.py",
            "src/billiards_ai/lab/mainline1_capture_hub.py",
            "src/billiards_ai/lab/mainline1_recording_processing.py",
            "src/billiards_ai/edge/route_assistance/persistence.py",
            "src/billiards_ai/edge/route_assistance/recording_capture.py",
            "src/billiards_ai/lab/field_assistance/recording_recovery.py",
            "src/billiards_ai/lab/field_assistance/pages.py",
            "src/billiards_ai/lab/mainline1_recording_media.py",
            "src/billiards_ai/lab/mainline1_shot_jobs.py",
            "src/billiards_ai/lab/mainline1_shot_worker.py",
        )
    }
    server = thread = prepared = None
    at_end, release_tail = threading.Event(), threading.Event()
    started = time.monotonic()
    original_threads = set(threading.enumerate())
    from scripts.algorithm.recording_fullgame_progress import FullgameProgress

    progress = FullgameProgress(output, report, started=started)
    report["until_complete"] = getattr(args, "until_complete", False)

    class HeldTailSource(ScenarioVideoCaptureBackend):
        """A camera-level final-frame latch, with no boundary or rule inputs."""

        emitted = 0

        def frames(self, stop):
            for frame in super().frames(stop):
                self.emitted += 1
                yield frame
                if frame.capture_frame_id == FRAME_COUNT - 1:
                    at_end.set()
                    while not self._stopped(stop) and not release_tail.wait(0.02):
                        pass

    try:
        options = [
            "--scenario-video",
            str(source_root / "input.avi"),
            "--capture-setup",
            str(source_root / "calibration/capture-setup.json"),
            "--calibration-root",
            str(source_root / "calibration/calibration-registry"),
            "--config",
            str(source_root / "source/configs/algorithm/shot-local.yaml"),
            "--state-db",
            str(output / "state.sqlite3"),
            "--ring-root",
            str(output / "ring"),
            "--jobs-root",
            str(output / "jobs"),
            "--artifact-root",
            str(output / "artifacts"),
            "--runtime-environment",
            "development",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--device",
            "cpu",
            "--session-policy",
            "recording-v1",
        ]
        parsed = blackbox._parser().parse_args(options)
        blackbox._validate_paths(parsed)
        with (
            progress.observe(),
            patch.object(blackbox, "ScenarioVideoCaptureBackend", HeldTailSource),
        ):
            calibration, factory, prepared = blackbox.prepare_runtime(
                parsed, initial_source_epoch_id="r4-synthetic-fullgame"
            )
            token = secrets.token_urlsafe(32)
            server = service.build_live_field_assistance_server(
                artifact_path=output / "artifacts",
                state_db_path=output / "state.sqlite3",
                host="127.0.0.1",
                port=0,
                control_token=token,
                source_epoch_factory=lambda: "r4-synthetic-fullgame",
                calibration_id=calibration,
                runtime_factory=factory,
                fresh_session_on_start=True,
                session_policy="recording-v1",
            )
            state = server.field_state
            # A finite diagnostic input must not silently replay after failure.
            state.managed_blackbox = True
            state.monotonic = lambda: 0.0  # Keep ended-review rollover out of this observer.
            report["session_id"] = state.session_id
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            live_ready = dict(
                contract="recording-first-live-ready/v1",
                url=f"http://127.0.0.1:{server.server_address[1]}/screen",
                session_id=state.session_id,
            )
            with (output / "live-ready.json").open("x", encoding="utf-8") as ready_stream:
                json.dump(live_ready, ready_stream)
                ready_stream.write("\n")
            print(json.dumps(dict(stage="live_ready", **live_ready)), flush=True)
            report["commands"] = _setup_players(state)
            state.wait_until_capture_ready(timeout_seconds=30)
            deadline, last_progress = time.monotonic() + args.capture_seconds, 0.0
            max_backlog = max_writer = 0
            while not at_end.wait(1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("whole input capture budget exhausted")
                runtime = state.projection_source.runtime
                with state.match_lock:
                    sample = _capture_sample(state, runtime, started=started)
                    max_backlog = max(max_backlog, sample["detection_backlog_frames"])
                    max_writer = max(max_writer, sample["pending_chunks"])
                if time.monotonic() - last_progress >= 15:
                    progress.heartbeat(sample)
                    report["samples"].append(sample)
                    print(json.dumps(sample), flush=True)
                    last_progress = time.monotonic()
            report["capture_seconds"] = time.monotonic() - started
            report["sampled_detection_backlog_high_water"] = max_backlog
            report["sampled_writer_high_water"] = max_writer
            report["emitted_frames"] = prepared.emitted
            stop_started = time.monotonic()
            ended = _request(
                server,
                "/api/control",
                token=token,
                payload=dict(command="end_session", session_id=state.session_id),
            )
            report["capture_stop_seconds"] = time.monotonic() - stop_started
            receipt = ended["recording_end"]
            report["capture_stop"] = receipt
            if receipt["cutoff"]["frame_id"] != FRAME_COUNT - 1:
                raise ValueError("capture stop did not fence the original whole input")
            print(
                json.dumps(
                    dict(
                        stage="capture_stopped",
                        cutoff=receipt["cutoff"]["frame_id"],
                        seconds=report["capture_stop_seconds"],
                    )
                ),
                flush=True,
            )
            frozen, review = _wait_for_background_completion(
                server,
                state,
                receipt,
                deadline=None
                if report["until_complete"]
                else time.monotonic() + args.encoding_seconds,
                report=report,
                progress=progress,
            )
            report["recording_end"] = frozen
            report["consumer_health"] = asdict(runtime._runtime.capture_health)
            report["capture_health"] = asdict(runtime.video_health)
            report["writer_health"] = asdict(runtime.recording_writer_health)
            report["runtime_diagnostics"] = asdict(runtime.diagnostics)
            print(
                json.dumps(
                    dict(stage="ended", shots=len(frozen["shots"]), emitted_frames=prepared.emitted)
                ),
                flush=True,
            )
            report["evidence"] = [
                state.record_store.get_recording_evidence(state.session_id, s["shot_id"])
                for s in frozen["shots"]
            ]
            binding = json.loads(
                (source_root / "build-evidence/oracle-interval-binding.json").read_bytes()
            )
            report["differences"] = compare_intervals(report["evidence"], binding["shot_intervals"])
            print(
                json.dumps(dict(stage="encoded", available=review["available_count"])), flush=True
            )
            report["review"] = review
            report["media"] = []
            for job in state.media_job_store.list_session_media_jobs(state.session_id):
                snapshot = state.media_job_store.snapshot(job.job_id)
                if snapshot.artifact_id is None:
                    continue
                artifact = verify_recording_playback_artifact(
                    output / "artifacts/postgame" / state.session_id / snapshot.artifact_id, job
                )
                clip = next(clip for clip in review["clips"] if clip["shot_id"] == job.shot_id)
                digest, size = _media_hash(server, clip["media_url"])
                if (digest, size) != (artifact.content_sha256, artifact.byte_size):
                    raise RuntimeError("HTTP media differs from verified original artifact")
                report["media"].append(
                    dict(
                        shot_id=job.shot_id, frames=artifact.frame_count, bytes=size, sha256=digest
                    )
                )
                print(
                    json.dumps(
                        dict(stage="verified_media", shot=job.shot_id, count=len(report["media"]))
                    ),
                    flush=True,
                )
            report["analysis_jobs"] = len(state.media_job_store.list_session_jobs(state.session_id))
            report["saved_outbox"] = len(state.record_store.list_saved_media_outbox())
            report["peak_rss_bytes"] = _rss()
            raw_started = time.monotonic()
            report["raw_recording"] = verify_raw_recording(frozen, output)
            report["raw_verification_seconds"] = time.monotonic() - raw_started
            report["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            report["product_pass"] = bool(
                not report["differences"]
                and report["emitted_frames"] == FRAME_COUNT
                and frozen["last_frame_id"] == FRAME_COUNT - 1
                and review["available_count"] == 28
                and len(report["media"]) == 28
                and report["analysis_jobs"] == report["saved_outbox"] == 0
                and report["raw_recording"]["decoded_frames"] == FRAME_COUNT
                and report["capture_health"]["received_frames"] == FRAME_COUNT
                and not report["capture_health"]["reason"]
                and report["consumer_health"]["received_frames"] == FRAME_COUNT
                and not report["consumer_health"]["reason"]
            )
            _hold_for_browser(server, report, output, browser_seconds)
    except KeyboardInterrupt:
        report["interrupted"] = True
        report["failure"] = dict(type="KeyboardInterrupt", reason="episode interrupted")
    except Exception as exc:
        report["failure"] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        # Cancel before releasing the source latch, so cleanup cannot advance into EOF.
        cleanup_errors = []
        cleanups = []
        if server is not None:
            if thread is not None and thread.ident is not None:
                cleanups.append(server.shutdown)
            cleanups.append(server.server_close)
        if prepared is not None:
            cleanups.append(prepared.cancel)
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
        release_tail.set()
        if thread is not None and thread.ident is not None:
            thread.join(5)
        report["remaining_threads"] = [
            t.name for t in threading.enumerate() if t not in original_threads and t.is_alive()
        ]
        report["cleanup_errors"] = cleanup_errors
        report["cleanup_confirmed"] = (
            not cleanup_errors
            and (thread is None or not thread.is_alive())
            and not report["remaining_threads"]
        )
        with progress.lock:
            report["clip_timeline"] = list(progress.rows.values())
            report["observation_status"] = (
                "PASS"
                if not report.get("observer_errors")
                and not report.get("timeline_overflow")
                and len(
                    {
                        row["job_sha256"]
                        for row in report["clip_timeline"]
                        if row["status"] == "available" and "done_seconds" in row
                    }
                )
                == 28
                else "INCOMPLETE"
            )
        report["elapsed_seconds"] = time.monotonic() - started
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                dict(
                    output=str(output),
                    product_pass=report["product_pass"],
                    failure=report.get("failure"),
                    cleanup=report["cleanup_confirmed"],
                )
            ),
            flush=True,
        )
    return (
        0
        if report["product_pass"] and report["cleanup_confirmed"] and not report.get("failure")
        else 1
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture-seconds", type=float, default=360)
    parser.add_argument("--encoding-seconds", type=float, default=600)
    parser.add_argument("--browser-seconds", type=float, default=0)
    parser.add_argument("--until-complete", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
