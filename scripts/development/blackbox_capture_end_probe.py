"""Verify capture-end on the full fixed input, explicitly excluding postgame acceptance."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from scripts.development import blackbox_report as reports
from scripts.development import blackbox_source as sources

CONTRACT = "blackbox-capture-end-probe/v1"
EXPECTED_FRAMES = 15880


class CaptureEndProbeComplete(Exception):
    """Tool-only stop after the original HTTP end and cutoff checks have completed."""


def verify_end(state, receipt, report):
    from billiards_ai.lab.mainline1_session_recording import CommittedRecordingSource

    runtime = state.projection_source.runtime
    stored = state.record_store.get_recording_capture_end(state.session_id)
    if (
        not isinstance(runtime._analysis_source, CommittedRecordingSource)
        or receipt.get("contract") != "recording-capture-end/v1"
        or receipt.get("session_id") != state.session_id
        or receipt.get("source_epoch_id") != state.projection_source.source_epoch_id
        or receipt.get("capture_status") != "STOPPED"
        or receipt.get("cutoff", {}).get("frame_id") != EXPECTED_FRAMES - 1
        or state.record_store.load_session(state.session_id).status.value != "ended"
        or stored is None
        or any(
            stored[key] != receipt[key]
            for key in ("session_id", "source_epoch_id", "cutoff", "accepted_at_utc")
        )
        or report.get("emitted_frames") != EXPECTED_FRAMES
        or type(report.get("capture_stop_seconds")) not in {int, float}
        or not 0 <= report["capture_stop_seconds"] < 1
    ):
        raise ValueError("capture end probe assertions failed")
    return {
        "acknowledged": True,
        "durable_session_ended": True,
        "original_cutoff_unchanged": True,
        "disk_detection_source": True,
        "emitted_frames": EXPECTED_FRAMES,
        "cutoff_frame_id": EXPECTED_FRAMES - 1,
        "end_seconds": report["capture_stop_seconds"],
        "sampled_backlog_high_water": report.get("sampled_detection_backlog_high_water"),
    }


def run(fullgame, argv, root, source_commit, input_sha256):
    observation = None

    def after_original_end(server, state, receipt, *, deadline, report, progress=None):
        nonlocal observation
        observation = verify_end(state, receipt, report)
        observation[
            "detector_pending_at_ack"
        ] = not state.projection_source._runtime_exited.is_set()
        tail_deadline = min(deadline, time.monotonic() + 10)
        while True:
            if time.monotonic() >= tail_deadline:
                raise TimeoutError("capture tail observation exceeded deadline")
            current = state.record_store.get_recording_capture_end(state.session_id)
            if time.monotonic() >= tail_deadline:
                raise TimeoutError("capture tail observation exceeded deadline")
            if current is None or any(
                current[key] != receipt[key]
                for key in ("session_id", "source_epoch_id", "cutoff", "accepted_at_utc")
            ):
                raise ValueError("capture end receipt changed after ACK")
            if current["tail_status"] == "FAILED":
                raise ValueError("capture tail failed after ACK")
            if current["tail_status"] == "SAVED":
                observation["tail_saved"] = True
                break
            if time.monotonic() >= tail_deadline:
                raise TimeoutError("capture tail observation exceeded 10 seconds")
            time.sleep(0.05)
        observation["backpressure_case_covered"] = (
            observation["sampled_backlog_high_water"] is not None
            and observation["sampled_backlog_high_water"] > 180
        )
        raise CaptureEndProbeComplete("capture-end scope complete; postgame checks not run")

    with patch.object(fullgame, "_wait_for_background_completion", after_original_end):
        exit_code = fullgame.main(argv)
    path = Path(root) / "recording/report.json"
    report = json.loads(path.read_bytes())
    complete = bool(
        exit_code == 1
        and observation is not None
        and report.get("source_commit") == source_commit
        and report.get("source_sha256") == input_sha256
        and report.get("product_pass") is False
        and report.get("cleanup_confirmed") is True
        and report.get("failure", {}).get("type") == "CaptureEndProbeComplete"
        and report.get("emitted_frames") == EXPECTED_FRAMES
        and report.get("capture_stop", {}).get("cutoff", {}).get("frame_id") == EXPECTED_FRAMES - 1
        and not report.get("media")
    )
    value = {
        "contract": CONTRACT,
        "run_id": Path(root).name,
        "source_commit": source_commit,
        "input_sha256": input_sha256,
        "scope": "CAPTURE_END_ONLY",
        "capture_end_status": "PASS" if complete else "FAIL",
        "cleanup_confirmed": report.get("cleanup_confirmed") is True,
        "postgame_checks": "NOT_RUN",
        "fullgame_product_pass": False,
        "original_report": sources.pin(path),
        "observation": observation,
    }
    reports.atomic_json(Path(root) / "capture-end-probe.json", value)
    return 0 if complete else 1


def export(value, run_id, source_commit, input_sha256):
    if (
        not isinstance(value, dict)
        or value.get("contract") != CONTRACT
        or value.get("run_id") != run_id
        or value.get("source_commit") != source_commit
        or value.get("input_sha256") != input_sha256
        or value.get("scope") != "CAPTURE_END_ONLY"
        or value.get("postgame_checks") != "NOT_RUN"
        or value.get("fullgame_product_pass") is not False
        or value.get("capture_end_status") not in {"PASS", "FAIL"}
    ):
        raise ValueError("capture end probe identity mismatch")
    result = {
        key: value[key]
        for key in (
            "contract",
            "scope",
            "capture_end_status",
            "postgame_checks",
            "fullgame_product_pass",
        )
    }
    result["cleanup_confirmed"] = value.get("cleanup_confirmed") is True
    observation = value.get("observation")
    clean = {}
    if isinstance(observation, dict):
        for key in (
            "acknowledged",
            "durable_session_ended",
            "original_cutoff_unchanged",
            "disk_detection_source",
            "tail_saved",
            "detector_pending_at_ack",
            "backpressure_case_covered",
        ):
            if type(observation.get(key)) is bool:
                clean[key] = observation[key]
        for key in (
            "emitted_frames",
            "cutoff_frame_id",
            "end_seconds",
            "sampled_backlog_high_water",
        ):
            if reports.finite(observation.get(key)):
                clean[key] = observation[key]
    result["observation"] = clean
    if result["capture_end_status"] == "PASS" and (
        not result["cleanup_confirmed"]
        or not all(
            clean.get(key) is True
            for key in (
                "acknowledged",
                "durable_session_ended",
                "original_cutoff_unchanged",
                "disk_detection_source",
                "tail_saved",
            )
        )
        or clean.get("emitted_frames") != EXPECTED_FRAMES
        or clean.get("cutoff_frame_id") != EXPECTED_FRAMES - 1
        or not 0 <= clean.get("end_seconds", 2) < 1
    ):
        raise ValueError("capture end probe false pass")
    return result


def controller_passed(summary, observation):
    return (
        summary.get("status") == "PASS"
        and summary.get("exit_code") == 0
        and summary.get("owned_cleanup") is True
        and observation.get("capture_end_status") == "PASS"
    )
