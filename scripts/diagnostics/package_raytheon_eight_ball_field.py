#!/usr/bin/env python3
"""Build the independent development-only Raytheon eight-ball field packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

PACKET_NAME = "billiardsAI-raytheon-eight-ball-field-20260907-v14"
BASELINE_COMMIT = "9a144955377b289ce47de71769d1305dd8585de4"
PROFILE_ID = "cbsa-chinese-pool-2017-observable-v1"
DEVELOPMENT_ENVIRONMENT_ID = "cloud1-public-example"
DEVICE_LOCATOR = r"USB\VID_32E4&PID_6678&MI_00\6&B920475&0&0000"
DEVICE_FINGERPRINT = "865a64805fad161c4f09aeb35214fa23b061be0d8a08c9e6fc99f39c86481a93"
DETECTOR_ROOT = (
    "bootstrap-training-runs/"
    "2759358fe07f948a3c056a18f2b592e0b0ebad76364159f0a774c4eca70ca186/"
    "rfdetr-seg-small-bootstrap-short-r1/"
    "967612dd02c72a6af9149b6790820c4e998068df4536d3b006b917c774ac4a51"
)
IDENTITY_ROOT = (
    "identity-group-training-runs/2098055fcba31010d88cda072e5ec33b46f125e91c8ea6512673cb2070349b97"
)

GATE_STATUSES = frozenset({"PASS", "FAIL", "NOT_RUN"})
DISCONNECT_ABSTENTION_MARKERS = (
    "capture_interrupted",
    "capture_disconnected",
    "source_disconnected",
    "device_disconnected",
)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _gate(
    gate_id: str,
    status: str,
    determination: str,
    evidence: Mapping[str, object],
    reason: str | None,
) -> dict[str, object]:
    if status not in GATE_STATUSES:
        raise ValueError("field gate status is invalid")
    return {
        "gate_id": gate_id,
        "required": True,
        "status": status,
        "determination": determination,
        "evidence": dict(evidence),
        "reason": reason,
    }


def _is_disconnect_abstention(reason: object) -> bool:
    if not isinstance(reason, str):
        return False
    normalized = reason.strip().casefold().replace("-", "_")
    return "disconnect" in normalized or any(
        normalized == marker or normalized.startswith(marker + ":")
        for marker in DISCONNECT_ABSTENTION_MARKERS
    )


def inspect_worker_publication(
    *, state_db: Path, jobs_db: Path, session_id: str
) -> dict[str, object]:
    """Verify session media sync and the published AVAILABLE analysis revision."""

    receipt: dict[str, object] = {
        "schema_version": "1.0",
        "session_id": session_id,
        "status": "FAIL",
        "field_sync_status": None,
        "session_analysis_pending_count": None,
        "cloud_analysis_status": None,
        "verified": False,
        "reason": None,
    }
    if not session_id.strip():
        receipt["reason"] = "session ID is missing"
        return receipt
    if (
        not state_db.is_file()
        or state_db.is_symlink()
        or not jobs_db.is_file()
        or jobs_db.is_symlink()
    ):
        receipt["reason"] = "worker publication databases are missing or unsafe"
        return receipt
    try:
        with sqlite3.connect(state_db.resolve().as_uri() + "?mode=ro", uri=True) as state:
            state.row_factory = sqlite3.Row
            sync_rows = state.execute(
                "SELECT sync_job_id,status,last_error_code FROM field_sync_jobs WHERE session_id=?",
                (session_id,),
            ).fetchall()
        with sqlite3.connect(jobs_db.resolve().as_uri() + "?mode=ro", uri=True) as jobs:
            jobs.row_factory = sqlite3.Row
            publication_rows = jobs.execute(
                "SELECT revision,payload_sha256,payload_json,status,attempt_count,last_error "
                "FROM session_analysis_publications WHERE session_id=? "
                "ORDER BY revision",
                (session_id,),
            ).fetchall()
    except (OSError, sqlite3.DatabaseError) as exc:
        receipt["reason"] = f"worker publication query failed: {type(exc).__name__}"
        return receipt

    if len(sync_rows) != 1:
        receipt["reason"] = "exactly one field sync job is required"
        return receipt
    sync = sync_rows[0]
    receipt["field_sync_status"] = sync["status"]
    if sync["status"] != "complete" or sync["last_error_code"] is not None:
        receipt["reason"] = "field sync job is not cleanly complete"
        return receipt
    if not publication_rows:
        receipt["reason"] = "session analysis outbox has no durable publication"
        return receipt
    if any(
        hashlib.sha256(str(row["payload_json"]).encode()).hexdigest() != row["payload_sha256"]
        for row in publication_rows
    ):
        receipt["reason"] = "session analysis outbox payload hash is invalid"
        return receipt
    pending_count = sum(row["status"] != "published" for row in publication_rows)
    receipt["session_analysis_pending_count"] = pending_count
    if pending_count:
        receipt["reason"] = "session analysis outbox still has pending revisions"
        return receipt
    try:
        latest_payload = json.loads(publication_rows[-1]["payload_json"])
    except (TypeError, json.JSONDecodeError):
        receipt["reason"] = "session analysis outbox payload is invalid"
        return receipt
    if not isinstance(latest_payload, dict):
        receipt["reason"] = "session analysis outbox payload is invalid"
        return receipt
    cloud_status = latest_payload.get("status")
    receipt["cloud_analysis_status"] = cloud_status
    if latest_payload.get("session_id") != session_id or cloud_status != "AVAILABLE":
        receipt["reason"] = "latest published cloud analysis is not explicitly AVAILABLE"
        return receipt
    receipt["verified"] = True
    receipt["status"] = "PASS"
    return receipt


def _read_gate_rows(
    state_db: Path, session_ids: tuple[str, ...]
) -> tuple[list[dict[str, object]], set[tuple[str, str]], str | None]:
    if not state_db.is_file() or not session_ids:
        return [], set(), "state database or frozen session inventory is unavailable"
    placeholders = ",".join("?" for _ in session_ids)
    try:
        uri = state_db.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            decision_rows = connection.execute(
                "SELECT session_id,shot_id,source_epoch_id,status,elapsed_ms,"
                "facts_json,decision_json "
                f"FROM field_rule_decisions WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()
            shot_rows = connection.execute(
                "SELECT session_id,shot_id,payload_json FROM field_shots "
                f"WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()
    except (OSError, sqlite3.DatabaseError) as exc:
        return [], set(), f"state database gate query failed: {type(exc).__name__}"

    decisions: list[dict[str, object]] = []
    try:
        for row in decision_rows:
            decisions.append(
                {
                    "session_id": row["session_id"],
                    "shot_id": row["shot_id"],
                    "source_epoch_id": row["source_epoch_id"],
                    "status": row["status"],
                    "elapsed_ms": row["elapsed_ms"],
                    "facts": json.loads(row["facts_json"]),
                    "decision": json.loads(row["decision_json"]),
                }
            )
        settled_shots = {
            (str(row["session_id"]), str(row["shot_id"]))
            for row in shot_rows
            if _mapping(json.loads(row["payload_json"])).get("settled_ms") is not None
        }
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [], set(), f"state database gate evidence is invalid: {type(exc).__name__}"
    return decisions, settled_shots, None


def evaluate_field_gates(
    *,
    state_db: Path,
    context: Mapping[str, object],
    operator_evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    """Reduce durable and explicit operator evidence without inventing PASS results."""

    state = _mapping(context.get("state"))
    launch = _mapping(context.get("launch_audit"))
    server_exit = _mapping(context.get("server_exit"))
    phase = str(context.get("phase") or "INTERIM").upper()
    session_results = [_mapping(item) for item in _sequence(context.get("session_results"))]
    session_receipts = [_mapping(item) for item in _sequence(state.get("session_receipts"))]
    session_ids = tuple(
        str(item.get("session_id"))
        for item in session_receipts
        if isinstance(item.get("session_id"), str) and item.get("session_id")
    )
    decisions, settled_shots, database_error = _read_gate_rows(state_db, session_ids)
    settled_count = len(settled_shots)
    gates: list[dict[str, object]] = []
    operator = _mapping(operator_evidence)

    started = _timestamp(launch.get("started_at_utc"))
    stopped = _timestamp(server_exit.get("stopped_at_utc"))
    capture = _mapping(operator.get("capture_30m"))
    capture_declared = str(capture.get("status") or "NOT_RUN").upper()
    capture_observed_at = _timestamp(capture.get("observed_at_utc"))
    capture_observed = (
        capture.get("continuous_healthy_capture") is True
        and started is not None
        and stopped is not None
        and capture_observed_at is not None
        and started <= capture_observed_at <= stopped
        and bool(_sequence(capture.get("evidence")))
    )
    if capture_declared == "FAIL":
        capture_seconds = (
            None if started is None or stopped is None else (stopped - started).total_seconds()
        )
        capture_status = "FAIL"
        capture_reason = str(capture.get("reason") or "operator recorded capture failure")
    elif started is None or stopped is None:
        capture_status, capture_reason, capture_seconds = (
            "NOT_RUN",
            "complete launch and server-exit timestamps are unavailable",
            None,
        )
    else:
        capture_seconds = (stopped - started).total_seconds()
        automatic_capture_passed = (
            capture_seconds >= 1800
            and server_exit.get("ready_observed") is True
            and server_exit.get("exit_code") == 0
        )
        if not automatic_capture_passed:
            capture_status = "FAIL"
            capture_reason = "capture process was not ready and clean for 30 minutes"
        elif capture_declared == "PASS" and capture_observed:
            capture_status, capture_reason = "PASS", None
        elif capture_declared == "PASS":
            capture_status = "FAIL"
            capture_reason = "operator capture PASS lacks continuous-health evidence"
        else:
            capture_status = "NOT_RUN"
            capture_reason = "operator did not attest continuous healthy capture"
    gates.append(
        _gate(
            "healthy_capture_30m",
            capture_status,
            "automatic_and_operator",
            {
                "started_at_utc": launch.get("started_at_utc"),
                "stopped_at_utc": server_exit.get("stopped_at_utc"),
                "duration_seconds": capture_seconds,
                "ready_observed": server_exit.get("ready_observed"),
                "server_exit_code": server_exit.get("exit_code"),
                "operator": dict(capture),
            },
            capture_reason,
        )
    )

    elapsed = [item.get("elapsed_ms") for item in decisions]
    valid_elapsed = bool(elapsed) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        for value in elapsed
    )
    decision_shots = {
        (str(item.get("session_id") or ""), str(item.get("shot_id") or "")) for item in decisions
    }
    complete_rule_inventory = decision_shots == settled_shots
    if database_error is not None or not decisions:
        latency_status = "NOT_RUN"
        latency_reason = database_error or "no persisted realtime rule decision exists"
        maximum_elapsed = None
    elif not valid_elapsed:
        latency_status = "FAIL"
        latency_reason = "persisted rule latency evidence is invalid"
        maximum_elapsed = None
    elif not complete_rule_inventory:
        latency_status = "NOT_RUN" if phase != "FINAL" else "FAIL"
        latency_reason = "settled shots and persisted rule decisions are not one-to-one"
        maximum_elapsed = max(float(value) for value in elapsed)
    else:
        maximum_elapsed = max(float(value) for value in elapsed)
        latency_status = "PASS" if maximum_elapsed <= 2000 else "FAIL"
        latency_reason = None if latency_status == "PASS" else "a rule decision exceeded 2 seconds"
    gates.append(
        _gate(
            "rule_decision_latency_2s",
            latency_status,
            "automatic",
            {
                "decision_count": len(decisions),
                "settled_shot_count": settled_count,
                "complete_rule_inventory": complete_rule_inventory,
                "maximum_elapsed_ms": maximum_elapsed,
            },
            latency_reason,
        )
    )

    disconnect_tombstones = [
        item
        for item in decisions
        if any(
            _is_disconnect_abstention(reason)
            for reason in _sequence(_mapping(item.get("facts")).get("abstention_reasons"))
        )
    ]
    missing_candidates = [
        item
        for item in decisions
        if _sequence(_mapping(item.get("facts")).get("abstention_reasons"))
        and item not in disconnect_tombstones
    ]
    if database_error is not None or not decisions:
        missing_status = "NOT_RUN"
        missing_reason = database_error or "no persisted realtime rule decision exists"
    elif not missing_candidates:
        missing_status = "NOT_RUN"
        missing_reason = "no shot with missing or ambiguous facts was exercised"
    elif all(
        item.get("status") == "abstained"
        and _mapping(item.get("decision")).get("reason_code") == "fact_adapter_abstained"
        for item in missing_candidates
    ):
        missing_status = "PASS"
        missing_reason = None
    else:
        missing_status = "FAIL"
        missing_reason = "a shot with incomplete facts did not abstain"
    gates.append(
        _gate(
            "missing_fact_abstain",
            missing_status,
            "automatic",
            {
                "candidate_count": len(missing_candidates),
                "shots": [item.get("shot_id") for item in missing_candidates],
                "excluded_disconnect_shots": [
                    item.get("shot_id") for item in disconnect_tombstones
                ],
            },
            missing_reason,
        )
    )

    disconnect = _mapping(operator.get("disconnect_recovery"))
    declared_status = str(disconnect.get("status") or "NOT_RUN").upper()
    required_observations = (
        disconnect.get("idle_disconnect_abstained") is True
        and disconnect.get("in_shot_disconnect_abstained") is True
        and disconnect.get("reconnect_ready") is True
        and disconnect.get("synthesized_shot_end") is False
        and _timestamp(disconnect.get("observed_at_utc")) is not None
        and bool(_sequence(disconnect.get("evidence")))
    )
    if declared_status == "PASS" and required_observations:
        disconnect_status, disconnect_reason = "PASS", None
    elif declared_status == "FAIL":
        disconnect_status = "FAIL"
        disconnect_reason = str(disconnect.get("reason") or "operator recorded failure")
    elif declared_status == "PASS":
        disconnect_status = "FAIL"
        disconnect_reason = "operator PASS lacks all required observations and evidence"
    else:
        disconnect_status = "NOT_RUN"
        disconnect_reason = "operator did not complete both disconnect/recovery checks"
    gates.append(
        _gate(
            "disconnect_recovery",
            disconnect_status,
            "operator",
            dict(disconnect),
            disconnect_reason,
        )
    )

    if database_error is not None:
        shot_status, shot_reason = "NOT_RUN", database_error
    elif settled_count < 10 and phase != "FINAL":
        shot_status = "NOT_RUN"
        shot_reason = "10-shot exercise is not complete"
    else:
        shot_status = "PASS" if settled_count >= 10 else "FAIL"
        shot_reason = (
            None if shot_status == "PASS" else "fewer than 10 settled shots were persisted"
        )
    gates.append(
        _gate(
            "settled_shots_10",
            shot_status,
            "automatic",
            {"settled_shot_count": settled_count},
            shot_reason,
        )
    )

    result_ids = tuple(str(item.get("session_id") or "") for item in session_results)
    epoch_sets = {
        session_id: {
            str(item.get("source_epoch_id") or "")
            for item in decisions
            if item.get("session_id") == session_id and item.get("source_epoch_id")
        }
        for session_id in session_ids
    }
    fresh_source_epochs = (
        len(session_ids) == 2
        and all(epoch_sets.get(session_id) for session_id in session_ids)
        and epoch_sets[session_ids[0]].isdisjoint(epoch_sets[session_ids[1]])
    )
    two_sessions = (
        len(session_ids) == 2
        and len(set(session_ids)) == 2
        and len(result_ids) == 2
        and set(result_ids) == set(session_ids)
        and isinstance(state.get("server_pid"), int)
        and state.get("server_pid") == launch.get("server_pid")
        and fresh_source_epochs
    )
    rollover_status = "PASS" if two_sessions else ("NOT_RUN" if phase != "FINAL" else "FAIL")
    gates.append(
        _gate(
            "same_process_two_session_rollover",
            rollover_status,
            "automatic",
            {
                "session_ids": list(session_ids),
                "server_pid": state.get("server_pid"),
                "launch_server_pid": launch.get("server_pid"),
                "source_epochs": {
                    session_id: sorted(epoch_sets.get(session_id, set()))
                    for session_id in session_ids
                },
            },
            (
                None
                if two_sessions
                else (
                    "exactly two frozen sessions with disjoint source epochs "
                    "from one server are required"
                )
            ),
        )
    )

    expected_qr = state.get("join_qr_sha256")
    current_qr = context.get("current_join_qr_sha256")
    qr_passed = (
        two_sessions
        and isinstance(expected_qr, str)
        and len(expected_qr) == 64
        and current_qr == expected_qr
        and all(item.get("join_qr_sha256") == expected_qr for item in session_receipts)
    )
    local_qr_not_run = (
        state.get("local_only") is True
        and expected_qr is None
        and current_qr is None
        and all(item.get("join_qr_sha256") is None for item in session_receipts)
    )
    qr_status = (
        "NOT_RUN"
        if local_qr_not_run
        else ("PASS" if qr_passed else ("NOT_RUN" if phase != "FINAL" else "FAIL"))
    )
    qr_reason = None
    if local_qr_not_run:
        qr_reason = "LocalOnly ran without a join QR; permanent QR reuse was not exercised"
    elif not qr_passed:
        qr_reason = "two sessions did not retain the same permanent QR hash"
    gates.append(
        _gate(
            "permanent_qr_hash",
            qr_status,
            "automatic",
            {
                "expected_sha256": expected_qr,
                "current_sha256": current_qr,
                "session_count": len(session_receipts),
            },
            qr_reason,
        )
    )

    roots = [str(item.get("artifact_root") or "") for item in session_results]
    reports = [str(item.get("report_session_id") or "") for item in session_results]
    job_sets = [set(map(str, _sequence(item.get("job_ids")))) for item in session_results]
    artifact_sets = [set(map(str, _sequence(item.get("artifact_ids")))) for item in session_results]
    isolated = (
        two_sessions
        and len(set(roots)) == 2
        and all(
            Path(root).name == session_id
            for root, session_id in zip(roots, result_ids, strict=True)
        )
        and reports == list(result_ids)
        and all(job_sets)
        and all(artifact_sets)
        and job_sets[0].isdisjoint(job_sets[1])
        and artifact_sets[0].isdisjoint(artifact_sets[1])
    )
    isolation_status = "PASS" if isolated else ("NOT_RUN" if phase != "FINAL" else "FAIL")
    gates.append(
        _gate(
            "session_artifact_isolation",
            isolation_status,
            "automatic",
            {"artifact_roots": roots, "report_session_ids": reports},
            None if isolated else "session job, artifact, report, or root isolation is incomplete",
        )
    )

    timing_rows = []
    postgame_valid = two_sessions
    for item in session_results:
        ended = _timestamp(item.get("ended_at_utc"))
        deadline = _timestamp(item.get("deadline_at_utc"))
        completed = _timestamp(item.get("completed_at_utc"))
        deadline_seconds = (
            (deadline - ended).total_seconds()
            if deadline is not None and ended is not None
            else None
        )
        completion_seconds = (
            (completed - ended).total_seconds()
            if completed is not None and ended is not None
            else None
        )
        within = (
            deadline_seconds is not None
            and completion_seconds is not None
            and 0 <= deadline_seconds <= 120
            and 0 <= completion_seconds <= 120
            and completed <= deadline
        )
        local_worker_verified = (
            item.get("worker_exit_code") == 0
            and item.get("worker_timed_out") is False
            and item.get("analysis_status") == "AVAILABLE"
        )
        local_accepted = item.get("local_accepted") is True and local_worker_verified
        postgame_valid &= within and local_accepted
        timing_rows.append(
            {
                "session_id": item.get("session_id"),
                "ended_at_utc": item.get("ended_at_utc"),
                "deadline_at_utc": item.get("deadline_at_utc"),
                "completed_at_utc": item.get("completed_at_utc"),
                "deadline_seconds_after_end": deadline_seconds,
                "completion_seconds_after_end": completion_seconds,
                "within_deadline": within,
                "worker_exit_code": item.get("worker_exit_code"),
                "worker_timed_out": item.get("worker_timed_out"),
                "analysis_status": item.get("analysis_status"),
                "field_sync_status": item.get("field_sync_status"),
                "session_analysis_pending_count": item.get("session_analysis_pending_count"),
                "cloud_analysis_status": item.get("cloud_analysis_status"),
                "publication_verified": item.get("publication_verified"),
                "local_session_accepted": local_accepted,
            }
        )
    postgame_status = "PASS" if postgame_valid else ("NOT_RUN" if phase != "FINAL" else "FAIL")
    gates.append(
        _gate(
            "postgame_120s",
            postgame_status,
            "automatic",
            {"sessions": timing_rows},
            None if postgame_valid else "both sessions were not finalized within 120 seconds",
        )
    )

    local_only = state.get("local_only") is True
    configured_analysis_sync = state.get("analysis_sync_enabled")
    analysis_sync_enabled = (
        not local_only if configured_analysis_sync is None else configured_analysis_sync is True
    )
    cloud_rows = [
        {
            "session_id": item.get("session_id"),
            "field_sync_status": item.get("field_sync_status"),
            "session_analysis_pending_count": item.get("session_analysis_pending_count"),
            "cloud_analysis_status": item.get("cloud_analysis_status"),
            "publication_verified": item.get("publication_verified"),
        }
        for item in session_results
    ]
    cloud_valid = two_sessions and all(
        item.get("field_sync_status") == "complete"
        and item.get("session_analysis_pending_count") == 0
        and item.get("cloud_analysis_status") == "AVAILABLE"
        and item.get("publication_verified") is True
        for item in session_results
    )
    if not analysis_sync_enabled:
        cloud_status = "NOT_RUN"
        cloud_reason = "This run mode explicitly disables development cloud publication"
    else:
        cloud_status = "PASS" if cloud_valid else ("NOT_RUN" if phase != "FINAL" else "FAIL")
        cloud_reason = (
            None if cloud_valid else "durable development cloud AVAILABLE was not verified"
        )
    gates.append(
        _gate(
            "development_cloud_publication",
            cloud_status,
            "automatic",
            {
                "local_only": local_only,
                "analysis_sync_enabled": analysis_sync_enabled,
                "sessions": cloud_rows,
            },
            cloud_reason,
        )
    )

    all_required_pass = all(item["status"] == "PASS" for item in gates)
    rule_authority_accepted = (
        not local_only
        and state.get("observation_only") is False
        and state.get("identity_gate_status") == "PASS"
    )
    accepted = phase == "FINAL" and all_required_pass and rule_authority_accepted and two_sessions
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "phase": phase,
        "development_evidence_only": True,
        "production_ready": False,
        "gates": gates,
        "all_required_pass": all_required_pass,
        "rule_authority_accepted": rule_authority_accepted,
        "observation_only": state.get("observation_only"),
        "local_only": local_only,
        "identity_gate_status": state.get("identity_gate_status"),
        "distinct_session_count": len(set(session_ids)),
        "accepted": accepted,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_from_directory(root: Path) -> dict[str, bytes]:
    candidate = root.resolve()
    if not candidate.is_dir() or candidate.is_symlink():
        raise ValueError("candidate root is missing or unsafe")
    files: dict[str, bytes] = {}
    for path in candidate.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"candidate contains an unsafe symlink: {path}")
        if path.is_file():
            relative = path.relative_to(candidate).as_posix()
            files[f"candidate/{relative}"] = path.read_bytes()
    return files


def _candidate_from_archive(path: Path) -> dict[str, bytes]:
    source = path.resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("candidate archive is missing or unsafe")
    files: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                name = item.filename
                if "\\" in name:
                    raise ValueError("candidate archive contains an unsafe path")
                parts = PurePosixPath(name).parts
                if PurePosixPath(name).is_absolute() or ".." in parts:
                    raise ValueError("candidate archive contains an unsafe path")
                positions = [index for index, part in enumerate(parts) if part == "candidate"]
                if not positions:
                    continue
                if len(positions) != 1 or positions[0] == len(parts) - 1:
                    raise ValueError("candidate archive inventory is ambiguous")
                relative = PurePosixPath(*parts[positions[0] + 1 :]).as_posix()
                target = f"candidate/{relative}"
                if target in files:
                    raise ValueError("candidate archive contains duplicate inventory")
                files[target] = archive.read(item)
    except zipfile.BadZipFile as exc:
        raise ValueError("candidate archive is not a valid ZIP") from exc
    return files


def _validate_candidate(files: dict[str, bytes]) -> str | None:
    for required in (
        "candidate/capture-setup.json",
        "candidate/calibrations/registry.json",
    ):
        if required not in files:
            raise ValueError(f"candidate asset is missing: {required}")
    if not any(
        name.startswith("candidate/calibrations/") and name.endswith("/calibration.json")
        for name in files
    ):
        raise ValueError("candidate calibration artifact is missing")
    qr_assets = [
        name
        for name in files
        if name
        in {
            "candidate/join-qr.png",
            "candidate/join-qr.jpg",
            "candidate/join-qr.jpeg",
        }
    ]
    if len(qr_assets) > 1:
        raise ValueError("candidate contains ambiguous join-qr assets")
    if not qr_assets:
        return None
    qr_asset = qr_assets[0]
    payload = files[qr_asset]
    if not (payload.startswith(b"\x89PNG\r\n\x1a\n") or payload.startswith(b"\xff\xd8\xff")):
        raise ValueError("candidate join QR image format is invalid")
    return qr_asset


def _required_project_file(project: Path, relative: str) -> bytes:
    path = project / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"packet source is missing or unsafe: {path}")
    return path.read_bytes()


def build_packet(
    project_root: Path,
    output_root: Path,
    *,
    candidate_root: Path | None = None,
    candidate_archive: Path | None = None,
    join_qr_asset: Path | None = None,
) -> Path:
    """Build one packet from a candidate directory or an earlier packet ZIP."""

    if (candidate_root is None) == (candidate_archive is None):
        raise ValueError("provide exactly one candidate root or candidate archive")
    project = project_root.resolve()
    destination = output_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    candidate_files = (
        _candidate_from_directory(candidate_root)
        if candidate_root is not None
        else _candidate_from_archive(candidate_archive)
    )
    if join_qr_asset is not None:
        qr = join_qr_asset.resolve()
        if (
            qr.is_symlink()
            or not qr.is_file()
            or qr.suffix.lower() not in {".png", ".jpg", ".jpeg"}
        ):
            raise ValueError("join QR asset is missing, unsafe, or not PNG/JPEG")
        for existing in tuple(candidate_files):
            if existing.startswith("candidate/join-qr."):
                del candidate_files[existing]
        candidate_files[f"candidate/join-qr{qr.suffix.lower()}"] = qr.read_bytes()
    join_qr_relative = _validate_candidate(candidate_files)

    files: dict[str, bytes] = {}
    source_root = project / "src" / "billiards_ai"
    for path in source_root.rglob("*.py"):
        if path.is_symlink():
            raise ValueError(f"source tree contains an unsafe symlink: {path}")
        files[path.relative_to(project).as_posix()] = path.read_bytes()
    for relative in (
        "scripts/algorithm/serve_mainline1_usb_field.py",
        "scripts/algorithm/serve_mainline1_blackbox_field.py",
        "scripts/algorithm/run_mainline1_product_realtime.py",
        "scripts/algorithm/run_mainline1_shot_worker.py",
        "scripts/algorithm/windows/serve-mainline1-usb-field.ps1",
        "configs/algorithm/shot-local.yaml",
        "native/windows/mainline1_uvc_capture_csharp/main.cs",
        "native/windows/mainline1_uvc_capture/protocol.md",
    ):
        files[relative] = _required_project_file(project, relative)
    packet_sources = project / "scripts" / "diagnostics" / "raytheon_eight_ball_field"
    for name in (
        "run.ps1",
        "README.txt",
        "export_evidence.py",
        "retire_existing_session.py",
    ):
        files[name] = _required_project_file(packet_sources, name)
    files["field_gate_evaluator.py"] = Path(__file__).read_bytes()
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    files.update(candidate_files)

    manifest = {
        "schema_version": "1.0",
        "packet_version": PACKET_NAME,
        "kind": "mainline1_eight_ball_field_development_acceptance",
        "development_evidence_only": True,
        "production_ready": False,
        "baseline_commit": BASELINE_COMMIT,
        "source_identity": "packet_manifest_per_file_sha256",
        "production_gates_pending": [
            "trusted_pnp_to_capture_handle_binding",
            "product_controls_lock",
            "low_level_camera_read_timeout_and_cancel",
            "actual_table_corner_detector",
            "signed_native_release",
            "non_windows_product_hardware_base",
            "cross_process_revocable_gpu_preemption",
        ],
        "writes_installed_service_or_state_db": True,
        "installed_state_write_scope": ("retire_one_existing_development_session_only"),
        "writes_production_state": False,
        "writes_development_cloud_state": True,
        "native_helper": {
            "trust": "unsigned_source_compiled_at_runtime",
            "production_signature_required_for_release": True,
        },
        "run": {
            "phases": ["Start", "EndSession"],
            "rule_profile": PROFILE_ID,
            "development_environment_id": DEVELOPMENT_ENVIRONMENT_ID,
            "identity_gate_required_for_rule_authority": "PASS",
            "failed_identity_behavior": "observation_only_or_block",
            "postgame_deadline_seconds": 120,
            "postgame_execution": "per_session_async_live_priority_then_offline_final",
            "worker_launch_receipt": "run_state_session_scoped_pid_command_and_paths",
            "qr_source": (
                "packet_if_present_else_current_json_in_existing_development_control_only"
            ),
            "sync_source": "current_json_in_existing_development_control_only",
            "local_only_supported": True,
            "existing_development_control_only_supported": True,
            "recommended_field_mode": "observation_only_existing_development_control_only",
            "existing_development_control_base_url": (
                "https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com"
            ),
            "existing_development_device_id": "raytheon-01",
            "existing_development_credential_target": ("billiardsAI/development/raytheon-01"),
            "analysis_sync_enabled_in_recommended_mode": False,
            "live_priority_isolation": "development_best_effort_not_proven",
            "device_locator": DEVICE_LOCATOR,
            "device_fingerprint": DEVICE_FINGERPRINT,
            "join_qr_asset": join_qr_relative,
            "state_scope": "unique_result_root_only",
            "synthesizes_participants_or_breaker": False,
            "launch_parameters_redacted": True,
            "result_evidence": [
                "launch-audit.json",
                "native-helper.json",
                "server-exit.json",
                "state-audit.json",
                "field-gates.json",
                "summary.json",
                "existing-session-retirement.json",
                "session-worker-result-<session_id>.json",
                "live-resource-<session_id>.jsonl",
            ],
        },
        "assets": {
            "detector": {
                "relative_root": DETECTOR_ROOT,
                "manifest_sha256": (
                    "355acf72c8e698e03a1f90265537a2e78e4cd883a1df93e8be8e40376e60934b"
                ),
                "checkpoint_sha256": (
                    "d9cd5efe6c6deb8b7762e08c0f720cb264a2f41bce9174ddc77154a56f434541"
                ),
            },
            "identity": {
                "relative_root": IDENTITY_ROOT,
                "manifest_sha256": (
                    "233a9e63829eceb674dbbe67741d4b4b9afffe84264831f6167e87934c685443"
                ),
                "head_sha256": ("2737c03cf7418da8fd308f44c58460261b4801791ce1d0808c0a589eadd73d03"),
                "metrics_sha256": (
                    "c9a86ad22a56a4351e9889bdb3375c2c3fa36b58c9ca632ffb184aa21fc182f6"
                ),
                "gate_status": "FAIL",
                "development_only": True,
            },
            "dinov2": {
                "relative_root": "dinov2-small-ed25f3a",
                "manifest_sha256": (
                    "875d184852d921f3a9cbee54f18771640a511bd71d6f47829dc5d6ecf30cf21c"
                ),
                "model_sha256": (
                    "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
                ),
                "config_sha256": (
                    "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
                ),
                "preprocessor_sha256": (
                    "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
                ),
            },
        },
        "files": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(files.items())
        },
    }
    files["packet-manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    archive_path = destination / f"{PACKET_NAME}.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(f"{PACKET_NAME}/{name}", data)
    return archive_path


def _read_json_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object is required: {path}")
    return value


def _evaluate_field_gates_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Evaluate development field gate evidence")
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--operator-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    operator = (
        None
        if args.operator_evidence is None or not args.operator_evidence.is_file()
        else _read_json_object(args.operator_evidence)
    )
    receipt = evaluate_field_gates(
        state_db=args.state_db,
        context=_read_json_object(args.context),
        operator_evidence=operator,
    )
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


def _inspect_worker_publication_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Inspect durable worker publication")
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--jobs-db", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = inspect_worker_publication(
        state_db=args.state_db,
        jobs_db=args.jobs_db,
        session_id=args.session_id,
    )
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt["verified"] else 2


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate-field-gates":
        return _evaluate_field_gates_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect-worker-publication":
        return _inspect_worker_publication_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate", type=Path)
    candidate.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--join-qr-asset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    archive = build_packet(
        project,
        args.output_root,
        candidate_root=args.candidate,
        candidate_archive=args.candidate_archive,
        join_qr_asset=args.join_qr_asset,
    )
    print(
        json.dumps(
            {
                "bytes": archive.stat().st_size,
                "sha256": _sha256(archive.read_bytes()),
                "zip": str(archive),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
