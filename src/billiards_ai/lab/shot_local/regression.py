"""Pure multi-video regression scoring for shot-local product artifacts."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

_SUPPORTED_EVENTS = {"ball_collision", "cushion", "pot"}


def _shots(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    try:
        rows = payload["shots"]
        if not isinstance(rows, list):
            raise TypeError
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            shot_id = row["shot_id"]
            if not isinstance(shot_id, str) or not shot_id:
                raise TypeError
            if shot_id in result:
                raise ValueError(f"{label} shot IDs must be unique")
            result[shot_id] = row
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{label} is invalid") from exc


def _score(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
    }


def _event_key(row: dict[str, Any], event_type: str) -> tuple[str, ...] | None:
    actor = row.get("actor_id")
    if not isinstance(actor, str) or not actor:
        return None
    if event_type == "ball_collision":
        target = row.get("target_id")
        return (actor, target) if isinstance(target, str) and target else None
    if event_type == "cushion":
        rail = row.get("rail")
        return (actor, rail) if isinstance(rail, str) and rail else None
    if event_type == "pot":
        pocket = row.get("pocket")
        return (actor, pocket) if isinstance(pocket, str) and pocket else None
    return None


def _events(row: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    events = row.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"{label} events are invalid")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"{label} events are invalid")
        event_type = event.get("event_type")
        if event_type not in _SUPPORTED_EVENTS:
            raise ValueError(f"{label} event type is unsupported: {event_type}")
        frame_id = event.get("contact_frame_id")
        if not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"{label} event frame is invalid")
        if _event_key(event, event_type) is None:
            raise ValueError(f"{label} event identity is incomplete")
    return events


def _identity_result(
    active_shots: dict[str, dict[str, Any]], truth_shots: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    passed = 0
    audit: list[dict[str, Any]] = []
    total = 0
    for shot_id, truth_shot in truth_shots.items():
        checkpoints = truth_shot.get("identity_checkpoints", [])
        if not isinstance(checkpoints, list):
            raise ValueError("truth identity checkpoints are invalid")
        tracks = active_shots.get(shot_id, {}).get("tracks", [])
        if not isinstance(tracks, list):
            raise ValueError("active tracks are invalid")
        samples: dict[tuple[str, int], dict[str, Any]] = {}
        for track in tracks:
            try:
                track_id = track["track_id"]
                track_samples = track["samples"]
                if not isinstance(track_id, str) or not isinstance(track_samples, list):
                    raise TypeError
                for sample in track_samples:
                    frame_id = sample["frame_id"]
                    if not isinstance(frame_id, int):
                        raise TypeError
                    key = (track_id, frame_id)
                    if key in samples:
                        raise ValueError("active track samples must be unique")
                    samples[key] = sample
            except (KeyError, TypeError) as exc:
                raise ValueError("active tracks are invalid") from exc
        for checkpoint in checkpoints:
            total += 1
            try:
                track_id = checkpoint["track_id"]
                frame_id = checkpoint["frame_id"]
                expected_x = float(checkpoint["x_mm"])
                expected_y = float(checkpoint["y_mm"])
                tolerance = float(checkpoint["tolerance_mm"])
                if (
                    not isinstance(track_id, str)
                    or not isinstance(frame_id, int)
                    or frame_id < 0
                    or tolerance < 0
                    or not all(math.isfinite(v) for v in (expected_x, expected_y, tolerance))
                ):
                    raise TypeError
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("truth identity checkpoint is invalid") from exc
            sample = samples.get((track_id, frame_id))
            reason = "missing_track_sample"
            distance = None
            if sample is not None:
                try:
                    distance = math.hypot(
                        float(sample["x_mm"]) - expected_x,
                        float(sample["y_mm"]) - expected_y,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("active track sample position is invalid") from exc
                reason = "matched" if distance <= tolerance else "position_mismatch"
            if reason == "matched":
                passed += 1
            audit.append(
                {
                    "shot_id": shot_id,
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "result": "PASS" if reason == "matched" else "FAIL",
                    "reason": reason,
                    "distance_mm": distance,
                    "tolerance_mm": tolerance,
                }
            )
    failed = total - passed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": None if total == 0 else passed / total,
        "audit": audit,
    }


def evaluate_regression_case(
    world_events: dict[str, Any],
    active_tracks: dict[str, Any],
    truth: dict[str, Any],
    *,
    frame_tolerance: int = 2,
) -> dict[str, Any]:
    """Compare one product artifact with reviewed events and identity checkpoints."""

    if frame_tolerance < 0:
        raise ValueError("frame tolerance must be non-negative")
    predicted_shots = _shots(world_events, label="prediction")
    active_shots = _shots(active_tracks, label="active tracks")
    truth_shots = _shots(truth, label="truth")
    reviewed_count = sum(
        len(_events(row, label="truth")) + len(row.get("identity_checkpoints", []))
        for row in truth_shots.values()
    )
    if reviewed_count == 0:
        raise ValueError("truth requires at least one manual reviewed fact")

    counters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    audit: list[dict[str, Any]] = []
    for shot_id in sorted(set(predicted_shots) | set(truth_shots)):
        candidates = [
            event
            for event in _events(predicted_shots.get(shot_id, {}), label="prediction")
            if event.get("status") == "observed"
        ]
        expected = _events(truth_shots.get(shot_id, {}), label="truth")
        event_types = sorted({event["event_type"] for event in candidates + expected})
        for event_type in event_types:
            predicted_type = [event for event in candidates if event["event_type"] == event_type]
            truth_type = [event for event in expected if event["event_type"] == event_type]
            remaining = set(range(len(truth_type)))
            for event in predicted_type:
                key = _event_key(event, event_type)
                choices = sorted(
                    (
                        abs(truth_type[index]["contact_frame_id"] - event["contact_frame_id"]),
                        index,
                    )
                    for index in remaining
                    if _event_key(truth_type[index], event_type) == key
                )
                if choices and choices[0][0] <= frame_tolerance:
                    delta, index = choices[0]
                    remaining.remove(index)
                    counters[event_type][0] += 1
                    audit.append(
                        {
                            "shot_id": shot_id,
                            "event_type": event_type,
                            "result": "TP",
                            "delta_frames": delta,
                            "matching_key": list(key or ()),
                        }
                    )
                else:
                    counters[event_type][1] += 1
                    audit.append({"shot_id": shot_id, "event_type": event_type, "result": "FP"})
            counters[event_type][2] += len(remaining)
            audit.extend(
                {"shot_id": shot_id, "event_type": event_type, "result": "FN"} for _ in remaining
            )

    by_type = {name: _score(*counts) for name, counts in sorted(counters.items())}
    overall = _score(
        sum(counts[0] for counts in counters.values()),
        sum(counts[1] for counts in counters.values()),
        sum(counts[2] for counts in counters.values()),
    )
    return {
        "schema_version": "1.0",
        "frame_tolerance": frame_tolerance,
        "overall": overall,
        "by_event_type": by_type,
        "identity_checkpoints": _identity_result(active_shots, truth_shots),
        "audit": audit,
    }


def aggregate_regression_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-aggregate completed case evaluations without hiding invalid cases."""

    counters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    identity_total = identity_passed = 0
    for case in cases:
        evaluation = case.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        for event_type, score in evaluation.get("by_event_type", {}).items():
            counters[event_type][0] += int(score["tp"])
            counters[event_type][1] += int(score["fp"])
            counters[event_type][2] += int(score["fn"])
        identity = evaluation["identity_checkpoints"]
        identity_total += int(identity["total"])
        identity_passed += int(identity["passed"])
    overall = _score(
        sum(row[0] for row in counters.values()),
        sum(row[1] for row in counters.values()),
        sum(row[2] for row in counters.values()),
    )
    return {
        "overall": overall,
        "by_event_type": {name: _score(*counts) for name, counts in sorted(counters.items())},
        "identity_checkpoints": {
            "total": identity_total,
            "passed": identity_passed,
            "failed": identity_total - identity_passed,
            "pass_rate": None if identity_total == 0 else identity_passed / identity_total,
        },
    }
