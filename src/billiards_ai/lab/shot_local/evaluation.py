"""Pure manual-truth comparison for shot-local event timelines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _shot_rows(payload: dict[str, Any], *, label: str) -> dict[str, list[dict[str, Any]]]:
    try:
        shots = payload["shots"]
        if not isinstance(shots, list):
            raise TypeError
        result: dict[str, list[dict[str, Any]]] = {}
        for shot in shots:
            shot_id = shot["shot_id"]
            events = shot["events"]
            if not isinstance(shot_id, str) or not isinstance(events, list):
                raise TypeError
            if shot_id in result:
                raise ValueError(f"{label} shot IDs must be unique")
            result[shot_id] = events
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{label} timeline is invalid") from exc


def _score(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
    }


def _reason_value(row: dict[str, Any], prefix: str) -> str | None:
    direct = row.get(prefix)
    if isinstance(direct, str) and direct.strip():
        return direct
    reason = row.get("reason")
    marker = f"{prefix}:"
    if isinstance(reason, str) and reason.startswith(marker) and len(reason) > len(marker):
        return reason[len(marker) :]
    return None


def _matching_key(row: dict[str, Any], event_type: str) -> tuple[str, ...] | None:
    if event_type in {"shot_start", "shot_end"}:
        return ()
    if event_type == "first_contact":
        target = row.get("target_entity")
        return (str(target),) if isinstance(target, str) and target else None
    actor = row.get("actor_entity")
    if not isinstance(actor, str) or not actor:
        return None
    if event_type == "ball_ball_collision":
        target = row.get("target_entity")
        return (actor, str(target)) if isinstance(target, str) and target else None
    if event_type == "cushion":
        rail = _reason_value(row, "rail")
        return (actor, rail) if rail is not None else None
    if event_type == "pot":
        pocket = _reason_value(row, "pocket")
        return (actor, pocket) if pocket is not None else None
    return None


def evaluate_event_timeline(
    timeline: dict[str, Any],
    truth: dict[str, Any],
    *,
    tolerance_ms: int = 120,
) -> dict[str, Any]:
    """Greedily match admitted predictions to manual events by shot, type, and time."""

    if tolerance_ms < 0:
        raise ValueError("event tolerance must be non-negative")
    predicted_shots = _shot_rows(timeline, label="prediction")
    truth_shots = _shot_rows(truth, label="truth")
    counters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    audit: list[dict[str, Any]] = []
    for shot_id in sorted(set(predicted_shots) | set(truth_shots)):
        predicted = [
            row for row in predicted_shots.get(shot_id, []) if row.get("status") == "admitted"
        ]
        expected = list(truth_shots.get(shot_id, []))
        event_types = sorted(
            {str(row.get("event_type")) for row in predicted + expected if row.get("event_type")}
        )
        for event_type in event_types:
            candidates = [row for row in predicted if row.get("event_type") == event_type]
            targets = [row for row in expected if row.get("event_type") == event_type]
            remaining = set(range(len(targets)))
            for row in candidates:
                timestamp = row.get("timestamp_ms")
                matching_key = _matching_key(row, event_type)
                choices = [
                    (abs(int(targets[index]["timestamp_ms"]) - int(timestamp)), index)
                    for index in remaining
                    if timestamp is not None
                    and targets[index].get("timestamp_ms") is not None
                    and matching_key is not None
                    and _matching_key(targets[index], event_type) == matching_key
                ]
                choices.sort()
                if choices and choices[0][0] <= tolerance_ms:
                    delta, index = choices[0]
                    remaining.remove(index)
                    counters[event_type][0] += 1
                    audit.append(
                        {
                            "shot_id": shot_id,
                            "event_type": event_type,
                            "result": "TP",
                            "delta_ms": delta,
                            "matching_key": list(matching_key),
                        }
                    )
                else:
                    counters[event_type][1] += 1
                    audit.append({"shot_id": shot_id, "event_type": event_type, "result": "FP"})
            counters[event_type][2] += len(remaining)
            audit.extend(
                {"shot_id": shot_id, "event_type": event_type, "result": "FN"} for _ in remaining
            )
    by_type = {event_type: _score(*counts) for event_type, counts in sorted(counters.items())}
    overall = _score(
        sum(counts[0] for counts in counters.values()),
        sum(counts[1] for counts in counters.values()),
        sum(counts[2] for counts in counters.values()),
    )
    return {
        "schema_version": "1.0",
        "tolerance_ms": tolerance_ms,
        "overall": overall,
        "by_event_type": by_type,
        "audit": audit,
    }
