"""Standard-library-only public projection of passive first-shot metadata.

This contract describes bounded producer diagnostics, never authoritative rule,
media, model, or product acceptance. Invalid optional diagnostics are isolated.
"""

from __future__ import annotations

import json
import math
import re

CONTRACT = "offline_first_shot_diagnostics_v1"
MAX_FRAMES = 4096
MAX_BALLS = 128
MAX_EXPORTED_COUNT_ROWS = 128
MAX_BYTES = 16 * 1024
_SAFE_INT = 2**53 - 1
_SHOT = re.compile(r"shot-[0-9]{4,12}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ERROR = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,127}\Z")
_ERROR_TYPES = {
    "ValueError",
    "TypeError",
    "AttributeError",
    "RuntimeError",
    "KeyError",
    "IndexError",
    "OverflowError",
    "MemoryError",
}
_REASONS = {
    "tracker source frames must be contiguous",
    "tracker source gap exceeds safe flow window",
    "tracker source geometry changed",
    "tracker processing p95 exceeds realtime budget",
    "tracker source lag exceeds realtime budget",
}
_ABSTENTIONS = {
    "ring_incomplete",
    "source_frame_gap",
    "tracker_degraded",
    "shot_limit_exceeded",
    "trusted_before_missing",
    "trusted_after_missing",
}
_SEQUENCE_KEYS = {
    "count",
    "inspected_count",
    "truncated",
    "first_frame_id",
    "last_frame_id",
    "inspected_order_sha256",
    "expected_count",
    "expected_range_truncated",
    "missing_count",
    "first_missing_ids",
    "duplicate_count",
    "first_duplicate_indices",
    "reversed_count",
    "first_reversed_indices",
    "outside_interval_count",
    "first_outside_interval_indices",
}
_SHOT_KEYS = {
    "status",
    "shot_id",
    "error_type",
    "interval_start",
    "interval_end",
    "declared_source_count",
    "shot_start",
    "shot_end",
    "source",
    "tracked",
    "rule",
    "active_rule",
    "anchors",
    "publishable",
    "abstention_reason",
    "tracker_snapshot_at_refinement",
    "tracker_policy",
    "unknown_group_counts",
    "unknown_counts_truncated",
    "active_rule_input_truncated",
}
_HEALTH_KEYS = {
    "state",
    "processed_frames",
    "p50_processing_ms",
    "p95_processing_ms",
    "maximum_processing_ms",
    "maximum_source_lag_frames",
    "reason",
}
_SNAPSHOT_KEYS = {
    "state",
    "processed_frames",
    "maximum_source_lag_frames",
    "reason",
    "last_computed_health",
}
_POLICY_KEYS = {
    "allow_accounted_frame_gaps",
    "maximum_accounted_frame_gap",
    "enforce_processing_budget",
    "processing_budget_ms",
    "maximum_source_lag_frames",
    "history_frames",
}


def _require(ok):
    if not ok:
        raise ValueError("invalid first-shot diagnostic metadata")


def _keys(value, names):
    # Check length before constructing a set: oversized dictionaries are not scanned.
    _require(type(value) is dict and len(value) == len(names) and set(value) == names)


def _integer(value, lo=0, hi=_SAFE_INT):
    _require(type(value) is int and lo <= value <= hi)


def _number(value, *, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value))
    _require(0 < value <= 1_000_000_000 if positive else 0 <= value <= 1_000_000_000)


def _choice(value, choices, *, maximum_length=96):
    _require(isinstance(value, str) and len(value) <= maximum_length and value in choices)


def _sequence(value, *, start=None, end=None):
    _keys(value, _SEQUENCE_KEYS)
    count, inspected = value["count"], value["inspected_count"]
    _integer(count)
    _integer(inspected, hi=MAX_FRAMES)
    _require(inspected == min(count, MAX_FRAMES))
    _require(type(value["truncated"]) is bool and value["truncated"] == (count > inspected))
    for name in ("first_frame_id", "last_frame_id"):
        if count:
            _integer(value[name])
        else:
            _require(value[name] is None)
    sha = value["inspected_order_sha256"]
    _require(isinstance(sha, str) and len(sha) == 64 and _SHA.fullmatch(sha))
    expected = None if start is None else end - start + 1
    _require(value["expected_count"] == expected)
    if expected is not None:
        _integer(value["expected_count"], lo=1)
    _require(type(value["expected_range_truncated"]) is bool)
    _require(value["expected_range_truncated"] == (expected is not None and expected > MAX_FRAMES))
    for count_key, examples_key, maximum, lower, upper in (
        ("missing_count", "first_missing_ids", min(expected or 0, MAX_FRAMES), start, end),
        ("duplicate_count", "first_duplicate_indices", max(0, inspected - 1), 0, inspected - 1),
        ("reversed_count", "first_reversed_indices", max(0, inspected - 1), 0, inspected - 1),
        ("outside_interval_count", "first_outside_interval_indices", inspected, 0, inspected - 1),
    ):
        n, examples = value[count_key], value[examples_key]
        _require(type(examples) in (list, tuple) and len(examples) <= 8)
        if count_key == "missing_count" and expected is None:
            _require(n is None and not examples)
            continue
        _integer(n, hi=maximum)
        _require(len(examples) == min(n, 8))
        previous = -1
        for item in examples:
            _integer(item, lo=lower, hi=upper)
            _require(item > previous)
            previous = item
        if count_key == "outside_interval_count" and expected is None:
            _require(n == 0)


def _health(value):
    _keys(value, _HEALTH_KEYS)
    _choice(value["state"], ("running", "degraded"))
    _integer(value["processed_frames"])
    _integer(value["maximum_source_lag_frames"])
    if value["reason"] is not None:
        _choice(value["reason"], _REASONS)
    for key in ("p50_processing_ms", "p95_processing_ms", "maximum_processing_ms"):
        if value[key] is not None:
            _number(value[key])


def _snapshot(value):
    _keys(value, _SNAPSHOT_KEYS)
    _choice(value["state"], ("running", "degraded"))
    _integer(value["processed_frames"])
    _integer(value["maximum_source_lag_frames"])
    if value["reason"] is not None:
        _choice(value["reason"], _REASONS)
    cached = value["last_computed_health"]
    if cached is not None:
        _health(cached)
        _require(cached["processed_frames"] <= value["processed_frames"])
        _require(cached["maximum_source_lag_frames"] <= value["maximum_source_lag_frames"])


def _validate(value, *, exported=False):
    _keys(value, _SHOT_KEYS | ({"unknown_group_counts_omitted"} if exported else set()))
    _require(isinstance(value["shot_id"], str) and len(value["shot_id"]) <= 17)
    _require(_SHOT.fullmatch(value["shot_id"]))
    status = value["status"]
    _choice(status, ("OBSERVED_METADATA", "PARTIAL_LIMIT", "UNAVAILABLE"))
    if status == "UNAVAILABLE":
        error = value["error_type"]
        _require(isinstance(error, str) and len(error) <= 128 and _ERROR.fullmatch(error))
        for key in _SHOT_KEYS - {
            "status",
            "shot_id",
            "error_type",
            "unknown_group_counts",
            "unknown_counts_truncated",
            "active_rule_input_truncated",
        }:
            _require(value[key] is None)
        _require(type(value["unknown_group_counts"]) in (tuple, list))
        _require(not value["unknown_group_counts"])
        _require(value["unknown_counts_truncated"] is False)
        _require(value["active_rule_input_truncated"] is False)
        if exported:
            _integer(value["unknown_group_counts_omitted"], hi=0)
        return
    _require(value["error_type"] is None and type(value["publishable"]) is bool)
    for key in (
        "interval_start",
        "interval_end",
        "shot_start",
        "shot_end",
        "declared_source_count",
    ):
        _integer(value[key])
    start, end = value["interval_start"], value["interval_end"]
    shot_start, shot_end = value["shot_start"], value["shot_end"]
    _require(start <= shot_start <= shot_end <= end)
    if value["abstention_reason"] is not None:
        _choice(value["abstention_reason"], _ABSTENTIONS)
    for key in ("source", "tracked", "rule"):
        if key == "source" and value[key] is None:
            continue
        _sequence(value[key], start=start, end=end)
    _sequence(value["active_rule"], start=shot_start, end=shot_end)
    _sequence(value["anchors"])
    if value["tracker_snapshot_at_refinement"] is not None:
        _snapshot(value["tracker_snapshot_at_refinement"])
    policy = value["tracker_policy"]
    if policy is not None:
        _keys(policy, _POLICY_KEYS)
        for key in ("allow_accounted_frame_gaps", "enforce_processing_budget"):
            _require(type(policy[key]) is bool)
        for key in ("maximum_accounted_frame_gap", "history_frames"):
            _integer(policy[key], lo=2)
        _integer(policy["maximum_source_lag_frames"])
        _number(policy["processing_budget_ms"], positive=True)
    counts = value["unknown_group_counts"]
    cap = MAX_EXPORTED_COUNT_ROWS if exported else MAX_FRAMES
    _require(type(counts) in (tuple, list) and len(counts) <= cap)
    inspected = value["tracked"]["inspected_count"]
    if exported:
        _integer(value["unknown_group_counts_omitted"], hi=MAX_FRAMES)
        _require(len(counts) == min(inspected, MAX_EXPORTED_COUNT_ROWS))
        _require(value["unknown_group_counts_omitted"] == inspected - len(counts))
    else:
        _require(len(counts) == inspected)
    for row in counts:
        _require(type(row) in (tuple, list) and len(row) == 3)
        _integer(row[0])
        _integer(row[1], hi=MAX_BALLS)
        _integer(row[2], lo=row[1], hi=MAX_BALLS)
    for key in ("unknown_counts_truncated", "active_rule_input_truncated"):
        _require(type(value[key]) is bool)
    _require(value["active_rule_input_truncated"] == value["rule"]["truncated"])
    limited = value["unknown_counts_truncated"] or value["active_rule_input_truncated"]
    for key in ("source", "tracked", "rule", "active_rule", "anchors"):
        sequence = value[key]
        if sequence is not None:
            limited |= sequence["truncated"] or sequence["expected_range_truncated"]
    _require((status == "PARTIAL_LIMIT") == limited)


def _envelope(status, diagnostic=None):
    return {"contract": CONTRACT, "status": status, "diagnostic": diagnostic}


def _bounded_envelope(status, diagnostic):
    result = _envelope(status, diagnostic)
    encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    _require(len(encoded) <= MAX_BYTES)
    return result


def project_first_shot_diagnostics(value):
    """Return only this version's public schema; optional errors never escape."""
    if value is None:
        return _envelope("NOT_CAPTURED")
    try:
        _validate(value)
        result = dict(value)
        result["unknown_group_counts"] = value["unknown_group_counts"][:MAX_EXPORTED_COUNT_ROWS]
        result["unknown_group_counts_omitted"] = len(value["unknown_group_counts"]) - len(
            result["unknown_group_counts"]
        )
        # Validation bounded every sequence/string before this copy/encoding.
        encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        _require(len(encoded) <= MAX_BYTES)
        result = json.loads(encoded)
        if result["status"] == "UNAVAILABLE" and result["error_type"] not in _ERROR_TYPES:
            result["error_type"] = "OTHER"
        return _bounded_envelope(result["status"], result)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return _envelope("INVALID")


def validate_first_shot_diagnostics_export(value):
    """Revalidate the driver envelope at the native export boundary."""
    try:
        _keys(value, {"contract", "status", "diagnostic"})
        _require(value["contract"] == CONTRACT)
        if value["status"] in ("NOT_CAPTURED", "INVALID", "BUDGET_OMITTED"):
            _require(value["diagnostic"] is None)
            return _envelope(value["status"])
        diagnostic = value["diagnostic"]
        _validate(diagnostic, exported=True)
        _require(diagnostic["status"] == value["status"])
        encoded = json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        _require(len(encoded) <= MAX_BYTES)
        diagnostic = json.loads(encoded)
        if diagnostic["status"] == "UNAVAILABLE" and diagnostic["error_type"] not in _ERROR_TYPES:
            diagnostic["error_type"] = "OTHER"
        return _bounded_envelope(value["status"], diagnostic)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return _envelope("INVALID")


def fit_first_shot_diagnostics(document, *, owner, render, max_bytes):
    """Drop only optional diagnostics before an existing document size fallback.

    render is the caller's existing exact encoding. No cap is expanded, and no
    pre-existing snapshot/failure is removed to make room for diagnostic data.
    """
    key = "first_shot_diagnostics"
    if key not in owner:
        return
    if len(render(document)) <= max_bytes:
        return
    owner[key] = _envelope("BUDGET_OMITTED")
    if len(render(document)) > max_bytes:
        del owner[key]
