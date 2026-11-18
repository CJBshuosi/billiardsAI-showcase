"""Manifest-driven multi-video regression orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from billiards_ai.lab.shot_local.regression import (
    aggregate_regression_cases,
    evaluate_regression_case,
)


@dataclass(frozen=True, slots=True)
class RegressionSuiteArtifact:
    path: Path
    artifact_id: str
    status: str
    case_count: int


@dataclass(frozen=True, slots=True)
class RegressionProductRun:
    """Canonical product path plus the calibration decision that admitted it."""

    path: Path
    calibration_gate: dict[str, Any]


class RegressionCalibrationBlocked(RuntimeError):
    """A regression case stopped at the calibration gate, before perception scoring."""

    def __init__(self, *, status: str, calibration_gate: dict[str, Any]) -> None:
        if status not in {"CALIBRATION_REQUIRED", "CALIBRATION_DRIFTED"}:
            raise ValueError("calibration block status is invalid")
        super().__init__(status)
        self.status = status
        self.calibration_gate = calibration_gate


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid: {path}")
    return value


def _case_path(base: Path, value: object, *, label: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is invalid")
    raw = Path(value)
    path = raw if raw.is_absolute() else base / raw
    if path.is_symlink():
        raise ValueError(f"{label} is unsafe: {path}")
    resolved = path.resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def _number(value: object, *, label: str, lower: float, upper: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < lower or (upper is not None and number > upper):
        raise ValueError(f"{label} is outside the allowed range")
    return number


def _parse_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.is_symlink() or not path.resolve().is_file():
        raise ValueError("suite manifest is missing or unsafe")
    manifest = _load_json(path.resolve(), label="suite manifest")
    if manifest.get("schema_version") != "1.0":
        raise ValueError("suite manifest schema is unsupported")
    suite_id = manifest.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise ValueError("suite ID is invalid")
    frame_tolerance = manifest.get("frame_tolerance")
    if (
        not isinstance(frame_tolerance, int)
        or isinstance(frame_tolerance, bool)
        or frame_tolerance < 0
    ):
        raise ValueError("frame tolerance is invalid")
    gates = manifest.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("suite gates are invalid")
    _number(gates.get("minimum_overall_precision"), label="minimum precision", lower=0, upper=1)
    _number(gates.get("minimum_overall_recall"), label="minimum recall", lower=0, upper=1)
    _number(
        gates.get("minimum_identity_checkpoint_pass_rate"),
        label="minimum identity checkpoint pass rate",
        lower=0,
        upper=1,
    )
    _number(gates.get("maximum_case_wall_seconds"), label="maximum case wall seconds", lower=0)
    by_type = gates.get("minimum_recall_by_event_type", {})
    if not isinstance(by_type, dict):
        raise ValueError("per-event recall gates are invalid")
    for event_type, threshold in by_type.items():
        if event_type not in {"ball_collision", "cushion", "pot"}:
            raise ValueError(f"per-event recall gate is unsupported: {event_type}")
        _number(threshold, label=f"minimum {event_type} recall", lower=0, upper=1)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite requires at least one case")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError("suite case ID is invalid")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("suite case IDs must be unique")
    base = path.resolve().parent
    parsed = []
    for case in cases:
        parsed.append(
            {
                "case_id": case["case_id"],
                "video": _case_path(base, case.get("video"), label="video", directory=False),
                "source_artifact": _case_path(
                    base, case.get("source_artifact"), label="source artifact", directory=True
                ),
                "shot_local_config": _case_path(
                    base, case.get("shot_local_config"), label="shot-local config", directory=False
                ),
                "truth": _case_path(base, case.get("truth"), label="truth", directory=False),
                "capture_setup": _case_path(
                    base,
                    case.get("capture_setup"),
                    label="capture setup",
                    directory=False,
                ),
            }
        )
    return manifest, parsed


def _gate_failures(
    aggregate: dict[str, Any], cases: list[dict[str, Any]], gates: dict[str, Any]
) -> list[str]:
    failures: list[str] = []

    def minimum(metric: object, threshold: float, label: str) -> None:
        if metric is None or float(metric) < threshold:
            failures.append(f"{label} below {threshold}")

    minimum(
        aggregate["overall"]["precision"],
        float(gates["minimum_overall_precision"]),
        "overall precision",
    )
    minimum(
        aggregate["overall"]["recall"],
        float(gates["minimum_overall_recall"]),
        "overall recall",
    )
    minimum(
        aggregate["identity_checkpoints"]["pass_rate"],
        float(gates["minimum_identity_checkpoint_pass_rate"]),
        "identity checkpoint pass rate",
    )
    for event_type, threshold in gates.get("minimum_recall_by_event_type", {}).items():
        metric = aggregate["by_event_type"].get(event_type, {}).get("recall")
        minimum(metric, float(threshold), f"{event_type} recall")
    maximum_wall = float(gates["maximum_case_wall_seconds"])
    for case in cases:
        wall_seconds = case.get("wall_seconds")
        if wall_seconds is not None and float(wall_seconds) > maximum_wall:
            failures.append(f"{case['case_id']} wall seconds above {maximum_wall}")
    return failures


def run_regression_suite(
    manifest_path: Path,
    output_root: Path,
    calibration_root: Path,
    product_runner: Callable[..., RegressionProductRun],
) -> RegressionSuiteArtifact:
    """Run every manifest case and publish one immutable aggregate report."""

    manifest, cases = _parse_manifest(manifest_path)
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    product_root = root / "product-artifacts"
    product_root.mkdir(exist_ok=True)
    resolved_calibration_root = calibration_root.resolve()
    results: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        try:
            product_run = product_runner(
                video=case["video"],
                source_artifact=case["source_artifact"],
                capture_setup=case["capture_setup"],
                calibration_root=resolved_calibration_root,
                config_path=case["shot_local_config"],
                output_root=product_root,
            )
            if not isinstance(product_run, RegressionProductRun):
                raise TypeError("product runner did not return calibration provenance")
            product_path = product_run.path.resolve()
            gate = product_run.calibration_gate
            if gate.get("status") != "READY":
                raise ValueError(f"{case_id} product runner returned a non-ready gate")
            required = {
                name: _load_json(product_path / name, label=f"{case_id} {name}")
                for name in (
                    "world-events.json",
                    "active-tracks.json",
                    "summary.json",
                    "performance.json",
                    "manifest.json",
                )
            }
            product_manifest = required["manifest.json"]
            if (
                product_manifest.get("kind") != "mainline1_shot_local_video"
                or product_manifest.get("pipeline_version") != "shot-local-unified-product-v1"
            ):
                raise ValueError(f"{case_id} is not a canonical Mainline 1 product artifact")
            truth = _load_json(case["truth"], label=f"{case_id} truth")
            evaluation = evaluate_regression_case(
                required["world-events.json"],
                required["active-tracks.json"],
                truth,
                frame_tolerance=manifest["frame_tolerance"],
            )
            summary = required["summary.json"]
            wall_ms = _number(
                required["performance.json"].get("total_artifact_ms"),
                label=f"{case_id} wall seconds",
                lower=0,
            )
            wall_seconds = wall_ms / 1000
            results.append(
                {
                    "case_id": case_id,
                    "status": "PASS",
                    "product_artifact_id": product_manifest.get("artifact_id", product_path.name),
                    "shot_count": summary.get("shot_count"),
                    "wall_seconds": wall_seconds,
                    "calibration_gate": gate,
                    "evaluation": evaluation,
                }
            )
        except RegressionCalibrationBlocked as exc:
            results.append(
                {
                    "case_id": case_id,
                    "status": exc.status,
                    "calibration_gate": exc.calibration_gate,
                    "evaluation": None,
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            results.append({"case_id": case_id, "status": "ERROR", "error": str(exc)})

    aggregate = aggregate_regression_cases(results)
    has_errors = any(case["status"] == "ERROR" for case in results)
    blocked_statuses = {
        case["status"]
        for case in results
        if case["status"] in {"CALIBRATION_REQUIRED", "CALIBRATION_DRIFTED"}
    }
    failures = (
        []
        if has_errors or blocked_statuses
        else _gate_failures(aggregate, results, manifest["gates"])
    )
    if has_errors:
        status = "ERROR"
    elif "CALIBRATION_DRIFTED" in blocked_statuses:
        status = "CALIBRATION_DRIFTED"
    elif "CALIBRATION_REQUIRED" in blocked_statuses:
        status = "CALIBRATION_REQUIRED"
    else:
        status = "FAIL" if failures else "PASS"
    evidence = {
        "suite_manifest_sha256": _sha256(manifest_path.resolve()),
        "truth_sha256": {case["case_id"]: _sha256(case["truth"]) for case in cases},
        "capture_setup_sha256": {case["case_id"]: _sha256(case["capture_setup"]) for case in cases},
        "product_artifact_ids": {
            case["case_id"]: case.get("product_artifact_id") for case in results
        },
    }
    identity = {
        "schema_version": "1.0",
        "suite_id": manifest["suite_id"],
        "evidence": evidence,
    }
    artifact_id = hashlib.sha256(_canonical(identity)).hexdigest()
    destination = root / artifact_id
    if destination.exists():
        report = _load_json(destination / "report.json", label="existing regression report")
        return RegressionSuiteArtifact(
            destination, artifact_id, str(report["status"]), len(results)
        )
    temporary = root / f".pending-{uuid4().hex}"
    report = identity | {
        "artifact_id": artifact_id,
        "status": status,
        "gate_failures": failures,
        "aggregate": aggregate,
        "cases": results,
    }
    try:
        temporary.mkdir()
        (temporary / "report.json").write_bytes(_canonical(report))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return RegressionSuiteArtifact(destination, artifact_id, status, len(results))
