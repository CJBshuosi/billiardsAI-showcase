"""Blind set-level evaluator and frozen human-review sample for SAM2 pilot runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import (
    InstanceAbsence,
    InstancePrompt,
    MaskObservation,
    PilotQaIdentity,
    PropagationResult,
)
from billiards_ai.lab.propagation.pilot_artifacts import (
    PilotArtifactError,
    PilotPayloadInventoryItem,
    PilotRunManifest,
    verify_pilot_run_evidence,
)
from billiards_ai.lab.propagation.pilot_data import (
    FROZEN_CIRCLE_COUNT,
    FROZEN_EVALUATION_FRAME_COUNT,
    FROZEN_PROMPT_FRAME_COUNT,
    FROZEN_REVIEW_FRAME_COUNT,
    FROZEN_SHOT_COUNT,
    FROZEN_VIDEO_IDS,
    EvaluationCircle,
    EvaluationManifest,
    PilotDataError,
    verify_prompt_manifest,
)
from billiards_ai.lab.propagation.pilot_planning import WindowPlan

ReviewStratum = Literal["clean_match", "flagged_or_ambiguous_match", "miss", "false_positive"]
ReviewUnitType = Literal["matched_pair", "unmatched_ground_truth", "unmatched_prediction"]

_RAW_STRUCTURED_FILES = frozenset(
    {
        "manifest.json",
        "inventory.json",
        "prompts.jsonl",
        "observations.jsonl",
        "absences.jsonl",
        "links.jsonl",
        "qa.jsonl",
        "results.jsonl",
        "failures.jsonl",
        "telemetry.jsonl",
        "plan.json",
        "coverage.jsonl",
    }
)
_QUOTAS: dict[ReviewStratum, int] = {
    "clean_match": 40,
    "flagged_or_ambiguous_match": 30,
    "miss": 15,
    "false_positive": 15,
}


class PilotEvaluationError(RuntimeError):
    """Evaluation inputs or immutable artifacts failed integrity validation."""


class EvaluationMatch(FrozenContract):
    ground_truth: EvaluationCircle
    prediction: MaskObservation
    normalized_center_error: float = Field(ge=0)


class FrameAssignment(FrozenContract):
    matches: tuple[EvaluationMatch, ...] = ()
    unmatched_ground_truth: tuple[EvaluationCircle, ...] = ()
    unmatched_predictions: tuple[MaskObservation, ...] = ()


class SetMetrics(FrozenContract):
    ground_truth_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    true_positive_support: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    recall: float = Field(ge=0, le=1)
    median_normalized_center_error: float | None = Field(default=None, ge=0)
    p90_normalized_center_error: float | None = Field(default=None, ge=0)


class AutomatedGate(FrozenContract):
    value: float | int | None
    threshold: float
    comparison: Literal[">=", "<="]
    passed: bool


class HumanGate(FrozenContract):
    threshold: float
    comparison: Literal[">=", "<="]
    status: Literal["pending_human_review"] = "pending_human_review"


class FrozenGates(FrozenContract):
    coverage: AutomatedGate
    recall: AutomatedGate
    median_center_error: AutomatedGate
    p90_center_error: AutomatedGate
    severe_drift_or_wrong_target: HumanGate
    acceptable_without_redraw: HumanGate
    automated_gates_passed: bool
    pilot_status: Literal["automated_gate_failed", "pending_human_review"]


class ReviewUnit(FrozenContract):
    unit_id: NonEmptyStr
    unit_type: ReviewUnitType
    stratum: ReviewStratum
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    ground_truth_id: str | None = None
    prediction_id: str | None = None
    ground_truth: EvaluationCircle | None = None
    prediction: MaskObservation | None = None
    normalized_center_error: float | None = Field(default=None, ge=0)
    mask_ref: str | None = None
    qa_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def shape_matches_unit_type(self) -> ReviewUnit:
        if self.unit_type == "matched_pair" and (
            self.ground_truth_id is None or self.prediction_id is None
        ):
            raise ValueError("matched review units require both identities")
        if self.unit_type == "unmatched_ground_truth" and (
            self.ground_truth_id is None or self.prediction_id is not None
        ):
            raise ValueError("miss review units require only a ground-truth identity")
        if self.unit_type == "unmatched_prediction" and (
            self.ground_truth_id is not None or self.prediction_id is None
        ):
            raise ValueError("false-positive review units require only a prediction identity")
        expected_strata: dict[ReviewUnitType, set[ReviewStratum]] = {
            "matched_pair": {"clean_match", "flagged_or_ambiguous_match"},
            "unmatched_ground_truth": {"miss"},
            "unmatched_prediction": {"false_positive"},
        }
        if self.stratum not in expected_strata[self.unit_type]:
            raise ValueError("review unit stratum does not match its unit type")
        if self.ground_truth is not None and (
            self.ground_truth.annotation_id != self.ground_truth_id
            or self.ground_truth.video_id != self.asset_id
            or self.ground_truth.source_frame_index != self.frame_index
        ):
            raise ValueError("review unit ground-truth evidence does not match its identity")
        if self.prediction is not None and (
            self.prediction.instance_id != self.prediction_id
            or self.prediction.asset_id != self.asset_id
            or self.prediction.frame_index != self.frame_index
        ):
            raise ValueError("review unit prediction evidence does not match its identity")
        return self


class ReviewSample(FrozenContract):
    population_counts: dict[ReviewStratum, int]
    primary_quota_sample_counts: dict[ReviewStratum, int]
    sample_counts: dict[ReviewStratum, int]
    units: tuple[ReviewUnit, ...]


class CoverageSummary(FrozenContract):
    completed_shots: int = Field(ge=0)
    total_shots: int = Field(ge=0)
    failed_shots: int = Field(ge=0)


class ExploratoryTrackContinuity(FrozenContract):
    status: Literal["not_evaluated"] = "not_evaluated"
    included_in_set_metrics: Literal[False] = False


class EvaluationReport(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    raw_run_hash: Sha256
    evaluation_manifest_hash: Sha256
    dataset_id: Sha256
    prompt_manifest_id: Sha256
    input_bundle_id: Sha256
    qa_identity: PilotQaIdentity = Field(
        default_factory=lambda: PilotQaIdentity(artifact_kind="evaluation_report")
    )
    raw_outcome: Literal["all_shots_success", "partial_success", "failed"]
    coverage: CoverageSummary
    set_metrics: SetMetrics
    frozen_gates: FrozenGates
    failed_shot_count: int = Field(ge=0)
    explicit_absence_count: int = Field(ge=0)
    exploratory_track_continuity: ExploratoryTrackContinuity = Field(
        default_factory=ExploratoryTrackContinuity
    )
    review_population_counts: dict[ReviewStratum, int]

    @model_validator(mode="after")
    def qa_kind_is_evaluation(self) -> EvaluationReport:
        if self.qa_identity.artifact_kind != "evaluation_report":
            raise ValueError("evaluation report must use evaluation_report QA identity")
        return self


class PilotEvaluationResult(FrozenContract):
    report: EvaluationReport
    evaluation_path: Path
    review_sample_path: Path


class _LoadedRawRun(FrozenContract):
    manifest: PilotRunManifest
    observations: tuple[MaskObservation, ...]
    absences: tuple[InstanceAbsence, ...]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distance(ground_truth: EvaluationCircle, prediction: MaskObservation) -> float:
    return (
        math.hypot(
            prediction.center_px[0] - ground_truth.center_x_px,
            prediction.center_px[1] - ground_truth.center_y_px,
        )
        / ground_truth.radius_px
    )


def _hungarian(costs: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return the minimum-cost column for each row of a rectangular integer matrix."""

    if not costs:
        return ()
    row_count = len(costs)
    column_count = len(costs[0])
    if column_count < row_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment cost matrix must be rectangular with enough columns")
    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta: int | float = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            if math.isinf(delta):  # pragma: no cover - every row has a private dummy column.
                raise ValueError("assignment matrix has no complete solution")
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += int(delta)
                    column_potential[column] -= int(delta)
                else:
                    minimum[column] -= int(delta)
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while current_column:
            previous = predecessor[current_column]
            matched_row[current_column] = matched_row[previous]
            current_column = previous
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return tuple(assignment)


def match_frame(
    ground_truth: Sequence[EvaluationCircle], predictions: Sequence[MaskObservation]
) -> FrameAssignment:
    """Return a deterministic maximum-cardinality, minimum-cost same-class assignment."""

    grounds = tuple(sorted(ground_truth, key=lambda item: item.annotation_id))
    predicted = tuple(sorted(predictions, key=lambda item: item.instance_id))
    if grounds:
        frame_key = (grounds[0].video_id, grounds[0].source_frame_index)
        if any((item.video_id, item.source_frame_index) != frame_key for item in grounds):
            raise ValueError("ground truth must belong to one frame")
        if any((item.asset_id, item.frame_index) != frame_key for item in predicted):
            raise ValueError("predictions must belong to the ground-truth frame")

    distances = tuple(tuple(_distance(gt, pred) for pred in predicted) for gt in grounds)
    column_count = len(predicted) + len(grounds)
    radix = column_count + 1
    tie_span = radix ** len(grounds)
    distance_scale = 10**12
    unmatched_distance = (len(grounds) + 1) * distance_scale
    costs: list[list[int]] = []
    for ground_index, row in enumerate(distances):
        positional_weight = radix ** (len(grounds) - ground_index - 1)
        costs.append(
            [
                (
                    round(distance * distance_scale) * tie_span
                    if predicted[prediction_index].class_name == "ball" and distance <= 1.0
                    else unmatched_distance * 2 * tie_span
                )
                + prediction_index * positional_weight
                for prediction_index, distance in enumerate(row)
            ]
            + [
                unmatched_distance * tie_span + dummy_column * positional_weight
                for dummy_column in range(len(predicted), column_count)
            ]
        )
    columns = _hungarian(costs)
    pairs = tuple(
        (ground_index, column)
        for ground_index, column in enumerate(columns)
        if column < len(predicted)
        and predicted[column].class_name == "ball"
        and distances[ground_index][column] <= 1.0
    )
    ground_matched = {ground_index for ground_index, _ in pairs}
    prediction_matched = {prediction_index for _, prediction_index in pairs}
    matches = tuple(
        EvaluationMatch(
            ground_truth=grounds[ground_index],
            prediction=predicted[prediction_index],
            normalized_center_error=distances[ground_index][prediction_index],
        )
        for ground_index, prediction_index in pairs
    )
    return FrameAssignment(
        matches=matches,
        unmatched_ground_truth=tuple(
            item for index, item in enumerate(grounds) if index not in ground_matched
        ),
        unmatched_predictions=tuple(
            item for index, item in enumerate(predicted) if index not in prediction_matched
        ),
    )


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def compute_set_metrics(assignments: Sequence[FrameAssignment]) -> SetMetrics:
    matches = tuple(match for assignment in assignments for match in assignment.matches)
    misses = sum(len(item.unmatched_ground_truth) for item in assignments)
    false_positives = sum(len(item.unmatched_predictions) for item in assignments)
    ground_truth_count = len(matches) + misses
    errors = [item.normalized_center_error for item in matches]
    return SetMetrics(
        ground_truth_count=ground_truth_count,
        prediction_count=len(matches) + false_positives,
        true_positive_support=len(matches),
        false_negative_count=misses,
        false_positive_count=false_positives,
        recall=len(matches) / ground_truth_count if ground_truth_count else 0.0,
        median_normalized_center_error=float(statistics.median(errors)) if errors else None,
        p90_normalized_center_error=_nearest_rank(errors, 0.9),
    )


def compute_frozen_gates(
    *, metrics: SetMetrics, completed_shots: int, total_shots: int
) -> FrozenGates:
    coverage_value = completed_shots / total_shots if total_shots else 0.0
    median = metrics.median_normalized_center_error
    p90 = metrics.p90_normalized_center_error
    gates = {
        "coverage": coverage_value >= 24 / 25,
        "recall": metrics.recall >= 0.90,
        "median": median is not None and median <= 0.5,
        "p90": p90 is not None and p90 <= 1.0,
    }
    automated = all(gates.values())
    return FrozenGates(
        coverage=AutomatedGate(
            value=coverage_value, threshold=24 / 25, comparison=">=", passed=gates["coverage"]
        ),
        recall=AutomatedGate(
            value=metrics.recall, threshold=0.90, comparison=">=", passed=gates["recall"]
        ),
        median_center_error=AutomatedGate(
            value=median, threshold=0.5, comparison="<=", passed=gates["median"]
        ),
        p90_center_error=AutomatedGate(
            value=p90, threshold=1.0, comparison="<=", passed=gates["p90"]
        ),
        severe_drift_or_wrong_target=HumanGate(threshold=0.05, comparison="<="),
        acceptable_without_redraw=HumanGate(threshold=0.60, comparison=">="),
        automated_gates_passed=automated,
        pilot_status="pending_human_review" if automated else "automated_gate_failed",
    )


def _stable_review_key(unit: ReviewUnit) -> tuple[str, str]:
    return hashlib.sha256(_canonical_bytes(unit.model_dump(mode="json"))).hexdigest(), unit.unit_id


def freeze_review_sample(units: Sequence[ReviewUnit]) -> ReviewSample:
    by_stratum: dict[ReviewStratum, list[ReviewUnit]] = {
        "clean_match": [],
        "flagged_or_ambiguous_match": [],
        "miss": [],
        "false_positive": [],
    }
    seen: set[str] = set()
    for unit in units:
        if unit.unit_id in seen:
            raise ValueError("review unit identities must be unique")
        seen.add(unit.unit_id)
        by_stratum[unit.stratum].append(unit)
    for population in by_stratum.values():
        population.sort(key=_stable_review_key)
    selected: list[ReviewUnit] = []
    primary_counts: dict[ReviewStratum, int] = {}
    leftovers: list[ReviewUnit] = []
    for stratum, quota in _QUOTAS.items():
        primary = by_stratum[stratum][:quota]
        selected.extend(primary)
        primary_counts[stratum] = len(primary)
        leftovers.extend(by_stratum[stratum][quota:])
    leftovers.sort(key=_stable_review_key)
    selected.extend(leftovers[: max(0, 100 - len(selected))])
    selected.sort(key=_stable_review_key)
    sample_counts = {stratum: 0 for stratum in _QUOTAS}
    for unit in selected:
        sample_counts[unit.stratum] += 1
    return ReviewSample(
        population_counts={stratum: len(population) for stratum, population in by_stratum.items()},
        primary_quota_sample_counts=primary_counts,
        sample_counts=sample_counts,
        units=tuple(selected),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotEvaluationError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise PilotEvaluationError(f"JSON object required: {path.name}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotEvaluationError(f"invalid JSONL artifact: {path.name}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise PilotEvaluationError(f"JSONL object rows required: {path.name}")
    return rows


def _safe_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
        raise PilotEvaluationError("raw-run inventory path is unsafe")
    candidate = root.joinpath(*path.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise PilotEvaluationError(f"raw-run inventory payload is missing: {relative}")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PilotEvaluationError("raw-run inventory payload escapes its root") from exc
    return candidate


def _load_raw_run(raw_run: Path) -> _LoadedRawRun:
    raw_run = raw_run.absolute()
    if raw_run.is_symlink() or not raw_run.is_dir() or len(raw_run.name) != 64:
        raise PilotEvaluationError("raw run must be a content-addressed regular directory")
    try:
        int(raw_run.name, 16)
    except ValueError as exc:
        raise PilotEvaluationError("raw run directory name is not a SHA-256 identity") from exc
    entries = (raw_run, *raw_run.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise PilotEvaluationError("raw run contains a symlink")
    if os.name != "nt" and any(stat.S_IMODE(path.stat().st_mode) & 0o222 for path in entries):
        raise PilotEvaluationError("raw run is mutable")
    stored_manifest = _read_json(raw_run / "manifest.json")
    if stored_manifest.get("content_hash") != raw_run.name:
        raise PilotEvaluationError("raw-run content identity mismatch")
    try:
        manifest_payload = {
            key: value for key, value in stored_manifest.items() if key != "content_hash"
        }
        manifest_payload["plans"] = tuple(
            WindowPlan.model_validate(
                {key: value for key, value in plan.items() if key != "plan_hash"}
            )
            for plan in manifest_payload["plans"]
        )
        manifest = PilotRunManifest.model_validate(manifest_payload)
    except (PilotDataError, TypeError, ValueError) as exc:
        raise PilotEvaluationError("raw-run manifest is structurally incomplete") from exc
    inventory_raw = _read_json(raw_run / "inventory.json")
    if (
        set(inventory_raw) != {"schema_version", "files"}
        or inventory_raw["schema_version"] != "1.0"
    ):
        raise PilotEvaluationError("raw-run inventory schema is invalid")
    try:
        inventory = tuple(
            PilotPayloadInventoryItem.model_validate(item) for item in inventory_raw["files"]
        )
    except (TypeError, ValueError) as exc:
        raise PilotEvaluationError("raw-run inventory entries are invalid") from exc
    inventory_paths = [item.path for item in inventory]
    if len(inventory_paths) != len(set(inventory_paths)):
        raise PilotEvaluationError("raw-run inventory contains duplicate paths")
    actual_files = {
        path.relative_to(raw_run).as_posix() for path in raw_run.rglob("*") if path.is_file()
    }
    if actual_files != _RAW_STRUCTURED_FILES | set(inventory_paths):
        raise PilotEvaluationError("raw-run payload inventory is not exact")
    for item in inventory:
        payload = _safe_file(raw_run, item.path)
        if payload.stat().st_size != item.bytes or _sha256(payload) != item.sha256:
            raise PilotEvaluationError(f"raw-run inventory mismatch: {item.path}")
    try:
        tuple(InstancePrompt.model_validate(row) for row in _read_jsonl(raw_run / "prompts.jsonl"))
        results = tuple(
            PropagationResult.model_validate(row) for row in _read_jsonl(raw_run / "results.jsonl")
        )
        observations = tuple(
            MaskObservation.model_validate(row)
            for row in _read_jsonl(raw_run / "observations.jsonl")
        )
        absences = tuple(
            InstanceAbsence.model_validate(row) for row in _read_jsonl(raw_run / "absences.jsonl")
        )
    except (TypeError, ValueError) as exc:
        raise PilotEvaluationError("raw-run structured propagation rows are invalid") from exc
    links = _read_jsonl(raw_run / "links.jsonl")
    qa = _read_jsonl(raw_run / "qa.jsonl")
    failures = _read_jsonl(raw_run / "failures.jsonl")
    telemetry = _read_jsonl(raw_run / "telemetry.jsonl")
    coverage = _read_jsonl(raw_run / "coverage.jsonl")
    plan = _read_json(raw_run / "plan.json")
    expected_plan = {
        "plans": sorted(
            (item.model_dump(mode="json") for item in manifest.plans), key=_canonical_bytes
        )
    }
    if plan != expected_plan:
        raise PilotEvaluationError("raw-run plan evidence does not match its manifest")
    observation_keys = {
        (item.asset_id, item.frame_index, item.instance_id) for item in observations
    }
    if len(observation_keys) != len(observations):
        raise PilotEvaluationError("raw-run observations are not unique")
    ordered_results = tuple(
        sorted(results, key=lambda item: _canonical_bytes(item.model_dump(mode="json")))
    )
    try:
        required_references = verify_pilot_run_evidence(manifest, ordered_results)
    except (PilotArtifactError, TypeError, ValueError) as exc:
        raise PilotEvaluationError("raw-run results do not match the active plan") from exc
    if tuple(inventory_paths) != required_references:
        raise PilotEvaluationError("raw-run inventory does not match required payload references")
    inventory_payload = [item.model_dump(mode="json") for item in inventory]
    identity = {
        "schema_version": "1.0",
        "manifest": manifest.model_dump(mode="json"),
        "results": [item.model_dump(mode="json") for item in ordered_results],
        "inventory": inventory_payload,
    }
    if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != raw_run.name:
        raise PilotEvaluationError("raw-run content identity does not verify")
    expected_prompts = sorted(
        (prompt.model_dump(mode="json") for result in ordered_results for prompt in result.prompts),
        key=_canonical_bytes,
    )
    expected_observations = sorted(
        (
            observation.model_dump(mode="json")
            for result in ordered_results
            for observation in result.observations
        ),
        key=_canonical_bytes,
    )
    expected_absences = sorted(
        (
            absence.model_dump(mode="json")
            for result in ordered_results
            for absence in result.absences
        ),
        key=_canonical_bytes,
    )
    expected_qa = sorted(
        (
            [
                {
                    "kind": "observation",
                    "asset_id": item["asset_id"],
                    "frame_index": item["frame_index"],
                    "instance_id": item["instance_id"],
                    "qa_flags": item["qa_flags"],
                }
                for item in expected_observations
            ]
            + [
                {
                    "kind": "absence",
                    "asset_id": item["asset_id"],
                    "frame_index": item["frame_index"],
                    "instance_id": item["instance_id"],
                    "qa_flags": item["qa_flags"],
                }
                for item in expected_absences
            ]
        ),
        key=_canonical_bytes,
    )
    expected_failures = sorted(
        (
            [
                {"kind": "worker_attempt", **item.model_dump(mode="json")}
                for item in manifest.attempt_failures
            ]
            + [{"kind": "shot", **item.model_dump(mode="json")} for item in manifest.failed_shots]
            + [
                {"kind": "propagation", **failure.model_dump(mode="json")}
                for result in ordered_results
                for failure in result.failures
            ]
        ),
        key=_canonical_bytes,
    )
    expected_links = sorted(
        (item.model_dump(mode="json") for item in manifest.links), key=_canonical_bytes
    )
    expected_telemetry = sorted(manifest.telemetry, key=_canonical_bytes)
    expected_coverage = sorted(
        (item.coverage_proof.model_dump(mode="json") for item in manifest.plans),
        key=_canonical_bytes,
    )
    stored_rows = {
        "prompts": list(_read_jsonl(raw_run / "prompts.jsonl")),
        "observations": list(_read_jsonl(raw_run / "observations.jsonl")),
        "absences": list(_read_jsonl(raw_run / "absences.jsonl")),
        "results": list(_read_jsonl(raw_run / "results.jsonl")),
        "links": list(links),
        "qa": list(qa),
        "failures": list(failures),
        "telemetry": list(telemetry),
        "coverage": list(coverage),
    }
    expected_rows = {
        "prompts": expected_prompts,
        "observations": expected_observations,
        "absences": expected_absences,
        "results": [item.model_dump(mode="json") for item in ordered_results],
        "links": expected_links,
        "qa": expected_qa,
        "failures": expected_failures,
        "telemetry": expected_telemetry,
        "coverage": expected_coverage,
    }
    if stored_rows != expected_rows:
        raise PilotEvaluationError("raw-run structured evidence does not verify")
    return _LoadedRawRun(manifest=manifest, observations=observations, absences=absences)


def _load_evaluation_manifest(path: Path) -> EvaluationManifest:
    try:
        manifest = EvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PilotEvaluationError("evaluation manifest is invalid") from exc
    payload = manifest.model_dump(mode="json", exclude={"manifest_id"})
    if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != manifest.manifest_id:
        raise PilotEvaluationError("evaluation manifest content identity mismatch")
    if (
        path.parent.name not in {"", "."}
        and len(path.parent.name) == 64
        and path.parent.name != manifest.manifest_id
    ):
        raise PilotEvaluationError("evaluation manifest path identity mismatch")
    return manifest


def _verify_frozen_evaluation_population(
    *, manifest: EvaluationManifest, raw_manifest: PilotRunManifest
) -> None:
    prompt = raw_manifest.prompt_manifest
    try:
        verify_prompt_manifest(prompt)
    except (PilotDataError, TypeError, ValueError) as exc:
        raise PilotEvaluationError("raw prompt manifest snapshot is invalid") from exc
    if len(prompt.shots) != FROZEN_SHOT_COUNT or len(manifest.shots) != FROZEN_SHOT_COUNT:
        raise PilotEvaluationError("frozen evaluation requires exactly 25 shots")
    if {shot.video_id for shot in prompt.shots} != set(FROZEN_VIDEO_IDS) or {
        shot.video_id for shot in manifest.shots
    } != set(FROZEN_VIDEO_IDS):
        raise PilotEvaluationError("frozen evaluation uses an unapproved video set")
    if manifest.shots != prompt.shots:
        raise PilotEvaluationError("evaluation shots do not match the frozen prompt shots")
    prompt_frames = {(circle.video_id, circle.source_frame_index) for circle in prompt.circles}
    if len(prompt_frames) != FROZEN_PROMPT_FRAME_COUNT:
        raise PilotEvaluationError("frozen prompt frame count is invalid")
    evaluation_frames = {
        (circle.video_id, circle.source_frame_index) for circle in manifest.circles
    }
    if len(evaluation_frames) != FROZEN_EVALUATION_FRAME_COUNT:
        raise PilotEvaluationError("frozen evaluation requires exactly 50 held-out frames")
    if len(prompt_frames | evaluation_frames) != FROZEN_REVIEW_FRAME_COUNT:
        raise PilotEvaluationError("frozen prompt/evaluation frame relationship is invalid")
    if len(prompt.circles) + len(manifest.circles) != FROZEN_CIRCLE_COUNT:
        raise PilotEvaluationError("frozen prompt/evaluation circle population is invalid")
    role_frames: dict[tuple[str, int], dict[str, set[int]]] = {}
    for circle in manifest.circles:
        roles = role_frames.setdefault(
            (circle.video_id, circle.shot_index), {"middle": set(), "end": set()}
        )
        roles[circle.role].add(circle.source_frame_index)
    shot_keys = {(shot.video_id, shot.shot_index) for shot in manifest.shots}
    if set(role_frames) != shot_keys:
        raise PilotEvaluationError("every frozen shot requires held-out role evidence")
    shots_by_key = {(shot.video_id, shot.shot_index): shot for shot in manifest.shots}
    for key, roles in role_frames.items():
        if len(roles["middle"]) != 1 or len(roles["end"]) != 1:
            raise PilotEvaluationError("each frozen shot requires one middle and one end frame")
        middle = next(iter(roles["middle"]))
        end = next(iter(roles["end"]))
        shot = shots_by_key[key]
        if not shot.start_frame < middle < end < shot.end_frame:
            raise PilotEvaluationError("frozen evaluation role frames are out of order")


def _unit_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _review_units(assignments: Sequence[FrameAssignment]) -> tuple[ReviewUnit, ...]:
    units: list[ReviewUnit] = []
    for assignment in assignments:
        for match in assignment.matches:
            stratum: ReviewStratum = (
                "flagged_or_ambiguous_match" if match.prediction.qa_flags else "clean_match"
            )
            payload = {
                "unit_type": "matched_pair",
                "ground_truth_id": match.ground_truth.annotation_id,
                "prediction_id": match.prediction.instance_id,
                "asset_id": match.prediction.asset_id,
                "frame_index": match.prediction.frame_index,
            }
            units.append(
                ReviewUnit(
                    unit_id=_unit_id(payload),
                    unit_type="matched_pair",
                    stratum=stratum,
                    asset_id=match.prediction.asset_id,
                    frame_index=match.prediction.frame_index,
                    ground_truth_id=match.ground_truth.annotation_id,
                    prediction_id=match.prediction.instance_id,
                    ground_truth=match.ground_truth,
                    prediction=match.prediction,
                    normalized_center_error=match.normalized_center_error,
                    mask_ref=match.prediction.mask_ref,
                    qa_flags=match.prediction.qa_flags,
                )
            )
        for ground_truth in assignment.unmatched_ground_truth:
            payload = {
                "unit_type": "unmatched_ground_truth",
                "ground_truth_id": ground_truth.annotation_id,
                "asset_id": ground_truth.video_id,
                "frame_index": ground_truth.source_frame_index,
            }
            units.append(
                ReviewUnit(
                    unit_id=_unit_id(payload),
                    unit_type="unmatched_ground_truth",
                    stratum="miss",
                    asset_id=ground_truth.video_id,
                    frame_index=ground_truth.source_frame_index,
                    ground_truth_id=ground_truth.annotation_id,
                    ground_truth=ground_truth,
                )
            )
        for prediction in assignment.unmatched_predictions:
            payload = {
                "unit_type": "unmatched_prediction",
                "prediction_id": prediction.instance_id,
                "asset_id": prediction.asset_id,
                "frame_index": prediction.frame_index,
                "mask_ref": prediction.mask_ref,
            }
            units.append(
                ReviewUnit(
                    unit_id=_unit_id(payload),
                    unit_type="unmatched_prediction",
                    stratum="false_positive",
                    asset_id=prediction.asset_id,
                    frame_index=prediction.frame_index,
                    prediction_id=prediction.instance_id,
                    prediction=prediction,
                    mask_ref=prediction.mask_ref,
                    qa_flags=prediction.qa_flags,
                )
            )
    return tuple(units)


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise PilotEvaluationError("pending evaluation artifact contains a symlink")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _remove_temporary(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), reverse=True):
        item.chmod(0o755 if item.is_dir() else 0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def _publish_json_artifact(*, root: Path, filename: str, payload: Mapping[str, object]) -> Path:
    body = _canonical_bytes(payload) + b"\n"
    content_hash = hashlib.sha256(body).hexdigest()
    root = root.absolute()
    if ".." in root.parts or root.is_symlink():
        raise PilotEvaluationError("evaluation output root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise PilotEvaluationError("evaluation output root is unsafe")
    target = root / content_hash
    expected = {filename: body}
    if target.exists():
        entries = tuple(target.rglob("*"))
        actual = {path.relative_to(target).as_posix() for path in entries if path.is_file()}
        if (
            target.is_symlink()
            or any(path.is_symlink() for path in entries)
            or actual != set(expected)
        ):
            raise PilotEvaluationError("existing evaluation artifact inventory changed")
        if (target / filename).read_bytes() != body:
            raise PilotEvaluationError("existing evaluation artifact content changed")
        if os.name != "nt" and any(
            stat.S_IMODE(path.stat().st_mode) & 0o222 for path in (target, *target.rglob("*"))
        ):
            raise PilotEvaluationError("existing evaluation artifact is mutable")
        return target
    temporary = target.parent / f".pending-{content_hash}-{uuid4().hex}"
    temporary.mkdir()
    try:
        (temporary / filename).write_bytes(body)
        _make_immutable(temporary)
        os.replace(temporary, target)
    finally:
        _remove_temporary(temporary)
    return target


def evaluate_pilot_run(
    *, raw_run: Path, evaluation_manifest_path: Path, output_root: Path
) -> PilotEvaluationResult:
    """Evaluate one frozen raw run against isolated labels and publish two artifacts."""

    loaded = _load_raw_run(raw_run)
    evaluation = _load_evaluation_manifest(evaluation_manifest_path)
    manifest = loaded.manifest
    if manifest.dataset_id != evaluation.dataset_id:
        raise PilotEvaluationError("raw-run/evaluation dataset identity mismatch")
    if manifest.prompt_manifest_id != evaluation.prompt_manifest_id:
        raise PilotEvaluationError("raw-run/evaluation prompt-manifest identity mismatch")
    _verify_frozen_evaluation_population(manifest=evaluation, raw_manifest=manifest)
    raw_shots = {(item.asset_id, item.start_frame, item.end_frame) for item in manifest.plans}
    evaluation_shots = {
        (item.video_id, item.start_frame, item.end_frame) for item in evaluation.shots
    }
    if (
        raw_shots != evaluation_shots
        or len(raw_shots) != len(manifest.plans)
        or len(evaluation_shots) != len(evaluation.shots)
    ):
        raise PilotEvaluationError("raw-run/evaluation shot contract mismatch")
    shots_by_key = {(item.video_id, item.shot_index): item for item in evaluation.shots}
    if any(
        (circle.video_id, circle.shot_index) not in shots_by_key
        or not shots_by_key[(circle.video_id, circle.shot_index)].start_frame
        < circle.source_frame_index
        < shots_by_key[(circle.video_id, circle.shot_index)].end_frame
        for circle in evaluation.circles
    ):
        raise PilotEvaluationError("evaluation circle is incompatible with its shot")
    evaluation_circle_keys = {
        (circle.video_id, circle.source_frame_index, circle.annotation_id)
        for circle in evaluation.circles
    }
    if len(evaluation_circle_keys) != len(evaluation.circles):
        raise PilotEvaluationError("evaluation circles must have unique frame identities")

    failed = {(item.asset_id, item.start_frame, item.end_frame) for item in manifest.failed_shots}
    failed_assets_frames = {
        (shot.video_id, circle.source_frame_index)
        for shot in evaluation.shots
        for circle in evaluation.circles
        if (circle.video_id, circle.shot_index) == (shot.video_id, shot.shot_index)
        and (shot.video_id, shot.start_frame, shot.end_frame) in failed
    }
    circles_by_frame: dict[tuple[str, int], list[EvaluationCircle]] = {}
    for circle in evaluation.circles:
        circles_by_frame.setdefault((circle.video_id, circle.source_frame_index), []).append(circle)
    predictions_by_frame: dict[tuple[str, int], list[MaskObservation]] = {}
    for observation in loaded.observations:
        predictions_by_frame.setdefault((observation.asset_id, observation.frame_index), []).append(
            observation
        )
    evaluation_frames = set(circles_by_frame)
    assignments: list[FrameAssignment] = []
    for frame in sorted(evaluation_frames | (set(predictions_by_frame) & evaluation_frames)):
        circles = tuple(circles_by_frame.get(frame, ()))
        predictions = tuple(predictions_by_frame.get(frame, ()))
        if frame in failed_assets_frames:
            assignments.append(
                FrameAssignment(
                    unmatched_ground_truth=tuple(
                        sorted(circles, key=lambda item: item.annotation_id)
                    ),
                    unmatched_predictions=tuple(
                        sorted(predictions, key=lambda item: item.instance_id)
                    ),
                )
            )
        else:
            assignments.append(match_frame(circles, predictions))
    metrics = compute_set_metrics(assignments)
    completed = len(manifest.plans) - len(manifest.failed_shots)
    gates = compute_frozen_gates(
        metrics=metrics, completed_shots=completed, total_shots=len(manifest.plans)
    )
    sample = freeze_review_sample(_review_units(assignments))
    report = EvaluationReport(
        raw_run_hash=raw_run.name,
        evaluation_manifest_hash=evaluation.manifest_id,
        dataset_id=manifest.dataset_id,
        prompt_manifest_id=manifest.prompt_manifest_id,
        input_bundle_id=manifest.input_bundle_id,
        raw_outcome=manifest.outcome,
        coverage=CoverageSummary(
            completed_shots=completed,
            total_shots=len(manifest.plans),
            failed_shots=len(manifest.failed_shots),
        ),
        set_metrics=metrics,
        frozen_gates=gates,
        failed_shot_count=len(manifest.failed_shots),
        explicit_absence_count=len(loaded.absences),
        review_population_counts=sample.population_counts,
    )
    evaluation_path = _publish_json_artifact(
        root=output_root / "evaluations",
        filename="report.json",
        payload=report.model_dump(mode="json"),
    )
    review_payload = {
        "schema_version": "1.0",
        "raw_run_hash": raw_run.name,
        "evaluation_manifest_hash": evaluation.manifest_id,
        "dataset_id": manifest.dataset_id,
        "prompt_manifest_id": manifest.prompt_manifest_id,
        "input_bundle_id": manifest.input_bundle_id,
        "qa_identity": report.qa_identity.model_dump(mode="json"),
        **sample.model_dump(mode="json"),
    }
    review_sample_path = _publish_json_artifact(
        root=output_root / "review-samples",
        filename="review-sample.json",
        payload=review_payload,
    )
    return PilotEvaluationResult(
        report=report,
        evaluation_path=evaluation_path,
        review_sample_path=review_sample_path,
    )
