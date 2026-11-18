"""Loopback-only review session, verdict persistence, and immutable export."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from pydantic import Field, StrictBool, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import PilotQaIdentity
from billiards_ai.lab.propagation.evaluation import (
    EvaluationReport,
    ReviewSample,
    ReviewUnit,
    _load_raw_run,
)
from billiards_ai.lab.propagation.overlays import (
    EvaluationRenderInventoryItem,
    EvaluationRenderManifest,
    _canonical_bytes,
    _has_symlink_ancestor,
    _load_sample,
    _validate_json_artifact,
    _verify_render,
)
from billiards_ai.lab.propagation.review_page import (
    REVIEW_CSS,
    ReviewPageModel,
    render_complete_page,
    render_review_page,
)

try:  # POSIX is used by the pilot; Windows fallback retains in-process serialization.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows.
    fcntl = None

_CSP = (
    "default-src 'none'; img-src 'self'; style-src 'self'; "
    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREDICTED_STRATA = frozenset({"clean_match", "flagged_or_ambiguous_match", "false_positive"})


class ReviewServerError(RuntimeError):
    """Review state, evidence, request, or export failed closed."""


class ReviewConflictError(ReviewServerError):
    """A stale browser attempted to overwrite a newer review revision."""


class ReviewVerdict(FrozenContract):
    unit_id: Sha256
    stratum: NonEmptyStr
    unit_type: NonEmptyStr
    acceptable_without_redraw: StrictBool
    severe_drift_or_wrong_target: StrictBool
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def human_labels_are_not_contradictory(self) -> ReviewVerdict:
        if self.acceptable_without_redraw and self.severe_drift_or_wrong_target:
            raise ValueError("acceptable and severe-drift labels cannot both be true")
        return self


class ReviewState(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    raw_run_hash: Sha256
    evaluation_artifact_hash: Sha256
    review_sample_hash: Sha256
    render_artifact_hash: Sha256
    revision: int = Field(ge=0)
    total_count: int = Field(ge=0)
    verdicts: tuple[ReviewVerdict, ...] = ()

    @model_validator(mode="after")
    def revision_tracks_unique_verdicts(self) -> ReviewState:
        if len({item.unit_id for item in self.verdicts}) != len(self.verdicts):
            raise ValueError("verdict unit identities must be unique")
        if self.revision != len(self.verdicts) or len(self.verdicts) > self.total_count:
            raise ValueError("review revision must equal completed verdict count")
        return self

    @property
    def completed_count(self) -> int:
        return len(self.verdicts)


class VerdictRates(FrozenContract):
    acceptable_without_redraw: float = Field(ge=0, le=1)
    severe_drift_or_wrong_target: float = Field(ge=0, le=1)
    acceptable_denominator: int = Field(gt=0)
    severe_denominator: int = Field(gt=0)


class HumanGateResult(FrozenContract):
    value: float | None
    threshold: float
    comparison: Literal[">=", "<="]
    passed: bool | None


class VerdictSummary(FrozenContract):
    status: Literal["pending", "human_gates_passed", "human_gates_failed"]
    reviewed_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    population_counts: dict[str, int]
    sample_counts: dict[str, int]
    raw_challenge_sample: VerdictRates | None = None
    population_weighted: VerdictRates | None = None
    acceptable_without_redraw_gate: HumanGateResult
    severe_drift_or_wrong_target_gate: HumanGateResult


def compute_verdict_summary(
    sample: ReviewSample, verdicts: tuple[ReviewVerdict, ...]
) -> VerdictSummary:
    """Compute complete-only raw and frozen-population-weighted human gate rates."""
    pending_gate_acceptable = HumanGateResult(
        value=None, threshold=0.60, comparison=">=", passed=None
    )
    pending_gate_severe = HumanGateResult(value=None, threshold=0.05, comparison="<=", passed=None)
    if len(verdicts) != len(sample.units):
        return VerdictSummary(
            status="pending",
            reviewed_count=len(verdicts),
            total_count=len(sample.units),
            population_counts=sample.population_counts,
            sample_counts=sample.sample_counts,
            acceptable_without_redraw_gate=pending_gate_acceptable,
            severe_drift_or_wrong_target_gate=pending_gate_severe,
        )
    by_stratum: dict[str, list[ReviewVerdict]] = {
        stratum: [] for stratum in sample.population_counts
    }
    for verdict in verdicts:
        try:
            by_stratum[verdict.stratum].append(verdict)
        except KeyError as exc:
            raise ReviewServerError("verdict references an unknown review stratum") from exc
    predicted = tuple(item for item in verdicts if item.stratum in _PREDICTED_STRATA)
    if not verdicts or not predicted:
        raise ReviewServerError("complete review requires both overall and predicted-mask units")
    raw = VerdictRates(
        acceptable_without_redraw=sum(item.acceptable_without_redraw for item in verdicts)
        / len(verdicts),
        severe_drift_or_wrong_target=sum(item.severe_drift_or_wrong_target for item in predicted)
        / len(predicted),
        acceptable_denominator=len(verdicts),
        severe_denominator=len(predicted),
    )
    acceptable_population = sum(sample.population_counts.values())
    severe_population = sum(
        count for stratum, count in sample.population_counts.items() if stratum in _PREDICTED_STRATA
    )
    if acceptable_population <= 0 or severe_population <= 0:
        raise ReviewServerError("frozen population denominators must be positive")

    def weighted(field: str, strata: set[str] | frozenset[str]) -> float:
        numerator = 0.0
        denominator = 0
        for stratum, population in sample.population_counts.items():
            if stratum not in strata or population == 0:
                continue
            observed = by_stratum[stratum]
            if not observed or len(observed) != sample.sample_counts[stratum]:
                raise ReviewServerError("complete verdicts do not match frozen stratum counts")
            numerator += population * (
                sum(bool(getattr(item, field)) for item in observed) / len(observed)
            )
            denominator += population
        if denominator <= 0:
            raise ReviewServerError("weighted verdict denominator is empty")
        return numerator / denominator

    all_strata = set(sample.population_counts)
    weighted_rates = VerdictRates(
        acceptable_without_redraw=weighted("acceptable_without_redraw", all_strata),
        severe_drift_or_wrong_target=weighted("severe_drift_or_wrong_target", _PREDICTED_STRATA),
        acceptable_denominator=acceptable_population,
        severe_denominator=severe_population,
    )
    acceptable_passed = weighted_rates.acceptable_without_redraw >= 0.60
    severe_passed = weighted_rates.severe_drift_or_wrong_target <= 0.05
    return VerdictSummary(
        status=(
            "human_gates_passed" if acceptable_passed and severe_passed else "human_gates_failed"
        ),
        reviewed_count=len(verdicts),
        total_count=len(sample.units),
        population_counts=sample.population_counts,
        sample_counts=sample.sample_counts,
        raw_challenge_sample=raw,
        population_weighted=weighted_rates,
        acceptable_without_redraw_gate=HumanGateResult(
            value=weighted_rates.acceptable_without_redraw,
            threshold=0.60,
            comparison=">=",
            passed=acceptable_passed,
        ),
        severe_drift_or_wrong_target_gate=HumanGateResult(
            value=weighted_rates.severe_drift_or_wrong_target,
            threshold=0.05,
            comparison="<=",
            passed=severe_passed,
        ),
    )


def _state_bytes(state: ReviewState) -> bytes:
    return _canonical_bytes(state.model_dump(mode="json")) + b"\n"


def _artifact_is_immutable(root: Path) -> bool:
    return os.name == "nt" or not any(
        stat.S_IMODE(path.stat().st_mode) & 0o222 for path in (root, *root.rglob("*"))
    )


def _verify_verdict_artifact(target: Path, body: bytes) -> None:
    entries = tuple(target.rglob("*"))
    actual = {path.relative_to(target).as_posix() for path in entries if path.is_file()}
    if (
        target.is_symlink()
        or any(path.is_symlink() for path in entries)
        or actual != {"verdict.json"}
        or (target / "verdict.json").read_bytes() != body
        or hashlib.sha256(body).hexdigest() != target.name
        or not _artifact_is_immutable(target)
    ):
        raise ReviewServerError("verdict artifact inventory or content changed")


def _read_verified_inventory_bytes(*, root: Path, item: EvaluationRenderInventoryItem) -> bytes:
    relative = PurePosixPath(item.path)
    candidate = root.joinpath(*relative.parts)
    if _has_symlink_ancestor(candidate):
        raise ReviewServerError("render media path has a symlink ancestor")
    try:
        before = os.stat(candidate, follow_symlinks=False)
    except OSError as exc:
        raise ReviewServerError("render media is missing") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ReviewServerError("render media must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    elif not getattr(before, "st_ino", 0):  # pragma: no cover - platform fallback.
        raise ReviewServerError("portable no-follow media verification is unavailable")
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ReviewServerError("render media could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ReviewServerError("render media changed while it was opened")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReviewServerError("render media changed while it was read")
    body = b"".join(blocks)
    if len(body) != item.bytes or hashlib.sha256(body).hexdigest() != item.sha256:
        raise ReviewServerError("render media does not match its exact inventory record")
    return body


class ReviewSession:
    """One resumable reviewer over one immutable sample and render artifact."""

    def __init__(
        self,
        *,
        raw_run: Path,
        evaluation_artifact: Path,
        review_sample: Path,
        render_artifact: Path,
        state_path: Path,
        report: EvaluationReport,
        sample: ReviewSample,
        render_manifest: EvaluationRenderManifest,
        sample_qa_identity: PilotQaIdentity,
        media_items: dict[str, EvaluationRenderInventoryItem],
        media_bytes: dict[str, bytes],
        interaction_source_bytes: dict[Path, bytes],
        state: ReviewState,
    ) -> None:
        self.raw_run = raw_run
        self.evaluation_artifact = evaluation_artifact
        self.review_sample = review_sample
        self.render_artifact = render_artifact
        self.state_path = state_path
        self._report = report
        self._sample = sample
        self._render_manifest = render_manifest
        self._sample_qa_identity = sample_qa_identity
        self._media_items = media_items
        self._media_bytes = media_bytes
        self._interaction_source_bytes = interaction_source_bytes
        self._state = state
        self._thread_lock = threading.Lock()

    @classmethod
    def open(
        cls,
        *,
        raw_run: Path,
        evaluation_artifact: Path,
        review_sample: Path,
        render_artifact: Path,
        state_path: Path,
    ) -> ReviewSession:
        raw_run = raw_run.absolute()
        evaluation_artifact = evaluation_artifact.absolute()
        review_sample = review_sample.absolute()
        render_artifact = render_artifact.absolute()
        state_path = state_path.absolute()
        if _has_symlink_ancestor(state_path) or any(
            _has_symlink_ancestor(path)
            for path in (raw_run, evaluation_artifact, review_sample, render_artifact)
        ):
            raise ReviewServerError("review source or state path is unsafe")
        loaded_raw = _load_raw_run(raw_run)
        report_payload = _validate_json_artifact(
            evaluation_artifact, filename="report.json", label="evaluation artifact"
        )
        sample_payload = _validate_json_artifact(
            review_sample, filename="review-sample.json", label="review sample"
        )
        try:
            report = EvaluationReport.model_validate(report_payload)
            sample = _load_sample(sample_payload)
            sample_qa_identity = PilotQaIdentity.model_validate(sample_payload["qa_identity"])
            render_payload = json.loads(
                (render_artifact / "render-manifest.json").read_text(encoding="utf-8")
            )
            render_manifest = EvaluationRenderManifest.model_validate(render_payload)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ReviewServerError("review source contract is invalid") from exc
        if not _SHA256.fullmatch(render_artifact.name):
            raise ReviewServerError("render artifact is not content-addressed")
        _verify_render(render_artifact, render_manifest, render_artifact.name)
        raw_manifest = loaded_raw.manifest
        bindings = (
            render_manifest.raw_run_hash == raw_run.name == report.raw_run_hash,
            render_manifest.evaluation_artifact_hash == evaluation_artifact.name,
            render_manifest.review_sample_hash == review_sample.name,
            render_manifest.evaluation_manifest_hash == report.evaluation_manifest_hash,
            report.dataset_id == raw_manifest.dataset_id == render_manifest.dataset_id,
            report.prompt_manifest_id
            == raw_manifest.prompt_manifest_id
            == render_manifest.prompt_manifest_id,
            report.input_bundle_id
            == raw_manifest.input_bundle_id
            == render_manifest.input_bundle_id,
            report.qa_identity == sample_qa_identity == render_manifest.qa_identity,
            report.qa_identity.pilot_qa_config_hash
            == raw_manifest.qa_identity.pilot_qa_config_hash,
            sample_payload.get("raw_run_hash") == raw_run.name,
            sample_payload.get("evaluation_manifest_hash") == report.evaluation_manifest_hash,
            sample_payload.get("dataset_id") == raw_manifest.dataset_id,
            sample_payload.get("prompt_manifest_id") == raw_manifest.prompt_manifest_id,
            sample_payload.get("input_bundle_id") == raw_manifest.input_bundle_id,
        )
        if not all(bindings):
            raise ReviewServerError("raw/evaluation/sample/render hashes do not match")
        expected_jpegs = {item.unit_id: item.jpeg_path for item in render_manifest.unit_evidence}
        if set(expected_jpegs) != {item.unit_id for item in sample.units}:
            raise ReviewServerError("render inventory does not exactly cover the frozen sample")
        for unit_id, relative in expected_jpegs.items():
            if relative != f"review-units/{unit_id}.jpg":
                raise ReviewServerError("render JPEG path is not unit-derived")
        inventory_by_path = {item.path: item for item in render_manifest.inventory}
        if len(inventory_by_path) != len(render_manifest.inventory):
            raise ReviewServerError("render inventory contains duplicate paths")
        media_items = {
            unit_id: inventory_by_path[relative] for unit_id, relative in expected_jpegs.items()
        }
        if len(media_items) != len(expected_jpegs):
            raise ReviewServerError("render inventory does not exactly bind unit JPEGs")
        media_bytes = {
            unit_id: _read_verified_inventory_bytes(root=render_artifact, item=item)
            for unit_id, item in media_items.items()
        }
        interaction_source_bytes = {
            path: path.read_bytes()
            for path in (
                raw_run / "manifest.json",
                raw_run / "inventory.json",
                evaluation_artifact / "report.json",
                review_sample / "review-sample.json",
                render_artifact / "render-manifest.json",
            )
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            state_path.parent.is_symlink()
            or not state_path.parent.is_dir()
            or state_path.is_symlink()
        ):
            raise ReviewServerError("review state path is unsafe")
        initial = ReviewState(
            raw_run_hash=raw_run.name,
            evaluation_artifact_hash=evaluation_artifact.name,
            review_sample_hash=review_sample.name,
            render_artifact_hash=render_artifact.name,
            revision=0,
            total_count=len(sample.units),
        )
        state = cls._read_state(state_path) if state_path.exists() else initial
        cls._validate_state(state, initial=initial, sample=sample)
        return cls(
            raw_run=raw_run,
            evaluation_artifact=evaluation_artifact,
            review_sample=review_sample,
            render_artifact=render_artifact,
            state_path=state_path,
            report=report,
            sample=sample,
            render_manifest=render_manifest,
            sample_qa_identity=sample_qa_identity,
            media_items=media_items,
            media_bytes=media_bytes,
            interaction_source_bytes=interaction_source_bytes,
            state=state,
        )

    @staticmethod
    def _read_state(path: Path) -> ReviewState:
        if path.is_symlink() or not path.is_file():
            raise ReviewServerError("review state must be a regular file")
        try:
            return ReviewState.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ReviewServerError("review state is invalid") from exc

    @staticmethod
    def _validate_state(state: ReviewState, *, initial: ReviewState, sample: ReviewSample) -> None:
        for field in (
            "raw_run_hash",
            "evaluation_artifact_hash",
            "review_sample_hash",
            "render_artifact_hash",
            "total_count",
        ):
            if getattr(state, field) != getattr(initial, field):
                raise ReviewServerError("review state source binding changed")
        unit_by_id = {item.unit_id: item for item in sample.units}
        expected_prefix = [item.unit_id for item in sample.units[: len(state.verdicts)]]
        if [item.unit_id for item in state.verdicts] != expected_prefix:
            raise ReviewServerError("review state is not a frozen sample prefix")
        for verdict in state.verdicts:
            unit = unit_by_id[verdict.unit_id]
            if verdict.stratum != unit.stratum or verdict.unit_type != unit.unit_type:
                raise ReviewServerError("review verdict identity fields changed")
            if unit.unit_type != "matched_pair" and verdict.acceptable_without_redraw:
                raise ReviewServerError("unmatched units cannot be accepted without redrawing")

    @contextmanager
    def _locked_state(self) -> Iterator[None]:
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        if lock_path.is_symlink():
            raise ReviewServerError("review state lock path is unsafe")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _atomic_write(self, state: ReviewState) -> None:
        pending = self.state_path.parent / f".pending-{self.state_path.name}-{uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(pending, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_state_bytes(state))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, self.state_path)
        finally:
            if pending.exists():
                pending.unlink()

    @property
    def revision(self) -> int:
        return self._state.revision

    @property
    def current_unit(self) -> ReviewUnit | None:
        return (
            self._sample.units[self._state.completed_count]
            if self._state.completed_count < len(self._sample.units)
            else None
        )

    @property
    def remaining_units(self) -> tuple[ReviewUnit, ...]:
        return self._sample.units[self._state.completed_count :]

    @property
    def summary(self) -> VerdictSummary:
        return compute_verdict_summary(self._sample, self._state.verdicts)

    def submit(
        self,
        *,
        unit_id: str,
        revision: int,
        acceptable_without_redraw: bool,
        severe_drift_or_wrong_target: bool,
        note: str | None = None,
    ) -> ReviewState:
        if (
            type(acceptable_without_redraw) is not bool
            or type(severe_drift_or_wrong_target) is not bool
        ):
            raise ReviewServerError("both verdict values must be booleans")
        if acceptable_without_redraw and severe_drift_or_wrong_target:
            raise ReviewServerError("两个选项不能同时选择“是”。")
        with self._thread_lock, self._locked_state():
            self._validate_interaction_sources(unit_id=unit_id)
            disk = self._read_state(self.state_path) if self.state_path.exists() else self._state
            initial = self._state.model_copy(update={"revision": 0, "verdicts": ()})
            self._validate_state(disk, initial=initial, sample=self._sample)
            if disk.revision != revision:
                self._state = disk
                raise ReviewConflictError("review revision is stale")
            if disk.completed_count >= len(self._sample.units):
                raise ReviewConflictError("review sample is already complete")
            unit = self._sample.units[disk.completed_count]
            if unit.unit_id != unit_id:
                raise ReviewConflictError("submitted unit is not the current frozen review unit")
            if unit.unit_type != "matched_pair" and acceptable_without_redraw:
                raise ReviewServerError("unmatched units cannot be accepted without redrawing")
            verdict = ReviewVerdict(
                unit_id=unit.unit_id,
                stratum=unit.stratum,
                unit_type=unit.unit_type,
                acceptable_without_redraw=acceptable_without_redraw,
                severe_drift_or_wrong_target=severe_drift_or_wrong_target,
                note=note,
            )
            updated = disk.model_copy(
                update={"revision": disk.revision + 1, "verdicts": (*disk.verdicts, verdict)}
            )
            ReviewState.model_validate(updated.model_dump())
            self._atomic_write(updated)
            try:
                self._validate_interaction_sources(unit_id=unit_id)
            except ReviewServerError as integrity_error:
                try:
                    self._atomic_write(disk)
                except (OSError, ReviewServerError) as rollback_error:
                    raise ReviewServerError(
                        "review source changed and atomic state rollback failed"
                    ) from rollback_error
                self._state = disk
                raise integrity_error
            self._state = updated
            return updated

    def page(
        self,
        *,
        acceptable_without_redraw: bool | None = None,
        severe_drift_or_wrong_target: bool | None = None,
        note: str = "",
        validation_error: str | None = None,
    ) -> str:
        unit = self.current_unit
        self._validate_interaction_sources(unit_id=unit.unit_id if unit is not None else None)
        if unit is None:
            return render_complete_page(
                total_count=len(self._sample.units), revision=self._state.revision
            )
        evidence: list[str] = [f"类型：{unit.unit_type}", f"分层：{unit.stratum}"]
        if unit.ground_truth_id:
            evidence.append(f"标注：{unit.ground_truth_id}")
        if unit.prediction_id:
            evidence.append(f"预测：{unit.prediction_id}")
        if unit.normalized_center_error is not None:
            evidence.append(f"归一化中心误差：{unit.normalized_center_error:.3f}")
        if unit.qa_flags:
            evidence.append("QA：" + ", ".join(unit.qa_flags))
        return render_review_page(
            ReviewPageModel(
                unit_id=unit.unit_id,
                unit_type=unit.unit_type,
                stratum=unit.stratum,
                asset_id=unit.asset_id,
                frame_index=unit.frame_index,
                media_url=f"/media/{unit.unit_id}.jpg",
                evidence_lines=tuple(evidence),
                completed_count=self._state.completed_count,
                total_count=len(self._sample.units),
                revision=self._state.revision,
                acceptable_without_redraw=acceptable_without_redraw,
                severe_drift_or_wrong_target=severe_drift_or_wrong_target,
                note=note,
                validation_error=validation_error,
            )
        )

    def current_media_bytes(self, unit_id: str) -> bytes:
        unit = self.current_unit
        if unit is None or unit.unit_id != unit_id or not _SHA256.fullmatch(unit_id):
            raise ReviewServerError("only the current frozen unit media is available")
        self._validate_interaction_sources(unit_id=unit_id)
        return self._media_bytes[unit_id]

    def _validate_interaction_sources(self, *, unit_id: str | None) -> None:
        try:
            for path, expected in self._interaction_source_bytes.items():
                if _has_symlink_ancestor(path) or not path.is_file():
                    raise ReviewServerError("review identity source is unsafe or missing")
                if path.read_bytes() != expected:
                    raise ReviewServerError("review identity source changed")
            if unit_id is not None:
                item = self._media_items.get(unit_id)
                expected_media = self._media_bytes.get(unit_id)
                if item is None or expected_media is None:
                    raise ReviewServerError("review media is not bound to the frozen sample")
                current = _read_verified_inventory_bytes(root=self.render_artifact, item=item)
                if current != expected_media:
                    raise ReviewServerError("render media changed after session validation")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ReviewServerError("review source or media evidence changed") from exc

    def _validate_sources(self) -> None:
        try:
            loaded_raw = _load_raw_run(self.raw_run)
            report_payload = _validate_json_artifact(
                self.evaluation_artifact, filename="report.json", label="evaluation artifact"
            )
            sample_payload = _validate_json_artifact(
                self.review_sample, filename="review-sample.json", label="review sample"
            )
            _verify_render(self.render_artifact, self._render_manifest, self.render_artifact.name)
            for unit_id, item in self._media_items.items():
                current = _read_verified_inventory_bytes(root=self.render_artifact, item=item)
                if current != self._media_bytes[unit_id]:
                    raise ReviewServerError("render media changed after session validation")
            raw_manifest = loaded_raw.manifest
            if (
                report_payload.get("raw_run_hash") != self.raw_run.name
                or self._report.dataset_id != raw_manifest.dataset_id
                or self._report.prompt_manifest_id != raw_manifest.prompt_manifest_id
                or self._report.input_bundle_id != raw_manifest.input_bundle_id
                or self._report.qa_identity != self._sample_qa_identity
                or self._render_manifest.qa_identity != self._report.qa_identity
                or sample_payload.get("qa_identity")
                != self._sample_qa_identity.model_dump(mode="json")
            ):
                raise ReviewServerError("review source provenance changed")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ReviewServerError("review source evidence changed") from exc

    def export_verdict_artifact(self, output_root: Path) -> Path:
        summary = self.summary
        if summary.status == "pending":
            raise ReviewServerError("review must be complete before verdict export")
        self._validate_sources()
        payload = {
            "schema_version": "1.0",
            "raw_run_hash": self.raw_run.name,
            "evaluation_artifact_hash": self.evaluation_artifact.name,
            "review_sample_hash": self.review_sample.name,
            "render_artifact_hash": self.render_artifact.name,
            "evaluation_manifest_hash": self._report.evaluation_manifest_hash,
            "dataset_id": self._report.dataset_id,
            "prompt_manifest_id": self._report.prompt_manifest_id,
            "input_bundle_id": self._report.input_bundle_id,
            "qa_identity": self._report.qa_identity.model_dump(mode="json"),
            "revision": self._state.revision,
            "verdicts": [item.model_dump(mode="json") for item in self._state.verdicts],
            "summary": summary.model_dump(mode="json"),
            "pilot_status": (
                "passed"
                if self._report.frozen_gates.automated_gates_passed
                and summary.status == "human_gates_passed"
                else "failed"
            ),
        }
        body = _canonical_bytes(payload) + b"\n"
        content_hash = hashlib.sha256(body).hexdigest()
        output_root = output_root.absolute()
        root = output_root / "verdicts"
        if _has_symlink_ancestor(output_root) or _has_symlink_ancestor(root):
            raise ReviewServerError("verdict output root is unsafe")
        root.mkdir(parents=True, exist_ok=True)
        target = root / content_hash
        if target.exists():
            _verify_verdict_artifact(target, body)
            return target
        pending = root / f".pending-{content_hash}-{uuid4().hex}"
        pending.mkdir()
        try:
            (pending / "verdict.json").write_bytes(body)
            (pending / "verdict.json").chmod(0o444)
            pending.chmod(0o555)
            self._validate_sources()
            if target.exists():
                raise ReviewServerError("verdict target appeared during publication")
            os.replace(pending, target)
            try:
                self._validate_sources()
                _verify_verdict_artifact(target, body)
            except ReviewServerError:
                target.chmod(0o755)
                for path in target.rglob("*"):
                    if not path.is_symlink():
                        path.chmod(0o755 if path.is_dir() else 0o644)
                shutil.rmtree(target)
                raise
        except (OSError, ValueError) as exc:
            raise ReviewServerError("verdict artifact publication failed safely") from exc
        finally:
            if pending.exists() and not pending.is_symlink():
                pending.chmod(0o755)
                for path in pending.rglob("*"):
                    if not path.is_symlink():
                        path.chmod(0o644)
                shutil.rmtree(pending)
        return target


class _ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], session: ReviewSession):
        self.review_session = session
        super().__init__(address, _ReviewRequestHandler)


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    server: _ReviewHTTPServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _allowed_host(self) -> bool:
        host = self.headers.get("Host", "")
        return host == "127.0.0.1" or bool(re.fullmatch(r"127\.0\.0\.1:\d+", host))

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send(
        self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8"
    ) -> None:
        self._headers(status, content_type, len(body))
        self.wfile.write(body)

    def _reject_host(self) -> bool:
        if self._allowed_host():
            return False
        self._send(HTTPStatus.BAD_REQUEST, b"invalid Host")
        return True

    def _send_form_error(
        self,
        *,
        status: HTTPStatus,
        message: str,
        acceptable_without_redraw: bool | None,
        severe_drift_or_wrong_target: bool | None,
        note: str,
    ) -> None:
        try:
            page = self.server.review_session.page(
                acceptable_without_redraw=acceptable_without_redraw,
                severe_drift_or_wrong_target=severe_drift_or_wrong_target,
                note=note,
                validation_error=message,
            ).encode("utf-8")
        except ReviewServerError:
            self._send(
                HTTPStatus.CONFLICT,
                "证据完整性校验失败，未记录任何复核结果。".encode(),
            )
            return
        self._send(status, page, "text/html; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self._reject_host():
            return
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            self._send(HTTPStatus.NOT_FOUND, b"not found")
            return
        if parsed.path == "/":
            body = self.server.review_session.page().encode("utf-8")
            self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
            return
        if parsed.path == "/assets/review.css":
            self._send(HTTPStatus.OK, REVIEW_CSS.encode(), "text/css; charset=utf-8")
            return
        match = re.fullmatch(r"/media/([0-9a-f]{64})\.jpg", parsed.path)
        if match:
            try:
                body = self.server.review_session.current_media_bytes(match.group(1))
            except (OSError, ReviewServerError):
                self._send(HTTPStatus.NOT_FOUND, b"not found")
                return
            self._send(HTTPStatus.OK, body, "image/jpeg")
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found")

    def do_POST(self) -> None:  # noqa: N802
        if self._reject_host():
            return
        parsed = urlsplit(self.path)
        if parsed.path != "/verdict" or parsed.query or parsed.fragment:
            self._send(HTTPStatus.NOT_FOUND, b"not found")
            return
        if self.headers.get("Content-Type") != "application/x-www-form-urlencoded":
            self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, b"form encoding required")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send(HTTPStatus.BAD_REQUEST, b"invalid content length")
            return
        if length < 1 or length > 4096:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"invalid form size")
            return
        acceptable: bool | None = None
        severe: bool | None = None
        note = ""
        try:
            fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            allowed = {
                "unit_id",
                "revision",
                "acceptable_without_redraw",
                "severe_drift_or_wrong_target",
                "note",
            }
            if set(fields) != allowed or any(len(value) != 1 for value in fields.values()):
                raise ValueError("表单字段不完整或重复，请重新确认。")
            note = fields["note"][0]
            booleans = {
                name: fields[name][0]
                for name in ("acceptable_without_redraw", "severe_drift_or_wrong_target")
            }
            if any(value not in {"true", "false"} for value in booleans.values()):
                raise ValueError("两个判断都必须明确选择“是”或“否”。")
            acceptable = booleans["acceptable_without_redraw"] == "true"
            severe = booleans["severe_drift_or_wrong_target"] == "true"
            if acceptable and severe:
                raise ValueError("两个选项不能同时选择“是”。")
            current = self.server.review_session.current_unit
            if current is not None and current.unit_type != "matched_pair" and acceptable:
                raise ValueError("漏检或误检不能标记为可直接使用。")
            self.server.review_session.submit(
                unit_id=fields["unit_id"][0],
                revision=int(fields["revision"][0]),
                acceptable_without_redraw=acceptable,
                severe_drift_or_wrong_target=severe,
                note=note,
            )
        except ReviewConflictError:
            self._send_form_error(
                status=HTTPStatus.CONFLICT,
                message="页面修订号已更新，请根据当前项目重新确认后提交。",
                acceptable_without_redraw=acceptable,
                severe_drift_or_wrong_target=severe,
                note=note,
            )
            return
        except (UnicodeError, ValueError) as exc:
            localized_messages = {
                "表单字段不完整或重复，请重新确认。",
                "两个判断都必须明确选择“是”或“否”。",
                "两个选项不能同时选择“是”。",
                "漏检或误检不能标记为可直接使用。",
            }
            message = str(exc)
            self._send_form_error(
                status=HTTPStatus.BAD_REQUEST,
                message=(
                    message if message in localized_messages else "表单内容无效，请重新确认。"
                ),
                acceptable_without_redraw=acceptable,
                severe_drift_or_wrong_target=severe,
                note=note,
            )
            return
        except ReviewServerError:
            self._send_form_error(
                status=HTTPStatus.CONFLICT,
                message="证据或复核状态已变化，未记录本次提交。",
                acceptable_without_redraw=acceptable,
                severe_drift_or_wrong_target=severe,
                note=note,
            )
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()


def create_review_server(
    *, session: ReviewSession, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Create a server that cannot bind to a non-loopback interface."""

    if host != "127.0.0.1":
        raise ReviewServerError("review server must bind exactly to 127.0.0.1")
    if not 0 <= port <= 65535:
        raise ReviewServerError("review server port is invalid")
    return _ReviewHTTPServer((host, port), session)
