"""Payload-verified, content-addressed publication for frozen SAM2 raw runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import (
    InstancePrompt,
    PilotQaIdentity,
    PropagationResult,
    PropagationWindow,
)
from billiards_ai.lab.propagation.pilot_data import (
    PilotDataError,
    PromptManifest,
    verify_prompt_manifest,
)
from billiards_ai.lab.propagation.pilot_planning import WindowPlan
from billiards_ai.lab.propagation.stitching import TrackLink


class PilotArtifactError(RuntimeError):
    """A raw-run artifact cannot be safely frozen or reused."""


class PilotPayloadInventoryItem(FrozenContract):
    path: NonEmptyStr
    bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("payload inventory path must be relative and confined")
        return value


class PilotAttemptFailure(FrozenContract):
    """Immutable evidence for a failed worker attempt, including recovered OOMs."""

    window: PropagationWindow
    code: NonEmptyStr
    message: NonEmptyStr
    worker_result_sha256: Sha256 | None = None


class PilotShotFailure(FrozenContract):
    """Terminal per-shot failure without suppressing other shot results."""

    asset_id: NonEmptyStr
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    code: NonEmptyStr
    message: NonEmptyStr

    @model_validator(mode="after")
    def interval_is_non_empty(self) -> PilotShotFailure:
        if self.end_frame <= self.start_frame:
            raise ValueError("failed shot interval must be non-empty")
        return self


class ExecutedGoldPrompt(FrozenContract):
    """Gold prompt evidence bound to the shot that actually executed it."""

    asset_id: NonEmptyStr
    shot_index: int = Field(ge=0)
    shot_start: int = Field(ge=0)
    shot_end: int = Field(gt=0)
    prompt: InstancePrompt

    @model_validator(mode="after")
    def prompt_matches_owner(self) -> ExecutedGoldPrompt:
        if self.shot_end <= self.shot_start:
            raise ValueError("executed gold prompt owner interval must be non-empty")
        if self.prompt.source_kind != "gold" or self.prompt.class_name != "ball":
            raise ValueError("executed gold prompt evidence must be a gold ball prompt")
        if self.prompt.asset_id != self.asset_id or self.prompt.frame_index != self.shot_start:
            raise ValueError("executed gold prompt must match its owning shot asset and start")
        return self


class PilotRunManifest(FrozenContract):
    """All non-payload raw-run evidence needed for immutable local publication."""

    schema_version: Literal["2.0"] = "2.0"
    input_bundle_id: Sha256
    prompt_manifest: PromptManifest
    executed_gold_prompts: tuple[ExecutedGoldPrompt, ...] = Field(min_length=1)
    qa_identity: PilotQaIdentity = Field(
        default_factory=lambda: PilotQaIdentity(artifact_kind="raw_run")
    )
    structural_status: Literal["complete"] = "complete"
    outcome: Literal["all_shots_success", "partial_success", "failed"] = "all_shots_success"
    plans: tuple[WindowPlan, ...] = Field(min_length=1)
    links: tuple[TrackLink, ...] = ()
    attempt_failures: tuple[PilotAttemptFailure, ...] = ()
    failed_shots: tuple[PilotShotFailure, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()

    @property
    def dataset_id(self) -> str:
        return self.prompt_manifest.dataset_id

    @property
    def prompt_manifest_id(self) -> str:
        return self.prompt_manifest.manifest_id

    @model_validator(mode="after")
    def raw_run_uses_frozen_qa_identity(self) -> PilotRunManifest:
        try:
            verify_prompt_manifest(self.prompt_manifest)
        except (PilotDataError, TypeError, ValueError) as exc:
            raise ValueError("pilot run prompt manifest snapshot is invalid") from exc
        if self.qa_identity.artifact_kind != "raw_run":
            raise ValueError("pilot run manifest must use a raw_run QA identity")
        if len({(plan.asset_id, plan.start_frame, plan.end_frame) for plan in self.plans}) != len(
            self.plans
        ):
            raise ValueError("pilot run plans must be unique")
        if any(attempt.status == "planned" for plan in self.plans for attempt in plan.attempts):
            raise ValueError("structurally complete pilot runs cannot contain planned attempts")
        failed_attempts = {
            (
                attempt.window.asset_id,
                attempt.window.ordinal,
                attempt.window.start_frame,
                attempt.window.end_frame,
                attempt.failure_code,
            )
            for plan in self.plans
            for attempt in plan.attempts
            if attempt.status == "failed"
        }
        evidenced_attempts = {
            (
                failure.window.asset_id,
                failure.window.ordinal,
                failure.window.start_frame,
                failure.window.end_frame,
                failure.code,
            )
            for failure in self.attempt_failures
        }
        if len(evidenced_attempts) != len(self.attempt_failures):
            raise ValueError("attempt failure evidence must be unique")
        if failed_attempts != evidenced_attempts:
            raise ValueError("failed attempts require exact attempt failure evidence")
        plan_keys = {(plan.asset_id, plan.start_frame, plan.end_frame) for plan in self.plans}
        failed_keys = {
            (failure.asset_id, failure.start_frame, failure.end_frame)
            for failure in self.failed_shots
        }
        if not failed_keys <= plan_keys or len(failed_keys) != len(self.failed_shots):
            raise ValueError("failed shots must identify unique manifest plans")
        successful_count = len(self.plans) - len(failed_keys)
        expected_outcome = (
            "all_shots_success"
            if not failed_keys
            else "partial_success"
            if successful_count
            else "failed"
        )
        if self.outcome != expected_outcome:
            raise ValueError("pilot run outcome does not match its explicit failed shots")
        for plan in self.plans:
            key = (plan.asset_id, plan.start_frame, plan.end_frame)
            if key not in failed_keys and plan.coverage_proof.missing_frames:
                raise ValueError("successful shots require complete planned-frame coverage")
        return self


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confined_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise PilotArtifactError(f"{label} must be relative and confined")
    return path


def _require_regular_directory(path: Path, *, label: str) -> Path:
    if ".." in path.parts or path.is_symlink() or not path.is_dir():
        raise PilotArtifactError(f"{label} must be a regular directory")
    return path.absolute()


def _safe_payload(root: Path, reference: str, *, label: str) -> Path:
    relative = _confined_path(reference, label=label)
    candidate = root.joinpath(*relative.parts)
    if any(path.is_symlink() for path in (candidate, *candidate.parents)):
        raise PilotArtifactError(f"{label} is a symlink")
    if not candidate.is_file():
        raise PilotArtifactError(f"missing payload: {reference}")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PilotArtifactError(f"{label} escapes its staging root") from exc
    return candidate


def _overlay_path(asset_id: str, frame_index: int) -> str:
    return f"overlays/{asset_id}/{frame_index:08d}.png"


_STRUCTURED_ARTIFACTS = frozenset(
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


def _expected_payload_references(results: Sequence[PropagationResult]) -> tuple[str, ...]:
    references: list[str] = []
    for result in results:
        if not isinstance(result, PropagationResult):
            raise TypeError("pilot publication requires propagation result contracts")
        for observation in result.observations:
            if observation.provenance != "pseudo" or observation.review_status != "unreviewed":
                raise PilotArtifactError("raw observations must remain pseudo/unreviewed")
        for absence in result.absences:
            if absence.provenance != "pseudo" or absence.review_status != "unreviewed":
                raise PilotArtifactError("raw absences must remain pseudo/unreviewed")
        references.extend(
            prompt.mask_ref for prompt in result.prompts if prompt.mask_ref is not None
        )
        references.extend(observation.mask_ref for observation in result.observations)
        references.extend(
            _overlay_path(result.window.asset_id, frame_index)
            for frame_index in {
                item.frame_index for item in (*result.observations, *result.absences)
            }
        )
    if len(references) != len(set(references)):
        raise PilotArtifactError("duplicate payload reference")
    for reference in references:
        path = _confined_path(reference, label="payload reference")
        if path.parts[0] in _STRUCTURED_ARTIFACTS:
            raise PilotArtifactError(
                "payload reference collides with a reserved artifact namespace"
            )
    return tuple(sorted(references))


def _validate_results_match_plan(
    manifest: PilotRunManifest, results: Sequence[PropagationResult]
) -> None:
    plan_attempts = {
        (
            attempt.window.asset_id,
            attempt.window.ordinal,
            attempt.window.start_frame,
            attempt.window.end_frame,
        ): attempt.status
        for plan in manifest.plans
        for attempt in plan.attempts
    }
    result_windows = {
        (
            result.window.asset_id,
            result.window.ordinal,
            result.window.start_frame,
            result.window.end_frame,
        )
        for result in results
    }
    if len(result_windows) != len(results) or not result_windows <= set(plan_attempts):
        raise PilotArtifactError("results must map one-to-one to manifest windows")
    active_windows = {
        window for window, status in plan_attempts.items() if status in {"planned", "committed"}
    }
    if result_windows != active_windows:
        raise PilotArtifactError("results must cover exactly the active manifest windows")


def verify_pilot_run_evidence(
    manifest: PilotRunManifest, results: Sequence[PropagationResult]
) -> tuple[str, ...]:
    """Apply the publisher's exact result/plan and payload-reference policy."""

    if not isinstance(manifest, PilotRunManifest):
        raise TypeError("pilot verification requires a PilotRunManifest")
    try:
        verify_prompt_manifest(manifest.prompt_manifest)
    except (PilotDataError, TypeError, ValueError) as exc:
        raise PilotArtifactError("prompt manifest snapshot identity is invalid") from exc
    prompt_shots = {
        (shot.video_id, shot.start_frame, shot.end_frame) for shot in manifest.prompt_manifest.shots
    }
    planned_shots = {(plan.asset_id, plan.start_frame, plan.end_frame) for plan in manifest.plans}
    if planned_shots != prompt_shots:
        raise PilotArtifactError("run plans must exactly match prompt manifest shots")
    _validate_results_match_plan(manifest, results)
    shot_owners = {
        (shot.video_id, shot.shot_index): (
            shot.video_id,
            shot.shot_index,
            shot.start_frame,
            shot.end_frame,
        )
        for shot in manifest.prompt_manifest.shots
    }
    snapshot_gold = Counter(
        (
            *shot_owners[(circle.video_id, circle.shot_index)],
            circle.video_id,
            circle.source_frame_index,
            circle.annotation_id,
            "ball",
        )
        for circle in manifest.prompt_manifest.circles
    )
    executed_gold: Counter[tuple[object, ...]] = Counter()
    for record in manifest.executed_gold_prompts:
        prompt = record.prompt
        owner = (
            record.asset_id,
            record.shot_index,
            record.shot_start,
            record.shot_end,
        )
        if (
            owner not in shot_owners.values()
            or prompt.source_kind != "gold"
            or prompt.class_name != "ball"
            or prompt.asset_id != record.asset_id
            or prompt.frame_index != record.shot_start
        ):
            raise PilotArtifactError(
                "executed gold prompt must exactly match its snapshot shot owner"
            )
        executed_gold[
            (
                *owner,
                prompt.asset_id,
                prompt.frame_index,
                prompt.source_id,
                prompt.class_name,
            )
        ] += 1
    if executed_gold != snapshot_gold:
        raise PilotArtifactError(
            "executed gold prompts must bind one-to-one to prompt manifest circles"
        )
    plan_owners = {
        (plan.asset_id, plan.start_frame, plan.end_frame): next(
            owner
            for owner in shot_owners.values()
            if (owner[0], owner[2], owner[3]) == (plan.asset_id, plan.start_frame, plan.end_frame)
        )
        for plan in manifest.plans
    }
    owners_by_active_window: dict[tuple[str, int, int, int], list[tuple[str, int, int, int]]] = {}
    for plan in manifest.plans:
        owner = plan_owners[(plan.asset_id, plan.start_frame, plan.end_frame)]
        for attempt in plan.attempts:
            if attempt.status not in {"planned", "committed"}:
                continue
            window = attempt.window
            owners_by_active_window.setdefault(
                (window.asset_id, window.ordinal, window.start_frame, window.end_frame), []
            ).append(owner)
    result_owners: list[tuple[str, int, int, int]] = []
    for result in results:
        window = result.window
        owners = owners_by_active_window.get(
            (window.asset_id, window.ordinal, window.start_frame, window.end_frame), []
        )
        if len(owners) != 1:
            raise PilotArtifactError(
                "result active attempt must resolve to exactly one snapshot shot owner"
            )
        result_owners.append(owners[0])
    owners_with_results = set(result_owners)
    expected_result_gold = Counter(
        (
            *shot_owners[(circle.video_id, circle.shot_index)],
            circle.video_id,
            circle.source_frame_index,
            circle.annotation_id,
            "ball",
        )
        for circle in manifest.prompt_manifest.circles
        if shot_owners[(circle.video_id, circle.shot_index)] in owners_with_results
    )
    result_gold = Counter(
        (
            *owner,
            prompt.asset_id,
            prompt.frame_index,
            prompt.source_id,
            prompt.class_name,
        )
        for result, owner in zip(results, result_owners, strict=True)
        for prompt in result.prompts
        if prompt.source_kind == "gold"
    )
    if result_gold != expected_result_gold:
        raise PilotArtifactError(
            "result gold prompts must bind one-to-one to executed prompt circles"
        )
    return _expected_payload_references(results)


def _discovered_files(root: Path) -> set[str]:
    files: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise PilotArtifactError(f"payload staging contains a symlink: {candidate}")
        if candidate.is_file():
            try:
                files.add(candidate.relative_to(root).as_posix())
            except ValueError as exc:  # pragma: no cover - defensive against filesystem races.
                raise PilotArtifactError("payload staging escapes its root") from exc
    return files


def _inventory(
    staging_root: Path, references: Sequence[str]
) -> tuple[tuple[Path, PilotPayloadInventoryItem], ...]:
    entries: list[tuple[Path, PilotPayloadInventoryItem]] = []
    for reference in references:
        source = _safe_payload(staging_root, reference, label="payload reference")
        entries.append(
            (
                source,
                PilotPayloadInventoryItem(
                    path=reference, bytes=source.stat().st_size, sha256=_sha256(source)
                ),
            )
        )
    actual = _discovered_files(staging_root)
    expected = set(references)
    if actual != expected:
        raise PilotArtifactError("payload set is missing or has extra files")
    return tuple(entries)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _ordered_rows(results: Sequence[PropagationResult], attribute: str) -> list[dict[str, Any]]:
    rows = [
        item.model_dump(mode="json") for result in results for item in getattr(result, attribute)
    ]
    return sorted(rows, key=lambda row: _canonical_bytes(row))


def _structured_payloads(
    *,
    manifest: PilotRunManifest,
    results: Sequence[PropagationResult],
    inventory: Sequence[PilotPayloadInventoryItem],
    content_hash: str,
) -> dict[str, bytes]:
    observations = _ordered_rows(results, "observations")
    absences = _ordered_rows(results, "absences")
    qa_rows = [
        {
            "kind": "observation",
            "asset_id": item["asset_id"],
            "frame_index": item["frame_index"],
            "instance_id": item["instance_id"],
            "qa_flags": item["qa_flags"],
        }
        for item in observations
    ] + [
        {
            "kind": "absence",
            "asset_id": item["asset_id"],
            "frame_index": item["frame_index"],
            "instance_id": item["instance_id"],
            "qa_flags": item["qa_flags"],
        }
        for item in absences
    ]
    plans = sorted((plan.model_dump(mode="json") for plan in manifest.plans), key=_canonical_bytes)
    coverage = sorted(
        (plan.coverage_proof.model_dump(mode="json") for plan in manifest.plans),
        key=_canonical_bytes,
    )
    stored_manifest = {"content_hash": content_hash, **manifest.model_dump(mode="json")}
    failure_rows = (
        [
            {"kind": "worker_attempt", **item.model_dump(mode="json")}
            for item in manifest.attempt_failures
        ]
        + [{"kind": "shot", **item.model_dump(mode="json")} for item in manifest.failed_shots]
        + [{"kind": "propagation", **row} for row in _ordered_rows(results, "failures")]
    )
    return {
        "manifest.json": _canonical_bytes(stored_manifest) + b"\n",
        "inventory.json": _canonical_bytes(
            {"schema_version": "1.0", "files": [item.model_dump(mode="json") for item in inventory]}
        )
        + b"\n",
        "prompts.jsonl": _jsonl(_ordered_rows(results, "prompts")),
        "results.jsonl": _jsonl([result.model_dump(mode="json") for result in results]),
        "observations.jsonl": _jsonl(observations),
        "absences.jsonl": _jsonl(absences),
        "links.jsonl": _jsonl(
            sorted((item.model_dump(mode="json") for item in manifest.links), key=_canonical_bytes)
        ),
        "qa.jsonl": _jsonl(sorted(qa_rows, key=_canonical_bytes)),
        "failures.jsonl": _jsonl(sorted(failure_rows, key=_canonical_bytes)),
        "telemetry.jsonl": _jsonl(sorted(manifest.telemetry, key=_canonical_bytes)),
        "plan.json": _canonical_bytes({"plans": plans}) + b"\n",
        "coverage.jsonl": _jsonl(coverage),
    }


def _make_immutable(root: Path) -> None:
    if root.is_symlink():
        raise PilotArtifactError("temporary publication directory is a symlink")
    entries = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise PilotArtifactError("temporary publication contains a symlink")
    for path in sorted(entries, reverse=True):
        if path.is_symlink():
            raise PilotArtifactError("temporary publication contains a symlink")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _remove_own_temporary(path: Path) -> None:
    if not path.exists():
        return
    for entry in sorted(path.rglob("*"), reverse=True):
        if entry.is_symlink():
            entry.unlink()
            continue
        entry.chmod(0o755 if entry.is_dir() else 0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def _verify_existing(
    path: Path,
    *,
    structured: Mapping[str, bytes],
    inventory: Sequence[PilotPayloadInventoryItem],
) -> None:
    _verify_tree(
        path,
        structured=structured,
        inventory=inventory,
        label="published raw run",
        require_immutable=True,
    )


def _verify_tree(
    path: Path,
    *,
    structured: Mapping[str, bytes],
    inventory: Sequence[PilotPayloadInventoryItem],
    label: str,
    require_immutable: bool,
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PilotArtifactError(f"{label} is unsafe")
    expected = {*structured, *(item.path for item in inventory)}
    actual = _discovered_files(path)
    if actual != expected:
        raise PilotArtifactError(f"{label} payload set is invalid")
    for relative, expected_bytes in structured.items():
        candidate = _safe_payload(path, relative, label=f"{label} structured evidence")
        if candidate.read_bytes() != expected_bytes:
            raise PilotArtifactError(f"{label} structured evidence changed")
    for item in inventory:
        candidate = _safe_payload(path, item.path, label=f"{label} payload")
        if candidate.stat().st_size != item.bytes or _sha256(candidate) != item.sha256:
            raise PilotArtifactError(f"{label} payload changed")
    if require_immutable and os.name != "nt":
        entries = (path, *path.rglob("*"))
        if any(stat.S_IMODE(entry.stat().st_mode) & 0o222 for entry in entries):
            raise PilotArtifactError(f"{label} is writable")


def publish_pilot_run(
    *,
    staging_root: Path,
    manifest: PilotRunManifest,
    results: Sequence[PropagationResult],
    output_root: Path,
) -> Path:
    """Atomically freeze one raw SAM2 run after validating every referenced payload."""

    staging_root = _require_regular_directory(staging_root, label="payload staging root")
    output_root = output_root.absolute()
    if ".." in output_root.parts or output_root.is_symlink():
        raise PilotArtifactError("output root must be a confined regular directory")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise PilotArtifactError("output root must be a regular directory")
    if not isinstance(manifest, PilotRunManifest):
        raise TypeError("pilot publication requires a PilotRunManifest")
    if manifest.qa_identity.artifact_kind != "raw_run":
        raise PilotArtifactError("pilot run manifest must identify a raw run")

    ordered_results = tuple(
        sorted(results, key=lambda result: _canonical_bytes(result.model_dump(mode="json")))
    )
    references = verify_pilot_run_evidence(manifest, ordered_results)
    entries = _inventory(staging_root, references)
    inventory = tuple(item for _, item in entries)
    identity = {
        "schema_version": "1.0",
        "manifest": manifest.model_dump(mode="json"),
        "results": [result.model_dump(mode="json") for result in ordered_results],
        "inventory": [item.model_dump(mode="json") for item in inventory],
    }
    content_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    structured = _structured_payloads(
        manifest=manifest,
        results=ordered_results,
        inventory=inventory,
        content_hash=content_hash,
    )
    target = output_root / content_hash
    if target.exists():
        _verify_existing(target, structured=structured, inventory=inventory)
        return target

    temporary = output_root / f".pending-{content_hash}-{uuid4().hex}"
    temporary.mkdir()
    try:
        for relative, payload in structured.items():
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        for source, item in entries:
            destination = temporary / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if (
                source.stat().st_size != item.bytes
                or _sha256(source) != item.sha256
                or destination.stat().st_size != item.bytes
                or _sha256(destination) != item.sha256
            ):
                raise PilotArtifactError(f"payload changed while publishing: {item.path}")
        _make_immutable(temporary)
        _verify_tree(
            temporary,
            structured=structured,
            inventory=inventory,
            label="pending publication",
            require_immutable=True,
        )
        try:
            os.replace(temporary, target)
        except OSError:
            if target.exists():
                _verify_existing(target, structured=structured, inventory=inventory)
                return target
            raise
    finally:
        _remove_own_temporary(temporary)
    return target
