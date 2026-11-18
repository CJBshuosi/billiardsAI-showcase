"""Post-freeze evaluation renderings built from immutable raw overlays."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

import cv2
import numpy as np
from pydantic import Field, field_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import PilotQaIdentity
from billiards_ai.lab.propagation.evaluation import (
    EvaluationReport,
    ReviewSample,
    ReviewUnit,
    _load_raw_run,
)
from billiards_ai.lab.propagation.pilot_artifacts import PilotPayloadInventoryItem

_SAFE_ASSET = re.compile(r"^[A-Za-z0-9_.-]+$")
_FRAME_SIZE = (640, 360)
_REVIEW_STRATA = {
    "clean_match",
    "flagged_or_ambiguous_match",
    "miss",
    "false_positive",
}


class EvaluationOverlayError(RuntimeError):
    """Frozen evaluation render inputs or outputs failed integrity checks."""


class EvaluationRenderInventoryItem(FrozenContract):
    path: NonEmptyStr
    bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("render inventory paths must be relative and confined")
        return value


class UnitRenderEvidence(FrozenContract):
    unit_id: Sha256
    kind: Literal["match", "miss", "false_positive"]
    stratum: NonEmptyStr
    jpeg_path: NonEmptyStr
    annotations: tuple[NonEmptyStr, ...]


class EvaluationRenderManifest(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    raw_run_hash: Sha256
    evaluation_artifact_hash: Sha256
    review_sample_hash: Sha256
    evaluation_manifest_hash: Sha256
    dataset_id: Sha256
    prompt_manifest_id: Sha256
    input_bundle_id: Sha256
    qa_identity: PilotQaIdentity
    fps: float = Field(gt=0, le=60)
    inventory: tuple[EvaluationRenderInventoryItem, ...]
    unit_evidence: tuple[UnitRenderEvidence, ...]


class EvaluationOverlayResult(FrozenContract):
    path: Path
    content_hash: Sha256
    manifest: EvaluationRenderManifest


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationOverlayError(f"{label} contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationOverlayError(f"{label} must contain a JSON object")
    return value


def _has_symlink_ancestor(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _validate_json_artifact(root: Path, *, filename: str, label: str) -> dict[str, Any]:
    root = root.absolute()
    if (
        _has_symlink_ancestor(root)
        or not root.is_dir()
        or not re.fullmatch(r"[0-9a-f]{64}", root.name)
    ):
        raise EvaluationOverlayError(f"{label} must be a content-addressed directory")
    entries = tuple(root.rglob("*"))
    files = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    if any(path.is_symlink() for path in (root, *entries)) or files != {filename}:
        raise EvaluationOverlayError(f"{label} inventory is invalid")
    artifact = root / filename
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != root.name:
        raise EvaluationOverlayError(f"{label} content hash is invalid")
    if os.name != "nt" and any(
        stat.S_IMODE(path.stat().st_mode) & 0o222 for path in (root, *entries)
    ):
        raise EvaluationOverlayError(f"{label} is mutable")
    return _json_object(artifact, label=label)


def _load_sample(payload: dict[str, Any]) -> ReviewSample:
    try:
        sample = ReviewSample.model_validate(
            {
                key: payload[key]
                for key in (
                    "population_counts",
                    "primary_quota_sample_counts",
                    "sample_counts",
                    "units",
                )
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationOverlayError("review sample contract is invalid") from exc
    if any(
        set(counts) != _REVIEW_STRATA
        for counts in (
            sample.population_counts,
            sample.primary_quota_sample_counts,
            sample.sample_counts,
        )
    ):
        raise EvaluationOverlayError("review sample must preserve all four frozen strata")
    actual = {name: 0 for name in sample.sample_counts}
    for unit in sample.units:
        if unit.stratum not in actual:
            raise EvaluationOverlayError("review sample contains an undeclared stratum")
        actual[unit.stratum] += 1
    if actual != sample.sample_counts or any(
        sample.sample_counts[name] > sample.population_counts.get(name, -1) for name in actual
    ):
        raise EvaluationOverlayError("review sample counts do not match its frozen units")
    if len({unit.unit_id for unit in sample.units}) != len(sample.units):
        raise EvaluationOverlayError("review sample unit identities are duplicated")
    return sample


def _raw_overlay_path(raw_run: Path, unit: ReviewUnit, inventory: set[str]) -> Path | None:
    if not _SAFE_ASSET.fullmatch(unit.asset_id) or unit.asset_id in {".", ".."}:
        raise EvaluationOverlayError("review unit asset identifier is unsafe")
    relative = f"overlays/{unit.asset_id}/{unit.frame_index:08d}.png"
    if relative not in inventory:
        return None
    candidate = raw_run.joinpath(*PurePosixPath(relative).parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise EvaluationOverlayError("inventoried raw overlay is missing or unsafe")
    return candidate


def _render_unit(base_path: Path | None, unit: ReviewUnit) -> tuple[np.ndarray, tuple[str, ...]]:
    if base_path is None:
        image = np.full((_FRAME_SIZE[1], _FRAME_SIZE[0], 3), 24, dtype=np.uint8)
        source_width, source_height = _FRAME_SIZE
        unavailable = True
    else:
        source = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        if source is None:
            raise EvaluationOverlayError("raw overlay cannot be decoded")
        source_height, source_width = source.shape[:2]
        image = cv2.resize(source, _FRAME_SIZE, interpolation=cv2.INTER_AREA)
        unavailable = False
    scale_x = _FRAME_SIZE[0] / source_width
    scale_y = _FRAME_SIZE[1] / source_height
    annotations: list[str] = []
    if unit.ground_truth is not None:
        center = (
            round(unit.ground_truth.center_x_px * scale_x),
            round(unit.ground_truth.center_y_px * scale_y),
        )
        radius = max(2, round(unit.ground_truth.radius_px * (scale_x + scale_y) / 2))
        cv2.circle(image, center, radius, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        annotations.append("held_out_ground_truth")
    if unit.prediction is not None:
        center = (
            round(unit.prediction.center_px[0] * scale_x),
            round(unit.prediction.center_px[1] * scale_y),
        )
        cv2.drawMarker(image, center, (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
        annotations.append("prediction")
    kind: Literal["match", "miss", "false_positive"]
    if unit.unit_type == "matched_pair":
        kind, color = "match", (0, 200, 0)
        annotations.append("match")
    elif unit.unit_type == "unmatched_ground_truth":
        kind, color = "miss", (0, 0, 255)
        annotations.append("miss")
    else:
        kind, color = "false_positive", (0, 140, 255)
        annotations.append("false_positive")
    if unit.qa_flags:
        annotations.append("qa:" + ",".join(unit.qa_flags))
    annotations.extend(("gate:acceptable_without_redraw", "gate:severe_drift_or_wrong_target"))
    if unavailable:
        annotations.append("raw_overlay_unavailable")
    cv2.rectangle(image, (0, 0), (_FRAME_SIZE[0] - 1, 46), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"{kind.upper()} | {unit.asset_id}:{unit.frame_index}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "HUMAN GATES: acceptable? severe drift/wrong target?",
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return image, tuple(annotations)


def _inventory_item(root: Path, relative: str) -> EvaluationRenderInventoryItem:
    path = root.joinpath(*PurePosixPath(relative).parts)
    return EvaluationRenderInventoryItem(
        path=relative, bytes=path.stat().st_size, sha256=_sha256(path)
    )


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise EvaluationOverlayError("pending render contains a symlink")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _remove_pending(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    shutil.rmtree(root)


def _verify_render(root: Path, manifest: EvaluationRenderManifest, content_hash: str) -> None:
    entries = tuple(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in entries):
        raise EvaluationOverlayError("render artifact contains a symlink")
    expected = {"render-manifest.json", *(item.path for item in manifest.inventory)}
    actual = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    if actual != expected:
        raise EvaluationOverlayError("render artifact inventory changed")
    manifest_bytes = _canonical_bytes(manifest.model_dump(mode="json")) + b"\n"
    if hashlib.sha256(manifest_bytes).hexdigest() != content_hash:
        raise EvaluationOverlayError("render manifest identity changed")
    if (root / "render-manifest.json").read_bytes() != manifest_bytes:
        raise EvaluationOverlayError("render manifest content changed")
    for item in manifest.inventory:
        path = root.joinpath(*PurePosixPath(item.path).parts)
        if path.stat().st_size != item.bytes or _sha256(path) != item.sha256:
            raise EvaluationOverlayError("render payload changed")
    if os.name != "nt" and any(
        stat.S_IMODE(path.stat().st_mode) & 0o222 for path in (root, *entries)
    ):
        raise EvaluationOverlayError("render artifact is mutable")


def render_evaluation_overlays(
    *,
    raw_run: Path,
    evaluation_artifact: Path,
    review_sample: Path,
    output_root: Path,
    fps: float = 2.0,
) -> EvaluationOverlayResult:
    """Freeze review MP4/JPEG evidence without modifying the inference-only raw run."""

    if not 0 < fps <= 60:
        raise EvaluationOverlayError("render FPS must be in (0, 60]")
    raw_run = raw_run.absolute()
    if _has_symlink_ancestor(raw_run):
        raise EvaluationOverlayError("raw run path is unsafe")
    loaded_raw = _load_raw_run(raw_run)
    report_payload = _validate_json_artifact(
        evaluation_artifact, filename="report.json", label="evaluation artifact"
    )
    sample_payload = _validate_json_artifact(
        review_sample, filename="review-sample.json", label="review sample"
    )
    try:
        report = EvaluationReport.model_validate(report_payload)
        sample_qa_identity = PilotQaIdentity.model_validate(sample_payload["qa_identity"])
    except (TypeError, ValueError) as exc:
        raise EvaluationOverlayError("evaluation artifact contract is invalid") from exc
    except KeyError as exc:
        raise EvaluationOverlayError("review sample QA identity is missing") from exc
    sample = _load_sample(sample_payload)
    if sample.population_counts != report.review_population_counts:
        raise EvaluationOverlayError(
            "review sample population does not match the frozen evaluation report"
        )
    raw_manifest = loaded_raw.manifest
    if (
        report.raw_run_hash != raw_run.name
        or report.dataset_id != raw_manifest.dataset_id
        or report.prompt_manifest_id != raw_manifest.prompt_manifest_id
        or report.input_bundle_id != raw_manifest.input_bundle_id
        or report.qa_identity.pilot_qa_config_hash != raw_manifest.qa_identity.pilot_qa_config_hash
    ):
        raise EvaluationOverlayError("evaluation report has foreign raw-run provenance")
    if (
        sample_qa_identity.artifact_kind != "evaluation_report"
        or sample_qa_identity != report.qa_identity
    ):
        raise EvaluationOverlayError("review sample QA identity is invalid")
    binding = {
        "raw_run_hash": raw_run.name,
        "evaluation_manifest_hash": report.evaluation_manifest_hash,
        "dataset_id": report.dataset_id,
        "prompt_manifest_id": report.prompt_manifest_id,
        "input_bundle_id": report.input_bundle_id,
    }
    if any(report_payload.get(key) != value for key, value in binding.items()) or any(
        sample_payload.get(key) != value for key, value in binding.items()
    ):
        raise EvaluationOverlayError("raw/evaluation/review sample binding mismatch")

    inventory_payload = _json_object(raw_run / "inventory.json", label="raw-run inventory")
    try:
        raw_inventory = tuple(
            PilotPayloadInventoryItem.model_validate(item) for item in inventory_payload["files"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationOverlayError("raw-run inventory contract is invalid") from exc
    raw_paths = {item.path for item in raw_inventory}

    output_root = output_root.absolute()
    render_root = output_root / "evaluation-renders"
    if _has_symlink_ancestor(output_root) or _has_symlink_ancestor(render_root):
        raise EvaluationOverlayError("render output root is unsafe")
    render_root.mkdir(parents=True, exist_ok=True)
    if render_root.is_symlink() or not render_root.is_dir():
        raise EvaluationOverlayError("render output root is unsafe")
    pending = render_root / f".pending-{uuid4().hex}"
    pending.mkdir()
    try:
        unit_directory = pending / "review-units"
        unit_directory.mkdir()
        evidence: list[UnitRenderEvidence] = []
        frames: list[np.ndarray] = []
        for unit in sample.units:
            base = _raw_overlay_path(raw_run, unit, raw_paths)
            image, annotations = _render_unit(base, unit)
            relative = f"review-units/{unit.unit_id}.jpg"
            destination = pending.joinpath(*PurePosixPath(relative).parts)
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise EvaluationOverlayError("review JPEG could not be written")
            kind = {
                "matched_pair": "match",
                "unmatched_ground_truth": "miss",
                "unmatched_prediction": "false_positive",
            }[unit.unit_type]
            evidence.append(
                UnitRenderEvidence(
                    unit_id=unit.unit_id,
                    kind=kind,
                    stratum=unit.stratum,
                    jpeg_path=relative,
                    annotations=annotations,
                )
            )
            frames.append(image)
        video_path = pending / "review.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, _FRAME_SIZE)
        if not writer.isOpened():
            raise EvaluationOverlayError("review MP4 writer could not be opened")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise EvaluationOverlayError("review MP4 was not produced")
        relative_payloads = ["review.mp4", *(item.jpeg_path for item in evidence)]
        inventory = tuple(
            _inventory_item(pending, relative) for relative in sorted(relative_payloads)
        )
        manifest = EvaluationRenderManifest(
            raw_run_hash=raw_run.name,
            evaluation_artifact_hash=evaluation_artifact.name,
            review_sample_hash=review_sample.name,
            evaluation_manifest_hash=report.evaluation_manifest_hash,
            dataset_id=report.dataset_id,
            prompt_manifest_id=report.prompt_manifest_id,
            input_bundle_id=report.input_bundle_id,
            qa_identity=report.qa_identity,
            fps=fps,
            inventory=inventory,
            unit_evidence=tuple(evidence),
        )
        manifest_bytes = _canonical_bytes(manifest.model_dump(mode="json")) + b"\n"
        content_hash = hashlib.sha256(manifest_bytes).hexdigest()
        (pending / "render-manifest.json").write_bytes(manifest_bytes)
        _make_immutable(pending)

        _load_raw_run(raw_run)
        _validate_json_artifact(
            evaluation_artifact, filename="report.json", label="evaluation artifact"
        )
        _validate_json_artifact(review_sample, filename="review-sample.json", label="review sample")
        target = render_root / content_hash
        if target.exists():
            _verify_render(target, manifest, content_hash)
            return EvaluationOverlayResult(
                path=target, content_hash=content_hash, manifest=manifest
            )
        if target.exists():  # Defensive recheck immediately before the atomic rename.
            raise EvaluationOverlayError("render target appeared during publication")
        os.replace(pending, target)
        try:
            _load_raw_run(raw_run)
            _validate_json_artifact(
                evaluation_artifact, filename="report.json", label="evaluation artifact"
            )
            _validate_json_artifact(
                review_sample, filename="review-sample.json", label="review sample"
            )
            _verify_render(target, manifest, content_hash)
        except (EvaluationOverlayError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _remove_pending(target)
            raise EvaluationOverlayError("rendering source changed during publication") from exc
        return EvaluationOverlayResult(path=target, content_hash=content_hash, manifest=manifest)
    except EvaluationOverlayError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationOverlayError("evaluation rendering failed safely") from exc
    finally:
        _remove_pending(pending)
