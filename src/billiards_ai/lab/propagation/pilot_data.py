"""Fail-closed split of reviewed pilot frames into prompt and evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256

_ROLES: tuple[Literal["start", "middle", "end"], ...] = ("start", "middle", "end")
_FRAME_NAME = re.compile(r"-frame-(\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)
FROZEN_VIDEO_IDS = (
    "video1",
    "video2",
    "video11",
    "video17",
    "video27",
    "video41",
    "video102",
    "video107",
    "video141",
    "video150",
)
FROZEN_SHOT_COUNT = 25
FROZEN_PROMPT_FRAME_COUNT = 25
FROZEN_EVALUATION_FRAME_COUNT = 50
FROZEN_REVIEW_FRAME_COUNT = 75
FROZEN_CIRCLE_COUNT = 768


class PilotDataError(RuntimeError):
    """The reviewed pilot inputs cannot safely form the blind split."""


class ShotDefinition(FrozenContract):
    video_id: NonEmptyStr
    shot_index: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def interval_is_non_empty(self) -> ShotDefinition:
        if self.end_frame <= self.start_frame:
            raise ValueError("shot interval must be non-empty")
        return self


class PromptCircle(FrozenContract):
    annotation_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    source_frame_sha256: Sha256
    center_x_px: float = Field(ge=0)
    center_y_px: float = Field(ge=0)
    radius_px: float = Field(gt=0)


class EvaluationCircle(PromptCircle):
    role: Literal["middle", "end"]


class PromptManifest(FrozenContract):
    schema_version: Literal["2.0"] = "2.0"
    manifest_id: Sha256
    dataset_id: Sha256
    shots: tuple[ShotDefinition, ...]
    circles: tuple[PromptCircle, ...]


class EvaluationManifest(FrozenContract):
    schema_version: Literal["2.0"] = "2.0"
    manifest_id: Sha256
    dataset_id: Sha256
    prompt_manifest_id: Sha256
    shots: tuple[ShotDefinition, ...]
    circles: tuple[EvaluationCircle, ...]


class PilotManifestBundle(FrozenContract):
    prompt_manifest: PromptManifest
    evaluation_manifest: EvaluationManifest
    prompt_manifest_path: Path
    evaluation_manifest_path: Path


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotDataError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PilotDataError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PilotDataError(f"missing review file: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotDataError(f"invalid review JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise PilotDataError(f"review JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _confined(root: Path, relative: str, *, label: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or relative in {"", "."}:
        raise PilotDataError(f"unsafe {label}: {relative}")
    path = root.joinpath(*candidate.parts)
    if path.is_symlink() or not path.is_file():
        raise PilotDataError(f"missing {label}: {relative}")
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise PilotDataError(f"unsafe {label}: {relative}")
    return resolved


def _frame_index(filename: str) -> int:
    match = _FRAME_NAME.search(filename)
    if match is None:
        raise PilotDataError(f"review evidence filename lacks a source frame index: {filename}")
    return int(match.group(1))


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotDataError(f"{label} must be a non-negative integer")
    return value


def _review_records(
    review_pack: Path,
) -> tuple[tuple[ShotDefinition, ...], dict[tuple[str, int], tuple[str, str]]]:
    videos_root = review_pack / "videos"
    if not videos_root.is_dir():
        raise PilotDataError("review pack lacks videos directory")
    shots: list[ShotDefinition] = []
    evidence: dict[tuple[str, int], tuple[str, str]] = {}
    seen_roles: set[tuple[str, int, str]] = set()
    seen_shots: set[tuple[str, int]] = set()
    for review_file in sorted(videos_root.glob("*/shot-review.jsonl")):
        video_id = review_file.parent.name
        for row in _read_jsonl(review_file):
            shot_index = _integer(row.get("shot_index"), label="shot_index")
            candidate = row.get("candidate")
            if not isinstance(candidate, dict):
                raise PilotDataError("review row lacks candidate")
            start_frame = _integer(candidate.get("start_frame"), label="candidate start_frame")
            end_frame = _integer(candidate.get("end_frame"), label="candidate end_frame")
            frames = row.get("evidence_frames")
            if not isinstance(frames, dict) or set(frames) != set(_ROLES):
                raise PilotDataError(f"missing required review roles for {video_id}/{shot_index}")
            role_indices: dict[str, int] = {}
            for role in _ROLES:
                filename = frames[role]
                if not isinstance(filename, str):
                    raise PilotDataError("review evidence filename must be a string")
                key = (video_id, shot_index, role)
                if key in seen_roles:
                    raise PilotDataError(
                        f"duplicate video/shot/role: {video_id}/{shot_index}/{role}"
                    )
                seen_roles.add(key)
                frame_index = _frame_index(filename)
                role_indices[role] = frame_index
                if role == "start" and frame_index != start_frame:
                    raise PilotDataError(
                        f"review start frame disagrees with shot candidate: {video_id}/{shot_index}"
                    )
                relative = f"videos/{video_id}/review-frames/{filename}"
                source = _confined(review_pack, relative, label="review evidence")
                join_key = (video_id, frame_index)
                if join_key in evidence:
                    raise PilotDataError(f"duplicate review source frame: {video_id}/{frame_index}")
                evidence[join_key] = (role, _sha256(source))
            if not (
                role_indices["start"] < role_indices["middle"] < role_indices["end"] < end_frame
            ):
                raise PilotDataError(
                    f"review role frames must satisfy start < middle < end < candidate end: "
                    f"{video_id}/{shot_index}"
                )
            if (video_id, shot_index) in seen_shots:
                raise PilotDataError(f"duplicate video/shot: {video_id}/{shot_index}")
            seen_shots.add((video_id, shot_index))
            try:
                shots.append(
                    ShotDefinition(
                        video_id=video_id,
                        shot_index=shot_index,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                )
            except ValueError as exc:
                raise PilotDataError(f"invalid shot definition: {video_id}/{shot_index}") from exc
    if not shots:
        raise PilotDataError("review pack has no shots")
    return tuple(sorted(shots, key=lambda item: (item.video_id, item.shot_index))), evidence


def _circle(
    annotation: Mapping[str, object], image: Mapping[str, object]
) -> tuple[str, float, float, float]:
    annotation_id = str(annotation.get("annotation_id") or annotation.get("id") or "")
    bbox = annotation.get("bbox")
    area = annotation.get("area")
    if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(area, (int, float)):
        raise PilotDataError("COCO annotation requires bbox and area")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in bbox):
        raise PilotDataError("COCO bbox must be finite")
    left, top, width, height = (float(value) for value in bbox)
    numeric_area = float(area)
    if (
        not annotation_id
        or width <= 0
        or height <= 0
        or numeric_area <= 0
        or not math.isfinite(numeric_area)
    ):
        raise PilotDataError("COCO circle geometry must be non-empty")
    radius = math.sqrt(numeric_area / math.pi)
    tolerance = max(1e-6, radius * 1e-6)
    if abs(width / 2 - radius) > tolerance or abs(height / 2 - radius) > tolerance:
        raise PilotDataError("COCO circle bbox is not circular")
    return annotation_id, left + width / 2, top + height / 2, radius


def _circle_review_to_source(
    transform: object, center_x: float, center_y: float, radius: float
) -> tuple[float, float, float]:
    """Map a review-pixel circle into display-oriented source pixels.

    The circle's review bounding box is mapped through the same FrameTransform,
    and the source radius is the geometric
    mean of the mapped box half-extents, so evaluation ground truth lands in the
    exact pixel space of the SAM2 mask centroids it is scored against.
    """

    box = transform.map_review_box(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    )
    source_x = (box[0] + box[2]) / 2
    source_y = (box[1] + box[3]) / 2
    source_radius = (((box[2] - box[0]) * (box[3] - box[1])) ** 0.5) / 2
    return source_x, source_y, source_radius


def _manifest_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def verify_prompt_manifest(manifest: PromptManifest) -> None:
    """Verify one self-contained prompt-only provenance snapshot."""

    if not isinstance(manifest, PromptManifest):
        raise TypeError("prompt provenance requires a PromptManifest")
    payload = manifest.model_dump(mode="json", exclude={"manifest_id"})
    if _manifest_id(payload) != manifest.manifest_id:
        raise PilotDataError("prompt manifest content identity mismatch")
    if not manifest.shots:
        raise PilotDataError("prompt manifest requires shots")
    shot_keys = {(shot.video_id, shot.shot_index) for shot in manifest.shots}
    if len(shot_keys) != len(manifest.shots):
        raise PilotDataError("prompt manifest shot identities must be unique")
    prompt_frames = {(shot.video_id, shot.start_frame) for shot in manifest.shots}
    if len(prompt_frames) != len(manifest.shots):
        raise PilotDataError("prompt manifest shots require unique start frames")
    circle_keys = {
        (circle.video_id, circle.shot_index, circle.annotation_id) for circle in manifest.circles
    }
    if len(circle_keys) != len(manifest.circles):
        raise PilotDataError("prompt manifest circle identities must be unique")
    circles_by_shot: dict[tuple[str, int], list[PromptCircle]] = {}
    for circle in manifest.circles:
        key = (circle.video_id, circle.shot_index)
        if key not in shot_keys:
            raise PilotDataError("prompt circle references an unknown shot")
        circles_by_shot.setdefault(key, []).append(circle)
    if set(circles_by_shot) != shot_keys:
        raise PilotDataError("every prompt shot requires start-label circles")
    shots_by_key = {(shot.video_id, shot.shot_index): shot for shot in manifest.shots}
    for key, circles in circles_by_shot.items():
        shot = shots_by_key[key]
        if any(circle.source_frame_index != shot.start_frame for circle in circles):
            raise PilotDataError("prompt circle source frame must equal its shot start")
        if len({circle.source_frame_sha256 for circle in circles}) != 1:
            raise PilotDataError("prompt circles on one frame require one source identity")


def dataset_id_for_videos(dataset_root: Path, video_ids: Sequence[str]) -> str:
    """Hash the exact, sorted approved video payload descriptors."""

    descriptors: list[dict[str, object]] = []
    for video_id in sorted(video_ids):
        relative = f"videos/{video_id}.mp4"
        video = _confined(dataset_root, relative, label="pilot video")
        descriptors.append(
            {
                "video_id": video_id,
                "path": relative,
                "bytes": video.stat().st_size,
                "sha256": _sha256(video),
            }
        )
    return hashlib.sha256(_canonical_bytes(descriptors)).hexdigest()


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_bytes(payload) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_pilot_manifests(
    *, annotation_export: Path, review_pack: Path, dataset_root: Path, output_root: Path
) -> PilotManifestBundle:
    """Join reviewed COCO circles to review evidence without exposing held-out circles."""

    annotation_export = annotation_export.resolve()
    review_pack = review_pack.resolve()
    dataset_root = dataset_root.resolve()
    shots, evidence = _review_records(review_pack)
    expected_videos = {shot.video_id for shot in shots}
    review_videos = {
        path.parent.name for path in (review_pack / "videos").glob("*/shot-review.jsonl")
    }
    if review_videos != expected_videos:
        raise PilotDataError("unexpected review video")
    dataset_id = dataset_id_for_videos(dataset_root, tuple(expected_videos))
    # Local import: frame_mapping imports ShotDefinition from this module, so a
    # module-level import here would be circular.  Both modules are fully loaded
    # by the time build_pilot_manifests runs.
    from billiards_ai.lab.propagation.frame_mapping import (
        FrameExtractionError,
        FrameMappingError,
        FrameTransform,
        probe_source_geometry,
    )

    # Held-out evaluation circles are annotated in review pixels but SAM2
    # predictions live in display-oriented source pixels (the prompt path maps
    # its circles through the same frame transform). Probe each
    # source video's geometry so evaluation ground truth can be mapped into the
    # prediction pixel space; otherwise the blind evaluator compares two spaces.
    source_geometry: dict[str, object] = {}
    for video_id in sorted(expected_videos):
        try:
            source_geometry[video_id] = probe_source_geometry(
                _confined(dataset_root, f"videos/{video_id}.mp4", label="pilot video")
            )
        except FrameExtractionError as exc:
            raise PilotDataError(f"source video geometry is unreadable: {video_id}") from exc
    payload = _read_object(
        _confined(annotation_export, "annotations.json", label="annotation export")
    )
    summary = _read_object(_confined(annotation_export, "summary.json", label="annotation summary"))
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise PilotDataError("COCO export requires images and annotations")
    if _integer(summary.get("annotation_count"), label="annotation summary count") != len(
        annotations
    ):
        raise PilotDataError("annotation summary count mismatch")
    images_by_id: dict[int, dict[str, object]] = {}
    image_hashes: dict[tuple[str, int], str] = {}
    review_sizes: dict[tuple[str, int], tuple[int, int]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise PilotDataError("COCO image must be an object")
        image_id = _integer(image.get("id"), label="COCO image id")
        if image_id in images_by_id:
            raise PilotDataError("duplicate COCO image id")
        video_id = image.get("source_video")
        frame_index = image.get("source_frame_index")
        filename = image.get("file_name")
        if not isinstance(video_id, str) or not isinstance(filename, str):
            raise PilotDataError("COCO image lacks source video or file name")
        index = _integer(frame_index, label="COCO source frame index")
        key = (video_id, index)
        if key not in evidence:
            raise PilotDataError(f"unexpected COCO source frame: {video_id}/{index}")
        if key in image_hashes:
            raise PilotDataError(f"duplicate COCO source frame: {video_id}/{index}")
        exported = _confined(annotation_export / "images", filename, label="exported review image")
        image_hash = _sha256(exported)
        if image_hash != evidence[key][1]:
            raise PilotDataError(f"review/export source-frame hash mismatch: {video_id}/{index}")
        review_width = image.get("width")
        review_height = image.get("height")
        if (
            isinstance(review_width, bool)
            or isinstance(review_height, bool)
            or not isinstance(review_width, int)
            or not isinstance(review_height, int)
            or review_width <= 0
            or review_height <= 0
        ):
            raise PilotDataError(f"COCO image lacks review dimensions: {video_id}/{index}")
        images_by_id[image_id] = image
        image_hashes[key] = image_hash
        review_sizes[key] = (review_width, review_height)
    if set(image_hashes) != set(evidence):
        raise PilotDataError("review/COCO frame count mismatch")
    circles_by_frame: dict[tuple[str, int], list[tuple[str, float, float, float]]] = {
        key: [] for key in image_hashes
    }
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise PilotDataError("COCO annotation must be an object")
        image_id = _integer(annotation.get("image_id"), label="COCO annotation image id")
        image = images_by_id.get(image_id)
        if image is None:
            raise PilotDataError("COCO annotation references unknown image")
        video_id = image["source_video"]
        frame_index = image["source_frame_index"]
        assert isinstance(video_id, str) and isinstance(frame_index, int)
        circles_by_frame[(video_id, frame_index)].append(_circle(annotation, image))
    if any(not circles for circles in circles_by_frame.values()):
        raise PilotDataError("complete review frame has no circles")
    shot_by_frame: dict[tuple[str, int], ShotDefinition] = {}
    # Re-read stable review rows to associate non-start roles with their shot index.
    role_shots = _role_shot_index(review_pack)
    for key, shot_index in role_shots.items():
        shot = next(
            (item for item in shots if (item.video_id, item.shot_index) == (key[0], shot_index)),
            None,
        )
        if shot is None:
            raise PilotDataError("review role references unknown shot")
        shot_by_frame[(key[0], key[1])] = shot
    prompts: list[PromptCircle] = []
    evaluations: list[EvaluationCircle] = []
    for (video_id, frame_index), circles in sorted(circles_by_frame.items()):
        shot = shot_by_frame.get((video_id, frame_index))
        role_hash = evidence.get((video_id, frame_index))
        if shot is None or role_hash is None:
            raise PilotDataError("review/COCO join is incomplete")
        role, source_hash = role_hash
        for annotation_id, center_x, center_y, radius in sorted(circles):
            identity = dict(
                annotation_id=annotation_id,
                video_id=video_id,
                shot_index=shot.shot_index,
                source_frame_index=frame_index,
                source_frame_sha256=source_hash,
            )
            if role == "start":
                # Prompt circles stay in review pixels: the execution boundary
                # maps them into source pixels, and the frozen
                # raw-run snapshot pins that review-space prompt manifest.
                prompts.append(
                    PromptCircle(
                        center_x_px=center_x,
                        center_y_px=center_y,
                        radius_px=radius,
                        **identity,
                    )
                )
            else:
                transform = FrameTransform(
                    review_width=review_sizes[(video_id, frame_index)][0],
                    review_height=review_sizes[(video_id, frame_index)][1],
                    source_width=source_geometry[video_id].source_width,
                    source_height=source_geometry[video_id].source_height,
                    rotation_degrees=source_geometry[video_id].rotation_degrees,
                )
                try:
                    source_x, source_y, source_radius = _circle_review_to_source(
                        transform, center_x, center_y, radius
                    )
                except FrameMappingError as exc:
                    raise PilotDataError(
                        f"evaluation circle escapes the review frame: {video_id}/{frame_index}"
                    ) from exc
                evaluations.append(
                    EvaluationCircle(
                        role=role,
                        center_x_px=source_x,
                        center_y_px=source_y,
                        radius_px=source_radius,
                        **identity,
                    )
                )
    prompt_payload: dict[str, object] = {
        "schema_version": "2.0",
        "dataset_id": dataset_id,
        "shots": [item.model_dump(mode="json") for item in shots],
        "circles": [item.model_dump(mode="json") for item in prompts],
    }
    prompt_manifest = PromptManifest(manifest_id=_manifest_id(prompt_payload), **prompt_payload)
    evaluation_payload: dict[str, object] = {
        "schema_version": "2.0",
        "dataset_id": dataset_id,
        "prompt_manifest_id": prompt_manifest.manifest_id,
        "shots": [item.model_dump(mode="json") for item in shots],
        "circles": [item.model_dump(mode="json") for item in evaluations],
    }
    evaluation_manifest = EvaluationManifest(
        manifest_id=_manifest_id(evaluation_payload), **evaluation_payload
    )
    prompt_path = (
        output_root / "prompt-manifests" / prompt_manifest.manifest_id / "prompt-manifest.json"
    )
    evaluation_path = (
        output_root
        / "evaluation-manifests"
        / evaluation_manifest.manifest_id
        / "evaluation-manifest.json"
    )
    _write_manifest(prompt_path, prompt_manifest.model_dump(mode="json"))
    _write_manifest(evaluation_path, evaluation_manifest.model_dump(mode="json"))
    return PilotManifestBundle(
        prompt_manifest=prompt_manifest,
        evaluation_manifest=evaluation_manifest,
        prompt_manifest_path=prompt_path,
        evaluation_manifest_path=evaluation_path,
    )


def _role_shot_index(review_pack: Path) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for review_file in sorted((review_pack / "videos").glob("*/shot-review.jsonl")):
        video_id = review_file.parent.name
        for row in _read_jsonl(review_file):
            shot_index = _integer(row.get("shot_index"), label="shot_index")
            frames = row.get("evidence_frames")
            assert isinstance(frames, dict)
            for role in _ROLES:
                filename = frames.get(role)
                assert isinstance(filename, str)
                key = (video_id, _frame_index(filename))
                if key in result:
                    raise PilotDataError(f"duplicate review source frame: {video_id}/{key[1]}")
                result[key] = shot_index
    return result


def preflight_frozen_pilot_counts(bundle: PilotManifestBundle) -> None:
    """Explicitly assert the approved production population; never call on import."""

    prompt_frames = {
        (circle.video_id, circle.source_frame_index) for circle in bundle.prompt_manifest.circles
    }
    evaluation_frames = {
        (circle.video_id, circle.source_frame_index)
        for circle in bundle.evaluation_manifest.circles
    }
    all_videos = {shot.video_id for shot in bundle.prompt_manifest.shots}
    if len(all_videos) != 10:
        raise PilotDataError("frozen pilot requires exactly 10 videos")
    if all_videos != set(FROZEN_VIDEO_IDS):
        raise PilotDataError("frozen pilot requires the approved video set")
    if bundle.evaluation_manifest.shots != bundle.prompt_manifest.shots:
        raise PilotDataError("evaluation shots must exactly match prompt shots")
    if bundle.evaluation_manifest.dataset_id != bundle.prompt_manifest.dataset_id:
        raise PilotDataError("evaluation dataset identity must match prompt dataset identity")
    if bundle.evaluation_manifest.prompt_manifest_id != bundle.prompt_manifest.manifest_id:
        raise PilotDataError("evaluation prompt identity must match prompt manifest identity")
    if len(bundle.prompt_manifest.shots) != FROZEN_SHOT_COUNT:
        raise PilotDataError("frozen pilot requires exactly 25 shots")
    if len(prompt_frames) != FROZEN_PROMPT_FRAME_COUNT:
        raise PilotDataError("frozen pilot requires exactly 25 prompt frames")
    if len(evaluation_frames) != FROZEN_EVALUATION_FRAME_COUNT:
        raise PilotDataError("frozen pilot requires exactly 50 evaluation frames")
    if len(prompt_frames | evaluation_frames) != FROZEN_REVIEW_FRAME_COUNT:
        raise PilotDataError("frozen pilot requires exactly 75 review frames")
    if (
        len(bundle.prompt_manifest.circles) + len(bundle.evaluation_manifest.circles)
        != FROZEN_CIRCLE_COUNT
    ):
        raise PilotDataError("frozen pilot requires exactly 768 circles")
    prompt_shots = {(shot.video_id, shot.shot_index) for shot in bundle.prompt_manifest.shots}
    shots_by_key = {(shot.video_id, shot.shot_index): shot for shot in bundle.prompt_manifest.shots}
    prompt_circle_shots = {
        (circle.video_id, circle.shot_index) for circle in bundle.prompt_manifest.circles
    }
    evaluation_circle_shots = {
        (circle.video_id, circle.shot_index) for circle in bundle.evaluation_manifest.circles
    }
    if prompt_circle_shots != prompt_shots:
        raise PilotDataError("prompt circles must correspond exactly to prompt shots")
    if evaluation_circle_shots != prompt_shots:
        raise PilotDataError("evaluation circles must correspond exactly to prompt shots")
    for circle in bundle.prompt_manifest.circles:
        shot = shots_by_key[(circle.video_id, circle.shot_index)]
        if circle.source_frame_index != shot.start_frame:
            raise PilotDataError("prompt circle source frame must equal its shot start")
    for circle in bundle.evaluation_manifest.circles:
        shot = shots_by_key[(circle.video_id, circle.shot_index)]
        if not shot.start_frame < circle.source_frame_index < shot.end_frame:
            raise PilotDataError("evaluation circle source frame must fall inside its shot")
    evaluation_roles: dict[tuple[str, int], set[str]] = {}
    for circle in bundle.evaluation_manifest.circles:
        evaluation_roles.setdefault((circle.video_id, circle.shot_index), set()).add(circle.role)
    if any(roles != {"middle", "end"} for roles in evaluation_roles.values()):
        raise PilotDataError("evaluation circles must contain middle and end roles for every shot")
    evaluation_videos = {circle.video_id for circle in bundle.evaluation_manifest.circles}
    if evaluation_videos != all_videos:
        raise PilotDataError("evaluation circles must use the approved video set")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify frozen SAM2 pilot manifest counts.")
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    arguments = parser.parse_args(argv)
    prompt = PromptManifest.model_validate_json(
        arguments.prompt_manifest.read_text(encoding="utf-8")
    )
    evaluation = EvaluationManifest.model_validate_json(
        arguments.evaluation_manifest.read_text(encoding="utf-8")
    )
    preflight_frozen_pilot_counts(
        PilotManifestBundle(
            prompt_manifest=prompt,
            evaluation_manifest=evaluation,
            prompt_manifest_path=arguments.prompt_manifest,
            evaluation_manifest_path=arguments.evaluation_manifest,
        )
    )
    print("PASS: frozen SAM2 pilot counts.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
