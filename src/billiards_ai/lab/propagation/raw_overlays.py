"""Inference-only renderings for immutable SAM2 raw-run inspection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from pydantic import Field, field_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import (
    InstanceAbsence,
    MaskObservation,
    PropagationResult,
)
from billiards_ai.lab.propagation.frame_mapping import ExtractedWindow, _verify_extracted_window


class RawOverlayEvidence(FrozenContract):
    """The inference facts rendered into one raw overlay, without evaluation data."""

    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    instance_ids: tuple[NonEmptyStr, ...]
    qa_flags: tuple[NonEmptyStr, ...]
    trajectory: tuple[tuple[float, float], ...]
    window: dict[str, int]


class RawOverlayPayload(FrozenContract):
    """A confined raw overlay payload and the evidence rendered into it."""

    path: NonEmptyStr
    bytes: int = Field(gt=0)
    sha256: Sha256
    evidence: RawOverlayEvidence

    @field_validator("path")
    @classmethod
    def path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("overlay path must be relative and confined")
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_payload(root: Path, reference: str, *, label: str) -> Path:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or reference in {"", "."}:
        raise ValueError(f"{label} must be relative and confined")
    candidate = root.joinpath(*relative.parts)
    if any(path.is_symlink() for path in (candidate, *candidate.parents)):
        raise ValueError(f"{label} is a symlink")
    if not candidate.is_file():
        raise ValueError(f"{label} is missing")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its root") from exc
    return candidate


def _overlay_reference(asset_id: str, frame_index: int) -> str:
    if (
        not asset_id
        or asset_id in {".", ".."}
        or "/" in asset_id
        or "\\" in asset_id
        or "\x00" in asset_id
        or not asset_id.isascii()
        or any(not (character.isalnum() or character in {"-", "_", "."}) for character in asset_id)
    ):
        raise ValueError("raw overlay asset identifier is unsafe")
    return f"overlays/{asset_id}/{frame_index:08d}.png"


def _safe_destination(root: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("raw overlay destination is unsafe")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("raw overlay output root is unsafe")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("raw overlay destination is unsafe")
        else:
            current.mkdir()
            if current.is_symlink() or not current.is_dir():
                raise ValueError("raw overlay destination is unsafe")
    destination = current / relative.name
    if destination.is_symlink() or destination.exists():
        raise ValueError("raw overlay destination already exists or is unsafe")
    try:
        destination.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError("raw overlay destination escapes output root") from exc
    return destination


def _draw_mask(image: np.ndarray, mask: np.ndarray) -> None:
    if mask.shape != image.shape[:2]:
        raise ValueError("mask dimensions do not match prepared frame")
    if mask.dtype != np.uint8:
        raise ValueError("mask must be an 8-bit image")
    image[mask > 0] = (0.45 * image[mask > 0] + np.array([0, 0, 255])).astype(np.uint8)


def _evidence(
    *,
    asset_id: str,
    frame_index: int,
    result: PropagationResult,
    observations: Sequence[MaskObservation],
    absences: Sequence[InstanceAbsence],
) -> RawOverlayEvidence:
    instance_ids = {item.instance_id for item in (*observations, *absences)}
    trajectory = tuple(
        observation.center_px
        for observation in sorted(result.observations, key=lambda item: item.frame_index)
        if observation.instance_id in instance_ids and observation.frame_index <= frame_index
    )
    return RawOverlayEvidence(
        asset_id=asset_id,
        frame_index=frame_index,
        instance_ids=tuple(sorted(instance_ids)),
        qa_flags=tuple(
            sorted({flag for item in (*observations, *absences) for flag in item.qa_flags})
        ),
        trajectory=trajectory,
        window={
            "ordinal": result.window.ordinal,
            "start_frame": result.window.start_frame,
            "end_frame": result.window.end_frame,
        },
    )


def render_raw_overlays(
    *,
    prepared_windows: Sequence[ExtractedWindow],
    results: Sequence[PropagationResult],
    payload_root: Path,
    output_root: Path,
) -> tuple[RawOverlayPayload, ...]:
    """Render raw overlays from prepared frames and raw inference contracts only.

    The API deliberately has no evaluation manifest, labels, or metrics parameter.
    """

    if payload_root.is_symlink() or not payload_root.is_dir():
        raise ValueError("payload root must be a regular directory")
    if any(path.is_symlink() for path in payload_root.rglob("*")):
        raise ValueError("payload root contains a symlink")
    try:
        relative_output = output_root.resolve().relative_to(payload_root.resolve())
    except ValueError as exc:
        raise ValueError("overlay output must stay inside the payload root") from exc
    if relative_output.as_posix() != "overlays":
        raise ValueError("raw overlays must use the confined overlays directory")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise ValueError("overlay output must not be a symlink")

    prepared_frames: dict[tuple[str, int, int], dict[int, Path]] = {}
    for prepared in prepared_windows:
        if not isinstance(prepared, ExtractedWindow):
            raise TypeError("raw overlays require verified ExtractedWindow contracts")
        _verify_extracted_window(prepared)
        key = (
            prepared.shot.video_id,
            prepared.start_frame,
            prepared.end_frame,
        )
        if key in prepared_frames:
            raise ValueError("prepared window is duplicated")
        prepared_frames[key] = {
            frame.source_frame_index: prepared.frame_directory / frame.jpeg_filename
            for frame in prepared.frames
        }

    grouped: dict[
        tuple[str, int], tuple[PropagationResult, list[MaskObservation], list[InstanceAbsence]]
    ] = {}
    for result in results:
        if not isinstance(result, PropagationResult):
            raise TypeError("raw overlays require propagation result contracts")
        for observation in result.observations:
            if observation.provenance != "pseudo" or observation.review_status != "unreviewed":
                raise ValueError("raw observations must remain pseudo/unreviewed")
        for absence in result.absences:
            if absence.provenance != "pseudo" or absence.review_status != "unreviewed":
                raise ValueError("raw absences must remain pseudo/unreviewed")
        items: defaultdict[int, list[MaskObservation]] = defaultdict(list)
        missing: defaultdict[int, list[InstanceAbsence]] = defaultdict(list)
        for observation in result.observations:
            items[observation.frame_index].append(observation)
        for absence in result.absences:
            missing[absence.frame_index].append(absence)
        for frame_index in sorted(set(items) | set(missing)):
            key = (result.window.asset_id, frame_index)
            if key in grouped:
                raise ValueError("raw overlay frame is duplicated across inference results")
            grouped[key] = (result, items[frame_index], missing[frame_index])

    payloads: list[RawOverlayPayload] = []
    for (asset_id, frame_index), (result, observations, absences) in sorted(grouped.items()):
        window_key = (
            result.window.asset_id,
            result.window.start_frame,
            result.window.end_frame,
        )
        try:
            prepared = prepared_frames[window_key][frame_index]
        except KeyError as exc:
            raise ValueError("prepared window is missing for raw overlay") from exc
        if prepared.is_symlink() or not prepared.is_file():
            raise ValueError("prepared frame is missing or unsafe")
        image = cv2.imread(str(prepared), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("prepared frame cannot be decoded")
        for observation in observations:
            mask_path = _safe_payload(payload_root, observation.mask_ref, label="mask payload")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("mask payload cannot be decoded")
            _draw_mask(image, mask)
            left, top, right, bottom = (int(value) for value in observation.box_xyxy_px)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 1)
            cv2.putText(
                image,
                observation.instance_id,
                (left, max(10, top - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )
            points = [
                item.center_px
                for item in sorted(result.observations, key=lambda item: item.frame_index)
                if item.instance_id == observation.instance_id and item.frame_index <= frame_index
            ]
            if len(points) > 1:
                cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (255, 0, 0), 1)
        evidence = _evidence(
            asset_id=asset_id,
            frame_index=frame_index,
            result=result,
            observations=observations,
            absences=absences,
        )
        labels = " | ".join((*evidence.instance_ids, *evidence.qa_flags))
        cv2.putText(
            image,
            f"window {result.window.start_frame}:{result.window.end_frame} {labels}",
            (1, image.shape[0] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )
        reference = _overlay_reference(asset_id, frame_index)
        destination = _safe_destination(payload_root, reference)
        if not cv2.imwrite(str(destination), image):
            raise ValueError("raw overlay could not be written")
        payloads.append(
            RawOverlayPayload(
                path=reference,
                bytes=destination.stat().st_size,
                sha256=_sha256(destination),
                evidence=evidence,
            )
        )
    return tuple(payloads)
