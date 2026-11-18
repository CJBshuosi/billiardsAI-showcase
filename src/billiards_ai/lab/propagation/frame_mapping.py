"""Coordinate transforms and verified display-oriented SAM2 frame directories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from uuid import uuid4

import cv2
from pydantic import Field, computed_field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import (
    PilotQaIdentity,
    PropagationWindow,
    pilot_qa_identity,
)
from billiards_ai.lab.propagation.pilot_data import ShotDefinition

PointPx = tuple[float, float]
BoxPx = tuple[float, float, float, float]
AffineCoefficients = tuple[float, float, float, float, float, float]
_ROTATIONS = frozenset((0, 90, 180, 270))


class FrameMappingError(RuntimeError):
    """Review coordinates cannot be safely mapped into source-frame pixels."""


class FrameExtractionError(RuntimeError):
    """A source video or its verified JPEG frame cache changed or decoded incorrectly."""


def _finite_pair(value: Sequence[float], *, label: str) -> PointPx:
    if len(value) != 2:
        raise FrameMappingError(f"{label} must contain two coordinates")
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise FrameMappingError(f"{label} coordinates must be numeric") from exc
    if not all(math.isfinite(component) for component in point):
        raise FrameMappingError(f"{label} coordinates must be finite")
    return point


def _finite_box(value: Sequence[float], *, label: str) -> BoxPx:
    if len(value) != 4:
        raise FrameMappingError(f"{label} must contain four coordinates")
    try:
        box = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise FrameMappingError(f"{label} coordinates must be numeric") from exc
    if not all(math.isfinite(component) for component in box):
        raise FrameMappingError(f"{label} coordinates must be finite")
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise FrameMappingError(f"{label} must be non-empty")
    return left, top, right, bottom


class FrameTransform(FrozenContract):
    """Affine conversion from review pixels into display-oriented source pixels.

    ``source_width`` and ``source_height`` describe the encoded source frame.  The
    recorded rotation is applied after review-to-source scaling, so mapped source
    points always address the display-oriented pixels written for SAM2.
    """

    review_width: int = Field(gt=0)
    review_height: int = Field(gt=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    rotation_degrees: int = 0

    @field_validator("rotation_degrees")
    @classmethod
    def rotation_is_supported(cls, value: int) -> int:
        if value not in _ROTATIONS:
            raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees")
        return value

    @computed_field(return_type=tuple[int, int])
    @property
    def display_size(self) -> tuple[int, int]:
        if self.rotation_degrees in {90, 270}:
            return self.source_height, self.source_width
        return self.source_width, self.source_height

    @computed_field(return_type=tuple[float, float, float, float, float, float])
    @property
    def forward_affine(self) -> AffineCoefficients:
        scale_x = self.source_width / self.review_width
        scale_y = self.source_height / self.review_height
        if self.rotation_degrees == 0:
            return scale_x, 0.0, 0.0, 0.0, scale_y, 0.0
        if self.rotation_degrees == 90:
            return 0.0, -scale_y, float(self.source_height - 1), scale_x, 0.0, 0.0
        if self.rotation_degrees == 180:
            return (
                -scale_x,
                0.0,
                float(self.source_width - 1),
                0.0,
                -scale_y,
                float(self.source_height - 1),
            )
        return 0.0, scale_y, 0.0, -scale_x, 0.0, float(self.source_width - 1)

    @computed_field(return_type=tuple[float, float, float, float, float, float])
    @property
    def inverse_affine(self) -> AffineCoefficients:
        return self._inverse_coefficients(self.forward_affine)

    @property
    def _forward_box_affine(self) -> AffineCoefficients:
        scale_x = self.source_width / self.review_width
        scale_y = self.source_height / self.review_height
        if self.rotation_degrees == 0:
            return scale_x, 0.0, 0.0, 0.0, scale_y, 0.0
        if self.rotation_degrees == 90:
            return 0.0, -scale_y, float(self.source_height), scale_x, 0.0, 0.0
        if self.rotation_degrees == 180:
            return -scale_x, 0.0, float(self.source_width), 0.0, -scale_y, float(self.source_height)
        return 0.0, scale_y, 0.0, -scale_x, 0.0, float(self.source_width)

    @property
    def _inverse_box_affine(self) -> AffineCoefficients:
        return self._inverse_coefficients(self._forward_box_affine)

    @staticmethod
    def _inverse_coefficients(coefficients: AffineCoefficients) -> AffineCoefficients:
        a, b, c, d, e, f = coefficients
        determinant = a * e - b * d
        if determinant == 0.0:  # pragma: no cover - positive dimensions prevent this.
            raise FrameMappingError("coordinate transform is not invertible")
        return (
            e / determinant,
            -b / determinant,
            (b * f - e * c) / determinant,
            -d / determinant,
            a / determinant,
            (d * c - a * f) / determinant,
        )

    def map_review_point(self, point: Sequence[float]) -> PointPx:
        """Map a confined review point to display-oriented source coordinates."""

        review_point = _finite_pair(point, label="review point")
        self._require_point_in_bounds(
            review_point, width=self.review_width, height=self.review_height, label="review point"
        )
        source_point = self._apply(self.forward_affine, review_point)
        self._require_point_in_bounds(
            source_point,
            width=self.display_size[0],
            height=self.display_size[1],
            label="source point",
        )
        recovered_review = self._apply(self.inverse_affine, source_point)
        recovered_source = self._apply(self.forward_affine, recovered_review)
        self.validate_source_round_trip(
            source_point=source_point, recovered_source_point=recovered_source
        )
        return source_point

    def map_source_point(self, point: Sequence[float]) -> PointPx:
        """Map a confined display-oriented source point back to review pixels."""

        source_point = _finite_pair(point, label="source point")
        self._require_point_in_bounds(
            source_point,
            width=self.display_size[0],
            height=self.display_size[1],
            label="source point",
        )
        review_point = self._apply(self.inverse_affine, source_point)
        self._require_point_in_bounds(
            review_point, width=self.review_width, height=self.review_height, label="review point"
        )
        recovered_source = self._apply(self.forward_affine, review_point)
        self.validate_source_round_trip(
            source_point=source_point, recovered_source_point=recovered_source
        )
        return review_point

    def map_review_box(self, box: Sequence[float]) -> BoxPx:
        """Map a review box, rejecting a box that escapes the review frame."""

        review_box = _finite_box(box, label="review box")
        self._require_box_in_bounds(
            review_box, width=self.review_width, height=self.review_height, label="review box"
        )
        mapped = self._map_box(self._forward_box_affine, review_box)
        self._require_box_in_bounds(
            mapped, width=self.display_size[0], height=self.display_size[1], label="source box"
        )
        return mapped

    def map_source_box(self, box: Sequence[float]) -> BoxPx:
        """Map a display-oriented source box back to the review coordinate system."""

        source_box = _finite_box(box, label="source box")
        self._require_box_in_bounds(
            source_box, width=self.display_size[0], height=self.display_size[1], label="source box"
        )
        mapped = self._map_box(self._inverse_box_affine, source_box)
        self._require_box_in_bounds(
            mapped, width=self.review_width, height=self.review_height, label="review box"
        )
        return mapped

    def validate_source_round_trip(
        self, *, source_point: Sequence[float], recovered_source_point: Sequence[float]
    ) -> None:
        """Fail closed if a mapped source point cannot return within half a pixel."""

        original = _finite_pair(source_point, label="source point")
        recovered = _finite_pair(recovered_source_point, label="recovered source point")
        if math.dist(original, recovered) - 0.5 > 1e-9:
            raise FrameMappingError("source coordinate round-trip error exceeds 0.5 pixels")

    @staticmethod
    def _apply(coefficients: AffineCoefficients, point: PointPx) -> PointPx:
        a, b, c, d, e, f = coefficients
        x, y = point
        return a * x + b * y + c, d * x + e * y + f

    @classmethod
    def _map_box(cls, coefficients: AffineCoefficients, box: BoxPx) -> BoxPx:
        left, top, right, bottom = box
        points = tuple(
            cls._apply(coefficients, point)
            for point in ((left, top), (left, bottom), (right, top), (right, bottom))
        )
        return (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        )

    @staticmethod
    def _require_point_in_bounds(point: PointPx, *, width: int, height: int, label: str) -> None:
        x, y = point
        if not 0.0 <= x < width or not 0.0 <= y < height:
            raise FrameMappingError(f"{label} escapes its image bounds")

    @classmethod
    def _require_box_in_bounds(cls, box: BoxPx, *, width: int, height: int, label: str) -> None:
        left, top, right, bottom = box
        if not 0.0 <= left < right <= width or not 0.0 <= top < bottom <= height:
            raise FrameMappingError(f"{label} escapes its image bounds")


class SourceVideoMetadata(FrozenContract):
    source_sha256: Sha256
    source_size_bytes: int = Field(gt=0)
    source_mtime_ns: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    display_width: int = Field(gt=0)
    display_height: int = Field(gt=0)
    rotation_degrees: int
    frame_count: int = Field(gt=0)
    fps: float = Field(gt=0)

    @field_validator("rotation_degrees")
    @classmethod
    def rotation_is_supported(cls, value: int) -> int:
        if value not in _ROTATIONS:
            raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees")
        return value

    @model_validator(mode="after")
    def display_dimensions_follow_rotation(self) -> SourceVideoMetadata:
        expected = (
            (self.source_height, self.source_width)
            if self.rotation_degrees in {90, 270}
            else (self.source_width, self.source_height)
        )
        if (self.display_width, self.display_height) != expected:
            raise ValueError("display dimensions do not match source rotation")
        return self


def _jpeg_filename(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.name != value
        or path.suffix.lower() != ".jpg"
    ):
        raise ValueError("JPEG filename must be a confined .jpg basename")
    return value


class ExtractedShotFrame(FrozenContract):
    source_frame_index: int = Field(ge=0)
    shot_local_index: int = Field(ge=0)
    jpeg_filename: NonEmptyStr
    jpeg_sha256: Sha256

    _filename_is_jpeg = field_validator("jpeg_filename")(_jpeg_filename)


class ExtractedShot(FrozenContract):
    shot: ShotDefinition
    source_video_path: Path
    frame_directory: Path
    source_metadata: SourceVideoMetadata
    frames: tuple[ExtractedShotFrame, ...]
    publication_id: Sha256
    publication_directory: Path
    manifest_path: Path

    @model_validator(mode="after")
    def frames_are_exact_shot_cache(self) -> ExtractedShot:
        expected_count = self.shot.end_frame - self.shot.start_frame
        if len(self.frames) != expected_count:
            raise ValueError("shot cache must contain every source frame in the half-open shot")
        for local_index, frame in enumerate(self.frames):
            if frame.source_frame_index != self.shot.start_frame + local_index:
                raise ValueError("shot cache source-frame indices must be exact and consecutive")
            if frame.shot_local_index != local_index:
                raise ValueError("shot cache local indices must be exact and consecutive")
            if frame.jpeg_filename != f"{frame.source_frame_index:08d}.jpg":
                raise ValueError("shot cache JPEG filename must encode its source frame index")
        if self.frame_directory != self.publication_directory / "frames":
            raise ValueError("shot cache must use its published frames directory")
        if self.manifest_path != self.publication_directory / "manifest.json":
            raise ValueError("shot cache must use its published manifest")
        return self


class ExtractedWindowFrame(FrozenContract):
    source_frame_index: int = Field(ge=0)
    shot_local_index: int = Field(ge=0)
    sam2_local_index: int = Field(ge=0)
    jpeg_filename: NonEmptyStr
    jpeg_sha256: Sha256

    _filename_is_jpeg = field_validator("jpeg_filename")(_jpeg_filename)


class ExtractedWindow(FrozenContract):
    shot: ShotDefinition
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    frame_directory: Path
    frames: tuple[ExtractedWindowFrame, ...]
    source_metadata: SourceVideoMetadata
    source_shot_publication_id: Sha256
    publication_id: Sha256
    publication_directory: Path
    manifest_path: Path

    @model_validator(mode="after")
    def frames_are_exact_window(self) -> ExtractedWindow:
        if not self.shot.start_frame <= self.start_frame < self.end_frame <= self.shot.end_frame:
            raise ValueError("window must be a non-empty half-open interval within its shot")
        if len(self.frames) != self.end_frame - self.start_frame:
            raise ValueError("window must contain every frame in its half-open interval")
        for local_index, frame in enumerate(self.frames):
            if frame.source_frame_index != self.start_frame + local_index:
                raise ValueError("window source-frame indices must be exact and consecutive")
            if frame.shot_local_index != frame.source_frame_index - self.shot.start_frame:
                raise ValueError("window shot-local index must map exactly to the source frame")
            if frame.sam2_local_index != local_index:
                raise ValueError("window SAM2-local indices must be exact and consecutive")
            if frame.jpeg_filename != f"{local_index:05d}.jpg":
                raise ValueError("window JPEG filename must encode its SAM2-local frame index")
        if self.frame_directory != self.publication_directory / "frames":
            raise ValueError("window must use its published frames directory")
        if self.manifest_path != self.publication_directory / "manifest.json":
            raise ValueError("window must use its published manifest")
        return self

    @property
    def sam2_frame_directory(self) -> Path:
        """The only directory a SAM2 worker may receive for this window."""

        return self.frame_directory

    @property
    def sam2_worker_input(self) -> Sam2WindowInput:
        return sam2_worker_input(self)


class Sam2WindowInput(FrozenContract):
    """Verified worker handoff that can only be constructed from a window artifact."""

    window: ExtractedWindow
    propagation_window: PropagationWindow | None = None
    qa_identity: PilotQaIdentity = Field(
        default_factory=lambda: pilot_qa_identity(artifact_kind="worker_job")
    )

    @model_validator(mode="after")
    def uses_a_worker_job_qa_identity(self) -> Sam2WindowInput:
        if self.qa_identity.artifact_kind != "worker_job":
            raise ValueError("SAM2 worker input requires a worker-job QA identity")
        if self.propagation_window is not None and (
            self.propagation_window.asset_id != self.window.shot.video_id
            or self.propagation_window.start_frame != self.window.start_frame
            or self.propagation_window.end_frame != self.window.end_frame
        ):
            raise ValueError("bound propagation window does not match the extracted window")
        return self

    @property
    def frame_directory(self) -> Path:
        return self.window.frame_directory


def sam2_worker_input(
    window: ExtractedWindow, *, propagation_window: PropagationWindow | None = None
) -> Sam2WindowInput:
    """Revalidate and hand an official worker only an immutable window directory."""

    if not isinstance(window, ExtractedWindow):
        raise TypeError("SAM2 worker input requires an ExtractedWindow")
    _verify_extracted_window(window)
    return Sam2WindowInput(window=window, propagation_window=propagation_window)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_from_capture(capture: cv2.VideoCapture) -> int:
    property_id = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    raw_rotation = 0.0 if property_id is None else capture.get(property_id)
    if not math.isfinite(raw_rotation):
        raise FrameExtractionError("source video rotation metadata is invalid")
    rotation = int(round(raw_rotation)) % 360
    if abs(raw_rotation - round(raw_rotation)) > 1e-6 or rotation not in _ROTATIONS:
        raise FrameExtractionError("source video rotation metadata is unsupported")
    return rotation


def _positive_integer_metadata(value: float, *, label: str) -> int:
    if not math.isfinite(value) or value <= 0 or abs(value - round(value)) > 1e-6:
        raise FrameExtractionError(f"source video {label} metadata is invalid")
    return int(round(value))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _confined_path(path: Path, *, label: str, directory: bool) -> Path:
    if ".." in path.parts:
        raise FrameExtractionError(f"unsafe {label}: {path}")
    absolute = path.absolute()
    if any(ancestor.is_symlink() for ancestor in (absolute, *absolute.parents)):
        raise FrameExtractionError(f"unsafe {label}: {path}")
    if directory:
        if absolute.exists() and not absolute.is_dir():
            raise FrameExtractionError(f"unsafe {label}: {path}")
        absolute.mkdir(parents=True, exist_ok=True)
        if absolute.is_symlink() or not absolute.is_dir():
            raise FrameExtractionError(f"unsafe {label}: {path}")
    elif not absolute.is_file() or absolute.is_symlink():
        raise FrameExtractionError(f"unsafe {label}: {path}")
    return absolute


def _new_publication_staging(output_root: Path, *, category: str) -> tuple[Path, Path, Path]:
    root = _confined_path(output_root, label="output directory", directory=True)
    publication_root = root / category
    if publication_root.is_symlink():
        raise FrameExtractionError(f"unsafe output directory: {output_root}")
    publication_root.mkdir(exist_ok=True)
    if publication_root.is_symlink() or not publication_root.is_dir():
        raise FrameExtractionError(f"unsafe output directory: {output_root}")
    temporary = publication_root / f".pending-{uuid4().hex}"
    temporary.mkdir()
    frames = temporary / "frames"
    frames.mkdir()
    return publication_root, temporary, frames


def _manifest_bytes(*, kind: str, identity: dict[str, object]) -> tuple[str, bytes]:
    payload = {"schema_version": "1.0", "kind": kind, "identity": identity}
    publication_id = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    manifest = {"publication_id": publication_id, **payload}
    return publication_id, _canonical_bytes(manifest) + b"\n"


def _make_read_only(root: Path) -> None:
    """Best-effort portability: inability to chmod never weakens hash validation."""

    try:
        entries = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
        for entry in entries:
            mode = entry.stat().st_mode
            if entry.is_dir():
                os.chmod(entry, (mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) & ~0o222)
            else:
                os.chmod(entry, mode & ~0o222)
        os.chmod(root, (root.stat().st_mode | stat.S_IXUSR) & ~0o222)
    except OSError:
        return


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for entry in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            os.chmod(entry, entry.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass
    try:
        os.chmod(path, path.stat().st_mode | stat.S_IWUSR)
    except OSError:
        pass
    shutil.rmtree(path)


def _rotate_to_display(frame: object, rotation_degrees: int) -> object:
    if rotation_degrees == 0:
        return frame
    if rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _validate_jpeg(path: Path, *, width: int, height: int) -> None:
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (height, width):
        raise FrameExtractionError(f"JPEG payload failed decode validation: {path}")


def _verify_exact_jpegs(
    directory: Path,
    *,
    frames: Sequence[ExtractedShotFrame] | Sequence[ExtractedWindowFrame],
    width: int,
    height: int,
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise FrameExtractionError(f"unsafe frame directory: {directory}")
    expected = {frame.jpeg_filename for frame in frames}
    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise FrameExtractionError(f"cannot read frame directory: {directory}") from exc
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise FrameExtractionError(f"frame directory contains an unsafe payload: {directory}")
    if {entry.name for entry in entries} != expected:
        raise FrameExtractionError(f"frame directory payload set changed: {directory}")
    for frame in frames:
        payload = directory / frame.jpeg_filename
        if _sha256(payload) != frame.jpeg_sha256:
            raise FrameExtractionError(f"verified JPEG payload changed: {payload}")
        _validate_jpeg(payload, width=width, height=height)


def _verify_publication(
    publication_directory: Path,
    *,
    manifest_bytes: bytes,
    frames: Sequence[ExtractedShotFrame] | Sequence[ExtractedWindowFrame],
    width: int,
    height: int,
) -> Path:
    if publication_directory.is_symlink() or not publication_directory.is_dir():
        raise FrameExtractionError(f"unsafe published directory: {publication_directory}")
    entries = tuple(publication_directory.iterdir())
    if any(entry.is_symlink() for entry in entries) or {entry.name for entry in entries} != {
        "frames",
        "manifest.json",
    }:
        raise FrameExtractionError(
            f"published directory payload set changed: {publication_directory}"
        )
    manifest_path = publication_directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.read_bytes() != manifest_bytes:
        raise FrameExtractionError(f"published manifest changed: {publication_directory}")
    frame_directory = publication_directory / "frames"
    _verify_exact_jpegs(frame_directory, frames=frames, width=width, height=height)
    return frame_directory


def _reuse_or_publish(
    *,
    publication_root: Path,
    temporary: Path,
    publication_id: str,
    manifest_bytes: bytes,
    frames: Sequence[ExtractedShotFrame] | Sequence[ExtractedWindowFrame],
    width: int,
    height: int,
) -> Path:
    target = publication_root / publication_id
    if target.exists():
        _verify_publication(
            target,
            manifest_bytes=manifest_bytes,
            frames=frames,
            width=width,
            height=height,
        )
        return target
    _verify_exact_jpegs(temporary / "frames", frames=frames, width=width, height=height)
    (temporary / "manifest.json").write_bytes(manifest_bytes)
    _make_read_only(temporary)
    try:
        os.replace(temporary, target)
    except OSError as exc:
        if target.exists():
            _verify_publication(
                target,
                manifest_bytes=manifest_bytes,
                frames=frames,
                width=width,
                height=height,
            )
            return target
        raise FrameExtractionError(f"could not publish frame directory: {target}") from exc
    _verify_publication(
        target,
        manifest_bytes=manifest_bytes,
        frames=frames,
        width=width,
        height=height,
    )
    return target


def _shot_manifest(
    *, metadata: SourceVideoMetadata, shot: ShotDefinition, frames: Sequence[ExtractedShotFrame]
) -> tuple[str, bytes]:
    return _manifest_bytes(
        kind="shot-cache",
        identity={
            "source_metadata": metadata.model_dump(mode="json"),
            "shot": shot.model_dump(mode="json"),
            "frames": [frame.model_dump(mode="json") for frame in frames],
        },
    )


def _window_manifest(
    *,
    shot: ExtractedShot,
    start_frame: int,
    end_frame: int,
    frames: Sequence[ExtractedWindowFrame],
) -> tuple[str, bytes]:
    return _manifest_bytes(
        kind="sam2-window",
        identity={
            "shot_publication_id": shot.publication_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": [frame.model_dump(mode="json") for frame in frames],
        },
    )


class SourceGeometry(NamedTuple):
    """The encoded source-frame geometry read from a video's container metadata."""

    source_width: int
    source_height: int
    rotation_degrees: int


def _read_source_geometry(capture: cv2.VideoCapture) -> SourceGeometry:
    """Read encoded dimensions and rotation from an already-opened capture.

    Shared by extract_shot_frames() and probe_source_geometry() so both derive an
    identical FrameTransform: the review->source mapping applied to evaluation
    ground-truth circles must match the one applied to prompt circles exactly.
    """

    source_width = _positive_integer_metadata(capture.get(cv2.CAP_PROP_FRAME_WIDTH), label="width")
    source_height = _positive_integer_metadata(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT), label="height"
    )
    rotation_degrees = _rotation_from_capture(capture)
    return SourceGeometry(
        source_width=source_width,
        source_height=source_height,
        rotation_degrees=rotation_degrees,
    )


def probe_source_geometry(video_path: Path) -> SourceGeometry:
    """Read a source video's encoded width, height and rotation without decoding.

    Used at prepare-bundle time to map held-out evaluation circles from review
    pixels into the same display-oriented source pixels the SAM2 predictions live
    in, using the same cv2 metadata reads extract_shot_frames() performs.
    """

    video_path = _confined_path(video_path, label="source video", directory=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FrameExtractionError(f"cannot open source video: {video_path}")
    auto_orientation = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if auto_orientation is not None:
        capture.set(auto_orientation, 0)
    try:
        return _read_source_geometry(capture)
    finally:
        capture.release()


def extract_shot_frames(
    *, video_path: Path, shot: ShotDefinition, output_dir: Path
) -> ExtractedShot:
    """Decode a complete source video once and freeze its half-open shot cache as JPEGs."""

    video_path = _confined_path(video_path, label="source video", directory=False)
    before_stat = video_path.stat()
    source_sha256 = _sha256(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FrameExtractionError(f"cannot open source video: {video_path}")
    auto_orientation = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if auto_orientation is not None:
        capture.set(auto_orientation, 0)
    try:
        source_width, source_height, rotation_degrees = _read_source_geometry(capture)
        frame_count = _positive_integer_metadata(
            capture.get(cv2.CAP_PROP_FRAME_COUNT), label="frame count"
        )
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise FrameExtractionError("source video FPS metadata is invalid")
        if frame_count < shot.end_frame:
            raise FrameExtractionError("source video is truncated before the requested shot")
        display_width, display_height = (
            (source_height, source_width)
            if rotation_degrees in {90, 270}
            else (source_width, source_height)
        )
        publication_root, temporary, frame_staging = _new_publication_staging(
            output_dir, category="shot-caches"
        )
        frames: list[ExtractedShotFrame] = []
        try:
            decoded_count = 0
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                if frame is None or frame.shape[:2] != (source_height, source_width):
                    raise FrameExtractionError("source video decode metadata changed")
                if shot.start_frame <= decoded_count < shot.end_frame:
                    display_frame = _rotate_to_display(frame, rotation_degrees)
                    if display_frame.shape[:2] != (display_height, display_width):
                        raise FrameExtractionError("source video display orientation is invalid")
                    filename = f"{decoded_count:08d}.jpg"
                    destination = frame_staging / filename
                    if not cv2.imwrite(str(destination), display_frame):
                        raise FrameExtractionError(f"could not write JPEG frame: {destination}")
                    _validate_jpeg(destination, width=display_width, height=display_height)
                    frames.append(
                        ExtractedShotFrame(
                            source_frame_index=decoded_count,
                            shot_local_index=decoded_count - shot.start_frame,
                            jpeg_filename=filename,
                            jpeg_sha256=_sha256(destination),
                        )
                    )
                decoded_count += 1
            if decoded_count != frame_count:
                raise FrameExtractionError(
                    "source video decode is truncated or frame count changed"
                )
            after_stat = video_path.stat()
            if (
                after_stat.st_size != before_stat.st_size
                or after_stat.st_mtime_ns != before_stat.st_mtime_ns
                or _sha256(video_path) != source_sha256
            ):
                raise FrameExtractionError("source video changed while decoding")
            metadata = SourceVideoMetadata(
                source_sha256=source_sha256,
                source_size_bytes=before_stat.st_size,
                source_mtime_ns=before_stat.st_mtime_ns,
                source_width=source_width,
                source_height=source_height,
                display_width=display_width,
                display_height=display_height,
                rotation_degrees=rotation_degrees,
                frame_count=frame_count,
                fps=fps,
            )
            publication_id, manifest_bytes = _shot_manifest(
                metadata=metadata, shot=shot, frames=frames
            )
            publication_directory = _reuse_or_publish(
                publication_root=publication_root,
                temporary=temporary,
                publication_id=publication_id,
                manifest_bytes=manifest_bytes,
                frames=frames,
                width=display_width,
                height=display_height,
            )
            extracted = ExtractedShot(
                shot=shot,
                source_video_path=video_path,
                frame_directory=publication_directory / "frames",
                source_metadata=metadata,
                frames=tuple(frames),
                publication_id=publication_id,
                publication_directory=publication_directory,
                manifest_path=publication_directory / "manifest.json",
            )
            return extracted
        finally:
            if temporary.exists():
                _remove_tree(temporary)
    finally:
        capture.release()


def _verify_extracted_shot(shot: ExtractedShot) -> None:
    video_path = _confined_path(shot.source_video_path, label="source video", directory=False)
    stat = video_path.stat()
    metadata = shot.source_metadata
    if stat.st_size != metadata.source_size_bytes or stat.st_mtime_ns != metadata.source_mtime_ns:
        raise FrameExtractionError("source video changed since shot extraction")
    if _sha256(video_path) != metadata.source_sha256:
        raise FrameExtractionError("source video changed since shot extraction")
    publication_id, manifest_bytes = _shot_manifest(
        metadata=metadata, shot=shot.shot, frames=shot.frames
    )
    if publication_id != shot.publication_id:
        raise FrameExtractionError("shot publication identity changed")
    _verify_publication(
        shot.publication_directory,
        manifest_bytes=manifest_bytes,
        frames=shot.frames,
        width=metadata.display_width,
        height=metadata.display_height,
    )


def materialize_window_frames(
    *, shot: ExtractedShot, start_frame: int, end_frame: int, output_dir: Path
) -> ExtractedWindow:
    """Copy one verified half-open shot interval into a SAM2-only frame directory."""

    if not shot.shot.start_frame <= start_frame < end_frame <= shot.shot.end_frame:
        raise FrameExtractionError("window must be a non-empty half-open interval within its shot")
    _verify_extracted_shot(shot)
    selected = shot.frames[start_frame - shot.shot.start_frame : end_frame - shot.shot.start_frame]
    publication_root, temporary, frame_staging = _new_publication_staging(
        output_dir, category="window-frames"
    )
    frames: list[ExtractedWindowFrame] = []
    try:
        for sam2_local_index, source_frame in enumerate(selected):
            filename = f"{sam2_local_index:05d}.jpg"
            destination = frame_staging / filename
            shutil.copyfile(shot.frame_directory / source_frame.jpeg_filename, destination)
            jpeg_sha256 = _sha256(destination)
            if jpeg_sha256 != source_frame.jpeg_sha256:
                raise FrameExtractionError("window JPEG hash differs from its verified shot cache")
            _validate_jpeg(
                destination,
                width=shot.source_metadata.display_width,
                height=shot.source_metadata.display_height,
            )
            frames.append(
                ExtractedWindowFrame(
                    source_frame_index=source_frame.source_frame_index,
                    shot_local_index=source_frame.shot_local_index,
                    sam2_local_index=sam2_local_index,
                    jpeg_filename=filename,
                    jpeg_sha256=jpeg_sha256,
                )
            )
        publication_id, manifest_bytes = _window_manifest(
            shot=shot, start_frame=start_frame, end_frame=end_frame, frames=frames
        )
        publication_directory = _reuse_or_publish(
            publication_root=publication_root,
            temporary=temporary,
            publication_id=publication_id,
            manifest_bytes=manifest_bytes,
            frames=frames,
            width=shot.source_metadata.display_width,
            height=shot.source_metadata.display_height,
        )
        window = ExtractedWindow(
            shot=shot.shot,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_directory=publication_directory / "frames",
            frames=tuple(frames),
            source_metadata=shot.source_metadata,
            source_shot_publication_id=shot.publication_id,
            publication_id=publication_id,
            publication_directory=publication_directory,
            manifest_path=publication_directory / "manifest.json",
        )
        return window
    finally:
        if temporary.exists():
            _remove_tree(temporary)


def _verify_extracted_window(window: ExtractedWindow) -> None:
    publication_id, manifest_bytes = _manifest_bytes(
        kind="sam2-window",
        identity={
            "shot_publication_id": window.source_shot_publication_id,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "frames": [frame.model_dump(mode="json") for frame in window.frames],
        },
    )
    if publication_id != window.publication_id:
        raise FrameExtractionError("window publication identity changed")
    _verify_publication(
        window.publication_directory,
        manifest_bytes=manifest_bytes,
        frames=window.frames,
        width=window.source_metadata.display_width,
        height=window.source_metadata.display_height,
    )
