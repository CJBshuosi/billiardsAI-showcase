"""Capture-source identity used by the canonical Mainline 1 calibration gate."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaptureSourceKind(StrEnum):
    FILE = "file"
    USB = "usb"


class FrameGeometry(_FrozenModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crop: tuple[int, int, int, int]
    rotation_degrees: Literal[0, 90, 180, 270] = 0

    @model_validator(mode="after")
    def validate_crop(self) -> FrameGeometry:
        left, top, right, bottom = self.crop
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("crop must be a positive left/top/right/bottom rectangle")
        if right > self.width or bottom > self.height:
            raise ValueError("crop must remain inside the decoded frame")
        return self


class CaptureSetup(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    capture_setup_id: str = Field(min_length=1)
    table_id: str = Field(min_length=1)
    camera_device_id: str = Field(min_length=1)
    lens_id: str = Field(min_length=1)
    mount_revision: str = Field(min_length=1)
    distortion_revision: str = Field(min_length=1)
    source_kind: CaptureSourceKind
    geometry: FrameGeometry
    requested_fps: float | None = Field(default=None, gt=0)
    observed_fps: float | None = Field(default=None, gt=0)
    device_locator_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    table_reference_points_px: tuple[tuple[float, float], ...] | None = None
    validation_corners_px: tuple[tuple[float, float], ...] | None = None
    image_controls: tuple[tuple[str, float], ...] = ()

    @model_validator(mode="after")
    def validate_corner_evidence(self) -> CaptureSetup:
        corners = self.validation_corners_px
        if corners is not None and (
            len(corners) != 4
            or not all(math.isfinite(value) for point in corners for value in point)
        ):
            raise ValueError("validation_corners_px must contain exactly four finite corners")
        references = self.table_reference_points_px
        if references is not None and (
            not references
            or not all(math.isfinite(value) for point in references for value in point)
        ):
            raise ValueError("table_reference_points_px must contain finite points")
        control_names = [name for name, _ in self.image_controls]
        if len(control_names) != len(set(control_names)) or any(
            not name.strip() or not math.isfinite(value) for name, value in self.image_controls
        ):
            raise ValueError("image_controls must contain unique finite named values")
        return self


class FileSourceMetadata(_FrozenModel):
    path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry: FrameGeometry
    frame_count: int = Field(gt=0)
    fps: float = Field(gt=0)


class UsbSourceMetadata(_FrozenModel):
    device_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry: FrameGeometry
    fps: float = Field(gt=0)
    source_epoch_id: str = Field(min_length=1)
    evidence_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_corners_px: tuple[tuple[float, float], ...] | None = None
    image_controls: tuple[tuple[str, float], ...]

    @model_validator(mode="after")
    def validate_live_evidence(self) -> UsbSourceMetadata:
        if self.validation_corners_px is not None and (
            len(self.validation_corners_px) != 4
            or not all(
                math.isfinite(value) for point in self.validation_corners_px for value in point
            )
        ):
            raise ValueError("validation_corners_px must contain exactly four finite corners")
        names = [name for name, _ in self.image_controls]
        if len(names) != len(set(names)) or any(
            not name.strip() or not math.isfinite(value) for name, value in self.image_controls
        ):
            raise ValueError("image_controls must contain unique finite named values")
        return self


def _safe_existing_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return resolved


def load_capture_setup(path: Path) -> CaptureSetup:
    source = _safe_existing_file(path, label="capture setup")
    return CaptureSetup.model_validate_json(source.read_bytes())


def capture_setup_fingerprint(setup: CaptureSetup) -> str:
    payload = setup.model_dump(mode="json", exclude={"validation_corners_px"})
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_file_source(path: Path) -> FileSourceMetadata:
    source = _safe_existing_file(path, label="video source")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("video source could not be opened")
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
            raise ValueError("video source metadata is incomplete")
    finally:
        capture.release()
    return FileSourceMetadata(
        path=source,
        source_sha256=_sha256(source),
        geometry=FrameGeometry(width=width, height=height, crop=(0, 0, width, height)),
        frame_count=frame_count,
        fps=fps,
    )
