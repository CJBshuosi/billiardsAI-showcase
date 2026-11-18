"""Immutable, model-independent vision frame contracts."""

from __future__ import annotations

import hashlib
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256


class PixelPoint(FrozenContract):
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)


class PixelBox(FrozenContract):
    left: float = Field(ge=0, allow_inf_nan=False)
    top: float = Field(ge=0, allow_inf_nan=False)
    right: float = Field(gt=0, allow_inf_nan=False)
    bottom: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("pixel box must have positive area")
        return self


class BallInstanceObservation(FrozenContract):
    detection_id: NonEmptyStr
    bbox: PixelBox
    center: PixelPoint
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    class_label: Literal["ball", "unknown"]
    mask_ref: NonEmptyStr | None = None
    warnings: tuple[NonEmptyStr, ...] = ()


class VisionPackageIdentity(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    package_id: Sha256
    manifest_sha256: Sha256


class VisionFrameInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: Literal["1.0"] = "1.0"
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    pixel_sha256: Sha256
    image: np.ndarray = Field(exclude=True, repr=False)
    table_roi_ref: NonEmptyStr | None = None

    @model_validator(mode="after")
    def image_is_bound_to_metadata(self) -> Self:
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("vision frame image must be an HxWx3 uint8 array")
        if self.image.shape[:2] != (self.height_px, self.width_px):
            raise ValueError("vision frame image shape does not match its metadata")
        if self.image.flags.writeable:
            raise ValueError("vision frame image must be read-only")
        actual_hash = hashlib.sha256(self.image.tobytes()).hexdigest()
        if actual_hash != self.pixel_sha256:
            raise ValueError("vision frame pixel hash mismatch")
        return self


class VisionFrameObservation(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    pixel_sha256: Sha256
    package: VisionPackageIdentity
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    detections: tuple[BallInstanceObservation, ...] = ()
    runtime_ref: NonEmptyStr

    @model_validator(mode="after")
    def detections_match_source_frame(self) -> Self:
        detection_ids = [item.detection_id for item in self.detections]
        if len(detection_ids) != len(set(detection_ids)):
            raise ValueError("detection IDs must be unique within a frame")
        for item in self.detections:
            box = item.bbox
            center = item.center
            if (
                box.right > self.width_px
                or box.bottom > self.height_px
                or center.x >= self.width_px
                or center.y >= self.height_px
            ):
                raise ValueError("detection geometry must remain inside the source frame")
        return self


def make_vision_frame_input(
    image: np.ndarray,
    *,
    frame_id: int,
    timestamp_ms: int,
    table_roi_ref: str | None = None,
) -> VisionFrameInput:
    pixels = np.array(image, copy=True)
    pixels.setflags(write=False)
    height_px, width_px = pixels.shape[:2]
    return VisionFrameInput(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        width_px=width_px,
        height_px=height_px,
        pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        image=pixels,
        table_roi_ref=table_roi_ref,
    )
