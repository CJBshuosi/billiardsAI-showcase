"""Versioned normalized-media contracts at the AD-13h perception boundary."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _reject_control_characters(value: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("control characters are not allowed")
    return value


NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_reject_control_characters),
]


class MediaStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _controlled_relative_path(value: str) -> str:
    if "\\" in value or (len(value) >= 2 and value[1] == ":"):
        raise ValueError("path must be a controlled relative path")
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError("path must be a controlled relative path")
    return value


def frame_sequence_sha256(frame_hashes: tuple[str, ...]) -> str:
    """Hash an ordered decoded-frame sequence with an unambiguous separator."""

    payload = "\n".join(frame_hashes) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class FrameRateStrategy(StrEnum):
    PRESERVE_VFR_SAMPLE_BY_PTS = "PRESERVE_VFR_SAMPLE_BY_PTS"


class MediaEquivalenceRule(StrEnum):
    DECODED_FRAME_SHA256_V1 = "DECODED_FRAME_SHA256_V1"


class TimeBase(MediaStrictFrozenModel):
    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)


class MediaResolution(MediaStrictFrozenModel):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class ToolVersions(MediaStrictFrozenModel):
    ffmpeg: NonEmptyStr
    ffprobe: NonEmptyStr
    decoder: NonEmptyStr


class FrameRecord(MediaStrictFrozenModel):
    index: int = Field(ge=0)
    requested_timestamp_ms: int = Field(ge=0)
    source_pts: int
    image_path: NonEmptyStr
    content_sha256: Sha256

    _image_path_is_controlled = field_validator("image_path")(_controlled_relative_path)


class FrameManifest(MediaStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    input_sha256: Sha256
    processing_config_sha256: Sha256
    interval_ms: int = Field(gt=0)
    source_time_base: TimeBase
    first_source_pts: int
    tool_versions: ToolVersions
    equivalence_rule: Literal[MediaEquivalenceRule.DECODED_FRAME_SHA256_V1]
    decoded_frames_sha256: Sha256
    frames: tuple[FrameRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def frames_are_canonical_and_at_or_after_targets(self) -> FrameManifest:
        previous_pts: int | None = None
        for expected_index, frame in enumerate(self.frames):
            if frame.index != expected_index:
                raise ValueError("frame indices must be contiguous from zero")
            expected_request = expected_index * self.interval_ms
            if frame.requested_timestamp_ms != expected_request:
                raise ValueError("requested timestamps must be exact interval multiples")
            relative_ticks = frame.source_pts - self.first_source_pts
            if (
                relative_ticks * self.source_time_base.numerator * 1_000
                < expected_request * self.source_time_base.denominator
            ):
                raise ValueError("source PTS is before requested timestamp")
            if previous_pts is not None and frame.source_pts < previous_pts:
                raise ValueError("sampled source PTS must be non-decreasing")
            previous_pts = frame.source_pts
        expected_hash = frame_sequence_sha256(tuple(frame.content_sha256 for frame in self.frames))
        if self.decoded_frames_sha256 != expected_hash:
            raise ValueError("decoded frame equivalence hash does not match frames")
        return self


class NormalizedMedia(MediaStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    input_sha256: Sha256
    processing_config_sha256: Sha256
    media_path: NonEmptyStr
    container: NonEmptyStr
    video_codec: NonEmptyStr
    video_profile: NonEmptyStr
    pixel_format: NonEmptyStr
    resolution: MediaResolution
    source_orientation_degrees: Literal[0, 90, 180, 270]
    orientation_degrees: Literal[0, 90, 180, 270]
    time_base: TimeBase
    frame_rate_strategy: Literal[FrameRateStrategy.PRESERVE_VFR_SAMPLE_BY_PTS]
    audio_removed: StrictBool
    tool_versions: ToolVersions
    content_sha256: Sha256
    equivalence_rule: Literal[MediaEquivalenceRule.DECODED_FRAME_SHA256_V1]
    decoded_frames_sha256: Sha256

    _media_path_is_controlled = field_validator("media_path")(_controlled_relative_path)

    @model_validator(mode="after")
    def output_is_canonical(self) -> NormalizedMedia:
        if self.orientation_degrees != 0:
            raise ValueError("normalized media orientation must be zero")
        if not self.audio_removed:
            raise ValueError("normalized media must have audio removed")
        return self

    def is_equivalent_to(self, other: object) -> bool:
        """Compare contract and decoded evidence, not unstable encoded bytes."""

        if not isinstance(other, NormalizedMedia):
            return False
        fields = (
            "schema_version",
            "input_sha256",
            "processing_config_sha256",
            "container",
            "video_codec",
            "video_profile",
            "pixel_format",
            "resolution",
            "source_orientation_degrees",
            "orientation_degrees",
            "time_base",
            "frame_rate_strategy",
            "audio_removed",
            "tool_versions",
            "equivalence_rule",
            "decoded_frames_sha256",
        )
        return all(getattr(self, field) == getattr(other, field) for field in fields)
