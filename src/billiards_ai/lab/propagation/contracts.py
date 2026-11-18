"""Teacher-neutral contracts for bounded video-mask propagation."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256

PointPx = tuple[float, float]
BoxPx = tuple[float, float, float, float]


def _confined_reference(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("path reference must be relative and confined")
    return value


def _non_empty_box(value: BoxPx | None) -> BoxPx | None:
    if value is not None:
        left, top, right, bottom = value
        if right <= left or bottom <= top:
            raise ValueError("box must be non-empty")
    return value


class InstancePrompt(FrozenContract):
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    instance_id: NonEmptyStr
    class_name: NonEmptyStr
    source_kind: Literal["gold", "pseudo", "detector"]
    source_id: NonEmptyStr
    box_xyxy_px: BoxPx | None = None
    mask_ref: str | None = None

    _box_is_non_empty = field_validator("box_xyxy_px")(_non_empty_box)

    @field_validator("mask_ref")
    @classmethod
    def mask_reference_is_confined(cls, value: str | None) -> str | None:
        return None if value is None else _confined_reference(value)

    @model_validator(mode="after")
    def carries_exactly_one_prompt_geometry(self) -> InstancePrompt:
        if (self.box_xyxy_px is None) == (self.mask_ref is None):
            raise ValueError("prompt requires exactly one box or mask reference")
        return self


class PropagationWindow(FrozenContract):
    asset_id: NonEmptyStr
    ordinal: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def interval_is_non_empty(self) -> PropagationWindow:
        if self.end_frame <= self.start_frame:
            raise ValueError("window must be a non-empty half-open interval")
        return self


class MaskObservation(FrozenContract):
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    instance_id: NonEmptyStr
    class_name: NonEmptyStr
    prompt_frame_index: int = Field(ge=0)
    center_px: PointPx
    box_xyxy_px: BoxPx
    area_px: int = Field(gt=0)
    mask_ref: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    qa_flags: tuple[str, ...] = ()
    provenance: Literal["pseudo"] = "pseudo"
    review_status: Literal["unreviewed"] = "unreviewed"

    _box_is_non_empty = field_validator("box_xyxy_px")(_non_empty_box)

    @field_validator("mask_ref")
    @classmethod
    def mask_reference_is_confined(cls, value: str) -> str:
        return _confined_reference(value)

    @field_validator("qa_flags")
    @classmethod
    def qa_flags_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("QA flags must be unique")
        return tuple(sorted(value))


class InstanceAbsence(FrozenContract):
    """Explicit evidence that an instance has no usable mask on one frame."""

    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    instance_id: NonEmptyStr
    class_name: NonEmptyStr
    prompt_frame_index: int = Field(ge=0)
    reason: NonEmptyStr
    qa_flags: tuple[str, ...] = ()
    provenance: Literal["pseudo"] = "pseudo"
    review_status: Literal["unreviewed"] = "unreviewed"

    @field_validator("qa_flags")
    @classmethod
    def qa_flags_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("QA flags must be unique")
        return tuple(sorted(value))


class PilotQaConfig(FrozenContract):
    """The fixed, content-addressed SAM2 pilot QA thresholds."""

    minimum_overlap_iou: float = 0.25
    maximum_overlap_center_distance_radii: float = 1.0
    maximum_center_jump_radii_per_source_frame: float = 4.0
    maximum_area_ratio: float = 2.0
    ambiguity_iou_margin: float = 0.05
    ambiguity_center_distance_margin_radii: float = 0.25

    @model_validator(mode="after")
    def thresholds_are_exactly_frozen(self) -> PilotQaConfig:
        expected = {
            "minimum_overlap_iou": 0.25,
            "maximum_overlap_center_distance_radii": 1.0,
            "maximum_center_jump_radii_per_source_frame": 4.0,
            "maximum_area_ratio": 2.0,
            "ambiguity_iou_margin": 0.05,
            "ambiguity_center_distance_margin_radii": 0.25,
        }
        if self.model_dump() != expected:
            raise ValueError("pilot QA thresholds are frozen and cannot be overridden")
        return self

    @property
    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class PilotQaIdentity(FrozenContract):
    """Reusable QA identity for worker jobs, frozen raw runs, and evaluations."""

    artifact_kind: Literal["worker_job", "raw_run", "evaluation_report"]
    pilot_qa_config_hash: Sha256 = Field(default_factory=lambda: PilotQaConfig().canonical_hash)

    @model_validator(mode="after")
    def identifies_the_frozen_qa_configuration(self) -> PilotQaIdentity:
        if self.pilot_qa_config_hash != PilotQaConfig().canonical_hash:
            raise ValueError("artifact must use the frozen QA configuration hash")
        return self


def pilot_qa_identity(
    *, artifact_kind: Literal["worker_job", "raw_run", "evaluation_report"]
) -> PilotQaIdentity:
    return PilotQaIdentity(artifact_kind=artifact_kind)


class PropagationFailure(FrozenContract):
    asset_id: NonEmptyStr
    frame_index: int | None = Field(default=None, ge=0)
    code: NonEmptyStr
    message: NonEmptyStr


class PropagationResult(FrozenContract):
    backend_id: NonEmptyStr
    backend_version: NonEmptyStr
    pilot_qa_config_hash: Sha256 = Field(default_factory=lambda: PilotQaConfig().canonical_hash)
    window: PropagationWindow
    prompts: tuple[InstancePrompt, ...]
    observations: tuple[MaskObservation, ...] = ()
    absences: tuple[InstanceAbsence, ...] = ()
    failures: tuple[PropagationFailure, ...] = ()

    @model_validator(mode="after")
    def contents_are_confined_to_window(self) -> PropagationResult:
        if self.pilot_qa_config_hash != PilotQaConfig().canonical_hash:
            raise ValueError("result must use the frozen QA configuration hash")
        prompt_keys: set[tuple[int, str]] = set()
        prompted_instances: set[str] = set()
        for prompt in self.prompts:
            if prompt.asset_id != self.window.asset_id:
                raise ValueError("prompt must reference the result asset")
            if not self.window.start_frame <= prompt.frame_index < self.window.end_frame:
                raise ValueError("prompt must fall inside the result window")
            key = (prompt.frame_index, prompt.instance_id)
            if key in prompt_keys:
                raise ValueError("prompt instance IDs must be unique on each frame")
            prompt_keys.add(key)
            prompted_instances.add(prompt.instance_id)

        observation_keys: set[tuple[int, str]] = set()
        for observation in self.observations:
            if observation.asset_id != self.window.asset_id:
                raise ValueError("observation must reference the result asset")
            if not self.window.start_frame <= observation.frame_index < self.window.end_frame:
                raise ValueError("observation must fall inside the result window")
            if observation.instance_id not in prompted_instances:
                raise ValueError("observation must reference a prompted instance")

            key = (observation.frame_index, observation.instance_id)
            if key in observation_keys:
                raise ValueError("observation evidence must be unique per instance and frame")
            observation_keys.add(key)

        absence_keys: set[tuple[int, str]] = set()
        for absence in self.absences:
            if absence.asset_id != self.window.asset_id:
                raise ValueError("absence must reference the result asset")
            if not self.window.start_frame <= absence.frame_index < self.window.end_frame:
                raise ValueError("absence must fall inside the result window")
            if absence.instance_id not in prompted_instances:
                raise ValueError("absence must reference a prompted instance")
            key = (absence.frame_index, absence.instance_id)
            if key in absence_keys:
                raise ValueError("absence evidence must be unique per instance and frame")
            if key in observation_keys:
                raise ValueError("an instance cannot both be observed and absent on one frame")
            absence_keys.add(key)

        for failure in self.failures:
            if failure.asset_id != self.window.asset_id:
                raise ValueError("failure must reference the result asset")
            if failure.frame_index is not None and not (
                self.window.start_frame <= failure.frame_index < self.window.end_frame
            ):
                raise ValueError("failure must fall inside the result window")
        return self
