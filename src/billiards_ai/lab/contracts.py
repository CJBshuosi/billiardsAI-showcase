"""Versioned contracts for frozen algorithm datasets and experiment runs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.models import Point

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BallIdentity = Annotated[str, Field(pattern=r"^(cue|unknown|[1-9]|1[0-5])$")]


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetPurpose(StrEnum):
    LOCAL_VIEW = "local_view"
    ANNOTATION = "annotation"
    BENCHMARK = "benchmark"
    LOCAL_TRAINING = "local_training"
    CLOUD_TRAINING = "cloud_training"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    LOCKED_TEST = "locked-test"


class Visibility(StrEnum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    FIRST_CONTACT = "first_contact"
    CUSHION = "cushion"
    POT = "pot"
    FOUL = "foul"


class AdjudicationStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    AGREED = "agreed"
    ADJUDICATED = "adjudicated"
    UNSUPPORTED = "unsupported"


class CalibrationReference(FrozenContract):
    revision: NonEmptyStr
    sha256: Sha256


class VideoAsset(FrozenContract):
    asset_id: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: Sha256
    is_real_person: bool
    person_group_id: NonEmptyStr
    session_group_id: NonEmptyStr
    table_group_id: NonEmptyStr
    camera_group_id: NonEmptyStr
    calibration: CalibrationReference
    gate_revision_id: UUID | None = None
    approved_uses: tuple[DatasetPurpose, ...] = ()

    @field_validator("relative_path")
    @classmethod
    def path_is_relative_and_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("asset path must be relative and confined")
        return value

    @field_validator("approved_uses")
    @classmethod
    def approved_uses_are_unique(
        cls, value: tuple[DatasetPurpose, ...]
    ) -> tuple[DatasetPurpose, ...]:
        if len(value) != len(set(value)):
            raise ValueError("approved uses must be unique")
        return tuple(sorted(value, key=lambda item: item.value))

    @model_validator(mode="after")
    def real_assets_carry_gate_evidence(self) -> VideoAsset:
        if self.is_real_person and (self.gate_revision_id is None or not self.approved_uses):
            raise ValueError("real-person assets require gate revision and approved uses")
        if not self.is_real_person and self.gate_revision_id is not None:
            raise ValueError("synthetic assets cannot claim a real-video gate revision")
        return self


class BallFrameAnnotation(FrozenContract):
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    track_id: NonEmptyStr
    identity: BallIdentity
    visibility: Visibility
    center_mm: Point
    segmentation_polygon_mm: tuple[Point, ...] | None = None

    @field_validator("center_mm")
    @classmethod
    def center_is_on_table(cls, value: Point) -> Point:
        if not value.is_on_table():
            raise ValueError("ball center must be inside calibrated table coordinates")
        return value

    @field_validator("segmentation_polygon_mm")
    @classmethod
    def polygon_is_valid(cls, value: tuple[Point, ...] | None) -> tuple[Point, ...] | None:
        if value is not None:
            if len(value) < 3:
                raise ValueError("segmentation polygon requires at least three points")
            if any(not point.is_on_table() for point in value):
                raise ValueError("segmentation polygon must use table millimetres")
        return value


class EventAnnotation(FrozenContract):
    event_type: EventType
    timestamp_ms: int = Field(ge=0)
    actor_identity: BallIdentity | None = None
    target_identity: BallIdentity | None = None
    evidence_frame_ids: tuple[int, ...] = Field(min_length=1)
    eligible: bool
    adjudication_status: AdjudicationStatus = AdjudicationStatus.UNREVIEWED

    @field_validator("evidence_frame_ids")
    @classmethod
    def evidence_frames_are_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(frame_id < 0 for frame_id in value):
            raise ValueError("evidence frame IDs must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("evidence frame IDs must be unique")
        return tuple(sorted(value))


class ShotAnnotation(FrozenContract):
    shot_id: NonEmptyStr
    asset_id: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    ball_frames: tuple[BallFrameAnnotation, ...] = ()
    events: tuple[EventAnnotation, ...] = ()

    @model_validator(mode="after")
    def interval_and_evidence_are_consistent(self) -> ShotAnnotation:
        if self.end_ms <= self.start_ms:
            raise ValueError("shot interval must be a non-empty half-open interval")
        if any(not self.start_ms <= frame.timestamp_ms < self.end_ms for frame in self.ball_frames):
            raise ValueError("ball frame timestamp must fall inside shot interval")
        if any(not self.start_ms <= event.timestamp_ms < self.end_ms for event in self.events):
            raise ValueError("event timestamp must fall inside shot interval")
        return self


class DatasetManifest(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    revision: NonEmptyStr
    coordinate_unit: Literal["mm"] = "mm"
    locked: bool = False
    assets: tuple[VideoAsset, ...]
    splits: dict[DatasetSplit, tuple[str, ...]]
    shots: tuple[ShotAnnotation, ...] = ()

    @model_validator(mode="after")
    def manifest_is_consistent(self) -> DatasetManifest:
        if not self.assets:
            raise ValueError("dataset requires at least one asset")
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset IDs must be unique")
        asset_hashes = [asset.sha256 for asset in self.assets]
        if len(asset_hashes) != len(set(asset_hashes)):
            raise ValueError("asset sha256 values must be unique")
        if not self.splits:
            raise ValueError("dataset requires at least one split")

        assigned: dict[str, DatasetSplit] = {}
        asset_by_id = {asset.asset_id: asset for asset in self.assets}
        group_owners: dict[tuple[str, str], DatasetSplit] = {}
        for split, split_asset_ids in self.splits.items():
            if not split_asset_ids:
                raise ValueError("dataset splits cannot be empty")
            for asset_id in split_asset_ids:
                if asset_id not in asset_by_id:
                    raise ValueError("split references an unknown asset")
                if asset_id in assigned:
                    raise ValueError("asset cannot appear in multiple splits")
                assigned[asset_id] = split
                asset = asset_by_id[asset_id]
                groups = {
                    ("person", asset.person_group_id),
                    ("session", asset.session_group_id),
                    ("table", asset.table_group_id),
                    ("camera", asset.camera_group_id),
                }
                for group in groups:
                    owner = group_owners.setdefault(group, split)
                    if owner is not split:
                        raise ValueError("group isolation violation")
        if set(assigned) != set(asset_ids):
            raise ValueError("every asset must belong to exactly one split")

        shots_by_asset: dict[str, list[ShotAnnotation]] = {}
        shot_ids: set[str] = set()
        for shot in self.shots:
            if shot.shot_id in shot_ids:
                raise ValueError("shot IDs must be unique")
            shot_ids.add(shot.shot_id)
            if shot.asset_id not in asset_by_id:
                raise ValueError("shot references an unknown asset")
            shots_by_asset.setdefault(shot.asset_id, []).append(shot)
        for asset_shots in shots_by_asset.values():
            ordered = sorted(asset_shots, key=lambda shot: (shot.start_ms, shot.end_ms))
            adjacent_shots = zip(ordered, ordered[1:], strict=False)
            if any(previous.end_ms > current.start_ms for previous, current in adjacent_shots):
                raise ValueError("shot intervals overlap")
        return self


class RunManifest(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    run_id: NonEmptyStr
    dataset_revision: NonEmptyStr
    dataset_manifest_sha256: Sha256
    candidate_id: NonEmptyStr
    candidate_config_sha256: Sha256
    code_commit: NonEmptyStr
    dependency_lock_sha256: Sha256
    hardware: NonEmptyStr
    seed: int
