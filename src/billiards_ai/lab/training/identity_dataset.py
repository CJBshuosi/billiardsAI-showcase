"""Strict crop provenance and balancing for the separate ball-identity model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.lab.training.identity_labels import (
    BallGroup,
    ExactBallIdentity,
    ExactIdentityEvidence,
    validate_identity_label,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SafeId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$", min_length=1),
]


class IdentityDatasetError(RuntimeError):
    """Identity crops are changed, leaking, or missing provenance."""


class IdentityCrop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    crop_id: SafeId
    relative_path: str
    sha256: Sha256
    source_asset_id: SafeId
    source_frame_id: SafeId
    source_frame_sha256: Sha256
    calibration_revision: SafeId | None = None
    calibration_sha256: Sha256 | None = None
    detector_bbox_xywh: tuple[float, float, float, float]
    detector_mask_polygon_px: tuple[tuple[float, float], ...] = Field(min_length=3)
    group_split: SafeId
    group: BallGroup
    exact_identity: ExactBallIdentity | None = None
    exact_evidence: ExactIdentityEvidence | None = None
    visibility: Literal["visible", "partial", "occluded", "unknown"]
    orientation: Literal["number_visible", "number_hidden", "ambiguous"]

    @field_validator("relative_path")
    @classmethod
    def path_is_relative_and_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("crop path must be relative and confined")
        return value

    @field_validator("detector_bbox_xywh")
    @classmethod
    def bbox_is_non_empty(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x, y, width, height = value
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("detector box must be non-empty and non-negative")
        return value

    @model_validator(mode="after")
    def calibration_reference_is_complete_when_present(self) -> Self:
        if (self.calibration_revision is None) != (self.calibration_sha256 is None):
            raise ValueError("identity crop calibration reference must be complete")
        validate_identity_label(
            group=self.group,
            exact_identity=self.exact_identity,
            exact_evidence=self.exact_evidence,
        )
        return self


@dataclass(frozen=True)
class IdentityDataset:
    records: tuple[IdentityCrop, ...]
    group_class_weights: dict[str, float]
    exact_class_weights: dict[str, float]
    exact_labeled_count: int
    split: str

    @property
    def class_weights(self) -> dict[str, float]:
        """Compatibility alias for pre-v2 training backends."""

        return self.exact_class_weights


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _group_and_split(value: str) -> tuple[str, str]:
    if ":" not in value:
        return value, value
    return tuple(value.rsplit(":", 1))  # type: ignore[return-value]


def build_identity_dataset(
    *, crops: tuple[IdentityCrop, ...], source_root: Path, split: str
) -> IdentityDataset:
    if not crops:
        raise IdentityDatasetError("identity dataset requires crops")
    if len({crop.crop_id for crop in crops}) != len(crops):
        raise IdentityDatasetError("crop IDs must be unique")

    group_owners: dict[str, str] = {}
    for crop in crops:
        group, declared_split = _group_and_split(crop.group_split)
        owner = group_owners.setdefault(group, declared_split)
        if owner != declared_split:
            raise IdentityDatasetError("group leakage across identity splits")

    root = source_root.resolve()
    selected = []
    for crop in sorted(crops, key=lambda item: item.crop_id):
        _, declared_split = _group_and_split(crop.group_split)
        if declared_split != split:
            continue
        unresolved = source_root / crop.relative_path
        if unresolved.is_symlink():
            raise IdentityDatasetError("identity crop cannot be a symbolic link")
        path = unresolved.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise IdentityDatasetError("identity crop is missing or escapes source root")
        if _sha256_file(path) != crop.sha256:
            raise IdentityDatasetError("identity crop sha256 mismatch")
        selected.append(crop)
    if not selected:
        raise IdentityDatasetError("requested identity split has no crops")

    group_counts: dict[str, int] = {}
    exact_counts: dict[str, int] = {}
    for crop in selected:
        group_counts[crop.group] = group_counts.get(crop.group, 0) + 1
        if crop.exact_identity is not None:
            exact_counts[crop.exact_identity] = exact_counts.get(crop.exact_identity, 0) + 1
    group_weights = {group: 1.0 / count for group, count in sorted(group_counts.items())}
    exact_weights = {identity: 1.0 / count for identity, count in sorted(exact_counts.items())}
    return IdentityDataset(
        records=tuple(selected),
        group_class_weights=group_weights,
        exact_class_weights=exact_weights,
        exact_labeled_count=sum(exact_counts.values()),
        split=split,
    )
