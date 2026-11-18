"""Teacher-neutral pseudo labels and deterministic active-review selection.

Teacher output is deliberately confined to ``pseudo`` / ``unreviewed`` state.  Human
adjudication belongs to the frozen dataset contracts and cannot happen in this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256

PointPx = tuple[float, float]
BoxPx = tuple[float, float, float, float]


class FrameGeometry(FrozenContract):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    table_bounds_px: BoxPx
    minimum_ball_diameter_px: float = Field(default=8.0, gt=0)
    maximum_ball_diameter_px: float = Field(default=80.0, gt=0)
    fusion_center_distance_px: float = Field(default=16.0, gt=0)

    @model_validator(mode="after")
    def bounds_and_diameters_are_valid(self) -> FrameGeometry:
        left, top, right, bottom = self.table_bounds_px
        if not (0 <= left < right <= self.width_px and 0 <= top < bottom <= self.height_px):
            raise ValueError("table bounds must be confined to the frame")
        if self.maximum_ball_diameter_px <= self.minimum_ball_diameter_px:
            raise ValueError("maximum ball diameter must exceed minimum")
        return self


class MotionContext(FrozenContract):
    active_ratio: float = Field(ge=0.0, le=1.0)
    speed_px_per_s: float = Field(ge=0.0)
    blur_score: float = Field(default=0.0, ge=0.0, le=1.0)


class InstanceCandidate(FrozenContract):
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    teacher_id: NonEmptyStr
    teacher_version: NonEmptyStr
    teacher_sha256: Sha256
    class_name: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)
    center_px: PointPx
    box_xyxy_px: BoxPx
    mask_polygon_px: tuple[PointPx, ...] | None = None
    proposed_track_id: str | None = None
    identity_candidates: tuple[str, ...] = ()
    occluded: bool = False

    @field_validator("box_xyxy_px")
    @classmethod
    def box_is_non_empty(cls, value: BoxPx) -> BoxPx:
        left, top, right, bottom = value
        if right <= left or bottom <= top:
            raise ValueError("candidate box must be non-empty")
        return value

    @field_validator("mask_polygon_px")
    @classmethod
    def mask_is_polygon(cls, value: tuple[PointPx, ...] | None) -> tuple[PointPx, ...] | None:
        if value is not None and len(value) < 3:
            raise ValueError("mask polygon requires at least three points")
        return value


class TrackCandidate(FrozenContract):
    proposed_track_id: NonEmptyStr
    asset_id: NonEmptyStr
    frame_indices: tuple[int, ...] = Field(min_length=1)
    teacher_id: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)


class PseudoLabel(FrozenContract):
    label_id: Sha256
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    class_name: NonEmptyStr
    center_px: PointPx
    box_xyxy_px: BoxPx
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: Literal["pseudo"] = "pseudo"
    review_status: Literal["unreviewed"] = "unreviewed"
    teacher_ids: tuple[str, ...] = Field(min_length=1)
    teacher_evidence: tuple[InstanceCandidate, ...] = Field(min_length=1)
    teacher_agreement: bool
    occluded: bool
    blur_score: float = Field(ge=0.0, le=1.0)
    speed_px_per_s: float = Field(ge=0.0)
    near_table_edge: bool
    identity_ambiguous: bool


ReviewReason = Literal[
    "teacher_disagreement",
    "occlusion",
    "low_confidence",
    "blur",
    "rail_or_pocket_proximity",
    "high_speed",
    "identity_orientation",
    "easy_sample",
]


class ReviewItem(FrozenContract):
    label_id: Sha256
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    reason: ReviewReason
    priority: int = Field(ge=0)
    review_status: Literal["unreviewed"] = "unreviewed"


class ReviewStrata(FrozenContract):
    low_confidence_below: float = Field(default=0.60, ge=0.0, le=1.0)
    blur_at_least: float = Field(default=0.65, ge=0.0, le=1.0)
    high_speed_at_least_px_per_s: float = Field(default=700.0, ge=0.0)


def _distance(first: PointPx, second: PointPx) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _is_valid(candidate: InstanceCandidate, geometry: FrameGeometry) -> bool:
    x_px, y_px = candidate.center_px
    left, top, right, bottom = geometry.table_bounds_px
    if not (left <= x_px <= right and top <= y_px <= bottom):
        return False
    box_left, box_top, box_right, box_bottom = candidate.box_xyxy_px
    diameter = max(box_right - box_left, box_bottom - box_top)
    return geometry.minimum_ball_diameter_px <= diameter <= geometry.maximum_ball_diameter_px


def _cluster_candidates(
    candidates: Sequence[InstanceCandidate], geometry: FrameGeometry
) -> tuple[tuple[InstanceCandidate, ...], ...]:
    clusters: list[list[InstanceCandidate]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.asset_id,
            item.frame_index,
            item.class_name,
            item.center_px,
            item.teacher_id,
        ),
    ):
        matching = next(
            (
                cluster
                for cluster in clusters
                if cluster[0].asset_id == candidate.asset_id
                and cluster[0].frame_index == candidate.frame_index
                and cluster[0].class_name == candidate.class_name
                and _distance(cluster[0].center_px, candidate.center_px)
                <= geometry.fusion_center_distance_px
            ),
            None,
        )
        if matching is None:
            clusters.append([candidate])
        else:
            matching.append(candidate)
    return tuple(tuple(cluster) for cluster in clusters)


def _stable_label_id(evidence: tuple[InstanceCandidate, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in evidence]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fuse_candidates(
    candidates: Sequence[InstanceCandidate],
    geometry: FrameGeometry,
    motion: MotionContext,
) -> tuple[PseudoLabel, ...]:
    """Reject impossible detections and fuse nearby teacher evidence as pseudo labels."""

    valid = tuple(candidate for candidate in candidates if _is_valid(candidate, geometry))
    labels: list[PseudoLabel] = []
    table_left, table_top, table_right, table_bottom = geometry.table_bounds_px
    for cluster in _cluster_candidates(valid, geometry):
        evidence = tuple(sorted(cluster, key=lambda item: item.teacher_id))
        count = len(evidence)
        center = (
            round(sum(item.center_px[0] for item in evidence) / count, 6),
            round(sum(item.center_px[1] for item in evidence) / count, 6),
        )
        box = tuple(
            round(sum(item.box_xyxy_px[index] for item in evidence) / count, 6)
            for index in range(4)
        )
        teacher_ids = tuple(sorted({item.teacher_id for item in evidence}))
        edge_margin = geometry.maximum_ball_diameter_px
        near_edge = (
            center[0] - table_left <= edge_margin
            or table_right - center[0] <= edge_margin
            or center[1] - table_top <= edge_margin
            or table_bottom - center[1] <= edge_margin
        )
        identities = {identity for item in evidence for identity in item.identity_candidates}
        labels.append(
            PseudoLabel(
                label_id=_stable_label_id(evidence),
                asset_id=evidence[0].asset_id,
                frame_index=evidence[0].frame_index,
                timestamp_ms=evidence[0].timestamp_ms,
                class_name=evidence[0].class_name,
                center_px=center,
                box_xyxy_px=box,  # type: ignore[arg-type]
                confidence=round(sum(item.confidence for item in evidence) / count, 6),
                teacher_ids=teacher_ids,
                teacher_evidence=evidence,
                teacher_agreement=len(teacher_ids) >= 2,
                occluded=any(item.occluded for item in evidence),
                blur_score=motion.blur_score,
                speed_px_per_s=motion.speed_px_per_s,
                near_table_edge=near_edge,
                identity_ambiguous=len(identities) > 1,
            )
        )
    return tuple(sorted(labels, key=lambda item: (item.asset_id, item.frame_index, item.label_id)))


def _review_reason(label: PseudoLabel, strata: ReviewStrata) -> tuple[ReviewReason, int]:
    if not label.teacher_agreement and not label.occluded:
        return "teacher_disagreement", 0
    if label.occluded:
        return "occlusion", 1
    if label.confidence < strata.low_confidence_below:
        return "low_confidence", 2
    if label.blur_score >= strata.blur_at_least:
        return "blur", 3
    if label.near_table_edge:
        return "rail_or_pocket_proximity", 4
    if label.speed_px_per_s >= strata.high_speed_at_least_px_per_s:
        return "high_speed", 5
    if label.identity_ambiguous:
        return "identity_orientation", 6
    return "easy_sample", 7


def select_review_items(
    labels: Sequence[PseudoLabel], *, limit: int, strata: ReviewStrata
) -> tuple[ReviewItem, ...]:
    """Rank each label once, prioritising hard cases before consensus samples."""

    if limit < 0:
        raise ValueError("review limit must be non-negative")
    by_id: dict[str, PseudoLabel] = {}
    for label in labels:
        by_id.setdefault(label.label_id, label)
    buckets: dict[int, list[ReviewItem]] = defaultdict(list)
    for label in by_id.values():
        reason, priority = _review_reason(label, strata)
        buckets[priority].append(
            ReviewItem(
                label_id=label.label_id,
                asset_id=label.asset_id,
                frame_index=label.frame_index,
                timestamp_ms=label.timestamp_ms,
                reason=reason,
                priority=priority,
            )
        )
    ordered_buckets = {
        priority: sorted(
            items, key=lambda value: (value.asset_id, value.frame_index, value.label_id)
        )
        for priority, items in buckets.items()
    }
    ordered = [ordered_buckets[priority][0] for priority in sorted(ordered_buckets)]
    ordered.extend(
        item for priority in sorted(ordered_buckets) for item in ordered_buckets[priority][1:]
    )
    return tuple(ordered[:limit])
