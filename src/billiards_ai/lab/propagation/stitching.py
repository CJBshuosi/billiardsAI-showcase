"""Deterministic, abstention-aware matching across window overlaps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr
from billiards_ai.lab.propagation.contracts import MaskObservation, PilotQaConfig


class TrackLink(FrozenContract):
    left_instance_id: str | None
    right_instance_id: NonEmptyStr
    status: Literal["linked", "abstained"]
    reason: Literal["matched", "ambiguous", "no_candidate", "no_overlap", "left_already_linked"]
    overlap_frames: tuple[int, ...]
    mean_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_center_distance_px: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def status_and_identity_are_consistent(self) -> TrackLink:
        if self.status == "linked" and (self.left_instance_id is None or self.reason != "matched"):
            raise ValueError("linked tracks require a left instance and matched reason")
        if self.status == "abstained" and self.left_instance_id is not None:
            raise ValueError("abstained links cannot claim a left instance")
        return self


@dataclass(frozen=True)
class _Candidate:
    left_instance_id: str
    right_instance_id: str
    overlap_frames: tuple[int, ...]
    mean_iou: float
    mean_distance: float


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _by_instance(
    observations: Sequence[MaskObservation],
) -> dict[str, dict[int, MaskObservation]]:
    grouped: dict[str, dict[int, MaskObservation]] = defaultdict(dict)
    for observation in observations:
        grouped[observation.instance_id][observation.frame_index] = observation
    return dict(grouped)


def stitch_overlap(
    left: Sequence[MaskObservation],
    right: Sequence[MaskObservation],
    *,
    successor_start_frame: int,
    minimum_iou: float | None = None,
    maximum_center_distance_px: float | None = None,
    prompt_radii_px: Mapping[tuple[str, str, int], float] | None = None,
    qa_config: PilotQaConfig | None = None,
) -> tuple[TrackLink, ...]:
    normalized = prompt_radii_px is not None or qa_config is not None
    if successor_start_frame < 0:
        raise ValueError("successor start frame must be non-negative")
    if normalized and (prompt_radii_px is None or qa_config is None):
        raise ValueError(
            "normalized stitching requires prompt radii and the frozen QA configuration"
        )
    if not normalized and (minimum_iou is None or maximum_center_distance_px is None):
        raise ValueError("pixel stitching requires explicit IoU and center thresholds")
    if minimum_iou is not None and not 0 <= minimum_iou <= 1:
        raise ValueError("minimum IoU must be between zero and one")
    if maximum_center_distance_px is not None and maximum_center_distance_px < 0:
        raise ValueError("maximum center distance must be non-negative")
    if prompt_radii_px is not None and any(radius <= 0 for radius in prompt_radii_px.values()):
        raise ValueError("prompt radii must be positive")

    left_by_instance = _by_instance(left)
    right_by_instance = _by_instance(right)
    left_frames = {item.frame_index for item in left}
    right_frames = {item.frame_index for item in right}
    shared_window_frames = tuple(sorted((left_frames & right_frames) - {successor_start_frame}))
    if not shared_window_frames:
        return tuple(
            TrackLink(
                left_instance_id=None,
                right_instance_id=right_id,
                status="abstained",
                reason="no_overlap",
                overlap_frames=(),
            )
            for right_id in sorted(right_by_instance)
        )

    proposals: dict[str, _Candidate] = {}
    resolved: dict[str, TrackLink] = {}
    for right_id, right_frames_by_index in sorted(right_by_instance.items()):
        candidates: list[_Candidate] = []
        if normalized:
            assert prompt_radii_px is not None
            assert qa_config is not None
            right_sample = next(iter(right_frames_by_index.values()))
            radius = prompt_radii_px.get(
                (right_sample.asset_id, right_id, right_sample.prompt_frame_index)
            )
            if radius is None:
                raise ValueError("every normalized successor instance requires a prompt radius")
            required_iou = qa_config.minimum_overlap_iou
            distance_limit = qa_config.maximum_overlap_center_distance_radii * radius
        else:
            assert minimum_iou is not None
            assert maximum_center_distance_px is not None
            required_iou = minimum_iou
            distance_limit = maximum_center_distance_px
        right_sample = next(iter(right_frames_by_index.values()))
        for left_id, left_frames_by_index in sorted(left_by_instance.items()):
            left_sample = next(iter(left_frames_by_index.values()))
            if (
                left_sample.asset_id != right_sample.asset_id
                or left_sample.class_name != right_sample.class_name
            ):
                continue
            overlap_frames = tuple(
                sorted(
                    (set(left_frames_by_index) & set(right_frames_by_index))
                    - {successor_start_frame}
                )
            )
            if not overlap_frames:
                continue
            ious = [
                _box_iou(
                    left_frames_by_index[frame].box_xyxy_px,
                    right_frames_by_index[frame].box_xyxy_px,
                )
                for frame in overlap_frames
            ]
            distances = [
                _distance(
                    left_frames_by_index[frame].center_px,
                    right_frames_by_index[frame].center_px,
                )
                for frame in overlap_frames
            ]
            mean_iou = sum(ious) / len(ious)
            mean_distance = sum(distances) / len(distances)
            if mean_iou >= required_iou and mean_distance <= distance_limit:
                candidates.append(
                    _Candidate(
                        left_instance_id=left_id,
                        right_instance_id=right_id,
                        overlap_frames=overlap_frames,
                        mean_iou=mean_iou,
                        mean_distance=mean_distance,
                    )
                )
        candidates.sort(
            key=lambda item: (-item.mean_iou, item.mean_distance, item.left_instance_id)
        )
        if not candidates:
            resolved[right_id] = TrackLink(
                left_instance_id=None,
                right_instance_id=right_id,
                status="abstained",
                reason="no_candidate",
                overlap_frames=shared_window_frames,
            )
            continue
        best = candidates[0]
        ambiguous = len(candidates) > 1 and (
            (
                abs(candidates[1].mean_iou - best.mean_iou) < 1e-12
                and abs(candidates[1].mean_distance - best.mean_distance) < 1e-12
            )
            or (
                normalized
                and qa_config is not None
                and radius is not None
                and best.mean_iou - candidates[1].mean_iou <= qa_config.ambiguity_iou_margin
                and candidates[1].mean_distance - best.mean_distance
                <= qa_config.ambiguity_center_distance_margin_radii * radius
            )
        )
        if ambiguous:
            resolved[right_id] = TrackLink(
                left_instance_id=None,
                right_instance_id=right_id,
                status="abstained",
                reason="ambiguous",
                overlap_frames=best.overlap_frames,
                mean_iou=best.mean_iou,
                mean_center_distance_px=best.mean_distance,
            )
            continue
        proposals[right_id] = best

    used_left: set[str] = set()
    for right_id, candidate in sorted(
        proposals.items(),
        key=lambda item: (
            -item[1].mean_iou,
            item[1].mean_distance,
            item[0],
            item[1].left_instance_id,
        ),
    ):
        if candidate.left_instance_id in used_left:
            resolved[right_id] = TrackLink(
                left_instance_id=None,
                right_instance_id=right_id,
                status="abstained",
                reason="left_already_linked",
                overlap_frames=candidate.overlap_frames,
                mean_iou=candidate.mean_iou,
                mean_center_distance_px=candidate.mean_distance,
            )
            continue
        used_left.add(candidate.left_instance_id)
        resolved[right_id] = TrackLink(
            left_instance_id=candidate.left_instance_id,
            right_instance_id=right_id,
            status="linked",
            reason="matched",
            overlap_frames=candidate.overlap_frames,
            mean_iou=candidate.mean_iou,
            mean_center_distance_px=candidate.mean_distance,
        )
    return tuple(resolved[right_id] for right_id in sorted(resolved))
