"""Deterministic temporal fusion for per-frame hierarchical identity evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from billiards_ai.lab.adapters.identity import identity_group


class FusionStatus(StrEnum):
    RESOLVED = "resolved"
    GROUP_ONLY = "group_only"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IdentityFrameEvidence:
    frame_id: int
    track_id: str
    exact_identity: str | None
    group: str
    confidence: float
    source_ref: str

    def __post_init__(self) -> None:
        if self.frame_id < 0 or not self.track_id or not self.source_ref:
            raise ValueError("identity evidence identifiers are invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("identity evidence confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    track_id: str
    identity: str | None
    group: str
    confidence: float
    status: FusionStatus
    supporting_frame_ids: tuple[int, ...]
    source_refs: tuple[str, ...]


class FrameIdentityAdapter(Protocol):
    def resolve_identities(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]: ...


def _rank_votes(votes: dict[str, float]) -> tuple[str, float, float, float]:
    if not votes:
        return "unknown", 0.0, 0.0, 0.0
    ordered = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    winner, weight = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    total = sum(votes.values())
    return winner, weight / total, (weight - runner_up) / total, total


def _resolve_track(
    track_id: str,
    frames: Sequence[IdentityFrameEvidence],
    *,
    resolution_threshold: float,
    conflict_margin: float,
) -> TrackIdentity:
    group_votes: dict[str, float] = defaultdict(float)
    exact_votes: dict[str, float] = defaultdict(float)
    for item in frames:
        if item.group != "unknown":
            group_votes[item.group] += item.confidence
        if item.exact_identity not in {None, "unknown"}:
            exact_votes[item.exact_identity] += item.confidence

    group, group_posterior, group_margin, _ = _rank_votes(group_votes)
    exact, exact_posterior, exact_margin, _ = _rank_votes(exact_votes)
    group_resolved = (
        group != "unknown"
        and group_posterior >= resolution_threshold
        and group_margin >= conflict_margin
    )
    exact_resolved = (
        exact != "unknown"
        and exact_posterior >= resolution_threshold
        and exact_margin >= conflict_margin
    )

    if exact_resolved and group_resolved and identity_group(exact) == group:
        status = FusionStatus.RESOLVED
        confidence = min(exact_posterior, group_posterior)
    elif not exact_votes and group_resolved:
        exact = None
        status = FusionStatus.GROUP_ONLY
        confidence = group_posterior
    elif group_resolved and not exact_resolved:
        exact = None
        status = FusionStatus.GROUP_ONLY
        confidence = group_posterior
    elif not exact_votes and not group_votes:
        exact = None
        group = "unknown"
        status = FusionStatus.UNKNOWN
        confidence = 0.0
    else:
        exact = None
        group = "unknown"
        status = FusionStatus.AMBIGUOUS
        confidence = max(exact_posterior, group_posterior)

    ordered = sorted(frames, key=lambda item: (item.frame_id, item.source_ref))
    return TrackIdentity(
        track_id=track_id,
        identity=exact,
        group=group,
        confidence=confidence,
        status=status,
        supporting_frame_ids=tuple(item.frame_id for item in ordered),
        source_refs=tuple(item.source_ref for item in ordered),
    )


def fuse_track_identities(
    evidence: Sequence[IdentityFrameEvidence],
    *,
    resolution_threshold: float = 0.70,
    conflict_margin: float = 0.15,
) -> dict[str, TrackIdentity]:
    if not 0 <= resolution_threshold <= 1 or not 0 <= conflict_margin <= 1:
        raise ValueError("fusion thresholds must be between zero and one")
    grouped: dict[str, list[IdentityFrameEvidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.track_id].append(item)
    return {
        track_id: _resolve_track(
            track_id,
            grouped[track_id],
            resolution_threshold=resolution_threshold,
            conflict_margin=conflict_margin,
        )
        for track_id in sorted(grouped)
    }


def classify_and_fuse_track_identities(
    *,
    adapter: FrameIdentityAdapter,
    observations: Sequence[Mapping[str, Any]],
    resolution_threshold: float = 0.70,
    conflict_margin: float = 0.15,
) -> dict[str, TrackIdentity]:
    """Classify per-frame crops with an injected eye, then fuse identities by track."""

    resolved = adapter.resolve_identities(observations)
    if len(resolved) != len(observations):
        raise ValueError("identity adapter changed the observation count")
    evidence = []
    for observation, classification in zip(observations, resolved, strict=True):
        try:
            track_id = str(observation["track_id"])
            if classification.get("track_id") != observation["track_id"]:
                raise ValueError("identity adapter changed the track ID")
            evidence.append(
                IdentityFrameEvidence(
                    frame_id=int(observation["frame_id"]),
                    track_id=track_id,
                    exact_identity=(
                        str(classification["exact_identity"])
                        if classification["exact_identity"] is not None
                        else None
                    ),
                    group=str(classification["group"]),
                    confidence=float(classification["confidence"]),
                    source_ref=str(observation["source_ref"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("identity classification evidence is invalid") from exc
    return fuse_track_identities(
        evidence,
        resolution_threshold=resolution_threshold,
        conflict_margin=conflict_margin,
    )
