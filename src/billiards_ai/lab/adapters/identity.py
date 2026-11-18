"""Hierarchical ball identity inference with explicit unknown outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class IdentityBackend(Protocol):
    version: str

    def predict(self, crop: object) -> Mapping[str, Mapping[str, float]]: ...


def _rank(scores: Mapping[str, float]) -> tuple[str, float, float]:
    if not scores:
        raise ValueError("identity backend returned no scores")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    label, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return label, confidence, confidence - runner_up


def identity_group(identity: str) -> str:
    if identity == "cue":
        return "cue"
    if identity == "8":
        return "eight"
    if identity.isdigit() and 1 <= int(identity) <= 7:
        return "solid"
    if identity.isdigit() and 9 <= int(identity) <= 15:
        return "stripe"
    return "unknown"


class HierarchicalIdentityAdapter:
    def __init__(
        self,
        backend: IdentityBackend,
        *,
        exact_confidence_min: float = 0.80,
        exact_margin_min: float = 0.20,
        group_confidence_min: float = 0.80,
    ) -> None:
        self._backend = backend
        self.exact_confidence_min = exact_confidence_min
        self.exact_margin_min = exact_margin_min
        self.group_confidence_min = group_confidence_min

    def resolve_identities(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        results = []
        for observation in observations:
            scores = self._backend.predict(observation["crop"])
            exact, confidence, margin = _rank(scores["exact_scores"])
            group, group_confidence, _ = _rank(scores["group_scores"])
            visibility, _, _ = _rank(scores["visibility_scores"])
            unknown_reason = None
            exact_unresolved = (
                exact == "unknown"
                or confidence < self.exact_confidence_min
                or margin < self.exact_margin_min
            )
            if group_confidence < self.group_confidence_min or group == "unknown":
                unknown_reason = (
                    "low_exact_confidence"
                    if confidence < self.exact_confidence_min
                    else "low_group_confidence"
                )
                exact = None
                group = "unknown"
            elif exact_unresolved:
                unknown_reason = "exact_unresolved"
                exact = None
            elif identity_group(exact) != group:
                unknown_reason = "hierarchy_conflict"
                exact = None
            results.append(
                {
                    "track_id": observation["track_id"],
                    "exact_identity": exact,
                    "group": group,
                    "visibility": visibility,
                    "confidence": confidence,
                    "exact_confidence": confidence,
                    "group_confidence": group_confidence,
                    "unknown_reason": unknown_reason,
                    "forced_label": False,
                    "identity_backend_version": self._backend.version,
                }
            )
        return tuple(results)
