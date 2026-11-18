"""Truthful V0 capability registry and research debt boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    capability_id: str
    status: Literal["supported", "deferred"]
    fallback: str
    research_branch: str | None = None

    def __post_init__(self) -> None:
        if not self.capability_id or not self.fallback:
            raise ValueError("capability identity and fallback must not be empty")


def capability_registry() -> tuple[CapabilityRecord, ...]:
    """Return the deterministic product capability matrix for this V0."""

    records = (
        CapabilityRecord(
            capability_id="stable_table_route",
            status="supported",
            fallback="abstain_when_stationary_snapshot_is_not_plannable",
        ),
        CapabilityRecord(
            capability_id="ML1-TRACK-001",
            status="deferred",
            fallback="stationary_snapshot_reinitialization",
            research_branch="codex/global-fixed-id-research",
        ),
        CapabilityRecord(
            capability_id="ML1-EVENT-002",
            status="deferred",
            fallback="manual_exception_confirmation",
            research_branch="codex/global-fixed-id-research",
        ),
        CapabilityRecord(
            capability_id="ML1-RULE-003",
            status="deferred",
            fallback="manual_exception_confirmation",
            research_branch="codex/global-fixed-id-research",
        ),
        CapabilityRecord(
            capability_id="GUIDANCE-POWER-001",
            status="deferred",
            fallback="generic_five_band_power",
        ),
        CapabilityRecord(
            capability_id="MEDIA-HIGHLIGHT-001",
            status="deferred",
            fallback="user_favorites_only",
        ),
    )
    return tuple(sorted(records, key=lambda item: item.capability_id))
