from __future__ import annotations

import pytest

from billiards_ai.edge.route_assistance.capabilities import capability_registry
from billiards_ai.edge.route_assistance.models import (
    AssistanceSnapshot,
    GuidanceDisplay,
    PowerDisplay,
    SceneObservation,
)


def test_v0_never_claims_global_identity_or_rule_adjudication() -> None:
    records = {item.capability_id: item for item in capability_registry()}

    assert records["stable_table_route"].status == "supported"
    assert records["ML1-TRACK-001"].status == "deferred"
    assert records["ML1-RULE-003"].status == "deferred"
    assert records["ML1-TRACK-001"].fallback == "stationary_snapshot_reinitialization"
    assert records["ML1-TRACK-001"].research_branch == "codex/global-fixed-id-research"


def test_capability_registry_is_deterministic_and_complete() -> None:
    records = capability_registry()

    assert tuple(item.capability_id for item in records) == tuple(
        sorted(item.capability_id for item in records)
    )
    assert {item.capability_id for item in records} == {
        "GUIDANCE-POWER-001",
        "MEDIA-HIGHLIGHT-001",
        "ML1-EVENT-002",
        "ML1-RULE-003",
        "ML1-TRACK-001",
        "stable_table_route",
    }


def test_scene_observation_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SceneObservation(frame_id=0, timestamp_ms=-1, centers_mm=())


def test_power_display_is_explicitly_generic() -> None:
    display = PowerDisplay(band="medium_light", approximate_percent=35)

    assert display.personalized is False


def test_snapshot_rejects_stale_guidance_generation() -> None:
    guidance = GuidanceDisplay(
        recommendation_id="recommendation-1",
        candidate_id="candidate-1",
        scene_generation=1,
        power=PowerDisplay(band="medium", approximate_percent=50),
    )

    with pytest.raises(ValueError, match="scene generation"):
        AssistanceSnapshot(scene_generation=2, guidance=guidance)
