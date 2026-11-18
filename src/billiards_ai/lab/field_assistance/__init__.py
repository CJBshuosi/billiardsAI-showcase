"""Local adapters for the field route-assistance domain."""

from billiards_ai.lab.field_assistance.guidance import (
    FieldGuidanceResult,
    build_field_guidance,
    generic_power_display,
)
from billiards_ai.lab.field_assistance.live_state import (
    ReplayClock,
    ReplayProjection,
    ReplayTimeline,
    ReplayTimelineFrame,
    load_replay_timeline,
)
from billiards_ai.lab.field_assistance.observations import (
    build_snapshot_table_state,
    load_cached_observations,
    scene_observation,
)
from billiards_ai.lab.field_assistance.pages import render_screen_page, render_tablet_page

_REPLAY_EXPORTS = frozenset(
    {
        "FieldReplayArtifact",
        "FieldReplayConfig",
        "run_field_route_replay",
        "verify_field_route_replay_artifact",
    }
)


def __getattr__(name: str):
    if name in _REPLAY_EXPORTS:
        from billiards_ai.lab.field_assistance import replay

        return getattr(replay, name)
    raise AttributeError(name)


__all__ = [
    "FieldGuidanceResult",
    "FieldReplayArtifact",
    "FieldReplayConfig",
    "ReplayClock",
    "ReplayProjection",
    "ReplayTimeline",
    "ReplayTimelineFrame",
    "build_field_guidance",
    "build_snapshot_table_state",
    "generic_power_display",
    "load_cached_observations",
    "load_replay_timeline",
    "run_field_route_replay",
    "render_screen_page",
    "render_tablet_page",
    "scene_observation",
    "verify_field_route_replay_artifact",
]
