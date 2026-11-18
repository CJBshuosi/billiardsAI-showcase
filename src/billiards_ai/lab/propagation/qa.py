"""Pure QA classification for propagated mask observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from billiards_ai.lab.propagation.contracts import MaskObservation, PilotQaConfig


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def evaluate_observations(
    observations: Sequence[MaskObservation],
    *,
    frame_width_px: int,
    frame_height_px: int,
    max_center_jump_px: float | None = None,
    max_area_ratio: float | None = None,
    prompt_radii_px: Mapping[tuple[str, str, int], float] | None = None,
    qa_config: PilotQaConfig | None = None,
) -> tuple[MaskObservation, ...]:
    if frame_width_px <= 0 or frame_height_px <= 0:
        raise ValueError("frame dimensions must be positive")
    normalized = prompt_radii_px is not None or qa_config is not None
    if normalized and (prompt_radii_px is None or qa_config is None):
        raise ValueError("normalized QA requires prompt radii and the frozen QA configuration")
    if not normalized and (max_center_jump_px is None or max_area_ratio is None):
        raise ValueError("pixel QA requires explicit center-jump and area thresholds")
    if max_center_jump_px is not None and max_center_jump_px <= 0:
        raise ValueError("maximum center jump must be positive")
    if max_area_ratio is not None and max_area_ratio <= 1:
        raise ValueError("maximum area ratio must exceed one")
    if prompt_radii_px is not None and any(radius <= 0 for radius in prompt_radii_px.values()):
        raise ValueError("prompt radii must be positive")

    ordered = tuple(
        sorted(
            observations,
            key=lambda item: (item.asset_id, item.instance_id, item.frame_index),
        )
    )
    previous_by_instance: dict[tuple[str, str], MaskObservation] = {}
    evaluated: list[MaskObservation] = []
    for observation in ordered:
        flags = set(observation.qa_flags)
        left, top, right, bottom = observation.box_xyxy_px
        center_x, center_y = observation.center_px
        if (
            left < 0
            or top < 0
            or right > frame_width_px
            or bottom > frame_height_px
            or not 0 <= center_x <= frame_width_px
            or not 0 <= center_y <= frame_height_px
        ):
            flags.add("outside_frame")

        key = (observation.asset_id, observation.instance_id)
        previous = previous_by_instance.get(key)
        if previous is not None:
            if normalized:
                assert prompt_radii_px is not None
                assert qa_config is not None
                radius = prompt_radii_px.get((*key, observation.prompt_frame_index))
                if radius is None:
                    raise ValueError("every normalized observation requires a prompt radius")
                source_frame_gap = observation.frame_index - previous.frame_index
                center_jump_limit = (
                    qa_config.maximum_center_jump_radii_per_source_frame * radius * source_frame_gap
                )
                area_limit = qa_config.maximum_area_ratio
            else:
                assert max_center_jump_px is not None
                assert max_area_ratio is not None
                center_jump_limit = max_center_jump_px
                area_limit = max_area_ratio
            if _distance(previous.center_px, observation.center_px) > center_jump_limit:
                flags.add("center_jump")
            area_ratio = max(previous.area_px, observation.area_px) / min(
                previous.area_px, observation.area_px
            )
            if area_ratio > area_limit:
                flags.add("area_jump")

        updated = observation.model_copy(update={"qa_flags": tuple(sorted(flags))})
        evaluated.append(updated)
        previous_by_instance[key] = observation
    return tuple(evaluated)
