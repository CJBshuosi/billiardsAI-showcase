from __future__ import annotations

from billiards_ai.lab.propagation import MaskObservation, PilotQaConfig
from billiards_ai.lab.propagation.qa import evaluate_observations


def _observation(
    frame_index: int,
    *,
    instance_id: str = "ball-1",
    center_px: tuple[float, float] = (50.0, 50.0),
    box_xyxy_px: tuple[float, float, float, float] = (45.0, 45.0, 55.0, 55.0),
    area_px: int = 80,
    prompt_frame_index: int = 0,
) -> MaskObservation:
    return MaskObservation(
        asset_id="video-1",
        frame_index=frame_index,
        instance_id=instance_id,
        class_name="ball",
        prompt_frame_index=prompt_frame_index,
        center_px=center_px,
        box_xyxy_px=box_xyxy_px,
        area_px=area_px,
        mask_ref=f"masks/{instance_id}/{frame_index:06d}.png",
    )


def _evaluate(observations: tuple[MaskObservation, ...]) -> tuple[MaskObservation, ...]:
    return evaluate_observations(
        observations,
        frame_width_px=100,
        frame_height_px=100,
        max_center_jump_px=20.0,
        max_area_ratio=2.0,
    )


def test_qa_leaves_valid_continuous_observations_unflagged() -> None:
    first = _observation(0)
    second = _observation(
        1,
        center_px=(55.0, 50.0),
        box_xyxy_px=(50.0, 45.0, 60.0, 55.0),
    )

    evaluated = _evaluate((second, first))

    assert [(item.frame_index, item.qa_flags) for item in evaluated] == [(0, ()), (1, ())]


def test_qa_flags_observations_outside_the_frame() -> None:
    outside = _observation(
        0,
        center_px=(98.0, 50.0),
        box_xyxy_px=(93.0, 45.0, 103.0, 55.0),
    )

    assert _evaluate((outside,))[0].qa_flags == ("outside_frame",)


def test_qa_flags_only_the_later_area_and_center_jump() -> None:
    first = _observation(0, area_px=80)
    jump = _observation(
        1,
        center_px=(90.0, 90.0),
        box_xyxy_px=(80.0, 80.0, 100.0, 100.0),
        area_px=240,
    )

    evaluated = _evaluate((first, jump))

    assert evaluated[0].qa_flags == ()
    assert evaluated[1].qa_flags == ("area_jump", "center_jump")


def test_qa_does_not_compare_different_instances() -> None:
    first = _observation(0, instance_id="ball-a", center_px=(10.0, 10.0))
    other = _observation(1, instance_id="ball-b", center_px=(90.0, 90.0))

    assert [item.qa_flags for item in _evaluate((first, other))] == [(), ()]


def test_qa_derives_each_center_jump_limit_from_its_prompt_radius_and_frame_gap() -> None:
    first = _observation(0, center_px=(10.0, 10.0))
    later = _observation(2, center_px=(85.0, 10.0))

    evaluated = evaluate_observations(
        (first, later),
        frame_width_px=100,
        frame_height_px=100,
        prompt_radii_px={("video-1", "ball-1", 0): 10.0},
        qa_config=PilotQaConfig(),
    )

    assert [item.qa_flags for item in evaluated] == [(), ()]


def test_qa_uses_each_reprompted_segment_source_frame_radius() -> None:
    initial = _observation(0, center_px=(10.0, 10.0), prompt_frame_index=0)
    initial_next = _observation(1, center_px=(15.0, 10.0), prompt_frame_index=0)
    relayed = _observation(20, center_px=(10.0, 50.0), prompt_frame_index=20)
    relayed_next = _observation(21, center_px=(30.0, 50.0), prompt_frame_index=20)

    evaluated = evaluate_observations(
        (initial, initial_next, relayed, relayed_next),
        frame_width_px=100,
        frame_height_px=100,
        prompt_radii_px={
            ("video-1", "ball-1", 0): 1.0,
            ("video-1", "ball-1", 20): 6.0,
        },
        qa_config=PilotQaConfig(),
    )

    assert [item.qa_flags for item in evaluated] == [(), ("center_jump",), (), ()]
