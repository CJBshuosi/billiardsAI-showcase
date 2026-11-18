from __future__ import annotations

from billiards_ai.lab.propagation import MaskObservation, PilotQaConfig
from billiards_ai.lab.propagation.stitching import stitch_overlap


def _observation(
    frame_index: int,
    instance_id: str,
    *,
    left: float,
    class_name: str = "ball",
    prompt_frame_index: int | None = None,
) -> MaskObservation:
    return MaskObservation(
        asset_id="video-1",
        frame_index=frame_index,
        instance_id=instance_id,
        class_name=class_name,
        prompt_frame_index=frame_index if prompt_frame_index is None else prompt_frame_index,
        center_px=(left + 5.0, 25.0),
        box_xyxy_px=(left, 20.0, left + 10.0, 30.0),
        area_px=80,
        mask_ref=f"masks/{instance_id}/{frame_index:06d}.png",
    )


def _stitch(
    left: tuple[MaskObservation, ...],
    right: tuple[MaskObservation, ...],
):
    return stitch_overlap(
        left,
        right,
        successor_start_frame=0,
        minimum_iou=0.5,
        maximum_center_distance_px=8.0,
    )


def test_stitching_matches_multiple_same_class_instances_one_to_one() -> None:
    left = (
        _observation(8, "left-a", left=10.0),
        _observation(8, "left-b", left=50.0),
        _observation(9, "left-a", left=11.0),
        _observation(9, "left-b", left=51.0),
    )
    right = (
        _observation(8, "right-b", left=50.0),
        _observation(8, "right-a", left=10.0),
        _observation(9, "right-b", left=51.0),
        _observation(9, "right-a", left=11.0),
    )

    links = _stitch(left, right)

    assert [(item.left_instance_id, item.right_instance_id, item.status) for item in links] == [
        ("left-a", "right-a", "linked"),
        ("left-b", "right-b", "linked"),
    ]


def test_stitching_abstains_when_two_left_tracks_are_equally_plausible() -> None:
    left = (
        _observation(8, "left-a", left=10.0),
        _observation(8, "left-b", left=10.0),
    )
    right = (_observation(8, "right-a", left=10.0),)

    links = _stitch(left, right)

    assert len(links) == 1
    assert links[0].status == "abstained"
    assert links[0].reason == "ambiguous"
    assert links[0].left_instance_id is None


def test_stitching_enforces_one_to_one_links() -> None:
    left = (_observation(8, "left-a", left=10.0),)
    right = (
        _observation(8, "right-best", left=10.0),
        _observation(8, "right-weaker", left=11.0),
    )

    links = _stitch(left, right)

    by_right = {item.right_instance_id: item for item in links}
    assert by_right["right-best"].status == "linked"
    assert by_right["right-weaker"].status == "abstained"
    assert by_right["right-weaker"].reason == "left_already_linked"


def test_stitching_abstains_when_windows_have_no_common_frames() -> None:
    left = (_observation(7, "left-a", left=10.0),)
    right = (_observation(8, "right-a", left=10.0),)

    links = _stitch(left, right)

    assert len(links) == 1
    assert links[0].status == "abstained"
    assert links[0].reason == "no_overlap"


def test_stitching_uses_frozen_normalized_ambiguity_margins() -> None:
    left = (
        _observation(8, "left-best", left=10.0),
        _observation(8, "left-near", left=10.2),
    )
    right = (_observation(8, "right-a", left=10.0),)

    links = stitch_overlap(
        left,
        right,
        successor_start_frame=0,
        prompt_radii_px={("video-1", "right-a", 8): 10.0},
        qa_config=PilotQaConfig(),
    )

    assert links[0].status == "abstained"
    assert links[0].reason == "ambiguous"


def test_stitching_excludes_the_successor_relay_frame_from_link_agreement() -> None:
    relay_only_left = (_observation(20, "left-a", left=10.0),)
    relay_only_right = (_observation(20, "right-a", left=10.0),)

    links = stitch_overlap(
        relay_only_left,
        relay_only_right,
        successor_start_frame=20,
        minimum_iou=0.5,
        maximum_center_distance_px=8.0,
    )

    assert links[0].status == "abstained"
    assert links[0].reason == "no_overlap"
    assert links[0].overlap_frames == ()


def test_stitching_scores_only_real_nonrelay_intersection_frames() -> None:
    left = (
        _observation(20, "left-a", left=60.0),
        _observation(21, "left-a", left=10.0),
    )
    right = (
        _observation(20, "right-a", left=10.0),
        _observation(21, "right-a", left=10.0),
    )

    links = stitch_overlap(
        left,
        right,
        successor_start_frame=20,
        minimum_iou=0.5,
        maximum_center_distance_px=8.0,
    )

    assert links[0].status == "linked"
    assert links[0].overlap_frames == (21,)


def test_stitching_uses_the_radius_of_the_reprompted_successor_segment() -> None:
    left = (_observation(21, "left-a", left=10.0),)
    right = (_observation(21, "right-a", left=15.0, prompt_frame_index=20),)

    links = stitch_overlap(
        left,
        right,
        successor_start_frame=20,
        prompt_radii_px={
            ("video-1", "right-a", 0): 1.0,
            ("video-1", "right-a", 20): 25.0,
        },
        qa_config=PilotQaConfig(),
    )

    assert links[0].status == "linked"
