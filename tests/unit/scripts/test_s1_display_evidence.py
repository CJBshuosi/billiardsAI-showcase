import copy

import pytest

from scripts.development import s1_screen_acceptance as screen


def report():
    first = dict(
        cue_path=[[1, 2], [3, 4]], object_path=[[3, 4], [5, 6]], region=[[2, 2], [4, 2], [3, 4]]
    )
    following = dict(cue_path=[[2, 3], [4, 5]], object_path=[[4, 5], [6, 7]], region=None)

    def observed(at, *, feedback=False, overlay=first):
        return dict(
            monotonic=at,
            observed_at=at + 0.02,
            phase="已进入建议区" if feedback else "系统自动推荐",
            has_halo=not feedback,
            has_dash=not feedback,
            feedback_motion_frame_id=10 if feedback else None,
            observed_frame_id=20 if feedback or overlay is following else 0,
            route_lines=[]
            if feedback
            else [" ".join(f"{x},{y}" for x, y in overlay[k]) for k in ("cue_path", "object_path")],
            region_points="2,2 4,2 3,4" if feedback or overlay["region"] else None,
        )

    return dict(
        recommendations={"0": dict(overlay=first), "20": dict(overlay=following)},
        feedbacks={
            "10": dict(
                overlay=first,
                review=dict(region_result="inside", observed_frame_id=20, motion_frame_id=10),
            )
        },
        dom=[
            observed(10.18),
            observed(10.2, feedback=True),
            observed(10.7, feedback=True),
            observed(11.18, feedback=True),
            observed(11.22, overlay=following),
        ],
    )


def test_real_display_requires_both_selected_paths_and_one_second_feedback_handoff():
    result = screen.assess_display_flow(report())
    assert result["selected_pocket_routes"] == 2
    assert result["feedback"][0]["visible_seconds_bounds"][0] < 1
    assert result["feedback"][0]["visible_seconds_bounds"][1] > 1


def with_bounds(lower, upper):
    value = report()
    value["dom"][0].update(monotonic=0.0, observed_at=0.02)
    value["dom"][1].update(monotonic=0.05, observed_at=0.1)
    value["dom"][2].update(monotonic=0.2, observed_at=0.22)
    value["dom"][3].update(monotonic=0.1 + lower, observed_at=0.11 + lower)
    value["dom"][4].update(monotonic=upper - 0.01, observed_at=upper)
    return value


def test_precise_actual_about_one_second_bounds_do_not_have_to_contain_exact_one():
    bounds = [1.129328001383692, 1.261671998985121]
    result = screen.assess_display_flow(with_bounds(*bounds))
    assert result["feedback"][0]["visible_seconds_bounds"] == pytest.approx(bounds)


@pytest.mark.parametrize("bounds", [(0.7, 0.9), (1.1, 1.3), (0.9, 1.15)])
def test_about_one_second_policy_accepts_the_whole_precise_boundary_interval(bounds):
    assert screen.assess_display_flow(with_bounds(*bounds))["feedback"]


@pytest.mark.parametrize(
    "bounds",
    [
        (0.4, 0.6),
        (1.5, 1.7),
        (0.65, 0.85),  # An acceptable midpoint cannot rescue an outside bound.
        (1.15, 1.35),
        (0.85, 1.15),  # Entirely near one second, but too uncertain.
        (1.2, 1.1),
        (float("nan"), 1.2),
        (0.9, float("nan")),
        (float("inf"), 1.2),
        (0.9, float("inf")),
    ],
)
def test_about_one_second_policy_rejects_short_long_uncertain_or_invalid_intervals(bounds):
    with pytest.raises(ValueError, match="one-second"):
        screen.assess_display_flow(with_bounds(*bounds))


@pytest.mark.parametrize(
    "fault", ["missing_pocket", "wrong_pocket", "old_route", "wrong_region", "blink"]
)
def test_missing_or_misbound_visible_advice_cannot_be_accepted(fault):
    value = copy.deepcopy(report())
    if fault == "missing_pocket":
        value["dom"][0]["route_lines"].pop()
    elif fault == "wrong_pocket":
        value["dom"][0]["route_lines"][1] = "3,4 500,600"
    elif fault == "old_route":
        value["dom"][2]["route_lines"] = ["1,2 3,4"]
    elif fault == "wrong_region":
        value["dom"][2]["region_points"] = "200,200 400,200 300,400"
    else:
        for index, row in enumerate(value["dom"]):
            row["monotonic"] = 10 + index * 0.05
            row["observed_at"] = row["monotonic"] + 0.02
    with pytest.raises(ValueError, match="display|feedback|pocket"):
        screen.assess_display_flow(value)


def test_handoff_cannot_show_old_route_before_correct_next_route_appears_later():
    value = report()
    old = copy.deepcopy(value["dom"][0])
    old.update(monotonic=11.2, observed_at=11.22)
    value["dom"].insert(-1, old)
    with pytest.raises(ValueError, match="handoff"):
        screen.assess_display_flow(value)


def test_two_separate_inside_feedbacks_are_bound_to_their_own_shots():
    value = report()
    region = value["recommendations"]["0"]["overlay"]["region"]
    second_overlay = copy.deepcopy(value["recommendations"]["20"]["overlay"])
    second_overlay["region"] = region
    value["feedbacks"]["30"] = dict(
        overlay=second_overlay,
        review=dict(
            region_result="inside",
            observed_frame_id=40,
            motion_frame_id=30,
        ),
    )
    value["recommendations"]["40"] = copy.deepcopy(value["recommendations"]["0"])
    preceding = copy.deepcopy(value["dom"][-1])
    preceding.update(monotonic=12.18, observed_at=12.20)
    value["dom"].append(preceding)
    for at in [12.2, 12.7, 13.18]:
        row = copy.deepcopy(value["dom"][1])
        row.update(
            monotonic=at, observed_at=at + 0.02, feedback_motion_frame_id=30, observed_frame_id=40
        )
        value["dom"].append(row)
    following = copy.deepcopy(value["dom"][0])
    following.update(monotonic=13.22, observed_at=13.24, observed_frame_id=40)
    value["dom"].append(following)
    result = screen.assess_display_flow(value)
    assert len(result["feedback"]) == 2


@pytest.mark.parametrize("frame_id", [20, 99])
def test_abstention_handoff_is_also_bound_to_the_post_shot_snapshot(frame_id):
    value = report()
    value["recommendations"].pop("20")
    value["dom"][-1].update(
        observed_frame_id=frame_id,
        phase="暂无可靠方案",
        route_lines=[],
        has_halo=False,
        has_region=False,
    )
    if frame_id == 20:
        assert screen.assess_display_flow(value)["feedback"]
    else:
        with pytest.raises(ValueError, match="handoff"):
            screen.assess_display_flow(value)
