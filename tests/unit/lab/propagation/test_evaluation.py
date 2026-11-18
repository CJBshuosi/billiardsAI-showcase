from __future__ import annotations

import pytest

from billiards_ai.lab.propagation.contracts import MaskObservation
from billiards_ai.lab.propagation.evaluation import (
    ReviewUnit,
    compute_frozen_gates,
    compute_set_metrics,
    freeze_review_sample,
    match_frame,
)
from billiards_ai.lab.propagation.pilot_data import EvaluationCircle


def _circle(
    annotation_id: str,
    x: float,
    *,
    y: float = 0.0,
    radius: float = 10.0,
) -> EvaluationCircle:
    return EvaluationCircle(
        annotation_id=annotation_id,
        video_id="video-1",
        shot_index=0,
        source_frame_index=10,
        source_frame_sha256="a" * 64,
        center_x_px=x,
        center_y_px=y,
        radius_px=radius,
        role="middle",
    )


def _prediction(
    instance_id: str,
    x: float,
    *,
    y: float = 0.0,
    qa_flags: tuple[str, ...] = (),
) -> MaskObservation:
    return MaskObservation(
        asset_id="video-1",
        frame_index=10,
        instance_id=instance_id,
        class_name="ball",
        prompt_frame_index=0,
        center_px=(x, y),
        box_xyxy_px=(x - 2, y - 2, x + 2, y + 2),
        area_px=12,
        mask_ref=f"masks/{instance_id}.png",
        qa_flags=qa_flags,
    )


def test_match_frame_is_minimum_cost_one_to_one_for_multiple_same_class_balls() -> None:
    assignment = match_frame(
        (_circle("ground-a", 0), _circle("ground-b", 12)),
        (_prediction("prediction-a", 3), _prediction("prediction-b", 11)),
    )

    assert [
        (match.ground_truth.annotation_id, match.prediction.instance_id)
        for match in assignment.matches
    ] == [("ground-a", "prediction-a"), ("ground-b", "prediction-b")]
    assert [match.normalized_center_error for match in assignment.matches] == [0.3, 0.1]
    assert assignment.unmatched_ground_truth == ()
    assert assignment.unmatched_predictions == ()


def test_match_frame_resolves_equal_cost_ties_by_stable_identity() -> None:
    assignment = match_frame(
        (_circle("ground-b", 0), _circle("ground-a", 0)),
        (_prediction("prediction-b", 0), _prediction("prediction-a", 0)),
    )

    assert [
        (match.ground_truth.annotation_id, match.prediction.instance_id)
        for match in assignment.matches
    ] == [("ground-a", "prediction-a"), ("ground-b", "prediction-b")]


def test_match_frame_accepts_exact_radius_boundary_and_keeps_misses_and_false_positives() -> None:
    assignment = match_frame(
        (_circle("accepted", 0), _circle("miss", 100)),
        (_prediction("at-boundary", 10), _prediction("false-positive", 111)),
    )

    assert [
        (item.ground_truth.annotation_id, item.normalized_center_error)
        for item in assignment.matches
    ] == [("accepted", 1.0)]
    assert [item.annotation_id for item in assignment.unmatched_ground_truth] == ["miss"]
    assert [item.instance_id for item in assignment.unmatched_predictions] == ["false-positive"]


def test_match_frame_never_matches_a_different_class_and_keeps_it_as_a_false_positive() -> None:
    wrong_class = _prediction("wrong-class", 0).model_copy(update={"class_name": "cue"})

    assignment = match_frame((_circle("ball", 0),), (wrong_class,))

    assert assignment.matches == ()
    assert [item.annotation_id for item in assignment.unmatched_ground_truth] == ["ball"]
    assert [item.instance_id for item in assignment.unmatched_predictions] == ["wrong-class"]


def test_set_metrics_use_full_ground_truth_denominator_and_report_tp_support() -> None:
    first = match_frame(
        (_circle("a", 0), _circle("b", 20)),
        (_prediction("pa", 1), _prediction("fp", 100)),
    )
    failed_or_absent = match_frame((_circle("c", 40),), ())

    metrics = compute_set_metrics((first, failed_or_absent))

    assert metrics.ground_truth_count == 3
    assert metrics.true_positive_support == 1
    assert metrics.false_positive_count == 1
    assert metrics.false_negative_count == 2
    assert metrics.recall == 1 / 3
    assert metrics.median_normalized_center_error == 0.1
    assert metrics.p90_normalized_center_error == 0.1


def test_set_metrics_use_nearest_rank_for_the_registered_p90_tail() -> None:
    assignment = match_frame(
        (_circle("a", 0), _circle("b", 20)),
        (_prediction("pa", 1), _prediction("pb", 22)),
    )

    metrics = compute_set_metrics((assignment,))

    assert metrics.median_normalized_center_error == pytest.approx(0.15)
    assert metrics.p90_normalized_center_error == 0.2


def test_frozen_gates_keep_human_verdicts_pending_and_do_not_mix_track_continuity() -> None:
    assignment = match_frame(
        tuple(_circle(f"g-{index}", index * 20.0) for index in range(10)),
        tuple(_prediction(f"p-{index}", index * 20.0 + 4.0) for index in range(9)),
    )
    metrics = compute_set_metrics((assignment,))

    gates = compute_frozen_gates(metrics=metrics, completed_shots=24, total_shots=25)

    assert gates.coverage.passed is True
    assert gates.recall.passed is True
    assert gates.median_center_error.passed is True
    assert gates.p90_center_error.passed is True
    assert gates.severe_drift_or_wrong_target.status == "pending_human_review"
    assert gates.acceptable_without_redraw.status == "pending_human_review"
    assert gates.automated_gates_passed is True
    assert gates.pilot_status == "pending_human_review"


def _unit(index: int, stratum: str) -> ReviewUnit:
    unit_type = {
        "clean_match": "matched_pair",
        "flagged_or_ambiguous_match": "matched_pair",
        "miss": "unmatched_ground_truth",
        "false_positive": "unmatched_prediction",
    }[stratum]
    return ReviewUnit(
        unit_id=f"{stratum}-{index:03d}",
        unit_type=unit_type,
        stratum=stratum,
        asset_id="video-1",
        frame_index=index,
        ground_truth_id=None if stratum == "false_positive" else f"g-{index}",
        prediction_id=None if stratum == "miss" else f"p-{index}",
        qa_flags=("ambiguous",) if stratum == "flagged_or_ambiguous_match" else (),
    )


def test_review_sample_applies_quotas_then_stable_hash_fills_remaining_capacity() -> None:
    populations = {
        "clean_match": 55,
        "flagged_or_ambiguous_match": 35,
        "miss": 20,
        "false_positive": 20,
    }
    units = tuple(
        _unit(index, stratum) for stratum, count in populations.items() for index in range(count)
    )

    first = freeze_review_sample(units)
    second = freeze_review_sample(tuple(reversed(units)))

    assert first == second
    assert first.population_counts == populations
    assert len(first.units) == 100
    assert first.primary_quota_sample_counts == {
        "clean_match": 40,
        "flagged_or_ambiguous_match": 30,
        "miss": 15,
        "false_positive": 15,
    }
    assert sum(first.sample_counts.values()) == 100
