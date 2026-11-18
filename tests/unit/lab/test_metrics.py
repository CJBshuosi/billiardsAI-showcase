import pytest

from billiards_ai.lab.metrics import (
    classification_metrics,
    event_metrics,
    first_hit_metrics,
    position_error_metrics,
    shot_boundary_metrics,
    tracking_metrics,
)


def test_abstained_first_hit_counts_as_uncovered_not_correct() -> None:
    result = first_hit_metrics(
        truth=["3", "8"],
        prediction=["3", None],
        eligible=[True, True],
    )

    assert result.accuracy == 0.5
    assert result.coverage == 0.5
    assert result.denominator == 2
    assert result.errors == 1
    assert result.abstentions == 1


def test_position_metrics_keep_tail_error_visible() -> None:
    result = position_error_metrics([0.0, 5.0, 10.0, 40.0])

    assert result.median == 7.5
    assert result.p95 == 40.0


def test_identity_macro_f1_does_not_hide_a_failed_class() -> None:
    result = classification_metrics(
        truth=["cue", "cue", "8", "8"],
        prediction=["cue", "cue", "cue", None],
        classes=["cue", "8"],
    )

    assert result.class_f1["cue"] == pytest.approx(0.8)
    assert result.class_f1["8"] == 0.0
    assert result.macro_f1 == pytest.approx(0.4)
    assert result.coverage == 0.75


def test_tracking_metrics_report_identity_failures_and_switch_rate() -> None:
    result = tracking_metrics(
        identity_true_positive=8,
        identity_false_positive=1,
        identity_false_negative=1,
        detection_accuracy=0.9,
        association_accuracy=0.81,
        id_switches=1,
        ball_shots=20,
    )

    assert result.idf1 == pytest.approx(8 / 9)
    assert result.hota == pytest.approx(0.8538149682)
    assert result.id_switch_per_ball_shot == 0.05


def test_shot_boundaries_match_only_inside_tolerance() -> None:
    result = shot_boundary_metrics(
        truth=[(0, 1000), (2000, 3000)],
        prediction=[(100, 1100), (2300, 3300)],
        tolerance_ms=200,
    )

    assert result.f1 == 0.5
    assert result.matched == 1
    assert result.boundary_median_error_ms == 100
    assert result.boundary_p95_error_ms == 100


def test_identity_aware_events_require_matching_ball_and_time() -> None:
    result = event_metrics(
        truth=[("3", 1000), ("8", 2000)],
        prediction=[("3", 1050), ("5", 2000)],
        tolerance_ms=100,
    )

    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
