from __future__ import annotations

import pytest

from billiards_ai.lab.shot_local.regression import (
    aggregate_regression_cases,
    evaluate_regression_case,
)


def test_case_matches_world_events_by_frame_and_physical_identity() -> None:
    world_events = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "ball_collision",
                        "status": "observed",
                        "contact_frame_id": 51,
                        "actor_id": "shot-0001-anchor-09",
                        "target_id": "shot-0001-anchor-14",
                    },
                    {
                        "event_type": "cushion",
                        "status": "observed",
                        "contact_frame_id": 73,
                        "actor_id": "shot-0001-anchor-14",
                        "rail": "left",
                    },
                    {
                        "event_type": "cushion",
                        "status": "abstained",
                        "contact_frame_id": 80,
                        "actor_id": "shot-0001-anchor-14",
                        "rail": "bottom",
                    },
                ],
            }
        ]
    }
    truth = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "ball_collision",
                        "contact_frame_id": 52,
                        "actor_id": "shot-0001-anchor-09",
                        "target_id": "shot-0001-anchor-14",
                    },
                    {
                        "event_type": "cushion",
                        "contact_frame_id": 72,
                        "actor_id": "shot-0001-anchor-14",
                        "rail": "left",
                    },
                ],
                "identity_checkpoints": [],
            }
        ]
    }

    result = evaluate_regression_case(world_events, {"shots": []}, truth, frame_tolerance=1)

    assert result["overall"] == {
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert result["by_event_type"]["ball_collision"]["tp"] == 1
    assert result["identity_checkpoints"]["total"] == 0


def test_case_rejects_time_only_match_when_collision_identity_is_wrong() -> None:
    prediction = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "ball_collision",
                        "status": "observed",
                        "contact_frame_id": 20,
                        "actor_id": "shot-0001-anchor-01",
                        "target_id": "shot-0001-anchor-02",
                    }
                ],
            }
        ]
    }
    truth = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "ball_collision",
                        "contact_frame_id": 20,
                        "actor_id": "shot-0001-anchor-03",
                        "target_id": "shot-0001-anchor-04",
                    }
                ],
                "identity_checkpoints": [],
            }
        ]
    }

    result = evaluate_regression_case(prediction, {"shots": []}, truth, frame_tolerance=0)

    assert result["overall"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
    }


def test_identity_checkpoints_require_the_same_track_near_reviewed_position() -> None:
    truth = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [],
                "identity_checkpoints": [
                    {
                        "track_id": "shot-0001-anchor-10",
                        "frame_id": 130,
                        "x_mm": 95.0,
                        "y_mm": 79.9,
                        "tolerance_mm": 5.0,
                    },
                    {
                        "track_id": "shot-0001-anchor-10",
                        "frame_id": 154,
                        "x_mm": 500.0,
                        "y_mm": 500.0,
                        "tolerance_mm": 5.0,
                    },
                ],
            }
        ]
    }
    active_tracks = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "tracks": [
                    {
                        "track_id": "shot-0001-anchor-10",
                        "samples": [
                            {"frame_id": 130, "x_mm": 95.0, "y_mm": 79.9},
                            {"frame_id": 154, "x_mm": 346.2, "y_mm": 281.0},
                        ],
                    }
                ],
            }
        ]
    }

    result = evaluate_regression_case({"shots": []}, active_tracks, truth)

    assert result["identity_checkpoints"]["total"] == 2
    assert result["identity_checkpoints"]["passed"] == 1
    assert result["identity_checkpoints"]["failed"] == 1
    assert result["identity_checkpoints"]["pass_rate"] == 0.5
    assert result["identity_checkpoints"]["audit"][1]["reason"] == "position_mismatch"


def test_truth_requires_unique_shots_and_at_least_one_reviewed_fact() -> None:
    duplicate = {
        "shots": [
            {"shot_id": "shot-0001", "events": [], "identity_checkpoints": []},
            {"shot_id": "shot-0001", "events": [], "identity_checkpoints": []},
        ]
    }

    with pytest.raises(ValueError, match="truth shot IDs must be unique"):
        evaluate_regression_case({"shots": []}, {"shots": []}, duplicate)

    with pytest.raises(ValueError, match="manual reviewed fact"):
        evaluate_regression_case(
            {"shots": []},
            {"shots": []},
            {"shots": [{"shot_id": "shot-0001", "events": [], "identity_checkpoints": []}]},
        )


def test_aggregate_uses_micro_counts_instead_of_averaging_percentages() -> None:
    cases = [
        {
            "case_id": "one",
            "status": "PASS",
            "evaluation": {
                "overall": {"tp": 8, "fp": 2, "fn": 0},
                "by_event_type": {"cushion": {"tp": 8, "fp": 2, "fn": 0}},
                "identity_checkpoints": {"total": 2, "passed": 2, "failed": 0},
            },
        },
        {
            "case_id": "two",
            "status": "PASS",
            "evaluation": {
                "overall": {"tp": 1, "fp": 0, "fn": 1},
                "by_event_type": {"ball_collision": {"tp": 1, "fp": 0, "fn": 1}},
                "identity_checkpoints": {"total": 1, "passed": 0, "failed": 1},
            },
        },
    ]

    result = aggregate_regression_cases(cases)

    assert result["overall"] == {
        "tp": 9,
        "fp": 2,
        "fn": 1,
        "precision": 9 / 11,
        "recall": 0.9,
    }
    assert result["identity_checkpoints"]["pass_rate"] == 2 / 3
