from __future__ import annotations

from billiards_ai.lab.shot_local.evaluation import evaluate_event_timeline


def test_evaluator_counts_only_admitted_predictions_and_matches_by_time() -> None:
    timeline = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "first_contact",
                        "status": "admitted",
                        "timestamp_ms": 100,
                        "target_entity": "shot-0001-anchor-01",
                    },
                    {"event_type": "cushion", "status": "pending", "timestamp_ms": 180},
                    {"event_type": "pot", "status": "admitted", "timestamp_ms": 300},
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
                        "event_type": "first_contact",
                        "timestamp_ms": 115,
                        "target_entity": "shot-0001-anchor-01",
                    },
                    {"event_type": "cushion", "timestamp_ms": 180},
                    {"event_type": "ball_ball_collision", "timestamp_ms": 220},
                ],
            }
        ]
    }

    result = evaluate_event_timeline(timeline, truth, tolerance_ms=30)

    assert result["overall"] == {"tp": 1, "fp": 1, "fn": 2, "precision": 0.5, "recall": 1 / 3}
    assert result["by_event_type"]["cushion"] == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "precision": None,
        "recall": 0.0,
    }


def test_evaluator_rejects_duplicate_truth_shot_ids() -> None:
    duplicate = {"shots": [{"shot_id": "shot-0001", "events": []}] * 2}

    try:
        evaluate_event_timeline({"shots": []}, duplicate)
    except ValueError as exc:
        assert str(exc) == "truth shot IDs must be unique"
    else:
        raise AssertionError("duplicate truth was accepted")


def test_collision_identity_mismatch_is_fp_and_fn_even_at_same_timestamp() -> None:
    prediction = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "ball_ball_collision",
                        "status": "admitted",
                        "timestamp_ms": 100,
                        "actor_entity": "shot-0001-anchor-01",
                        "target_entity": "shot-0001-anchor-02",
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
                        "event_type": "ball_ball_collision",
                        "timestamp_ms": 100,
                        "actor_entity": "shot-0001-anchor-03",
                        "target_entity": "shot-0001-anchor-04",
                    }
                ],
            }
        ]
    }

    result = evaluate_event_timeline(prediction, truth)

    assert result["overall"]["tp"] == 0
    assert result["overall"]["fp"] == 1
    assert result["overall"]["fn"] == 1


def test_cushion_requires_actor_and_rail_and_pot_requires_actor_and_pocket() -> None:
    prediction = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "cushion",
                        "status": "admitted",
                        "timestamp_ms": 100,
                        "actor_entity": "shot-0001-anchor-01",
                        "rail": "left",
                    },
                    {
                        "event_type": "pot",
                        "status": "admitted",
                        "timestamp_ms": 200,
                        "actor_entity": "shot-0001-anchor-02",
                        "pocket": "left-bottom",
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
                        "event_type": "cushion",
                        "timestamp_ms": 100,
                        "actor_entity": "shot-0001-anchor-01",
                        "rail": "right",
                    },
                    {
                        "event_type": "pot",
                        "timestamp_ms": 200,
                        "actor_entity": "shot-0001-anchor-02",
                        "pocket": "right-bottom",
                    },
                ],
            }
        ]
    }

    result = evaluate_event_timeline(prediction, truth)

    assert result["overall"] == {
        "tp": 0,
        "fp": 2,
        "fn": 2,
        "precision": 0.0,
        "recall": 0.0,
    }


def test_pot_without_pocket_identity_cannot_match_by_time_only() -> None:
    prediction = {
        "shots": [
            {
                "shot_id": "shot-0001",
                "events": [
                    {
                        "event_type": "pot",
                        "status": "admitted",
                        "timestamp_ms": 200,
                        "actor_entity": "shot-0001-anchor-02",
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
                        "event_type": "pot",
                        "timestamp_ms": 200,
                        "actor_entity": "shot-0001-anchor-02",
                    }
                ],
            }
        ]
    }

    result = evaluate_event_timeline(prediction, truth)

    assert result["overall"]["tp"] == 0
    assert result["overall"]["fp"] == 1
    assert result["overall"]["fn"] == 1
