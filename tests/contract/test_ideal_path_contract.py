"""Strict, versioned business contract for Story 5.2 ideal paths."""

from __future__ import annotations

import importlib
import importlib.util
import json

import pytest
from pydantic import ValidationError


def _contract():
    assert importlib.util.find_spec("billiards_ai.shared.models.ideal_path") is not None, (
        "Story 5.2 ideal-path contract is missing"
    )
    return importlib.import_module("billiards_ai.shared.models.ideal_path")


def _table(models):
    return models.IdealPathTableConfiguration(
        configuration_version="chinese-eight-ball-9ft-v1",
        length_mm=2540,
        width_mm=1270,
        ball_diameter_mm=57,
        corner_pocket_mouth_mm=86,
        side_pocket_mouth_mm=92,
        cushion_nose_width_mm=40,
    )


def _request(models, **updates):
    payload = {
        "cue_ball": {"ball_id": 0, "position": {"x_mm": 500, "y_mm": 635}},
        "object_balls": (
            {"ball_id": 3, "position": {"x_mm": 1500, "y_mm": 635}},
            {"ball_id": 8, "position": {"x_mm": 2100, "y_mm": 300}},
        ),
        "target_ball_id": 3,
        "target_pocket": "TOP_RIGHT",
        "intent": "DIRECT_POT",
        "target_ball_reviewed": True,
        "target_pocket_reviewed": True,
        "intent_reviewed": True,
        "calibration_status": "ACTIVE",
        "table": _table(models),
    }
    payload.update(updates)
    return models.IdealPathRequest(**payload)


def test_contract_is_strict_frozen_versioned_and_rejects_unknown_fields():
    models = _contract()
    request = _request(models)

    assert request.schema_version == "1.0"
    assert request.model_config["strict"] is True
    assert request.model_config["frozen"] is True
    assert request.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError, match="unexpected"):
        models.IdealPathRequest(**request.model_dump(), unexpected="raw")
    with pytest.raises(ValidationError):
        request.cue_ball.position.x_mm = 501


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cue_ball": {"ball_id": 0, "position": {"x_mm": 500.5, "y_mm": 635}}}, "int"),
        ({"cue_ball": {"ball_id": 0, "position": {"x_mm": -1, "y_mm": 635}}}, "table"),
        (
            {
                "object_balls": (
                    {"ball_id": 3, "position": {"x_mm": 1500, "y_mm": 635}},
                    {"ball_id": 3, "position": {"x_mm": 1600, "y_mm": 635}},
                )
            },
            "unique",
        ),
        (
            {
                "table": {
                    "configuration_version": "chinese-eight-ball-9ft-v1",
                    "length_mm": 2500,
                    "width_mm": 1270,
                    "ball_diameter_mm": 57,
                    "corner_pocket_mouth_mm": 86,
                    "side_pocket_mouth_mm": 92,
                    "cushion_nose_width_mm": 40,
                }
            },
            "supported",
        ),
    ],
)
def test_mm_coordinates_ball_ids_and_fixed_table_configuration_fail_closed(updates, message):
    models = _contract()

    with pytest.raises(ValidationError, match=message):
        _request(models, **updates)


def test_incomplete_or_unreliable_intent_is_constructible_for_typed_abstention():
    models = _contract()

    request = _request(
        models,
        target_ball_id=None,
        target_pocket=None,
        intent=None,
        target_ball_reviewed=False,
        target_pocket_reviewed=False,
        intent_reviewed=False,
    )

    assert request.target_ball_id is None
    assert request.target_pocket is None
    assert request.intent is None


def test_every_business_point_is_bounded_by_the_versioned_table_plane():
    models = _contract()

    with pytest.raises(ValidationError):
        models.IdealPathPoint(x_mm=-1, y_mm=0)
    with pytest.raises(ValidationError):
        models.IdealPathPoint(x_mm=2541, y_mm=0)
    with pytest.raises(ValidationError):
        models.IdealPathPoint(x_mm=0, y_mm=1271)


def test_result_contains_only_versioned_mm_business_events():
    models = _contract()
    result = models.IdealPathResult(
        status="SUCCEEDED",
        reason=None,
        adapter_version="1.0",
        engine_name="pooltool-billiards",
        engine_version="0.6.0",
        source_configuration_version="chinese-eight-ball-9ft-v1",
        events=(
            {
                "sequence": 1,
                "kind": "CUE_STRIKE",
                "ball_id": 0,
                "position": {"x_mm": 500, "y_mm": 635},
            },
            {
                "sequence": 2,
                "kind": "BALL_CONTACT",
                "ball_id": 0,
                "other_ball_id": 3,
                "position": {"x_mm": 1500, "y_mm": 635},
            },
            {
                "sequence": 3,
                "kind": "POCKETED",
                "ball_id": 3,
                "pocket": "TOP_RIGHT",
                "position": {"x_mm": 2540, "y_mm": 1270},
            },
        ),
    )

    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    schema = json.dumps(models.IdealPathResult.model_json_schema()).lower()
    assert result.schema_version == "1.0"
    assert result.unit == "mm"
    assert result.coordinate_system == "LEFT_BOTTOM_POCKET_X_LONG_Y_SHORT"
    assert all(
        forbidden not in (serialized + schema).lower()
        for forbidden in (
            "pooltool object",
            "raw_observation",
            "trajectory",
            "media_url",
            "image_url",
            "video_url",
            "execution_parameters",
        )
    )


@pytest.mark.parametrize(
    "event",
    [
        {
            "sequence": 1,
            "kind": "BALL_CONTACT",
            "ball_id": 0,
            "position": {"x_mm": 1500, "y_mm": 635},
        },
        {
            "sequence": 1,
            "kind": "POCKETED",
            "ball_id": 3,
            "position": {"x_mm": 2540, "y_mm": 1270},
        },
        {
            "sequence": 1,
            "kind": "STOP",
            "ball_id": 3,
            "pocket": "TOP_RIGHT",
            "position": {"x_mm": 1500, "y_mm": 635},
        },
    ],
)
def test_event_relationship_and_pocket_fields_fail_closed(event):
    models = _contract()

    with pytest.raises(ValidationError):
        models.IdealPathEvent(**event)


def test_failed_result_cannot_carry_fabricated_events():
    models = _contract()

    with pytest.raises(ValidationError, match="events"):
        models.IdealPathResult(
            status="SIMULATION_FAILED",
            reason="adapter_failure",
            adapter_version="1.0",
            engine_name="pooltool-billiards",
            engine_version="0.6.0",
            source_configuration_version="chinese-eight-ball-9ft-v1",
            events=(
                {
                    "sequence": 1,
                    "kind": "STOP",
                    "ball_id": 0,
                    "position": {"x_mm": 500, "y_mm": 635},
                },
            ),
        )
