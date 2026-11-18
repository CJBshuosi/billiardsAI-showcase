"""Story 8.3 behavioral ATDD for the approved v2 metric evidence contract."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricShotEvidence,
    EvidenceApprovalKind,
    FieldSource,
    FoulResult,
    GameCode,
    MetricFieldQuality,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    ReviewedFieldQuality,
    ReviewedPoint,
    ReviewedTableGeometry,
    ReviewedTargetZone,
    TargetZoneEvaluationStatus,
    TargetZoneResult,
    TargetZoneShape,
)
from billiards_ai.shared.models.shot import TargetZoneSource


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _quality(confidence: float = 1.0) -> ReviewedFieldQuality:
    return ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=confidence)


def _field_quality(confidence: float = 1.0) -> MetricFieldQuality:
    quality = _quality(confidence)
    return MetricFieldQuality(
        geometry=quality,
        intent=quality,
        pot_attempt=quality,
        pot_result=quality,
        foul_result=quality,
        target_zone=quality,
        cue_ball_end=quality,
    )


def _calibration() -> dict[str, object]:
    return {
        "calibration_id": _id(1),
        "calibration_revision_id": _id(2),
        "calibration_revision_number": 1,
    }


def _point(x_mm: float, y_mm: float) -> ReviewedPoint:
    return ReviewedPoint(**_calibration(), x_mm=x_mm, y_mm=y_mm)


def _table() -> ReviewedTableGeometry:
    return ReviewedTableGeometry(
        **_calibration(),
        table_length_mm=2540,
        table_width_mm=1270,
        ball_diameter_mm=57.15,
        corner_pocket_mouth_mm=86,
        side_pocket_mouth_mm=92,
    )


def _metric_shot(**updates: object) -> ApprovedMetricShotEvidence:
    values: dict[str, object] = {
        "shot_id": _id(10),
        "session_id": _id(11),
        "sequence": 1,
        "review_revision": 1,
        "game_code": GameCode.CHINESE_EIGHT_BALL,
        "table_geometry": _table(),
        "cue_ball_start_mm": _point(100.0, 300.0),
        "object_ball_start_mm": _point(500.0, 300.0),
        "intended_object_ball": 3,
        "intended_target_pocket": PocketCode.BOTTOM_RIGHT,
        "pot_attempt_status": PotAttemptStatus.ATTEMPTED,
        "pot_result": PotResult.LEGAL_SUCCESS,
        "foul_result": FoulResult.LEGAL,
        "target_zone_source": TargetZoneSource.NONE,
        "target_zone": None,
        "cue_ball_end_mm": None,
        "target_zone_evaluation_status": TargetZoneEvaluationStatus.NOT_APPLICABLE,
        "target_zone_result": None,
        "field_quality": _field_quality(),
    }
    return ApprovedMetricShotEvidence.model_validate(values | updates)


def test_reviewed_metric_evidence_is_strict_frozen_raw_free_and_versioned() -> None:
    evidence = _metric_shot()

    assert evidence.schema_version == "1.0"
    assert evidence.game_code is GameCode.CHINESE_EIGHT_BALL
    with pytest.raises(ValidationError):
        ApprovedMetricShotEvidence.model_validate(
            evidence.model_dump(mode="python") | {"travel_distance_mm": 400}
        )
    with pytest.raises(ValidationError):
        evidence.__setattr__("sequence", 2)
    with pytest.raises(ValidationError):
        ApprovedMetricShotEvidence.model_validate(
            evidence.model_dump(mode="python") | {"sequence": True}
        )


@pytest.mark.parametrize(
    ("attempt", "result", "valid"),
    [
        (PotAttemptStatus.ATTEMPTED, PotResult.LEGAL_SUCCESS, True),
        (PotAttemptStatus.ATTEMPTED, PotResult.MISS, True),
        (PotAttemptStatus.ATTEMPTED, None, False),
        (PotAttemptStatus.NOT_ATTEMPTED, None, True),
        (PotAttemptStatus.NOT_ATTEMPTED, PotResult.MISS, False),
        (PotAttemptStatus.UNSUPPORTED, None, True),
        (PotAttemptStatus.ABSTAINED, None, True),
    ],
)
def test_pot_attempt_result_matrix_is_fail_closed(
    attempt: PotAttemptStatus,
    result: PotResult | None,
    valid: bool,
) -> None:
    values = _metric_shot().model_dump(mode="python") | {
        "pot_attempt_status": attempt,
        "pot_result": result,
    }
    if attempt is not PotAttemptStatus.ATTEMPTED:
        values |= {"intended_object_ball": None, "intended_target_pocket": None}

    if valid:
        assert ApprovedMetricShotEvidence.model_validate(values).pot_result is result
    else:
        with pytest.raises(ValidationError, match="pot"):
            ApprovedMetricShotEvidence.model_validate(values)


def test_target_zone_matrix_requires_reviewed_closed_geometry_and_result() -> None:
    zone = ReviewedTargetZone(
        **_calibration(),
        shape=TargetZoneShape.RECTANGLE,
        minimum_x_mm=1000.0,
        minimum_y_mm=400.0,
        maximum_x_mm=1200.0,
        maximum_y_mm=600.0,
        vertices=(),
    )
    evidence = _metric_shot(
        target_zone_source=TargetZoneSource.SCENE_PRESET,
        target_zone=zone,
        cue_ball_end_mm=_point(1220.0, 500.0),
        target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
        target_zone_result=TargetZoneResult.INSIDE_OR_TOLERATED,
    )

    assert evidence.target_zone_result is TargetZoneResult.INSIDE_OR_TOLERATED
    for update in (
        {"target_zone": None},
        {"cue_ball_end_mm": None},
        {"target_zone_result": None},
        {"target_zone_source": TargetZoneSource.NONE},
    ):
        with pytest.raises(ValidationError, match="target zone"):
            ApprovedMetricShotEvidence.model_validate(evidence.model_dump(mode="python") | update)

    unsupported = evidence.model_dump(mode="python") | {
        "target_zone_evaluation_status": TargetZoneEvaluationStatus.UNSUPPORTED,
        "target_zone_result": None,
        "target_zone": None,
        "cue_ball_end_mm": None,
    }
    assert (
        ApprovedMetricShotEvidence.model_validate(unsupported).target_zone_source
        is TargetZoneSource.SCENE_PRESET
    )
    with pytest.raises(ValidationError, match="approved source"):
        ApprovedMetricShotEvidence.model_validate(
            unsupported | {"target_zone_source": TargetZoneSource.NONE}
        )


def test_target_zone_tolerance_uses_half_up_millimetre_boundary() -> None:
    zone = ReviewedTargetZone(
        **_calibration(),
        shape=TargetZoneShape.RECTANGLE,
        minimum_x_mm=1000.0,
        minimum_y_mm=400.0,
        maximum_x_mm=1200.0,
        maximum_y_mm=600.0,
    )
    assert (
        _metric_shot(
            target_zone_source=TargetZoneSource.SCENE_PRESET,
            target_zone=zone,
            cue_ball_end_mm=_point(1220.0004, 500.0),
            target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
            target_zone_result=TargetZoneResult.INSIDE_OR_TOLERATED,
        ).target_zone_result
        is TargetZoneResult.INSIDE_OR_TOLERATED
    )
    assert (
        _metric_shot(
            target_zone_source=TargetZoneSource.SCENE_PRESET,
            target_zone=zone,
            cue_ball_end_mm=_point(1220.0005, 500.0),
            target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
            target_zone_result=TargetZoneResult.OUTSIDE,
        ).target_zone_result
        is TargetZoneResult.OUTSIDE
    )


def test_polygon_zone_requires_canonical_clockwise_non_self_intersecting_vertices() -> None:
    clockwise = (
        _point(100.0, 100.0),
        _point(100.0, 200.0),
        _point(200.0, 200.0),
        _point(200.0, 100.0),
    )

    assert (
        ReviewedTargetZone(
            **_calibration(),
            shape=TargetZoneShape.POLYGON,
            vertices=clockwise,
        ).vertices
        == clockwise
    )
    with pytest.raises(ValidationError, match="canonical"):
        ReviewedTargetZone(
            **_calibration(),
            shape=TargetZoneShape.POLYGON,
            vertices=clockwise[1:] + clockwise[:1],
        )
    with pytest.raises(ValidationError, match="self-intersect"):
        ReviewedTargetZone(
            **_calibration(),
            shape=TargetZoneShape.POLYGON,
            vertices=(clockwise[0], clockwise[2], clockwise[1], clockwise[3]),
        )

    touching = (
        _point(100.0, 100.0),
        _point(100.0, 300.0),
        _point(300.0, 300.0),
        _point(300.0, 100.0),
        _point(150.0, 100.0),
        _point(150.0, 250.0),
        _point(250.0, 250.0),
        _point(250.0, 100.0),
    )
    with pytest.raises(ValidationError, match="self-intersect"):
        ReviewedTargetZone(**_calibration(), shape=TargetZoneShape.POLYGON, vertices=touching)

    collinear_overlap = (
        _point(100.0, 100.0),
        _point(100.0, 300.0),
        _point(300.0, 300.0),
        _point(300.0, 100.0),
        _point(100.0, 200.0),
        _point(250.0, 200.0),
    )
    with pytest.raises(ValidationError, match="self-intersect"):
        ReviewedTargetZone(
            **_calibration(),
            shape=TargetZoneShape.POLYGON,
            vertices=collinear_overlap,
        )


def test_coincident_cue_and_object_centres_are_invalid_reviewed_geometry() -> None:
    with pytest.raises(ValidationError, match="centres must differ"):
        _metric_shot(object_ball_start_mm=_point(100.0, 300.0))


def test_negative_zero_coordinates_canonicalize_to_positive_zero() -> None:
    negative = _point(-0.0, -0.0)
    positive = _point(0.0, 0.0)
    assert negative.model_dump_json() == positive.model_dump_json()
    assert negative.x_mm.hex() == positive.x_mm.hex() == "0x0.0p+0"


def test_unknown_field_source_is_only_valid_at_zero_confidence_and_non_finite_is_rejected() -> None:
    assert ReviewedFieldQuality(source=FieldSource.UNKNOWN, confidence=0.0).confidence == 0.0
    with pytest.raises(ValidationError):
        ReviewedFieldQuality(source=FieldSource.UNKNOWN, confidence=0.1)
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=value)


def test_approval_kind_is_a_closed_wire_enum() -> None:
    assert tuple(EvidenceApprovalKind) == (
        EvidenceApprovalKind.HUMAN_REVIEW,
        EvidenceApprovalKind.AUTOMATED_POLICY,
    )
    with pytest.raises(ValueError):
        EvidenceApprovalKind("future")
