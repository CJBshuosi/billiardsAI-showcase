"""Executable Story 2.7 contracts for objective shot observations."""

import pytest
from pydantic import ValidationError


def _api():
    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.observation import (
        BallMovement,
        ObjectiveShotObservation,
        ObservationConfidence,
        ObservationField,
        ObservationSource,
        RawObservation,
        StruckBallCandidate,
        UnsupportedReason,
    )
    from billiards_ai.shared.models.segmentation import BallPosition
    from billiards_ai.shared.models.shot import TargetZoneSource

    return {
        "BallMovement": BallMovement,
        "BallPosition": BallPosition,
        "ObjectiveShotObservation": ObjectiveShotObservation,
        "ObservationConfidence": ObservationConfidence,
        "ObservationField": ObservationField,
        "ObservationSource": ObservationSource,
        "Point": Point,
        "RawObservation": RawObservation,
        "StruckBallCandidate": StruckBallCandidate,
        "TargetZoneSource": TargetZoneSource,
        "UnsupportedReason": UnsupportedReason,
    }


def test_observation_contract_is_strict_frozen_and_keeps_candidates_only_in_raw_data():
    api = _api()
    source = api["ObservationSource"].SEGMENT_SNAPSHOT_DELTA
    high = api["ObservationConfidence"].HIGH
    unknown = api["ObservationConfidence"].UNKNOWN
    unsupported = api["UnsupportedReason"].FIRST_CONTACT_NOT_OBSERVABLE
    before = (
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=100, y_mm=200)),
        api["BallPosition"](ball_id=1, position=api["Point"](x_mm=500, y_mm=600)),
    )
    after = (
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=200, y_mm=200)),
        api["BallPosition"](ball_id=1, position=api["Point"](x_mm=550, y_mm=600)),
    )
    candidate = api["StruckBallCandidate"](
        ball_id=1,
        confidence=api["ObservationConfidence"].LOW,
        source=source,
        unsupported_reason=unsupported,
    )
    observation = api["ObjectiveShotObservation"](
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before=api["ObservationField"](value=before, confidence=high, source=source),
        ball_positions_after=api["ObservationField"](value=after, confidence=high, source=source),
        potted_balls=api["ObservationField"](value=(), confidence=high, source=source),
        cue_ball_end=api["ObservationField"](
            value=api["Point"](x_mm=200, y_mm=200), confidence=high, source=source
        ),
        ball_movements=api["ObservationField"](
            value=(
                api["BallMovement"](ball_id=0, distance_mm=100),
                api["BallMovement"](ball_id=1, distance_mm=50),
            ),
            confidence=high,
            source=source,
        ),
        travel_distance_mm=api["ObservationField"](value=100, confidence=high, source=source),
        cushion_count=api["ObservationField"](
            value=None,
            confidence=unknown,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
            unsupported_reason=api["UnsupportedReason"].CUSHION_CONTACT_NOT_OBSERVABLE,
        ),
        struck_ball=api["ObservationField"](
            value=None,
            confidence=unknown,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
            unsupported_reason=unsupported,
        ),
        is_foul=api["ObservationField"](
            value=None,
            confidence=unknown,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
            unsupported_reason=api["UnsupportedReason"].FOUL_NOT_OBSERVABLE,
        ),
        target_zone_source=api["TargetZoneSource"].NONE,
        deviation_mm=None,
        raw_observation=api["RawObservation"](struck_ball_candidates=(candidate,)),
    )

    assert observation.schema_version == "1.0"
    assert observation.struck_ball_candidates == (candidate,)
    assert observation.foul_evidence == ()
    assert "struck_ball_candidates" not in observation.model_dump(exclude={"raw_observation"})

    with pytest.raises(ValidationError):
        observation.t_end_ms = 901
    with pytest.raises(ValidationError):
        observation.raw_observation.struck_ball_candidates = ()
    with pytest.raises(ValidationError):
        api["ObjectiveShotObservation"].model_validate(
            observation.model_dump() | {"trajectory": [{"x_mm": 1, "y_mm": 2}]}
        )


def test_observation_fields_require_honest_unknown_reasons_and_canonical_evidence():
    api = _api()

    with pytest.raises(ValidationError, match="unsupported reason"):
        api["ObservationField"](
            value=None,
            confidence=api["ObservationConfidence"].UNKNOWN,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
        )
    with pytest.raises(ValidationError, match="unsupported reason"):
        api["ObservationField"](
            value=1,
            confidence=api["ObservationConfidence"].HIGH,
            source=api["ObservationSource"].SEGMENT_SNAPSHOT_DELTA,
            unsupported_reason=api["UnsupportedReason"].FOUL_NOT_OBSERVABLE,
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        api["RawObservation"](
            struck_ball_candidates=(
                api["StruckBallCandidate"](
                    ball_id=2,
                    confidence=api["ObservationConfidence"].LOW,
                    source=api["ObservationSource"].SEGMENT_SNAPSHOT_DELTA,
                    unsupported_reason=api["UnsupportedReason"].FIRST_CONTACT_NOT_OBSERVABLE,
                ),
                api["StruckBallCandidate"](
                    ball_id=1,
                    confidence=api["ObservationConfidence"].LOW,
                    source=api["ObservationSource"].SEGMENT_SNAPSHOT_DELTA,
                    unsupported_reason=api["UnsupportedReason"].FIRST_CONTACT_NOT_OBSERVABLE,
                ),
            )
        )


def test_observation_rejects_movement_or_cue_distance_inconsistent_with_snapshots():
    api = _api()
    source = api["ObservationSource"].SEGMENT_SNAPSHOT_DELTA
    high = api["ObservationConfidence"].HIGH
    unknown = api["ObservationConfidence"].UNKNOWN
    before = (
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=100, y_mm=200)),
        api["BallPosition"](ball_id=1, position=api["Point"](x_mm=500, y_mm=600)),
    )
    after = (
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=200, y_mm=200)),
        api["BallPosition"](ball_id=1, position=api["Point"](x_mm=550, y_mm=600)),
    )
    values = dict(
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before=api["ObservationField"](value=before, confidence=high, source=source),
        ball_positions_after=api["ObservationField"](value=after, confidence=high, source=source),
        potted_balls=api["ObservationField"](value=(), confidence=high, source=source),
        cue_ball_end=api["ObservationField"](
            value=after[0].position, confidence=high, source=source
        ),
        ball_movements=api["ObservationField"](
            value=(
                api["BallMovement"](ball_id=0, distance_mm=99),
                api["BallMovement"](ball_id=1, distance_mm=50),
            ),
            confidence=high,
            source=source,
        ),
        travel_distance_mm=api["ObservationField"](value=100, confidence=high, source=source),
        cushion_count=api["ObservationField"](
            value=None,
            confidence=unknown,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
            unsupported_reason=api["UnsupportedReason"].CUSHION_CONTACT_NOT_OBSERVABLE,
        ),
        struck_ball=api["ObservationField"](
            value=None,
            confidence=unknown,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
            unsupported_reason=api["UnsupportedReason"].NO_OBJECT_BALL_MOVEMENT_EVIDENCE,
        ),
        is_foul=api["ObservationField"](
            value=None,
            confidence=unknown,
            source=api["ObservationSource"].UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
            unsupported_reason=api["UnsupportedReason"].FOUL_NOT_OBSERVABLE,
        ),
        target_zone_source=api["TargetZoneSource"].NONE,
        deviation_mm=None,
        raw_observation=api["RawObservation"](),
    )

    with pytest.raises(ValidationError, match="movement distance"):
        api["ObjectiveShotObservation"](**values)
