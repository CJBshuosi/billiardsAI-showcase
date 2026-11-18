"""Executable Story 2.8 contracts for optional ambiguity adjudication."""

from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError


def _api():
    from billiards_ai.shared.models.adjudication import (
        AdjudicatedStruckBall,
        AdjudicationCandidate,
        AdjudicationRawObservation,
        AdjudicationRequest,
        AdjudicationResult,
        AdjudicationSource,
        AdjudicationStatus,
    )
    from billiards_ai.shared.models.observation import ObservationConfidence, ObservationSource

    return {
        "AdjudicatedStruckBall": AdjudicatedStruckBall,
        "AdjudicationCandidate": AdjudicationCandidate,
        "AdjudicationRawObservation": AdjudicationRawObservation,
        "AdjudicationRequest": AdjudicationRequest,
        "AdjudicationResult": AdjudicationResult,
        "AdjudicationSource": AdjudicationSource,
        "AdjudicationStatus": AdjudicationStatus,
        "ObservationConfidence": ObservationConfidence,
        "ObservationSource": ObservationSource,
    }


def test_provider_request_is_frozen_and_contains_only_minimal_observation_evidence():
    api = _api()
    request = api["AdjudicationRequest"](
        shot_id=uuid4(),
        idempotency_key="a" * 64,
        evidence_signature="d" * 64,
        candidates=(
            api["AdjudicationCandidate"](
                ball_id=1,
                confidence=api["ObservationConfidence"].LOW,
                source=api["ObservationSource"].SEGMENT_SNAPSHOT_DELTA,
            ),
        ),
        estimated_cost_cny=Decimal("0.50"),
    )

    assert set(request.model_dump()) == {
        "schema_version",
        "shot_id",
        "idempotency_key",
        "evidence_signature",
        "candidates",
        "estimated_cost_cny",
    }
    assert not {
        "frame",
        "image",
        "video",
        "locator",
        "ball_positions_before",
        "ball_positions_after",
        "trajectory",
    }.intersection(request.model_dump())

    with pytest.raises(ValidationError):
        request.candidates = ()
    with pytest.raises(ValidationError):
        api["AdjudicationRequest"].model_validate(
            request.model_dump() | {"image_url": "https://example.invalid/keyframe.jpg"}
        )


def test_result_contract_is_strict_and_keeps_provider_text_only_in_raw_observation():
    api = _api()
    result = api["AdjudicationResult"](
        shot_id=uuid4(),
        idempotency_key="b" * 64,
        evidence_signature="d" * 64,
        result=api["AdjudicatedStruckBall"](ball_id=1),
        source=api["AdjudicationSource"].EXTERNAL_PROVIDER,
        confidence=api["ObservationConfidence"].MEDIUM,
        status=api["AdjudicationStatus"].SUCCEEDED,
        retryable=False,
        estimated_cost_cny=Decimal("0.50"),
        raw_observation=api["AdjudicationRawObservation"](
            provider_output='{"ball_id":1,"confidence":"MEDIUM","free_text":"candidate one"}'
        ),
    )

    business_fields = result.model_dump(exclude={"raw_observation"})
    assert "candidate one" not in repr(business_fields)
    assert "provider_output" not in business_fields
    assert result.raw_observation.provider_output is not None

    with pytest.raises(ValidationError):
        result.status = api["AdjudicationStatus"].PARSE_FAILED
    with pytest.raises(ValidationError, match="successful adjudication"):
        api["AdjudicationResult"].model_validate(result.model_dump() | {"result": None})
    with pytest.raises(ValidationError):
        api["AdjudicationResult"].model_validate(
            result.model_dump() | {"provider_model": "unapproved-model"}
        )


def test_retryable_is_reserved_for_typed_provider_failure():
    api = _api()
    common = {
        "shot_id": uuid4(),
        "idempotency_key": "c" * 64,
        "evidence_signature": "d" * 64,
        "result": None,
        "source": api["AdjudicationSource"].EXTERNAL_PROVIDER,
        "confidence": api["ObservationConfidence"].UNKNOWN,
        "estimated_cost_cny": Decimal("0.50"),
        "raw_observation": api["AdjudicationRawObservation"](failure_detail="temporary"),
    }

    api["AdjudicationResult"](
        **common,
        status=api["AdjudicationStatus"].PROVIDER_FAILED,
        retryable=True,
    )
    with pytest.raises(ValidationError, match="retryable"):
        api["AdjudicationResult"](
            **common,
            status=api["AdjudicationStatus"].PARSE_FAILED,
            retryable=True,
        )
