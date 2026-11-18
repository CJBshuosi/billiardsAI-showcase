from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.edge.coach_runtime.replay import (
    MAX_REALTIME_OBSERVATION_LINE_BYTES,
    MAX_REALTIME_OBSERVATION_RECORDS,
    MAX_REALTIME_OBSERVATIONS_BYTES,
    decode_replay_observations,
)
from billiards_ai.shared.models.quality import (
    FieldQuality,
    FinalFieldSource,
    ShotAnalysisStatus,
    ShotFieldQuality,
)
from billiards_ai.shared.models.realtime import (
    RealtimeObservationStatus,
    RealtimeShotObservation,
)
from billiards_ai.shared.models.shot import Confidence, TargetZoneSource

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "core_intelligence"
    / "v1"
    / "realtime-shot-observation.json"
)

EXPECTED_WIRE_FIELDS = {
    "schema_version",
    "event_id",
    "session_id",
    "shot_sequence",
    "monotonic_timestamp_ms",
    "shot",
    "status",
    "evidence_sha256",
    "perception_model_version",
    "calibration_revision",
    "observed_at",
}

HANDOFF_STATUSES = (
    RealtimeObservationStatus.OBSERVED,
    RealtimeObservationStatus.ABSTAINED,
    RealtimeObservationStatus.UNSUPPORTED,
)


def _field_quality(*, known: bool) -> ShotFieldQuality:
    value = FieldQuality(
        source=FinalFieldSource.CV if known else FinalFieldSource.UNKNOWN,
        confidence=0.95 if known else 0.0,
    )
    return ShotFieldQuality(
        ball_positions_before=value,
        ball_positions_after=value,
        cue_ball_end=value,
        struck_ball=value,
        potted_balls=value,
        cushion_count=value,
        travel_distance_mm=value,
        is_foul=value,
    )


def _canonical_bytes(observation: RealtimeShotObservation) -> bytes:
    return json.dumps(
        observation.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _observation(
    sequence: int,
    status: RealtimeObservationStatus,
) -> RealtimeShotObservation:
    base = RealtimeShotObservation.model_validate_json(FIXTURE.read_bytes())
    observed = status is RealtimeObservationStatus.OBSERVED
    shot = base.shot.model_copy(
        update={
            "shot_id": UUID(f"20000000-0000-4000-8000-{sequence:012d}"),
            "sequence": sequence,
            "t_start_ms": sequence * 1_000,
            "t_end_ms": sequence * 1_000 + 250,
            "ball_positions_before": base.shot.ball_positions_before if observed else (),
            "ball_positions_after": base.shot.ball_positions_after if observed else (),
            "cue_ball_end": base.shot.cue_ball_end if observed else None,
            "struck_ball": base.shot.struck_ball if observed else None,
            "potted_balls": base.shot.potted_balls if observed else (),
            "cushion_count": base.shot.cushion_count if observed else None,
            "travel_distance_mm": base.shot.travel_distance_mm if observed else None,
            "is_foul": base.shot.is_foul if observed else None,
            "target_zone_source": (
                base.shot.target_zone_source if observed else TargetZoneSource.NONE
            ),
            "deviation_mm": base.shot.deviation_mm if observed else None,
            "confidence": Confidence.HIGH if observed else Confidence.ABSTAIN,
            "field_quality": _field_quality(known=observed),
            "analysis_status": (
                ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE
                if status is RealtimeObservationStatus.UNSUPPORTED
                else ShotAnalysisStatus.ANALYZED
            ),
        }
    )
    candidate = base.model_copy(
        update={
            "event_id": UUID(f"10000000-0000-4000-8000-{sequence:012d}"),
            "shot_sequence": sequence,
            "monotonic_timestamp_ms": sequence * 1_000 + 1_500,
            "shot": shot,
            "status": status,
            "evidence_sha256": hashlib.sha256(
                f"story-10.7-observation:{sequence}".encode()
            ).hexdigest(),
        }
    )
    return RealtimeShotObservation.model_validate(candidate.model_dump(), strict=True)


def test_story_10_7_handoff_freezes_existing_wire_fields_and_statuses() -> None:
    assert set(RealtimeShotObservation.model_fields) == EXPECTED_WIRE_FIELDS
    assert {status.value for status in RealtimeObservationStatus} == {
        "OBSERVED",
        "ABSTAINED",
        "UNSUPPORTED",
    }
    assert "INCOMPLETE" not in RealtimeObservationStatus.__members__


def test_three_status_handoff_vectors_use_the_production_replay_decoder() -> None:
    observations = tuple(
        _observation(sequence, status) for sequence, status in enumerate(HANDOFF_STATUSES, start=1)
    )
    payload = b"".join(_canonical_bytes(item) + b"\n" for item in observations)

    decoded = decode_replay_observations(
        payload,
        expected_session_id=observations[0].session_id,
    )

    assert decoded == observations
    assert tuple(item.status for item in decoded) == HANDOFF_STATUSES
    assert decoded[1].shot.confidence is Confidence.ABSTAIN
    assert decoded[2].shot.confidence is Confidence.ABSTAIN
    assert decoded[2].shot.analysis_status is ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE


def test_handoff_helper_supports_high_sequences_without_invalid_hashes() -> None:
    observation = _observation(16, RealtimeObservationStatus.OBSERVED)

    assert observation.shot_sequence == 16
    assert len(observation.evidence_sha256) == 64


def test_three_status_vectors_carry_explicit_valid_field_quality() -> None:
    observations = tuple(
        _observation(sequence, status) for sequence, status in enumerate(HANDOFF_STATUSES, start=1)
    )

    assert all(item.shot.field_quality is not None for item in observations)
    assert all(
        quality.source is not FinalFieldSource.UNKNOWN and quality.confidence > 0.0
        for quality in observations[0].shot.field_quality.values()
    )
    assert all(
        quality.source is FinalFieldSource.UNKNOWN and quality.confidence == 0.0
        for observation in observations[1:]
        for quality in observation.shot.field_quality.values()
    )


def test_positive_corpus_separates_exact_retries_from_new_sequences() -> None:
    observed = _observation(1, RealtimeObservationStatus.OBSERVED)
    abstained = _observation(2, RealtimeObservationStatus.ABSTAINED)
    unsupported = _observation(3, RealtimeObservationStatus.UNSUPPORTED)
    payload = b"".join(
        _canonical_bytes(item) + b"\n" for item in (observed, observed, abstained, unsupported)
    )

    decoded = decode_replay_observations(
        payload,
        expected_session_id=observed.session_id,
    )

    assert decoded == (observed, observed, abstained, unsupported)


def test_negative_corpus_rejects_same_key_with_different_full_payload() -> None:
    observation = _observation(1, RealtimeObservationStatus.OBSERVED)
    conflicting = observation.model_copy(
        update={"perception_model_version": ("model:conflicting@1.0.0+sha256:" + "a" * 64)}
    )
    payload = _canonical_bytes(observation) + b"\n" + _canonical_bytes(conflicting) + b"\n"

    with pytest.raises(ValueError, match="conflicting replay observation sequence"):
        decode_replay_observations(payload, expected_session_id=observation.session_id)


def test_negative_corpus_rejects_sequence_gap() -> None:
    first = _observation(1, RealtimeObservationStatus.OBSERVED)
    third = _observation(3, RealtimeObservationStatus.UNSUPPORTED)
    payload = _canonical_bytes(first) + b"\n" + _canonical_bytes(third) + b"\n"

    with pytest.raises(ValueError, match="replay observation sequence gap"):
        decode_replay_observations(payload, expected_session_id=first.session_id)


@pytest.mark.parametrize(
    ("field", "invalid", "match"),
    (
        ("schema_version", "2.0", "schema_version"),
        ("event_id", "10000000-0000-1000-8000-000000000001", "UUID version 4"),
        ("shot_sequence", 0, "greater than or equal to 1"),
        ("monotonic_timestamp_ms", -1, "greater than or equal to 0"),
        ("evidence_sha256", "A" * 64, "String should match pattern"),
        ("perception_model_version", " ", "at least 1 character"),
        ("calibration_revision", 0, "greater than or equal to 1"),
        ("observed_at", "2026-08-04T16:30:00+08:00", "datetime must use UTC"),
    ),
)
def test_wire_schema_rejects_invalid_top_level_values(
    field: str,
    invalid: object,
    match: str,
) -> None:
    values = _observation(1, RealtimeObservationStatus.OBSERVED).model_dump(mode="json")
    values[field] = invalid

    with pytest.raises(ValidationError, match=match):
        RealtimeShotObservation.model_validate_json(json.dumps(values))


def test_wire_schema_forbids_extra_fields_and_cross_identity() -> None:
    values = _observation(1, RealtimeObservationStatus.OBSERVED).model_dump(mode="json")
    with_extra = {**values, "stream_generation": 1}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RealtimeShotObservation.model_validate_json(json.dumps(with_extra))

    mismatched = deepcopy(values)
    mismatched["shot"]["sequence"] = 2
    with pytest.raises(ValidationError, match="observation and shot sequence must match"):
        RealtimeShotObservation.model_validate_json(json.dumps(mismatched))

    cross_session = deepcopy(values)
    cross_session["shot"]["session_id"] = "00000000-0000-4000-8000-000000000003"
    with pytest.raises(ValidationError, match="observation and shot session must match"):
        RealtimeShotObservation.model_validate_json(json.dumps(cross_session))

    invalid_interval = deepcopy(values)
    invalid_interval["shot"]["t_end_ms"] = invalid_interval["shot"]["t_start_ms"]
    with pytest.raises(ValidationError, match="finalized shot interval must be positive"):
        RealtimeShotObservation.model_validate_json(json.dumps(invalid_interval))


def test_wire_contract_is_frozen() -> None:
    observation = _observation(1, RealtimeObservationStatus.OBSERVED)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        observation.shot_sequence = 2


def test_current_shared_validator_gap_is_explicitly_frozen() -> None:
    observation = _observation(1, RealtimeObservationStatus.OBSERVED)
    invalid_shot = observation.shot.model_copy(update={"confidence": Confidence.ABSTAIN})

    accepted = RealtimeShotObservation.model_validate(
        observation.model_copy(update={"shot": invalid_shot}).model_dump(),
        strict=True,
    )

    assert accepted.status is RealtimeObservationStatus.OBSERVED
    assert accepted.shot.confidence is Confidence.ABSTAIN


def test_known_empty_and_unknown_fields_are_distinguished_by_field_quality() -> None:
    observation = _observation(1, RealtimeObservationStatus.OBSERVED)
    known_empty = observation.shot.model_copy(update={"potted_balls": ()})
    unknown = _observation(2, RealtimeObservationStatus.ABSTAINED).shot

    assert known_empty.potted_balls == ()
    assert known_empty.field_quality.potted_balls.source is not FinalFieldSource.UNKNOWN
    assert unknown.potted_balls == ()
    assert unknown.field_quality.potted_balls.source is FinalFieldSource.UNKNOWN


def test_live_replay_wire_budgets_are_frozen_for_the_missing_public_codec_seam() -> None:
    observation = _observation(1, RealtimeObservationStatus.OBSERVED)
    encoded = _canonical_bytes(observation)

    assert len(encoded) <= MAX_REALTIME_OBSERVATION_LINE_BYTES == 65_536
    assert MAX_REALTIME_OBSERVATIONS_BYTES == 8_388_608
    assert MAX_REALTIME_OBSERVATION_RECORDS == 1_000


def test_abstained_handoff_cannot_claim_non_abstained_shot_confidence() -> None:
    observation = _observation(1, RealtimeObservationStatus.ABSTAINED)
    invalid_shot = observation.shot.model_copy(update={"confidence": Confidence.HIGH})

    with pytest.raises(ValidationError, match="abstained observation"):
        RealtimeShotObservation.model_validate(
            observation.model_copy(update={"shot": invalid_shot}).model_dump(),
            strict=True,
        )
