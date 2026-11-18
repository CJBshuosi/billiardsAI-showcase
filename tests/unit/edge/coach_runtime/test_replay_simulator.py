from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.edge.coach_runtime.execution_policy import (
    AbstentionAction,
    DrillExecutionProjection,
    ExecutionBlockProjection,
    MetricCombination,
    MetricRuleProjection,
    ReplanContinueAction,
    SuccessPolicyProjection,
    _issue_verified_execution_projection,
)
from billiards_ai.edge.coach_runtime.replay import (
    COACH_RUNTIME_REPLAY_VERSION,
    MAX_REALTIME_OBSERVATION_RECORDS,
    MAX_REALTIME_OBSERVATIONS_BYTES,
    ReplayEvidenceLineageV1,
    ReplayPlanFixtureV1,
    canonical_replay_plan_fixture_bytes,
    canonical_replay_result_bytes,
    decode_replay_observations,
    decode_replay_plan_fixture,
    run_replay,
)
from billiards_ai.edge.coach_runtime.state_machine import CoachRuntimeState, VerifiedPlanAdmission
from billiards_ai.shared.models.growth import PlanOutcomeStatus
from billiards_ai.shared.models.quality import FieldQuality, FinalFieldSource, ShotFieldQuality
from billiards_ai.shared.models.realtime import RealtimeObservationStatus, RealtimeShotObservation
from billiards_ai.shared.models.shot import Confidence

OBSERVATION_FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "core_intelligence"
    / "v1"
    / "realtime-shot-observation.json"
)
REPLAY_FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "coach_runtime" / "replay" / "v1"

PLAN_FIXTURE_BYTES = (
    b'{"admission_id":"admission-1","bundle_content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaa'
    b'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bundle_id":"00000000-0000-4000-8000-'
    b'000000000010","evidence_lineage":[{"evidence_envelope_id":"00000000-0000-4000-'
    b'8000-000000000020","schema_version":"1.0","shot_sequence":1}],"outcome_recorded_at":'
    b'"2026-08-10T08:30:00Z",'
    b'"projection_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
    b'"runtime_version":"coach-runtime-v1","schema_version":"1.0","session_started_at":'
    b'"2026-08-10T08:00:00Z","session_started_monotonic_ms":1000}'
)


def test_plan_fixture_is_strict_frozen_and_canonical() -> None:
    fixture = decode_replay_plan_fixture(PLAN_FIXTURE_BYTES)

    assert type(fixture) is ReplayPlanFixtureV1
    assert fixture.session_started_at == datetime(2026, 8, 10, 8, tzinfo=UTC)
    assert fixture.evidence_lineage[0].evidence_envelope_id == UUID(
        "00000000-0000-4000-8000-000000000020"
    )
    assert canonical_replay_plan_fixture_bytes(fixture) == PLAN_FIXTURE_BYTES
    with pytest.raises(ValidationError):
        fixture.runtime_version = "changed"  # type: ignore[misc]


def test_frozen_synthetic_fixture_files_use_the_production_decoders() -> None:
    plan_bytes = (REPLAY_FIXTURE_ROOT / "plan-fixture.json").read_bytes()
    observation_bytes = (REPLAY_FIXTURE_ROOT / "observations.jsonl").read_bytes()

    assert plan_bytes == PLAN_FIXTURE_BYTES
    fixture = decode_replay_plan_fixture(plan_bytes)
    observations = decode_replay_observations(
        observation_bytes,
        expected_session_id=UUID("00000000-0000-4000-8000-000000000002"),
    )
    assert fixture.admission_id == "admission-1"
    assert tuple(item.shot_sequence for item in observations) == (1,)


@pytest.mark.parametrize(
    "bad",
    (
        PLAN_FIXTURE_BYTES.replace(
            b'"schema_version":"1.0"', b'"schema_version":"1.0","schema_version":"1.0"'
        ),
        PLAN_FIXTURE_BYTES.replace(
            b'"runtime_version":"coach-runtime-v1",',
            b'"unknown":1,"runtime_version":"coach-runtime-v1",',
        ),
        b" " + PLAN_FIXTURE_BYTES,
        PLAN_FIXTURE_BYTES.replace(b'"1.0"', b'"2.0"', 1),
        PLAN_FIXTURE_BYTES.replace(b"2026-08-10T08:30:00Z", b"2026-08-10T07:30:00Z"),
        PLAN_FIXTURE_BYTES.replace(
            b'"shot_sequence":1}',
            b'"shot_sequence":1},{"evidence_envelope_id":"00000000-0000-4000-8000-'
            b'000000000021","shot_sequence":1}',
        ),
        PLAN_FIXTURE_BYTES.replace(
            b'"evidence_lineage":[',
            b'"evidence_lineage":[{"evidence_envelope_id":"00000000-0000-4000-'
            b'8000-000000000020","schema_version":"1.0","shot_sequence":2},',
        ),
        PLAN_FIXTURE_BYTES + b"\n",
        b"x" * 65_537,
        b"\xff",
    ),
)
def test_plan_fixture_rejects_ambiguous_or_noncanonical_input(bad: bytes) -> None:
    with pytest.raises(ValueError):
        decode_replay_plan_fixture(bad)


VALID_FIELD_QUALITY = ShotFieldQuality(
    ball_positions_before=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    ball_positions_after=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    cue_ball_end=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    struck_ball=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    potted_balls=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    cushion_count=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    travel_distance_mm=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
    is_foul=FieldQuality(source=FinalFieldSource.CV, confidence=0.9),
)
USER_ID = UUID("00000000-0000-4000-8000-000000000001")


def _observation(
    sequence: int = 1,
    *,
    digest: str | None = None,
    status: RealtimeObservationStatus = RealtimeObservationStatus.OBSERVED,
    succeeds: bool = True,
) -> RealtimeShotObservation:
    base = RealtimeShotObservation.model_validate_json(OBSERVATION_FIXTURE.read_bytes())
    observed_at = datetime(2026, 8, 10, 8, 0, sequence, tzinfo=UTC)
    shot = base.shot.model_copy(
        update={
            "shot_id": UUID(f"20000000-0000-4000-8000-{sequence:012d}"),
            "sequence": sequence,
            "t_start_ms": sequence * 1000,
            "t_end_ms": sequence * 1000 + 250,
            "potted_balls": base.shot.potted_balls if succeeds else (),
            "created_at": observed_at,
            "confidence": (
                Confidence.ABSTAIN
                if status is RealtimeObservationStatus.ABSTAINED
                else Confidence.HIGH
            ),
            "field_quality": VALID_FIELD_QUALITY,
        }
    )
    return RealtimeShotObservation.model_validate(
        base.model_copy(
            update={
                "event_id": UUID(f"10000000-0000-4000-8000-{sequence:012d}"),
                "shot_sequence": sequence,
                "monotonic_timestamp_ms": 1500 + sequence * 1000,
                "evidence_sha256": digest or f"{sequence:x}" * 64,
                "shot": shot,
                "status": status,
                "observed_at": observed_at,
            }
        ).model_dump(),
        strict=True,
    )


def _observation_bytes(observation: RealtimeShotObservation) -> bytes:
    return json.dumps(
        observation.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_observation_jsonl_decodes_exact_models_and_preserves_exact_replay() -> None:
    observation = _observation()
    line = _observation_bytes(observation)

    decoded = decode_replay_observations(
        line + b"\n" + line + b"\n", expected_session_id=observation.session_id
    )

    assert decoded == (observation, observation)
    assert all(type(item) is RealtimeShotObservation for item in decoded)


@pytest.mark.parametrize(
    "build_bad",
    (
        lambda one, two: one,
        lambda one, two: one + b"\r\n",
        lambda one, two: one + b"\n\n",
        lambda one, two: b" " + one + b"\n",
        lambda one, two: one.replace(b'"schema_version":"1.0"', b'"x":1,"x":2,', 1) + b"\n",
        lambda one, two: one.replace(b'"schema_version":"1.0"', b'"unknown":1,', 1) + b"\n",
        lambda one, two: (
            one.replace(b'"schema_version":"1.0"', b'"schema_version":"2.0"', 1) + b"\n"
        ),
        lambda one, two: (
            one.replace(
                str(_observation(1).session_id).encode(),
                b"00000000-0000-4000-8000-000000000099",
                1,
            )
            + b"\n"
        ),
        lambda one, two: b"x" * 65_537 + b"\n",
        lambda one, two: one + b"\n" + _observation_bytes(_observation(3)) + b"\n",
        lambda one, two: one + b"\n" + _observation_bytes(_observation(1, digest="f" * 64)) + b"\n",
        lambda one, two: (
            one
            + b"\n"
            + _observation_bytes(two.model_copy(update={"event_id": _observation(1).event_id}))
            + b"\n"
        ),
        lambda one, two: b"\xff\n",
    ),
)
def test_observation_jsonl_rejects_noncanonical_gap_conflict_and_identity_reuse(
    build_bad,
) -> None:
    one = _observation_bytes(_observation(1))
    two = _observation(2)
    bad = build_bad(one, two)

    with pytest.raises(ValueError):
        decode_replay_observations(bad, expected_session_id=_observation(1).session_id)


def test_observation_jsonl_rejects_total_size_and_record_count_before_decoding() -> None:
    with pytest.raises(ValueError, match="wire size"):
        decode_replay_observations(
            b"x" * (MAX_REALTIME_OBSERVATIONS_BYTES + 1),
            expected_session_id=_observation().session_id,
        )
    with pytest.raises(ValueError, match="record count"):
        decode_replay_observations(
            b"{}\n" * (MAX_REALTIME_OBSERVATION_RECORDS + 1),
            expected_session_id=_observation().session_id,
        )


def _drill(
    *,
    drill_id: str = "primary",
    abstention_count: int = 2,
    abstention_action: AbstentionAction = AbstentionAction.USE_PUBLISHED_FALLBACK,
    fallback_key: tuple[str, int] | None = ("fallback", 1),
) -> DrillExecutionProjection:
    return DrillExecutionProjection(
        drill_id=drill_id,
        drill_revision=1,
        drill_content_sha256=("1" if drill_id == "primary" else "2") * 64,
        required_capabilities=("TABLE_MM_COORDINATES",),
        initial_difficulty=3,
        minimum_runtime_difficulty=2,
        maximum_runtime_difficulty=4,
        progression_count=2,
        progression_delta=1,
        regression_count=2,
        regression_delta=-1,
        abstention_count=abstention_count,
        abstention_action=abstention_action,
        success_policy=SuccessPolicyProjection(
            evaluator_policy_version="edge-success-v1",
            combination=MetricCombination.ALL,
            metric_rules=(
                MetricRuleProjection(
                    metric_code="POT_SUCCESS_COUNT",
                    target_attempts=2,
                    minimum_successes=1,
                    evaluator_code="POT_RESULT",
                    evaluator_version="1.0",
                    required_capabilities=(),
                ),
            ),
        ),
        replan_reason_codes=("FALLBACK_UNAVAILABLE", "TARGET_OUT_OF_RANGE"),
        replan_continue_action=(
            ReplanContinueAction.USE_PUBLISHED_FALLBACK
            if fallback_key is not None
            else ReplanContinueAction.SAFE_PAUSE
        ),
        fallback_key=fallback_key,
        target_zone=(100, 200, 300, 400),
    )


def _projection(*, safe_pause_on_first_abstention: bool = False, target_repetitions: int = 2):
    primary = _drill(
        abstention_count=1 if safe_pause_on_first_abstention else 2,
        abstention_action=(
            AbstentionAction.SAFE_PAUSE
            if safe_pause_on_first_abstention
            else AbstentionAction.USE_PUBLISHED_FALLBACK
        ),
        fallback_key=None if safe_pause_on_first_abstention else ("fallback", 1),
    )
    fallback = _drill(
        drill_id="fallback",
        abstention_count=1,
        abstention_action=AbstentionAction.SAFE_PAUSE,
        fallback_key=None,
    )
    return _issue_verified_execution_projection(
        admission_id="admission-1",
        bundle_id=UUID("00000000-0000-4000-8000-000000000010"),
        bundle_content_sha256="a" * 64,
        user_id=USER_ID,
        session_id=_observation().session_id,
        plan_id=UUID("00000000-0000-4000-8000-000000000004"),
        plan_version=3,
        target_device_id="edge-1",
        publication_sequence=7,
        expires_at=datetime(2026, 8, 10, 10, tzinfo=UTC),
        plan_valid_until=datetime(2026, 8, 11, tzinfo=UTC),
        authorization_id=UUID("00000000-0000-4000-8000-000000000012"),
        authorization_version=1,
        authorization_expires_at=datetime(2026, 8, 11, tzinfo=UTC),
        signing_key_id="key-1",
        signing_key_version=1,
        signing_key_expires_at=datetime(2026, 9, 1, tzinfo=UTC),
        execution_policy_catalog_version="2026.08",
        execution_policy_catalog_sha256="c" * 64,
        capability_manifest_id="manifest-1",
        capability_manifest_version=1,
        capability_manifest_content_sha256="b" * 64,
        capabilities=("TABLE_MM_COORDINATES",),
        equipment_codes=(),
        blocks=(
            ExecutionBlockProjection(
                block_id=UUID("00000000-0000-4000-8000-000000000006"),
                order=1,
                target_repetitions=target_repetitions,
                drill=primary,
            ),
        ),
        terminal_fallback=fallback,
    )


def _admission(projection) -> VerifiedPlanAdmission:
    return VerifiedPlanAdmission(
        admission_id=projection.admission_id,
        session_id=projection.session_id,
        plan_id=projection.plan_id,
        plan_version=projection.plan_version,
        execution_projection=projection,
    )


def _plan_bytes(projection, *, evidence: bool = True) -> bytes:
    return canonical_replay_plan_fixture_bytes(
        ReplayPlanFixtureV1(
            admission_id=projection.admission_id,
            bundle_id=projection.bundle_id,
            bundle_content_sha256=projection.bundle_content_sha256,
            projection_sha256=projection.projection_sha256,
            session_started_at=datetime(2026, 8, 10, 8, tzinfo=UTC),
            outcome_recorded_at=datetime(2026, 8, 10, 8, 30, tzinfo=UTC),
            session_started_monotonic_ms=1000,
            runtime_version=COACH_RUNTIME_REPLAY_VERSION,
            evidence_lineage=(
                (
                    ReplayEvidenceLineageV1(
                        shot_sequence=1,
                        evidence_envelope_id=UUID("00000000-0000-4000-8000-000000000020"),
                    ),
                    ReplayEvidenceLineageV1(
                        shot_sequence=2,
                        evidence_envelope_id=UUID("00000000-0000-4000-8000-000000000021"),
                    ),
                )
                if evidence
                else ()
            ),
        )
    )


def test_replay_completes_using_existing_runtime_policy_and_event_producer() -> None:
    projection = _projection()
    one = _observation_bytes(_observation(1))
    two = _observation_bytes(_observation(2))
    observations = one + b"\n" + one + b"\n" + two + b"\n"

    first = run_replay(_admission(projection), _plan_bytes(projection), observations)
    second = run_replay(_admission(projection), _plan_bytes(projection), observations)

    assert first == second
    assert canonical_replay_result_bytes(first) == canonical_replay_result_bytes(second)
    assert first.terminal_runtime_state is CoachRuntimeState.COMPLETED
    assert first.input_observation_count == 3
    assert first.consumed_observation_count == 2
    assert first.replayed_observation_count == 1
    assert first.remaining_observation_count == 0
    assert first.plan_outcome.status is PlanOutcomeStatus.COMPLETED
    assert first.plan_outcome.blocks[0].attempted_shots == 2
    assert first.plan_outcome.blocks[0].completed_shots == 2
    assert first.plan_outcome.user_id == USER_ID
    assert first.plan_outcome.direct_ability_update_allowed is False
    assert first.plan_outcome.evidence_envelope_ids == (
        UUID("00000000-0000-4000-8000-000000000020"),
        UUID("00000000-0000-4000-8000-000000000021"),
    )
    assert tuple(event.sequence for event in first.coaching_events) == tuple(
        range(1, len(first.coaching_events) + 1)
    )
    assert first.plan_outcome.outcome_id == UUID("c5735670-a9ec-440f-99e7-d66f50731079")
    assert (
        first.plan_outcome.content_sha256
        == "f74fad2a59e479be5746ae142d8eb06b7dd3e98b442a35c378724c02a39a07a4"
    )
    assert first.content_sha256 == (
        "15115e8224d6cf2a3094d25223452608797d29224dc3ed37c699dced31111b35"
    )


def test_replay_emits_progression_regression_and_counts_failures_as_completed() -> None:
    projection = _projection(target_repetitions=4)
    observations = b"".join(
        _observation_bytes(_observation(sequence, succeeds=sequence <= 2)) + b"\n"
        for sequence in range(1, 5)
    )

    result = run_replay(_admission(projection), _plan_bytes(projection), observations)

    assert result.terminal_runtime_state is CoachRuntimeState.COMPLETED
    assert result.plan_outcome.blocks[0].attempted_shots == 4
    assert result.plan_outcome.blocks[0].completed_shots == 4
    assert {event.event_type.value for event in result.coaching_events} >= {
        "PROGRESSION",
        "REGRESSION",
    }
    assert {event.outcome_status.value for event in result.coaching_events} >= {
        "SUCCESS",
        "FAILURE",
    }


def test_safe_pause_emits_replan_and_does_not_consume_remaining_observation() -> None:
    projection = _projection(safe_pause_on_first_abstention=True)
    observations = (
        _observation_bytes(_observation(1, status=RealtimeObservationStatus.ABSTAINED))
        + b"\n"
        + _observation_bytes(_observation(2))
        + b"\n"
    )

    result = run_replay(_admission(projection), _plan_bytes(projection), observations)

    assert result.terminal_runtime_state is CoachRuntimeState.PAUSED
    assert result.consumed_observation_count == 1
    assert result.remaining_observation_count == 1
    assert len(result.replan_intents) == 1
    assert result.replan_intents[0].reason_code == "FALLBACK_UNAVAILABLE"
    assert result.plan_outcome.status is PlanOutcomeStatus.PARTIAL
    assert result.plan_outcome.blocks[0].attempted_shots == 1
    assert result.plan_outcome.blocks[0].completed_shots == 0
    assert result.plan_outcome.evidence_envelope_ids == (
        UUID("00000000-0000-4000-8000-000000000020"),
    )


def test_published_fallback_continues_then_safe_pauses_without_failure_accounting() -> None:
    projection = _projection()
    observations = b"".join(
        _observation_bytes(_observation(sequence, status=RealtimeObservationStatus.ABSTAINED))
        + b"\n"
        for sequence in (1, 2, 3)
    )

    result = run_replay(_admission(projection), _plan_bytes(projection), observations)

    assert result.terminal_runtime_state is CoachRuntimeState.PAUSED
    assert result.consumed_observation_count == 3
    assert result.plan_outcome.blocks[0].attempted_shots == 3
    assert result.plan_outcome.blocks[0].completed_shots == 0
    assert len(result.replan_intents) == 1
    assert result.replan_intents[0].shot_sequence == 3
    assert all(event.outcome_status.value != "FAILURE" for event in result.coaching_events)


def test_outcome_uses_empty_evidence_lineage_and_exact_fixture_times() -> None:
    projection = _projection(safe_pause_on_first_abstention=True)
    observations = (
        _observation_bytes(_observation(1, status=RealtimeObservationStatus.UNSUPPORTED)) + b"\n"
    )

    result = run_replay(
        _admission(projection),
        _plan_bytes(projection, evidence=False),
        observations,
    )

    assert result.plan_outcome.evidence_envelope_ids == ()
    assert result.plan_outcome.started_at == datetime(2026, 8, 10, 8, tzinfo=UTC)
    assert result.plan_outcome.completed_at == datetime(2026, 8, 10, 8, 30, tzinfo=UTC)
    assert result.plan_outcome.status is PlanOutcomeStatus.PARTIAL


def test_replay_rejects_non_nominal_admission_and_fixture_pin_mismatch() -> None:
    projection = _projection()
    observations = _observation_bytes(_observation()) + b"\n"

    with pytest.raises(TypeError):
        run_replay(object(), _plan_bytes(projection), observations)  # type: ignore[arg-type]
    mismatched = decode_replay_plan_fixture(_plan_bytes(projection)).model_copy(
        update={"projection_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="verified plan admission"):
        run_replay(
            _admission(projection),
            canonical_replay_plan_fixture_bytes(mismatched),
            observations,
        )


def test_replay_rejects_unknown_runtime_version_and_expired_projection() -> None:
    with pytest.raises(ValidationError):
        ReplayPlanFixtureV1(
            **{
                **decode_replay_plan_fixture(PLAN_FIXTURE_BYTES).model_dump(),
                "runtime_version": "future-runtime-v99",
            }
        )
    projection = _projection()
    expired_fixture = decode_replay_plan_fixture(_plan_bytes(projection)).model_copy(
        update={
            "session_started_at": datetime(2030, 1, 1, tzinfo=UTC),
            "outcome_recorded_at": datetime(2030, 1, 1, 1, tzinfo=UTC),
        }
    )
    future_observation = RealtimeShotObservation.model_validate(
        _observation(1)
        .model_copy(
            update={
                "observed_at": datetime(2030, 1, 1, 0, 1, tzinfo=UTC),
                "shot": _observation(1).shot.model_copy(
                    update={"created_at": datetime(2030, 1, 1, 0, 1, tzinfo=UTC)}
                ),
            }
        )
        .model_dump(),
        strict=True,
    )

    with pytest.raises(ValueError, match="validity"):
        run_replay(
            _admission(projection),
            canonical_replay_plan_fixture_bytes(expired_fixture),
            _observation_bytes(future_observation) + b"\n",
        )


def test_result_canonicalizer_rejects_nested_tampering() -> None:
    projection = _projection(safe_pause_on_first_abstention=True)
    result = run_replay(
        _admission(projection),
        _plan_bytes(projection),
        _observation_bytes(_observation(1, status=RealtimeObservationStatus.ABSTAINED)) + b"\n",
    )
    tampered = replace(
        result,
        plan_outcome=result.plan_outcome.model_copy(
            update={"user_id": UUID("00000000-0000-4000-8000-000000000099")}
        ),
    )

    with pytest.raises(ValueError, match="content hash"):
        canonical_replay_result_bytes(tampered)

    with pytest.raises(ValueError, match="status"):
        replace(result, terminal_runtime_state=CoachRuntimeState.COMPLETED)

    foreign_intent = replace(
        result.replan_intents[0],
        admission_id="foreign-admission",
    )
    with pytest.raises(ValueError, match="lineage"):
        canonical_replay_result_bytes(replace(result, replan_intents=(foreign_intent,)))


def test_replay_rejects_observation_outside_explicit_fixture_timeline() -> None:
    projection = _projection()
    before_start = RealtimeShotObservation.model_validate(
        _observation(1)
        .model_copy(update={"observed_at": datetime(2026, 8, 10, 7, 59, tzinfo=UTC)})
        .model_dump(),
        strict=True,
    )

    with pytest.raises(ValueError, match="fixture timeline"):
        run_replay(
            _admission(projection),
            _plan_bytes(projection),
            _observation_bytes(before_start) + b"\n",
        )


def test_replay_rejects_decreasing_monotonic_observation_time() -> None:
    projection = _projection()
    one = _observation(1)
    two = RealtimeShotObservation.model_validate(
        _observation(2)
        .model_copy(
            update={
                "monotonic_timestamp_ms": 2400,
                "shot": _observation(2).shot.model_copy(
                    update={"t_start_ms": 1000, "t_end_ms": 1250}
                ),
            }
        )
        .model_dump(),
        strict=True,
    )

    with pytest.raises(ValueError, match="monotonic"):
        run_replay(
            _admission(projection),
            _plan_bytes(projection),
            _observation_bytes(one) + b"\n" + _observation_bytes(two) + b"\n",
        )


@pytest.mark.parametrize(
    "shot_update",
    (
        {"t_start_ms": 1600, "t_end_ms": 1700},
        {"created_at": datetime(2026, 8, 10, 7, 59, tzinfo=UTC)},
    ),
)
def test_replay_rejects_shot_timeline_outside_observation_and_fixture(shot_update) -> None:
    projection = _projection()
    base = _observation(1)
    invalid = RealtimeShotObservation.model_validate(
        base.model_copy(update={"shot": base.shot.model_copy(update=shot_update)}).model_dump(),
        strict=True,
    )

    with pytest.raises(ValueError, match="shot timeline"):
        run_replay(
            _admission(projection),
            _plan_bytes(projection),
            _observation_bytes(invalid) + b"\n",
        )
