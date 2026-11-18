from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from billiards_ai.edge.coach_runtime.execution_policy import (
    AbstentionAction,
    DrillExecutionProjection,
    EvaluationConclusion,
    ExecutionBlockProjection,
    ExecutionDecisionKind,
    ExecutionPolicyState,
    MetricCombination,
    MetricRuleProjection,
    ReplanContinueAction,
    SuccessPolicyProjection,
    VerifiedExecutionProjection,
    _issue_verified_execution_projection,
    apply_observation,
    evaluate_observation,
    evaluator_registry_compatible,
)
from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
from billiards_ai.shared.models.evidence import FinalizedShotEvidence
from billiards_ai.shared.models.geometry import Point
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
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, TargetZoneSource

SESSION_ID = UUID("00000000-0000-4000-8000-000000000002")
PLAN_ID = UUID("00000000-0000-4000-8000-000000000004")
DEFAULT_CUE_BALL_END = Point(x_mm=150, y_mm=350)
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


def _drill(
    drill_id: str,
    *,
    evaluator_code: str = "POT_RESULT",
    combination: MetricCombination = MetricCombination.ALL,
    fallback_key: tuple[str, int] | None = ("fallback", 1),
    abstention_action: AbstentionAction = AbstentionAction.USE_PUBLISHED_FALLBACK,
) -> DrillExecutionProjection:
    metric_code = {
        "POT_RESULT": "POT_SUCCESS_COUNT",
        "TARGET_ZONE_RESULT": "TARGET_ZONE_SUCCESS_COUNT",
        "BANK_POT_RESULT": "BANK_POT_SUCCESS_COUNT",
    }.get(evaluator_code, "UNKNOWN_SUCCESS_COUNT")
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
        abstention_count=2,
        abstention_action=abstention_action,
        success_policy=SuccessPolicyProjection(
            evaluator_policy_version="edge-success-v1",
            combination=combination,
            metric_rules=(
                MetricRuleProjection(
                    metric_code=metric_code,
                    target_attempts=20,
                    minimum_successes=12,
                    evaluator_code=evaluator_code,
                    evaluator_version="1.0",
                    required_capabilities=("TABLE_MM_COORDINATES",),
                ),
            ),
        ),
        replan_reason_codes=("FALLBACK_UNAVAILABLE", "TARGET_OUT_OF_RANGE"),
        replan_continue_action=(
            ReplanContinueAction.USE_PUBLISHED_FALLBACK
            if fallback_key
            else ReplanContinueAction.SAFE_PAUSE
        ),
        fallback_key=fallback_key,
        target_zone=(100, 200, 300, 400),
    )


def _projection(*, primary: DrillExecutionProjection | None = None) -> VerifiedExecutionProjection:
    primary = primary or _drill("primary")
    fallback = _drill(
        "fallback",
        fallback_key=None,
        abstention_action=AbstentionAction.SAFE_PAUSE,
    )
    return _issue_verified_execution_projection(
        admission_id="bundle:verified",
        bundle_id=UUID("00000000-0000-4000-8000-000000000099"),
        bundle_content_sha256="a" * 64,
        user_id=UUID("00000000-0000-4000-8000-000000000001"),
        session_id=SESSION_ID,
        plan_id=PLAN_ID,
        plan_version=3,
        target_device_id="edge-1",
        publication_sequence=7,
        expires_at=datetime(2026, 8, 10, tzinfo=UTC),
        plan_valid_until=datetime(2026, 8, 11, tzinfo=UTC),
        authorization_id=UUID("00000000-0000-4000-8000-000000000088"),
        authorization_version=2,
        authorization_expires_at=datetime(2026, 8, 11, tzinfo=UTC),
        signing_key_id="key-1",
        signing_key_version=1,
        signing_key_expires_at=datetime(2026, 9, 1, tzinfo=UTC),
        execution_policy_catalog_version="2026.08",
        execution_policy_catalog_sha256="c" * 64,
        capability_manifest_id="manifest-4",
        capability_manifest_version=4,
        capability_manifest_content_sha256="b" * 64,
        capabilities=("TABLE_MM_COORDINATES",),
        equipment_codes=(),
        blocks=(
            ExecutionBlockProjection(
                block_id=UUID("00000000-0000-4000-8000-000000000066"),
                order=1,
                target_repetitions=20,
                drill=primary,
            ),
        ),
        terminal_fallback=fallback,
    )


def _observation(
    sequence: int,
    *,
    status: RealtimeObservationStatus = RealtimeObservationStatus.OBSERVED,
    struck_ball: int | None = 1,
    potted_balls: tuple[int, ...] = (1,),
    cue_ball_end: Point | None = DEFAULT_CUE_BALL_END,
    is_foul: bool | None = False,
    field_quality: ShotFieldQuality | None = VALID_FIELD_QUALITY,
    target_zone_source: TargetZoneSource = TargetZoneSource.SCENE_PRESET,
    digest: str | None = None,
) -> RealtimeShotObservation:
    confidence = (
        Confidence.ABSTAIN if status is RealtimeObservationStatus.ABSTAINED else Confidence.HIGH
    )
    shot = FinalizedShotEvidence(
        shot_id=UUID(f"00000000-0000-4000-8000-{sequence:012d}"),
        session_id=SESSION_ID,
        sequence=sequence,
        t_start_ms=sequence * 1000,
        t_end_ms=sequence * 1000 + 500,
        ball_positions_before=(),
        ball_positions_after=(),
        cue_ball_end=cue_ball_end,
        struck_ball=struck_ball,
        potted_balls=potted_balls,
        cushion_count=2,
        travel_distance_mm=500,
        is_foul=is_foul,
        target_zone_source=target_zone_source,
        deviation_mm=None,
        confidence=confidence,
        field_quality=field_quality,
        analysis_status=ShotAnalysisStatus.ANALYZED,
        review_status=ReviewStatus.APPROVED,
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    return RealtimeShotObservation(
        event_id=UUID(f"10000000-0000-4000-8000-{sequence:012d}"),
        session_id=SESSION_ID,
        shot_sequence=sequence,
        monotonic_timestamp_ms=sequence * 1000 + 500,
        shot=shot,
        status=status,
        evidence_sha256=digest or f"{sequence:x}" * 64,
        perception_model_version="synthetic-v1",
        calibration_revision=1,
        observed_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


def test_versioned_evaluators_are_strict_and_raw_free() -> None:
    projection = _projection()
    drill = projection.blocks[0].drill

    assert (
        evaluate_observation(projection, drill, _observation(1)).conclusion
        is EvaluationConclusion.SUCCESS
    )
    assert (
        evaluate_observation(projection, drill, _observation(1, potted_balls=())).conclusion
        is EvaluationConclusion.FAILURE
    )

    zone = _drill("primary", evaluator_code="TARGET_ZONE_RESULT")
    assert (
        evaluate_observation(_projection(primary=zone), zone, _observation(1)).conclusion
        is EvaluationConclusion.SUCCESS
    )
    assert (
        evaluate_observation(
            _projection(primary=zone), zone, _observation(1, cue_ball_end=Point(x_mm=99, y_mm=350))
        ).conclusion
        is EvaluationConclusion.FAILURE
    )

    bank = _drill("primary", evaluator_code="BANK_POT_RESULT")
    assert (
        evaluate_observation(_projection(primary=bank), bank, _observation(1)).conclusion
        is EvaluationConclusion.UNSUPPORTED
    )
    unknown = _drill("primary", evaluator_code="UNPUBLISHED_EVALUATOR")
    assert (
        evaluate_observation(_projection(primary=unknown), unknown, _observation(1)).conclusion
        is EvaluationConclusion.UNSUPPORTED
    )
    assert (
        evaluate_observation(
            projection,
            drill,
            _observation(1, struck_ball=None),
        ).conclusion
        is EvaluationConclusion.UNSUPPORTED
    )


@pytest.mark.parametrize("field_name", ["struck_ball", "potted_balls", "is_foul"])
@pytest.mark.parametrize(
    "required_quality",
    [
        FieldQuality(source=FinalFieldSource.UNKNOWN, confidence=0.9),
        FieldQuality(
            source=FinalFieldSource.CV,
            confidence=CONFIDENCE_ABSTAIN_BELOW - 0.01,
        ),
    ],
)
def test_pot_requires_authoritative_quality_for_every_required_field(
    field_name: str,
    required_quality: FieldQuality,
) -> None:
    projection = _projection()
    drill = projection.blocks[0].drill
    quality = VALID_FIELD_QUALITY.model_copy(update={field_name: required_quality})

    result = evaluate_observation(
        projection,
        drill,
        _observation(1, field_quality=quality),
    )

    assert result.conclusion is EvaluationConclusion.UNSUPPORTED


def test_pot_requires_complete_field_quality() -> None:
    projection = _projection()
    drill = projection.blocks[0].drill

    result = evaluate_observation(
        projection,
        drill,
        _observation(1, field_quality=None),
    )

    assert result.conclusion is EvaluationConclusion.UNSUPPORTED


@pytest.mark.parametrize(
    ("target_zone_source", "field_quality"),
    [
        (TargetZoneSource.NONE, VALID_FIELD_QUALITY),
        (TargetZoneSource.POST_HOC_INFERRED, VALID_FIELD_QUALITY),
        (TargetZoneSource.SCENE_PRESET, None),
        (
            TargetZoneSource.SCENE_PRESET,
            VALID_FIELD_QUALITY.model_copy(
                update={
                    "cue_ball_end": FieldQuality(
                        source=FinalFieldSource.UNKNOWN,
                        confidence=0.9,
                    )
                }
            ),
        ),
        (
            TargetZoneSource.SCENE_PRESET,
            VALID_FIELD_QUALITY.model_copy(
                update={
                    "cue_ball_end": FieldQuality(
                        source=FinalFieldSource.CV,
                        confidence=CONFIDENCE_ABSTAIN_BELOW - 0.01,
                    )
                }
            ),
        ),
    ],
)
def test_target_zone_requires_scene_preset_and_authoritative_cue_end_quality(
    target_zone_source: TargetZoneSource,
    field_quality: ShotFieldQuality | None,
) -> None:
    drill = _drill("primary", evaluator_code="TARGET_ZONE_RESULT")
    projection = _projection(primary=drill)

    result = evaluate_observation(
        projection,
        drill,
        _observation(
            1,
            target_zone_source=target_zone_source,
            field_quality=field_quality,
        ),
    )

    assert result.conclusion is EvaluationConclusion.UNSUPPORTED


def test_required_field_quality_threshold_is_inclusive() -> None:
    threshold = FieldQuality(
        source=FinalFieldSource.CV,
        confidence=CONFIDENCE_ABSTAIN_BELOW,
    )
    pot_quality = VALID_FIELD_QUALITY.model_copy(
        update={name: threshold for name in ("struck_ball", "potted_balls", "is_foul")}
    )
    pot_projection = _projection()
    assert (
        evaluate_observation(
            pot_projection,
            pot_projection.blocks[0].drill,
            _observation(1, field_quality=pot_quality),
        ).conclusion
        is EvaluationConclusion.SUCCESS
    )

    zone = _drill("primary", evaluator_code="TARGET_ZONE_RESULT")
    zone_quality = VALID_FIELD_QUALITY.model_copy(update={"cue_ball_end": threshold})
    assert (
        evaluate_observation(
            _projection(primary=zone),
            zone,
            _observation(1, field_quality=zone_quality),
        ).conclusion
        is EvaluationConclusion.SUCCESS
    )


@pytest.mark.parametrize(
    "status", [RealtimeObservationStatus.ABSTAINED, RealtimeObservationStatus.UNSUPPORTED]
)
def test_no_conclusion_never_counts_as_failure(status: RealtimeObservationStatus) -> None:
    state = ExecutionPolicyState.start(_projection())
    result = apply_observation(state, _observation(1, status=status))
    assert result.state.success_streak == 0
    assert result.state.failure_streak == 0
    assert result.state.abstention_streak == 1
    assert result.evaluation.conclusion.value == status.value


def test_progression_regression_clamp_and_streak_reset_are_session_local() -> None:
    state = ExecutionPolicyState.start(_projection())
    first = apply_observation(state, _observation(1))
    progressed = apply_observation(first.state, _observation(2))
    assert progressed.state.difficulty == 4
    assert progressed.state.success_streak == 0
    failed = apply_observation(progressed.state, _observation(3, potted_balls=()))
    regressed = apply_observation(failed.state, _observation(4, potted_balls=()))
    assert regressed.state.difficulty == 3
    assert regressed.state.failure_streak == 0
    capped = apply_observation(
        apply_observation(regressed.state, _observation(5)).state, _observation(6)
    )
    assert capped.state.difficulty == 4
    assert state.difficulty == 3


def test_new_block_state_resets_streaks_and_replays_deterministically() -> None:
    projection = _projection()
    first = projection.blocks[0]
    values = {
        name: getattr(projection, name)
        for name in (
            "admission_id",
            "bundle_id",
            "bundle_content_sha256",
            "user_id",
            "session_id",
            "plan_id",
            "plan_version",
            "target_device_id",
            "publication_sequence",
            "expires_at",
            "plan_valid_until",
            "authorization_id",
            "authorization_version",
            "authorization_expires_at",
            "signing_key_id",
            "signing_key_version",
            "signing_key_expires_at",
            "execution_policy_catalog_version",
            "execution_policy_catalog_sha256",
            "capability_manifest_id",
            "capability_manifest_version",
            "capability_manifest_content_sha256",
            "capabilities",
            "equipment_codes",
            "terminal_fallback",
        )
    }
    projection = _issue_verified_execution_projection(
        **values,
        blocks=(
            first,
            replace(
                first,
                block_id=UUID("00000000-0000-4000-8000-000000000065"),
                order=2,
            ),
        ),
    )
    state = ExecutionPolicyState.start(projection, block_index=1)
    assert (state.success_streak, state.failure_streak, state.abstention_streak) == (0, 0, 0)
    assert apply_observation(state, _observation(1)) == apply_observation(
        ExecutionPolicyState.start(projection, 1), _observation(1)
    )


def test_signed_block_horizon_advances_and_preserves_global_sequence() -> None:
    projection = _projection()
    first = replace(projection.blocks[0], target_repetitions=2)
    second = replace(
        first,
        block_id=UUID("00000000-0000-4000-8000-000000000064"),
        order=2,
        drill=replace(first.drill, drill_id="primary-two", drill_content_sha256="3" * 64),
    )
    projection = _issue_verified_execution_projection(
        **{
            field.name: getattr(projection, field.name)
            for field in fields(projection)
            if field.name not in {"_issuer", "schema_version", "projection_sha256", "blocks"}
        },
        blocks=(first, second),
    )
    state = ExecutionPolicyState.start(projection, expected_shot_sequence=7)
    one = apply_observation(state, _observation(7))
    advanced = apply_observation(one.state, _observation(8))
    assert advanced.decision.next_block_index == 1
    assert advanced.state.block_index == 1
    assert advanced.state.expected_shot_sequence == 9
    assert advanced.state.current_drill_key == second.drill.key
    assert advanced.state.completed_repetitions == 0
    assert (advanced.state.success_streak, advanced.state.failure_streak) == (0, 0)


def test_last_signed_block_horizon_requests_10_2_completion() -> None:
    projection = _projection()
    projection = _issue_verified_execution_projection(
        **{
            field.name: getattr(projection, field.name)
            for field in fields(projection)
            if field.name not in {"_issuer", "schema_version", "projection_sha256", "blocks"}
        },
        blocks=(replace(projection.blocks[0], target_repetitions=2),),
    )
    state = ExecutionPolicyState.start(projection)
    state = apply_observation(state, _observation(1)).state
    result = apply_observation(state, _observation(2))
    assert result.decision.kind is ExecutionDecisionKind.BEGIN_COMPLETION
    assert result.state.session_complete
    with pytest.raises(ValueError, match="completed"):
        apply_observation(result.state, _observation(3))


def test_projection_tampering_is_rejected_at_policy_seam() -> None:
    projection = _projection()
    forged = replace(
        projection,
        blocks=(
            replace(
                projection.blocks[0],
                drill=replace(projection.blocks[0].drill, progression_count=1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="integrity"):
        ExecutionPolicyState.start(forged)
    with pytest.raises(ValueError, match="integrity"):
        ExecutionPolicyState.start(replace(projection, _issuer=object()))


def test_bound_exhaustion_executes_signed_replan_action() -> None:
    state = ExecutionPolicyState.start(_projection())
    for sequence in range(1, 4):
        state = apply_observation(state, _observation(sequence)).state
    result = apply_observation(state, _observation(4))
    assert result.decision.kind is ExecutionDecisionKind.USE_PUBLISHED_FALLBACK
    assert result.state.fallback_active
    assert result.decision.replan_intent is not None
    assert result.decision.replan_intent.reason_code == "TARGET_OUT_OF_RANGE"


def test_abstention_breaks_consecutive_supported_streaks() -> None:
    state = apply_observation(ExecutionPolicyState.start(_projection()), _observation(1)).state
    abstained = apply_observation(
        state, _observation(2, status=RealtimeObservationStatus.ABSTAINED)
    )
    assert (abstained.state.success_streak, abstained.state.failure_streak) == (0, 0)


def test_replay_after_fallback_returns_original_evaluation() -> None:
    state = ExecutionPolicyState.start(_projection())
    first_observation = _observation(1, status=RealtimeObservationStatus.ABSTAINED)
    first = apply_observation(state, first_observation)
    fallback = apply_observation(
        first.state,
        _observation(2, status=RealtimeObservationStatus.ABSTAINED),
    )
    replay = apply_observation(fallback.state, first_observation)
    assert replay.replayed
    assert replay.evaluation == first.evaluation


@pytest.mark.parametrize("combination", [MetricCombination.ALL, MetricCombination.ANY])
def test_multi_metric_policy_never_masks_unsupported(
    combination: MetricCombination,
) -> None:
    drill = _drill("primary", combination=combination)
    supported = drill.success_policy.metric_rules[0]
    unsupported = replace(
        supported,
        metric_code="UNKNOWN_SUCCESS_COUNT",
        evaluator_code="UNKNOWN_RESULT",
    )
    drill = replace(
        drill,
        success_policy=replace(
            drill.success_policy,
            metric_rules=(supported, unsupported),
        ),
    )
    projection = _projection(primary=drill)
    assert (
        evaluate_observation(projection, drill, _observation(1)).conclusion
        is EvaluationConclusion.UNSUPPORTED
    )


def test_frozen_registry_rejects_unknown_policy_before_execution() -> None:
    unknown = _drill("primary", evaluator_code="UNPUBLISHED_EVALUATOR")
    assert evaluator_registry_compatible(_projection(primary=unknown)) is False


def test_abstention_switches_to_signed_terminal_fallback_then_safe_pauses() -> None:
    state = ExecutionPolicyState.start(_projection())
    one = apply_observation(state, _observation(1, status=RealtimeObservationStatus.ABSTAINED))
    fallback = apply_observation(
        one.state, _observation(2, status=RealtimeObservationStatus.UNSUPPORTED)
    )
    assert fallback.decision.kind is ExecutionDecisionKind.USE_PUBLISHED_FALLBACK
    assert fallback.state.fallback_active
    assert fallback.state.current_drill_key == ("fallback", 1)
    three = apply_observation(
        fallback.state, _observation(3, status=RealtimeObservationStatus.ABSTAINED)
    )
    paused = apply_observation(
        three.state, _observation(4, status=RealtimeObservationStatus.ABSTAINED)
    )
    assert paused.decision.kind is ExecutionDecisionKind.SAFE_PAUSE
    assert paused.state.safe_paused
    assert paused.decision.replan_intent is not None
    assert paused.decision.replan_intent.reason_code == "FALLBACK_UNAVAILABLE"


def test_exact_replay_is_idempotent_and_conflicting_digest_is_rejected() -> None:
    state = ExecutionPolicyState.start(_projection())
    observation = _observation(1)
    first = apply_observation(state, observation)
    replay = apply_observation(first.state, observation)
    assert replay.replayed
    assert replay.state == first.state
    assert replay.decision == first.decision
    with pytest.raises(ValueError, match="conflicting observation"):
        apply_observation(first.state, _observation(1, digest="f" * 64))


def test_structural_projection_and_wrong_session_cannot_execute() -> None:
    projection = _projection()
    state = ExecutionPolicyState.start(projection)
    fake = (
        projection.__dict__
        if hasattr(projection, "__dict__")
        else {"projection_sha256": projection.projection_sha256}
    )
    with pytest.raises(TypeError):
        ExecutionPolicyState.start(fake)  # type: ignore[arg-type]
    wrong = _observation(1).model_copy(
        update={"session_id": UUID("00000000-0000-4000-8000-000000000003")}
    )
    with pytest.raises(ValueError, match="session"):
        apply_observation(state, wrong)
