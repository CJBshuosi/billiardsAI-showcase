from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from billiards_ai.edge.coach_runtime import (
    AbstentionAction,
    AcceptObservation,
    AdmitVerifiedPlan,
    BeginCompletion,
    BeginPlanLoad,
    DrillExecutionProjection,
    EvaluationConclusion,
    ExecutionBlockProjection,
    ExecutionDecisionKind,
    ExecutionPolicyState,
    FailRuntime,
    FinishCompletion,
    MetricCombination,
    MetricRuleProjection,
    PauseRuntime,
    PresentFeedback,
    PresentSetup,
    RecoverRuntime,
    ReplanContinueAction,
    ResumeRuntime,
    RuntimeEffectKind,
    RuntimeSnapshot,
    SuccessPolicyProjection,
    VerifiedPlanAdmission,
    WaitForShot,
    apply_observation,
    build_realtime_coaching_event,
    expected_recovery_proof,
    transition,
)
from billiards_ai.edge.coach_runtime.execution_policy import (
    _issue_verified_execution_projection,
)
from billiards_ai.shared.models.evidence import FinalizedShotEvidence
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.overlay import OverlaySemanticRole
from billiards_ai.shared.models.quality import (
    FieldQuality,
    FinalFieldSource,
    ShotAnalysisStatus,
    ShotFieldQuality,
)
from billiards_ai.shared.models.realtime import (
    CoachingDeliveryStatus,
    CoachingEventType,
    CoachingOutcomeStatus,
    RealtimeObservationStatus,
    RealtimeShotObservation,
    canonical_realtime_coaching_event_bytes,
)
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, TargetZoneSource

SESSION_ID = UUID("00000000-0000-4000-8000-000000000002")
PLAN_ID = UUID("00000000-0000-4000-8000-000000000004")
BLOCK_ID = UUID("00000000-0000-4000-8000-000000000006")
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
    *,
    progression_count: int = 2,
    regression_count: int = 2,
    abstention_count: int = 2,
    abstention_action: AbstentionAction = AbstentionAction.USE_PUBLISHED_FALLBACK,
    fallback_key: tuple[str, int] | None = ("fallback", 1),
) -> DrillExecutionProjection:
    return DrillExecutionProjection(
        drill_id="primary",
        drill_revision=1,
        drill_content_sha256="1" * 64,
        required_capabilities=("TABLE_MM_COORDINATES",),
        initial_difficulty=3,
        minimum_runtime_difficulty=2,
        maximum_runtime_difficulty=4,
        progression_count=progression_count,
        progression_delta=1,
        regression_count=regression_count,
        regression_delta=-1,
        abstention_count=abstention_count,
        abstention_action=abstention_action,
        success_policy=SuccessPolicyProjection(
            evaluator_policy_version="edge-success-v1",
            combination=MetricCombination.ALL,
            metric_rules=(
                MetricRuleProjection(
                    metric_code="POT_SUCCESS_COUNT",
                    target_attempts=20,
                    minimum_successes=12,
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


def _projection(*, drill: DrillExecutionProjection | None = None):
    primary = drill or _drill()
    fallback = replace(
        _drill(
            abstention_count=1,
            abstention_action=AbstentionAction.SAFE_PAUSE,
            fallback_key=None,
        ),
        drill_id="fallback",
        drill_content_sha256="2" * 64,
    )
    return _issue_verified_execution_projection(
        admission_id="admission-1",
        bundle_id=UUID("00000000-0000-4000-8000-000000000010"),
        bundle_content_sha256="a" * 64,
        user_id=UUID("00000000-0000-4000-8000-000000000001"),
        session_id=SESSION_ID,
        plan_id=PLAN_ID,
        plan_version=3,
        target_device_id="edge-1",
        publication_sequence=7,
        expires_at=datetime(2026, 8, 10, tzinfo=UTC),
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
                block_id=BLOCK_ID,
                order=1,
                target_repetitions=20,
                drill=primary,
            ),
        ),
        terminal_fallback=fallback,
    )


def _observation(
    *,
    status: RealtimeObservationStatus = RealtimeObservationStatus.OBSERVED,
    digest: str = "d" * 64,
    potted_balls: tuple[int, ...] = (1,),
    cue_ball_end: Point | None = DEFAULT_CUE_BALL_END,
    field_quality: ShotFieldQuality | None = VALID_FIELD_QUALITY,
) -> RealtimeShotObservation:
    confidence = (
        Confidence.ABSTAIN if status is RealtimeObservationStatus.ABSTAINED else Confidence.HIGH
    )
    return RealtimeShotObservation(
        event_id=UUID("00000000-0000-4000-8000-000000000014"),
        session_id=SESSION_ID,
        shot_sequence=1,
        monotonic_timestamp_ms=1500,
        shot=FinalizedShotEvidence(
            shot_id=UUID("00000000-0000-4000-8000-000000000016"),
            session_id=SESSION_ID,
            sequence=1,
            t_start_ms=1000,
            t_end_ms=1500,
            ball_positions_before=(),
            ball_positions_after=(),
            cue_ball_end=cue_ball_end,
            struck_ball=1,
            potted_balls=potted_balls,
            cushion_count=0,
            travel_distance_mm=500,
            is_foul=False,
            target_zone_source=TargetZoneSource.SCENE_PRESET,
            deviation_mm=None,
            confidence=confidence,
            field_quality=field_quality,
            analysis_status=ShotAnalysisStatus.ANALYZED,
            review_status=ReviewStatus.APPROVED,
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
        ),
        status=status,
        evidence_sha256=digest,
        perception_model_version="synthetic-v1",
        calibration_revision=1,
        observed_at=datetime(2026, 8, 10, tzinfo=UTC),
    )


def _admitted_runtime(projection) -> RuntimeSnapshot:
    loading = transition(RuntimeSnapshot.initial(), BeginPlanLoad()).snapshot
    return transition(
        loading,
        AdmitVerifiedPlan(
            VerifiedPlanAdmission(
                admission_id=projection.admission_id,
                session_id=projection.session_id,
                plan_id=projection.plan_id,
                plan_version=projection.plan_version,
                execution_projection=projection,
            )
        ),
    ).snapshot


def _feedback_runtime(projection, observation, *, command_id="present-feedback"):
    runtime = transition(_admitted_runtime(projection), PresentSetup()).snapshot
    runtime = transition(runtime, WaitForShot()).snapshot
    runtime = transition(runtime, AcceptObservation(observation)).snapshot
    command = PresentFeedback(command_id=command_id)
    effect = transition(runtime, command).effects[0]
    return runtime, command, effect


def _map(
    projection,
    effect,
    runtime_before,
    runtime_command,
    *,
    policy_state=None,
    policy_result=None,
    observation=None,
    created_monotonic_ms=1500,
):
    return build_realtime_coaching_event(
        projection,
        effect,
        runtime_before=runtime_before,
        runtime_command=runtime_command,
        sequence=11,
        created_monotonic_ms=created_monotonic_ms,
        policy_state=policy_state,
        policy_result=policy_result,
        observation=observation,
    )


def _feedback(
    state: ExecutionPolicyState,
    observation: RealtimeShotObservation,
):
    result = apply_observation(state, observation)
    runtime_before, runtime_command, effect = _feedback_runtime(state.projection, observation)
    event = _map(
        state.projection,
        effect,
        runtime_before,
        runtime_command,
        created_monotonic_ms=observation.monotonic_timestamp_ms,
        policy_state=state,
        policy_result=result,
        observation=observation,
    )
    return event, result, effect


def test_success_maps_authoritative_target_and_cue_end_with_full_lineage() -> None:
    state = ExecutionPolicyState.start(_projection())
    event, result, effect = _feedback(state, _observation())

    assert result.evaluation.conclusion is EvaluationConclusion.SUCCESS
    assert event.event_type is CoachingEventType.POST_SHOT_FEEDBACK
    assert event.outcome_status is CoachingOutcomeStatus.SUCCESS
    assert event.delivery_status is CoachingDeliveryStatus.NORMAL
    assert event.text_summary == "本杆达到已发布训练标准。"
    assert event.speech is not None and event.speech.text == event.text_summary
    assert event.overlay is not None
    assert tuple(item.semantic_role for item in event.overlay.primitives) == (
        OverlaySemanticRole.ACTUAL_CUE_BALL_END,
        OverlaySemanticRole.TARGET_ZONE,
    )
    assert event.lineage.projection_sha256 == state.projection.projection_sha256
    assert event.lineage.block_id == BLOCK_ID
    assert event.lineage.drill_id == "primary"
    assert event.lineage.shot_id == _observation().shot.shot_id
    assert event.lineage.decision_id == result.decision.decision_id
    assert event.lineage.effect_id == effect.effect_id


def test_untrusted_cue_end_is_omitted_while_published_target_zone_remains() -> None:
    quality = VALID_FIELD_QUALITY.model_copy(
        update={"cue_ball_end": FieldQuality(source=FinalFieldSource.UNKNOWN, confidence=0.9)}
    )
    state = ExecutionPolicyState.start(_projection())
    event, _, _ = _feedback(state, _observation(field_quality=quality))

    assert event.overlay is not None
    assert tuple(item.semantic_role for item in event.overlay.primitives) == (
        OverlaySemanticRole.TARGET_ZONE,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RealtimeObservationStatus.ABSTAINED, CoachingOutcomeStatus.ABSTAINED),
        (RealtimeObservationStatus.UNSUPPORTED, CoachingOutcomeStatus.UNSUPPORTED),
    ],
)
def test_abstained_and_unsupported_are_distinct_never_failure(
    status: RealtimeObservationStatus,
    expected: CoachingOutcomeStatus,
) -> None:
    state = ExecutionPolicyState.start(_projection())
    event, result, _ = _feedback(state, _observation(status=status))

    assert result.evaluation.conclusion.value == expected.value
    assert event.event_type is CoachingEventType.ABSTENTION
    assert event.outcome_status is expected
    assert event.overlay is None
    assert "本杆不计为失败" in event.text_summary
    assert event.speech is not None and "本杆不计为失败" in event.speech.text


@pytest.mark.parametrize(
    ("potted_balls", "progression_count", "regression_count", "event_type", "outcome"),
    [
        ((1,), 1, 2, CoachingEventType.PROGRESSION, CoachingOutcomeStatus.SUCCESS),
        ((), 2, 1, CoachingEventType.REGRESSION, CoachingOutcomeStatus.FAILURE),
    ],
)
def test_difficulty_change_uses_only_the_signed_policy_decision(
    potted_balls: tuple[int, ...],
    progression_count: int,
    regression_count: int,
    event_type: CoachingEventType,
    outcome: CoachingOutcomeStatus,
) -> None:
    projection = _projection(
        drill=_drill(
            progression_count=progression_count,
            regression_count=regression_count,
        )
    )
    state = ExecutionPolicyState.start(projection)
    event, _, _ = _feedback(state, _observation(potted_balls=potted_balls))

    assert event.event_type is event_type
    assert event.outcome_status is outcome


def test_published_fallback_and_safe_pause_are_explicit_and_honest() -> None:
    fallback_state = ExecutionPolicyState.start(_projection(drill=_drill(abstention_count=1)))
    fallback_event, fallback_result, _ = _feedback(
        fallback_state, _observation(status=RealtimeObservationStatus.ABSTAINED)
    )
    assert fallback_result.decision.kind is ExecutionDecisionKind.USE_PUBLISHED_FALLBACK
    assert fallback_event.outcome_status is CoachingOutcomeStatus.ABSTAINED
    assert fallback_event.delivery_status is CoachingDeliveryStatus.NORMAL
    assert "已发布的安全练习" in fallback_event.text_summary

    pause_drill = _drill(
        abstention_count=1,
        abstention_action=AbstentionAction.SAFE_PAUSE,
        fallback_key=None,
    )
    pause_state = ExecutionPolicyState.start(_projection(drill=pause_drill))
    pause_event, pause_result, _ = _feedback(
        pause_state, _observation(status=RealtimeObservationStatus.UNSUPPORTED)
    )
    assert pause_result.decision.kind is ExecutionDecisionKind.SAFE_PAUSE
    assert pause_event.outcome_status is CoachingOutcomeStatus.UNSUPPORTED
    assert pause_event.delivery_status is CoachingDeliveryStatus.DEGRADED
    assert pause_event.event_type is CoachingEventType.SYSTEM_STATUS
    assert "本杆不计为失败" in pause_event.text_summary
    assert "安全暂停" in pause_event.text_summary


def _generic_transition(projection, kind: RuntimeEffectKind):
    ready = _admitted_runtime(projection)
    if kind is RuntimeEffectKind.PLAN_READY:
        before = transition(RuntimeSnapshot.initial(), BeginPlanLoad()).snapshot
        command = AdmitVerifiedPlan(
            VerifiedPlanAdmission(
                projection.admission_id,
                projection.session_id,
                projection.plan_id,
                projection.plan_version,
                projection,
            )
        )
    elif kind is RuntimeEffectKind.PRESENT_SETUP:
        before, command = ready, PresentSetup()
    elif kind is RuntimeEffectKind.WAIT_FOR_SHOT:
        before = transition(ready, PresentSetup()).snapshot
        command = WaitForShot()
    elif kind is RuntimeEffectKind.ANALYZE_SHOT:
        before = transition(transition(ready, PresentSetup()).snapshot, WaitForShot()).snapshot
        command = AcceptObservation(_observation())
    elif kind is RuntimeEffectKind.PAUSED:
        before, command = ready, PauseRuntime("synthetic pause")
    elif kind is RuntimeEffectKind.RESUMED:
        before = transition(ready, PauseRuntime("synthetic pause")).snapshot
        command = ResumeRuntime()
    elif kind in {RuntimeEffectKind.SAFE_PAUSE, RuntimeEffectKind.RECOVERED}:
        waiting = transition(transition(ready, PresentSetup()).snapshot, WaitForShot()).snapshot
        analyzing = transition(waiting, AcceptObservation(_observation())).snapshot
        degraded = transition(
            analyzing,
            AcceptObservation(_observation(digest="9" * 64)),
        ).snapshot
        if kind is RuntimeEffectKind.SAFE_PAUSE:
            before = analyzing
            command = AcceptObservation(_observation(digest="9" * 64))
        else:
            proof = expected_recovery_proof(degraded)
            assert proof is not None
            before = degraded
            command = RecoverRuntime(SESSION_ID, 1, proof)
    elif kind is RuntimeEffectKind.COMPLETE_SESSION:
        before, command = ready, BeginCompletion()
    elif kind is RuntimeEffectKind.SESSION_COMPLETED:
        before = transition(ready, BeginCompletion()).snapshot
        command = FinishCompletion()
    elif kind is RuntimeEffectKind.RUNTIME_FAILED:
        before, command = ready, FailRuntime("synthetic-failure")
    else:  # pragma: no cover - test helper protects its own table
        raise AssertionError(kind)
    result = transition(before, command)
    assert result.error is None and len(result.effects) == 1
    return before, command, result.effects[0]


@pytest.mark.parametrize(
    ("kind", "event_type", "delivery", "text"),
    [
        (
            RuntimeEffectKind.PLAN_READY,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.NORMAL,
            "可以开始",
        ),
        (
            RuntimeEffectKind.PRESENT_SETUP,
            CoachingEventType.SETUP,
            CoachingDeliveryStatus.NORMAL,
            "训练设置",
        ),
        (
            RuntimeEffectKind.WAIT_FOR_SHOT,
            CoachingEventType.PRE_SHOT_CUE,
            CoachingDeliveryStatus.NORMAL,
            "击球",
        ),
        (
            RuntimeEffectKind.ANALYZE_SHOT,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.NORMAL,
            "分析",
        ),
        (
            RuntimeEffectKind.PAUSED,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.NORMAL,
            "已暂停",
        ),
        (
            RuntimeEffectKind.RESUMED,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.NORMAL,
            "已恢复",
        ),
        (
            RuntimeEffectKind.SAFE_PAUSE,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.DEGRADED,
            "安全暂停",
        ),
        (
            RuntimeEffectKind.RECOVERED,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.NORMAL,
            "验证并恢复",
        ),
        (
            RuntimeEffectKind.COMPLETE_SESSION,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.NORMAL,
            "正在完成",
        ),
        (
            RuntimeEffectKind.SESSION_COMPLETED,
            CoachingEventType.SESSION_SUMMARY,
            CoachingDeliveryStatus.NORMAL,
            "已完成",
        ),
        (
            RuntimeEffectKind.RUNTIME_FAILED,
            CoachingEventType.SYSTEM_STATUS,
            CoachingDeliveryStatus.ERROR,
            "运行时故障",
        ),
    ],
)
def test_generic_templates_are_bound_to_real_runtime_transitions(
    kind: RuntimeEffectKind,
    event_type: CoachingEventType,
    delivery: CoachingDeliveryStatus,
    text: str,
) -> None:
    projection = _projection()
    before, command, effect = _generic_transition(projection, kind)
    event = _map(projection, effect, before, command)

    assert event.event_type is event_type
    assert event.outcome_status is CoachingOutcomeStatus.INFORMATIONAL
    assert event.delivery_status is delivery
    assert text in event.text_summary
    assert event.speech is not None and text in event.speech.text


def test_same_inputs_replay_to_identical_bytes_and_identity() -> None:
    state = ExecutionPolicyState.start(_projection())
    observation = _observation()
    result = apply_observation(state, observation)
    before, command, effect = _feedback_runtime(state.projection, observation)
    arguments = dict(
        policy_state=state,
        policy_result=result,
        observation=observation,
    )
    first = _map(state.projection, effect, before, command, **arguments)
    second = _map(state.projection, effect, before, command, **arguments)
    changed_command = PresentFeedback(command_id="changed-feedback")
    changed_effect = transition(before, changed_command).effects[0]
    changed = _map(state.projection, changed_effect, before, changed_command, **arguments)

    assert first.event_id == second.event_id
    assert first.idempotency_key == second.idempotency_key
    assert canonical_realtime_coaching_event_bytes(first) == (
        canonical_realtime_coaching_event_bytes(second)
    )
    assert changed.event_id != first.event_id


@dataclass(frozen=True)
class StructuralState:
    projection: object


def test_mapper_rejects_structural_fakes_and_lineage_mismatch() -> None:
    state = ExecutionPolicyState.start(_projection())
    observation = _observation()
    before, command, effect = _feedback_runtime(state.projection, observation)
    with pytest.raises(TypeError, match="exact"):
        _map(
            StructuralState(state.projection),
            effect,
            before,
            command,
        )
    with pytest.raises(ValueError, match="transition"):
        _map(
            state.projection,
            replace(effect, effect_id="9" * 64),
            before,
            command,
        )

    result = apply_observation(state, observation)
    forged = replace(result, state=replace(result.state, difficulty=2))
    with pytest.raises(ValueError, match="verified inputs"):
        _map(
            state.projection,
            effect,
            before,
            command,
            policy_state=state,
            policy_result=forged,
            observation=observation,
        )


def test_pre_admission_effect_is_rejected_without_inventing_full_lineage() -> None:
    projection = _projection()
    before = RuntimeSnapshot.initial()
    command = BeginPlanLoad()
    effect = transition(before, command).effects[0]
    with pytest.raises(ValueError, match="complete admitted"):
        _map(projection, effect, before, command)


def test_off_table_cue_end_is_omitted_instead_of_breaking_event() -> None:
    state = ExecutionPolicyState.start(_projection())
    event, _, _ = _feedback(
        state,
        _observation(cue_ball_end=Point(x_mm=3000, y_mm=350)),
    )
    assert event.overlay is not None
    assert tuple(item.semantic_role for item in event.overlay.primitives) == (
        OverlaySemanticRole.TARGET_ZONE,
    )


def test_mapper_rejects_exact_state_with_structural_projection() -> None:
    projection = _projection()
    state = ExecutionPolicyState.start(projection)
    observation = _observation()
    before, command, effect = _feedback_runtime(projection, observation)
    fake_state = replace(state, projection=StructuralState(projection))
    with pytest.raises(ValueError, match="verified projection"):
        _map(
            projection,
            effect,
            before,
            command,
            policy_state=fake_state,
            policy_result=apply_observation(state, observation),
            observation=observation,
        )
