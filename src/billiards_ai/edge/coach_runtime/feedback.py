"""Deterministic mapping from verified runtime effects and policy decisions to shared v2 events."""

from __future__ import annotations

from billiards_ai.edge.coach_runtime.execution_policy import (
    EvaluationConclusion,
    ExecutionDecisionKind,
    ExecutionPolicyResult,
    ExecutionPolicyState,
    VerifiedExecutionProjection,
    apply_observation,
)
from billiards_ai.edge.coach_runtime.state_machine import (
    AcceptObservation,
    AdmitVerifiedPlan,
    AdvanceToNextShot,
    BeginCompletion,
    BeginPlanLoad,
    FailRuntime,
    FinishCompletion,
    PauseRuntime,
    PresentFeedback,
    PresentSetup,
    RecoverRuntime,
    ResumeRuntime,
    RuntimeCommand,
    RuntimeEffect,
    RuntimeEffectKind,
    RuntimeSnapshot,
    WaitForShot,
    transition,
)
from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
from billiards_ai.shared.models.overlay import (
    CoachingDeliveryStatus,
    CoachingOutcomeStatus,
    OverlayCoordinateSpace,
    OverlayLineStyle,
    OverlayPayloadV1,
    OverlayPrimitiveKind,
    OverlayPrimitiveV1,
    OverlaySemanticRole,
    TablePointMmV1,
    TableZoneMmV1,
)
from billiards_ai.shared.models.quality import FinalFieldSource
from billiards_ai.shared.models.realtime import (
    CoachingEventType,
    RealtimeCoachingEventV2,
    RealtimeCoachingLineageV1,
    RealtimeShotObservation,
    SpeechCueV1,
    issue_realtime_coaching_event_v2,
)
from billiards_ai.shared.models.shot import TargetZoneSource

_GENERIC_FEEDBACK: dict[
    RuntimeEffectKind,
    tuple[CoachingEventType, CoachingDeliveryStatus, str, str],
] = {
    RuntimeEffectKind.PLAN_READY: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.NORMAL,
        "plan-ready",
        "训练计划已验证，可以开始。",
    ),
    RuntimeEffectKind.PRESENT_SETUP: (
        CoachingEventType.SETUP,
        CoachingDeliveryStatus.NORMAL,
        "present-setup",
        "请按已发布训练设置准备下一杆。",
    ),
    RuntimeEffectKind.WAIT_FOR_SHOT: (
        CoachingEventType.PRE_SHOT_CUE,
        CoachingDeliveryStatus.NORMAL,
        "wait-for-shot",
        "准备好后请击球。",
    ),
    RuntimeEffectKind.ANALYZE_SHOT: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.NORMAL,
        "analyze-shot",
        "正在分析本杆。",
    ),
    RuntimeEffectKind.PAUSED: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.NORMAL,
        "runtime-paused",
        "训练已暂停。",
    ),
    RuntimeEffectKind.RESUMED: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.NORMAL,
        "runtime-resumed",
        "训练已恢复。",
    ),
    RuntimeEffectKind.SAFE_PAUSE: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.DEGRADED,
        "safe-pause",
        "训练指导已安全暂停。",
    ),
    RuntimeEffectKind.RECOVERED: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.NORMAL,
        "runtime-recovered",
        "训练状态已验证并恢复。",
    ),
    RuntimeEffectKind.COMPLETE_SESSION: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.NORMAL,
        "session-completing",
        "正在完成本次训练。",
    ),
    RuntimeEffectKind.SESSION_COMPLETED: (
        CoachingEventType.SESSION_SUMMARY,
        CoachingDeliveryStatus.NORMAL,
        "session-completed",
        "本次训练已完成。",
    ),
    RuntimeEffectKind.RUNTIME_FAILED: (
        CoachingEventType.SYSTEM_STATUS,
        CoachingDeliveryStatus.ERROR,
        "runtime-failed",
        "训练指导发生运行时故障，已停止输出指导。",
    ),
}

_EXACT_RUNTIME_COMMAND_TYPES = (
    BeginPlanLoad,
    AdmitVerifiedPlan,
    PresentSetup,
    WaitForShot,
    AcceptObservation,
    PresentFeedback,
    AdvanceToNextShot,
    PauseRuntime,
    ResumeRuntime,
    RecoverRuntime,
    BeginCompletion,
    FinishCompletion,
    FailRuntime,
)


def _current_drill(state: ExecutionPolicyState):
    return (
        state.projection.terminal_fallback
        if state.fallback_active
        else state.projection.blocks[state.block_index].drill
    )


def _target_primitive(state: ExecutionPolicyState) -> OverlayPrimitiveV1 | None:
    zone = _current_drill(state).target_zone
    if zone is None:
        return None
    return OverlayPrimitiveV1(
        kind=OverlayPrimitiveKind.ZONE,
        semantic_role=OverlaySemanticRole.TARGET_ZONE,
        label_code="published-target-zone",
        line_style=OverlayLineStyle.DASHED,
        zone=TableZoneMmV1(
            minimum_x_mm=zone[0],
            maximum_x_mm=zone[1],
            minimum_y_mm=zone[2],
            maximum_y_mm=zone[3],
        ),
    )


def _cue_end_primitive(observation: RealtimeShotObservation) -> OverlayPrimitiveV1 | None:
    quality = observation.shot.field_quality
    point = observation.shot.cue_ball_end
    if (
        observation.shot.target_zone_source is not TargetZoneSource.SCENE_PRESET
        or quality is None
        or quality.cue_ball_end.source is FinalFieldSource.UNKNOWN
        or quality.cue_ball_end.confidence < CONFIDENCE_ABSTAIN_BELOW
        or point is None
        or not point.is_on_table()
    ):
        return None
    return OverlayPrimitiveV1(
        kind=OverlayPrimitiveKind.POINT,
        semantic_role=OverlaySemanticRole.ACTUAL_CUE_BALL_END,
        label_code="actual-cue-ball-end",
        line_style=OverlayLineStyle.SOLID,
        point=TablePointMmV1(x_mm=point.x_mm, y_mm=point.y_mm),
    )


def _overlay(
    state: ExecutionPolicyState,
    outcome: CoachingOutcomeStatus,
    delivery: CoachingDeliveryStatus,
    observation: RealtimeShotObservation | None,
) -> OverlayPayloadV1 | None:
    if outcome in {CoachingOutcomeStatus.ABSTAINED, CoachingOutcomeStatus.UNSUPPORTED}:
        return None
    primitives = tuple(
        item
        for item in (
            None if observation is None else _cue_end_primitive(observation),
            _target_primitive(state),
        )
        if item is not None
    )
    if not primitives:
        return None
    return OverlayPayloadV1(
        coordinate_space=OverlayCoordinateSpace.TABLE_MM,
        outcome_status=outcome,
        delivery_status=delivery,
        primitives=primitives,
    )


def _policy_feedback(
    state: ExecutionPolicyState,
    result: ExecutionPolicyResult,
) -> tuple[CoachingEventType, CoachingOutcomeStatus, CoachingDeliveryStatus, str, str]:
    conclusion = result.evaluation.conclusion
    outcome = CoachingOutcomeStatus(conclusion.value)
    delivery = CoachingDeliveryStatus.NORMAL
    if conclusion is EvaluationConclusion.SUCCESS:
        event_type = CoachingEventType.POST_SHOT_FEEDBACK
        cue_code = "shot-success"
        text = "本杆达到已发布训练标准。"
    elif conclusion is EvaluationConclusion.FAILURE:
        event_type = CoachingEventType.POST_SHOT_FEEDBACK
        cue_code = "shot-failure"
        text = "本杆未达到已发布训练标准，请按当前训练设置继续。"
    elif conclusion is EvaluationConclusion.ABSTAINED:
        event_type = CoachingEventType.ABSTENTION
        cue_code = "shot-abstained"
        text = "本杆未获得可靠分析结果，本杆不计为失败。"
    else:
        event_type = CoachingEventType.ABSTENTION
        cue_code = "shot-unsupported"
        text = "当前证据不支持可靠判断，本杆不计为失败。"

    if result.decision.kind is ExecutionDecisionKind.USE_PUBLISHED_FALLBACK:
        text += " 已切换到已发布的安全练习。"
        cue_code += "-published-fallback"
    elif result.decision.kind is ExecutionDecisionKind.SAFE_PAUSE:
        event_type = CoachingEventType.SYSTEM_STATUS
        delivery = CoachingDeliveryStatus.DEGRADED
        text += " 训练指导已安全暂停。"
        cue_code += "-safe-pause"
    elif result.state.difficulty > state.difficulty:
        event_type = CoachingEventType.PROGRESSION
        cue_code = "planned-progression"
        text = "本杆达到已发布训练标准，已按计划提高难度。"
    elif result.state.difficulty < state.difficulty:
        event_type = CoachingEventType.REGRESSION
        cue_code = "planned-regression"
        text = "本杆未达到已发布训练标准，已按计划降低难度。"
    return event_type, outcome, delivery, cue_code, text


def _validate_policy_state(
    state: ExecutionPolicyState,
    projection: VerifiedExecutionProjection,
    runtime_snapshot: RuntimeSnapshot,
) -> None:
    if type(state) is not ExecutionPolicyState:
        raise TypeError("policy state must be an exact ExecutionPolicyState")
    if (
        type(state.projection) is not VerifiedExecutionProjection
        or state.projection is not projection
    ):
        raise ValueError("policy state does not use the verified projection")
    if state.block_index != runtime_snapshot.block_index or not (
        0 <= state.block_index < len(projection.blocks)
    ):
        raise ValueError("policy state block does not match runtime")
    drill = _current_drill(state)
    if (
        state.current_drill_key != drill.key
        or type(state.expected_shot_sequence) is not int
        or state.expected_shot_sequence < 1
        or type(state.difficulty) is not int
        or not drill.minimum_runtime_difficulty
        <= state.difficulty
        <= drill.maximum_runtime_difficulty
        or any(
            type(value) is not bool
            for value in (
                state.fallback_active,
                state.safe_paused,
                state.session_complete,
            )
        )
        or type(state.receipts) is not tuple
    ):
        raise ValueError("policy state invariant mismatch")


def _verified_runtime_effect(
    projection: VerifiedExecutionProjection,
    effect: RuntimeEffect,
    runtime_before: RuntimeSnapshot,
    runtime_command: RuntimeCommand,
) -> RuntimeSnapshot:
    if type(effect) is not RuntimeEffect or type(runtime_before) is not RuntimeSnapshot:
        raise TypeError("runtime proof requires exact snapshot and effect")
    if type(runtime_command) not in _EXACT_RUNTIME_COMMAND_TYPES:
        raise TypeError("runtime proof requires an exact runtime command")
    verified = transition(runtime_before, runtime_command)
    if (
        verified.error is not None
        or verified.transition is None
        or len(verified.effects) != 1
        or verified.effects[0] != effect
        or verified.transition.effect_ids != (effect.effect_id,)
    ):
        raise ValueError("runtime effect is not bound to the supplied transition")
    snapshot = verified.snapshot
    if (
        snapshot.plan is None
        or type(snapshot.plan.execution_projection) is not VerifiedExecutionProjection
        or snapshot.plan.execution_projection is not projection
        or snapshot.session_id != projection.session_id
        or effect.session_id != projection.session_id
        or snapshot.block_index < 0
        or snapshot.block_index >= len(projection.blocks)
    ):
        raise ValueError("runtime transition lacks complete admitted projection lineage")
    return snapshot


def build_realtime_coaching_event(
    projection: VerifiedExecutionProjection,
    effect: RuntimeEffect,
    *,
    runtime_before: RuntimeSnapshot,
    runtime_command: RuntimeCommand,
    sequence: int,
    created_monotonic_ms: int,
    policy_state: ExecutionPolicyState | None = None,
    policy_result: ExecutionPolicyResult | None = None,
    observation: RealtimeShotObservation | None = None,
) -> RealtimeCoachingEventV2:
    """Map exact verified inputs without clocks, I/O, mutable state, or LLM calls."""

    if type(projection) is not VerifiedExecutionProjection:
        raise TypeError("feedback mapping requires an exact VerifiedExecutionProjection")
    if not projection.integrity_valid():
        raise ValueError("verified execution projection integrity mismatch")
    runtime_after = _verified_runtime_effect(projection, effect, runtime_before, runtime_command)
    if type(sequence) is not int or sequence < 1:
        raise ValueError("event sequence must be a positive integer")
    if type(created_monotonic_ms) is not int or created_monotonic_ms < 0:
        raise ValueError("event monotonic timestamp must be a non-negative integer")
    has_policy = policy_result is not None or observation is not None
    if has_policy and (
        type(policy_state) is not ExecutionPolicyState
        or type(policy_result) is not ExecutionPolicyResult
        or type(observation) is not RealtimeShotObservation
    ):
        raise TypeError("policy feedback requires exact state, result, and observation")
    if not has_policy and policy_state is not None:
        raise ValueError("policy state is only accepted with verified policy feedback")
    if policy_state is not None:
        _validate_policy_state(policy_state, projection, runtime_after)
    if effect.kind is RuntimeEffectKind.PRESENT_FEEDBACK and not has_policy:
        raise ValueError("feedback effect requires policy result and observation")
    if has_policy and effect.kind is not RuntimeEffectKind.PRESENT_FEEDBACK:
        raise ValueError("policy feedback effect kind must be PRESENT_FEEDBACK")
    if has_policy:
        assert policy_state is not None and policy_result is not None and observation is not None
        verified_result = apply_observation(policy_state, observation)
        if (
            verified_result != policy_result
            or observation.session_id != projection.session_id
            or effect.shot_sequence != observation.shot_sequence
            or created_monotonic_ms != observation.monotonic_timestamp_ms
            or runtime_before.shot_sequence != observation.shot_sequence
            or not any(
                receipt.key.session_id == observation.session_id
                and receipt.key.shot_sequence == observation.shot_sequence
                and receipt.key.evidence_sha256 == observation.evidence_sha256
                for receipt in runtime_before.observation_receipts
            )
        ):
            raise ValueError("policy feedback is not bound to verified inputs")
        event_type, outcome, delivery, cue_code, text = _policy_feedback(
            policy_state, policy_result
        )
    else:
        generic = _GENERIC_FEEDBACK.get(effect.kind)
        if generic is None:
            raise ValueError("runtime effect requires policy feedback")
        event_type, delivery, cue_code, text = generic
        outcome = CoachingOutcomeStatus.INFORMATIONAL

    block_index = runtime_after.block_index
    block = projection.blocks[block_index]
    drill = block.drill if policy_state is None else _current_drill(policy_state)
    lineage = RealtimeCoachingLineageV1(
        admission_id=projection.admission_id,
        bundle_id=projection.bundle_id,
        bundle_content_sha256=projection.bundle_content_sha256,
        projection_sha256=projection.projection_sha256,
        session_id=projection.session_id,
        plan_id=projection.plan_id,
        plan_version=projection.plan_version,
        block_id=block.block_id,
        block_order=block.order,
        drill_id=drill.drill_id,
        drill_revision=drill.drill_revision,
        drill_content_sha256=drill.drill_content_sha256,
        shot_id=None if observation is None else observation.shot.shot_id,
        shot_sequence=None if observation is None else observation.shot_sequence,
        evidence_sha256=None if observation is None else observation.evidence_sha256,
        decision_id=None if policy_result is None else policy_result.decision.decision_id,
        effect_id=effect.effect_id,
    )
    speech = SpeechCueV1(
        cue_code=cue_code,
        outcome_status=outcome,
        delivery_status=delivery,
        text=text,
    )
    return issue_realtime_coaching_event_v2(
        sequence=sequence,
        event_type=event_type,
        outcome_status=outcome,
        delivery_status=delivery,
        lineage=lineage,
        overlay=(
            None if policy_state is None else _overlay(policy_state, outcome, delivery, observation)
        ),
        speech=speech,
        text_summary=text,
        created_monotonic_ms=created_monotonic_ms,
    )


__all__ = ["build_realtime_coaching_event"]
