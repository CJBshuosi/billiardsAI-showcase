"""Deterministic, raw-free execution of one verified published plan projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import final
from uuid import UUID

from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
from billiards_ai.shared.models.quality import FinalFieldSource
from billiards_ai.shared.models.realtime import RealtimeObservationStatus, RealtimeShotObservation
from billiards_ai.shared.models.shot import Confidence, TargetZoneSource
from billiards_ai.shared.plan_publication.contracts import (
    AbstentionAction,
    MetricCombination,
    ReplanContinueAction,
)


class EvaluationConclusion(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ABSTAINED = "ABSTAINED"
    UNSUPPORTED = "UNSUPPORTED"


class ExecutionDecisionKind(StrEnum):
    NEXT_SHOT = "NEXT_SHOT"
    BEGIN_COMPLETION = "BEGIN_COMPLETION"
    USE_PUBLISHED_FALLBACK = "USE_PUBLISHED_FALLBACK"
    SAFE_PAUSE = "SAFE_PAUSE"


_PROJECTION_ISSUER = object()


def _canonical(value: object) -> bytes:
    def default(item: object) -> object:
        if isinstance(item, StrEnum):
            return item.value
        if isinstance(item, UUID | datetime):
            return str(item)
        if is_dataclass(item) and not isinstance(item, type):
            return {field.name: getattr(item, field.name) for field in fields(item)}
        raise TypeError(type(item).__name__)

    return json.dumps(value, default=default, sort_keys=True, separators=(",", ":")).encode()


def _sha(*parts: object) -> str:
    return hashlib.sha256(_canonical(parts)).hexdigest()


@final
@dataclass(frozen=True, slots=True)
class MetricRuleProjection:
    metric_code: str
    target_attempts: int
    minimum_successes: int
    evaluator_code: str
    evaluator_version: str
    required_capabilities: tuple[str, ...]


@final
@dataclass(frozen=True, slots=True)
class SuccessPolicyProjection:
    evaluator_policy_version: str
    combination: MetricCombination
    metric_rules: tuple[MetricRuleProjection, ...]


@final
@dataclass(frozen=True, slots=True)
class DrillExecutionProjection:
    drill_id: str
    drill_revision: int
    drill_content_sha256: str
    required_capabilities: tuple[str, ...]
    initial_difficulty: int
    minimum_runtime_difficulty: int
    maximum_runtime_difficulty: int
    progression_count: int
    progression_delta: int
    regression_count: int
    regression_delta: int
    abstention_count: int
    abstention_action: AbstentionAction
    success_policy: SuccessPolicyProjection
    replan_reason_codes: tuple[str, ...]
    replan_continue_action: ReplanContinueAction
    fallback_key: tuple[str, int] | None
    target_zone: tuple[int, int, int, int] | None

    @property
    def key(self) -> tuple[str, int]:
        return self.drill_id, self.drill_revision


@final
@dataclass(frozen=True, slots=True)
class ExecutionBlockProjection:
    block_id: UUID
    order: int
    target_repetitions: int
    drill: DrillExecutionProjection


@final
@dataclass(frozen=True, slots=True, repr=False)
class VerifiedExecutionProjection:
    _issuer: object = field(repr=False, compare=False)
    schema_version: str
    admission_id: str
    bundle_id: UUID
    bundle_content_sha256: str
    user_id: UUID
    session_id: UUID
    plan_id: UUID
    plan_version: int
    target_device_id: str
    publication_sequence: int
    expires_at: datetime
    plan_valid_until: datetime
    authorization_id: UUID
    authorization_version: int
    authorization_expires_at: datetime
    signing_key_id: str
    signing_key_version: int
    signing_key_expires_at: datetime
    execution_policy_catalog_version: str
    execution_policy_catalog_sha256: str
    capability_manifest_id: str
    capability_manifest_version: int
    capability_manifest_content_sha256: str
    capabilities: tuple[str, ...]
    equipment_codes: tuple[str, ...]
    blocks: tuple[ExecutionBlockProjection, ...]
    terminal_fallback: DrillExecutionProjection
    projection_sha256: str

    def integrity_valid(self) -> bool:
        material = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in {"_issuer", "projection_sha256"}
        }
        return self._issuer is _PROJECTION_ISSUER and self.projection_sha256 == _sha(
            "verified-execution-projection/v1", material
        )

    def __repr__(self) -> str:
        return "VerifiedExecutionProjection(<redacted>)"


def _issue_verified_execution_projection(**values: object) -> VerifiedExecutionProjection:
    """Private issuer used only by the verified admission projection builder."""

    material = {"schema_version": "1.0", **values}
    material.pop("projection_sha256", None)
    digest = _sha("verified-execution-projection/v1", material)
    return VerifiedExecutionProjection(
        _issuer=_PROJECTION_ISSUER,
        **material,
        projection_sha256=digest,
    )  # type: ignore[arg-type]


@final
@dataclass(frozen=True, slots=True)
class MetricEvaluation:
    metric_code: str
    evaluator_code: str
    evaluator_version: str
    conclusion: EvaluationConclusion


@final
@dataclass(frozen=True, slots=True)
class ObservationEvaluation:
    conclusion: EvaluationConclusion
    metrics: tuple[MetricEvaluation, ...]


@final
@dataclass(frozen=True, slots=True)
class ReplanIntent:
    schema_version: str
    intent_id: str
    admission_id: str
    session_id: UUID
    plan_id: UUID
    block_index: int
    reason_code: str
    shot_sequence: int
    evidence_sha256: str


@final
@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    decision_id: str
    kind: ExecutionDecisionKind
    next_shot_sequence: int
    next_block_index: int
    difficulty: int
    replan_intent: ReplanIntent | None = None


@final
@dataclass(frozen=True, slots=True)
class PolicyReceipt:
    shot_sequence: int
    evidence_sha256: str
    evaluation: ObservationEvaluation
    decision: ExecutionDecision


@final
@dataclass(frozen=True, slots=True)
class ExecutionPolicyState:
    projection: VerifiedExecutionProjection
    block_index: int
    current_drill_key: tuple[str, int]
    expected_shot_sequence: int
    difficulty: int
    completed_repetitions: int = 0
    success_streak: int = 0
    failure_streak: int = 0
    abstention_streak: int = 0
    fallback_active: bool = False
    safe_paused: bool = False
    session_complete: bool = False
    receipts: tuple[PolicyReceipt, ...] = ()

    @classmethod
    def start(
        cls,
        projection: VerifiedExecutionProjection,
        block_index: int = 0,
        expected_shot_sequence: int = 1,
    ) -> ExecutionPolicyState:
        if type(projection) is not VerifiedExecutionProjection:
            raise TypeError("projection must be an exact VerifiedExecutionProjection")
        if block_index < 0 or block_index >= len(projection.blocks):
            raise ValueError("block index is outside verified projection")
        if not projection.integrity_valid():
            raise ValueError("verified execution projection integrity mismatch")
        if type(expected_shot_sequence) is not int or expected_shot_sequence < 1:
            raise ValueError("expected shot sequence must be positive")
        drill = projection.blocks[block_index].drill
        return cls(
            projection=projection,
            block_index=block_index,
            current_drill_key=drill.key,
            expected_shot_sequence=expected_shot_sequence,
            difficulty=drill.initial_difficulty,
        )


@final
@dataclass(frozen=True, slots=True)
class ExecutionPolicyResult:
    state: ExecutionPolicyState
    evaluation: ObservationEvaluation
    decision: ExecutionDecision
    replayed: bool = False


def _required_field_quality_is_supported(
    observation: RealtimeShotObservation,
    field_names: tuple[str, ...],
) -> bool:
    quality = observation.shot.field_quality
    if quality is None:
        return False
    return all(
        (field_quality := getattr(quality, field_name)).source is not FinalFieldSource.UNKNOWN
        and field_quality.confidence >= CONFIDENCE_ABSTAIN_BELOW
        for field_name in field_names
    )


def _metric(
    rule: MetricRuleProjection,
    drill: DrillExecutionProjection,
    observation: RealtimeShotObservation,
    capabilities: frozenset[str],
) -> MetricEvaluation:
    conclusion = EvaluationConclusion.UNSUPPORTED
    registry_key = (
        drill.success_policy.evaluator_policy_version,
        rule.metric_code,
        rule.evaluator_code,
        rule.evaluator_version,
    )
    registry_capabilities = _EVALUATOR_REGISTRY.get(registry_key)
    if registry_capabilities is None:
        pass
    elif not registry_capabilities <= frozenset(rule.required_capabilities):
        pass
    elif not set(rule.required_capabilities) <= capabilities:
        pass
    elif observation.shot.confidence in {Confidence.LOW, Confidence.ABSTAIN}:
        pass
    elif rule.evaluator_code == "POT_RESULT":
        shot = observation.shot
        if (
            _required_field_quality_is_supported(
                observation,
                ("struck_ball", "potted_balls", "is_foul"),
            )
            and shot.struck_ball is not None
            and shot.is_foul is not None
        ):
            conclusion = (
                EvaluationConclusion.SUCCESS
                if (not shot.is_foul and shot.struck_ball in shot.potted_balls)
                else EvaluationConclusion.FAILURE
            )
    elif rule.evaluator_code == "TARGET_ZONE_RESULT":
        point, zone = observation.shot.cue_ball_end, drill.target_zone
        if (
            observation.shot.target_zone_source is TargetZoneSource.SCENE_PRESET
            and _required_field_quality_is_supported(observation, ("cue_ball_end",))
            and point is not None
            and zone is not None
        ):
            minimum_x, maximum_x, minimum_y, maximum_y = zone
            conclusion = (
                EvaluationConclusion.SUCCESS
                if minimum_x <= point.x_mm <= maximum_x and minimum_y <= point.y_mm <= maximum_y
                else EvaluationConclusion.FAILURE
            )
    # BANK_POT_RESULT intentionally remains unsupported: aggregate cushion_count cannot prove
    # that the target object ball contacted a cushion before the pot.
    return MetricEvaluation(
        rule.metric_code, rule.evaluator_code, rule.evaluator_version, conclusion
    )


_EVALUATOR_REGISTRY: dict[tuple[str, str, str, str], frozenset[str]] = {
    ("edge-success-v1", "POT_SUCCESS_COUNT", "POT_RESULT", "1.0"): frozenset(),
    (
        "edge-success-v1",
        "TARGET_ZONE_SUCCESS_COUNT",
        "TARGET_ZONE_RESULT",
        "1.0",
    ): frozenset({"TABLE_MM_COORDINATES"}),
    (
        "edge-success-v1",
        "BANK_POT_SUCCESS_COUNT",
        "BANK_POT_RESULT",
        "1.0",
    ): frozenset(),
}


def evaluator_registry_compatible(projection: VerifiedExecutionProjection) -> bool:
    """Validate every signed evaluator against the frozen edge registry."""

    if type(projection) is not VerifiedExecutionProjection or not projection.integrity_valid():
        return False
    for drill in tuple(block.drill for block in projection.blocks) + (
        projection.terminal_fallback,
    ):
        policy_version = drill.success_policy.evaluator_policy_version
        if "TARGET_OUT_OF_RANGE" not in drill.replan_reason_codes:
            return False
        for rule in drill.success_policy.metric_rules:
            key = (
                policy_version,
                rule.metric_code,
                rule.evaluator_code,
                rule.evaluator_version,
            )
            required = _EVALUATOR_REGISTRY.get(key)
            if required is None:
                return False
            declared = frozenset(rule.required_capabilities)
            if not required <= declared or not declared <= frozenset(projection.capabilities):
                return False
    return True


def evaluate_observation(
    projection: VerifiedExecutionProjection,
    drill: DrillExecutionProjection,
    observation: RealtimeShotObservation,
) -> ObservationEvaluation:
    if (
        type(projection) is not VerifiedExecutionProjection
        or type(drill) is not DrillExecutionProjection
        or type(observation) is not RealtimeShotObservation
    ):
        raise TypeError("evaluation requires exact verified projection, drill, and observation")
    if observation.status is RealtimeObservationStatus.ABSTAINED:
        return ObservationEvaluation(EvaluationConclusion.ABSTAINED, ())
    if observation.status is RealtimeObservationStatus.UNSUPPORTED:
        return ObservationEvaluation(EvaluationConclusion.UNSUPPORTED, ())
    metrics = tuple(
        _metric(rule, drill, observation, frozenset(projection.capabilities))
        for rule in drill.success_policy.metric_rules
    )
    values = tuple(item.conclusion for item in metrics)
    if EvaluationConclusion.UNSUPPORTED in values:
        conclusion = EvaluationConclusion.UNSUPPORTED
    elif drill.success_policy.combination is MetricCombination.ALL:
        conclusion = (
            EvaluationConclusion.FAILURE
            if EvaluationConclusion.FAILURE in values
            else EvaluationConclusion.SUCCESS
            if all(item is EvaluationConclusion.SUCCESS for item in values)
            else EvaluationConclusion.UNSUPPORTED
        )
    else:
        conclusion = (
            EvaluationConclusion.SUCCESS
            if EvaluationConclusion.SUCCESS in values
            else EvaluationConclusion.FAILURE
            if all(item is EvaluationConclusion.FAILURE for item in values)
            else EvaluationConclusion.UNSUPPORTED
        )
    return ObservationEvaluation(conclusion, metrics)


def _current_drill(state: ExecutionPolicyState) -> DrillExecutionProjection:
    if state.fallback_active:
        return state.projection.terminal_fallback
    return state.projection.blocks[state.block_index].drill


def _intent(
    state: ExecutionPolicyState, observation: RealtimeShotObservation, reason: str
) -> ReplanIntent:
    intent_id = _sha(
        "replan-intent/v1",
        state.projection.admission_id,
        state.block_index,
        reason,
        observation.shot_sequence,
        observation.evidence_sha256,
    )
    return ReplanIntent(
        "1.0",
        intent_id,
        state.projection.admission_id,
        state.projection.session_id,
        state.projection.plan_id,
        state.block_index,
        reason,
        observation.shot_sequence,
        observation.evidence_sha256,
    )


def _apply_replan_action(
    state: ExecutionPolicyState,
    drill: DrillExecutionProjection,
    difficulty: int,
) -> tuple[ExecutionDecisionKind, bool, bool, int]:
    if (
        drill.replan_continue_action is ReplanContinueAction.USE_PUBLISHED_FALLBACK
        and not state.fallback_active
    ):
        terminal = state.projection.terminal_fallback
        return (
            ExecutionDecisionKind.USE_PUBLISHED_FALLBACK,
            True,
            False,
            terminal.initial_difficulty,
        )
    return ExecutionDecisionKind.SAFE_PAUSE, state.fallback_active, True, difficulty


def apply_observation(
    state: ExecutionPolicyState, observation: RealtimeShotObservation
) -> ExecutionPolicyResult:
    if type(state) is not ExecutionPolicyState or type(observation) is not RealtimeShotObservation:
        raise TypeError("policy requires exact state and observation")
    if observation.session_id != state.projection.session_id:
        raise ValueError("observation session does not match verified projection")
    for receipt in state.receipts:
        if receipt.shot_sequence == observation.shot_sequence:
            if receipt.evidence_sha256 != observation.evidence_sha256:
                raise ValueError("conflicting observation evidence for shot sequence")
            return ExecutionPolicyResult(state, receipt.evaluation, receipt.decision, True)
    if observation.shot_sequence != state.expected_shot_sequence:
        raise ValueError("observation sequence is not the next policy shot")
    if state.safe_paused:
        raise ValueError("safe-paused policy cannot accept observations")
    if state.session_complete:
        raise ValueError("completed policy cannot accept observations")

    drill = _current_drill(state)
    evaluation = evaluate_observation(state.projection, drill, observation)
    success, failure, abstention = (
        state.success_streak,
        state.failure_streak,
        state.abstention_streak,
    )
    difficulty = state.difficulty
    kind = ExecutionDecisionKind.NEXT_SHOT
    fallback_active = state.fallback_active
    safe_paused = False
    intent = None
    completed_repetitions = state.completed_repetitions
    bound_exhausted = False
    if evaluation.conclusion is EvaluationConclusion.SUCCESS:
        completed_repetitions += 1
        success, failure, abstention = success + 1, 0, 0
        if success >= drill.progression_count:
            proposed = difficulty + drill.progression_delta
            difficulty = min(drill.maximum_runtime_difficulty, proposed)
            success = 0
            if (
                proposed > drill.maximum_runtime_difficulty
                and "TARGET_OUT_OF_RANGE" in drill.replan_reason_codes
            ):
                intent = _intent(state, observation, "TARGET_OUT_OF_RANGE")
                bound_exhausted = True
    elif evaluation.conclusion is EvaluationConclusion.FAILURE:
        completed_repetitions += 1
        success, failure, abstention = 0, failure + 1, 0
        if failure >= drill.regression_count:
            proposed = difficulty + drill.regression_delta
            difficulty = max(drill.minimum_runtime_difficulty, proposed)
            failure = 0
            if (
                proposed < drill.minimum_runtime_difficulty
                and "TARGET_OUT_OF_RANGE" in drill.replan_reason_codes
            ):
                intent = _intent(state, observation, "TARGET_OUT_OF_RANGE")
                bound_exhausted = True
    else:
        success = failure = 0
        abstention += 1
        if abstention >= drill.abstention_count:
            abstention = 0
            if (
                drill.abstention_action is AbstentionAction.USE_PUBLISHED_FALLBACK
                and not fallback_active
            ):
                kind = ExecutionDecisionKind.USE_PUBLISHED_FALLBACK
                fallback_active = True
                terminal = state.projection.terminal_fallback
                difficulty = terminal.initial_difficulty
                success = failure = 0
            elif drill.abstention_action is AbstentionAction.SAFE_PAUSE:
                kind = ExecutionDecisionKind.SAFE_PAUSE
                safe_paused = True
                if "FALLBACK_UNAVAILABLE" in drill.replan_reason_codes:
                    intent = _intent(state, observation, "FALLBACK_UNAVAILABLE")

    if bound_exhausted:
        kind, fallback_active, safe_paused, difficulty = _apply_replan_action(
            state, drill, difficulty
        )
        success = failure = abstention = 0
        completed_repetitions = 0

    next_block_index = state.block_index
    current_drill_key = state.projection.terminal_fallback.key if fallback_active else drill.key
    if (
        kind is ExecutionDecisionKind.NEXT_SHOT
        and not fallback_active
        and completed_repetitions >= state.projection.blocks[state.block_index].target_repetitions
        and state.block_index + 1 < len(state.projection.blocks)
    ):
        next_block_index = state.block_index + 1
        next_drill = state.projection.blocks[next_block_index].drill
        current_drill_key = next_drill.key
        difficulty = next_drill.initial_difficulty
        completed_repetitions = 0
        success = failure = abstention = 0
    elif (
        kind is ExecutionDecisionKind.NEXT_SHOT
        and not fallback_active
        and completed_repetitions >= state.projection.blocks[state.block_index].target_repetitions
    ):
        kind = ExecutionDecisionKind.BEGIN_COMPLETION
        success = failure = abstention = 0

    next_sequence = observation.shot_sequence + 1
    decision = ExecutionDecision(
        decision_id=_sha(
            "execution-decision/v1",
            state.projection.projection_sha256,
            state.block_index,
            state.current_drill_key,
            state.difficulty,
            state.success_streak,
            state.failure_streak,
            state.abstention_streak,
            state.completed_repetitions,
            observation.shot_sequence,
            observation.evidence_sha256,
            evaluation.conclusion,
            kind,
            difficulty,
            fallback_active,
            safe_paused,
            intent,
            next_block_index,
        ),
        kind=kind,
        next_shot_sequence=next_sequence,
        next_block_index=next_block_index,
        difficulty=difficulty,
        replan_intent=intent,
    )
    receipt = PolicyReceipt(
        observation.shot_sequence,
        observation.evidence_sha256,
        evaluation,
        decision,
    )
    next_state = replace(
        state,
        block_index=next_block_index,
        current_drill_key=current_drill_key,
        expected_shot_sequence=next_sequence,
        difficulty=difficulty,
        completed_repetitions=completed_repetitions,
        success_streak=success,
        failure_streak=failure,
        abstention_streak=abstention,
        fallback_active=fallback_active,
        safe_paused=safe_paused,
        session_complete=kind is ExecutionDecisionKind.BEGIN_COMPLETION,
        receipts=state.receipts + (receipt,),
    )
    return ExecutionPolicyResult(next_state, evaluation, decision)


__all__ = [
    "AbstentionAction",
    "DrillExecutionProjection",
    "EvaluationConclusion",
    "ExecutionBlockProjection",
    "ExecutionDecision",
    "ExecutionDecisionKind",
    "ExecutionPolicyResult",
    "ExecutionPolicyState",
    "MetricCombination",
    "MetricRuleProjection",
    "ObservationEvaluation",
    "ReplanContinueAction",
    "ReplanIntent",
    "SuccessPolicyProjection",
    "VerifiedExecutionProjection",
    "apply_observation",
    "evaluator_registry_compatible",
    "evaluate_observation",
]
