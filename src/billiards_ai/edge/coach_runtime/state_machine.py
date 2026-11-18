"""Pure deterministic state reducer for an admitted coaching plan."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from enum import Enum, StrEnum
from typing import final
from uuid import UUID

from pydantic import BaseModel

from billiards_ai.edge.coach_runtime.errors import RuntimeErrorCode, RuntimeTransitionError
from billiards_ai.edge.coach_runtime.execution_policy import VerifiedExecutionProjection
from billiards_ai.shared.models.realtime import RealtimeShotObservation


class CoachRuntimeState(StrEnum):
    IDLE = "IDLE"
    LOADING_PLAN = "LOADING_PLAN"
    READY = "READY"
    PRESENTING_SETUP = "PRESENTING_SETUP"
    WAITING_FOR_SHOT = "WAITING_FOR_SHOT"
    ANALYZING_SHOT = "ANALYZING_SHOT"
    PRESENTING_FEEDBACK = "PRESENTING_FEEDBACK"
    PAUSED = "PAUSED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class RuntimeEffectKind(StrEnum):
    PLAN_LOADING = "PLAN_LOADING"
    PLAN_READY = "PLAN_READY"
    PRESENT_SETUP = "PRESENT_SETUP"
    WAIT_FOR_SHOT = "WAIT_FOR_SHOT"
    ANALYZE_SHOT = "ANALYZE_SHOT"
    PRESENT_FEEDBACK = "PRESENT_FEEDBACK"
    PAUSED = "PAUSED"
    RESUMED = "RESUMED"
    SAFE_PAUSE = "SAFE_PAUSE"
    RECOVERED = "RECOVERED"
    COMPLETE_SESSION = "COMPLETE_SESSION"
    SESSION_COMPLETED = "SESSION_COMPLETED"
    RUNTIME_FAILED = "RUNTIME_FAILED"


@final
@dataclass(frozen=True, slots=True)
class VerifiedPlanAdmission:
    """Nominal value seam produced after Story 10.1 verifies a published plan."""

    admission_id: str
    session_id: UUID
    plan_id: UUID
    plan_version: int
    execution_projection: VerifiedExecutionProjection | None = None


@dataclass(frozen=True, slots=True)
class AdmittedPlanIdentity:
    admission_id: str
    plan_id: UUID
    plan_version: int
    execution_projection: VerifiedExecutionProjection | None = None


@dataclass(frozen=True, slots=True)
class BeginPlanLoad:
    command_id: str = "begin-plan-load"


@dataclass(frozen=True, slots=True)
class AdmitVerifiedPlan:
    admission: VerifiedPlanAdmission
    command_id: str = "admit-verified-plan"


@dataclass(frozen=True, slots=True)
class PresentSetup:
    command_id: str = "present-setup"


@dataclass(frozen=True, slots=True)
class WaitForShot:
    command_id: str = "wait-for-shot"


@dataclass(frozen=True, slots=True)
class AcceptObservation:
    observation: RealtimeShotObservation
    command_id: str = "accept-observation"


@dataclass(frozen=True, slots=True)
class PresentFeedback:
    command_id: str = "present-feedback"


@dataclass(frozen=True, slots=True)
class AdvanceToNextShot:
    next_shot_sequence: int
    next_block_index: int
    command_id: str = "advance-to-next-shot"


@dataclass(frozen=True, slots=True)
class PauseRuntime:
    reason: str
    command_id: str = "pause-runtime"


@dataclass(frozen=True, slots=True)
class ResumeRuntime:
    command_id: str = "resume-runtime"


@dataclass(frozen=True, slots=True)
class RecoverRuntime:
    session_id: UUID
    expected_shot_sequence: int
    consistency_sha256: str
    command_id: str = "recover-runtime"


@dataclass(frozen=True, slots=True)
class BeginCompletion:
    command_id: str = "begin-completion"


@dataclass(frozen=True, slots=True)
class FinishCompletion:
    command_id: str = "finish-completion"


@dataclass(frozen=True, slots=True)
class FailRuntime:
    reason_code: str
    command_id: str = "fail-runtime"


RuntimeCommand = (
    BeginPlanLoad
    | AdmitVerifiedPlan
    | PresentSetup
    | WaitForShot
    | AcceptObservation
    | PresentFeedback
    | AdvanceToNextShot
    | PauseRuntime
    | ResumeRuntime
    | RecoverRuntime
    | BeginCompletion
    | FinishCompletion
    | FailRuntime
)

_KNOWN_COMMAND_TYPES = (
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


@dataclass(frozen=True, slots=True)
class ObservationKey:
    session_id: UUID
    shot_sequence: int
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeEffect:
    effect_id: str
    kind: RuntimeEffectKind
    session_id: UUID | None
    shot_sequence: int


@dataclass(frozen=True, slots=True)
class RuntimeTransition:
    transition_id: str
    from_state: CoachRuntimeState
    to_state: CoachRuntimeState
    command_type: str
    effect_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    key: ObservationKey
    transition: RuntimeTransition
    effects: tuple[RuntimeEffect, ...]


@dataclass(frozen=True, slots=True)
class ObservationOutcomeReceipt:
    """Replay identity for an observation that deterministically degraded runtime."""

    key: ObservationKey
    transition: RuntimeTransition
    effects: tuple[RuntimeEffect, ...]


@dataclass(frozen=True, slots=True)
class CompletionReceipt:
    transition: RuntimeTransition
    effects: tuple[RuntimeEffect, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    state: CoachRuntimeState = CoachRuntimeState.IDLE
    session_id: UUID | None = None
    plan: AdmittedPlanIdentity | None = None
    shot_sequence: int = 0
    block_index: int = 0
    # Counts authoritative advances; AC 9 deliberately excludes resume restoration.
    transition_count: int = 0
    observation_receipts: tuple[ObservationReceipt, ...] = ()
    observation_outcome_receipts: tuple[ObservationOutcomeReceipt, ...] = ()
    resume_state: CoachRuntimeState | None = None
    degraded_reason: RuntimeErrorCode | None = None
    last_trusted_state: CoachRuntimeState | None = None
    recovery_state: CoachRuntimeState | None = None
    completion_receipt: CompletionReceipt | None = None
    completion_finished_receipt: CompletionReceipt | None = None
    failure_reason: str | None = None

    @classmethod
    def initial(cls) -> RuntimeSnapshot:
        return cls()

    @property
    def accepted_observation_count(self) -> int:
        return len(self.observation_receipts)


@dataclass(frozen=True, slots=True)
class RuntimeTransitionResult:
    snapshot: RuntimeSnapshot
    transition: RuntimeTransition | None = None
    effects: tuple[RuntimeEffect, ...] = ()
    error: RuntimeTransitionError | None = None
    replayed: bool = False


_ACTIVITY_STATES = frozenset(
    {
        CoachRuntimeState.READY,
        CoachRuntimeState.PRESENTING_SETUP,
        CoachRuntimeState.WAITING_FOR_SHOT,
        CoachRuntimeState.ANALYZING_SHOT,
        CoachRuntimeState.PRESENTING_FEEDBACK,
    }
)
_TERMINAL_STATES = frozenset({CoachRuntimeState.COMPLETED, CoachRuntimeState.FAILED})
_ADMITTED_SESSION_STATES = _ACTIVITY_STATES | {
    CoachRuntimeState.PAUSED,
    CoachRuntimeState.DEGRADED,
    CoachRuntimeState.COMPLETING,
    CoachRuntimeState.COMPLETED,
}


def _is_uuid4(value: object) -> bool:
    return isinstance(value, UUID) and value.version == 4


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_plan_identity(value: object) -> bool:
    return (
        type(value) is AdmittedPlanIdentity
        and type(value.admission_id) is str
        and bool(value.admission_id.strip())
        and _is_uuid4(value.plan_id)
        and type(value.plan_version) is int
        and value.plan_version > 0
        and type(value.execution_projection) is VerifiedExecutionProjection
        and value.execution_projection.integrity_valid()
        and value.execution_projection.admission_id == value.admission_id
        and value.execution_projection.plan_id == value.plan_id
        and value.execution_projection.plan_version == value.plan_version
    )


def _snapshot_invariant_error(snapshot: RuntimeSnapshot) -> str | None:
    if type(snapshot) is not RuntimeSnapshot:
        return "snapshot must be a RuntimeSnapshot"
    if not isinstance(snapshot.state, CoachRuntimeState):
        return "snapshot state must be a CoachRuntimeState"
    if type(snapshot.transition_count) is not int or snapshot.transition_count < 0:
        return "transition_count must be a non-negative integer"
    if type(snapshot.shot_sequence) is not int or snapshot.shot_sequence < 0:
        return "shot_sequence must be a non-negative integer"
    if type(snapshot.block_index) is not int or snapshot.block_index < 0:
        return "block_index must be a non-negative integer"
    if snapshot.state in _ADMITTED_SESSION_STATES:
        if not _is_uuid4(snapshot.session_id):
            return "admitted runtime state requires a UUID4 session_id"
        if not _valid_plan_identity(snapshot.plan):
            return "admitted runtime state requires a valid admitted plan identity"
        if snapshot.shot_sequence < 1:
            return "admitted runtime state requires a positive shot_sequence"
    if snapshot.state is CoachRuntimeState.PAUSED:
        if snapshot.resume_state not in _ACTIVITY_STATES:
            return "PAUSED requires an activity resume_state"
    if snapshot.state is CoachRuntimeState.DEGRADED:
        if snapshot.degraded_reason not in {
            RuntimeErrorCode.SEQUENCE_GAP,
            RuntimeErrorCode.EVIDENCE_CONFLICT,
        }:
            return "DEGRADED requires a supported degradation reason"
        if snapshot.last_trusted_state not in _ACTIVITY_STATES:
            return "DEGRADED requires an activity last_trusted_state"
        if snapshot.recovery_state not in _ACTIVITY_STATES:
            return "DEGRADED requires an activity recovery_state"
    if snapshot.state is CoachRuntimeState.COMPLETING:
        receipt = snapshot.completion_receipt
        if (
            type(receipt) is not CompletionReceipt
            or type(receipt.transition) is not RuntimeTransition
            or type(receipt.effects) is not tuple
            or not receipt.effects
            or any(type(effect) is not RuntimeEffect for effect in receipt.effects)
            or receipt.transition.to_state is not CoachRuntimeState.COMPLETING
            or receipt.transition.command_type != BeginCompletion.__name__
            or receipt.transition.effect_ids
            != tuple(effect.effect_id for effect in receipt.effects)
            or any(
                effect.kind is not RuntimeEffectKind.COMPLETE_SESSION for effect in receipt.effects
            )
        ):
            return "COMPLETING requires a begin-completion receipt"
    return None


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(*parts: object) -> str:
    return hashlib.sha256(canonical_bytes(parts)).hexdigest()


def expected_recovery_proof(snapshot: RuntimeSnapshot) -> str | None:
    """Recompute the consistency token for one exact degraded snapshot."""

    if (
        snapshot.state is not CoachRuntimeState.DEGRADED
        or snapshot.degraded_reason is None
        or snapshot.last_trusted_state is None
        or snapshot.recovery_state is None
    ):
        return None
    return _digest("coach-runtime-recovery-proof/v1", snapshot)


def _command_payload(command: RuntimeCommand) -> object:
    if isinstance(command, AdmitVerifiedPlan):
        admission = command.admission
        return (
            command.command_id,
            getattr(admission, "admission_id", None),
            getattr(admission, "session_id", None),
            getattr(admission, "plan_id", None),
            getattr(admission, "plan_version", None),
            getattr(getattr(admission, "execution_projection", None), "projection_sha256", None),
        )
    if isinstance(command, AcceptObservation):
        observation = command.observation
        return (
            command.command_id,
            observation.session_id,
            observation.shot_sequence,
            observation.evidence_sha256,
        )
    return _jsonable(command)


def _make_transition(
    snapshot: RuntimeSnapshot,
    command: RuntimeCommand,
    to_state: CoachRuntimeState,
    effect_kind: RuntimeEffectKind,
    increment_transition_count: bool = True,
    **changes: object,
) -> RuntimeTransitionResult:
    next_transition_count = snapshot.transition_count + int(increment_transition_count)
    transition_id = _digest(
        "transition",
        snapshot,
        snapshot.state,
        to_state,
        next_transition_count,
        _command_payload(command),
    )
    effect = RuntimeEffect(
        effect_id=_digest("effect", transition_id, effect_kind, 0),
        kind=effect_kind,
        session_id=changes.get("session_id", snapshot.session_id),
        shot_sequence=int(changes.get("shot_sequence", snapshot.shot_sequence)),
    )
    transition_record = RuntimeTransition(
        transition_id=transition_id,
        from_state=snapshot.state,
        to_state=to_state,
        command_type=type(command).__name__,
        effect_ids=(effect.effect_id,),
    )
    next_snapshot = replace(
        snapshot,
        state=to_state,
        transition_count=next_transition_count,
        **changes,
    )
    return RuntimeTransitionResult(
        snapshot=next_snapshot,
        transition=transition_record,
        effects=(effect,),
    )


def _error(
    snapshot: RuntimeSnapshot,
    command: RuntimeCommand,
    code: RuntimeErrorCode,
    message: str,
) -> RuntimeTransitionResult:
    return RuntimeTransitionResult(
        snapshot=snapshot,
        error=RuntimeTransitionError(
            code=code,
            state=snapshot.state.value,
            command_type=type(command).__name__,
            message=message,
        ),
    )


def _invalid(snapshot: RuntimeSnapshot, command: RuntimeCommand) -> RuntimeTransitionResult:
    if snapshot.state in _TERMINAL_STATES:
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.TERMINAL_STATE,
            f"{snapshot.state.value} is terminal",
        )
    return _error(
        snapshot,
        command,
        RuntimeErrorCode.INVALID_TRANSITION,
        f"{type(command).__name__} is invalid from {snapshot.state.value}",
    )


def _valid_admission(value: object) -> bool:
    if type(value) is not VerifiedPlanAdmission:
        return False
    return (
        isinstance(value.admission_id, str)
        and bool(value.admission_id.strip())
        and _is_uuid4(value.session_id)
        and _is_uuid4(value.plan_id)
        and type(value.plan_version) is int
        and value.plan_version > 0
        and type(value.execution_projection) is VerifiedExecutionProjection
        and value.execution_projection.integrity_valid()
        and value.execution_projection.admission_id == value.admission_id
        and value.execution_projection.session_id == value.session_id
        and value.execution_projection.plan_id == value.plan_id
        and value.execution_projection.plan_version == value.plan_version
    )


def _degrade(
    snapshot: RuntimeSnapshot,
    command: AcceptObservation,
    reason: RuntimeErrorCode,
) -> RuntimeTransitionResult:
    result = _make_transition(
        snapshot,
        command,
        CoachRuntimeState.DEGRADED,
        RuntimeEffectKind.SAFE_PAUSE,
        degraded_reason=reason,
        last_trusted_state=snapshot.state,
        recovery_state=snapshot.state,
    )
    observation = command.observation
    receipt = ObservationOutcomeReceipt(
        key=ObservationKey(
            session_id=observation.session_id,
            shot_sequence=observation.shot_sequence,
            evidence_sha256=observation.evidence_sha256,
        ),
        transition=result.transition,
        effects=result.effects,
    )
    return replace(
        result,
        snapshot=replace(
            result.snapshot,
            observation_outcome_receipts=(
                *snapshot.observation_outcome_receipts,
                receipt,
            ),
        ),
    )


def _accept_observation(
    snapshot: RuntimeSnapshot,
    command: AcceptObservation,
) -> RuntimeTransitionResult:
    observation = command.observation
    if not isinstance(observation, RealtimeShotObservation):
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.INVALID_OBSERVATION,
            "observation must be a RealtimeShotObservation",
        )
    if observation.session_id != snapshot.session_id:
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.SESSION_MISMATCH,
            "observation session does not match runtime session",
        )

    key = ObservationKey(
        session_id=observation.session_id,
        shot_sequence=observation.shot_sequence,
        evidence_sha256=observation.evidence_sha256,
    )
    for receipt in snapshot.observation_receipts:
        if receipt.key == key:
            return RuntimeTransitionResult(
                snapshot=snapshot,
                transition=receipt.transition,
                effects=receipt.effects,
                replayed=True,
            )
    for receipt in snapshot.observation_outcome_receipts:
        if receipt.key == key:
            return RuntimeTransitionResult(
                snapshot=snapshot,
                transition=receipt.transition,
                effects=receipt.effects,
                replayed=True,
            )

    if snapshot.state in _TERMINAL_STATES:
        return _invalid(snapshot, command)

    same_sequence = any(
        receipt.key.session_id == key.session_id and receipt.key.shot_sequence == key.shot_sequence
        for receipt in snapshot.observation_receipts
    )
    observation_states = {
        CoachRuntimeState.WAITING_FOR_SHOT,
        CoachRuntimeState.ANALYZING_SHOT,
    }
    if snapshot.state not in observation_states:
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.OBSERVATION_OUT_OF_STATE,
            "new observations are accepted only while waiting for or analyzing a shot",
        )
    if same_sequence:
        return _degrade(snapshot, command, RuntimeErrorCode.EVIDENCE_CONFLICT)

    if snapshot.state is CoachRuntimeState.ANALYZING_SHOT:
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.OBSERVATION_OUT_OF_STATE,
            "only a conflicting digest is meaningful while analyzing a shot",
        )
    if observation.shot_sequence < snapshot.shot_sequence:
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.STALE_OBSERVATION,
            "observation sequence is older than the expected shot",
        )
    if observation.shot_sequence > snapshot.shot_sequence:
        return _degrade(snapshot, command, RuntimeErrorCode.SEQUENCE_GAP)

    result = _make_transition(
        snapshot,
        command,
        CoachRuntimeState.ANALYZING_SHOT,
        RuntimeEffectKind.ANALYZE_SHOT,
    )
    receipt = ObservationReceipt(
        key=key,
        transition=result.transition,
        effects=result.effects,
    )
    return replace(
        result,
        snapshot=replace(
            result.snapshot,
            observation_receipts=(*snapshot.observation_receipts, receipt),
        ),
    )


def transition(
    snapshot: RuntimeSnapshot,
    command: RuntimeCommand,
) -> RuntimeTransitionResult:
    """Apply one command without reading clocks, randomness, I/O, or mutable state."""

    snapshot_error = _snapshot_invariant_error(snapshot)
    if snapshot_error is not None:
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.INVALID_SNAPSHOT,
            snapshot_error,
        )

    if isinstance(command, _KNOWN_COMMAND_TYPES) and (
        type(command.command_id) is not str or not command.command_id.strip()
    ):
        return _error(
            snapshot,
            command,
            RuntimeErrorCode.INVALID_COMMAND,
            "command_id must be a non-empty string",
        )

    if isinstance(command, AcceptObservation):
        return _accept_observation(snapshot, command)

    if isinstance(command, BeginCompletion) and snapshot.state in {
        CoachRuntimeState.COMPLETING,
        CoachRuntimeState.COMPLETED,
    }:
        if snapshot.completion_receipt is None:
            return _invalid(snapshot, command)
        return RuntimeTransitionResult(
            snapshot=snapshot,
            transition=snapshot.completion_receipt.transition,
            effects=snapshot.completion_receipt.effects,
            replayed=True,
        )

    if isinstance(command, FinishCompletion) and snapshot.state is CoachRuntimeState.COMPLETED:
        if snapshot.completion_finished_receipt is None:
            return _invalid(snapshot, command)
        return RuntimeTransitionResult(
            snapshot=snapshot,
            transition=snapshot.completion_finished_receipt.transition,
            effects=snapshot.completion_finished_receipt.effects,
            replayed=True,
        )

    if snapshot.state in _TERMINAL_STATES:
        return _invalid(snapshot, command)

    if isinstance(command, BeginPlanLoad):
        if snapshot.state is not CoachRuntimeState.IDLE:
            return _invalid(snapshot, command)
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.LOADING_PLAN,
            RuntimeEffectKind.PLAN_LOADING,
        )

    if isinstance(command, AdmitVerifiedPlan):
        if snapshot.state is not CoachRuntimeState.LOADING_PLAN:
            return _invalid(snapshot, command)
        if not _valid_admission(command.admission):
            return _error(
                snapshot,
                command,
                RuntimeErrorCode.INVALID_PLAN_ADMISSION,
                "plan input did not cross the verified admission seam",
            )
        admission = command.admission
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.READY,
            RuntimeEffectKind.PLAN_READY,
            session_id=admission.session_id,
            plan=AdmittedPlanIdentity(
                admission_id=admission.admission_id,
                plan_id=admission.plan_id,
                plan_version=admission.plan_version,
                execution_projection=admission.execution_projection,
            ),
            shot_sequence=1,
            block_index=0,
        )

    if isinstance(command, PresentSetup):
        if snapshot.state is not CoachRuntimeState.READY:
            return _invalid(snapshot, command)
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.PRESENTING_SETUP,
            RuntimeEffectKind.PRESENT_SETUP,
        )

    if isinstance(command, WaitForShot):
        if snapshot.state is not CoachRuntimeState.PRESENTING_SETUP:
            return _invalid(snapshot, command)
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.WAITING_FOR_SHOT,
            RuntimeEffectKind.WAIT_FOR_SHOT,
        )

    if isinstance(command, PresentFeedback):
        if snapshot.state is not CoachRuntimeState.ANALYZING_SHOT:
            return _invalid(snapshot, command)
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.PRESENTING_FEEDBACK,
            RuntimeEffectKind.PRESENT_FEEDBACK,
        )

    if isinstance(command, AdvanceToNextShot):
        if snapshot.state is not CoachRuntimeState.PRESENTING_FEEDBACK:
            return _invalid(snapshot, command)
        if (
            type(command.next_shot_sequence) is not int
            or type(command.next_block_index) is not int
            or command.next_shot_sequence != snapshot.shot_sequence + 1
            or command.next_block_index < 0
        ):
            return _error(
                snapshot,
                command,
                RuntimeErrorCode.INVALID_ADVANCE,
                "external advance must name the next shot and a non-negative block",
            )
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.PRESENTING_SETUP,
            RuntimeEffectKind.PRESENT_SETUP,
            shot_sequence=command.next_shot_sequence,
            block_index=command.next_block_index,
        )

    if isinstance(command, PauseRuntime):
        if snapshot.state not in _ACTIVITY_STATES:
            return _invalid(snapshot, command)
        if type(command.reason) is not str or not command.reason.strip():
            return _error(
                snapshot,
                command,
                RuntimeErrorCode.INVALID_TRANSITION,
                "pause reason must be a non-empty string",
            )
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.PAUSED,
            RuntimeEffectKind.PAUSED,
            resume_state=snapshot.state,
        )

    if isinstance(command, ResumeRuntime):
        if snapshot.state is not CoachRuntimeState.PAUSED or snapshot.resume_state is None:
            return _invalid(snapshot, command)
        return _make_transition(
            snapshot,
            command,
            snapshot.resume_state,
            RuntimeEffectKind.RESUMED,
            increment_transition_count=False,
            resume_state=None,
        )

    if isinstance(command, RecoverRuntime):
        valid_hash = _is_sha256(command.consistency_sha256)
        valid_session = _is_uuid4(command.session_id)
        valid_sequence = (
            type(command.expected_shot_sequence) is int and command.expected_shot_sequence >= 1
        )
        if snapshot.state is not CoachRuntimeState.DEGRADED:
            return _invalid(snapshot, command)
        expected_proof = expected_recovery_proof(snapshot)
        if (
            not valid_session
            or not valid_sequence
            or command.session_id != snapshot.session_id
            or command.expected_shot_sequence != snapshot.shot_sequence
            or not valid_hash
            or expected_proof is None
            or not hmac.compare_digest(
                command.consistency_sha256,
                expected_proof,
            )
        ):
            return _error(
                snapshot,
                command,
                RuntimeErrorCode.RECOVERY_PROOF_MISMATCH,
                "consistency proof does not match the degraded snapshot",
            )
        return _make_transition(
            snapshot,
            command,
            snapshot.recovery_state,
            RuntimeEffectKind.RECOVERED,
            degraded_reason=None,
            last_trusted_state=None,
            recovery_state=None,
        )

    if isinstance(command, BeginCompletion):
        if snapshot.state not in _ACTIVITY_STATES | {
            CoachRuntimeState.PAUSED,
            CoachRuntimeState.DEGRADED,
        }:
            return _invalid(snapshot, command)
        result = _make_transition(
            snapshot,
            command,
            CoachRuntimeState.COMPLETING,
            RuntimeEffectKind.COMPLETE_SESSION,
            resume_state=None,
        )
        receipt = CompletionReceipt(result.transition, result.effects)
        return replace(result, snapshot=replace(result.snapshot, completion_receipt=receipt))

    if isinstance(command, FinishCompletion):
        if snapshot.state is not CoachRuntimeState.COMPLETING:
            return _invalid(snapshot, command)
        result = _make_transition(
            snapshot,
            command,
            CoachRuntimeState.COMPLETED,
            RuntimeEffectKind.SESSION_COMPLETED,
        )
        receipt = CompletionReceipt(result.transition, result.effects)
        return replace(
            result,
            snapshot=replace(result.snapshot, completion_finished_receipt=receipt),
        )

    if isinstance(command, FailRuntime):
        if type(command.reason_code) is not str or not command.reason_code.strip():
            return _error(
                snapshot,
                command,
                RuntimeErrorCode.INVALID_TRANSITION,
                "failure reason code must be a non-empty string",
            )
        return _make_transition(
            snapshot,
            command,
            CoachRuntimeState.FAILED,
            RuntimeEffectKind.RUNTIME_FAILED,
            failure_reason=command.reason_code,
        )

    return _invalid(snapshot, command)
