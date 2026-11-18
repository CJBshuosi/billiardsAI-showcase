"""Strict, deterministic inputs for the offline CoachRuntime replay path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator
from pydantic_core import to_jsonable_python

from billiards_ai.edge.coach_runtime.execution_policy import (
    EvaluationConclusion,
    ExecutionDecisionKind,
    ExecutionPolicyResult,
    ExecutionPolicyState,
    ReplanIntent,
    VerifiedExecutionProjection,
    apply_observation,
)
from billiards_ai.edge.coach_runtime.feedback import build_realtime_coaching_event
from billiards_ai.edge.coach_runtime.state_machine import (
    AcceptObservation,
    AdmitVerifiedPlan,
    AdvanceToNextShot,
    BeginCompletion,
    BeginPlanLoad,
    CoachRuntimeState,
    FinishCompletion,
    PauseRuntime,
    PresentFeedback,
    PresentSetup,
    RuntimeCommand,
    RuntimeSnapshot,
    VerifiedPlanAdmission,
    WaitForShot,
    canonical_bytes,
    transition,
)
from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr, Sha256, require_utc
from billiards_ai.shared.models.growth import (
    BlockOutcome,
    PlanOutcome,
    PlanOutcomeStatus,
)
from billiards_ai.shared.models.realtime import (
    RealtimeCoachingEventV2,
    RealtimeShotObservation,
    canonical_realtime_coaching_event_bytes,
)

MAX_REPLAY_PLAN_FIXTURE_BYTES = 65_536
MAX_REALTIME_OBSERVATION_LINE_BYTES = 65_536
MAX_REALTIME_OBSERVATIONS_BYTES = 8_388_608
MAX_REALTIME_OBSERVATION_RECORDS = 1_000
COACH_RUNTIME_REPLAY_VERSION = "coach-runtime-v1"


class ReplayEvidenceLineageV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    shot_sequence: StrictInt = Field(ge=1)
    evidence_envelope_id: UUID4


class ReplayPlanFixtureV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    admission_id: NonEmptyStr
    bundle_id: UUID4
    bundle_content_sha256: Sha256
    projection_sha256: Sha256
    session_started_at: datetime
    outcome_recorded_at: datetime
    session_started_monotonic_ms: StrictInt = Field(ge=0)
    runtime_version: Literal["coach-runtime-v1"] = COACH_RUNTIME_REPLAY_VERSION
    evidence_lineage: tuple[ReplayEvidenceLineageV1, ...] = Field(default=())

    _session_started_at_utc = field_validator("session_started_at")(require_utc)
    _outcome_recorded_at_utc = field_validator("outcome_recorded_at")(require_utc)

    @model_validator(mode="after")
    def timeline_and_lineage_are_canonical(self) -> ReplayPlanFixtureV1:
        if self.outcome_recorded_at < self.session_started_at:
            raise ValueError("outcome recording cannot precede session start")
        sequences = tuple(item.shot_sequence for item in self.evidence_lineage)
        envelope_ids = tuple(item.evidence_envelope_id for item in self.evidence_lineage)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise ValueError("evidence lineage must have unique canonical shot sequence order")
        if len(envelope_ids) != len(set(envelope_ids)):
            raise ValueError("evidence envelope lineage must be unique")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def canonical_replay_plan_fixture_bytes(fixture: ReplayPlanFixtureV1) -> bytes:
    if type(fixture) is not ReplayPlanFixtureV1:
        raise TypeError("fixture must be an exact ReplayPlanFixtureV1")
    validated = ReplayPlanFixtureV1.model_validate(fixture.model_dump(), strict=True)
    data = _canonical_json_bytes(validated.model_dump(mode="json"))
    if len(data) > MAX_REPLAY_PLAN_FIXTURE_BYTES:
        raise ValueError("replay plan fixture exceeds maximum wire size")
    return data


def decode_replay_plan_fixture(data: bytes) -> ReplayPlanFixtureV1:
    if type(data) is not bytes:
        raise TypeError("replay plan fixture input must be bytes")
    if not data or len(data) > MAX_REPLAY_PLAN_FIXTURE_BYTES:
        raise ValueError("invalid replay plan fixture wire size")
    try:
        json.loads(data, object_pairs_hook=_reject_duplicate_keys)
        fixture = ReplayPlanFixtureV1.model_validate_json(data)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("invalid replay plan fixture JSON") from exc
    if canonical_replay_plan_fixture_bytes(fixture) != data:
        raise ValueError("replay plan fixture JSON must be canonical")
    return fixture


def _canonical_observation_bytes(observation: RealtimeShotObservation) -> bytes:
    if type(observation) is not RealtimeShotObservation:
        raise TypeError("observation must be an exact RealtimeShotObservation")
    validated = RealtimeShotObservation.model_validate(observation.model_dump(), strict=True)
    return _canonical_json_bytes(validated.model_dump(mode="json"))


def decode_replay_observations(
    data: bytes,
    *,
    expected_session_id: UUID,
) -> tuple[RealtimeShotObservation, ...]:
    if type(data) is not bytes:
        raise TypeError("replay observations input must be bytes")
    if (
        not isinstance(expected_session_id, UUID)
        or expected_session_id.version != 4
        or not data
        or len(data) > MAX_REALTIME_OBSERVATIONS_BYTES
        or not data.endswith(b"\n")
        or b"\r" in data
    ):
        raise ValueError("invalid replay observations JSONL envelope or wire size")
    if data.count(b"\n") > MAX_REALTIME_OBSERVATION_RECORDS:
        raise ValueError("replay observations exceed maximum record count")
    lines = data.split(b"\n")[:-1]
    if not lines or any(not line for line in lines):
        raise ValueError("replay observations JSONL cannot contain empty lines")

    observations: list[RealtimeShotObservation] = []
    seen_sequences: dict[int, bytes] = {}
    seen_event_ids: set[UUID] = set()
    next_sequence = 1
    for line in lines:
        if len(line) > MAX_REALTIME_OBSERVATION_LINE_BYTES:
            raise ValueError("replay observation exceeds maximum line size")
        try:
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            observation = RealtimeShotObservation.model_validate_json(line)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("invalid replay observation JSON") from exc
        if _canonical_observation_bytes(observation) != line:
            raise ValueError("replay observation JSON must be canonical")
        if observation.session_id != expected_session_id:
            raise ValueError("replay observation session does not match admitted plan")
        previous = seen_sequences.get(observation.shot_sequence)
        if previous is not None:
            if previous != line:
                raise ValueError("conflicting replay observation sequence")
            observations.append(observation)
            continue
        if observation.shot_sequence != next_sequence:
            raise ValueError("replay observation sequence gap")
        if observation.event_id in seen_event_ids:
            raise ValueError("replay observation event identity conflict")
        seen_sequences[observation.shot_sequence] = line
        seen_event_ids.add(observation.event_id)
        observations.append(observation)
        next_sequence += 1
    return tuple(observations)


@dataclass(frozen=True, slots=True)
class _ReplayLedgerEntry:
    block_index: int
    shot_sequence: int
    conclusion: EvaluationConclusion


@dataclass(frozen=True, slots=True)
class ReplayResultV1:
    schema_version: str
    runtime_version: str
    admission_id: str
    projection_sha256: str
    terminal_runtime_state: CoachRuntimeState
    terminal_policy_safe_paused: bool
    input_observation_count: int
    consumed_observation_count: int
    replayed_observation_count: int
    remaining_observation_count: int
    coaching_events: tuple[RealtimeCoachingEventV2, ...]
    replan_intents: tuple[ReplanIntent, ...]
    plan_outcome: PlanOutcome
    content_sha256: str

    def __post_init__(self) -> None:
        expected_outcome_status = (
            PlanOutcomeStatus.COMPLETED
            if self.terminal_runtime_state is CoachRuntimeState.COMPLETED
            else PlanOutcomeStatus.ABORTED
            if self.terminal_runtime_state is CoachRuntimeState.FAILED
            else PlanOutcomeStatus.PARTIAL
        )
        if self.plan_outcome.status is not expected_outcome_status:
            raise ValueError("replay runtime and outcome status mismatch")
        if (
            self.terminal_policy_safe_paused
            and self.terminal_runtime_state is not CoachRuntimeState.PAUSED
        ):
            raise ValueError("safe-paused policy requires paused runtime")
        counts = (
            self.input_observation_count,
            self.consumed_observation_count,
            self.replayed_observation_count,
            self.remaining_observation_count,
        )
        if (
            self.schema_version != "1.0"
            or self.runtime_version != COACH_RUNTIME_REPLAY_VERSION
            or type(self.admission_id) is not str
            or not self.admission_id.strip()
            or not _is_sha256(self.projection_sha256)
            or type(self.terminal_runtime_state) is not CoachRuntimeState
            or type(self.terminal_policy_safe_paused) is not bool
            or any(type(value) is not int or value < 0 for value in counts)
            or self.input_observation_count
            != self.consumed_observation_count
            + self.replayed_observation_count
            + self.remaining_observation_count
            or tuple(event.sequence for event in self.coaching_events)
            != tuple(range(1, len(self.coaching_events) + 1))
            or any(type(event) is not RealtimeCoachingEventV2 for event in self.coaching_events)
            or any(type(intent) is not ReplanIntent for intent in self.replan_intents)
            or type(self.plan_outcome) is not PlanOutcome
            or not _is_sha256(self.content_sha256)
        ):
            raise ValueError("invalid replay result invariant")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(domain: str, material: object) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + canonical_bytes(material)).hexdigest()


def _uuid4_from_digest(digest: str) -> UUID:
    value = bytearray(bytes.fromhex(digest)[:16])
    value[6] = (value[6] & 0x0F) | 0x40
    value[8] = (value[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(value))


def _result_material(result: ReplayResultV1) -> dict[str, object]:
    return {
        "schema_version": result.schema_version,
        "runtime_version": result.runtime_version,
        "admission_id": result.admission_id,
        "projection_sha256": result.projection_sha256,
        "terminal_runtime_state": result.terminal_runtime_state,
        "terminal_policy_safe_paused": result.terminal_policy_safe_paused,
        "input_observation_count": result.input_observation_count,
        "consumed_observation_count": result.consumed_observation_count,
        "replayed_observation_count": result.replayed_observation_count,
        "remaining_observation_count": result.remaining_observation_count,
        "coaching_events": result.coaching_events,
        "replan_intents": result.replan_intents,
        "plan_outcome": result.plan_outcome,
    }


def canonical_replay_result_bytes(result: ReplayResultV1) -> bytes:
    if type(result) is not ReplayResultV1:
        raise TypeError("result must be an exact ReplayResultV1")
    if result.schema_version != "1.0":
        raise ValueError("unsupported replay result schema version")
    if any(type(event) is not RealtimeCoachingEventV2 for event in result.coaching_events):
        raise TypeError("replay result events must be exact RealtimeCoachingEventV2 values")
    for event in result.coaching_events:
        canonical_realtime_coaching_event_bytes(event)
    if any(type(intent) is not ReplanIntent for intent in result.replan_intents):
        raise TypeError("replay result intents must be exact ReplanIntent values")
    outcome = PlanOutcome.model_validate(result.plan_outcome.model_dump(), strict=True)
    if any(
        event.lineage.admission_id != result.admission_id
        or event.lineage.projection_sha256 != result.projection_sha256
        or event.lineage.session_id != outcome.session_id
        or event.lineage.plan_id != outcome.plan_id
        or event.lineage.plan_version != outcome.plan_version
        for event in result.coaching_events
    ):
        raise ValueError("replay coaching event lineage mismatch")
    if any(
        intent.schema_version != "1.0"
        or not _is_sha256(intent.intent_id)
        or intent.admission_id != result.admission_id
        or intent.session_id != outcome.session_id
        or intent.plan_id != outcome.plan_id
        or type(intent.block_index) is not int
        or not 0 <= intent.block_index < len(outcome.blocks)
        or type(intent.shot_sequence) is not int
        or intent.shot_sequence < 1
        or not _is_sha256(intent.evidence_sha256)
        for intent in result.replan_intents
    ):
        raise ValueError("replay replan intent lineage mismatch")
    outcome_values = outcome.model_dump(mode="python")
    outcome_content = {
        key: value
        for key, value in outcome_values.items()
        if key not in {"outcome_id", "content_sha256"}
    }
    if outcome.content_sha256 != _digest("plan-outcome-replay-content/v1", outcome_content):
        raise ValueError("replay plan outcome content hash mismatch")
    outcome_identity = _digest(
        "plan-outcome-replay-identity/v1",
        (
            result.projection_sha256,
            outcome.plan_id,
            outcome.session_id,
            outcome.started_at,
            outcome.completed_at,
        ),
    )
    if outcome.outcome_id != _uuid4_from_digest(outcome_identity):
        raise ValueError("replay plan outcome identity mismatch")
    expected = _digest("coach-runtime-replay-result/v1", _result_material(result))
    if result.content_sha256 != expected:
        raise ValueError("replay result content hash mismatch")
    return canonical_bytes(result)


def _validate_replay_admission(
    admission: VerifiedPlanAdmission,
    fixture: ReplayPlanFixtureV1,
) -> VerifiedExecutionProjection:
    if type(admission) is not VerifiedPlanAdmission:
        raise TypeError("replay requires an exact VerifiedPlanAdmission")
    projection = admission.execution_projection
    if (
        type(projection) is not VerifiedExecutionProjection
        or not projection.integrity_valid()
        or not isinstance(projection.user_id, UUID)
        or projection.user_id.version != 4
        or admission.admission_id != projection.admission_id
        or admission.session_id != projection.session_id
        or admission.plan_id != projection.plan_id
        or admission.plan_version != projection.plan_version
        or fixture.admission_id != projection.admission_id
        or fixture.bundle_id != projection.bundle_id
        or fixture.bundle_content_sha256 != projection.bundle_content_sha256
        or fixture.projection_sha256 != projection.projection_sha256
        or not all(
            fixture.session_started_at < expiry
            for expiry in (
                projection.expires_at,
                projection.plan_valid_until,
                projection.authorization_expires_at,
                projection.signing_key_expires_at,
            )
        )
    ):
        raise ValueError("replay fixture does not match verified plan admission or validity")
    return projection


def _validate_observation_timeline(
    fixture: ReplayPlanFixtureV1,
    observations: tuple[RealtimeShotObservation, ...],
) -> None:
    seen_sequences: set[int] = set()
    last_monotonic_ms = fixture.session_started_monotonic_ms
    last_observed_at = fixture.session_started_at
    for observation in observations:
        if not fixture.session_started_at <= observation.observed_at <= fixture.outcome_recorded_at:
            raise ValueError("replay observation is outside explicit fixture timeline")
        shot = observation.shot
        if (
            fixture.session_started_monotonic_ms + shot.t_start_ms
            >= fixture.session_started_monotonic_ms + shot.t_end_ms
            or fixture.session_started_monotonic_ms + shot.t_end_ms
            > observation.monotonic_timestamp_ms
            or not fixture.session_started_at <= shot.created_at <= observation.observed_at
        ):
            raise ValueError("replay shot timeline is outside observation and fixture timeline")
        if observation.shot_sequence in seen_sequences:
            continue
        if (
            observation.monotonic_timestamp_ms < last_monotonic_ms
            or observation.observed_at < last_observed_at
        ):
            raise ValueError("replay observation timeline must be monotonic")
        seen_sequences.add(observation.shot_sequence)
        last_monotonic_ms = observation.monotonic_timestamp_ms
        last_observed_at = observation.observed_at


def _apply_runtime_command(
    *,
    projection: VerifiedExecutionProjection,
    snapshot: RuntimeSnapshot,
    command: RuntimeCommand,
    event_sequence: int,
    created_monotonic_ms: int,
    policy_state: ExecutionPolicyState | None = None,
    policy_result: ExecutionPolicyResult | None = None,
    observation: RealtimeShotObservation | None = None,
) -> tuple[RuntimeSnapshot, RealtimeCoachingEventV2 | None]:
    result = transition(snapshot, command)
    if result.error is not None or result.transition is None or len(result.effects) != 1:
        raise ValueError("replay runtime command failed closed")
    effect = result.effects[0]
    if isinstance(command, BeginPlanLoad):
        return result.snapshot, None
    event = build_realtime_coaching_event(
        projection,
        effect,
        runtime_before=snapshot,
        runtime_command=command,
        sequence=event_sequence,
        created_monotonic_ms=created_monotonic_ms,
        policy_state=policy_state,
        policy_result=policy_result,
        observation=observation,
    )
    return result.snapshot, event


def _build_plan_outcome(
    *,
    projection: VerifiedExecutionProjection,
    fixture: ReplayPlanFixtureV1,
    runtime_state: CoachRuntimeState,
    ledger: tuple[_ReplayLedgerEntry, ...],
) -> PlanOutcome:
    blocks = tuple(
        BlockOutcome(
            block_id=block.block_id,
            attempted_shots=sum(item.block_index == index for item in ledger),
            completed_shots=sum(
                item.block_index == index
                and item.conclusion in {EvaluationConclusion.SUCCESS, EvaluationConclusion.FAILURE}
                for item in ledger
            ),
        )
        for index, block in enumerate(projection.blocks)
    )
    consumed_sequences = {item.shot_sequence for item in ledger}
    evidence_envelope_ids = tuple(
        item.evidence_envelope_id
        for item in fixture.evidence_lineage
        if item.shot_sequence in consumed_sequences
    )
    status = (
        PlanOutcomeStatus.COMPLETED
        if runtime_state is CoachRuntimeState.COMPLETED
        else PlanOutcomeStatus.ABORTED
        if runtime_state is CoachRuntimeState.FAILED
        else PlanOutcomeStatus.PARTIAL
    )
    material: dict[str, object] = {
        "schema_version": "1.0",
        "plan_id": projection.plan_id,
        "plan_version": projection.plan_version,
        "user_id": projection.user_id,
        "session_id": projection.session_id,
        "device_id": projection.target_device_id,
        "started_at": fixture.session_started_at,
        "completed_at": fixture.outcome_recorded_at,
        "status": status,
        "blocks": blocks,
        "evidence_envelope_ids": evidence_envelope_ids,
        "direct_ability_update_allowed": False,
    }
    identity = _digest(
        "plan-outcome-replay-identity/v1",
        (
            projection.projection_sha256,
            projection.plan_id,
            projection.session_id,
            fixture.session_started_at,
            fixture.outcome_recorded_at,
        ),
    )
    content_sha256 = _digest("plan-outcome-replay-content/v1", material)
    return PlanOutcome(
        outcome_id=_uuid4_from_digest(identity),
        content_sha256=content_sha256,
        **material,
    )


def run_replay(
    admission: VerifiedPlanAdmission,
    fixture_bytes: bytes,
    observations_jsonl: bytes,
) -> ReplayResultV1:
    fixture = decode_replay_plan_fixture(fixture_bytes)
    projection = _validate_replay_admission(admission, fixture)
    observations = decode_replay_observations(
        observations_jsonl,
        expected_session_id=projection.session_id,
    )
    _validate_observation_timeline(fixture, observations)
    events: list[RealtimeCoachingEventV2] = []
    intents: list[ReplanIntent] = []
    ledger: list[_ReplayLedgerEntry] = []
    event_sequence = 1

    def apply(
        snapshot: RuntimeSnapshot,
        command: RuntimeCommand,
        monotonic_ms: int,
        *,
        policy_state: ExecutionPolicyState | None = None,
        policy_result: ExecutionPolicyResult | None = None,
        observation: RealtimeShotObservation | None = None,
    ) -> RuntimeSnapshot:
        nonlocal event_sequence
        next_snapshot, event = _apply_runtime_command(
            projection=projection,
            snapshot=snapshot,
            command=command,
            event_sequence=event_sequence,
            created_monotonic_ms=monotonic_ms,
            policy_state=policy_state,
            policy_result=policy_result,
            observation=observation,
        )
        if event is not None:
            events.append(event)
            event_sequence += 1
        return next_snapshot

    snapshot = RuntimeSnapshot.initial()
    snapshot = apply(snapshot, BeginPlanLoad(), fixture.session_started_monotonic_ms)
    snapshot = apply(
        snapshot,
        AdmitVerifiedPlan(admission),
        fixture.session_started_monotonic_ms,
    )
    snapshot = apply(snapshot, PresentSetup(), fixture.session_started_monotonic_ms)
    snapshot = apply(snapshot, WaitForShot(), fixture.session_started_monotonic_ms)
    policy_state = ExecutionPolicyState.start(projection)

    seen: set[tuple[int, str]] = set()
    replayed_count = 0
    processed_line_count = 0
    for observation in observations:
        key = (observation.shot_sequence, observation.evidence_sha256)
        if key in seen:
            runtime_replay = transition(snapshot, AcceptObservation(observation))
            policy_replay = apply_observation(policy_state, observation)
            if not runtime_replay.replayed or not policy_replay.replayed:
                raise ValueError("exact replay did not preserve runtime and policy receipts")
            replayed_count += 1
            processed_line_count += 1
            continue

        seen.add(key)
        monotonic_ms = observation.monotonic_timestamp_ms
        snapshot = apply(snapshot, AcceptObservation(observation), monotonic_ms)
        policy_before = policy_state
        policy_result = apply_observation(policy_before, observation)
        snapshot = apply(
            snapshot,
            PresentFeedback(),
            monotonic_ms,
            policy_state=policy_before,
            policy_result=policy_result,
            observation=observation,
        )
        ledger.append(
            _ReplayLedgerEntry(
                block_index=policy_before.block_index,
                shot_sequence=observation.shot_sequence,
                conclusion=policy_result.evaluation.conclusion,
            )
        )
        if policy_result.decision.replan_intent is not None:
            intents.append(policy_result.decision.replan_intent)
        policy_state = policy_result.state
        processed_line_count += 1

        if policy_result.decision.kind is ExecutionDecisionKind.BEGIN_COMPLETION:
            snapshot = apply(snapshot, BeginCompletion(), monotonic_ms)
            snapshot = apply(snapshot, FinishCompletion(), monotonic_ms)
            break
        if policy_result.decision.kind is ExecutionDecisionKind.SAFE_PAUSE:
            snapshot = apply(
                snapshot,
                PauseRuntime(reason="execution-policy-safe-pause"),
                monotonic_ms,
            )
            break
        snapshot = apply(
            snapshot,
            AdvanceToNextShot(
                next_shot_sequence=policy_result.decision.next_shot_sequence,
                next_block_index=policy_result.decision.next_block_index,
            ),
            monotonic_ms,
        )
        snapshot = apply(snapshot, WaitForShot(), monotonic_ms)

    outcome = _build_plan_outcome(
        projection=projection,
        fixture=fixture,
        runtime_state=snapshot.state,
        ledger=tuple(ledger),
    )
    material = {
        "schema_version": "1.0",
        "runtime_version": fixture.runtime_version,
        "admission_id": projection.admission_id,
        "projection_sha256": projection.projection_sha256,
        "terminal_runtime_state": snapshot.state,
        "terminal_policy_safe_paused": policy_state.safe_paused,
        "input_observation_count": len(observations),
        "consumed_observation_count": len(ledger),
        "replayed_observation_count": replayed_count,
        "remaining_observation_count": len(observations) - processed_line_count,
        "coaching_events": tuple(events),
        "replan_intents": tuple(intents),
        "plan_outcome": outcome,
    }
    result = ReplayResultV1(
        **material,
        content_sha256=_digest("coach-runtime-replay-result/v1", material),
    )
    canonical_replay_result_bytes(result)
    return result


__all__ = [
    "COACH_RUNTIME_REPLAY_VERSION",
    "MAX_REPLAY_PLAN_FIXTURE_BYTES",
    "MAX_REALTIME_OBSERVATION_LINE_BYTES",
    "MAX_REALTIME_OBSERVATIONS_BYTES",
    "MAX_REALTIME_OBSERVATION_RECORDS",
    "ReplayEvidenceLineageV1",
    "ReplayPlanFixtureV1",
    "ReplayResultV1",
    "canonical_replay_result_bytes",
    "canonical_replay_plan_fixture_bytes",
    "decode_replay_plan_fixture",
    "decode_replay_observations",
    "run_replay",
]
