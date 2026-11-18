from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest

from billiards_ai.edge.coach_runtime.execution_policy import (
    _issue_verified_execution_projection,
)
from billiards_ai.shared.models.realtime import RealtimeShotObservation
from billiards_ai.shared.models.training_plan import PublishedPlanBundle

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "core_intelligence" / "v1"
SESSION_ID = UUID("00000000-0000-4000-8000-000000000002")
PLAN_ID = UUID("00000000-0000-4000-8000-000000000004")


def _projection(
    runtime: ModuleType, *, admission_id: str, plan_id: UUID, plan_version: int
) -> object:
    metric = runtime.MetricRuleProjection(
        metric_code="POT_SUCCESS_COUNT",
        target_attempts=10,
        minimum_successes=6,
        evaluator_code="POT_RESULT",
        evaluator_version="1.0",
        required_capabilities=("TABLE_MM_COORDINATES",),
    )
    policy = runtime.SuccessPolicyProjection(
        evaluator_policy_version="edge-success-v1",
        combination=runtime.MetricCombination.ALL,
        metric_rules=(metric,),
    )
    fallback = runtime.DrillExecutionProjection(
        drill_id="fallback",
        drill_revision=1,
        drill_content_sha256="2" * 64,
        required_capabilities=("TABLE_MM_COORDINATES",),
        initial_difficulty=2,
        minimum_runtime_difficulty=1,
        maximum_runtime_difficulty=3,
        progression_count=2,
        progression_delta=1,
        regression_count=2,
        regression_delta=-1,
        abstention_count=2,
        abstention_action=runtime.AbstentionAction.SAFE_PAUSE,
        success_policy=policy,
        replan_reason_codes=("FALLBACK_UNAVAILABLE",),
        replan_continue_action=runtime.ReplanContinueAction.SAFE_PAUSE,
        fallback_key=None,
        target_zone=None,
    )
    primary = runtime.DrillExecutionProjection(
        drill_id="primary",
        drill_revision=1,
        drill_content_sha256="1" * 64,
        required_capabilities=("TABLE_MM_COORDINATES",),
        initial_difficulty=2,
        minimum_runtime_difficulty=1,
        maximum_runtime_difficulty=3,
        progression_count=2,
        progression_delta=1,
        regression_count=2,
        regression_delta=-1,
        abstention_count=2,
        abstention_action=runtime.AbstentionAction.USE_PUBLISHED_FALLBACK,
        success_policy=policy,
        replan_reason_codes=("FALLBACK_UNAVAILABLE",),
        replan_continue_action=runtime.ReplanContinueAction.USE_PUBLISHED_FALLBACK,
        fallback_key=("fallback", 1),
        target_zone=None,
    )
    return _issue_verified_execution_projection(
        admission_id=admission_id,
        bundle_id=UUID("00000000-0000-4000-8000-000000000099"),
        bundle_content_sha256="a" * 64,
        user_id=UUID("00000000-0000-4000-8000-000000000001"),
        session_id=SESSION_ID,
        plan_id=plan_id,
        plan_version=plan_version,
        target_device_id="edge-1",
        publication_sequence=1,
        expires_at=datetime(2026, 8, 10, tzinfo=UTC),
        plan_valid_until=datetime(2026, 8, 11, tzinfo=UTC),
        authorization_id=UUID("00000000-0000-4000-8000-000000000098"),
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
            runtime.ExecutionBlockProjection(
                block_id=UUID("00000000-0000-4000-8000-000000000097"),
                order=1,
                target_repetitions=10,
                drill=primary,
            ),
        ),
        terminal_fallback=fallback,
    )


@dataclass(frozen=True)
class StructuralPlanAdmission:
    admission_id: str = "verified-admission:plan-4:session-2"
    session_id: UUID = SESSION_ID
    plan_id: UUID = PLAN_ID
    plan_version: int = 1


@dataclass(frozen=True)
class StructuralObservation:
    session_id: UUID = SESSION_ID
    shot_sequence: int = 1
    evidence_sha256: str = "9" * 64


def _admission(
    runtime: ModuleType,
    *,
    admission_id: str = "verified-admission:plan-4:session-2",
    plan_id: UUID = PLAN_ID,
    plan_version: int = 1,
) -> object:
    projection = _projection(
        runtime, admission_id=admission_id, plan_id=plan_id, plan_version=plan_version
    )
    return runtime.VerifiedPlanAdmission(
        admission_id=admission_id,
        session_id=SESSION_ID,
        plan_id=plan_id,
        plan_version=plan_version,
        execution_projection=projection,
    )


def _runtime() -> ModuleType:
    try:
        return importlib.import_module("billiards_ai.edge.coach_runtime")
    except ModuleNotFoundError:
        pytest.fail("CoachRuntime package is not implemented")


def _observation(
    *,
    sequence: int = 1,
    digest: str = "9" * 64,
    session_id: UUID = SESSION_ID,
) -> RealtimeShotObservation:
    base = RealtimeShotObservation.model_validate_json(
        (FIXTURE_ROOT / "realtime-shot-observation.json").read_text(encoding="utf-8")
    )
    return base.model_copy(
        update={
            "event_id": UUID(f"10000000-0000-4000-8000-{sequence:012d}"),
            "session_id": session_id,
            "shot_sequence": sequence,
            "evidence_sha256": digest,
            "shot": base.shot.model_copy(
                update={
                    "shot_id": UUID(f"00000000-0000-4000-8000-{sequence:012d}"),
                    "session_id": session_id,
                    "sequence": sequence,
                }
            ),
        }
    )


def _apply(runtime: ModuleType, snapshot: object, command: object) -> object:
    result = runtime.transition(snapshot, command)
    assert result.error is None
    return result


def _ready(runtime: ModuleType) -> object:
    snapshot = runtime.RuntimeSnapshot.initial()
    snapshot = _apply(runtime, snapshot, runtime.BeginPlanLoad()).snapshot
    return _apply(
        runtime,
        snapshot,
        runtime.AdmitVerifiedPlan(_admission(runtime)),
    ).snapshot


def _waiting(runtime: ModuleType) -> object:
    snapshot = _ready(runtime)
    snapshot = _apply(runtime, snapshot, runtime.PresentSetup()).snapshot
    return _apply(runtime, snapshot, runtime.WaitForShot()).snapshot


def test_runtime_defines_the_frozen_state_and_error_code_contract() -> None:
    runtime = _runtime()

    assert tuple(state.value for state in runtime.CoachRuntimeState) == (
        "IDLE",
        "LOADING_PLAN",
        "READY",
        "PRESENTING_SETUP",
        "WAITING_FOR_SHOT",
        "ANALYZING_SHOT",
        "PRESENTING_FEEDBACK",
        "PAUSED",
        "COMPLETING",
        "COMPLETED",
        "DEGRADED",
        "FAILED",
    )
    assert tuple(code.value for code in runtime.RuntimeErrorCode) == (
        "INVALID_TRANSITION",
        "INVALID_COMMAND",
        "INVALID_SNAPSHOT",
        "INVALID_OBSERVATION",
        "INVALID_PLAN_ADMISSION",
        "SESSION_MISMATCH",
        "OBSERVATION_OUT_OF_STATE",
        "STALE_OBSERVATION",
        "SEQUENCE_GAP",
        "EVIDENCE_CONFLICT",
        "RECOVERY_PROOF_MISMATCH",
        "INVALID_ADVANCE",
        "TERMINAL_STATE",
    )
    snapshot = runtime.RuntimeSnapshot.initial()
    assert snapshot.state is runtime.CoachRuntimeState.IDLE
    assert snapshot.transition_count == 0
    with pytest.raises(FrozenInstanceError):
        snapshot.transition_count = 99


def test_legal_main_chain_is_command_driven_and_uses_stable_effect_identities() -> None:
    runtime = _runtime()
    snapshot = runtime.RuntimeSnapshot.initial()
    commands = (
        runtime.BeginPlanLoad(),
        runtime.AdmitVerifiedPlan(_admission(runtime)),
        runtime.PresentSetup(),
        runtime.WaitForShot(),
        runtime.AcceptObservation(_observation()),
        runtime.PresentFeedback(),
        runtime.AdvanceToNextShot(next_shot_sequence=2, next_block_index=0),
        runtime.WaitForShot(),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
        runtime.PresentFeedback(),
        runtime.BeginCompletion(),
        runtime.FinishCompletion(),
    )
    expected_states = (
        "LOADING_PLAN",
        "READY",
        "PRESENTING_SETUP",
        "WAITING_FOR_SHOT",
        "ANALYZING_SHOT",
        "PRESENTING_FEEDBACK",
        "PRESENTING_SETUP",
        "WAITING_FOR_SHOT",
        "ANALYZING_SHOT",
        "PRESENTING_FEEDBACK",
        "COMPLETING",
        "COMPLETED",
    )

    identities: list[tuple[str, tuple[str, ...]]] = []
    for command, expected_state in zip(commands, expected_states, strict=True):
        result = _apply(runtime, snapshot, command)
        assert result.snapshot.state.value == expected_state
        assert result.transition is not None
        identities.append(
            (result.transition.transition_id, tuple(effect.effect_id for effect in result.effects))
        )
        snapshot = result.snapshot

    assert snapshot.transition_count == len(commands)
    assert snapshot.shot_sequence == 2
    assert snapshot.block_index == 0
    assert len(set(identities)) == len(identities)


def test_raw_published_bundle_cannot_cross_the_verified_plan_admission_seam() -> None:
    runtime = _runtime()
    bundle = PublishedPlanBundle.model_validate_json(
        (FIXTURE_ROOT / "published-plan-bundle.json").read_text(encoding="utf-8")
    )
    loading = _apply(
        runtime,
        runtime.RuntimeSnapshot.initial(),
        runtime.BeginPlanLoad(),
    ).snapshot
    before = runtime.canonical_bytes(loading)

    result = runtime.transition(loading, runtime.AdmitVerifiedPlan(bundle))

    assert result.error is not None
    assert result.error.code is runtime.RuntimeErrorCode.INVALID_PLAN_ADMISSION
    assert result.snapshot is loading
    assert runtime.canonical_bytes(result.snapshot) == before


def test_structural_lookalike_cannot_cross_the_nominal_plan_admission_seam() -> None:
    runtime = _runtime()
    loading = _apply(
        runtime,
        runtime.RuntimeSnapshot.initial(),
        runtime.BeginPlanLoad(),
    ).snapshot

    result = runtime.transition(
        loading,
        runtime.AdmitVerifiedPlan(StructuralPlanAdmission()),
    )

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_PLAN_ADMISSION
    assert result.snapshot is loading


def test_exact_projection_with_stale_integrity_hash_cannot_enter_ready() -> None:
    runtime = _runtime()
    loading = _apply(runtime, runtime.RuntimeSnapshot.initial(), runtime.BeginPlanLoad()).snapshot
    admission = _admission(runtime)
    assert admission.execution_projection is not None
    forged = replace(
        admission,
        execution_projection=replace(
            admission.execution_projection,
            publication_sequence=admission.execution_projection.publication_sequence + 1,
        ),
    )
    result = runtime.transition(loading, runtime.AdmitVerifiedPlan(forged))
    assert result.error.code is runtime.RuntimeErrorCode.INVALID_PLAN_ADMISSION
    assert result.snapshot is loading


@pytest.mark.parametrize(
    ("session_id", "plan_id"),
    (
        (UUID(int=0), PLAN_ID),
        (SESSION_ID, UUID(int=0)),
    ),
)
def test_plan_admission_requires_uuid4_identity(session_id: UUID, plan_id: UUID) -> None:
    runtime = _runtime()
    loading = _apply(
        runtime,
        runtime.RuntimeSnapshot.initial(),
        runtime.BeginPlanLoad(),
    ).snapshot
    admission = runtime.VerifiedPlanAdmission(
        admission_id="verified-admission:invalid-uuid",
        session_id=session_id,
        plan_id=plan_id,
        plan_version=1,
    )

    result = runtime.transition(loading, runtime.AdmitVerifiedPlan(admission))

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_PLAN_ADMISSION
    assert result.snapshot is loading


@pytest.mark.parametrize(
    ("state_factory", "command_factory", "expected_code"),
    (
        (lambda r: r.RuntimeSnapshot.initial(), lambda r: r.PresentSetup(), "INVALID_TRANSITION"),
        (lambda r: _ready(r), lambda r: r.WaitForShot(), "INVALID_TRANSITION"),
        (lambda r: _waiting(r), lambda r: r.PresentFeedback(), "INVALID_TRANSITION"),
        (
            lambda r: _apply(r, _waiting(r), r.AcceptObservation(_observation())).snapshot,
            lambda r: r.WaitForShot(),
            "INVALID_TRANSITION",
        ),
        (
            lambda r: _apply(r, _waiting(r), r.PauseRuntime("user-request")).snapshot,
            lambda r: r.PresentSetup(),
            "INVALID_TRANSITION",
        ),
    ),
)
def test_illegal_transition_matrix_preserves_authoritative_snapshot_bytes(
    state_factory: object,
    command_factory: object,
    expected_code: str,
) -> None:
    runtime = _runtime()
    snapshot = state_factory(runtime)
    before = runtime.canonical_bytes(snapshot)

    result = runtime.transition(snapshot, command_factory(runtime))

    assert result.error is not None
    assert result.error.code.value == expected_code
    assert result.snapshot is snapshot
    assert runtime.canonical_bytes(result.snapshot) == before
    assert result.transition is None
    assert result.effects == ()


def test_exact_observation_replay_returns_original_ids_without_advancing_again() -> None:
    runtime = _runtime()
    waiting = _waiting(runtime)
    command = runtime.AcceptObservation(_observation())

    first = _apply(runtime, waiting, command)
    replay = _apply(runtime, first.snapshot, command)

    assert replay.replayed is True
    assert replay.snapshot is first.snapshot
    assert replay.snapshot.transition_count == waiting.transition_count + 1
    assert replay.snapshot.accepted_observation_count == 1
    assert replay.transition.transition_id == first.transition.transition_id
    assert tuple(effect.effect_id for effect in replay.effects) == tuple(
        effect.effect_id for effect in first.effects
    )


@pytest.mark.parametrize(
    ("sequence", "digest", "expected_accepted_count"),
    (
        (2, "8" * 64, 0),
        (1, "7" * 64, 1),
    ),
)
def test_degrading_observation_replay_returns_first_outcome_identity_once(
    sequence: int,
    digest: str,
    expected_accepted_count: int,
) -> None:
    runtime = _runtime()
    snapshot = _waiting(runtime)
    if sequence == 1:
        snapshot = _apply(
            runtime,
            snapshot,
            runtime.AcceptObservation(_observation()),
        ).snapshot
    command = runtime.AcceptObservation(_observation(sequence=sequence, digest=digest))

    first = _apply(runtime, snapshot, command)
    replay = _apply(runtime, first.snapshot, command)

    assert first.snapshot.state is runtime.CoachRuntimeState.DEGRADED
    assert first.snapshot.accepted_observation_count == expected_accepted_count
    assert len(first.snapshot.observation_outcome_receipts) == 1
    assert replay.replayed is True
    assert replay.snapshot is first.snapshot
    assert replay.snapshot.transition_count == first.snapshot.transition_count
    assert replay.transition.transition_id == first.transition.transition_id
    assert tuple(effect.effect_id for effect in replay.effects) == tuple(
        effect.effect_id for effect in first.effects
    )


@pytest.mark.parametrize("observation", (None, object(), StructuralObservation()))
def test_malformed_observation_returns_typed_error_without_mutating_snapshot(
    observation: object,
) -> None:
    runtime = _runtime()
    waiting = _waiting(runtime)
    before = runtime.canonical_bytes(waiting)

    result = runtime.transition(waiting, runtime.AcceptObservation(observation))

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_OBSERVATION
    assert result.snapshot is waiting
    assert runtime.canonical_bytes(result.snapshot) == before


@pytest.mark.parametrize("command_id", (None, "", "   ", object()))
def test_every_known_command_rejects_invalid_command_id_before_hashing(
    command_id: object,
) -> None:
    runtime = _runtime()
    initial = runtime.RuntimeSnapshot.initial()
    commands = (
        runtime.BeginPlanLoad(),
        runtime.AdmitVerifiedPlan(_admission(runtime)),
        runtime.PresentSetup(),
        runtime.WaitForShot(),
        runtime.AcceptObservation(_observation()),
        runtime.PresentFeedback(),
        runtime.AdvanceToNextShot(2, 0),
        runtime.PauseRuntime("test"),
        runtime.ResumeRuntime(),
        runtime.RecoverRuntime(SESSION_ID, 1, "a" * 64),
        runtime.BeginCompletion(),
        runtime.FinishCompletion(),
        runtime.FailRuntime("test"),
    )

    for command in commands:
        malformed = replace(command, command_id=command_id)
        result = runtime.transition(initial, malformed)
        assert result.error.code is runtime.RuntimeErrorCode.INVALID_COMMAND
        assert result.snapshot is initial


@pytest.mark.parametrize(
    ("observation", "expected_code"),
    (
        (_observation(sequence=2, digest="8" * 64), "SEQUENCE_GAP"),
        (
            _observation(session_id=UUID("00000000-0000-4000-8000-000000000099")),
            "SESSION_MISMATCH",
        ),
    ),
)
def test_observation_sequence_and_session_gates_fail_closed(
    observation: RealtimeShotObservation,
    expected_code: str,
) -> None:
    runtime = _runtime()
    waiting = _waiting(runtime)

    result = runtime.transition(waiting, runtime.AcceptObservation(observation))

    if expected_code == "SEQUENCE_GAP":
        assert result.error is None
        assert result.snapshot.state is runtime.CoachRuntimeState.DEGRADED
        assert result.snapshot.degraded_reason is runtime.RuntimeErrorCode.SEQUENCE_GAP
        assert result.snapshot.last_trusted_state is runtime.CoachRuntimeState.WAITING_FOR_SHOT
        assert result.snapshot.recovery_state is runtime.CoachRuntimeState.WAITING_FOR_SHOT
        assert result.effects[0].kind is runtime.RuntimeEffectKind.SAFE_PAUSE
    else:
        assert result.error is not None
        assert result.error.code.value == expected_code
        assert result.snapshot is waiting
        assert result.effects == ()


def test_same_sequence_with_different_digest_degrades_from_last_trusted_state() -> None:
    runtime = _runtime()
    first = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation()),
    )

    conflict = runtime.transition(
        first.snapshot,
        runtime.AcceptObservation(_observation(digest="7" * 64)),
    )

    assert conflict.error is None
    assert conflict.snapshot.state is runtime.CoachRuntimeState.DEGRADED
    assert conflict.snapshot.degraded_reason is runtime.RuntimeErrorCode.EVIDENCE_CONFLICT
    assert conflict.snapshot.last_trusted_state is runtime.CoachRuntimeState.ANALYZING_SHOT
    assert conflict.snapshot.recovery_state is runtime.CoachRuntimeState.ANALYZING_SHOT
    assert conflict.snapshot.accepted_observation_count == 1


def test_new_observation_outside_waiting_and_old_digest_conflict_never_queue() -> None:
    runtime = _runtime()
    ready = _ready(runtime)
    wrong_state = runtime.transition(ready, runtime.AcceptObservation(_observation()))
    waiting_for_two = _apply(
        runtime,
        _apply(
            runtime,
            _apply(
                runtime,
                _waiting(runtime),
                runtime.AcceptObservation(_observation()),
            ).snapshot,
            runtime.PresentFeedback(),
        ).snapshot,
        runtime.AdvanceToNextShot(next_shot_sequence=2, next_block_index=0),
    ).snapshot
    waiting_for_two = _apply(runtime, waiting_for_two, runtime.WaitForShot()).snapshot
    conflict = runtime.transition(
        waiting_for_two,
        runtime.AcceptObservation(_observation(sequence=1, digest="6" * 64)),
    )

    assert wrong_state.error.code is runtime.RuntimeErrorCode.OBSERVATION_OUT_OF_STATE
    assert wrong_state.snapshot is ready
    assert conflict.error is None
    assert conflict.snapshot.state is runtime.CoachRuntimeState.DEGRADED
    assert conflict.snapshot.degraded_reason is runtime.RuntimeErrorCode.EVIDENCE_CONFLICT
    assert conflict.snapshot.accepted_observation_count == 1


@pytest.mark.parametrize(
    "state_name",
    (
        "READY",
        "PRESENTING_SETUP",
        "WAITING_FOR_SHOT",
        "ANALYZING_SHOT",
        "PRESENTING_FEEDBACK",
    ),
)
def test_pause_and_resume_restore_the_exact_activity_state(state_name: str) -> None:
    runtime = _runtime()
    snapshots = {"READY": _ready(runtime)}
    snapshots["PRESENTING_SETUP"] = _apply(
        runtime, snapshots["READY"], runtime.PresentSetup()
    ).snapshot
    snapshots["WAITING_FOR_SHOT"] = _apply(
        runtime, snapshots["PRESENTING_SETUP"], runtime.WaitForShot()
    ).snapshot
    snapshots["ANALYZING_SHOT"] = _apply(
        runtime,
        snapshots["WAITING_FOR_SHOT"],
        runtime.AcceptObservation(_observation()),
    ).snapshot
    snapshots["PRESENTING_FEEDBACK"] = _apply(
        runtime, snapshots["ANALYZING_SHOT"], runtime.PresentFeedback()
    ).snapshot
    active = snapshots[state_name]

    paused = _apply(runtime, active, runtime.PauseRuntime("user-request")).snapshot
    resumed = _apply(runtime, paused, runtime.ResumeRuntime()).snapshot

    assert paused.state is runtime.CoachRuntimeState.PAUSED
    assert paused.resume_state is active.state
    assert resumed.state is active.state
    assert resumed.resume_state is None
    assert resumed.session_id == active.session_id
    assert resumed.shot_sequence == active.shot_sequence
    assert resumed.block_index == active.block_index
    assert resumed.observation_receipts == active.observation_receipts
    assert paused.transition_count == active.transition_count + 1
    assert resumed.transition_count == paused.transition_count


def test_degraded_runtime_requires_matching_explicit_consistency_proof_to_recover() -> None:
    runtime = _runtime()
    degraded = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    before = runtime.canonical_bytes(degraded)
    proof = runtime.expected_recovery_proof(degraded)
    assert proof is not None
    assert not hasattr(degraded, "recovery_proof_sha256")

    rejected = runtime.transition(
        degraded,
        runtime.RecoverRuntime(
            session_id=SESSION_ID,
            expected_shot_sequence=1,
            consistency_sha256="a" * 64 if proof != "a" * 64 else "b" * 64,
        ),
    )
    recovered = _apply(
        runtime,
        degraded,
        runtime.RecoverRuntime(
            session_id=SESSION_ID,
            expected_shot_sequence=1,
            consistency_sha256=proof,
        ),
    ).snapshot

    assert rejected.error.code is runtime.RuntimeErrorCode.RECOVERY_PROOF_MISMATCH
    assert runtime.canonical_bytes(rejected.snapshot) == before
    assert recovered.state is runtime.CoachRuntimeState.WAITING_FOR_SHOT
    assert recovered.degraded_reason is None
    assert recovered.recovery_state is None


def test_recovery_proof_is_recomputed_and_bound_to_the_exact_degraded_snapshot() -> None:
    runtime = _runtime()
    gap = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    accepted = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation()),
    ).snapshot
    conflict = _apply(
        runtime,
        accepted,
        runtime.AcceptObservation(_observation(digest="7" * 64)),
    ).snapshot
    gap_proof = runtime.expected_recovery_proof(gap)
    conflict_proof = runtime.expected_recovery_proof(conflict)

    assert gap_proof != conflict_proof
    rejected = runtime.transition(
        gap,
        runtime.RecoverRuntime(
            session_id=SESSION_ID,
            expected_shot_sequence=1,
            consistency_sha256=conflict_proof,
        ),
    )
    assert rejected.error.code is runtime.RuntimeErrorCode.RECOVERY_PROOF_MISMATCH
    assert rejected.snapshot is gap


@pytest.mark.parametrize("expected_shot_sequence", (True, 1.0))
def test_recovery_requires_a_strict_integer_expected_sequence(
    expected_shot_sequence: object,
) -> None:
    runtime = _runtime()
    degraded = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    before = runtime.canonical_bytes(degraded)
    proof = runtime.expected_recovery_proof(degraded)
    assert proof is not None

    result = runtime.transition(
        degraded,
        runtime.RecoverRuntime(
            session_id=SESSION_ID,
            expected_shot_sequence=expected_shot_sequence,
            consistency_sha256=proof,
        ),
    )

    assert result.error.code is runtime.RuntimeErrorCode.RECOVERY_PROOF_MISMATCH
    assert runtime.canonical_bytes(result.snapshot) == before


@pytest.mark.parametrize(
    ("session_id", "consistency_sha256"),
    (
        (UUID(int=0), "a" * 64),
        (SESSION_ID, b"a" * 64),
    ),
)
def test_recovery_rejects_malformed_identity_inputs(
    session_id: object,
    consistency_sha256: object,
) -> None:
    runtime = _runtime()
    degraded = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    before = runtime.canonical_bytes(degraded)

    result = runtime.transition(
        degraded,
        runtime.RecoverRuntime(
            session_id=session_id,
            expected_shot_sequence=1,
            consistency_sha256=consistency_sha256,
        ),
    )

    assert result.error.code is runtime.RuntimeErrorCode.RECOVERY_PROOF_MISMATCH
    assert runtime.canonical_bytes(result.snapshot) == before


def test_degraded_safe_completion_preserves_the_degradation_evidence() -> None:
    runtime = _runtime()
    degraded = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    completing = _apply(runtime, degraded, runtime.BeginCompletion()).snapshot
    completed = _apply(runtime, completing, runtime.FinishCompletion()).snapshot

    for snapshot in (completing, completed):
        assert snapshot.degraded_reason is runtime.RuntimeErrorCode.SEQUENCE_GAP
        assert snapshot.last_trusted_state is runtime.CoachRuntimeState.WAITING_FOR_SHOT
        assert snapshot.recovery_state is runtime.CoachRuntimeState.WAITING_FOR_SHOT


def test_reconstructed_activity_snapshot_requires_uuid4_session_and_plan() -> None:
    runtime = _runtime()
    ready = _ready(runtime)
    forged_snapshots = (
        replace(ready, session_id=UUID(int=0)),
        replace(ready, plan=None),
        replace(
            ready,
            plan=runtime.AdmittedPlanIdentity(
                admission_id="verified-admission:forged",
                plan_id=UUID(int=0),
                plan_version=1,
            ),
        ),
    )

    for snapshot in forged_snapshots:
        before = runtime.canonical_bytes(snapshot)
        result = runtime.transition(snapshot, runtime.PresentSetup())

        assert result.error.code is runtime.RuntimeErrorCode.INVALID_SNAPSHOT
        assert runtime.canonical_bytes(result.snapshot) == before


@pytest.mark.parametrize(
    "resume_state_name",
    (None, "DEGRADED", "COMPLETED"),
)
def test_reconstructed_paused_snapshot_requires_an_activity_resume_state(
    resume_state_name: str | None,
) -> None:
    runtime = _runtime()
    resume_state = (
        None if resume_state_name is None else runtime.CoachRuntimeState[resume_state_name]
    )
    forged = replace(
        _ready(runtime),
        state=runtime.CoachRuntimeState.PAUSED,
        resume_state=resume_state,
    )
    before = runtime.canonical_bytes(forged)

    result = runtime.transition(forged, runtime.ResumeRuntime())

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_SNAPSHOT
    assert runtime.canonical_bytes(result.snapshot) == before


@pytest.mark.parametrize(
    "changes",
    (
        {"degraded_reason": None},
        {"last_trusted_state": None},
        {"recovery_state": "COMPLETED"},
        {"recovery_state": "FAILED"},
    ),
)
def test_reconstructed_degraded_snapshot_requires_recovery_invariants(
    changes: dict[str, object],
) -> None:
    runtime = _runtime()
    degraded = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    if isinstance(changes.get("recovery_state"), str):
        changes = {
            **changes,
            "recovery_state": runtime.CoachRuntimeState[changes["recovery_state"]],
        }
    forged = replace(degraded, **changes)
    before = runtime.canonical_bytes(forged)

    result = runtime.transition(
        forged,
        runtime.RecoverRuntime(
            session_id=SESSION_ID,
            expected_shot_sequence=1,
            consistency_sha256="a" * 64,
        ),
    )

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_SNAPSHOT
    assert runtime.canonical_bytes(result.snapshot) == before


def test_reconstructed_completing_snapshot_requires_begin_completion_receipt() -> None:
    runtime = _runtime()
    forged = replace(
        _ready(runtime),
        state=runtime.CoachRuntimeState.COMPLETING,
        completion_receipt=None,
    )
    before = runtime.canonical_bytes(forged)

    result = runtime.transition(forged, runtime.FinishCompletion())

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_SNAPSHOT
    assert runtime.canonical_bytes(result.snapshot) == before


def test_nonwaiting_digest_conflict_never_changes_authoritative_state() -> None:
    runtime = _runtime()
    analyzing = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(_observation()),
    ).snapshot
    feedback = _apply(runtime, analyzing, runtime.PresentFeedback()).snapshot
    paused = _apply(runtime, analyzing, runtime.PauseRuntime("user-request")).snapshot
    completing = _apply(runtime, feedback, runtime.BeginCompletion()).snapshot
    degraded = _apply(
        runtime,
        analyzing,
        runtime.AcceptObservation(_observation(digest="7" * 64)),
    ).snapshot

    for snapshot in (feedback, paused, completing, degraded):
        before = runtime.canonical_bytes(snapshot)
        result = runtime.transition(
            snapshot,
            runtime.AcceptObservation(_observation(digest="6" * 64)),
        )
        assert result.error.code is runtime.RuntimeErrorCode.OBSERVATION_OUT_OF_STATE
        assert runtime.canonical_bytes(result.snapshot) == before


@pytest.mark.parametrize(
    "reason",
    (None, "", "   ", b"bytes", object()),
)
def test_pause_rejects_invalid_reason_with_typed_error(reason: object) -> None:
    runtime = _runtime()
    ready = _ready(runtime)

    result = runtime.transition(ready, runtime.PauseRuntime(reason))

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_TRANSITION
    assert result.snapshot is ready


@pytest.mark.parametrize(
    "reason_code",
    (None, "", "   ", b"bytes", object()),
)
def test_fail_rejects_invalid_reason_code_with_typed_error(reason_code: object) -> None:
    runtime = _runtime()
    ready = _ready(runtime)

    result = runtime.transition(ready, runtime.FailRuntime(reason_code))

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_TRANSITION
    assert result.snapshot is ready


def test_completion_uses_two_steps_is_idempotent_and_completed_is_terminal() -> None:
    runtime = _runtime()
    active = _ready(runtime)
    completing = _apply(runtime, active, runtime.BeginCompletion())
    duplicate = _apply(runtime, completing.snapshot, runtime.BeginCompletion())
    completed = _apply(runtime, duplicate.snapshot, runtime.FinishCompletion())
    finish_replay = _apply(runtime, completed.snapshot, runtime.FinishCompletion())
    completed_again = _apply(runtime, completed.snapshot, runtime.BeginCompletion())
    observation = runtime.transition(
        completed_again.snapshot,
        runtime.AcceptObservation(_observation()),
    )

    assert completing.snapshot.state is runtime.CoachRuntimeState.COMPLETING
    assert duplicate.snapshot is completing.snapshot
    assert duplicate.transition.transition_id == completing.transition.transition_id
    assert completed.snapshot.state is runtime.CoachRuntimeState.COMPLETED
    assert finish_replay.snapshot is completed.snapshot
    assert finish_replay.replayed is True
    assert finish_replay.transition.transition_id == completed.transition.transition_id
    assert tuple(effect.effect_id for effect in finish_replay.effects) == tuple(
        effect.effect_id for effect in completed.effects
    )
    assert completed_again.snapshot is completed.snapshot
    assert completed_again.transition.transition_id == completing.transition.transition_id
    assert observation.error.code is runtime.RuntimeErrorCode.TERMINAL_STATE
    assert observation.snapshot is completed.snapshot


@pytest.mark.parametrize(
    ("next_shot_sequence", "next_block_index"),
    (
        (2.0, 0),
        ("2", 0),
        (2, 0.5),
        (2, "0"),
        (True, 0),
        (2, False),
    ),
)
def test_external_advance_rejects_non_strict_integers_without_raising(
    next_shot_sequence: object,
    next_block_index: object,
) -> None:
    runtime = _runtime()
    feedback = _apply(
        runtime,
        _apply(
            runtime,
            _waiting(runtime),
            runtime.AcceptObservation(_observation()),
        ).snapshot,
        runtime.PresentFeedback(),
    ).snapshot
    before = runtime.canonical_bytes(feedback)

    result = runtime.transition(
        feedback,
        runtime.AdvanceToNextShot(
            next_shot_sequence=next_shot_sequence,
            next_block_index=next_block_index,
        ),
    )

    assert result.error.code is runtime.RuntimeErrorCode.INVALID_ADVANCE
    assert runtime.canonical_bytes(result.snapshot) == before


def test_transition_identity_includes_admitted_plan_context() -> None:
    runtime = _runtime()

    def first_setup(admission: object) -> str:
        loading = _apply(
            runtime,
            runtime.RuntimeSnapshot.initial(),
            runtime.BeginPlanLoad(),
        ).snapshot
        ready = _apply(runtime, loading, runtime.AdmitVerifiedPlan(admission)).snapshot
        return _apply(runtime, ready, runtime.PresentSetup()).transition.transition_id

    first = first_setup(_admission(runtime))
    second = first_setup(
        _admission(
            runtime,
            admission_id="verified-admission:plan-5:session-2",
            plan_id=UUID("00000000-0000-4000-8000-000000000005"),
            plan_version=2,
        )
    )

    assert first != second


def test_state_by_command_matrix_is_exhaustive_and_fail_closed() -> None:
    runtime = _runtime()
    idle = runtime.RuntimeSnapshot.initial()
    loading = _apply(runtime, idle, runtime.BeginPlanLoad()).snapshot
    ready = _apply(
        runtime,
        loading,
        runtime.AdmitVerifiedPlan(_admission(runtime)),
    ).snapshot
    setup = _apply(runtime, ready, runtime.PresentSetup()).snapshot
    waiting = _apply(runtime, setup, runtime.WaitForShot()).snapshot
    analyzing = _apply(
        runtime,
        waiting,
        runtime.AcceptObservation(_observation()),
    ).snapshot
    feedback = _apply(runtime, analyzing, runtime.PresentFeedback()).snapshot
    paused = _apply(runtime, ready, runtime.PauseRuntime("matrix")).snapshot
    degraded = _apply(
        runtime,
        waiting,
        runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
    ).snapshot
    completing = _apply(runtime, ready, runtime.BeginCompletion()).snapshot
    completed = _apply(runtime, completing, runtime.FinishCompletion()).snapshot
    failed = _apply(runtime, ready, runtime.FailRuntime("matrix")).snapshot
    snapshots = {
        runtime.CoachRuntimeState.IDLE: idle,
        runtime.CoachRuntimeState.LOADING_PLAN: loading,
        runtime.CoachRuntimeState.READY: ready,
        runtime.CoachRuntimeState.PRESENTING_SETUP: setup,
        runtime.CoachRuntimeState.WAITING_FOR_SHOT: waiting,
        runtime.CoachRuntimeState.ANALYZING_SHOT: analyzing,
        runtime.CoachRuntimeState.PRESENTING_FEEDBACK: feedback,
        runtime.CoachRuntimeState.PAUSED: paused,
        runtime.CoachRuntimeState.DEGRADED: degraded,
        runtime.CoachRuntimeState.COMPLETING: completing,
        runtime.CoachRuntimeState.COMPLETED: completed,
        runtime.CoachRuntimeState.FAILED: failed,
    }
    state = runtime.CoachRuntimeState
    legal_destinations = {
        "BeginPlanLoad": {state.IDLE: state.LOADING_PLAN},
        "AdmitVerifiedPlan": {state.LOADING_PLAN: state.READY},
        "PresentSetup": {state.READY: state.PRESENTING_SETUP},
        "WaitForShot": {state.PRESENTING_SETUP: state.WAITING_FOR_SHOT},
        "AcceptObservation": {state.WAITING_FOR_SHOT: state.ANALYZING_SHOT},
        "PresentFeedback": {state.ANALYZING_SHOT: state.PRESENTING_FEEDBACK},
        "AdvanceToNextShot": {state.PRESENTING_FEEDBACK: state.PRESENTING_SETUP},
        "PauseRuntime": {
            state.READY: state.PAUSED,
            state.PRESENTING_SETUP: state.PAUSED,
            state.WAITING_FOR_SHOT: state.PAUSED,
            state.ANALYZING_SHOT: state.PAUSED,
            state.PRESENTING_FEEDBACK: state.PAUSED,
        },
        "ResumeRuntime": {state.PAUSED: state.READY},
        "RecoverRuntime": {state.DEGRADED: state.WAITING_FOR_SHOT},
        "BeginCompletion": {
            state.READY: state.COMPLETING,
            state.PRESENTING_SETUP: state.COMPLETING,
            state.WAITING_FOR_SHOT: state.COMPLETING,
            state.ANALYZING_SHOT: state.COMPLETING,
            state.PRESENTING_FEEDBACK: state.COMPLETING,
            state.PAUSED: state.COMPLETING,
            state.DEGRADED: state.COMPLETING,
            state.COMPLETING: state.COMPLETING,
            state.COMPLETED: state.COMPLETED,
        },
        "FinishCompletion": {
            state.COMPLETING: state.COMPLETED,
            state.COMPLETED: state.COMPLETED,
        },
        "FailRuntime": {
            state.IDLE: state.FAILED,
            state.LOADING_PLAN: state.FAILED,
            state.READY: state.FAILED,
            state.PRESENTING_SETUP: state.FAILED,
            state.WAITING_FOR_SHOT: state.FAILED,
            state.ANALYZING_SHOT: state.FAILED,
            state.PRESENTING_FEEDBACK: state.FAILED,
            state.PAUSED: state.FAILED,
            state.COMPLETING: state.FAILED,
            state.DEGRADED: state.FAILED,
        },
    }

    for state, snapshot in snapshots.items():
        observation_sequence = (
            snapshot.shot_sequence
            if state is runtime.CoachRuntimeState.WAITING_FOR_SHOT
            else max(1, snapshot.shot_sequence + 1)
        )
        commands = (
            runtime.BeginPlanLoad(),
            runtime.AdmitVerifiedPlan(_admission(runtime)),
            runtime.PresentSetup(),
            runtime.WaitForShot(),
            runtime.AcceptObservation(_observation(sequence=observation_sequence)),
            runtime.PresentFeedback(),
            runtime.AdvanceToNextShot(
                next_shot_sequence=snapshot.shot_sequence + 1,
                next_block_index=snapshot.block_index,
            ),
            runtime.PauseRuntime("matrix"),
            runtime.ResumeRuntime(),
            runtime.RecoverRuntime(
                session_id=snapshot.session_id or SESSION_ID,
                expected_shot_sequence=snapshot.shot_sequence,
                consistency_sha256=runtime.expected_recovery_proof(snapshot) or "a" * 64,
            ),
            runtime.BeginCompletion(),
            runtime.FinishCompletion(),
            runtime.FailRuntime("matrix"),
        )
        for command in commands:
            before = runtime.canonical_bytes(snapshot)
            result = runtime.transition(snapshot, command)
            expected_destination = legal_destinations[type(command).__name__].get(state)
            expected_legal = expected_destination is not None
            assert (result.error is None) is expected_legal, (state, command, result.error)
            if expected_legal:
                assert result.snapshot.state is expected_destination
            else:
                assert runtime.canonical_bytes(result.snapshot) == before


def test_failed_and_completed_are_terminal_states() -> None:
    runtime = _runtime()
    failed = _apply(runtime, _ready(runtime), runtime.FailRuntime("fatal-contract")).snapshot

    assert failed.state is runtime.CoachRuntimeState.FAILED
    for command in (
        runtime.PresentSetup(),
        runtime.PauseRuntime("ignored"),
        runtime.BeginCompletion(),
    ):
        result = runtime.transition(failed, command)
        assert result.error.code is runtime.RuntimeErrorCode.TERMINAL_STATE
        assert result.snapshot is failed


def test_two_complete_in_memory_replays_are_byte_equivalent() -> None:
    runtime = _runtime()

    def replay() -> tuple[bytes, tuple[bytes, ...]]:
        snapshot = runtime.RuntimeSnapshot.initial()
        results = []
        for command in (
            runtime.BeginPlanLoad(),
            runtime.AdmitVerifiedPlan(_admission(runtime)),
            runtime.PresentSetup(),
            runtime.WaitForShot(),
            runtime.AcceptObservation(_observation()),
            runtime.PauseRuntime("coach-break"),
            runtime.ResumeRuntime(),
            runtime.PresentFeedback(),
            runtime.AdvanceToNextShot(next_shot_sequence=2, next_block_index=1),
            runtime.WaitForShot(),
            runtime.AcceptObservation(_observation(sequence=2, digest="8" * 64)),
            runtime.PresentFeedback(),
            runtime.BeginCompletion(),
            runtime.FinishCompletion(),
        ):
            result = _apply(runtime, snapshot, command)
            snapshot = result.snapshot
            results.append(runtime.canonical_bytes(result))
        return runtime.canonical_bytes(snapshot), tuple(results)

    first = replay()
    second = replay()

    assert first == second


def test_completed_runtime_exact_observation_replay_keeps_original_effect_identity() -> None:
    runtime = _runtime()
    observation = _observation()
    first = _apply(
        runtime,
        _waiting(runtime),
        runtime.AcceptObservation(observation),
    )
    feedback = _apply(runtime, first.snapshot, runtime.PresentFeedback()).snapshot
    completing = _apply(runtime, feedback, runtime.BeginCompletion()).snapshot
    completed = _apply(runtime, completing, runtime.FinishCompletion()).snapshot

    replay = _apply(runtime, completed, runtime.AcceptObservation(observation))

    assert replay.replayed is True
    assert replay.snapshot is completed
    assert replay.transition.transition_id == first.transition.transition_id
    assert tuple(effect.effect_id for effect in replay.effects) == tuple(
        effect.effect_id for effect in first.effects
    )


def test_runtime_types_are_frozen_and_do_not_import_forbidden_systems() -> None:
    runtime = _runtime()
    result = _apply(
        runtime,
        runtime.RuntimeSnapshot.initial(),
        runtime.BeginPlanLoad(),
    )
    error = runtime.transition(result.snapshot, runtime.BeginPlanLoad()).error
    completing = _apply(runtime, _ready(runtime), runtime.BeginCompletion()).snapshot
    assert isinstance(completing.completion_receipt, runtime.CompletionReceipt)

    for value, field, replacement in (
        (runtime.BeginPlanLoad(), "command_id", "changed"),
        (_admission(runtime), "admission_id", "changed"),
        (result.transition, "transition_id", "changed"),
        (result.effects[0], "effect_id", "changed"),
        (result, "replayed", True),
        (error, "message", "changed"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


def test_runtime_import_does_not_load_forbidden_transitive_dependencies() -> None:
    project_root = Path(__file__).parents[4]
    source_root = project_root / "src"
    script = """
import sys
import billiards_ai.edge.coach_runtime
forbidden = (
    "billiards_ai.lab", "fastapi", "sqlite3", "sqlalchemy", "openai",
    "anthropic", "paho", "websockets", "psycopg",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if loaded:
    raise SystemExit("forbidden imports: " + ",".join(loaded))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
