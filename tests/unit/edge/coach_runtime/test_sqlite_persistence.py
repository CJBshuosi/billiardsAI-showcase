from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from billiards_ai.edge.coach_runtime.execution_policy import (
    AbstentionAction,
    DrillExecutionProjection,
    ExecutionBlockProjection,
    ExecutionPolicyState,
    MetricCombination,
    MetricRuleProjection,
    ReplanContinueAction,
    ReplanIntent,
    SuccessPolicyProjection,
    _issue_verified_execution_projection,
    apply_observation,
)
from billiards_ai.edge.coach_runtime.persistence import (
    PersistenceErrorCode,
    RecoveryStatus,
    RuntimePersistenceError,
    SqliteRuntimeStore,
)
from billiards_ai.edge.coach_runtime.state_machine import (
    AcceptObservation,
    AdmitVerifiedPlan,
    AdvanceToNextShot,
    BeginCompletion,
    BeginPlanLoad,
    FinishCompletion,
    PresentFeedback,
    PresentSetup,
    RuntimeSnapshot,
    VerifiedPlanAdmission,
    WaitForShot,
    transition,
)
from billiards_ai.shared.models.realtime import RealtimeShotObservation

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "core_intelligence" / "v1"
SESSION_ID = UUID("00000000-0000-4000-8000-000000000002")
USER_ID = UUID("00000000-0000-4000-8000-000000000001")
PLAN_ID = UUID("00000000-0000-4000-8000-000000000004")
ROOT_KEY = bytes.fromhex("11" * 32)
NOW = datetime(2026, 8, 18, 8, tzinfo=UTC)


def _drill(
    drill_id: str = "primary",
    *,
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
        abstention_count=2,
        abstention_action=(
            AbstentionAction.USE_PUBLISHED_FALLBACK
            if fallback_key is not None
            else AbstentionAction.SAFE_PAUSE
        ),
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
        replan_reason_codes=("FALLBACK_UNAVAILABLE",),
        replan_continue_action=(
            ReplanContinueAction.USE_PUBLISHED_FALLBACK
            if fallback_key is not None
            else ReplanContinueAction.SAFE_PAUSE
        ),
        fallback_key=fallback_key,
        target_zone=(100, 200, 300, 400),
    )


def _projection():
    return _issue_verified_execution_projection(
        admission_id="admission-1",
        bundle_id=UUID("00000000-0000-4000-8000-000000000010"),
        bundle_content_sha256="a" * 64,
        user_id=USER_ID,
        session_id=SESSION_ID,
        plan_id=PLAN_ID,
        plan_version=3,
        target_device_id="edge-1",
        publication_sequence=7,
        expires_at=datetime(2026, 8, 20, tzinfo=UTC),
        plan_valid_until=datetime(2026, 8, 20, tzinfo=UTC),
        authorization_id=UUID("00000000-0000-4000-8000-000000000012"),
        authorization_version=1,
        authorization_expires_at=datetime(2026, 8, 20, tzinfo=UTC),
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
                target_repetitions=2,
                drill=_drill(),
            ),
        ),
        terminal_fallback=_drill("fallback", fallback_key=None),
    )


def _ready_result():
    projection = _projection()
    snapshot = transition(RuntimeSnapshot.initial(), BeginPlanLoad()).snapshot
    admitted = transition(
        snapshot,
        AdmitVerifiedPlan(
            VerifiedPlanAdmission(
                admission_id=projection.admission_id,
                session_id=projection.session_id,
                plan_id=projection.plan_id,
                plan_version=projection.plan_version,
                execution_projection=projection,
            )
        ),
    )
    assert admitted.error is None
    setup = transition(admitted.snapshot, PresentSetup())
    waiting = transition(setup.snapshot, WaitForShot())
    observation = _observation()
    result = transition(waiting.snapshot, AcceptObservation(observation))
    assert result.error is None
    policy = apply_observation(ExecutionPolicyState.start(projection), observation).state
    return result, policy


def _completed_result():
    ready, policy = _ready_result()
    completing = transition(ready.snapshot, BeginCompletion())
    completed = transition(completing.snapshot, FinishCompletion())
    assert completed.error is None
    return completed, policy


def _observation() -> RealtimeShotObservation:
    return RealtimeShotObservation.model_validate_json(
        (FIXTURE_ROOT / "realtime-shot-observation.json").read_bytes()
    )


def _intent() -> ReplanIntent:
    return ReplanIntent(
        schema_version="1.0",
        intent_id="f" * 64,
        admission_id="admission-1",
        session_id=SESSION_ID,
        plan_id=PLAN_ID,
        block_index=0,
        reason_code="FALLBACK_UNAVAILABLE",
        shot_sequence=1,
        evidence_sha256="9" * 64,
    )


def _store(tmp_path: Path, **kwargs) -> SqliteRuntimeStore:
    return SqliteRuntimeStore(
        tmp_path / "runtime.sqlite3",
        device_root_key=ROOT_KEY,
        now=lambda: NOW,
        **kwargs,
    )


def _ready_commit_kwargs(result) -> dict[str, object]:
    observation = _observation()
    return {
        "previous_snapshot": replace(
            result.snapshot,
            state=result.transition.from_state,
            transition_count=result.snapshot.transition_count - 1,
            observation_receipts=result.snapshot.observation_receipts[:-1],
        ),
        "command": AcceptObservation(observation),
        "input_observation": observation,
    }


def _completed_commit_kwargs(result) -> dict[str, object]:
    return {
        "previous_snapshot": replace(
            result.snapshot,
            state=result.transition.from_state,
            transition_count=result.snapshot.transition_count - 1,
            completion_finished_receipt=None,
        ),
        "command": FinishCompletion(),
    }


def test_catalog_migrates_once_and_rejects_unknown_future_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.schema_version == 2
    store.close()
    assert _store(tmp_path).schema_version == 2

    with sqlite3.connect(tmp_path / "runtime.sqlite3") as connection:
        connection.execute("UPDATE runtime_catalog SET schema_version = 999")
    with pytest.raises(RuntimePersistenceError) as caught:
        _store(tmp_path)
    assert caught.value.code is PersistenceErrorCode.UNSUPPORTED_MIGRATION


@pytest.mark.parametrize(
    "fault_stage",
    ("after_session", "after_event", "after_outbox", "before_commit"),
)
def test_transition_snapshot_event_and_outbox_are_atomic(tmp_path: Path, fault_stage: str) -> None:
    def fail(stage: str) -> None:
        if stage == fault_stage:
            raise OSError("injected crash")

    store = _store(tmp_path, fault_injector=fail)
    result, policy = _ready_result()
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(
            result,
            policy,
            outbox=(_intent(),),
            **_ready_commit_kwargs(result),
        )
    assert caught.value.code is PersistenceErrorCode.TRANSACTION_FAILED
    assert store.catalog_counts() == {
        "sessions": 0,
        "runtime_events": 0,
        "runtime_inbox": 0,
        "runtime_outbox": 0,
        "runtime_receipts": 0,
        "runtime_outbox_acks": 0,
    }


def test_commit_unknown_is_exactly_replayable(tmp_path: Path) -> None:
    fired = False

    def fail_once(stage: str) -> None:
        nonlocal fired
        if stage == "after_commit" and not fired:
            fired = True
            raise OSError("commit result unknown")

    result, policy = _ready_result()
    store = _store(tmp_path, fault_injector=fail_once)
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(result, policy, **_ready_commit_kwargs(result))
    assert caught.value.code is PersistenceErrorCode.COMMIT_UNKNOWN
    receipt = store.commit_transition(result, policy, **_ready_commit_kwargs(result))
    assert receipt.replayed is True
    assert receipt.snapshot == result.snapshot
    assert store.catalog_counts()["runtime_events"] == 2


def test_restart_restores_exact_snapshot_policy_and_authority_outbox(tmp_path: Path) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    receipt = store.commit_transition(
        result,
        policy,
        outbox=(_intent(),),
        **_ready_commit_kwargs(result),
    )
    assert receipt.replayed is False
    store.close()

    reopened = _store(tmp_path)
    recovered = reopened.recover_session(SESSION_ID)
    assert recovered.status is RecoveryStatus.EXACT
    assert recovered.snapshot == result.snapshot
    assert recovered.policy_state == policy
    pending = reopened.pending_outbox(SESSION_ID)
    assert len(pending) == 1
    assert pending[0].payload == _intent()
    assert pending[0].content_sha256 == hashlib.sha256(pending[0].plaintext).hexdigest()


def test_inbox_and_ack_replay_are_idempotent_but_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    first = store.commit_transition(
        result,
        policy,
        outbox=(_intent(),),
        **_ready_commit_kwargs(result),
    )
    second = store.commit_transition(
        result,
        policy,
        outbox=(_intent(),),
        **_ready_commit_kwargs(result),
    )
    assert second == first.with_replayed(True)

    pending = store.pending_outbox(SESSION_ID)[0]
    store.ack_outbox(
        pending.event_id,
        pending.event_type,
        pending.content_sha256,
        acked_at=NOW,
    )
    store.ack_outbox(
        pending.event_id,
        pending.event_type,
        pending.content_sha256,
        acked_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(RuntimePersistenceError) as caught:
        store.ack_outbox(
            pending.event_id,
            pending.event_type,
            "0" * 64,
            acked_at=NOW,
        )
    assert caught.value.code is PersistenceErrorCode.IDENTITY_CONFLICT
    assert store.pending_outbox(SESSION_ID) == ()


def test_inbox_replay_cannot_rebind_original_transition_or_outbox(tmp_path: Path) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    store.commit_transition(
        result,
        policy,
        outbox=(_intent(),),
        **_ready_commit_kwargs(result),
    )
    conflicting_intent = replace(_intent(), reason_code="TARGET_OUT_OF_RANGE")
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(
            result,
            policy,
            outbox=(conflicting_intent,),
            **_ready_commit_kwargs(result),
        )
    assert caught.value.code is PersistenceErrorCode.IDENTITY_CONFLICT


def test_append_only_trigger_and_same_input_identity_conflict(tmp_path: Path) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    store.commit_transition(result, policy, **_ready_commit_kwargs(result))
    conflicting = _observation().model_copy(update={"perception_model_version": "tampered-v2"})
    with pytest.raises(RuntimePersistenceError) as caught:
        kwargs = _ready_commit_kwargs(result)
        kwargs["command"] = AcceptObservation(conflicting)
        kwargs["input_observation"] = conflicting
        store.commit_transition(result, policy, **kwargs)
    assert caught.value.code is PersistenceErrorCode.IDENTITY_CONFLICT

    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE runtime_events SET sequence = sequence + 1")


def test_stale_and_gapped_runtime_transitions_cannot_replace_authority(tmp_path: Path) -> None:
    first, policy = _ready_result()
    store = _store(tmp_path)
    store.commit_transition(first, policy, **_ready_commit_kwargs(first))

    stale = store.commit_transition(first, policy, **_ready_commit_kwargs(first))
    assert stale.replayed is True

    feedback = transition(first.snapshot, PresentFeedback())
    skipped = transition(
        feedback.snapshot,
        AdvanceToNextShot(next_shot_sequence=2, next_block_index=0),
    )
    assert skipped.error is None
    with pytest.raises(RuntimePersistenceError) as gap:
        store.commit_transition(
            skipped,
            policy,
            previous_snapshot=feedback.snapshot,
            command=AdvanceToNextShot(next_shot_sequence=2, next_block_index=0),
        )
    assert gap.value.code is PersistenceErrorCode.IDENTITY_CONFLICT
    recovered = store.recover_session(SESSION_ID)
    assert recovered.snapshot == first.snapshot


def test_tamper_wrong_key_and_deleted_key_only_return_safe_recovery(tmp_path: Path) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    store.commit_transition(result, policy, **_ready_commit_kwargs(result))
    store.close()

    wrong_key = SqliteRuntimeStore(
        tmp_path / "runtime.sqlite3",
        device_root_key=bytes.fromhex("22" * 32),
        now=lambda: NOW,
    )
    recovered = wrong_key.recover_session(SESSION_ID)
    assert recovered.status is RecoveryStatus.PAUSED
    assert recovered.executable is False
    assert recovered.error is PersistenceErrorCode.KEY_UNAVAILABLE
    wrong_key.close()

    with sqlite3.connect(tmp_path / "runtime.sqlite3") as connection:
        connection.execute(
            "UPDATE runtime_sessions SET snapshot_ciphertext = "
            "randomblob(length(snapshot_ciphertext))"
        )
    tampered = _store(tmp_path).recover_session(SESSION_ID)
    assert tampered.status is RecoveryStatus.DEGRADED
    assert tampered.executable is False
    assert tampered.error is PersistenceErrorCode.INTEGRITY_MISMATCH


def test_raw_database_bytes_do_not_expose_sensitive_current_session_data(
    tmp_path: Path,
) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    store.commit_transition(
        result,
        policy,
        outbox=(_intent(),),
        **_ready_commit_kwargs(result),
    )
    store.checkpoint()
    store.close()
    raw = (tmp_path / "runtime.sqlite3").read_bytes()
    forbidden = (
        str(USER_ID).encode(),
        str(SESSION_ID).encode(),
        str(PLAN_ID).encode(),
        b"openid-secret",
        b"phone-number",
        b"raw_frame",
        b"trajectory",
        b"media_locator",
        b"prompt",
    )
    assert all(value not in raw.lower() for value in forbidden)


def test_retention_ack_ttl_quarantine_and_crypto_erasure(tmp_path: Path) -> None:
    completed, policy = _completed_result()
    store = _store(tmp_path)
    store.commit_transition(
        completed,
        policy,
        outbox=(_intent(),),
        committed_at=NOW,
        **_completed_commit_kwargs(completed),
    )
    pending = store.pending_outbox(SESSION_ID)[0]
    report = store.run_retention(now=NOW + timedelta(days=8))
    assert report.quarantined_session_ids == (SESSION_ID,)
    assert report.deleted_session_ids == ()
    assert store.accepting_new_sessions is False

    store.ack_outbox(
        pending.event_id,
        pending.event_type,
        pending.content_sha256,
        acked_at=NOW + timedelta(days=8),
    )
    assert store.run_retention(now=NOW + timedelta(days=8, hours=23)).deleted_session_ids == ()
    assert store.run_retention(now=NOW + timedelta(days=9, hours=1)).deleted_session_ids == (
        SESSION_ID,
    )

    other_path = tmp_path / "erase.sqlite3"
    other = SqliteRuntimeStore(other_path, device_root_key=ROOT_KEY, now=lambda: NOW)
    ready, policy = _ready_result()
    other.commit_transition(ready, policy, **_ready_commit_kwargs(ready))
    other.erase_session(SESSION_ID, erased_at=NOW)
    erased = other.recover_session(SESSION_ID)
    assert erased.status is RecoveryStatus.PAUSED
    assert erased.error is PersistenceErrorCode.KEY_UNAVAILABLE
    assert other.catalog_counts()["sessions"] == 0


def test_database_lock_is_typed_retryable_and_never_bypasses_transaction(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, busy_timeout_ms=1)
    result, policy = _ready_result()
    lock = sqlite3.connect(store.path, isolation_level=None)
    lock.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimePersistenceError) as caught:
            store.commit_transition(result, policy, **_ready_commit_kwargs(result))
        assert caught.value.code is PersistenceErrorCode.DB_LOCKED
        assert caught.value.retryable is True
    finally:
        lock.rollback()
        lock.close()
    assert store.catalog_counts()["sessions"] == 0


def test_only_frozen_authority_outbox_types_are_allowed(tmp_path: Path) -> None:
    result, policy = _ready_result()
    store = _store(tmp_path)
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(
            result,
            policy,
            outbox=({"free": "dict"},),
            **_ready_commit_kwargs(result),
        )
    assert caught.value.code is PersistenceErrorCode.INVALID_PAYLOAD
