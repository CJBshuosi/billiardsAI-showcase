from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from billiards_ai.edge.coach_runtime.persistence import (
    PersistenceErrorCode,
    RecoveryStatus,
    RuntimePersistenceError,
    SqliteRuntimeStore,
)
from billiards_ai.edge.coach_runtime.state_machine import (
    AcceptObservation,
    FinishCompletion,
    PauseRuntime,
    PresentFeedback,
    ResumeRuntime,
    RuntimeEffect,
    RuntimeEffectKind,
    transition,
)
from tests.unit.edge.coach_runtime.test_sqlite_persistence import (
    NOW,
    ROOT_KEY,
    SESSION_ID,
    _completed_result,
    _intent,
    _observation,
    _ready_result,
)


def _store(path: Path, **kwargs: object) -> SqliteRuntimeStore:
    return SqliteRuntimeStore(path, device_root_key=ROOT_KEY, now=lambda: NOW, **kwargs)


def _ready_authority() -> tuple[object, object, object, object]:
    result, policy = _ready_result()
    observation = _observation()
    previous = replace(
        result.snapshot,
        state=result.transition.from_state,
        transition_count=result.snapshot.transition_count - 1,
        observation_receipts=result.snapshot.observation_receipts[:-1],
    )
    return previous, AcceptObservation(observation), result, policy


def _commit_ready(store: SqliteRuntimeStore, *, outbox: tuple[object, ...] = ()):
    previous, command, result, policy = _ready_authority()
    return store.commit_transition(
        result,
        policy,
        previous_snapshot=previous,
        command=command,
        input_observation=_observation(),
        outbox=outbox,
    )


def test_review_recomputes_transition_and_uses_independent_storage_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = _store(path)
    receipt = _commit_ready(store)
    assert receipt.storage_revision == 1

    previous, command, result, policy = _ready_authority()
    forged = replace(
        result,
        effects=(
            RuntimeEffect(
                effect_id="f" * 64,
                kind=RuntimeEffectKind.ANALYZE_SHOT,
                session_id=SESSION_ID,
                shot_sequence=1,
            ),
        ),
    )
    other = _store(tmp_path / "forged.sqlite3")
    with pytest.raises(RuntimePersistenceError) as caught:
        other.commit_transition(
            forged,
            policy,
            previous_snapshot=previous,
            command=command,
            input_observation=_observation(),
        )
    assert caught.value.code is PersistenceErrorCode.INVALID_PAYLOAD


def test_review_resume_without_runtime_increment_still_advances_storage_cas(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "runtime.sqlite3")
    first = _commit_ready(store)
    _, _, ready, policy = _ready_authority()
    feedback = transition(ready.snapshot, PresentFeedback())
    second = store.commit_transition(
        feedback,
        policy,
        previous_snapshot=ready.snapshot,
        command=PresentFeedback(),
    )
    paused = transition(feedback.snapshot, PauseRuntime("network"))
    third = store.commit_transition(
        paused,
        policy,
        previous_snapshot=feedback.snapshot,
        command=PauseRuntime("network"),
    )
    resumed = transition(paused.snapshot, ResumeRuntime())
    assert resumed.snapshot.transition_count == paused.snapshot.transition_count
    fourth = store.commit_transition(
        resumed,
        policy,
        previous_snapshot=paused.snapshot,
        command=ResumeRuntime(),
    )
    assert (first.storage_revision, second.storage_revision, third.storage_revision) == (1, 2, 3)
    assert fourth.storage_revision == 4


def test_review_rejects_policy_snapshot_difficulty_and_receipt_inconsistency(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "runtime.sqlite3")
    previous, command, result, policy = _ready_authority()
    bad_policy = replace(policy, difficulty=999, expected_shot_sequence=99)
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(
            result,
            bad_policy,
            previous_snapshot=previous,
            command=command,
            input_observation=_observation(),
        )
    assert caught.value.code is PersistenceErrorCode.INVALID_PAYLOAD


def test_review_inbox_binds_exact_current_transition_observation(tmp_path: Path) -> None:
    store = _store(tmp_path / "runtime.sqlite3")
    previous, command, result, policy = _ready_authority()
    unrelated = _observation().model_copy(
        update={"event_id": "00000000-0000-4000-8000-000000000099"}
    )
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(
            result,
            policy,
            previous_snapshot=previous,
            command=command,
            input_observation=unrelated,
        )
    assert caught.value.code is PersistenceErrorCode.INVALID_PAYLOAD


def test_review_two_writers_claim_once_and_return_exact_replay(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    first_store = _store(path)
    second_store = _store(path)
    first = _commit_ready(first_store)
    second = _commit_ready(second_store)
    assert first.replayed is False
    assert second == first.with_replayed(True)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runtime_receipts").fetchone()[0] == 1


def test_review_no_observation_commit_unknown_has_durable_receipt(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    fired = False

    def fail_after_second_commit(stage: str) -> None:
        nonlocal fired
        if stage == "after_commit" and not fired:
            fired = True
            raise OSError("commit result unknown")

    store = _store(path, fault_injector=fail_after_second_commit)
    previous, command, result, policy = _ready_authority()
    with pytest.raises(RuntimePersistenceError):
        store.commit_transition(
            result,
            policy,
            previous_snapshot=previous,
            command=command,
            input_observation=_observation(),
        )
    replay = store.commit_transition(
        result,
        policy,
        previous_snapshot=previous,
        command=command,
        input_observation=_observation(),
    )
    feedback = transition(result.snapshot, PresentFeedback())
    store._fault = fail_after_second_commit
    fired = False
    with pytest.raises(RuntimePersistenceError) as caught:
        store.commit_transition(
            feedback,
            policy,
            previous_snapshot=result.snapshot,
            command=PresentFeedback(),
        )
    assert caught.value.code is PersistenceErrorCode.COMMIT_UNKNOWN
    exact = store.commit_transition(
        feedback,
        policy,
        previous_snapshot=result.snapshot,
        command=PresentFeedback(),
    )
    assert replay.replayed is True
    assert exact.replayed is True


def test_review_outbox_contract_is_bound_to_runtime_authority(tmp_path: Path) -> None:
    store = _store(tmp_path / "runtime.sqlite3")
    bad = replace(_intent(), block_index=99)
    with pytest.raises(RuntimePersistenceError) as caught:
        _commit_ready(store, outbox=(bad,))
    assert caught.value.code is PersistenceErrorCode.INVALID_PAYLOAD


def test_review_ack_ledger_and_outbox_existence_are_recovery_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = _store(path)
    _commit_ready(store, outbox=(_intent(),))
    pending = store.pending_outbox(SESSION_ID)[0]
    store.ack_outbox(
        pending.event_id,
        pending.event_type,
        pending.content_sha256,
        acked_at=NOW,
    )
    with sqlite3.connect(path) as connection:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='runtime_outbox_acks_no_delete'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM runtime_outbox_acks")
        connection.execute("DROP TRIGGER runtime_outbox_acks_no_delete")
        connection.execute("DELETE FROM runtime_outbox_acks")
        connection.execute(trigger_sql)
    recovered = _store(path).recover_session(SESSION_ID)
    assert recovered.status is RecoveryStatus.DEGRADED
    assert recovered.error is PersistenceErrorCode.INTEGRITY_MISMATCH


def test_review_deletion_tombstone_cannot_be_overwritten_or_resurrected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = _store(path)
    _commit_ready(store)
    store.erase_session(SESSION_ID, erased_at=NOW)
    with pytest.raises(RuntimePersistenceError) as caught:
        _commit_ready(store)
    assert caught.value.code is PersistenceErrorCode.QUARANTINED
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE deletion_receipts SET reason_code='FORGED'")


def test_review_crypto_erasure_checkpoints_wal_and_scrubs_all_database_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = _store(path)
    _commit_ready(store, outbox=(_intent(),))
    store.erase_session(SESSION_ID, erased_at=NOW)
    assert store.catalog_counts()["sessions"] == 0
    store.close()
    forbidden = (str(SESSION_ID).encode(), str(_intent().plan_id).encode())
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            raw = candidate.read_bytes()
            assert all(item not in raw for item in forbidden)


def test_review_completed_empty_outbox_expires_from_completed_at(tmp_path: Path) -> None:
    store = _store(tmp_path / "runtime.sqlite3")
    completed, policy = _completed_result()
    previous = replace(
        completed.snapshot,
        state=completed.transition.from_state,
        transition_count=completed.snapshot.transition_count - 1,
        completion_finished_receipt=None,
    )
    store.commit_transition(
        completed,
        policy,
        previous_snapshot=previous,
        command=FinishCompletion(),
        committed_at=NOW,
    )
    report = store.run_retention(now=NOW + timedelta(hours=25))
    assert report.deleted_session_ids == (SESSION_ID,)


def test_review_last_ack_is_maximum_and_quarantine_clears_when_caught_up(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = _store(path)
    _commit_ready(store, outbox=(_intent(),))
    pending = store.pending_outbox(SESSION_ID)[0]
    store.run_retention(now=NOW + timedelta(days=8))
    assert store.accepting_new_sessions is False
    store.ack_outbox(
        pending.event_id,
        pending.event_type,
        pending.content_sha256,
        acked_at=NOW + timedelta(days=9),
    )
    store.ack_outbox(
        pending.event_id,
        pending.event_type,
        pending.content_sha256,
        acked_at=NOW + timedelta(days=8),
    )
    store.run_retention(now=NOW + timedelta(days=9))
    with sqlite3.connect(path) as connection:
        last_ack, quarantined = connection.execute(
            "SELECT last_ack_at, quarantined_at FROM runtime_sessions"
        ).fetchone()
    assert last_ack == (NOW + timedelta(days=9)).isoformat()
    assert quarantined is None
    assert store.accepting_new_sessions is True


def test_review_schema_manifest_rejects_missing_trigger_index_or_tombstone(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.sqlite3"
    _store(path).close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER runtime_events_no_update")
    with pytest.raises(RuntimePersistenceError) as caught:
        _store(path)
    assert caught.value.code is PersistenceErrorCode.CORRUPT_DATABASE


@pytest.mark.parametrize("operation", ("init", "ack", "erase"))
def test_review_init_ack_and_erase_lock_are_typed_retryable(tmp_path: Path, operation: str) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = _store(path, busy_timeout_ms=1)
    _commit_ready(store, outbox=(_intent(),))
    pending = store.pending_outbox(SESSION_ID)[0]
    lock = sqlite3.connect(path, isolation_level=None)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(RuntimePersistenceError) as caught:
            if operation == "init":
                _store(path, busy_timeout_ms=1)
            elif operation == "ack":
                store.ack_outbox(
                    pending.event_id,
                    pending.event_type,
                    pending.content_sha256,
                    acked_at=NOW,
                )
            else:
                store.erase_session(SESSION_ID, erased_at=NOW)
        assert caught.value.code is PersistenceErrorCode.DB_LOCKED
        assert caught.value.retryable is True
    finally:
        lock.rollback()
        lock.close()


def test_review_failure_recovery_is_not_a_fabricated_runtime_snapshot(tmp_path: Path) -> None:
    recovered = _store(tmp_path / "runtime.sqlite3").recover_session(SESSION_ID)
    assert recovered.status is RecoveryStatus.PAUSED
    assert recovered.error is PersistenceErrorCode.KEY_UNAVAILABLE
    assert not hasattr(recovered, "snapshot")
    assert recovered.executable is False
