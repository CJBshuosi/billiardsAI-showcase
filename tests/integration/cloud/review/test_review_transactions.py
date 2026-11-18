from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError

from billiards_ai.cloud.review import apply_shot_reviews, reject_review_session, submit_review
from billiards_ai.cloud.review.persistence import (
    IdempotencyConflictError,
    IdempotencyIdentity,
    PostgresOutboxRepository,
    PostgresQualityGateRepository,
    QualityGateUnitOfWork,
    ReviewSessionOwnership,
)
from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    ReportDocument,
    ReviewerIdentity,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    SessionRejectionReason,
    Shot,
    ShotReviewDecision,
    ShotReviewOutcome,
)
from billiards_ai.shared.models.events import DomainEvent

REPORT_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
NOW = datetime(2026, 8, 10, 13, tzinfo=UTC)
AUTHORITY_TABLES = (
    "review_tasks",
    "review_rejections",
    "quality_gate_outbox",
    "quality_gate_idempotency_receipts",
)


async def authority_snapshot(session_factory) -> dict[str, tuple[str, ...]]:
    async with session_factory() as session:
        return {
            table: tuple(
                (
                    await session.execute(
                        text(
                            f"SELECT row_to_json(authority)::text FROM {table} authority "  # noqa: S608
                            "ORDER BY 1"
                        )
                    )
                ).scalars()
            )
            for table in AUTHORITY_TABLES
        }


def draft_facts():
    report = ReportDocument.model_validate_json(REPORT_FIXTURE.read_text(encoding="utf-8"))
    task = ReviewTask(
        id=uuid4(),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        status=ReviewTaskStatus.PENDING_REVIEW,
    )
    shots = tuple(
        Shot(
            id=uuid4(),
            session_id=report.session_id,
            sequence=sequence,
            t_start_ms=(sequence - 1) * 1000,
            t_end_ms=sequence * 1000,
            ball_positions_before=None,
            ball_positions_after=None,
            cue_ball_end=None,
            struck_ball=sequence,
            potted_balls=(),
            cushion_count=0,
            travel_distance_mm=300,
            is_foul=False,
            confidence=Confidence.HIGH,
            review_status=ReviewStatus.PENDING,
            raw_observation={},
            created_at=NOW,
        )
        for sequence in range(1, report.session_statistics.total_shot_count + 1)
    )
    return report, task, shots


async def rejected_facts(repository):
    report, task, shots = draft_facts()
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=NOW,
    )
    claimed = await repository.claim_task(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-9",
        credential_id="credential-9",
        claimed_at=NOW,
    )
    reviewer = ReviewerIdentity(
        principal_id="reviewer-9",
        credential_id="credential-9",
        roles=(AccessActorRole.REVIEWER,),
    )
    result = reject_review_session(
        report=report,
        review_task=claimed.task,
        reviewer=reviewer,
        reason=SessionRejectionReason.OCCLUDED,
        rejection_id=uuid4(),
        notification_event_id=uuid4(),
        rejected_at=NOW,
    )
    event = DomainEvent(
        event_id=result.notification.event_id,
        event_type="review.session.rejected.v1",
        aggregate_type="review_session",
        aggregate_id=report.session_id,
        aggregate_version=report.revision,
        occurred_at=NOW,
        correlation_id=task.id,
        idempotency_key=f"review-rejected:{task.id}:{report.revision}",
        payload={
            "session_id": str(report.session_id),
            "reason": result.rejection.reason_code.value,
            "message": result.notification.message.value,
            "created_at": NOW.isoformat(),
        },
    )
    return task, result, event


async def test_rejection_commits_task_audit_and_notification_but_no_approved_evidence(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    task, result, event = await rejected_facts(repository)

    identity = IdempotencyIdentity(
        actor_principal_id=result.rejection.reviewer.principal_id,
        operation="reject_session",
        task_id=task.id,
        shot_id=None,
        idempotency_key="reject-session-1",
    )
    stored = await repository.commit_rejection(
        result=result,
        notification_event=event,
        expected_lock_version=1,
        identity=identity,
        request_digest="1" * 64,
    )
    replay = await repository.commit_rejection(
        result=result,
        notification_event=event,
        expected_lock_version=1,
        identity=identity,
        request_digest="1" * 64,
    )

    assert stored.task.status is ReviewTaskStatus.REJECTED
    assert replay == stored
    assert stored.lock_version == 2
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "review_rejections",
                "quality_gate_outbox",
                "approved_evidence_batches",
                "quality_gate_idempotency_receipts",
            )
        }
        outbox_payload = (
            await session.execute(
                text("SELECT payload FROM quality_gate_outbox WHERE event_id = :event_id"),
                {"event_id": event.event_id},
            )
        ).scalar_one()
    assert counts == {
        "review_rejections": 1,
        "quality_gate_outbox": 1,
        "approved_evidence_batches": 0,
        "quality_gate_idempotency_receipts": 1,
    }
    assert "reviewer" not in str(outbox_payload).lower()
    assert (await repository.load_task(task.id)).task.status is ReviewTaskStatus.REJECTED


async def test_claim_receipt_replays_immutable_claim_snapshot_after_terminal_rejection(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    report, task, shots = draft_facts()
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=NOW,
    )
    ownership = ReviewSessionOwnership(
        task_id=task.id,
        user_id=uuid4(),
        session_id=task.session_id,
        bound_at=NOW,
    )
    claim_identity = IdempotencyIdentity(
        actor_principal_id="reviewer-9",
        operation="claim_review_v1",
        task_id=task.id,
        shot_id=None,
        idempotency_key="claim-review-terminal-replay-74f154d1",
    )
    claim_digest = "d" * 64
    claimed = await repository.claim_task_with_ownership_and_receipt(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-9",
        credential_id="credential-9",
        claimed_at=NOW,
        ownership=ownership,
        identity=claim_identity,
        request_digest=claim_digest,
    )
    reviewer = ReviewerIdentity(
        principal_id="reviewer-9",
        credential_id="credential-9",
        roles=(AccessActorRole.REVIEWER,),
    )
    rejection = reject_review_session(
        report=report,
        review_task=claimed.task,
        reviewer=reviewer,
        reason=SessionRejectionReason.OCCLUDED,
        rejection_id=uuid4(),
        notification_event_id=uuid4(),
        rejected_at=NOW,
    )
    notification = DomainEvent(
        event_id=rejection.notification.event_id,
        event_type="review.session.rejected.v1",
        aggregate_type="review_session",
        aggregate_id=report.session_id,
        aggregate_version=report.revision,
        occurred_at=NOW,
        correlation_id=task.id,
        idempotency_key=f"review-rejected:{task.id}:{report.revision}",
        payload={
            "session_id": str(report.session_id),
            "reason": rejection.rejection.reason_code.value,
            "message": rejection.notification.message.value,
            "created_at": NOW.isoformat(),
        },
    )
    await repository.commit_rejection(
        result=rejection,
        notification_event=notification,
        expected_lock_version=claimed.lock_version,
        identity=IdempotencyIdentity(
            actor_principal_id="reviewer-9",
            operation="reject_session",
            task_id=task.id,
            shot_id=None,
            idempotency_key="reject-after-claim-receipt",
        ),
        request_digest="f" * 64,
    )

    replay = await repository.probe_claim_receipt(
        identity=claim_identity,
        request_digest=claim_digest,
    )

    assert replay == claimed
    assert replay.task.status is ReviewTaskStatus.IN_REVIEW
    assert replay.lock_version == 1
    assert replay.claimed_principal_id == "reviewer-9"
    assert replay.claimed_credential_id == "credential-9"
    assert await repository.load_session_ownership(task.id) == ownership
    with pytest.raises(IdempotencyConflictError):
        await repository.probe_claim_receipt(
            identity=claim_identity,
            request_digest="e" * 64,
        )


async def test_rejection_intent_probe_strictly_replays_original_result_without_factories(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    task, result, notification = await rejected_facts(repository)
    identity = IdempotencyIdentity(
        actor_principal_id=result.rejection.reviewer.principal_id,
        operation="reject_review_intent_v1",
        task_id=task.id,
        shot_id=None,
        idempotency_key="reject-intent-digest-only-ea36e05aa89a",
    )
    request_digest = "a" * 64

    assert (
        await repository.probe_rejection_receipt(
            identity=identity,
            request_digest=request_digest,
        )
        is None
    )
    await repository.commit_rejection(
        result=result,
        notification_event=notification,
        expected_lock_version=1,
        identity=identity,
        request_digest=request_digest,
    )
    loaded = await repository.load_rejected_result(task_id=task.id)
    probe = await repository.probe_rejection_receipt(
        identity=identity,
        request_digest=request_digest,
    )

    assert loaded.result == result
    assert loaded.notification_event == notification
    assert not loaded.replayed
    assert probe is not None and probe.replayed
    assert probe.result == result
    with pytest.raises(IdempotencyConflictError):
        await repository.probe_rejection_receipt(
            identity=identity,
            request_digest="b" * 64,
        )


async def test_locked_rejection_recheck_ignores_losing_candidate_ids(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    task, result, notification = await rejected_facts(repository)
    identity = IdempotencyIdentity(
        actor_principal_id=result.rejection.reviewer.principal_id,
        operation="reject_review_intent_v1",
        task_id=task.id,
        shot_id=None,
        idempotency_key="reject-intent-concurrent-recheck-71a662",
    )
    request_digest = "c" * 64
    await repository.commit_rejection(
        result=result,
        notification_event=notification,
        expected_lock_version=1,
        identity=identity,
        request_digest=request_digest,
    )
    losing_result = result.model_copy(
        update={
            "rejection": result.rejection.model_copy(update={"rejection_id": uuid4()}),
            "notification": result.notification.model_copy(update={"event_id": uuid4()}),
        }
    )
    losing_notification = notification.model_copy(
        update={"event_id": losing_result.notification.event_id}
    )

    replayed_task = await repository.commit_rejection(
        result=losing_result,
        notification_event=losing_notification,
        expected_lock_version=1,
        identity=identity,
        request_digest=request_digest,
    )

    assert replayed_task.task.status is ReviewTaskStatus.REJECTED
    stored = await repository.load_rejected_result(task_id=task.id)
    assert stored.result == result
    async with database_session_factory() as session:
        assert (
            await session.execute(text("SELECT count(*) FROM review_rejections"))
        ).scalar_one() == 1


async def test_two_connections_missing_the_same_reject_receipt_replay_one_winner(
    database_session_factory,
):
    both_missed = asyncio.Event()
    miss_count = 0

    class ReceiptMissBarrierRepository(PostgresQualityGateRepository):
        @staticmethod
        async def _receipt_row(session, identity, *, for_update):
            nonlocal miss_count
            row = await PostgresQualityGateRepository._receipt_row(
                session,
                identity,
                for_update=for_update,
            )
            if row is None and identity.operation == "reject_review_intent_v1":
                miss_count += 1
                if miss_count == 2:
                    both_missed.set()
                await both_missed.wait()
            return row

    repository_a = ReceiptMissBarrierRepository(database_session_factory)
    repository_b = ReceiptMissBarrierRepository(database_session_factory)
    task, winner_candidate, winner_notification = await rejected_facts(repository_a)
    losing_candidate = winner_candidate.model_copy(
        update={
            "rejection": winner_candidate.rejection.model_copy(update={"rejection_id": uuid4()}),
            "notification": winner_candidate.notification.model_copy(update={"event_id": uuid4()}),
        }
    )
    losing_notification = winner_notification.model_copy(
        update={"event_id": losing_candidate.notification.event_id}
    )
    identity = IdempotencyIdentity(
        actor_principal_id=winner_candidate.rejection.reviewer.principal_id,
        operation="reject_review_intent_v1",
        task_id=task.id,
        shot_id=None,
        idempotency_key="reject-intent-real-concurrency-ea36e05a",
    )

    first, second = await asyncio.gather(
        repository_a.commit_rejection(
            result=winner_candidate,
            notification_event=winner_notification,
            expected_lock_version=1,
            identity=identity,
            request_digest="9" * 64,
        ),
        repository_b.commit_rejection(
            result=losing_candidate,
            notification_event=losing_notification,
            expected_lock_version=1,
            identity=identity,
            request_digest="9" * 64,
        ),
    )

    assert first == second
    assert first.task.status is ReviewTaskStatus.REJECTED
    stored = await repository_a.load_rejected_result(task_id=task.id)
    assert stored.result in (winner_candidate, losing_candidate)
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "review_rejections",
                "quality_gate_outbox",
                "quality_gate_idempotency_receipts",
            )
        }
    assert counts == {
        "review_rejections": 1,
        "quality_gate_outbox": 1,
        "quality_gate_idempotency_receipts": 1,
    }


async def test_outbox_conflict_rolls_back_rejection_and_terminal_task(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    task, result, event = await rejected_facts(repository)
    conflicting = event.model_copy(update={"event_id": uuid4()})
    await PostgresOutboxRepository(database_session_factory).enqueue(conflicting, available_at=NOW)

    with pytest.raises(IntegrityError):
        await repository.commit_rejection(
            result=result,
            notification_event=event,
            expected_lock_version=1,
            identity=IdempotencyIdentity(
                actor_principal_id=result.rejection.reviewer.principal_id,
                operation="reject_session",
                task_id=task.id,
                shot_id=None,
                idempotency_key="reject-session-conflict",
            ),
            request_digest="2" * 64,
        )

    current = await repository.load_task(task.id)
    assert current.task.status is ReviewTaskStatus.IN_REVIEW
    assert current.lock_version == 1
    async with database_session_factory() as session:
        assert (
            await session.execute(text("SELECT count(*) FROM review_rejections"))
        ).scalar_one() == 0


@pytest.mark.parametrize(
    "statement_marker",
    (
        "UPDATE review_tasks SET status = 'rejected'",
        "INSERT INTO review_rejections",
        "INSERT INTO quality_gate_outbox",
        "INSERT INTO quality_gate_idempotency_receipts",
    ),
)
async def test_rejection_failure_points_restore_the_canonical_authority_snapshot(
    database_session_factory,
    statement_marker,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    task, result, notification = await rejected_facts(repository)
    before = await authority_snapshot(database_session_factory)
    engine = database_session_factory.kw["bind"]

    class InjectedFailure(RuntimeError):
        pass

    def fail_at_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if statement_marker in statement:
            raise InjectedFailure(statement_marker)

    event.listen(engine.sync_engine, "before_cursor_execute", fail_at_statement)
    try:
        with pytest.raises(InjectedFailure):
            await repository.commit_rejection(
                result=result,
                notification_event=notification,
                expected_lock_version=1,
                identity=IdempotencyIdentity(
                    actor_principal_id=result.rejection.reviewer.principal_id,
                    operation="reject_session",
                    task_id=task.id,
                    shot_id=None,
                    idempotency_key=f"failure-{statement_marker}",
                ),
                request_digest="f" * 64,
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail_at_statement)

    assert await authority_snapshot(database_session_factory) == before


async def test_private_submission_seam_stages_strict_results_without_exposing_terminal_state(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    report, task, shots = draft_facts()
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=NOW,
    )
    claimed = await repository.claim_task(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-9",
        credential_id="credential-9",
        claimed_at=NOW,
    )
    reviewer = ReviewerIdentity(
        principal_id="reviewer-9",
        credential_id="credential-9",
        roles=(AccessActorRole.REVIEWER,),
    )
    decisions = tuple(
        ShotReviewDecision(
            shot_id=shot.id,
            outcome=ShotReviewOutcome.APPROVED,
            reviewer=reviewer,
            reviewed_at=NOW,
        )
        for shot in shots
    )
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=decisions,
    )
    submission = submit_review(
        report=report,
        review_task=claimed.task,
        application=application,
        completed_at=NOW,
    )

    class RollbackSentinel(Exception):
        pass

    with pytest.raises(RollbackSentinel):
        async with QualityGateUnitOfWork(database_session_factory).transaction() as session:
            await repository._stage_review_submission(  # noqa: SLF001
                session,
                source_task=claimed,
                submission=submission,
            )
            assert (
                await session.execute(text("SELECT count(*) FROM review_shot_decisions"))
            ).scalar_one() == len(shots)
            raise RollbackSentinel
    current = await repository.load_task(task.id)
    assert current.task.status is ReviewTaskStatus.IN_REVIEW
    assert current.lock_version == 1
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "review_shot_decisions",
                "review_shot_results",
                "review_corrections",
                "approved_evidence_batches",
                "quality_gate_outbox",
            )
        }
    assert counts == {
        "review_shot_decisions": 0,
        "review_shot_results": 0,
        "review_corrections": 0,
        "approved_evidence_batches": 0,
        "quality_gate_outbox": 0,
    }
