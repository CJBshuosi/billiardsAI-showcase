from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import event, text

from billiards_ai.cloud.review.application import ReviewApplicationService
from billiards_ai.cloud.review.application_models import ClaimReviewCommand, TaskRef
from billiards_ai.cloud.review.persistence import (
    IdempotencyConflictError,
    IdempotencyIdentity,
    PersistenceIntegrityError,
    PostgresApprovedEvidencePublisher,
    PostgresQualityGateRepository,
    ReviewSessionOwnership,
    RevisionConflictError,
)
from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    ReportDocument,
    ReviewerIdentity,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    Shot,
)
from billiards_ai.shared.models.events import canonical_sha256

REPORT_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
NOW = datetime(2026, 8, 10, 9, tzinfo=UTC)


def make_report() -> ReportDocument:
    return ReportDocument.model_validate_json(REPORT_FIXTURE.read_text(encoding="utf-8"))


def make_task(report: ReportDocument) -> ReviewTask:
    return ReviewTask(
        id=uuid4(),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        status=ReviewTaskStatus.PENDING_REVIEW,
    )


def make_shots(report: ReportDocument) -> tuple[Shot, ...]:
    return tuple(
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
            travel_distance_mm=400,
            is_foul=False,
            deviation_mm=None,
            confidence=Confidence.HIGH,
            review_status=ReviewStatus.PENDING,
            trajectory=None,
            raw_observation={"media_url": "must-not-persist", "provider_prompt": "secret"},
            created_at=NOW,
        )
        for sequence in range(1, report.session_statistics.total_shot_count + 1)
    )


async def stage(repository: PostgresQualityGateRepository):
    report = make_report()
    task = make_task(report)
    shots = make_shots(report)
    envelopes = {shot.id: uuid4() for shot in shots}
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids=envelopes,
        created_at=NOW,
    )
    return report, task, shots, envelopes


async def stage_distinct(repository: PostgresQualityGateRepository, *, created_at: datetime):
    payload = make_report().model_dump(mode="python")
    payload.update({"report_id": uuid4(), "session_id": uuid4()})
    report = ReportDocument.model_validate(payload, strict=True)
    task = make_task(report)
    shots = tuple(
        shot.model_copy(
            update={"id": uuid4(), "session_id": report.session_id, "created_at": created_at}
        )
        for shot in make_shots(report)
    )
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=created_at,
    )
    return task


async def test_draft_round_trips_strict_models_without_raw_or_media(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    report, task, shots, envelopes = await stage(repository)

    stored_report = await repository.load_report(report.report_id)
    stored_task = await repository.load_task(task.id)
    stored_shots = await repository.load_source_shots(task.id)

    assert stored_report == report
    assert stored_task.task == task
    assert stored_task.lock_version == 0
    assert tuple(item.shot for item in stored_shots) == tuple(
        shot.model_copy(update={"raw_observation": {}}) for shot in shots
    )
    assert {item.shot.id: item.source_envelope_id for item in stored_shots} == envelopes

    serialized = json.dumps(
        [
            stored_report.model_dump(mode="json"),
            *[item.shot.model_dump(mode="json") for item in stored_shots],
        ]
    ).lower()
    assert "must-not-persist" not in serialized
    assert "media_url" not in serialized
    assert "provider_prompt" not in serialized


async def test_legal_json_tamper_and_column_drift_fail_closed(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    report, task, _, _ = await stage(repository)

    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE review_reports SET payload = jsonb_set(payload, '{revision}', '2') "
                "WHERE report_id = :report_id"
            ),
            {"report_id": report.report_id},
        )
    with pytest.raises(PersistenceIntegrityError, match="digest"):
        await repository.load_report(report.report_id)

    async with database_session_factory() as session, session.begin():
        await session.execute(
            text("UPDATE review_reports SET session_id = :session_id WHERE report_id = :report_id"),
            {"report_id": report.report_id, "session_id": uuid4()},
        )
    with pytest.raises(PersistenceIntegrityError):
        await repository.load_report(report.report_id)


async def test_two_independent_connections_claim_exactly_once(database_session_factory):
    repository_a = PostgresQualityGateRepository(database_session_factory)
    repository_b = PostgresQualityGateRepository(database_session_factory)
    _, task, _, _ = await stage(repository_a)
    barrier = asyncio.Event()

    async def claim(repository, actor):
        await barrier.wait()
        try:
            return await repository.claim_task(
                task_id=task.id,
                expected_lock_version=0,
                actor_principal_id=actor,
                credential_id=f"credential-{actor}",
                claimed_at=NOW,
            )
        except RevisionConflictError:
            return None

    attempts = [
        asyncio.create_task(claim(repository_a, "reviewer-a")),
        asyncio.create_task(claim(repository_b, "reviewer-b")),
    ]
    barrier.set()
    results = await asyncio.gather(*attempts)

    assert sum(result is not None for result in results) == 1
    winner = next(result for result in results if result is not None)
    assert winner.task.status is ReviewTaskStatus.IN_REVIEW
    assert winner.lock_version == 1
    assert winner.claimed_principal_id in {"reviewer-a", "reviewer-b"}


async def test_stale_claim_cannot_overwrite_current_owner(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    _, task, _, _ = await stage(repository)
    await repository.claim_task(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        claimed_at=NOW,
    )

    with pytest.raises(RevisionConflictError):
        await repository.claim_task(
            task_id=task.id,
            expected_lock_version=0,
            actor_principal_id="reviewer-b",
            credential_id="credential-b",
            claimed_at=NOW,
        )

    current = await repository.load_task(task.id)
    assert current.claimed_principal_id == "reviewer-a"
    assert current.lock_version == 1


async def test_visible_task_pages_include_pending_and_only_the_callers_active_claims(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    staged = []
    for index in range(4):
        staged.append(await stage_distinct(repository, created_at=NOW + timedelta(seconds=index)))
    tasks = tuple(staged)
    await repository.claim_task(
        task_id=tasks[1].id,
        expected_lock_version=0,
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        claimed_at=NOW + timedelta(minutes=1),
    )
    await repository.claim_task(
        task_id=tasks[2].id,
        expected_lock_version=0,
        actor_principal_id="reviewer-b",
        credential_id="credential-b",
        claimed_at=NOW + timedelta(minutes=1),
    )

    first = await repository.list_visible_tasks(
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        cursor=None,
        limit=2,
    )
    second = await repository.list_visible_tasks(
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        cursor=first.next_cursor,
        limit=2,
    )

    visible_ids = tuple(item.task.id for item in (*first.items, *second.items))
    assert visible_ids == (tasks[0].id, tasks[1].id, tasks[3].id)
    assert tasks[2].id not in visible_ids
    assert first.next_cursor is not None
    assert second.next_cursor is None
    with pytest.raises(ValueError, match="cursor"):
        await repository.list_visible_tasks(
            actor_principal_id="reviewer-a",
            credential_id="credential-a",
            cursor="not-a-cursor",
            limit=2,
        )


async def test_visible_task_page_reauthorizes_rows_after_a_concurrent_claim(
    database_session_factory,
):
    selected = asyncio.Event()
    claimed = asyncio.Event()

    class PausingLoadRepository(PostgresQualityGateRepository):
        async def load_task(self, task_id):
            if not selected.is_set():
                selected.set()
                await claimed.wait()
            return await super().load_task(task_id)

    repository = PausingLoadRepository(database_session_factory)
    claiming_repository = PostgresQualityGateRepository(database_session_factory)
    first = await stage_distinct(repository, created_at=NOW)
    second = await stage_distinct(repository, created_at=NOW + timedelta(seconds=1))

    page_task = asyncio.create_task(
        repository.list_visible_tasks(
            actor_principal_id="reviewer-a",
            credential_id="credential-a",
            cursor=None,
            limit=1,
        )
    )
    await selected.wait()
    await claiming_repository.claim_task(
        task_id=first.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-b",
        credential_id="credential-b",
        claimed_at=NOW + timedelta(minutes=1),
    )
    claimed.set()

    page = await page_task

    assert page.items == ()
    assert page.next_cursor is not None
    next_page = await repository.list_visible_tasks(
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        cursor=page.next_cursor,
        limit=1,
    )
    assert tuple(item.task.id for item in next_page.items) == (second.id,)


async def test_application_claim_binds_subject_and_replays_without_recalling_clock(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    _, task, _, _ = await stage(repository)
    reviewer = ReviewerIdentity(
        principal_id="reviewer-application",
        credential_id="credential-application",
        roles=(AccessActorRole.REVIEWER,),
    )
    subject_id = uuid4()
    clock_calls = 0

    class IdentityProvider:
        def current_identity(self):
            return reviewer

    class SubjectProvider:
        def resolve_subject(self, *, session_id):
            return SimpleNamespace(session_id=session_id, user_id=subject_id)

    class UnusedMetricAssembler:
        def assemble(self, **kwargs):
            raise AssertionError("claim must not build metric authority")

    def now():
        nonlocal clock_calls
        clock_calls += 1
        return NOW

    service = ReviewApplicationService(
        repository=repository,
        publisher=PostgresApprovedEvidencePublisher(database_session_factory),
        identity_provider=IdentityProvider(),
        subject_provider=SubjectProvider(),
        metric_assembler=UnusedMetricAssembler(),
        now=now,
        uuid4_factory=uuid4,
    )
    command = ClaimReviewCommand(
        task=TaskRef(
            task_id=task.id,
            expected_lock_version=0,
            source_report_id=task.report_id,
            source_report_revision=task.report_revision,
        ),
        idempotency_key="claim-application-0123456789abcdef",
    )

    first = await service.claim(command)
    replay = await service.claim(command)

    assert first == replay
    assert first.task.status is ReviewTaskStatus.IN_REVIEW
    assert clock_calls == 1
    ownership = await repository.load_session_ownership(task.id)
    assert ownership.user_id == subject_id


async def test_claim_ownership_and_receipt_commit_atomically_and_replay_by_digest(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    _, task, _, _ = await stage(repository)
    ownership = ReviewSessionOwnership(
        task_id=task.id,
        user_id=uuid4(),
        session_id=task.session_id,
        bound_at=NOW,
    )
    identity = IdempotencyIdentity(
        actor_principal_id="reviewer-a",
        operation="claim_review_v1",
        task_id=task.id,
        shot_id=None,
        idempotency_key="claim-review-74f154d1a68b4b33a3ad",
    )
    request_digest = "a" * 64

    assert (
        await repository.probe_claim_receipt(
            identity=identity,
            request_digest=request_digest,
        )
        is None
    )
    claimed = await repository.claim_task_with_ownership_and_receipt(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        claimed_at=NOW,
        ownership=ownership,
        identity=identity,
        request_digest=request_digest,
    )
    probe = await repository.probe_claim_receipt(
        identity=identity,
        request_digest=request_digest,
    )
    replay = await repository.claim_task_with_ownership_and_receipt(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-a",
        credential_id="credential-a",
        claimed_at=datetime(2026, 8, 10, 10, tzinfo=UTC),
        ownership=ownership.model_copy(update={"bound_at": datetime(2026, 8, 10, 10, tzinfo=UTC)}),
        identity=identity,
        request_digest=request_digest,
    )

    assert claimed.task.status is ReviewTaskStatus.IN_REVIEW
    assert probe == replay == claimed
    assert await repository.load_session_ownership(task.id) == ownership
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "review_session_ownership",
                "quality_gate_idempotency_receipts",
            )
        }
    assert counts == {
        "review_session_ownership": 1,
        "quality_gate_idempotency_receipts": 1,
    }
    with pytest.raises(IdempotencyConflictError):
        await repository.probe_claim_receipt(
            identity=identity,
            request_digest="b" * 64,
        )


async def test_two_reviewers_atomically_compete_for_claim_ownership_and_receipt(
    database_session_factory,
):
    repository_a = PostgresQualityGateRepository(database_session_factory)
    repository_b = PostgresQualityGateRepository(database_session_factory)
    _, task, _, _ = await stage(repository_a)
    barrier = asyncio.Event()
    candidates = {
        "reviewer-a": ("credential-a", uuid4(), "a" * 64),
        "reviewer-b": ("credential-b", uuid4(), "b" * 64),
    }

    async def claim(repository, actor):
        credential, user_id, request_digest = candidates[actor]
        identity = IdempotencyIdentity(
            actor_principal_id=actor,
            operation="claim_review_v1",
            task_id=task.id,
            shot_id=None,
            idempotency_key=f"claim-review-race-{actor}-74f154d1",
        )
        await barrier.wait()
        try:
            stored = await repository.claim_task_with_ownership_and_receipt(
                task_id=task.id,
                expected_lock_version=0,
                actor_principal_id=actor,
                credential_id=credential,
                claimed_at=NOW,
                ownership=ReviewSessionOwnership(
                    task_id=task.id,
                    user_id=user_id,
                    session_id=task.session_id,
                    bound_at=NOW,
                ),
                identity=identity,
                request_digest=request_digest,
            )
            return actor, stored
        except RevisionConflictError:
            return actor, None

    attempts = [
        asyncio.create_task(claim(repository_a, "reviewer-a")),
        asyncio.create_task(claim(repository_b, "reviewer-b")),
    ]
    barrier.set()
    results = await asyncio.gather(*attempts)

    winners = tuple(item for item in results if item[1] is not None)
    assert len(winners) == 1
    winner_actor, winner = winners[0]
    assert winner is not None
    winner_credential, winner_user_id, _digest = candidates[winner_actor]
    assert winner.task.status is ReviewTaskStatus.IN_REVIEW
    assert winner.claimed_principal_id == winner_actor
    assert winner.claimed_credential_id == winner_credential
    ownership = await repository_a.load_session_ownership(task.id)
    assert ownership.user_id == winner_user_id
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "review_session_ownership",
                "quality_gate_idempotency_receipts",
            )
        }
        receipt_actor = (
            await session.execute(
                text(
                    "SELECT actor_principal_id FROM quality_gate_idempotency_receipts "
                    "WHERE operation='claim_review_v1' AND task_id=:task_id"
                ),
                {"task_id": task.id},
            )
        ).scalar_one()
    assert counts == {
        "review_session_ownership": 1,
        "quality_gate_idempotency_receipts": 1,
    }
    assert receipt_actor == winner_actor


async def test_claim_receipt_failure_rolls_back_claim_and_ownership(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    _, task, _, _ = await stage(repository)
    ownership = ReviewSessionOwnership(
        task_id=task.id,
        user_id=uuid4(),
        session_id=task.session_id,
        bound_at=NOW,
    )
    identity = IdempotencyIdentity(
        actor_principal_id="reviewer-a",
        operation="claim_review_v1",
        task_id=task.id,
        shot_id=None,
        idempotency_key="claim-review-rollback-56c113fac57e",
    )
    engine = database_session_factory.kw["bind"]

    class InjectedFailure(RuntimeError):
        pass

    def fail_receipt(_connection, _cursor, statement, _parameters, _context, _many):
        if "INSERT INTO quality_gate_idempotency_receipts" in statement:
            raise InjectedFailure("receipt")

    event.listen(engine.sync_engine, "before_cursor_execute", fail_receipt)
    try:
        with pytest.raises(InjectedFailure):
            await repository.claim_task_with_ownership_and_receipt(
                task_id=task.id,
                expected_lock_version=0,
                actor_principal_id="reviewer-a",
                credential_id="credential-a",
                claimed_at=NOW,
                ownership=ownership,
                identity=identity,
                request_digest="c" * 64,
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail_receipt)

    current = await repository.load_task(task.id)
    assert current.task.status is ReviewTaskStatus.PENDING_REVIEW
    assert current.lock_version == 0
    async with database_session_factory() as session:
        assert (
            await session.execute(text("SELECT count(*) FROM review_session_ownership"))
        ).scalar_one() == 0
        assert (
            await session.execute(text("SELECT count(*) FROM quality_gate_idempotency_receipts"))
        ).scalar_one() == 0


async def test_idempotency_receipt_replays_same_result_and_rejects_changed_request(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    _, task, shots, _ = await stage(repository)
    identity = IdempotencyIdentity(
        actor_principal_id="reviewer-a",
        operation="show_task",
        task_id=task.id,
        shot_id=None,
        idempotency_key="idem-7cbf4a5f7f5e4d23a4e9",
    )
    result_digest = canonical_sha256(
        {"task_id": str(task.id), "status": "pending_review", "lock_version": 0}
    )

    first = await repository.record_idempotency_receipt(
        identity=identity,
        request_digest="1" * 64,
        result_kind="review_task",
        result_identity=str(task.id),
        result_digest=result_digest,
        created_at=NOW,
    )
    replay = await repository.record_idempotency_receipt(
        identity=identity,
        request_digest="1" * 64,
        result_kind="review_task",
        result_identity=str(task.id),
        result_digest=result_digest,
        created_at=datetime(2026, 8, 10, 12, tzinfo=UTC),
    )

    assert replay == first
    assert replay.created_at == NOW
    with pytest.raises(IdempotencyConflictError):
        await repository.record_idempotency_receipt(
            identity=identity,
            request_digest="3" * 64,
            result_kind="review_task",
            result_identity=str(task.id),
            result_digest=result_digest,
            created_at=NOW,
        )
