from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from sqlalchemy import text

from billiards_ai.cloud.review.application import ReviewApplicationService
from billiards_ai.cloud.review.application_models import (
    ClaimReviewCommand,
    RejectReviewCommand,
    TaskRef,
)
from billiards_ai.cloud.review.persistence import (
    PostgresApprovedEvidencePublisher,
    PostgresQualityGateRepository,
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
    UserNotificationStatus,
)

REPORT_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
NOW = datetime(2026, 8, 10, 20, tzinfo=UTC)
REJECTED_AT = datetime(2026, 8, 10, 20, 1, tzinfo=UTC)


class _IdentityProvider:
    def __init__(self, identity: ReviewerIdentity) -> None:
        self._identity = identity

    def current_identity(self) -> ReviewerIdentity:
        return self._identity


class _SubjectProvider:
    def __init__(self, user_id: UUID) -> None:
        self._user_id = user_id

    def resolve_subject(self, *, session_id: UUID) -> object:
        return SimpleNamespace(session_id=session_id, user_id=self._user_id)


class _UnusedMetricAssembler:
    def assemble(self, **_kwargs):
        raise AssertionError("reject application flow must not assemble metric authority")


class _Factories:
    def __init__(self) -> None:
        self.clock_calls = 0
        self.uuid_calls = 0
        self._times = iter((NOW, REJECTED_AT))
        self._ids = iter(
            (
                UUID("00000000-0000-4000-8000-000000000901"),
                UUID("00000000-0000-4000-8000-000000000902"),
            )
        )

    def now(self) -> datetime:
        self.clock_calls += 1
        return next(self._times)

    def uuid4(self) -> UUID:
        self.uuid_calls += 1
        return next(self._ids)


async def _stage_review(repository: PostgresQualityGateRepository):
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
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=NOW,
    )
    return task


async def test_application_claim_reject_result_and_replay_are_one_postgres_graph(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task = await _stage_review(repository)
    reviewer = ReviewerIdentity(
        principal_id="reviewer-application-pg",
        credential_id="credential-application-pg",
        roles=(AccessActorRole.REVIEWER,),
    )
    subject_id = UUID("00000000-0000-4000-8000-000000000900")
    factories = _Factories()
    service = ReviewApplicationService(
        repository=repository,
        publisher=publisher,
        identity_provider=_IdentityProvider(reviewer),
        subject_provider=_SubjectProvider(subject_id),
        metric_assembler=_UnusedMetricAssembler(),
        now=factories.now,
        uuid4_factory=factories.uuid4,
    )
    claim = ClaimReviewCommand(
        task=TaskRef(
            task_id=task.id,
            expected_lock_version=0,
            source_report_id=task.report_id,
            source_report_revision=task.report_revision,
        ),
        idempotency_key="claim-application-pg-000000000001",
    )

    claimed = await service.claim(claim)
    reject = RejectReviewCommand(
        task=TaskRef(
            task_id=task.id,
            expected_lock_version=claimed.task.lock_version,
            source_report_id=task.report_id,
            source_report_revision=task.report_revision,
        ),
        reason=SessionRejectionReason.OCCLUDED,
        idempotency_key="reject-application-pg-000000000001",
    )
    rejected = await service.reject(reject)
    result = await service.result(reject.task)
    calls_before_replay = (factories.clock_calls, factories.uuid_calls)
    replay = await service.reject(reject)

    assert claimed.task.status is ReviewTaskStatus.IN_REVIEW
    assert rejected == result == replay
    assert rejected.status is ReviewTaskStatus.REJECTED
    assert rejected.notification_event_id == UUID("00000000-0000-4000-8000-000000000902")
    assert rejected.notification_status is UserNotificationStatus.PENDING
    assert calls_before_replay == (2, 2)
    assert (factories.clock_calls, factories.uuid_calls) == calls_before_replay
    assert (await repository.load_session_ownership(task.id)).user_id == subject_id

    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "review_session_ownership",
                "review_rejections",
                "quality_gate_outbox",
                "approved_evidence_batches",
                "quality_gate_idempotency_receipts",
            )
        }
        receipt_operations = tuple(
            (
                await session.execute(
                    text(
                        "SELECT operation FROM quality_gate_idempotency_receipts "
                        "WHERE task_id=:task_id ORDER BY operation"
                    ),
                    {"task_id": task.id},
                )
            ).scalars()
        )
        notification = (
            (
                await session.execute(
                    text(
                        "SELECT event_id,event_type,correlation_id FROM quality_gate_outbox "
                        "WHERE correlation_id=:task_id"
                    ),
                    {"task_id": task.id},
                )
            )
            .mappings()
            .one()
        )

    assert counts == {
        "review_session_ownership": 1,
        "review_rejections": 1,
        "quality_gate_outbox": 1,
        "approved_evidence_batches": 0,
        "quality_gate_idempotency_receipts": 2,
    }
    assert receipt_operations == ("claim_review_v1", "reject_review_intent_v1")
    assert notification == {
        "event_id": rejected.notification_event_id,
        "event_type": "review.session.rejected.v1",
        "correlation_id": task.id,
    }
