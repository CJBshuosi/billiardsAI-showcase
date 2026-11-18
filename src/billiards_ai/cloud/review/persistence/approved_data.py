"""PostgreSQL adapter for the existing approved-data repository boundary."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.cloud.review.repository import InMemoryApprovedDataRepository
from billiards_ai.shared.models.report import ReportDocument
from billiards_ai.shared.models.review import (
    ReportNotDeliverableReason,
    ReviewTask,
    ReviewTaskStatus,
)
from billiards_ai.shared.models.shot import Shot

from .approved_publication import PostgresApprovedEvidencePublisher
from .errors import PersistenceIntegrityError, ResourceNotFoundError
from .repository import PostgresQualityGateRepository, _validate_json_model, _verify_digest


class PostgresApprovedDataRepository:
    """Hydrate the existing synchronous approved repository from PostgreSQL.

    Database I/O stays explicit and async.  The returned object is the existing
    ``ApprovedDataRepository`` implementation consumed synchronously downstream;
    this adapter therefore does not create a second public repository protocol.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory
        self._quality = PostgresQualityGateRepository(session_factory)
        self._publication = PostgresApprovedEvidencePublisher(session_factory)

    async def load_repository(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
    ) -> InMemoryApprovedDataRepository:
        report, task, shots = await self._load_result_facts(report_id, revision)
        if report.session_id != session_id:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.SESSION_MISMATCH)
        expected_head = {(report_id, revision)}
        if await self._head_identities(session_id) != expected_head:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        async with self._sessions() as session:
            batch_ids = (
                (
                    await session.execute(
                        text(
                            "SELECT batch_id FROM approved_evidence_batches "
                            "WHERE session_id=:session_id AND report_id=:report_id "
                            "AND review_revision=:revision"
                        ),
                        {
                            "session_id": session_id,
                            "report_id": report_id,
                            "revision": revision,
                        },
                    )
                )
                .scalars()
                .all()
            )
        if len(batch_ids) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        try:
            publication = await self._publication.load_publication(batch_ids[0])
        except (PersistenceIntegrityError, ResourceNotFoundError) as exc:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH) from exc
        if (
            publication.batch.session_id != session_id
            or publication.batch.report_id != report_id
            or publication.batch.review_revision != revision
        ):
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        if await self._head_identities(session_id) != expected_head:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        return InMemoryApprovedDataRepository(reports=(report,), review_tasks=(task,), shots=shots)

    async def _head_identities(self, session_id: UUID) -> set[tuple[UUID, int]]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        text(
                            "WITH identities AS ("
                            "SELECT report_id,revision FROM review_reports "
                            "WHERE session_id=:session_id UNION "
                            "SELECT report_id,revision FROM review_result_reports "
                            "WHERE session_id=:session_id), maximum AS ("
                            "SELECT max(revision) AS revision FROM identities) "
                            "SELECT report_id,identities.revision FROM identities,maximum "
                            "WHERE identities.revision=maximum.revision"
                        ),
                        {"session_id": session_id},
                    )
                )
                .mappings()
                .all()
            )
        return {(row["report_id"], row["revision"]) for row in rows}

    async def _load_result_facts(
        self, report_id: UUID, revision: int
    ) -> tuple[ReportDocument, ReviewTask, tuple[Shot, ...]]:
        report = await self._quality.load_result_report(report_id, revision)
        async with self._sessions() as session:
            task_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM review_tasks WHERE result_report_id=:report_id "
                            "AND result_report_revision=:revision"
                        ),
                        {"report_id": report_id, "revision": revision},
                    )
                )
                .mappings()
                .all()
            )
            if len(task_rows) != 1:
                task_row = None
                rows = ()
            else:
                task_row = task_rows[0]
                rows = (
                    (
                        await session.execute(
                            text(
                                "SELECT * FROM review_shot_results WHERE task_id=:task_id "
                                "AND result_report_id=:report_id "
                                "AND result_report_revision=:revision ORDER BY sequence,shot_id"
                            ),
                            {
                                "task_id": task_row["task_id"],
                                "report_id": report_id,
                                "revision": revision,
                            },
                        )
                    )
                    .mappings()
                    .all()
                )
        if task_row is None or not rows:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVIEW_TASK_NOT_FOUND)
        status = ReviewTaskStatus(task_row["status"])
        try:
            task = ReviewTask(
                id=task_row["task_id"],
                session_id=task_row["session_id"],
                report_id=report_id,
                report_revision=revision,
                status=status,
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("terminal task payload is invalid") from exc
        shots: list[Shot] = []
        for row in rows:
            _verify_digest(row["payload"], row["payload_sha256"])
            shot = _validate_json_model(Shot, row["payload"])
            if (
                shot.id != row["shot_id"]
                or shot.session_id != row["session_id"]
                or shot.sequence != row["sequence"]
                or shot.review_status.value != row["review_status"]
            ):
                raise PersistenceIntegrityError("result shot columns drift from payload")
            shots.append(shot)
        return report, task, tuple(shots)
