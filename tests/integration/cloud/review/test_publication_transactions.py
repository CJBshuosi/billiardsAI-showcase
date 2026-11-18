from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from billiards_ai.cloud.engine.reporting import revise_report_document
from billiards_ai.cloud.review.batch_builder import (
    ApprovedEvidenceBatchBuilderError,
    BatchBuilderErrorCode,
)
from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.cloud.review.persistence import (
    IdempotencyConflictError,
    IdempotencyIdentity,
    PersistenceIntegrityError,
    PostgresApprovedDataRepository,
    PostgresApprovedEvidencePublisher,
    PostgresQualityGateRepository,
    PublicationAuthorityError,
    ReviewPublicationAuthority,
    ReviewSessionOwnership,
    RevisionConflictError,
)
from billiards_ai.cloud.review.persistence.approved_publication import (
    canonical_finalize_request_digest,
)
from billiards_ai.cloud.review.service import apply_shot_reviews, submit_review
from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    ReportDocument,
    ReportNotDeliverableReason,
    ReviewableField,
    ReviewCorrection,
    ReviewerIdentity,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    Shot,
    ShotAnalysisStatus,
    ShotReviewDecision,
    ShotReviewOutcome,
    TargetZoneSource,
)
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricEvidenceAuthorityRecord,
    ApprovedMetricShotEvidence,
    EvidenceApprovalKind,
    FieldSource,
    FoulResult,
    GameCode,
    MetricFieldQuality,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    ReviewedFieldQuality,
    ReviewedPoint,
    ReviewedTableGeometry,
    TargetZoneEvaluationStatus,
)

REPORT_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
NOW = datetime(2026, 8, 10, 16, tzinfo=UTC)


def _draft_facts() -> tuple[ReportDocument, ReviewTask, tuple[Shot, ...]]:
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
            ball_positions_before={0: {"x_mm": 100, "y_mm": 100}},
            ball_positions_after={0: {"x_mm": 200, "y_mm": 100}},
            cue_ball_end={"x_mm": 200, "y_mm": 100},
            struck_ball=sequence,
            potted_balls=[sequence],
            cushion_count=0,
            travel_distance_mm=100,
            confidence=Confidence.HIGH,
            analysis_status=ShotAnalysisStatus.ANALYZED,
            review_status=ReviewStatus.PENDING,
            created_at=NOW,
        )
        for sequence in (1, 2, 3, 4)
    )
    return report, task, shots


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _quality() -> ReviewedFieldQuality:
    return ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=1.0)


def _metric(shot: Shot, revision: int) -> ApprovedMetricShotEvidence:
    calibration = {
        "calibration_id": _id(800),
        "calibration_revision_id": _id(801),
        "calibration_revision_number": 1,
    }
    quality = _quality()
    return ApprovedMetricShotEvidence(
        shot_id=shot.id,
        session_id=shot.session_id,
        sequence=shot.sequence,
        review_revision=revision,
        game_code=GameCode.CHINESE_EIGHT_BALL,
        table_geometry=ReviewedTableGeometry(
            **calibration,
            table_length_mm=2540,
            table_width_mm=1270,
            ball_diameter_mm=57.15,
            corner_pocket_mouth_mm=86,
            side_pocket_mouth_mm=92,
        ),
        cue_ball_start_mm=ReviewedPoint(**calibration, x_mm=100.0, y_mm=300.0),
        object_ball_start_mm=ReviewedPoint(**calibration, x_mm=500.0, y_mm=300.0),
        intended_object_ball=shot.sequence,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=PotResult.LEGAL_SUCCESS,
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.NONE,
        target_zone=None,
        cue_ball_end_mm=None,
        target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
        target_zone_result=None,
        field_quality=MetricFieldQuality(
            geometry=quality,
            intent=quality,
            pot_attempt=quality,
            pot_result=quality,
            foul_result=quality,
            target_zone=quality,
            cue_ball_end=quality,
        ),
    )


async def _approved_facts(
    repository,
    outcomes: tuple[ShotReviewOutcome, ...] | None = None,
    *,
    bind_ownership: bool = True,
):
    report, task, shots = _draft_facts()
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
    if bind_ownership:
        await repository.bind_session_ownership(ownership)
    claimed = await repository.claim_task(
        task_id=task.id,
        expected_lock_version=0,
        actor_principal_id="reviewer-publication",
        credential_id="credential-publication",
        claimed_at=NOW,
    )
    reviewer = ReviewerIdentity(
        principal_id="reviewer-publication",
        credential_id="credential-publication",
        roles=(AccessActorRole.REVIEWER,),
    )
    selected_outcomes = outcomes or (ShotReviewOutcome.APPROVED,) * len(shots)
    decisions = tuple(
        ShotReviewDecision(
            shot_id=shot.id,
            outcome=outcome,
            reviewer=reviewer,
            reviewed_at=NOW,
            corrections=(
                (
                    ReviewCorrection(
                        shot_id=shot.id,
                        field=ReviewableField.CUSHION_COUNT,
                        ai_value=shot.cushion_count,
                        correct_value=1,
                        note="visible rail contact",
                        reviewer=reviewer,
                        reviewed_at=NOW,
                    ),
                )
                if outcome is ShotReviewOutcome.CORRECTED
                else ()
            ),
        )
        for shot, outcome in zip(shots, selected_outcomes, strict=True)
    )
    application = apply_shot_reviews(session_id=report.session_id, shots=shots, decisions=decisions)
    submission = submit_review(
        report=report,
        review_task=claimed.task,
        application=application,
        completed_at=NOW,
        corrected_report_id=(uuid4() if application.corrections else None),
    )
    metric_authority = ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version="human-review-v1",
        metric_shots=tuple(
            _metric(shot, submission.report.revision)
            for shot in submission.shots
            if shot.review_status is not ReviewStatus.REJECTED
        )
        or tuple(_metric(shot, submission.report.revision) for shot in submission.shots),
    )
    authority = ReviewPublicationAuthority(
        task_id=task.id,
        session_id=task.session_id,
        result_report_id=submission.report.report_id,
        review_revision=submission.report.revision,
        task_status=submission.review_task.status,
        batch_id=uuid4(),
        event_id=uuid4(),
        approved_at=NOW,
        metric_authority=metric_authority,
    )
    identity = IdempotencyIdentity(
        actor_principal_id=reviewer.principal_id,
        operation="finalize_review",
        task_id=task.id,
        shot_id=None,
        idempotency_key="finalize-publication-1",
    )
    return task, claimed, submission, authority, identity


def _request_digest(claimed, submission, authority) -> str:
    return canonical_finalize_request_digest(
        source_task=claimed,
        submission=submission,
        metric_authority=authority.metric_authority,
    )


async def test_session_ownership_is_prebound_idempotent_and_cannot_cross_session(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    report, task, shots = _draft_facts()
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=NOW,
    )
    authority = ReviewSessionOwnership(
        task_id=task.id,
        user_id=uuid4(),
        session_id=task.session_id,
        bound_at=NOW,
    )

    first = await repository.bind_session_ownership(authority)
    replay = await repository.bind_session_ownership(authority)

    assert replay == first == authority
    async with database_session_factory() as session:
        assert (
            await session.execute(text("SELECT count(*) FROM review_session_ownership"))
        ).scalar_one() == 1

    invalid = authority.model_copy(update={"session_id": UUID(int=0)})
    with pytest.raises((ValueError, PersistenceIntegrityError)):
        await repository.bind_session_ownership(invalid)


async def test_session_ownership_database_rejects_legal_json_tamper(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    report, task, shots = _draft_facts()
    await repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids={shot.id: uuid4() for shot in shots},
        created_at=NOW,
    )
    authority = ReviewSessionOwnership(
        task_id=task.id,
        user_id=uuid4(),
        session_id=task.session_id,
        bound_at=NOW,
    )
    await repository.bind_session_ownership(authority)
    with pytest.raises(IntegrityError, match="ownership is immutable"):
        async with database_session_factory() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE review_session_ownership "
                    "SET payload=jsonb_set(payload, '{user_id}', "
                    "to_jsonb(CAST(:user_id AS text))) WHERE task_id=:task_id"
                ),
                {"task_id": task.id, "user_id": str(uuid4())},
            )


async def test_finalize_commits_one_v2_batch_event_terminal_task_and_durable_replay(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, authority, identity = await _approved_facts(repository)

    assert (
        await publisher.probe_finalize_receipt(
            identity=identity,
            source_task=claimed,
            submission=submission,
            metric_authority=authority.metric_authority,
        )
        is None
    )
    first = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    replay = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    probe = await publisher.probe_finalize_receipt(
        identity=identity,
        source_task=claimed,
        submission=submission,
        metric_authority=authority.metric_authority,
    )

    assert first.task.task.status is ReviewTaskStatus.APPROVED
    assert first.batch.schema_version == "2.0"
    assert first.event.payload == {"approvedEvidenceBatch": first.batch.model_dump(mode="json")}
    assert replay.replayed and probe is not None and probe.replayed
    assert replay.batch == probe.batch == first.batch
    by_task = await publisher.load_publication_for_task(task.id)
    assert by_task.batch == first.batch
    assert by_task.event == first.event
    assert by_task.task == first.task
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "approved_evidence_batches",
                "quality_gate_outbox",
                "quality_gate_idempotency_receipts",
                "review_publication_lineage",
            )
        }
    assert counts == {
        "approved_evidence_batches": 1,
        "quality_gate_outbox": 1,
        "quality_gate_idempotency_receipts": 1,
        "review_publication_lineage": 1,
    }
    assert (await repository.load_task(task.id)).task.status is ReviewTaskStatus.APPROVED

    approved_repository = PostgresApprovedDataRepository(database_session_factory)
    approved_snapshot = await approved_repository.load_repository(
        first.batch.session_id, first.batch.report_id, first.batch.review_revision
    )
    approved = approved_snapshot.get_approved_session_data(
        first.batch.session_id, first.batch.report_id, first.batch.review_revision
    )
    assert approved.report == submission.report
    assert tuple(shot.shot_id for shot in approved.shots) == tuple(
        shot.id for shot in submission.shots
    )


async def test_finalize_intent_receipt_can_be_probed_before_submission_construction(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, legacy_identity = await _approved_facts(repository)
    identity = IdempotencyIdentity(
        actor_principal_id=legacy_identity.actor_principal_id,
        operation="finalize_review_intent_v1",
        task_id=legacy_identity.task_id,
        shot_id=None,
        idempotency_key="finalize-intent-digest-only-18d8cc174722",
    )
    request_digest = "d" * 64

    assert (
        await publisher.probe_finalize_receipt_by_digest(
            identity=identity,
            request_digest=request_digest,
        )
        is None
    )
    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=request_digest,
    )
    probe = await publisher.probe_finalize_receipt_by_digest(
        identity=identity,
        request_digest=request_digest,
    )

    assert probe is not None and probe.replayed
    assert probe.batch == stored.batch
    with pytest.raises(IdempotencyConflictError):
        await publisher.probe_finalize_receipt_by_digest(
            identity=identity,
            request_digest="e" * 64,
        )


async def test_same_key_with_changed_metric_facts_is_an_idempotency_conflict(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, identity = await _approved_facts(repository)
    await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    changed_metric = authority.metric_authority.metric_shots[0].model_copy(
        update={"intended_target_pocket": PocketCode.BOTTOM_LEFT}
    )
    changed_metric_authority = authority.metric_authority.model_copy(
        update={
            "metric_shots": (
                changed_metric,
                *authority.metric_authority.metric_shots[1:],
            )
        }
    )
    changed_authority = authority.model_copy(update={"metric_authority": changed_metric_authority})

    with pytest.raises(IdempotencyConflictError):
        await publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=changed_authority,
            identity=identity,
            request_digest=_request_digest(claimed, submission, changed_authority),
        )


async def test_missing_prebound_ownership_rolls_back_finalize(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, authority, identity = await _approved_facts(
        repository, bind_ownership=False
    )

    with pytest.raises(PublicationAuthorityError, match="ownership"):
        await publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=authority,
            identity=identity,
            request_digest=_request_digest(claimed, submission, authority),
        )
    assert (await repository.load_task(task.id)).task.status is ReviewTaskStatus.IN_REVIEW


async def test_finalize_preserves_rejected_audit_but_excludes_it_from_publication(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    outcomes = (
        ShotReviewOutcome.APPROVED,
        ShotReviewOutcome.REJECTED,
        ShotReviewOutcome.APPROVED,
        ShotReviewOutcome.REJECTED,
    )
    task, claimed, submission, authority, identity = await _approved_facts(repository, outcomes)

    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )

    assert tuple(shot.sequence for shot in stored.batch.shots) == (1, 3)
    assert tuple(metric.sequence for metric in stored.batch.metric_shots) == (1, 3)
    async with database_session_factory() as session:
        result_count = (
            await session.execute(
                text("SELECT count(*) FROM review_shot_results WHERE task_id=:task_id"),
                {"task_id": task.id},
            )
        ).scalar_one()
        source_count = (
            await session.execute(
                text("SELECT count(*) FROM approved_batch_sources WHERE task_id=:task_id"),
                {"task_id": task.id},
            )
        ).scalar_one()
    assert result_count == 4
    assert source_count == 2


async def test_all_rejected_finalize_rolls_back_every_staged_authority(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, authority, identity = await _approved_facts(
        repository, (ShotReviewOutcome.REJECTED,) * 4
    )

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as exc_info:
        await publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=authority,
            identity=identity,
            request_digest=_request_digest(claimed, submission, authority),
        )
    assert exc_info.value.code is BatchBuilderErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(
                    text(f"SELECT count(*) FROM {table}")  # noqa: S608
                )
            ).scalar_one()
            for table in (
                "review_result_reports",
                "review_shot_decisions",
                "review_shot_results",
                "approved_evidence_batches",
                "quality_gate_outbox",
                "quality_gate_idempotency_receipts",
            )
        }
    assert counts == {table: 0 for table in counts}
    assert (await repository.load_task(task.id)).task.status is ReviewTaskStatus.IN_REVIEW


async def test_corrected_finalize_publishes_direct_child_and_corrected_value(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, authority, identity = await _approved_facts(
        repository,
        (
            ShotReviewOutcome.CORRECTED,
            ShotReviewOutcome.APPROVED,
            ShotReviewOutcome.APPROVED,
            ShotReviewOutcome.REJECTED,
        ),
    )

    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )

    assert stored.task.task.status is ReviewTaskStatus.CORRECTED
    assert stored.batch.report_revision == task.report_revision + 1
    assert tuple(shot.sequence for shot in stored.batch.shots) == (1, 2, 3)
    assert stored.batch.shots[0].cushion_count == 1
    assert submission.report.parent_revision_id == task.report_id


async def test_concurrent_same_finalize_key_commits_once_and_strictly_replays(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, identity = await _approved_facts(repository)

    first, second = await asyncio.gather(
        publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=authority,
            identity=identity,
            request_digest=_request_digest(claimed, submission, authority),
        ),
        publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=authority,
            identity=identity,
            request_digest=_request_digest(claimed, submission, authority),
        ),
    )

    assert sorted((first.replayed, second.replayed)) == [False, True]
    assert first.batch == second.batch
    assert first.event == second.event
    assert first.task == second.task
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "approved_evidence_batches",
                "quality_gate_outbox",
                "quality_gate_idempotency_receipts",
            )
        }
    assert counts == {
        "approved_evidence_batches": 1,
        "quality_gate_outbox": 1,
        "quality_gate_idempotency_receipts": 1,
    }


async def test_concurrent_different_finalize_keys_have_one_terminal_winner_and_no_overwrite(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, first_authority, first_identity = await _approved_facts(repository)
    second_authority = first_authority.model_copy(update={"batch_id": uuid4(), "event_id": uuid4()})
    second_identity = replace(first_identity, idempotency_key="finalize-publication-rival")

    results = await asyncio.gather(
        publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=first_authority,
            identity=first_identity,
            request_digest=_request_digest(claimed, submission, first_authority),
        ),
        publisher.commit_finalize(
            source_task=claimed,
            submission=submission,
            authority=second_authority,
            identity=second_identity,
            request_digest=_request_digest(claimed, submission, second_authority),
        ),
        return_exceptions=True,
    )

    winners = tuple(result for result in results if not isinstance(result, BaseException))
    failures = tuple(result for result in results if isinstance(result, BaseException))
    assert len(winners) == len(failures) == 1
    assert isinstance(failures[0], RevisionConflictError)
    winner = winners[0]
    loser_authority = (
        second_authority if winner.batch.batch_id == first_authority.batch_id else first_authority
    )
    loser_identity = (
        second_identity if winner.batch.batch_id == first_authority.batch_id else first_identity
    )
    assert winner.replayed is False
    assert (await repository.load_task(task.id)).task.status is winner.task.task.status
    async with database_session_factory() as session:
        counts = {
            table: (
                await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in (
                "approved_evidence_batches",
                "quality_gate_outbox",
                "quality_gate_idempotency_receipts",
            )
        }
        loser_rows = (
            await session.execute(
                text(
                    "SELECT "
                    "(SELECT count(*) FROM approved_evidence_batches WHERE batch_id=:batch_id),"
                    "(SELECT count(*) FROM quality_gate_outbox WHERE event_id=:event_id),"
                    "(SELECT count(*) FROM quality_gate_idempotency_receipts "
                    "WHERE task_id=:task_id AND idempotency_key=:key)"
                ),
                {
                    "batch_id": loser_authority.batch_id,
                    "event_id": loser_authority.event_id,
                    "task_id": task.id,
                    "key": loser_identity.idempotency_key,
                },
            )
        ).one()
    assert counts == {
        "approved_evidence_batches": 1,
        "quality_gate_outbox": 1,
        "quality_gate_idempotency_receipts": 1,
    }
    assert tuple(loser_rows) == (0, 0, 0)


@pytest.mark.parametrize(
    "failure_table",
    (
        "review_result_reports",
        "review_shot_decisions",
        "review_corrections",
        "review_shot_results",
        "review_publication_lineage",
        "review_metric_authority",
        "approved_evidence_batches",
        "approved_batch_sources",
        "quality_gate_outbox",
        "quality_gate_idempotency_receipts",
    ),
)
async def test_finalize_failure_at_each_authority_stage_rolls_back_logically(
    database_session_factory,
    failure_table,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, authority, identity = await _approved_facts(
        repository,
        (
            ShotReviewOutcome.CORRECTED,
            ShotReviewOutcome.APPROVED,
            ShotReviewOutcome.APPROVED,
            ShotReviewOutcome.REJECTED,
        ),
    )
    trigger_name = f"qg_test_fail_{failure_table}"
    function_name = f"{trigger_name}_fn"
    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                f"CREATE FUNCTION {function_name}() RETURNS trigger LANGUAGE plpgsql "
                "AS $$ BEGIN RAISE EXCEPTION 'injected publication failure'; END $$"  # noqa: S608
            )
        )
        await session.execute(
            text(
                f"CREATE TRIGGER {trigger_name} BEFORE INSERT ON {failure_table} "
                f"FOR EACH ROW EXECUTE FUNCTION {function_name}()"  # noqa: S608
            )
        )
    try:
        with pytest.raises(DBAPIError, match="injected publication failure"):
            await publisher.commit_finalize(
                source_task=claimed,
                submission=submission,
                authority=authority,
                identity=identity,
                request_digest=_request_digest(claimed, submission, authority),
            )
        async with database_session_factory() as session:
            counts = {
                table: (
                    await session.execute(
                        text(f"SELECT count(*) FROM {table}")  # noqa: S608
                    )
                ).scalar_one()
                for table in (
                    "review_result_reports",
                    "review_shot_decisions",
                    "review_corrections",
                    "review_shot_results",
                    "review_publication_lineage",
                    "review_metric_authority",
                    "approved_evidence_batches",
                    "approved_batch_sources",
                    "quality_gate_outbox",
                    "quality_gate_idempotency_receipts",
                )
            }
        assert counts == {table: 0 for table in counts}
        assert (await repository.load_task(task.id)).task.status is ReviewTaskStatus.IN_REVIEW
    finally:
        async with database_session_factory() as session, session.begin():
            await session.execute(
                text(f"DROP TRIGGER IF EXISTS {trigger_name} ON {failure_table}")  # noqa: S608
            )
            await session.execute(text(f"DROP FUNCTION IF EXISTS {function_name}()"))  # noqa: S608


async def test_terminal_task_cas_failure_rolls_back_publication(database_session_factory):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    task, claimed, submission, authority, identity = await _approved_facts(repository)
    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                "CREATE FUNCTION qg_test_fail_terminal_task_fn() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN IF NEW.status IN ('approved','corrected') "
                "THEN RAISE EXCEPTION 'injected terminal task failure'; END IF; "
                "RETURN NEW; END $$"
            )
        )
        await session.execute(
            text(
                "CREATE TRIGGER qg_test_fail_terminal_task BEFORE UPDATE ON review_tasks "
                "FOR EACH ROW EXECUTE FUNCTION qg_test_fail_terminal_task_fn()"
            )
        )
    try:
        with pytest.raises(DBAPIError, match="injected terminal task failure"):
            await publisher.commit_finalize(
                source_task=claimed,
                submission=submission,
                authority=authority,
                identity=identity,
                request_digest=_request_digest(claimed, submission, authority),
            )
        async with database_session_factory() as session:
            assert (
                await session.execute(text("SELECT count(*) FROM approved_evidence_batches"))
            ).scalar_one() == 0
        assert (await repository.load_task(task.id)).task.status is ReviewTaskStatus.IN_REVIEW
    finally:
        async with database_session_factory() as session, session.begin():
            await session.execute(
                text("DROP TRIGGER IF EXISTS qg_test_fail_terminal_task ON review_tasks")
            )
            await session.execute(text("DROP FUNCTION IF EXISTS qg_test_fail_terminal_task_fn()"))


async def test_publication_loader_rejects_legal_batch_payload_tamper(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, identity = await _approved_facts(repository)
    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE approved_evidence_batches SET payload=jsonb_set(payload, "
                "'{user_id}', to_jsonb(CAST(:user_id AS text))) WHERE batch_id=:batch_id"
            ),
            {"batch_id": stored.batch.batch_id, "user_id": str(uuid4())},
        )

    with pytest.raises(PersistenceIntegrityError, match="digest"):
        await publisher.load_publication(stored.batch.batch_id)


async def test_publication_loader_rejects_review_decision_payload_tamper(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, identity = await _approved_facts(repository)
    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE review_shot_decisions SET payload=jsonb_set(payload, "
                "'{trajectory_reviewed}', 'true'::jsonb) WHERE task_id=:task_id"
            ),
            {"task_id": stored.task.task.id},
        )

    with pytest.raises(PersistenceIntegrityError, match="digest"):
        await publisher.load_publication(stored.batch.batch_id)


async def test_newer_source_draft_makes_older_approved_snapshot_stale(
    database_session_factory,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, identity = await _approved_facts(repository)
    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    newer_report = revise_report_document(
        submission.report,
        report_id=uuid4(),
        generated_at=NOW + timedelta(seconds=1),
    )
    newer_task = ReviewTask(
        id=uuid4(),
        session_id=newer_report.session_id,
        report_id=newer_report.report_id,
        report_revision=newer_report.revision,
        status=ReviewTaskStatus.PENDING_REVIEW,
    )
    pending_shots = tuple(
        shot.model_copy(update={"review_status": ReviewStatus.PENDING}) for shot in submission.shots
    )
    await repository.stage_review_draft(
        report=newer_report,
        task=newer_task,
        shots=pending_shots,
        source_envelope_ids={shot.id: uuid4() for shot in pending_shots},
        created_at=NOW + timedelta(seconds=1),
    )

    approved_bridge = PostgresApprovedDataRepository(database_session_factory)
    with pytest.raises(ReportNotDeliverableError) as exc_info:
        await approved_bridge.load_repository(
            stored.batch.session_id,
            stored.batch.report_id,
            stored.batch.review_revision,
        )
    assert exc_info.value.reason is ReportNotDeliverableReason.REVISION_MISMATCH


async def test_approved_bridge_rechecks_head_after_publication_rebuild(
    database_session_factory,
    monkeypatch,
):
    repository = PostgresQualityGateRepository(database_session_factory)
    publisher = PostgresApprovedEvidencePublisher(database_session_factory)
    _task, claimed, submission, authority, identity = await _approved_facts(repository)
    stored = await publisher.commit_finalize(
        source_task=claimed,
        submission=submission,
        authority=authority,
        identity=identity,
        request_digest=_request_digest(claimed, submission, authority),
    )
    approved_bridge = PostgresApprovedDataRepository(database_session_factory)
    original_load = approved_bridge._publication.load_publication  # noqa: SLF001

    async def load_while_new_head_commits(batch_id):
        publication = await original_load(batch_id)
        newer_report = revise_report_document(
            submission.report,
            report_id=uuid4(),
            generated_at=NOW + timedelta(seconds=1),
        )
        pending_shots = tuple(
            shot.model_copy(update={"review_status": ReviewStatus.PENDING})
            for shot in submission.shots
        )
        await repository.stage_review_draft(
            report=newer_report,
            task=ReviewTask(
                id=uuid4(),
                session_id=newer_report.session_id,
                report_id=newer_report.report_id,
                report_revision=newer_report.revision,
                status=ReviewTaskStatus.PENDING_REVIEW,
            ),
            shots=pending_shots,
            source_envelope_ids={shot.id: uuid4() for shot in pending_shots},
            created_at=NOW + timedelta(seconds=1),
        )
        return publication

    monkeypatch.setattr(
        approved_bridge._publication,  # noqa: SLF001
        "load_publication",
        load_while_new_head_commits,
    )
    with pytest.raises(ReportNotDeliverableError) as exc_info:
        await approved_bridge.load_repository(
            stored.batch.session_id,
            stored.batch.report_id,
            stored.batch.review_revision,
        )
    assert exc_info.value.reason is ReportNotDeliverableReason.REVISION_MISMATCH
