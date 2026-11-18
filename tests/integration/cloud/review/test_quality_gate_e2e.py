from __future__ import annotations

from pathlib import Path

import pytest

from billiards_ai.cloud.review.application import ReviewApplicationError, ReviewErrorCode
from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.cloud.review.persistence import ResourceNotFoundError
from billiards_ai.shared.models import (
    Confidence,
    ReviewStatus,
    ReviewTaskStatus,
    ShotAnalysisStatus,
    ShotReviewOutcome,
)
from billiards_ai.shared.models.events import canonical_sha256
from tests.integration.cloud.review.quality_gate_support import (
    ReviewScenario,
    authority_counts,
    claim_command,
    compose_runtime,
    finalize_command,
    load_synthetic_fixture,
    reject_command,
    stage_synthetic_graph,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "quality_gate" / "v2" / "synthetic.json"


def test_synthetic_quality_gate_fixture_is_versioned_and_deterministic() -> None:
    """Catch a missing frozen input before the PostgreSQL E2E matrix runs."""

    fixture = load_synthetic_fixture(FIXTURE)

    assert fixture.fixture_schema_version == "quality-gate-e2e-fixture-v2"
    assert tuple(shot.sequence for shot in fixture.shots) == (1, 2, 3, 4)
    assert len({shot.shot_id for shot in fixture.shots}) == 4
    assert canonical_sha256(fixture.model_dump(mode="json")) == (
        "7093818d6b77e526b3eeadcb4ef9614ba6963b872dc213b43a9cf6ce415dabcf"
    )


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_revision", "expected_sequences"),
    (
        (ReviewScenario.APPROVE, ReviewTaskStatus.APPROVED, 1, (1, 2, 3, 4)),
        (ReviewScenario.CORRECT, ReviewTaskStatus.CORRECTED, 2, (1, 2, 3, 4)),
        (
            ReviewScenario.APPROVE_WITH_REJECTED_SHOT,
            ReviewTaskStatus.APPROVED,
            1,
            (1, 3, 4),
        ),
        (
            ReviewScenario.CORRECT_WITH_REJECTED_SHOT,
            ReviewTaskStatus.CORRECTED,
            2,
            (1, 3, 4),
        ),
    ),
)
async def test_application_approved_flows_commit_one_strict_publication_graph(
    database_session_factory,
    scenario: ReviewScenario,
    expected_status: ReviewTaskStatus,
    expected_revision: int,
    expected_sequences: tuple[int, ...],
) -> None:
    """Catch any application path that bypasses or splits the terminal publication graph."""

    fixture = load_synthetic_fixture(FIXTURE)
    runtime = compose_runtime(database_session_factory, fixture)
    graph = await stage_synthetic_graph(runtime, fixture)

    page = await runtime.service.list_tasks()
    assert tuple(item.task_id for item in page.items) == (graph.task.id,)
    claim = claim_command(graph)
    claimed = await runtime.service.claim(claim)
    shown = await runtime.service.show(
        claim.task.model_copy(update={"expected_lock_version": claimed.task.lock_version})
    )
    command = finalize_command(graph, lock_version=claimed.task.lock_version, scenario=scenario)
    preview = await runtime.service.decide(command)
    finalized = await runtime.service.finalize(command)
    result = await runtime.service.result(command.task)

    assert shown.task == claimed.task
    assert tuple(item.shot_id for item in preview.decisions) == tuple(
        shot.id for shot in graph.shots
    )
    assert finalized == result
    assert finalized.status is expected_status
    assert finalized.report_revision == expected_revision

    publication = await runtime.publisher.load_publication(finalized.batch_id)
    event = await runtime.outbox.load(publication.event.event_id)
    approved_repository = await runtime.approved.load_repository(
        finalized.session_id, finalized.report_id, finalized.report_revision
    )
    approved = approved_repository.get_approved_session_data(
        finalized.session_id, finalized.report_id, finalized.report_revision
    )

    assert publication.task.task.status is expected_status
    assert tuple(shot.sequence for shot in publication.batch.shots) == expected_sequences
    assert tuple(shot.sequence for shot in publication.batch.metric_shots) == expected_sequences
    assert publication.batch.source_envelope_ids == tuple(
        graph.source_envelope_ids[graph.shots[sequence - 1].id] for sequence in expected_sequences
    )
    assert publication.batch.statistics.total_shot_count == len(expected_sequences)
    assert (
        canonical_sha256(publication.batch.model_dump(mode="json", exclude={"content_sha256"}))
        == publication.batch.content_sha256
    )
    assert event == publication.event
    assert event.event_type == "evidence.session.approved.v1"
    assert event.aggregate_type == "review_session"
    assert event.aggregate_id == graph.report.session_id
    assert event.aggregate_version == expected_revision
    assert event.causation_id == graph.task.id
    assert event.correlation_id == graph.report.session_id
    assert event.idempotency_key == (
        f"evidence-approved:{graph.report.session_id}:{expected_revision}"
    )
    assert event.payload == {"approvedEvidenceBatch": publication.batch.model_dump(mode="json")}
    assert approved.report.report_id == finalized.report_id
    if expected_revision == 2:
        assert approved.report.parent_revision_id == graph.report.report_id
        assert approved.report.report_id != graph.report.report_id
        assert approved.shots[0].cushion_count == 1
    else:
        assert approved.report.report_id == graph.report.report_id
        assert approved.report.parent_revision_id is None
        assert approved.shots[0].cushion_count == 0
    assert tuple(shot.shot_id for shot in approved.shots) == tuple(shot.id for shot in graph.shots)
    rejected_sequences = {2} if len(expected_sequences) == 3 else set()
    assert {
        shot.sequence for shot in approved.shots if shot.review_status.value == "rejected"
    } == rejected_sequences
    source = await runtime.repository.load_source_shots(graph.task.id)
    assert tuple(item.shot.id for item in source) == tuple(shot.id for shot in graph.shots)
    assert all(item.shot.review_status is ReviewStatus.PENDING for item in source)
    assert source[0].shot.cushion_count == 0

    counts = await authority_counts(database_session_factory)
    assert counts["review_tasks"] == 1
    assert counts["review_shot_decisions"] == 4
    assert counts["review_shot_results"] == 4
    assert counts["review_corrections"] == (
        1 if scenario in {ReviewScenario.CORRECT, ReviewScenario.CORRECT_WITH_REJECTED_SHOT} else 0
    )
    assert counts["approved_evidence_batches"] == 1
    assert counts["quality_gate_outbox"] == 1
    assert counts["quality_gate_idempotency_receipts"] == 2


async def test_application_session_rejection_is_notification_only(database_session_factory) -> None:
    """Catch a rejected session accidentally publishing approved evidence."""

    fixture = load_synthetic_fixture(FIXTURE)
    runtime = compose_runtime(database_session_factory, fixture)
    graph = await stage_synthetic_graph(runtime, fixture)
    claimed = await runtime.service.claim(claim_command(graph))
    command = reject_command(graph, lock_version=claimed.task.lock_version)

    rejected = await runtime.service.reject(command)
    result = await runtime.service.result(command.task)
    event = await runtime.outbox.load(rejected.notification_event_id)

    assert rejected == result
    assert rejected.status is ReviewTaskStatus.REJECTED
    assert event.event_type == "review.session.rejected.v1"
    assert event.aggregate_type == "review_session"
    assert event.aggregate_id == graph.report.session_id
    assert event.aggregate_version == graph.report.revision
    assert event.causation_id is None
    assert event.correlation_id == graph.task.id
    assert event.idempotency_key == f"review-rejected:{graph.task.id}:{graph.report.revision}"
    assert set(event.payload) == {"session_id", "reason", "message", "created_at"}
    with pytest.raises((ReportNotDeliverableError, ResourceNotFoundError)):
        await runtime.approved.load_repository(
            graph.report.session_id, graph.report.report_id, graph.report.revision
        )
    counts = await authority_counts(database_session_factory)
    assert counts["review_rejections"] == 1
    assert counts["review_result_reports"] == 0
    assert counts["approved_evidence_batches"] == 0
    assert counts["approved_batch_sources"] == 0
    assert counts["quality_gate_outbox"] == 1
    assert counts["quality_gate_idempotency_receipts"] == 2


async def test_all_rejected_is_typed_and_leaves_no_terminal_publication(
    database_session_factory,
) -> None:
    """Catch an empty approved projection committing partial terminal authority."""

    fixture = load_synthetic_fixture(FIXTURE)
    runtime = compose_runtime(database_session_factory, fixture)
    graph = await stage_synthetic_graph(runtime, fixture)
    claimed = await runtime.service.claim(claim_command(graph))
    command = finalize_command(
        graph,
        lock_version=claimed.task.lock_version,
        scenario=ReviewScenario.ALL_REJECTED,
    )

    with pytest.raises(ReviewApplicationError) as error:
        await runtime.service.finalize(command)

    assert error.value.code is ReviewErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE
    stored = await runtime.repository.load_task(graph.task.id)
    assert stored.task.status is ReviewTaskStatus.IN_REVIEW
    with pytest.raises(ReviewApplicationError) as result_error:
        await runtime.service.result(command.task)
    assert result_error.value.code is ReviewErrorCode.REVISION_CONFLICT
    counts = await authority_counts(database_session_factory)
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
    ):
        assert counts[table] == 0
    assert counts["quality_gate_idempotency_receipts"] == 1


async def test_preview_only_all_rejected_returns_the_complete_normalized_decision_set(
    database_session_factory,
) -> None:
    """Keep preview useful even when the same intent cannot produce approved evidence."""

    fixture = load_synthetic_fixture(FIXTURE)
    runtime = compose_runtime(database_session_factory, fixture)
    graph = await stage_synthetic_graph(runtime, fixture)
    claimed = await runtime.service.claim(claim_command(graph))
    command = finalize_command(
        graph,
        lock_version=claimed.task.lock_version,
        scenario=ReviewScenario.ALL_REJECTED,
    )

    preview = await runtime.service.decide(command)

    assert preview.task_id == graph.task.id
    assert preview.lock_version == claimed.task.lock_version
    assert tuple((item.shot_id, item.outcome) for item in preview.decisions) == tuple(
        (shot.id, ShotReviewOutcome.REJECTED) for shot in graph.shots
    )
    counts = await authority_counts(database_session_factory)
    assert counts["review_shot_decisions"] == 0
    assert counts["review_result_reports"] == 0
    assert counts["approved_evidence_batches"] == 0
    assert counts["quality_gate_outbox"] == 0
    assert counts["quality_gate_idempotency_receipts"] == 1


@pytest.mark.parametrize(
    "source_override",
    (
        {"analysis_status": ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE},
        {"confidence": Confidence.ABSTAIN},
    ),
    ids=("all-non-analyzed", "all-abstain"),
)
async def test_promotable_decisions_without_projectable_source_leave_no_terminal_graph(
    database_session_factory,
    source_override: dict[str, object],
) -> None:
    """A promotable decision is insufficient without projectable source evidence."""

    fixture = load_synthetic_fixture(FIXTURE)
    runtime = compose_runtime(database_session_factory, fixture)
    graph = await stage_synthetic_graph(runtime, fixture, shot_overrides=source_override)
    claimed = await runtime.service.claim(claim_command(graph))
    command = finalize_command(
        graph,
        lock_version=claimed.task.lock_version,
        scenario=ReviewScenario.APPROVE,
    )

    with pytest.raises(ReviewApplicationError) as error:
        await runtime.service.finalize(command)

    assert error.value.code is ReviewErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE
    assert all(decision.outcome is ShotReviewOutcome.APPROVED for decision in command.decisions)
    stored = await runtime.repository.load_task(graph.task.id)
    assert stored.task.status is ReviewTaskStatus.IN_REVIEW
    counts = await authority_counts(database_session_factory)
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
    ):
        assert counts[table] == 0
    assert counts["quality_gate_idempotency_receipts"] == 1
