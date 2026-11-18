from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from billiards_ai.cloud.coach_agent.assessment_repository import (
    AssessmentRepositoryError,
    AssessmentRepositoryErrorCode,
    InMemoryAssessmentRepository,
)
from billiards_ai.cloud.coach_agent.assessment_service import (
    AssessmentActorRole,
    AssessmentAuthorizationReceipt,
    AssessmentPrincipal,
    AssessmentReviewService,
    AssessmentServiceResultCode,
    AssessmentValidationService,
)
from billiards_ai.cloud.coach_agent.growth_analysis import GrowthAnalysisCompleted
from billiards_ai.shared.models.coaching import AgentRunStatus, CoachAssessmentStatus
from tests.unit.cloud.coach_agent.test_assessment_validation import _replace_assessment
from tests.unit.cloud.coach_agent.test_growth_analysis import (
    ScriptedProvider,
    _context,
    _project,
    _response,
    _window_batches,
    _workflow,
)
from tests.unit.cloud.growth.snapshot_test_support import uid


async def _draft():
    snapshot = _project(_window_batches(0, 5))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "IMPROVEMENT",
                },
                "secondary": None,
            }
        )
    )
    context = _context(snapshot)
    result = await _workflow(provider).analyze(context)
    assert isinstance(result, GrowthAnalysisCompleted)
    return context, result.assessment, result.agent_run, metric


def _ids() -> Iterator[UUID]:
    yield uid(990_001)
    yield uid(990_002)


def _times(start: datetime) -> Iterator[datetime]:
    for seconds in range(8):
        yield start + timedelta(seconds=seconds)


class _AuthorizationPort:
    def __init__(self, roles: dict[str, AssessmentActorRole]) -> None:
        self.roles = roles

    def authorize(
        self,
        *,
        principal: AssessmentPrincipal,
        assessment_id: UUID,
        target_status: CoachAssessmentStatus,
        requested_at: datetime,
    ) -> AssessmentAuthorizationReceipt:
        return AssessmentAuthorizationReceipt.build(
            assessment_id=assessment_id,
            principal_id=principal.principal_id,
            actor_role=self.roles[principal.principal_id],
            target_status=target_status,
            issued_at=requested_at,
        )


def _review_service(repository: InMemoryAssessmentRepository) -> AssessmentReviewService:
    return AssessmentReviewService(
        repository=repository,
        authorization_port=_AuthorizationPort(
            {
                "human-reviewer": AssessmentActorRole.HUMAN_REVIEWER,
                "system": AssessmentActorRole.SYSTEM,
                "ai-provider": AssessmentActorRole.AI_PROVIDER,
            }
        ),
    )


class _SlowScriptedProvider(ScriptedProvider):
    async def generate_structured(self, request):
        self.requests.append(request)
        await asyncio.sleep(0.05)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_valid_draft_is_staged_for_human_review_without_provider_repair() -> None:
    context, draft, run, _ = await _draft()
    provider = ScriptedProvider(RuntimeError("repair must not be called"))
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 11, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: next(_ids()),
        clock=lambda: next(times),
    )

    result = await service.validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="validate:assessment-1",
    )

    assert result.code is AssessmentServiceResultCode.REVIEW_REQUIRED
    assert result.assessment is not None
    assert result.assessment.status is CoachAssessmentStatus.REVIEW_REQUIRED
    assert result.assessment.revision == draft.revision + 1
    assert result.agent_run.repair_count == 0
    assert result.agent_run == run
    assert provider.requests == []
    assert repository.get_head(draft.assessment_id) == result.assessment
    assert len(repository.list_agent_runs(draft.assessment_id)) == 1
    transition = repository.list_status_transitions(draft.assessment_id)[0]
    assert transition.actor_role == AssessmentActorRole.PROGRAM_VALIDATOR.value
    assert transition.from_status is CoachAssessmentStatus.DRAFT
    assert transition.to_status is CoachAssessmentStatus.REVIEW_REQUIRED
    assert transition.from_content_sha256 == draft.content_sha256
    assert transition.validation_report_sha256 == result.validation_reports[-1].report_sha256
    assert transition.transition_sha256
    record = repository.list_validation_records(draft.assessment_id)[0]
    assert record.promotion_transition_sha256 == transition.transition_sha256
    assert record.record_sha256
    assert result.validation_reports[0].agent_run_sha256


@pytest.mark.asyncio
async def test_invalid_draft_gets_exactly_one_bounded_repair_then_succeeds() -> None:
    context, draft, run, metric = await _draft()
    invalid = _replace_assessment(draft, primary_finding_id=None)
    invalid_run = run.model_copy(update={"output_sha256": invalid.content_sha256})
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "IMPROVEMENT",
                },
                "secondary": None,
            }
        )
    )
    repository = InMemoryAssessmentRepository()
    ids = _ids()
    times = _times(datetime(2026, 8, 16, 12, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: next(ids),
        clock=lambda: next(times),
    )

    result = await service.validate_and_stage(
        context=context,
        draft=invalid,
        agent_run=invalid_run,
        idempotency_key="repair:assessment-1",
    )

    assert result.code is AssessmentServiceResultCode.REVIEW_REQUIRED
    assert result.agent_run == invalid_run
    assert len(provider.requests) == 1
    repair_payload = provider.requests[0].input_payload
    assert set(repair_payload) == {
        "schema_version",
        "metric_candidates",
        "invalid_selection",
        "validation_violation_codes",
    }
    assert set(repair_payload["invalid_selection"][0]) == {
        "metric_identity_sha256",
        "finding_type",
    }
    assert set(repair_payload["metric_candidates"][0]) == {
        "metric_identity_sha256",
        "allowed_finding_types",
    }
    serialized = provider.requests[0].model_dump_json()
    for forbidden in (
        "batch_id",
        "session_id",
        "shot_ids",
        "evidence_ref_ids",
        "finding_code",
        "summary",
        "explanation",
        "location",
        "metric_value",
    ):
        assert forbidden not in serialized
    revisions = repository.get_revisions(draft.assessment_id)
    assert revisions[0].content_sha256 == result.validation_reports[-1].assessment_content_sha256
    assert revisions[0].content_sha256 != invalid.content_sha256


@pytest.mark.asyncio
async def test_failed_repair_saves_failed_run_but_no_publishable_revision() -> None:
    context, draft, run, _ = await _draft()
    invalid = _replace_assessment(draft, primary_finding_id=None)
    invalid_run = run.model_copy(update={"output_sha256": invalid.content_sha256})
    provider = ScriptedProvider(_response({"not": "the repair schema"}))
    repository = InMemoryAssessmentRepository()
    ids = _ids()
    times = _times(datetime(2026, 8, 16, 13, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: next(ids),
        clock=lambda: next(times),
    )

    result = await service.validate_and_stage(
        context=context,
        draft=invalid,
        agent_run=invalid_run,
        idempotency_key="failed-repair:assessment-1",
    )

    assert result.code is AssessmentServiceResultCode.NO_RELIABLE_ASSESSMENT
    assert result.assessment is None
    assert result.agent_run == invalid_run
    assert len(provider.requests) == 1
    assert repository.get_head(draft.assessment_id) is None
    assert repository.list_agent_runs(draft.assessment_id) == (result.agent_run,)


@pytest.mark.asyncio
async def test_nonselection_violation_is_audited_without_provider_repair() -> None:
    context, draft, run, _ = await _draft()
    finding = draft.findings[0].model_copy(update={"summary": "一百 because pii@example.com"})
    invalid = _replace_assessment(draft, findings=(finding,))
    invalid_run = run.model_copy(update={"output_sha256": invalid.content_sha256})
    provider = ScriptedProvider(RuntimeError("must not repair non-selection failures"))
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 13, 30, tzinfo=UTC))
    result = await AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    ).validate_and_stage(
        context=context,
        draft=invalid,
        agent_run=invalid_run,
        idempotency_key="unrepairable:text",
    )

    assert result.code is AssessmentServiceResultCode.NO_RELIABLE_ASSESSMENT
    assert result.agent_run == invalid_run
    assert provider.requests == []
    assert repository.list_validation_records(draft.assessment_id)


@pytest.mark.asyncio
async def test_review_transitions_are_human_or_system_only_and_cas_idempotent() -> None:
    context, draft, run, _ = await _draft()
    repository = InMemoryAssessmentRepository()
    ids = _ids()
    times = _times(datetime(2026, 8, 16, 14, tzinfo=UTC))
    result = await AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(RuntimeError("unused")),
        uuid_factory=lambda: next(ids),
        clock=lambda: next(times),
    ).validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="validate:review-flow",
    )
    assert result.assessment is not None
    review = _review_service(repository)

    with pytest.raises(AssessmentRepositoryError) as exc:
        review.transition(
            assessment_id=draft.assessment_id,
            expected_revision=result.assessment.revision,
            target_status=CoachAssessmentStatus.APPROVED,
            principal=AssessmentPrincipal(principal_id="ai-provider"),
            idempotency_key="approve:ai",
            occurred_at=datetime(2026, 8, 16, 15, tzinfo=UTC),
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.TRANSITION_NOT_ALLOWED

    with pytest.raises(AssessmentRepositoryError) as exc:
        review.transition(
            assessment_id=draft.assessment_id,
            expected_revision=result.assessment.revision,
            target_status=CoachAssessmentStatus.APPROVED,
            principal=AssessmentPrincipal(principal_id="unknown-principal"),
            idempotency_key="approve:unknown",
            occurred_at=datetime(2026, 8, 16, 15, tzinfo=UTC),
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.TRANSITION_NOT_ALLOWED

    approved = review.transition(
        assessment_id=draft.assessment_id,
        expected_revision=result.assessment.revision,
        target_status=CoachAssessmentStatus.APPROVED,
        principal=AssessmentPrincipal(principal_id="human-reviewer"),
        idempotency_key="approve:human",
        occurred_at=datetime(2026, 8, 16, 15, tzinfo=UTC),
    )
    replay = review.transition(
        assessment_id=draft.assessment_id,
        expected_revision=result.assessment.revision,
        target_status=CoachAssessmentStatus.APPROVED,
        principal=AssessmentPrincipal(principal_id="human-reviewer"),
        idempotency_key="approve:human",
        occurred_at=datetime(2026, 8, 16, 15, tzinfo=UTC),
    )
    assert replay == approved
    assert approved.revision == result.assessment.revision + 1
    assert approved.status is CoachAssessmentStatus.APPROVED
    transition = repository.list_status_transitions(draft.assessment_id)[1]
    assert transition.actor_role == AssessmentActorRole.HUMAN_REVIEWER.value
    assert transition.from_status is CoachAssessmentStatus.REVIEW_REQUIRED
    assert transition.to_status is CoachAssessmentStatus.APPROVED

    with pytest.raises(AssessmentRepositoryError) as exc:
        review.transition(
            assessment_id=draft.assessment_id,
            expected_revision=result.assessment.revision,
            target_status=CoachAssessmentStatus.REJECTED,
            principal=AssessmentPrincipal(principal_id="human-reviewer"),
            idempotency_key="stale:reject",
            occurred_at=datetime(2026, 8, 16, 15, 1, tzinfo=UTC),
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.VERSION_CONFLICT


@pytest.mark.asyncio
async def test_cancellation_propagates_and_does_not_write_audit() -> None:
    context, draft, run, _ = await _draft()
    invalid = _replace_assessment(draft, primary_finding_id=None)
    invalid_run = run.model_copy(update={"output_sha256": invalid.content_sha256})
    repository = InMemoryAssessmentRepository()
    ids = _ids()
    times = _times(datetime(2026, 8, 16, 16, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(asyncio.CancelledError()),
        uuid_factory=lambda: next(ids),
        clock=lambda: next(times),
    )

    with pytest.raises(asyncio.CancelledError):
        await service.validate_and_stage(
            context=context,
            draft=invalid,
            agent_run=invalid_run,
            idempotency_key="cancel:repair",
        )
    assert repository.get_head(draft.assessment_id) is None
    assert repository.list_agent_runs(draft.assessment_id) == ()


@pytest.mark.asyncio
async def test_repair_idempotency_replays_without_second_provider_call() -> None:
    context, draft, run, metric = await _draft()
    invalid = _replace_assessment(draft, primary_finding_id=None)
    invalid_run = run.model_copy(update={"output_sha256": invalid.content_sha256})
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "IMPROVEMENT",
                },
                "secondary": None,
            }
        )
    )
    repository = InMemoryAssessmentRepository()
    ids = _ids()
    times = _times(datetime(2026, 8, 16, 17, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: next(ids),
        clock=lambda: next(times),
    )

    first = await service.validate_and_stage(
        context=context,
        draft=invalid,
        agent_run=invalid_run,
        idempotency_key="repair:idempotent",
    )
    replay = await service.validate_and_stage(
        context=context,
        draft=invalid,
        agent_run=invalid_run,
        idempotency_key="repair:idempotent",
    )

    assert replay == first
    assert len(provider.requests) == 1
    record = repository.list_validation_records(draft.assessment_id)[0]
    assert record.repair_request_id == provider.requests[0].request_id
    assert record.repair_request_id != run.run_id
    assert record.repair_output_sha256 is not None
    assert tuple(item.revision for item in repository.get_revisions(draft.assessment_id)) == (
        1,
        2,
    )


@pytest.mark.asyncio
async def test_repository_recomputes_idempotency_fingerprint_from_complete_command() -> None:
    context, draft, run, _ = await _draft()
    provider = ScriptedProvider(RuntimeError("unused"))
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 17, 30, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    )
    await service.validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="fingerprint:complete-command",
    )
    altered = _replace_assessment(draft, primary_finding_id=None)
    altered_run = run.model_copy(update={"output_sha256": altered.content_sha256})

    with pytest.raises(AssessmentRepositoryError) as exc:
        await service.validate_and_stage(
            context=context,
            draft=altered,
            agent_run=altered_run,
            idempotency_key="fingerprint:complete-command",
        )

    assert exc.value.code is AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT
    assert provider.requests == []


@pytest.mark.asyncio
async def test_same_logical_run_cannot_repair_again_under_a_new_key() -> None:
    context, draft, run, _ = await _draft()
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 18, tzinfo=UTC))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(RuntimeError("unused")),
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    )
    await service.validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="logical-run:first",
    )

    with pytest.raises(AssessmentRepositoryError) as exc:
        await service.validate_and_stage(
            context=context,
            draft=draft,
            agent_run=run,
            idempotency_key="logical-run:second",
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_service_rejects_unsafe_construct_before_repository_or_provider_use() -> None:
    context, draft, run, _ = await _draft()
    recursive: list[object] = []
    recursive.append(recursive)
    unsafe = draft.model_construct(**(dict(vars(draft)) | {"findings": recursive}))
    repository = InMemoryAssessmentRepository()
    provider = ScriptedProvider(RuntimeError("unsafe input must not reach provider"))
    service = AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: uid(990_001),
        clock=lambda: datetime(2026, 8, 16, 18, tzinfo=UTC),
    )

    result = await service.validate_and_stage(
        context=context,
        draft=unsafe,
        agent_run=run,
        idempotency_key="unsafe:construct",
    )
    assert result.code is AssessmentServiceResultCode.NO_RELIABLE_ASSESSMENT
    assert provider.requests == []
    assert repository.get_head(draft.assessment_id) is None
    assert repository.list_validation_records(draft.assessment_id)


@pytest.mark.asyncio
async def test_system_supersedes_approved_head_and_removes_approval_eligibility() -> None:
    context, draft, run, _ = await _draft()
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 19, tzinfo=UTC))
    staged = await AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(RuntimeError("unused")),
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    ).validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="supersede:stage",
    )
    assert staged.assessment is not None
    review = _review_service(repository)
    approved = review.transition(
        assessment_id=draft.assessment_id,
        expected_revision=staged.assessment.revision,
        target_status=CoachAssessmentStatus.APPROVED,
        principal=AssessmentPrincipal(principal_id="human-reviewer"),
        idempotency_key="supersede:approve",
        occurred_at=datetime(2026, 8, 16, 19, 1, tzinfo=UTC),
    )
    assert repository.get_approved_head(draft.assessment_id) == approved

    superseded = review.transition(
        assessment_id=draft.assessment_id,
        expected_revision=approved.revision,
        target_status=CoachAssessmentStatus.SUPERSEDED,
        principal=AssessmentPrincipal(principal_id="system"),
        idempotency_key="supersede:system",
        occurred_at=datetime(2026, 8, 16, 19, 2, tzinfo=UTC),
    )
    assert superseded.status is CoachAssessmentStatus.SUPERSEDED
    assert repository.get_approved_head(draft.assessment_id) is None
    transitions = repository.list_status_transitions(draft.assessment_id)
    assert transitions[-1].actor_role == AssessmentActorRole.SYSTEM.value
    assert transitions[1].previous_transition_sha256 == transitions[0].transition_sha256
    assert transitions[2].previous_transition_sha256 == transitions[1].transition_sha256
    assert tuple(item.occurred_at for item in transitions) == tuple(
        sorted(item.occurred_at for item in transitions)
    )


@pytest.mark.asyncio
async def test_repository_rejects_direct_review_transition_that_mutates_assessment_facts() -> None:
    context, draft, run, _ = await _draft()
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 20, tzinfo=UTC))
    staged = await AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(RuntimeError("unused")),
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    ).validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="immutable:stage",
    )
    assert staged.assessment is not None
    tampered = _replace_assessment(
        staged.assessment,
        revision=staged.assessment.revision + 1,
        status=CoachAssessmentStatus.APPROVED,
        user_id=uid(990_099),
    )

    with pytest.raises(AssessmentRepositoryError) as exc:
        receipt = AssessmentAuthorizationReceipt.build(
            assessment_id=tampered.assessment_id,
            principal_id="human-reviewer",
            actor_role=AssessmentActorRole.HUMAN_REVIEWER,
            target_status=CoachAssessmentStatus.APPROVED,
            issued_at=datetime(2026, 8, 16, 20, 1, tzinfo=UTC),
        )
        repository.append_transition(
            assessment=tampered,
            expected_revision=staged.assessment.revision,
            idempotency_key="immutable:tampered",
            command={"tampered": True},
            authorization_receipt=receipt,
            occurred_at=datetime(2026, 8, 16, 20, 1, tzinfo=UTC),
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.TRANSITION_NOT_ALLOWED
    assert repository.get_head(draft.assessment_id) == staged.assessment


@pytest.mark.asyncio
async def test_repository_returns_deep_revalidated_copies_after_object_setattr_tamper() -> None:
    context, draft, run, _ = await _draft()
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 21, tzinfo=UTC))
    staged = await AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(RuntimeError("unused")),
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    ).validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="copy:stage",
    )
    assert staged.assessment is not None
    returned = repository.get_head(draft.assessment_id)
    assert returned is not None
    object.__setattr__(returned, "status", CoachAssessmentStatus.APPROVED)

    exact = repository.get_head(draft.assessment_id)
    assert exact is not None
    assert exact.status is CoachAssessmentStatus.REVIEW_REQUIRED
    assert repository.get_approved_head(draft.assessment_id) is None


@pytest.mark.asyncio
async def test_transition_time_cannot_precede_assessment_or_previous_transition() -> None:
    context, draft, run, _ = await _draft()
    repository = InMemoryAssessmentRepository()
    times = _times(datetime(2026, 8, 16, 22, tzinfo=UTC))
    staged = await AssessmentValidationService(
        repository=repository,
        repair_provider=ScriptedProvider(RuntimeError("unused")),
        uuid_factory=lambda: uid(990_001),
        clock=lambda: next(times),
    ).validate_and_stage(
        context=context,
        draft=draft,
        agent_run=run,
        idempotency_key="time:stage",
    )
    assert staged.assessment is not None

    with pytest.raises(AssessmentRepositoryError) as exc:
        _review_service(repository).transition(
            assessment_id=draft.assessment_id,
            expected_revision=staged.assessment.revision,
            target_status=CoachAssessmentStatus.APPROVED,
            principal=AssessmentPrincipal(principal_id="human-reviewer"),
            idempotency_key="time:backwards",
            occurred_at=draft.created_at - timedelta(seconds=1),
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.INVALID_RECORD


@pytest.mark.asyncio
async def test_revision_conflict_is_rejected_before_any_provider_call() -> None:
    context, draft, run, _ = await _draft()
    revision_two = _replace_assessment(draft, revision=2)
    revision_two_run = run.model_copy(update={"output_sha256": revision_two.content_sha256})
    provider = ScriptedProvider(RuntimeError("must not be called"))
    repository = InMemoryAssessmentRepository()
    with pytest.raises(AssessmentRepositoryError) as exc:
        await AssessmentValidationService(
            repository=repository,
            repair_provider=provider,
            uuid_factory=lambda: uid(990_001),
            clock=lambda: datetime(2026, 8, 16, 22, tzinfo=UTC),
        ).validate_and_stage(
            context=context,
            draft=revision_two,
            agent_run=revision_two_run,
            idempotency_key="revision:two",
        )
    assert exc.value.code is AssessmentRepositoryErrorCode.VERSION_CONFLICT
    assert provider.requests == []


@pytest.mark.asyncio
async def test_repository_claim_allows_one_repair_across_service_instances() -> None:
    context, draft, run, metric = await _draft()
    invalid = _replace_assessment(draft, primary_finding_id=None)
    invalid_run = run.model_copy(update={"output_sha256": invalid.content_sha256})
    provider = _SlowScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "IMPROVEMENT",
                },
                "secondary": None,
            }
        )
    )
    repository = InMemoryAssessmentRepository()
    services = tuple(
        AssessmentValidationService(
            repository=repository,
            repair_provider=provider,
            uuid_factory=lambda: uid(990_001),
            clock=lambda: datetime(2026, 8, 16, 23, tzinfo=UTC),
        )
        for _ in range(2)
    )

    first, second = await asyncio.gather(
        *(
            service.validate_and_stage(
                context=context,
                draft=invalid,
                agent_run=invalid_run,
                idempotency_key="concurrent:repair",
            )
            for service in services
        )
    )

    assert first == second
    assert first.code is AssessmentServiceResultCode.REVIEW_REQUIRED
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_fallback_run_is_bounded_to_failed_audit_without_repair() -> None:
    context, draft, run, _ = await _draft()
    fallback_run = run.model_copy(update={"status": AgentRunStatus.FALLBACK, "output_sha256": None})
    provider = ScriptedProvider(RuntimeError("fallback must not trigger repair"))
    repository = InMemoryAssessmentRepository()
    result = await AssessmentValidationService(
        repository=repository,
        repair_provider=provider,
        uuid_factory=lambda: uid(990_001),
        clock=lambda: datetime(2026, 8, 16, 23, 30, tzinfo=UTC),
    ).validate_and_stage(
        context=context,
        draft=draft,
        agent_run=fallback_run,
        idempotency_key="fallback:audit",
    )

    assert result.code is AssessmentServiceResultCode.NO_RELIABLE_ASSESSMENT
    assert result.agent_run == fallback_run
    assert provider.requests == []
    assert repository.list_validation_records(draft.assessment_id)
