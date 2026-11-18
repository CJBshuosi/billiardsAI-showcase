"""Transport-neutral controlled review application orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from billiards_ai.cloud.review.application import (
    ReviewApplicationError,
    ReviewApplicationService,
    ReviewErrorCode,
    _task_digest,
)
from billiards_ai.cloud.review.application_models import (
    ClaimReviewCommand,
    CorrectionIntent,
    DecideReviewCommand,
    FinalizeReviewCommand,
    RejectReviewCommand,
    ShotDecisionIntent,
    TaskRef,
)
from billiards_ai.cloud.review.metric_authority import (
    MetricFieldReviewStates,
    MetricPointInput,
    MetricReviewState,
    MetricShotReviewInput,
)
from billiards_ai.cloud.review.persistence import StoredReviewTask, StoredSourceShot
from billiards_ai.cloud.review.rejection import reject_review_session
from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    FoulResult,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    ReportDocument,
    ReviewableField,
    ReviewerIdentity,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    SessionRejectionReason,
    Shot,
    ShotReviewOutcome,
    TargetZoneEvaluationStatus,
    TargetZoneSource,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
NOW = datetime(2026, 8, 10, 18, tzinfo=UTC)


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _identity(*, reviewer: bool = True, credential: str = "credential-1") -> ReviewerIdentity:
    return ReviewerIdentity(
        principal_id="reviewer-1",
        credential_id=credential,
        roles=(AccessActorRole.REVIEWER if reviewer else AccessActorRole.SYSTEM,),
    )


class _IdentityProvider:
    def __init__(self, identity: ReviewerIdentity | Exception) -> None:
        self.identity = identity

    def current_identity(self) -> ReviewerIdentity:
        if isinstance(self.identity, Exception):
            raise self.identity
        return self.identity


def _facts(*, owner: bool = True):
    report = ReportDocument.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    task = ReviewTask(
        id=_id(1),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        status=ReviewTaskStatus.IN_REVIEW,
    )
    stored = StoredReviewTask(
        task=task,
        lock_version=2,
        claimed_principal_id="reviewer-1" if owner else "other-reviewer",
        claimed_credential_id="credential-1" if owner else "other-credential",
        claimed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        completed_at=None,
        result_report_id=None,
        result_report_revision=None,
    )
    shots = tuple(
        Shot(
            id=_id(100 + sequence),
            session_id=report.session_id,
            sequence=sequence,
            t_start_ms=(sequence - 1) * 1000,
            t_end_ms=sequence * 1000,
            ball_positions_before=None,
            ball_positions_after=None,
            cue_ball_end=None,
            struck_ball=sequence,
            potted_balls=[],
            cushion_count=0,
            travel_distance_mm=400,
            is_foul=False,
            deviation_mm=None,
            target_zone_source=TargetZoneSource.NONE,
            confidence=Confidence.HIGH,
            review_status=ReviewStatus.IN_REVIEW,
            raw_observation={"media_locator": "must-not-leak"},
            created_at=NOW,
        )
        for sequence in range(1, report.session_statistics.total_shot_count + 1)
    )
    return report, stored, shots


class _Repository:
    def __init__(self, *, owner: bool = True, missing: bool = False) -> None:
        self.report, self.task, self.shots = _facts(owner=owner)
        self.missing = missing
        self.mutations = 0
        self.claim_replay = None
        self.rejection_replay = None

    async def load_task(self, task_id):
        if self.missing:
            from billiards_ai.cloud.review.persistence import ResourceNotFoundError

            raise ResourceNotFoundError("specific storage detail")
        return self.task

    async def load_report(self, report_id):
        return self.report

    async def load_source_shots(self, task_id):
        return tuple(
            StoredSourceShot(shot=shot, source_envelope_id=_id(300 + shot.sequence))
            for shot in self.shots
        )

    async def probe_claim_receipt(self, **kwargs):
        return self.claim_replay

    async def probe_rejection_receipt(self, **kwargs):
        return self.rejection_replay

    async def list_visible_tasks(self, **kwargs):
        return SimpleNamespace(items=(self.task,), next_cursor=None)

    async def claim_task_with_ownership_and_receipt(self, **kwargs):
        raise AssertionError("claim mutation was not configured")

    async def commit_rejection(self, **kwargs):
        raise AssertionError("rejection mutation was not configured")


class _Publisher:
    def __init__(self) -> None:
        self.replay = None
        self.publication = None

    async def probe_finalize_receipt_by_digest(self, **kwargs):
        return self.replay

    async def load_publication_for_task(self, task_id):
        if self.publication is None:
            raise AssertionError("publication was not configured")
        return self.publication


class _SubjectProvider:
    def resolve_subject(self, *, session_id):
        return SimpleNamespace(session_id=session_id, user_id=_id(900), accepted_at=NOW)


class _MetricAssembler:
    def __init__(self) -> None:
        self.calls = 0

    def assemble(self, **kwargs):
        self.calls += 1
        return SimpleNamespace()


class _CountingFactories:
    def __init__(self) -> None:
        self.clock_calls = 0
        self.uuid_calls = 0

    def now(self):
        self.clock_calls += 1
        return NOW

    def uuid4(self):
        self.uuid_calls += 1
        return _id(990 + self.uuid_calls)


def _task_ref(stored: StoredReviewTask) -> TaskRef:
    return TaskRef(
        task_id=stored.task.id,
        expected_lock_version=stored.lock_version,
        source_report_id=stored.task.report_id,
        source_report_revision=stored.task.report_revision,
    )


def _metric(shot_id: UUID) -> MetricShotReviewInput:
    state = MetricReviewState.CONFIRMED
    return MetricShotReviewInput(
        shot_id=shot_id,
        cue_ball_start_mm=MetricPointInput(x_mm=100.0, y_mm=200.0),
        object_ball_start_mm=MetricPointInput(x_mm=500.0, y_mm=200.0),
        intended_object_ball=3,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=PotResult.MISS,
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.NONE,
        target_zone=None,
        cue_ball_end_mm=None,
        target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
        review_states=MetricFieldReviewStates(
            geometry=state,
            intent=state,
            pot_attempt=state,
            pot_result=state,
            foul_result=state,
            target_zone=state,
            cue_ball_end=MetricReviewState.UNSUPPORTED,
        ),
    )


def _service(repository: _Repository, *, reviewer: bool = True):
    factories = _CountingFactories()
    publisher = _Publisher()
    service = ReviewApplicationService(
        repository=repository,
        publisher=publisher,
        identity_provider=_IdentityProvider(_identity(reviewer=reviewer)),
        subject_provider=_SubjectProvider(),
        metric_assembler=_MetricAssembler(),
        now=factories.now,
        uuid4_factory=factories.uuid4,
    )
    return service, publisher, factories


@pytest.mark.asyncio
async def test_identity_and_role_fail_before_repository_or_factories() -> None:
    repository = _Repository()
    service, _, factories = _service(repository, reviewer=False)

    with pytest.raises(ReviewApplicationError) as error:
        await service.show(_task_ref(repository.task))

    assert error.value.code is ReviewErrorCode.REVIEW_ACCESS_DENIED
    assert error.value.safe_message == "operation could not be completed"
    assert factories.clock_calls == factories.uuid_calls == 0


@pytest.mark.asyncio
async def test_list_returns_only_repository_safe_summaries() -> None:
    repository = _Repository()
    service, _, _ = _service(repository)

    page = await service.list_tasks(limit=20)

    assert len(page.items) == 1
    assert page.items[0].task_id == repository.task.task.id
    assert "credential" not in page.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", [_Repository(owner=False), _Repository(missing=True)])
async def test_unknown_and_non_owner_are_indistinguishable(repository: _Repository) -> None:
    service, _, _ = _service(repository)

    with pytest.raises(ReviewApplicationError) as error:
        await service.show(_task_ref(repository.task))

    assert error.value.code is ReviewErrorCode.REVIEW_RESOURCE_UNAVAILABLE
    assert str(error.value) == "operation could not be completed"


@pytest.mark.asyncio
async def test_decide_rebuilds_ai_value_and_is_preview_only() -> None:
    repository = _Repository()
    service, _, _ = _service(repository)
    decisions = tuple(
        ShotDecisionIntent(
            shot_id=shot.id,
            outcome=(ShotReviewOutcome.CORRECTED if index == 0 else ShotReviewOutcome.APPROVED),
            trajectory_reviewed=False,
            corrections=(
                CorrectionIntent(
                    field=ReviewableField.CUSHION_COUNT,
                    correct_value=2,
                    note="visible second rail contact",
                ),
            )
            if index == 0
            else (),
        )
        for index, shot in enumerate(repository.shots)
    )
    command = DecideReviewCommand(
        task=_task_ref(repository.task),
        decisions=decisions,
        metric_shots=tuple(_metric(shot.id) for shot in repository.shots),
    )

    preview = await service.decide(command)

    assert preview.task_id == repository.task.task.id
    assert preview.decisions[0].outcome is ShotReviewOutcome.CORRECTED
    assert repository.mutations == 0


@pytest.mark.asyncio
async def test_decide_maps_noop_correction_to_invalid_correction() -> None:
    repository = _Repository()
    service, _, _ = _service(repository)
    decisions = tuple(
        ShotDecisionIntent(
            shot_id=shot.id,
            outcome=(ShotReviewOutcome.CORRECTED if index == 0 else ShotReviewOutcome.APPROVED),
            trajectory_reviewed=False,
            corrections=(
                CorrectionIntent(
                    field=ReviewableField.CUSHION_COUNT,
                    correct_value=0,
                    note="reviewed value matches the source",
                ),
            )
            if index == 0
            else (),
        )
        for index, shot in enumerate(repository.shots)
    )

    with pytest.raises(ReviewApplicationError) as error:
        await service.decide(
            DecideReviewCommand(
                task=_task_ref(repository.task),
                decisions=decisions,
                metric_shots=tuple(_metric(shot.id) for shot in repository.shots),
            )
        )

    assert error.value.code is ReviewErrorCode.INVALID_CORRECTION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ("show", "claim_probe", "finalize_probe", "reject_probe", "result_read")
)
async def test_precommit_dependency_failures_are_stable(operation: str) -> None:
    repository = _Repository()
    publisher = _Publisher()
    service, _, _ = _service(repository)
    service._publisher = publisher  # noqa: SLF001 - inject one failing boundary

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("database host and credentials must stay private")

    if operation == "show":
        repository.load_task = unavailable
        invocation = service.show(_task_ref(repository.task))
    elif operation == "claim_probe":
        repository.probe_claim_receipt = unavailable
        pending = repository.task.task.model_copy(
            update={"status": ReviewTaskStatus.PENDING_REVIEW}
        )
        repository.task = replace(repository.task, task=pending, lock_version=0)
        invocation = service.claim(
            ClaimReviewCommand(
                task=_task_ref(repository.task),
                idempotency_key="claim-precommit-dependency-0123456789abcdef",
            )
        )
    elif operation == "finalize_probe":
        publisher.probe_finalize_receipt_by_digest = unavailable
        invocation = service.finalize(_finalize_command(repository))
    elif operation == "reject_probe":
        repository.probe_rejection_receipt = unavailable
        invocation = service.reject(
            RejectReviewCommand(
                task=_task_ref(repository.task),
                reason=SessionRejectionReason.OCCLUDED,
                idempotency_key="reject-precommit-dependency-0123456789abcdef",
            )
        )
    else:
        source_ref = _task_ref(repository.task)
        terminal_task = repository.task.task.model_copy(
            update={"status": ReviewTaskStatus.APPROVED}
        )
        repository.task = replace(
            repository.task,
            task=terminal_task,
            lock_version=source_ref.expected_lock_version + 1,
            completed_at=NOW,
            result_report_id=repository.report.report_id,
            result_report_revision=repository.report.revision,
        )
        publisher.load_publication_for_task = unavailable
        invocation = service.result(source_ref)

    with pytest.raises(ReviewApplicationError) as error:
        await invocation

    assert error.value.code is ReviewErrorCode.DEPENDENCY_UNAVAILABLE


@pytest.mark.asyncio
async def test_claim_commit_uncertainty_is_not_misreported_as_invalid_authority() -> None:
    repository = _Repository()
    pending = repository.task.task.model_copy(update={"status": ReviewTaskStatus.PENDING_REVIEW})
    repository.task = replace(repository.task, task=pending, lock_version=0)
    service, _, _ = _service(repository)

    async def uncertain_commit(**_kwargs):
        raise RuntimeError("connection dropped after commit dispatch")

    repository.claim_task_with_ownership_and_receipt = uncertain_commit

    with pytest.raises(ReviewApplicationError) as error:
        await service.claim(
            ClaimReviewCommand(
                task=_task_ref(repository.task),
                idempotency_key="claim-commit-unknown-0123456789abcdef",
            )
        )

    assert error.value.code is ReviewErrorCode.COMMIT_STATE_UNKNOWN


@pytest.mark.asyncio
async def test_terminal_result_uses_owner_and_source_identity_without_terminal_lock_guess() -> None:
    repository = _Repository()
    source_ref = _task_ref(repository.task)
    terminal_task = repository.task.task.model_copy(update={"status": ReviewTaskStatus.APPROVED})
    repository.task = replace(
        repository.task,
        task=terminal_task,
        lock_version=source_ref.expected_lock_version + 1,
        completed_at=NOW,
        result_report_id=repository.report.report_id,
        result_report_revision=repository.report.revision,
    )
    service, publisher, _ = _service(repository)
    publisher.publication = SimpleNamespace(
        task=repository.task,
        batch=SimpleNamespace(
            batch_id=_id(778),
            report_id=repository.report.report_id,
            review_revision=repository.report.revision,
        ),
    )

    result = await service.result(source_ref)

    assert result.status is ReviewTaskStatus.APPROVED
    assert result.batch_id == _id(778)


def _finalize_command(repository: _Repository) -> FinalizeReviewCommand:
    return FinalizeReviewCommand(
        task=_task_ref(repository.task),
        decisions=tuple(
            ShotDecisionIntent(
                shot_id=shot.id,
                outcome=ShotReviewOutcome.APPROVED,
                trajectory_reviewed=False,
            )
            for shot in repository.shots
        ),
        metric_shots=tuple(_metric(shot.id) for shot in repository.shots),
        idempotency_key="finalize-canonical-intent-0123456789abcdef",
    )


def test_finalize_intent_digest_is_order_independent_for_shots_and_corrections() -> None:
    repository = _Repository()
    command = _finalize_command(repository)
    first, second = repository.shots[:2]
    corrected = ShotDecisionIntent(
        shot_id=first.id,
        outcome=ShotReviewOutcome.CORRECTED,
        trajectory_reviewed=False,
        corrections=(
            CorrectionIntent(
                field=ReviewableField.TRAVEL_DISTANCE_MM,
                correct_value=401,
                note="distance reviewed",
            ),
            CorrectionIntent(
                field=ReviewableField.CUSHION_COUNT,
                correct_value=1,
                note="rail contact reviewed",
            ),
        ),
    )
    other = ShotDecisionIntent(
        shot_id=second.id,
        outcome=ShotReviewOutcome.APPROVED,
        trajectory_reviewed=False,
    )
    forward = command.model_copy(
        update={
            "decisions": (corrected, other),
            "metric_shots": (_metric(first.id), _metric(second.id)),
        }
    )
    reversed_command = command.model_copy(
        update={
            "decisions": (
                other,
                corrected.model_copy(
                    update={"corrections": tuple(reversed(corrected.corrections))}
                ),
            ),
            "metric_shots": (_metric(second.id), _metric(first.id)),
        }
    )

    assert _task_digest(_identity(), "review-finalize-intent-v1", forward) == _task_digest(
        _identity(), "review-finalize-intent-v1", reversed_command
    )


@pytest.mark.asyncio
async def test_finalize_replay_probes_before_clock_uuid_and_authority_factories() -> None:
    repository = _Repository()
    service, publisher, factories = _service(repository)
    publisher.replay = SimpleNamespace(
        task=replace(
            repository.task,
            task=repository.task.task.model_copy(update={"status": ReviewTaskStatus.APPROVED}),
            result_report_id=repository.report.report_id,
            result_report_revision=repository.report.revision,
        ),
        batch=SimpleNamespace(
            batch_id=_id(777),
            report_id=repository.report.report_id,
            review_revision=repository.report.revision,
        ),
    )
    decisions = tuple(
        ShotDecisionIntent(
            shot_id=shot.id,
            outcome=ShotReviewOutcome.APPROVED,
            trajectory_reviewed=False,
        )
        for shot in repository.shots
    )
    command = FinalizeReviewCommand(
        task=_task_ref(repository.task),
        decisions=decisions,
        metric_shots=tuple(_metric(shot.id) for shot in repository.shots),
        idempotency_key="finalize-replay-0001-0123456789abcdef",
    )

    result = await service.finalize(command)

    assert result.batch_id == _id(777)
    assert factories.clock_calls == factories.uuid_calls == 0


@pytest.mark.asyncio
async def test_claim_and_reject_replays_probe_before_factories() -> None:
    repository = _Repository()
    service, _, factories = _service(repository)
    repository.claim_replay = repository.task
    rejected_task = repository.task.task.model_copy(update={"status": ReviewTaskStatus.REJECTED})
    domain_result = reject_review_session(
        report=repository.report,
        review_task=repository.task.task,
        reviewer=_identity(),
        reason=SessionRejectionReason.OCCLUDED,
        rejection_id=_id(887),
        notification_event_id=_id(888),
        rejected_at=NOW,
    )
    repository.rejection_replay = SimpleNamespace(
        task=SimpleNamespace(task=rejected_task),
        result=domain_result,
    )

    claimed = await service.claim(
        ClaimReviewCommand(
            task=_task_ref(repository.task),
            idempotency_key="claim-replay-00001-0123456789abcdef",
        )
    )
    rejected = await service.reject(
        RejectReviewCommand(
            task=_task_ref(repository.task),
            reason=SessionRejectionReason.OCCLUDED,
            idempotency_key="reject-replay-0001-0123456789abcdef",
        )
    )

    assert claimed.task.task_id == repository.task.task.id
    assert rejected.notification_event_id == _id(888)
    assert factories.clock_calls == factories.uuid_calls == 0
