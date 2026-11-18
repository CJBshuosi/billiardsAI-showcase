"""Transport-neutral controlled human-review orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from billiards_ai.cloud.engine.aggregation import aggregate_session_statistics
from billiards_ai.cloud.engine.reporting import render_report_document
from billiards_ai.cloud.engine.scoring import score_session_performance
from billiards_ai.shared.models.access import AccessActorRole, ReviewerIdentity
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.quality import ShotAnalysisStatus
from billiards_ai.shared.models.review import (
    ReportRevisionReplacements,
    ReviewCorrection,
    ReviewTaskStatus,
    ShotReviewDecision,
)
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, Shot

from .application_models import (
    ApprovedReviewResult,
    ClaimReviewCommand,
    DecideReviewCommand,
    DecisionPreview,
    FinalizeReviewCommand,
    RejectedReviewResult,
    RejectReviewCommand,
    ReviewTaskDetail,
    ReviewTaskPage,
    ReviewTaskSummary,
    ShotDecisionIntent,
    ShotDecisionPreview,
    TaskRef,
)
from .identity import ReviewerIdentityProvider, SessionSubjectAuthorityProvider
from .metric_authority import HumanReviewMetricAuthorityAssembler
from .persistence import (
    IdempotencyConflictError,
    IdempotencyIdentity,
    NoProjectableApprovedEvidenceError,
    PersistenceIntegrityError,
    PostgresApprovedEvidencePublisher,
    PostgresQualityGateRepository,
    PublicationAuthorityError,
    ResourceNotFoundError,
    ReviewPublicationAuthority,
    ReviewSessionOwnership,
    RevisionConflictError,
    StoredReviewTask,
)
from .rejection import project_rejected_session, reject_review_session
from .service import apply_shot_reviews, review_session_evidence, submit_review


class ReviewErrorCode(StrEnum):
    REVIEW_ACCESS_DENIED = "REVIEW_ACCESS_DENIED"
    REVIEW_RESOURCE_UNAVAILABLE = "REVIEW_RESOURCE_UNAVAILABLE"
    REVISION_CONFLICT = "REVISION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    INVALID_DECISION_SET = "INVALID_DECISION_SET"
    INVALID_CORRECTION = "INVALID_CORRECTION"
    INVALID_CONTRACT = "INVALID_CONTRACT"
    NO_PROJECTABLE_APPROVED_EVIDENCE = "NO_PROJECTABLE_APPROVED_EVIDENCE"
    INVALID_AUTHORITY = "INVALID_AUTHORITY"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    COMMIT_STATE_UNKNOWN = "COMMIT_STATE_UNKNOWN"


class ReviewApplicationError(RuntimeError):
    """Stable, redacted application failure safe for all transports."""

    safe_message = "operation could not be completed"

    def __init__(self, code: ReviewErrorCode) -> None:
        self.code = code
        super().__init__(self.safe_message)


class ReviewMetricAssembler(Protocol):
    def assemble(self, **kwargs): ...


def _task_digest(identity: ReviewerIdentity, operation: str, command) -> str:
    intent = command.model_dump(mode="json")
    if "decisions" in intent:
        for decision in intent["decisions"]:
            decision["corrections"] = sorted(
                decision["corrections"], key=lambda item: item["field"]
            )
        intent["decisions"] = sorted(intent["decisions"], key=lambda item: item["shot_id"])
    if "metric_shots" in intent:
        intent["metric_shots"] = sorted(intent["metric_shots"], key=lambda item: item["shot_id"])
    return canonical_sha256(
        {
            "digest_schema_version": operation,
            "actor_principal_id": identity.principal_id,
            "actor_credential_id": identity.credential_id,
            "command": intent,
        }
    )


def _idempotency(
    identity: ReviewerIdentity, *, operation: str, task_id: UUID, key: str
) -> IdempotencyIdentity:
    return IdempotencyIdentity(
        actor_principal_id=identity.principal_id,
        operation=operation,
        task_id=task_id,
        shot_id=None,
        idempotency_key=key,
    )


class ReviewApplicationService:
    """The sole business entrypoint shared by CLI and FastAPI adapters."""

    def __init__(
        self,
        *,
        repository: PostgresQualityGateRepository,
        publisher: PostgresApprovedEvidencePublisher,
        identity_provider: ReviewerIdentityProvider,
        subject_provider: SessionSubjectAuthorityProvider,
        metric_assembler: HumanReviewMetricAuthorityAssembler | ReviewMetricAssembler,
        now: Callable[[], datetime],
        uuid4_factory: Callable[[], UUID],
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._identity_provider = identity_provider
        self._subject_provider = subject_provider
        self._metric_assembler = metric_assembler
        self._now = now
        self._uuid4 = uuid4_factory

    def _identity(self) -> ReviewerIdentity:
        try:
            raw = self._identity_provider.current_identity()
            identity = ReviewerIdentity.model_validate(raw.model_dump(mode="python"), strict=True)
            if AccessActorRole.REVIEWER not in identity.roles:
                raise ValueError("reviewer role absent")
            return identity
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVIEW_ACCESS_DENIED) from exc

    @staticmethod
    def _summary(stored: StoredReviewTask) -> ReviewTaskSummary:
        return ReviewTaskSummary(
            task_id=stored.task.id,
            session_id=stored.task.session_id,
            source_report_id=stored.task.report_id,
            source_report_revision=stored.task.report_revision,
            status=stored.task.status,
            lock_version=stored.lock_version,
            created_at=stored.created_at,
        )

    @staticmethod
    def _ref_matches(stored: StoredReviewTask, ref: TaskRef) -> bool:
        return (
            stored.task.id == ref.task_id
            and stored.lock_version == ref.expected_lock_version
            and stored.task.report_id == ref.source_report_id
            and stored.task.report_revision == ref.source_report_revision
        )

    async def _load_owned(
        self,
        identity: ReviewerIdentity,
        ref: TaskRef,
        *,
        require_lock_version: bool = True,
    ) -> StoredReviewTask:
        try:
            stored = await self._repository.load_task(ref.task_id)
            if (
                stored.task.id != ref.task_id
                or stored.task.report_id != ref.source_report_id
                or stored.task.report_revision != ref.source_report_revision
                or (require_lock_version and stored.lock_version != ref.expected_lock_version)
                or stored.claimed_principal_id != identity.principal_id
                or stored.claimed_credential_id != identity.credential_id
            ):
                raise ValueError("not visible")
            return stored
        except (ResourceNotFoundError, PersistenceIntegrityError, ValueError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVIEW_RESOURCE_UNAVAILABLE) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc

    async def _detail(self, stored: StoredReviewTask) -> ReviewTaskDetail:
        try:
            source = await self._repository.load_source_shots(stored.task.id)
            evidence = review_session_evidence(
                session_id=stored.task.session_id,
                shots=tuple(item.shot for item in source),
            )
            return ReviewTaskDetail(task=self._summary(stored), evidence=evidence)
        except (ResourceNotFoundError, PersistenceIntegrityError, ValueError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVIEW_RESOURCE_UNAVAILABLE) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc

    async def list_tasks(self, *, cursor: str | None = None, limit: int = 50) -> ReviewTaskPage:
        identity = self._identity()
        try:
            page = await self._repository.list_visible_tasks(
                actor_principal_id=identity.principal_id,
                credential_id=identity.credential_id,
                cursor=cursor,
                limit=limit,
            )
            return ReviewTaskPage(
                items=tuple(self._summary(item) for item in page.items),
                next_cursor=page.next_cursor,
            )
        except (ValueError, PersistenceIntegrityError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.INVALID_CONTRACT) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc

    async def claim(self, command: ClaimReviewCommand) -> ReviewTaskDetail:
        identity = self._identity()
        idem = _idempotency(
            identity,
            operation="claim_review_v1",
            task_id=command.task.task_id,
            key=command.idempotency_key,
        )
        digest = _task_digest(identity, "review-claim-intent-v1", command)
        try:
            replay = await self._repository.probe_claim_receipt(
                identity=idem, request_digest=digest
            )
        except IdempotencyConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.IDEMPOTENCY_CONFLICT) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc
        if replay is not None:
            return await self._detail(replay)
        try:
            task = await self._repository.load_task(command.task.task_id)
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc
        try:
            if not self._ref_matches(task, command.task):
                raise RevisionConflictError("stale claim")
            subject = self._subject_provider.resolve_subject(session_id=task.task.session_id)
            claimed_at = self._now()
            ownership = ReviewSessionOwnership(
                task_id=task.task.id,
                session_id=task.task.session_id,
                user_id=subject.user_id,
                bound_at=claimed_at,
            )
        except RevisionConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVISION_CONFLICT) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.INVALID_AUTHORITY) from exc
        try:
            claimed = await self._repository.claim_task_with_ownership_and_receipt(
                task_id=task.task.id,
                expected_lock_version=task.lock_version,
                actor_principal_id=identity.principal_id,
                credential_id=identity.credential_id,
                claimed_at=claimed_at,
                ownership=ownership,
                identity=idem,
                request_digest=digest,
            )
        except IdempotencyConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.IDEMPOTENCY_CONFLICT) from exc
        except RevisionConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVISION_CONFLICT) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.COMMIT_STATE_UNKNOWN) from exc
        return await self._detail(claimed)

    async def show(self, task: TaskRef) -> ReviewTaskDetail:
        identity = self._identity()
        return await self._detail(await self._load_owned(identity, task))

    @staticmethod
    def _ai_value(shot: Shot, field) -> object:
        value = getattr(shot, field.value)
        return tuple(value or ()) if field.value == "potted_balls" else value

    def _decisions(
        self,
        *,
        identity: ReviewerIdentity,
        shots: tuple[Shot, ...],
        intents: tuple[ShotDecisionIntent, ...],
        reviewed_at,
    ) -> tuple[ShotReviewDecision, ...]:
        by_shot = {shot.id: shot for shot in shots}
        if len(intents) != len(by_shot) or {item.shot_id for item in intents} != set(by_shot):
            raise ReviewApplicationError(ReviewErrorCode.INVALID_DECISION_SET)
        decisions = []
        for intent in intents:
            shot = by_shot[intent.shot_id]
            try:
                corrections = tuple(
                    ReviewCorrection(
                        shot_id=shot.id,
                        field=item.field,
                        ai_value=self._ai_value(shot, item.field),
                        correct_value=item.correct_value,
                        note=item.note,
                        reviewer=identity,
                        reviewed_at=reviewed_at,
                    )
                    for item in intent.corrections
                )
            except ValidationError as exc:
                raise ReviewApplicationError(ReviewErrorCode.INVALID_CORRECTION) from exc
            try:
                decisions.append(
                    ShotReviewDecision(
                        shot_id=shot.id,
                        outcome=intent.outcome,
                        reviewer=identity,
                        reviewed_at=reviewed_at,
                        corrections=corrections,
                        trajectory_reviewed=intent.trajectory_reviewed,
                    )
                )
            except ValidationError as exc:
                raise ReviewApplicationError(ReviewErrorCode.INVALID_DECISION_SET) from exc
        return tuple(decisions)

    async def _review_application(self, identity, command, reviewed_at):
        stored = await self._load_owned(identity, command.task)
        if stored.task.status is not ReviewTaskStatus.IN_REVIEW:
            raise ReviewApplicationError(ReviewErrorCode.REVISION_CONFLICT)
        try:
            source = await self._repository.load_source_shots(stored.task.id)
        except (ResourceNotFoundError, PersistenceIntegrityError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVIEW_RESOURCE_UNAVAILABLE) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc
        shots = tuple(item.shot for item in source)
        decisions = self._decisions(
            identity=identity,
            shots=shots,
            intents=command.decisions,
            reviewed_at=reviewed_at,
        )
        try:
            application = apply_shot_reviews(
                session_id=stored.task.session_id, shots=shots, decisions=decisions
            )
            revision = stored.task.report_revision + (1 if application.corrections else 0)
            projectable = any(
                shot.review_status in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
                and shot.analysis_status is ShotAnalysisStatus.ANALYZED
                and shot.confidence is not Confidence.ABSTAIN
                for shot in application.shots
            )
            metric = (
                self._metric_assembler.assemble(
                    session_id=stored.task.session_id,
                    review_revision=revision,
                    result_shots=application.shots,
                    inputs=command.metric_shots,
                )
                if projectable
                else None
            )
            return stored, application, metric
        except ReviewApplicationError:
            raise
        except (ValidationError, ValueError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.INVALID_DECISION_SET) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.INVALID_AUTHORITY) from exc

    async def decide(self, command: DecideReviewCommand) -> DecisionPreview:
        identity = self._identity()
        reviewed_at = self._now()
        stored, application, _ = await self._review_application(identity, command, reviewed_at)
        return DecisionPreview(
            task_id=stored.task.id,
            lock_version=stored.lock_version,
            decisions=tuple(
                ShotDecisionPreview(shot_id=item.shot_id, outcome=item.outcome)
                for item in application.decisions
            ),
        )

    @staticmethod
    def _approved_result(publication) -> ApprovedReviewResult:
        return ApprovedReviewResult(
            task_id=publication.task.task.id,
            session_id=publication.task.task.session_id,
            status=publication.task.task.status,
            report_id=publication.batch.report_id,
            report_revision=publication.batch.review_revision,
            batch_id=publication.batch.batch_id,
        )

    async def finalize(self, command: FinalizeReviewCommand) -> ApprovedReviewResult:
        identity = self._identity()
        idem = _idempotency(
            identity,
            operation="finalize_review_intent_v1",
            task_id=command.task.task_id,
            key=command.idempotency_key,
        )
        digest = _task_digest(identity, "review-finalize-intent-v1", command)
        commit_started = False
        try:
            replay = await self._publisher.probe_finalize_receipt_by_digest(
                identity=idem, request_digest=digest
            )
            if replay is not None:
                return self._approved_result(replay)
            completed_at = self._now()
            stored, application, metric = await self._review_application(
                identity, command, completed_at
            )
            if metric is None:
                raise ReviewApplicationError(ReviewErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE)
            report = await self._repository.load_report(stored.task.report_id)
            corrected_report_id = self._uuid4() if application.corrections else None
            replacements = None
            if application.corrections:
                aggregate = aggregate_session_statistics(application.shots)
                if not aggregate.deliverable or aggregate.statistics is None:
                    raise NoProjectableApprovedEvidenceError("corrected report is unavailable")
                scores = score_session_performance(aggregate.statistics)
                rendered = render_report_document(
                    report_id=corrected_report_id,
                    session_id=report.session_id,
                    generated_at=completed_at,
                    analysis_version=report.analysis_version,
                    calibration_status=report.calibration_status,
                    statistics=aggregate.statistics,
                    dimension_scores=scores,
                )
                replacements = ReportRevisionReplacements(
                    quality_summary=rendered.quality_summary,
                    session_statistics=rendered.session_statistics,
                    dimension_scores=rendered.dimension_scores,
                    typical_shots=(),
                    coaching_recommendations=(),
                )
            submission = submit_review(
                report=report,
                review_task=stored.task,
                application=application,
                completed_at=completed_at,
                corrected_report_id=corrected_report_id,
                replacements=replacements,
            )
            authority = ReviewPublicationAuthority(
                task_id=stored.task.id,
                session_id=stored.task.session_id,
                result_report_id=submission.report.report_id,
                review_revision=submission.report.revision,
                task_status=submission.review_task.status,
                batch_id=self._uuid4(),
                event_id=self._uuid4(),
                approved_at=completed_at,
                metric_authority=metric,
            )
            commit_started = True
            publication = await self._publisher.commit_finalize(
                source_task=stored,
                submission=submission,
                authority=authority,
                identity=idem,
                request_digest=digest,
            )
            return self._approved_result(publication)
        except IdempotencyConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.IDEMPOTENCY_CONFLICT) from exc
        except RevisionConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVISION_CONFLICT) from exc
        except NoProjectableApprovedEvidenceError as exc:
            raise ReviewApplicationError(ReviewErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE) from exc
        except ReviewApplicationError:
            raise
        except (PublicationAuthorityError, ValidationError, ValueError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.INVALID_AUTHORITY) from exc
        except Exception as exc:
            code = (
                ReviewErrorCode.COMMIT_STATE_UNKNOWN
                if commit_started
                else ReviewErrorCode.DEPENDENCY_UNAVAILABLE
            )
            raise ReviewApplicationError(code) from exc

    @staticmethod
    def _rejected_result(stored) -> RejectedReviewResult:
        outcome = project_rejected_session(stored.result)
        return RejectedReviewResult(
            task_id=stored.task.task.id,
            session_id=outcome.session_id,
            status=outcome.status,
            reason=outcome.reason_code,
            notification_event_id=outcome.notification_event_id,
            notification_status=outcome.notification_status,
        )

    async def reject(self, command: RejectReviewCommand) -> RejectedReviewResult:
        identity = self._identity()
        idem = _idempotency(
            identity,
            operation="reject_review_intent_v1",
            task_id=command.task.task_id,
            key=command.idempotency_key,
        )
        digest = _task_digest(identity, "review-reject-intent-v1", command)
        try:
            replay = await self._repository.probe_rejection_receipt(
                identity=idem, request_digest=digest
            )
        except IdempotencyConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.IDEMPOTENCY_CONFLICT) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc
        if replay is not None:
            return self._rejected_result(replay)
        stored = await self._load_owned(identity, command.task)
        try:
            rejected_at = self._now()
            report = await self._repository.load_report(stored.task.report_id)
            result = reject_review_session(
                report=report,
                review_task=stored.task,
                reviewer=identity,
                reason=command.reason,
                rejection_id=self._uuid4(),
                notification_event_id=self._uuid4(),
                rejected_at=rejected_at,
            )
            event = DomainEvent(
                event_id=result.notification.event_id,
                event_type="review.session.rejected.v1",
                aggregate_type="review_session",
                aggregate_id=stored.task.session_id,
                aggregate_version=stored.task.report_revision,
                occurred_at=rejected_at,
                correlation_id=stored.task.id,
                idempotency_key=(f"review-rejected:{stored.task.id}:{stored.task.report_revision}"),
                payload={
                    "session_id": str(stored.task.session_id),
                    "reason": result.rejection.reason_code.value,
                    "message": result.notification.message.value,
                    "created_at": rejected_at.isoformat(),
                },
            )
        except (ValidationError, ValueError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.INVALID_AUTHORITY) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc
        try:
            await self._repository.commit_rejection(
                result=result,
                notification_event=event,
                expected_lock_version=stored.lock_version,
                identity=idem,
                request_digest=digest,
            )
        except IdempotencyConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.IDEMPOTENCY_CONFLICT) from exc
        except RevisionConflictError as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVISION_CONFLICT) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.COMMIT_STATE_UNKNOWN) from exc
        try:
            terminal = await self._repository.load_rejected_result(task_id=stored.task.id)
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc
        return self._rejected_result(terminal)

    async def result(self, task: TaskRef) -> ApprovedReviewResult | RejectedReviewResult:
        identity = self._identity()
        stored = await self._load_owned(identity, task, require_lock_version=False)
        try:
            if stored.task.status in {ReviewTaskStatus.APPROVED, ReviewTaskStatus.CORRECTED}:
                return self._approved_result(
                    await self._publisher.load_publication_for_task(stored.task.id)
                )
            if stored.task.status is ReviewTaskStatus.REJECTED:
                return self._rejected_result(
                    await self._repository.load_rejected_result(task_id=stored.task.id)
                )
            raise ReviewApplicationError(ReviewErrorCode.REVISION_CONFLICT)
        except ReviewApplicationError:
            raise
        except (ResourceNotFoundError, PersistenceIntegrityError, ValueError) as exc:
            raise ReviewApplicationError(ReviewErrorCode.REVIEW_RESOURCE_UNAVAILABLE) from exc
        except Exception as exc:
            raise ReviewApplicationError(ReviewErrorCode.DEPENDENCY_UNAVAILABLE) from exc


def build_local_review_application_service(
    *,
    database_url: str,
    identity_provider: ReviewerIdentityProvider,
    table_binding_repository,
    calibration_store: Path,
    calibration_device_id: str,
) -> ReviewApplicationService:
    """Compose the controlled local CLI from explicit trusted stores only."""

    from billiards_ai.cloud.calibration.repository import JsonCalibrationRepository
    from billiards_ai.cloud.db.database import create_database_engine, create_session_factory
    from billiards_ai.shared.models.calibration import CalibrationLineage

    from .identity import TableBindingSessionSubjectAuthority

    if not calibration_device_id or calibration_device_id != calibration_device_id.strip():
        raise ValueError("calibration device id is invalid")
    engine = create_database_engine(database_url)
    sessions = create_session_factory(engine)
    calibration_repository = JsonCalibrationRepository(calibration_store)

    class _CalibrationProvider:
        def calibration_for(self, *, session_id: UUID) -> CalibrationLineage:
            record = calibration_repository.get_current(calibration_device_id)
            if record is None:
                raise ValueError("calibration authority is unavailable")
            return CalibrationLineage.from_record(record)

    repository = PostgresQualityGateRepository(sessions)
    return ReviewApplicationService(
        repository=repository,
        publisher=PostgresApprovedEvidencePublisher(sessions),
        identity_provider=identity_provider,
        subject_provider=TableBindingSessionSubjectAuthority(table_binding_repository),
        metric_assembler=HumanReviewMetricAuthorityAssembler(_CalibrationProvider()),
        now=lambda: datetime.now(UTC),
        uuid4_factory=uuid4,
    )


__all__ = [
    "ReviewApplicationError",
    "ReviewApplicationService",
    "ReviewErrorCode",
    "build_local_review_application_service",
]
