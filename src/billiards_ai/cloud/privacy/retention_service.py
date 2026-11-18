"""Authoritative retention milestones and deletion-request state machine."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from billiards_ai.shared.constants import CLIP_RETENTION_DAYS, RETENTION_CLOCK_SKEW_MINUTES
from billiards_ai.shared.models import (
    AssetContext,
    DeletionAttempt,
    DeletionEvent,
    DeletionEventType,
    DeletionFailureCode,
    DeletionRequest,
    DeletionRequestStatus,
    DeletionScope,
    MaterialUseDecision,
    MaterialUsePurpose,
    MediaDeletionStatus,
    PublicDeletionStatus,
    RetainedMediaAsset,
    RetainedMediaAssetCreate,
)

from .authorization import authorize_real_video_use
from .media_store import ControlledFilesystemMediaStore
from .repository import JsonGateRepository
from .retention_repository import (
    JsonRetentionRepository,
    MediaUnavailableError,
    RetentionConflictError,
    RetentionError,
    RetentionNotFoundError,
    RetentionOwnershipError,
    RetentionRepositoryError,
)


class MediaRetentionService:
    def __init__(
        self,
        repository: JsonRetentionRepository,
        *,
        gate_repository: JsonGateRepository | None = None,
        media_store: ControlledFilesystemMediaStore | None = None,
        session_media_authority: object | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.repository = repository
        self.gate_repository = gate_repository
        self.media_store = media_store
        self.session_media_authority = session_media_authority
        self._now = now or (lambda: datetime.now(UTC))

    def register_asset(
        self,
        asset: RetainedMediaAssetCreate,
        *,
        gate_revision_id: UUID,
        asset_context: AssetContext,
        requested_purpose: MaterialUsePurpose,
    ) -> RetainedMediaAsset:
        """Register real media only after the Story 1.1 combined authorization."""

        if self.gate_repository is None:
            raise RetentionRepositoryError("a trusted gate repository is required")
        self._validate_gate_asset_binding(asset, asset_context)
        decision = authorize_real_video_use(
            self.gate_repository, gate_revision_id, asset_context, requested_purpose
        )
        if decision.decision is not MaterialUseDecision.ALLOW:
            raise RetentionOwnershipError(f"real media was not authorized: {decision.reason_code}")
        now = self._now().astimezone(UTC)
        retained = RetainedMediaAsset(
            **asset.model_dump(),
            deletion_status=MediaDeletionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        return self.repository._register_authorized_asset(retained)

    def register_pending_asset(
        self,
        asset: RetainedMediaAssetCreate,
        *,
        gate_revision_id: UUID,
        asset_context: AssetContext,
        requested_purpose: MaterialUsePurpose,
    ) -> RetainedMediaAsset:
        if self.gate_repository is None:
            raise RetentionRepositoryError("a trusted gate repository is required")
        self._validate_gate_asset_binding(asset, asset_context)
        decision = authorize_real_video_use(
            self.gate_repository, gate_revision_id, asset_context, requested_purpose
        )
        if decision.decision is not MaterialUseDecision.ALLOW:
            raise RetentionOwnershipError(f"real media was not authorized: {decision.reason_code}")
        now = self._now().astimezone(UTC)
        retained = RetainedMediaAsset(
            **asset.model_dump(),
            deletion_status=MediaDeletionStatus.PENDING_BINDING,
            created_at=now,
            updated_at=now,
        )
        return self.repository.register_pending_binding(retained)

    def activate_pending_asset(self, *, asset_id: UUID, session_id: UUID) -> RetainedMediaAsset:
        return self.repository.activate_pending_binding(asset_id, expected_session_id=session_id)

    def _validate_gate_asset_binding(
        self, asset: RetainedMediaAsset | RetainedMediaAssetCreate, asset_context: AssetContext
    ) -> None:
        if (
            asset_context.asset_id != str(asset.asset_id)
            or asset_context.asset_sha256 != asset.content_sha256
            or asset_context.captured_at != asset.captured_at
        ):
            raise RetentionConflictError("asset context does not match retention asset")
        provenance = asset_context.source_asset_provenance
        if provenance is None:
            return
        try:
            source_id = UUID(provenance.asset_id)
        except ValueError as exc:
            raise RetentionConflictError("source provenance asset ID is invalid") from exc
        if source_id not in asset.source_asset_ids:
            raise RetentionConflictError("source provenance is not a retention source")
        source = self.repository.get_asset(source_id)
        if (
            source.content_sha256 != provenance.sha256
            or source.captured_at != provenance.captured_at
        ):
            raise RetentionConflictError("source provenance does not match retention source")

    def _record_milestone(
        self, asset_id: UUID, field: str, occurred_at: datetime
    ) -> RetainedMediaAsset:
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("milestone time must include a timezone")
        occurred_at = occurred_at.astimezone(UTC)
        if occurred_at > self._now().astimezone(UTC) + timedelta(
            minutes=RETENTION_CLOCK_SKEW_MINUTES
        ):
            raise ValueError("milestone time cannot be more than five minutes in the future")
        asset = self.repository.get_asset(asset_id)
        existing = getattr(asset, field)
        # Repeated milestones are idempotent. Any correction that could alter an
        # active deadline needs a future explicit policy workflow.
        value = existing or occurred_at
        updates = {field: value, "updated_at": self._now().astimezone(UTC)}
        reviewed = value if field == "reviewed_at" else asset.reviewed_at
        delivered = value if field == "delivered_at" else asset.delivered_at
        if reviewed is not None and delivered is not None and asset.retention_expires_at is None:
            updates["retention_expires_at"] = max(reviewed, delivered) + timedelta(
                days=CLIP_RETENTION_DAYS
            )
        updated = asset.model_copy(update=updates)
        # Revalidate because model_copy intentionally does not run Pydantic validators.
        updated = RetainedMediaAsset.model_validate(updated.model_dump())
        return self.repository.save_asset(updated, expected_current=asset)

    def record_reviewed(self, asset_id: UUID, reviewed_at: datetime) -> RetainedMediaAsset:
        return self._record_milestone(asset_id, "reviewed_at", reviewed_at)

    def record_delivered(self, asset_id: UUID, delivered_at: datetime) -> RetainedMediaAsset:
        return self._record_milestone(asset_id, "delivered_at", delivered_at)

    def list_due_assets(self, *, at: datetime | None = None) -> tuple[RetainedMediaAsset, ...]:
        cutoff = (at or self._now()).astimezone(UTC)
        return tuple(
            item
            for item in self.repository.list_assets()
            if item.deletion_status is MediaDeletionStatus.ACTIVE
            and item.retention_expires_at is not None
            and item.retention_expires_at <= cutoff
        )

    def abandon_pending_asset(self, *, asset_id: UUID, session_id: UUID) -> None:
        self.repository.abandon_pending_binding(asset_id, expected_session_id=session_id)

    def quarantine_unbound_asset(self, *, asset_id: UUID, session_id: UUID) -> None:
        self.repository.quarantine_unbound_asset(asset_id, expected_session_id=session_id)

    def submit_due_deletions(self, *, at: datetime | None = None) -> tuple[DeletionRequest, ...]:
        """Create stable, repeatable deletion requests for due assets not already queued."""

        active_asset_ids = {
            asset_id
            for request in self.repository.list_requests()
            if request.status
            in {
                DeletionRequestStatus.RECEIVED,
                DeletionRequestStatus.IN_PROGRESS,
                DeletionRequestStatus.COMPLETED,
            }
            for asset_id in request.frozen_asset_ids
        }
        created: list[DeletionRequest] = []
        for asset in sorted(self.list_due_assets(at=at), key=lambda item: str(item.asset_id)):
            if asset.asset_id in active_asset_ids:
                continue
            created.append(
                self.submit_deletion(
                    current_user_id=asset.owner_user_id,
                    scope=DeletionScope(asset_ids=(asset.asset_id,)),
                    idempotency_key=f"retention-expiry:{asset.asset_id}",
                )
            )
            active_asset_ids.update(created[-1].frozen_asset_ids)
        return tuple(created)

    def _resolve_scope(self, user_id: str, scope: DeletionScope) -> tuple[UUID, ...]:
        assets = tuple(
            item
            for item in self.repository.list_assets()
            if item.deletion_status is not MediaDeletionStatus.PENDING_BINDING
        )
        by_id = {item.asset_id: item for item in assets}
        selected: set[UUID] = set()
        if scope.all_user_media:
            selected.update(
                item.asset_id for item in assets if user_id in item.owner_scope.subject_user_ids
            )
        else:
            for session_id in scope.session_ids:
                session_assets = [item for item in assets if item.session_id == session_id]
                if not session_assets:
                    raise RetentionNotFoundError("deletion scope was not found")
                if any(user_id not in item.owner_scope.subject_user_ids for item in session_assets):
                    raise RetentionOwnershipError("deletion scope is not wholly owned")
                selected.update(item.asset_id for item in session_assets)
            for asset_id in scope.asset_ids:
                item = by_id.get(asset_id)
                if item is None:
                    raise RetentionNotFoundError("deletion scope was not found")
                if user_id not in item.owner_scope.subject_user_ids:
                    raise RetentionOwnershipError("deletion scope is not wholly owned")
                selected.add(asset_id)
        if not selected and not scope.all_user_media:
            raise RetentionNotFoundError("deletion scope contains no registered media")
        changed = True
        while changed:
            changed = False
            for item in assets:
                if item.asset_id not in selected and selected.intersection(item.source_asset_ids):
                    if user_id not in item.owner_scope.subject_user_ids:
                        raise RetentionOwnershipError("derived deletion scope is not wholly owned")
                    selected.add(item.asset_id)
                    changed = True
        return tuple(sorted(selected, key=str))

    def submit_deletion(
        self, *, current_user_id: str, scope: DeletionScope, idempotency_key: str
    ) -> DeletionRequest:
        if not current_user_id.strip() or not idempotency_key.strip():
            raise ValueError("authenticated user and idempotency key are required")
        frozen = self._resolve_scope(current_user_id, scope)
        now = self._now().astimezone(UTC)
        if not frozen:
            request = DeletionRequest(
                request_id=uuid4(),
                user_id=current_user_id,
                scope=scope,
                idempotency_key=idempotency_key,
                status=DeletionRequestStatus.COMPLETED,
                frozen_asset_ids=(),
                created_at=now,
                updated_at=now,
                completed_at=now,
                completion_evidence_summary="verified 0/0 registered media objects absent",
            )
            return self.repository.create_request(request)
        request = DeletionRequest(
            request_id=uuid4(),
            user_id=current_user_id,
            scope=scope,
            idempotency_key=idempotency_key,
            status=DeletionRequestStatus.RECEIVED,
            frozen_asset_ids=frozen,
            created_at=now,
            updated_at=now,
        )
        return self.repository.create_request(request)

    def get_public_status(self, *, current_user_id: str, request_id: UUID) -> PublicDeletionStatus:
        request = self.repository.get_request(request_id)
        if request.user_id != current_user_id:
            # Deliberately indistinguishable from not found to prevent enumeration.
            raise RetentionNotFoundError("deletion request was not found")
        return PublicDeletionStatus.from_request(request)

    def _resolve_authorized_media_for_access(
        self,
        *,
        asset_id: UUID,
        gate_revision_id: UUID,
        asset_context: AssetContext,
        requested_purpose: MaterialUsePurpose,
    ) -> str:
        """Internal primitive used only after the Story 1.3 access checks."""

        if self.gate_repository is None:
            raise RetentionRepositoryError("a trusted gate repository is required")
        asset, locator = self.repository._get_active_asset_with_locator(asset_id)
        self._require_session_media_bound(asset)
        try:
            self._validate_gate_asset_binding(asset, asset_context)
        except RetentionConflictError as exc:
            raise MediaUnavailableError("media context does not match the requested asset") from exc
        decision = authorize_real_video_use(
            self.gate_repository, gate_revision_id, asset_context, requested_purpose
        )
        if decision.decision is not MaterialUseDecision.ALLOW:
            raise MediaUnavailableError("media use was not authorized")
        return locator

    def _require_session_media_bound(self, asset: RetainedMediaAsset) -> None:
        if self.session_media_authority is None:
            return
        try:
            bound = self.session_media_authority.is_media_bound(
                session_id=asset.session_id, asset_id=asset.asset_id
            )
        except Exception as exc:
            raise MediaUnavailableError("session media authority is unavailable") from exc
        if not bound:
            raise MediaUnavailableError("session media is not bound")

    def execute_deletion(self, *, request_id: UUID, executor_id: str) -> DeletionRequest:
        if self.media_store is None:
            raise RetentionRepositoryError("a controlled media store is required")
        request = self.repository.get_request(request_id)
        if request.status not in {
            DeletionRequestStatus.RECEIVED,
            DeletionRequestStatus.FAILED,
        }:
            raise RetentionConflictError("deletion request cannot be executed from this state")
        started = self._now().astimezone(UTC)
        in_progress = request.model_copy(
            update={
                "status": DeletionRequestStatus.IN_PROGRESS,
                "updated_at": started,
                "failure_reason_code": None,
                "retryable": False,
                "events": request.events
                + (
                    DeletionEvent(
                        event_number=len(request.events) + 1,
                        event_type=DeletionEventType.STARTED,
                        executor_id=executor_id,
                        occurred_at=started,
                        evidence_summary="deletion execution claimed by operator",
                    ),
                ),
            }
        )
        in_progress = DeletionRequest.model_validate(in_progress.model_dump())
        in_progress = self.repository.save_request(in_progress)
        request = in_progress
        failure_code = DeletionFailureCode.INVALID_STATE
        assets: list[RetainedMediaAsset] = []
        preexisting_absent_count = 0
        verified_deleted_count = 0
        try:
            expanded_ids = tuple(
                sorted(
                    set(in_progress.frozen_asset_ids).union(
                        self._resolve_scope(in_progress.user_id, in_progress.scope)
                    ),
                    key=str,
                )
            )
            in_progress = self.repository.expand_in_progress_scope(
                in_progress.request_id,
                expected_frozen_asset_ids=in_progress.frozen_asset_ids,
                expanded_frozen_asset_ids=expanded_ids,
                updated_at=started,
            )
            request = in_progress
            failure_code = DeletionFailureCode.STORAGE_DELETE_FAILED
            assets = [self.repository.get_asset(item) for item in request.frozen_asset_ids]
            for asset in assets:
                if asset.deletion_status is not MediaDeletionStatus.DELETED:
                    pending = RetainedMediaAsset.model_validate(
                        asset.model_copy(
                            update={
                                "deletion_status": MediaDeletionStatus.PENDING_DELETION,
                                "updated_at": started,
                            }
                        ).model_dump()
                    )
                    self.repository.save_asset(pending, expected_current=asset)
                    if pending.storage_locator is not None:
                        if self.media_store.exists(pending.storage_locator):
                            self.media_store.delete(pending.storage_locator)
                        else:
                            preexisting_absent_count += 1
                else:
                    preexisting_absent_count += 1
            failure_code = DeletionFailureCode.VERIFICATION_FAILED
            for asset_id in request.frozen_asset_ids:
                current = self.repository.get_asset(asset_id)
                if current.storage_locator is not None and self.media_store.exists(
                    current.storage_locator
                ):
                    raise RetentionRepositoryError("media verification found an existing object")
                verified_deleted_count += 1
            failure_code = DeletionFailureCode.REFERENCE_SCRUB_FAILED
            self.repository.scrub_media_references(request.frozen_asset_ids)
            finished = self._now().astimezone(UTC)
            failure_code = DeletionFailureCode.STORE_UNAVAILABLE
            for asset_id in request.frozen_asset_ids:
                current = self.repository.get_asset(asset_id)
                deleted = RetainedMediaAsset.model_validate(
                    current.model_copy(
                        update={
                            "deletion_status": MediaDeletionStatus.DELETED,
                            "storage_locator": None,
                            "updated_at": finished,
                        }
                    ).model_dump()
                )
                self.repository.save_asset(deleted, expected_current=current)
            failure_code = DeletionFailureCode.VERIFICATION_FAILED
            for asset_id in request.frozen_asset_ids:
                deleted = self.repository.get_asset(asset_id)
                if deleted.storage_locator is not None:
                    raise RetentionRepositoryError("deleted asset retained a locator")
            for candidate in self.repository.list_assets():
                if (
                    set(candidate.source_asset_ids).intersection(request.frozen_asset_ids)
                    and candidate.deletion_status is not MediaDeletionStatus.DELETED
                ):
                    raise RetentionRepositoryError("a late derived media asset remains available")
            evidence = (
                f"verified={verified_deleted_count}/{len(request.frozen_asset_ids)}; "
                f"preexisting_absent={preexisting_absent_count}"
            )
            attempt = DeletionAttempt(
                attempt_number=len(request.attempts) + 1,
                executor_id=executor_id,
                started_at=started,
                finished_at=finished,
                planned_asset_count=len(request.frozen_asset_ids),
                verified_deleted_count=verified_deleted_count,
                result="COMPLETED",
                evidence_summary=evidence,
            )
            completed = in_progress.model_copy(
                update={
                    "status": DeletionRequestStatus.COMPLETED,
                    "updated_at": finished,
                    "completed_at": finished,
                    "completion_evidence_summary": evidence,
                    "attempts": request.attempts + (attempt,),
                }
            )
            completed = DeletionRequest.model_validate(completed.model_dump())
            return self.repository.finalize_completed_request(completed)
        except BaseException as exc:
            finished = self._now().astimezone(UTC)
            for asset_id in request.frozen_asset_ids:
                try:
                    current = self.repository.get_asset(asset_id)
                    if current.deletion_status is not MediaDeletionStatus.DELETED:
                        failed_asset = RetainedMediaAsset.model_validate(
                            current.model_copy(
                                update={
                                    "deletion_status": MediaDeletionStatus.DELETION_FAILED,
                                    "updated_at": finished,
                                }
                            ).model_dump()
                        )
                        self.repository.save_asset(failed_asset, expected_current=current)
                except RetentionError:
                    pass
            attempt = DeletionAttempt(
                attempt_number=len(request.attempts) + 1,
                executor_id=executor_id,
                started_at=started,
                finished_at=finished,
                planned_asset_count=len(request.frozen_asset_ids),
                verified_deleted_count=verified_deleted_count,
                result="FAILED",
                failure_code=failure_code,
                retryable=True,
                evidence_summary=f"attempt failed; operator evidence type={type(exc).__name__}",
            )
            failed = in_progress.model_copy(
                update={
                    "status": DeletionRequestStatus.FAILED,
                    "updated_at": finished,
                    "failure_reason_code": failure_code,
                    "retryable": True,
                    "attempts": request.attempts + (attempt,),
                }
            )
            failed = DeletionRequest.model_validate(failed.model_dump())
            saved_failed = self.repository.save_request(failed)
            if not isinstance(exc, Exception):
                raise
            return saved_failed

    def reclaim_in_progress(
        self, *, request_id: UUID, executor_id: str, reason: str
    ) -> DeletionRequest:
        """Force a stranded IN_PROGRESS request to retryable FAILED after operator checks."""

        if not executor_id.strip() or not reason.strip():
            raise ValueError("executor and reclaim reason are required")
        request = self.repository.get_request(request_id)
        if request.status is not DeletionRequestStatus.IN_PROGRESS:
            raise RetentionConflictError("only an in-progress request can be reclaimed")
        now = self._now().astimezone(UTC)
        attempt = DeletionAttempt(
            attempt_number=len(request.attempts) + 1,
            executor_id=executor_id,
            started_at=request.updated_at,
            finished_at=now,
            planned_asset_count=len(request.frozen_asset_ids),
            verified_deleted_count=0,
            result="FAILED",
            failure_code=DeletionFailureCode.EXECUTION_INTERRUPTED,
            retryable=True,
            evidence_summary=f"stranded execution reclaimed: {reason}",
        )
        event = DeletionEvent(
            event_number=len(request.events) + 1,
            event_type=DeletionEventType.RECLAIMED,
            executor_id=executor_id,
            occurred_at=now,
            evidence_summary=f"operator reclaim: {reason}",
        )
        failed = DeletionRequest.model_validate(
            request.model_copy(
                update={
                    "status": DeletionRequestStatus.FAILED,
                    "updated_at": now,
                    "failure_reason_code": DeletionFailureCode.EXECUTION_INTERRUPTED,
                    "retryable": True,
                    "attempts": request.attempts + (attempt,),
                    "events": request.events + (event,),
                }
            ).model_dump()
        )
        return self.repository.save_request(failed)
