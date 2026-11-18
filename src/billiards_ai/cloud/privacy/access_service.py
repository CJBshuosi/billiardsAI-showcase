"""Single auditable entrypoint for controlled human and third-party media access."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, TypeVar
from uuid import UUID, uuid4

from billiards_ai.shared.models import (
    AccessAction,
    AccessActorRole,
    AccessAuditEvent,
    AccessDecision,
    AccessDecisionResult,
    AccessReasonCode,
    AssetContext,
    MaterialUsePurpose,
    MediaDeletionStatus,
    ReviewerIdentity,
    SessionAccessGrant,
    ThirdPartyProcessingApproval,
)

from .access_repository import AccessRepositoryError, JsonAccessRepository
from .retention_repository import MediaUnavailableError, RetentionError
from .retention_service import MediaRetentionService

T = TypeVar("T")


class ReviewerIdentityProvider(Protocol):
    def current_identity(self) -> ReviewerIdentity: ...


class AccessDeniedError(RuntimeError):
    """A controlled access decision denied release of media."""

    def __init__(self, decision: AccessDecision):
        super().__init__(decision.reason_code.value)
        self.decision = decision


def _role_for_event(identity: ReviewerIdentity) -> AccessActorRole:
    if AccessActorRole.REVIEWER in identity.roles:
        return AccessActorRole.REVIEWER
    if AccessActorRole.ACCESS_ADMIN in identity.roles:
        return AccessActorRole.ACCESS_ADMIN
    return AccessActorRole.SYSTEM


def _valid_reason(reason: str) -> str | None:
    normalized = reason.strip()
    if not normalized or any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        return None
    return normalized


class MediaAccessService:
    def __init__(
        self,
        *,
        access_repository: JsonAccessRepository,
        retention_service: MediaRetentionService,
        identity_provider: ReviewerIdentityProvider,
        now: Callable[[], datetime] | None = None,
    ):
        self.access_repository = access_repository
        self.retention_service = retention_service
        self.identity_provider = identity_provider
        self._now = now or (lambda: datetime.now(UTC))

    def _identity(self) -> ReviewerIdentity:
        try:
            return self.identity_provider.current_identity()
        except (AccessRepositoryError, OSError, ValueError) as exc:
            raise AccessRepositoryError("trusted reviewer identity is unavailable") from exc

    def grant_current_reviewer(self, *, session_id: UUID, reason: str) -> SessionAccessGrant:
        identity = self._identity()
        if AccessActorRole.ACCESS_ADMIN not in identity.roles:
            decision = AccessDecision(
                event_id=uuid4(),
                session_id=session_id,
                action=AccessAction.CONTROLLED_REVIEW_VIEW,
                result=AccessDecisionResult.DENY,
                reason_code=AccessReasonCode.ROLE_NOT_ALLOWED,
            )
            raise AccessDeniedError(decision)
        grant = SessionAccessGrant(
            grant_id=uuid4(),
            reviewer_id=identity.principal_id,
            session_id=session_id,
            reason=reason,
            granted_by=identity.principal_id,
            granted_at=self._now().astimezone(UTC),
        )
        return self.access_repository.create_grant(grant)

    def revoke_grant(self, *, grant_id: UUID, reason: str) -> SessionAccessGrant:
        identity = self._identity()
        if AccessActorRole.ACCESS_ADMIN not in identity.roles:
            raise AccessDeniedError(
                AccessDecision(
                    event_id=uuid4(),
                    session_id=uuid4(),
                    action=AccessAction.CONTROLLED_REVIEW_VIEW,
                    result=AccessDecisionResult.DENY,
                    reason_code=AccessReasonCode.ROLE_NOT_ALLOWED,
                )
            )
        return self.access_repository.revoke_grant(
            grant_id,
            revoked_at=self._now().astimezone(UTC),
            revocation_reason=reason,
        )

    def check_current_reviewer_grant(self, *, session_id: UUID) -> SessionAccessGrant | None:
        identity = self._identity()
        return self.access_repository.find_active_grant(identity.principal_id, session_id)

    def list_audit_events(self, *, session_id: UUID | None = None) -> tuple[AccessAuditEvent, ...]:
        identity = self._identity()
        if not set(identity.roles).intersection(
            {AccessActorRole.REVIEWER, AccessActorRole.ACCESS_ADMIN}
        ):
            raise AccessDeniedError(
                AccessDecision(
                    event_id=uuid4(),
                    session_id=session_id or uuid4(),
                    action=AccessAction.CONTROLLED_REVIEW_VIEW,
                    result=AccessDecisionResult.DENY,
                    reason_code=AccessReasonCode.ROLE_NOT_ALLOWED,
                )
            )
        return self.access_repository.list_events(session_id=session_id)

    def approve_third_party_processing(
        self,
        *,
        provider_id: str,
        endpoint_region: str,
        purpose: MaterialUsePurpose,
        cross_border: bool,
        cross_border_assessment: str,
        max_cost_cny: Decimal,
        approval_evidence_sha256: str,
    ) -> ThirdPartyProcessingApproval:
        identity = self._identity()
        if AccessActorRole.ACCESS_ADMIN not in identity.roles:
            raise AccessDeniedError(
                AccessDecision(
                    event_id=uuid4(),
                    session_id=uuid4(),
                    action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                    result=AccessDecisionResult.DENY,
                    reason_code=AccessReasonCode.ROLE_NOT_ALLOWED,
                )
            )
        approval = ThirdPartyProcessingApproval(
            approval_id=uuid4(),
            provider_id=provider_id,
            endpoint_region=endpoint_region,
            data_scope="KEYFRAME",
            purpose=purpose,
            cross_border=cross_border,
            cross_border_assessment=cross_border_assessment,
            max_cost_cny=max_cost_cny,
            approved_by=identity.principal_id,
            approved_at=self._now().astimezone(UTC),
            approval_evidence_sha256=approval_evidence_sha256,
        )
        return self.access_repository.create_provider_approval(approval)

    def revoke_third_party_approval(
        self, *, approval_id: UUID, reason: str
    ) -> ThirdPartyProcessingApproval:
        identity = self._identity()
        if AccessActorRole.ACCESS_ADMIN not in identity.roles:
            raise AccessDeniedError(
                AccessDecision(
                    event_id=uuid4(),
                    session_id=uuid4(),
                    action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                    result=AccessDecisionResult.DENY,
                    reason_code=AccessReasonCode.ROLE_NOT_ALLOWED,
                )
            )
        return self.access_repository.revoke_provider_approval(
            approval_id,
            revoked_at=self._now().astimezone(UTC),
            revocation_reason=reason,
        )

    def _record_decision(
        self,
        *,
        identity: ReviewerIdentity,
        session_id: UUID,
        action: AccessAction,
        reason: str,
        result: AccessDecisionResult,
        reason_code: AccessReasonCode,
    ) -> AccessDecision:
        event = AccessAuditEvent(
            event_id=uuid4(),
            sequence=1,
            actor_id=identity.principal_id,
            actor_role=_role_for_event(identity),
            occurred_at=self._now().astimezone(UTC),
            session_id=session_id,
            action=action,
            reason=reason,
            result=result,
            reason_code=reason_code,
            security_event=result is AccessDecisionResult.DENY,
        )
        stored = self.access_repository.append_event(event)
        return AccessDecision(
            event_id=stored.event_id,
            session_id=stored.session_id,
            action=stored.action,
            result=stored.result,
            reason_code=stored.reason_code,
        )

    def _deny(
        self,
        *,
        identity: ReviewerIdentity,
        session_id: UUID,
        action: AccessAction,
        reason: str,
        reason_code: AccessReasonCode,
    ) -> None:
        decision = self._record_decision(
            identity=identity,
            session_id=session_id,
            action=action,
            reason=reason,
            result=AccessDecisionResult.DENY,
            reason_code=reason_code,
        )
        raise AccessDeniedError(decision)

    def _identity_or_audited_deny(
        self, *, session_id: UUID, action: AccessAction, reason: str
    ) -> ReviewerIdentity:
        try:
            return self._identity()
        except AccessRepositoryError:
            unavailable = ReviewerIdentity(
                principal_id="identity-unavailable",
                roles=(AccessActorRole.SYSTEM,),
                credential_id="server-generated-denial",
            )
            self._deny(
                identity=unavailable,
                session_id=session_id,
                action=action,
                reason=_valid_reason(reason) or "access reason omitted or invalid",
                reason_code=AccessReasonCode.IDENTITY_UNAVAILABLE,
            )

    def authorize_and_record_access(
        self,
        *,
        asset_id: UUID,
        gate_revision_id: UUID,
        asset_context: AssetContext,
        reason: str,
        action: AccessAction,
        consumer: Callable[[str], T],
    ) -> T:
        try:
            asset = self.retention_service.repository.get_asset(asset_id)
        except RetentionError as exc:
            raise AccessRepositoryError("authoritative media inventory is unavailable") from exc
        identity = self._identity_or_audited_deny(
            session_id=asset.session_id, action=action, reason=reason
        )
        audit_reason = _valid_reason(reason)
        if audit_reason is None:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason="access reason omitted or invalid",
                reason_code=AccessReasonCode.INVALID_REASON,
            )
        if AccessActorRole.REVIEWER not in identity.roles:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason=audit_reason,
                reason_code=AccessReasonCode.ROLE_NOT_ALLOWED,
            )
        if action is AccessAction.EXPORT_OR_DOWNLOAD:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason=audit_reason,
                reason_code=AccessReasonCode.ACTION_DENIED,
            )
        if action is not AccessAction.CONTROLLED_REVIEW_VIEW:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason=audit_reason,
                reason_code=AccessReasonCode.ACTION_DENIED,
            )
        grant = self.access_repository.find_active_grant(identity.principal_id, asset.session_id)
        if grant is None:
            revoked = any(
                item.reviewer_id == identity.principal_id
                and item.session_id == asset.session_id
                and not item.is_active
                for item in self.access_repository.list_grants()
            )
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason=audit_reason,
                reason_code=(
                    AccessReasonCode.GRANT_REVOKED if revoked else AccessReasonCode.GRANT_NOT_FOUND
                ),
            )
        if asset.deletion_status is not MediaDeletionStatus.ACTIVE:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason=audit_reason,
                reason_code=AccessReasonCode.MEDIA_UNAVAILABLE,
            )
        try:
            locator = self.retention_service._resolve_authorized_media_for_access(
                asset_id=asset_id,
                gate_revision_id=gate_revision_id,
                asset_context=asset_context,
                requested_purpose=MaterialUsePurpose.NECESSARY_REVIEW,
            )
        except MediaUnavailableError:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=action,
                reason=audit_reason,
                reason_code=AccessReasonCode.GATE_DENIED,
            )
        self._record_decision(
            identity=identity,
            session_id=asset.session_id,
            action=action,
            reason=audit_reason,
            result=AccessDecisionResult.ALLOW,
            reason_code=AccessReasonCode.AUTHORIZED,
        )
        return consumer(locator)

    def authorize_third_party_keyframe_transfer(
        self,
        *,
        asset_id: UUID,
        gate_revision_id: UUID,
        asset_context: AssetContext,
        reason: str,
        provider_id: str,
        endpoint_region: str,
        purpose: MaterialUsePurpose,
        cross_border: bool,
        estimated_cost_cny: Decimal,
    ) -> AccessDecision:
        try:
            asset = self.retention_service.repository.get_asset(asset_id)
        except RetentionError as exc:
            raise AccessRepositoryError("authoritative media inventory is unavailable") from exc
        identity = self._identity_or_audited_deny(
            session_id=asset.session_id,
            action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
            reason=reason,
        )
        audit_reason = _valid_reason(reason)
        if audit_reason is None:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason="access reason omitted or invalid",
                reason_code=AccessReasonCode.INVALID_REASON,
            )
        if asset.deletion_status is not MediaDeletionStatus.ACTIVE:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason=audit_reason,
                reason_code=AccessReasonCode.MEDIA_UNAVAILABLE,
            )
        if asset.kind.value != "KEYFRAME":
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason=audit_reason,
                reason_code=AccessReasonCode.THIRD_PARTY_APPROVAL_MISMATCH,
            )
        approval = self.access_repository.find_active_provider_approval(provider_id)
        if approval is None:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason=audit_reason,
                reason_code=AccessReasonCode.THIRD_PARTY_APPROVAL_REQUIRED,
            )
        if (
            approval.endpoint_region != endpoint_region
            or approval.purpose is not purpose
            or approval.cross_border is not cross_border
        ):
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason=audit_reason,
                reason_code=AccessReasonCode.THIRD_PARTY_APPROVAL_MISMATCH,
            )
        if estimated_cost_cny > approval.max_cost_cny:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason=audit_reason,
                reason_code=AccessReasonCode.COST_LIMIT_EXCEEDED,
            )
        try:
            self.retention_service._resolve_authorized_media_for_access(
                asset_id=asset_id,
                gate_revision_id=gate_revision_id,
                asset_context=asset_context,
                requested_purpose=purpose,
            )
        except MediaUnavailableError:
            self._deny(
                identity=identity,
                session_id=asset.session_id,
                action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
                reason=audit_reason,
                reason_code=AccessReasonCode.GATE_DENIED,
            )
        return self._record_decision(
            identity=identity,
            session_id=asset.session_id,
            action=AccessAction.THIRD_PARTY_KEYFRAME_TRANSFER,
            reason=audit_reason,
            result=AccessDecisionResult.ALLOW,
            reason_code=AccessReasonCode.AUTHORIZED,
        )
