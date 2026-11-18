from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.access import (
    AccessAction,
    AccessActorRole,
    AccessAuditEvent,
    AccessDecisionResult,
    AccessReasonCode,
    SessionAccessGrant,
    ThirdPartyProcessingApproval,
)
from billiards_ai.shared.models.privacy import MaterialUsePurpose
from billiards_ai.shared.models.retention import MediaAssetKind

NOW = datetime(2026, 7, 28, 16, 0, tzinfo=UTC)


def test_access_enums_are_exact() -> None:
    assert {item.value for item in AccessAction} == {
        "CONTROLLED_REVIEW_VIEW",
        "EXPORT_OR_DOWNLOAD",
        "THIRD_PARTY_KEYFRAME_TRANSFER",
    }
    assert {item.value for item in AccessDecisionResult} == {"ALLOW", "DENY"}
    assert {item.value for item in AccessActorRole} == {
        "REVIEWER",
        "ACCESS_ADMIN",
        "SYSTEM",
    }


def test_session_grant_is_frozen_single_session_and_strict() -> None:
    grant = SessionAccessGrant(
        grant_id=uuid4(),
        reviewer_id="reviewer-1",
        session_id=uuid4(),
        reason="  review low-confidence shots  ",
        granted_by="admin-1",
        granted_at=NOW,
    )

    assert grant.reason == "review low-confidence shots"
    assert isinstance(grant.grant_id, UUID)
    assert grant.revoked_at is None
    assert "locator" not in grant.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SessionAccessGrant(**grant.model_dump(), wildcard=True)
    with pytest.raises(ValidationError):
        grant.reason = "changed"


@pytest.mark.parametrize("reason", ["", "   ", "review\x00session"])
def test_session_grant_rejects_empty_or_control_character_reason(reason: str) -> None:
    with pytest.raises(ValidationError):
        SessionAccessGrant(
            grant_id=uuid4(),
            reviewer_id="reviewer-1",
            session_id=uuid4(),
            reason=reason,
            granted_by="admin-1",
            granted_at=NOW,
        )


def test_audit_event_contains_only_minimal_non_media_fields() -> None:
    event = AccessAuditEvent(
        event_id=uuid4(),
        sequence=1,
        actor_id="reviewer-1",
        actor_role=AccessActorRole.REVIEWER,
        occurred_at=NOW,
        session_id=uuid4(),
        action=AccessAction.CONTROLLED_REVIEW_VIEW,
        reason="manual review",
        result=AccessDecisionResult.ALLOW,
        reason_code=AccessReasonCode.AUTHORIZED,
        security_event=False,
    )

    payload = event.model_dump(mode="json")
    assert set(payload) == {
        "schema_version",
        "event_id",
        "sequence",
        "actor_id",
        "actor_role",
        "occurred_at",
        "session_id",
        "action",
        "reason",
        "result",
        "reason_code",
        "security_event",
    }
    serialized = event.model_dump_json().lower()
    for forbidden in ("locator", "url", "sha256", "credential", "evidence_path"):
        assert forbidden not in serialized


def test_third_party_approval_is_keyframe_scoped_and_contains_no_secret() -> None:
    approval = ThirdPartyProcessingApproval(
        approval_id=uuid4(),
        provider_id="provider-cn-1",
        endpoint_region="cn-shanghai",
        data_scope=MediaAssetKind.KEYFRAME,
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        cross_border=False,
        cross_border_assessment="DOMESTIC_ONLY",
        max_cost_cny="2.50",
        approved_by="admin-1",
        approved_at=NOW,
        approval_evidence_sha256="a" * 64,
    )

    assert approval.data_scope is MediaAssetKind.KEYFRAME
    assert "api_key" not in approval.model_dump(mode="json")
    with pytest.raises(ValidationError):
        ThirdPartyProcessingApproval(
            **{**approval.model_dump(), "data_scope": MediaAssetKind.VIDEO}
        )


def test_models_reject_naive_datetimes_and_non_uuid4() -> None:
    with pytest.raises(ValidationError):
        SessionAccessGrant(
            grant_id=UUID("00000000-0000-1000-8000-000000000000"),
            reviewer_id="reviewer-1",
            session_id=uuid4(),
            reason="review",
            granted_by="admin-1",
            granted_at=NOW.replace(tzinfo=None),
        )
