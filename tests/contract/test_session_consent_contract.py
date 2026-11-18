"""Frozen public contracts for Story 6.2 session consent."""

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    CloudSession,
    CloudSessionStatus,
    ConsentDecision,
    ConsentDecisionRecord,
    ConsentNotice,
    ConsentPolicyView,
    ConsentSubmissionRequest,
    SessionAdmissionErrorCode,
    SessionAdmissionNextAction,
    SessionAdmissionView,
    SessionParticipant,
    TrainingConsentWithdrawalRequest,
)
from billiards_ai.shared.models.privacy import MaterialUsePurpose

NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)
IDS = [uuid4() for _ in range(10)]
DIGEST = "a" * 64
KEY = "A" * 43


def _notice(purpose: MaterialUsePurpose) -> ConsentNotice:
    return ConsentNotice(
        purpose=purpose,
        version="synthetic-v1",
        content="Synthetic controlled notice",
        content_sha256=hashlib.sha256(b"Synthetic controlled notice").hexdigest(),
        scope_fingerprint=DIGEST,
        gate_revision_id=IDS[0],
        canonicalization_schema_version="raw-utf8-v1",
        collection_scope="Synthetic table-only clips",
        use_description="Synthetic analysis only",
        retention_description="Synthetic retention policy",
        access_description="Synthetic least-privilege access",
        deletion_description="Synthetic deletion route",
        operator_name="Synthetic operator",
        operator_contact="privacy@example.invalid",
    )


def _request() -> ConsentSubmissionRequest:
    return ConsentSubmissionRequest(
        schema_version="1.0",
        binding_id=IDS[1],
        service_decision=ConsentDecision.GRANTED,
        training_decision=ConsentDecision.DECLINED,
        policy_id=DIGEST,
        gate_revision_id=IDS[0],
        service_notice_version="synthetic-v1",
        training_notice_version="synthetic-v1",
        idempotency_key=KEY,
    )


def test_request_is_strict_minimal_and_rejects_inconsistent_decisions() -> None:
    assert set(_request().model_dump()) == {
        "schema_version",
        "binding_id",
        "service_decision",
        "training_decision",
        "policy_id",
        "gate_revision_id",
        "service_notice_version",
        "training_notice_version",
        "idempotency_key",
    }
    for forbidden in (
        "user_id",
        "openid",
        "session_id",
        "venue_id",
        "table_id",
        "device_id",
        "qr_token",
        "jwt",
        "code",
        "time",
        "hash",
        "upload",
    ):
        with pytest.raises(ValidationError):
            ConsentSubmissionRequest.model_validate(
                {**_request().model_dump(mode="json"), forbidden: "synthetic"}
            )
    with pytest.raises(ValidationError):
        ConsentSubmissionRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                "service_decision": "DECLINED",
                "training_decision": "GRANTED",
            }
        )
    with pytest.raises(ValidationError):
        ConsentSubmissionRequest.model_validate(
            {**_request().model_dump(mode="json"), "service_decision": True}
        )


def test_notice_policy_and_owner_view_do_not_expose_internal_authority() -> None:
    policy = ConsentPolicyView(
        binding_id=IDS[1],
        policy_id=DIGEST,
        gate_revision_id=IDS[0],
        scope_fingerprint=DIGEST,
        canonicalization_schema_version="raw-utf8-v1",
        service_notice=_notice(MaterialUsePurpose.SERVICE_ANALYSIS),
        training_notice=_notice(MaterialUsePurpose.TRAINING),
    )
    assert policy.service_notice.content == "Synthetic controlled notice"
    view = SessionAdmissionView(
        binding_id=IDS[1],
        session_id=IDS[2],
        participant_id=IDS[3],
        venue_display_label="Synthetic venue",
        table_display_label="Table 1",
        service_decision=ConsentDecision.GRANTED,
        training_decision=ConsentDecision.DECLINED,
        service_notice_version="synthetic-v1",
        training_notice_version="synthetic-v1",
        session_status=CloudSessionStatus.OPEN,
        created_at=NOW,
        activated_at=NOW,
        next_action=SessionAdmissionNextAction.WAITING_FOR_UPLOAD,
    )
    dumped = view.model_dump(mode="json")
    assert "user_id" not in dumped
    assert "scope_fingerprint" not in dumped
    assert "content_sha256" not in dumped
    assert "gate_revision_id" not in dumped


def test_append_only_records_require_uuid4_aware_time_and_complete_grant_links() -> None:
    record = ConsentDecisionRecord(
        consent_id=IDS[4],
        user_id=IDS[5],
        binding_id=IDS[1],
        session_id=IDS[2],
        participant_id=IDS[3],
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        decision=ConsentDecision.GRANTED,
        notice_version="synthetic-v1",
        notice_sha256=DIGEST,
        gate_revision_id=IDS[0],
        gate_scope_fingerprint=DIGEST,
        decided_at=NOW,
    )
    assert record.decision is ConsentDecision.GRANTED
    for bad in (
        {"consent_id": "00000000-0000-1000-8000-000000000000"},
        {"decided_at": "2026-07-29T12:00:00"},
        {"session_id": None},
        {"participant_id": None},
    ):
        with pytest.raises(ValidationError):
            ConsentDecisionRecord.model_validate({**record.model_dump(mode="json"), **bad})
    with pytest.raises(ValidationError):
        ConsentDecisionRecord.model_validate(record.model_construct().model_dump(mode="python"))


def test_session_participant_and_withdrawal_contracts_are_separate() -> None:
    session = CloudSession(
        session_id=IDS[2],
        table_registration_id=IDS[6],
        status=CloudSessionStatus.OPEN,
        created_at=NOW,
    )
    participant = SessionParticipant(
        participant_id=IDS[3],
        session_id=session.session_id,
        user_id=IDS[5],
        binding_id=IDS[1],
        service_consent_id=IDS[4],
        training_consent_id=IDS[7],
        joined_at=NOW,
    )
    assert "user_id" not in CloudSession.model_fields
    assert participant.session_id == session.session_id
    TrainingConsentWithdrawalRequest(
        participant_id=participant.participant_id,
        purpose=MaterialUsePurpose.TRAINING,
        idempotency_key=KEY,
    )
    with pytest.raises(ValidationError):
        TrainingConsentWithdrawalRequest(
            participant_id=participant.participant_id,
            purpose=MaterialUsePurpose.ANNOTATION,
            idempotency_key=KEY,
        )


def test_closed_error_codes_cover_the_state_matrix() -> None:
    assert {item.value for item in SessionAdmissionErrorCode} >= {
        "IDENTITY_REQUIRED",
        "RESOURCE_NOT_FOUND",
        "CONSENT_POLICY_CHANGED",
        "CONSENT_SCOPE_MISMATCH",
        "REAL_VIDEO_GATE_NOT_APPROVED",
        "TABLE_CAPACITY_REACHED",
        "IDEMPOTENCY_CONFLICT",
        "SESSION_ADMISSION_REPOSITORY_UNAVAILABLE",
    }
