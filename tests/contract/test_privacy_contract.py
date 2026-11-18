"""Shared domain contracts for Story 1.1 privacy controls."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from billiards_ai.cloud import privacy
from billiards_ai.shared.models import (
    BystanderDisposition,
    EvidenceReference,
    GateChecklist,
    RealVideoGateStatus,
    SubjectClassification,
)


def test_privacy_package_exposes_stable_public_api():
    assert callable(privacy.authorize_real_video_use)
    assert privacy.RealVideoGateService is not None
    assert privacy.JsonGateRepository is not None


def test_gate_defaults_to_blocked_and_uses_uuid4(
    tmp_path, gate_service_factory, gate_scope_factory, gate_evidence, now
):
    gate_service = gate_service_factory(tmp_path)
    revision = gate_service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    assert revision.status is RealVideoGateStatus.BLOCKED
    assert revision.gate_id.version == 4
    assert revision.revision_id.version == 4


def test_models_reject_naive_datetimes_and_extra_fields(complete_checklist_factory):
    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_ref="controlled://evidence",
            sha256="3" * 64,
            captured_at=datetime(2026, 7, 27, 12, 0),
        )
    with pytest.raises(ValidationError):
        complete_checklist_factory(unexpected=True)


@pytest.mark.parametrize(
    "captured_at",
    [
        datetime.max.replace(tzinfo=timezone(-timedelta(hours=12))),
        datetime.min.replace(tzinfo=timezone(timedelta(hours=12))),
    ],
)
def test_datetime_timezone_extremes_are_validation_errors(captured_at):
    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_ref="controlled://extreme",
            sha256="3" * 64,
            captured_at=captured_at,
        )


def test_datetime_max_comparisons_do_not_overflow(monkeypatch, gate_scope_factory, gate_evidence):
    from uuid import uuid4

    from billiards_ai.shared.models import GateRevision
    from billiards_ai.shared.models import privacy as privacy_models

    maximum = datetime.max.replace(tzinfo=UTC)
    monkeypatch.setattr(privacy_models, "_utc_now", lambda: maximum)
    revision = GateRevision(
        gate_id=uuid4(),
        revision_id=uuid4(),
        revision_number=1,
        scope=gate_scope_factory(),
        inspector_id="inspector-extreme",
        inspected_at=maximum,
        evidence=tuple(item.model_copy(update={"captured_at": maximum}) for item in gate_evidence),
        created_at=maximum,
    )
    assert revision.created_at == maximum


def test_checklist_rejects_coercible_values_and_identifiers_reject_whitespace(
    gate_scope_factory,
):
    with pytest.raises(ValidationError):
        GateChecklist(table_view_only="yes")
    with pytest.raises(ValidationError):
        gate_scope_factory(venue_id="   ")


def test_consented_subject_requires_versioned_evidence():
    with pytest.raises(ValidationError):
        SubjectClassification(
            asset_id="asset-1",
            segment_ref="segment-1",
            subject_ref="subject-1",
            disposition=BystanderDisposition.CONSENTED_FOR_SERVICE,
        )


def test_subject_disposition_values_are_exact():
    assert {item.value for item in BystanderDisposition} == {
        "NOT_IDENTIFIABLE",
        "ANONYMIZED",
        "CONSENTED_FOR_SERVICE",
        "CONSENTED_FOR_TRAINING",
        "UNCONSENTED_IDENTIFIABLE",
    }


def test_scope_fingerprint_changes_with_camera_or_consent(gate_scope_factory):
    original = gate_scope_factory()
    assert original.fingerprint != gate_scope_factory(camera_config_id="top-view-v2").fingerprint
    assert original.fingerprint != gate_scope_factory(service_consent_sha256="5" * 64).fingerprint
