"""Unit tests for bystander-purpose authorization policy."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.privacy import authorize_real_video_use
from billiards_ai.shared.models import (
    AnonymizationRecord,
    AssetContext,
    BystanderDisposition,
    MaterialUseDecision,
    MaterialUsePurpose,
    SourceAssetProvenance,
    SubjectClassification,
)
from billiards_ai.shared.models import privacy as privacy_models


@pytest.mark.parametrize(
    ("disposition", "purpose", "expected"),
    [
        (disposition, purpose, expected)
        for disposition, allowed in {
            BystanderDisposition.NOT_IDENTIFIABLE: set(MaterialUsePurpose),
            BystanderDisposition.ANONYMIZED: set(MaterialUsePurpose),
            BystanderDisposition.CONSENTED_FOR_SERVICE: {
                MaterialUsePurpose.SERVICE_ANALYSIS,
                MaterialUsePurpose.NECESSARY_REVIEW,
            },
            BystanderDisposition.CONSENTED_FOR_TRAINING: {
                MaterialUsePurpose.ANNOTATION,
                MaterialUsePurpose.TRAINING,
            },
            BystanderDisposition.UNCONSENTED_IDENTIFIABLE: set(),
        }.items()
        for purpose in MaterialUsePurpose
        for expected in [
            MaterialUseDecision.ALLOW
            if purpose in allowed
            else (
                MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT
                if disposition
                in {
                    BystanderDisposition.CONSENTED_FOR_SERVICE,
                    BystanderDisposition.CONSENTED_FOR_TRAINING,
                    BystanderDisposition.UNCONSENTED_IDENTIFIABLE,
                }
                else MaterialUseDecision.DENY
            )
        ]
    ],
)
def test_full_disposition_purpose_matrix(
    tmp_path, disposition, purpose, expected, approved_gate_factory, now
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    anonymization = ()
    source_provenance = None
    asset_id = "asset-1"
    asset_sha256 = "8" * 64
    if disposition is BystanderDisposition.ANONYMIZED:
        asset_id = "asset-1-anonymized"
        asset_sha256 = "5" * 64
        anonymization = (
            AnonymizationRecord(
                source_asset_id="asset-1",
                source_asset_sha256="8" * 64,
                derived_asset_id=asset_id,
                method="crop-and-blur",
                operator_id="privacy-operator",
                processed_at=now,
                output_sha256="5" * 64,
                visual_evidence_sha256="6" * 64,
                metadata_removal_evidence_sha256="7" * 64,
            ),
        )
        source_provenance = {
            "asset_id": "asset-1",
            "sha256": "8" * 64,
            "captured_at": now,
            "gate_revision_id": approved.revision_id,
            "gate_scope_fingerprint": approved.scope.fingerprint,
        }
    consent_fields = {}
    if disposition is BystanderDisposition.CONSENTED_FOR_SERVICE:
        consent_fields = {
            "consent_version": approved.scope.service_consent_version,
            "consent_evidence_sha256": "9" * 64,
        }
    elif disposition is BystanderDisposition.CONSENTED_FOR_TRAINING:
        consent_fields = {
            "consent_version": approved.scope.training_consent_version,
            "consent_evidence_sha256": "9" * 64,
        }
    context = AssetContext(
        asset_id=asset_id,
        asset_sha256=asset_sha256,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=True,
        classifications=(
            SubjectClassification(
                asset_id=asset_id,
                segment_ref="segment-1",
                subject_ref="subject-1",
                disposition=disposition,
                **consent_fields,
            ),
        ),
        anonymization_records=anonymization,
        source_asset_provenance=source_provenance,
    )
    result = authorize_real_video_use(
        gate_service.repository, approved.revision_id, context, purpose
    )
    assert result.decision is expected


def test_no_bystanders_is_not_automatic_rejection(tmp_path, approved_gate_factory, now):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )
    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        context,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.ALLOW


def test_asset_context_requires_capture_provenance(approved_gate_factory, tmp_path):
    _, _, approved = approved_gate_factory(tmp_path)
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1",
            asset_sha256="8" * 64,
            gate_scope_fingerprint=approved.scope.fingerprint,
            bystanders_present=False,
        )


def test_asset_capture_time_cannot_exceed_future_clock_skew(monkeypatch, now):
    monkeypatch.setattr(privacy_models, "_utc_now", lambda: now, raising=False)
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-future",
            asset_sha256="8" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now + timedelta(minutes=5, microseconds=1),
            captured_under_revision_id=uuid4(),
            bystanders_present=False,
        )


def test_asset_capture_datetime_max_does_not_overflow(monkeypatch):
    maximum = datetime.max.replace(tzinfo=UTC)
    monkeypatch.setattr(privacy_models, "_utc_now", lambda: maximum, raising=False)

    context = AssetContext(
        asset_id="asset-maximum",
        asset_sha256="8" * 64,
        gate_scope_fingerprint="9" * 64,
        captured_at=maximum,
        captured_under_revision_id=uuid4(),
        bystanders_present=False,
    )
    assert context.captured_at == maximum


class _GateBoundSourceProvenance(SourceAssetProvenance):
    gate_revision_id: UUID
    gate_scope_fingerprint: str


@pytest.mark.parametrize(
    ("revision_selector", "scope_selector", "capture_offset"),
    [
        ("other", "current", timedelta()),
        ("current", "other", timedelta()),
        ("current", "current", timedelta(microseconds=-1)),
    ],
)
def test_anonymized_source_must_belong_to_current_approval(
    tmp_path,
    approved_gate_factory,
    now,
    revision_selector,
    scope_selector,
    capture_offset,
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    source = _GateBoundSourceProvenance(
        asset_id="asset-source",
        sha256="8" * 64,
        captured_at=approved.approval.approved_at + capture_offset,
        gate_revision_id=(approved.revision_id if revision_selector == "current" else uuid4()),
        gate_scope_fingerprint=(
            approved.scope.fingerprint if scope_selector == "current" else "0" * 64
        ),
    )
    context = AssetContext(
        asset_id="asset-derived",
        asset_sha256="5" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=True,
        classifications=(
            SubjectClassification(
                asset_id="asset-derived",
                segment_ref="segment-1",
                subject_ref="subject-1",
                disposition=BystanderDisposition.ANONYMIZED,
            ),
        ),
        anonymization_records=(
            AnonymizationRecord(
                source_asset_id="asset-source",
                source_asset_sha256="8" * 64,
                derived_asset_id="asset-derived",
                method="blur",
                operator_id="privacy-operator",
                processed_at=now,
                output_sha256="5" * 64,
                visual_evidence_sha256="6" * 64,
                metadata_removal_evidence_sha256="7" * 64,
            ),
        ),
        source_asset_provenance=source,
    )

    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        context,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.DENY


def test_anonymization_requires_changed_media_digest(now):
    with pytest.raises(ValidationError):
        AnonymizationRecord(
            source_asset_id="asset-source",
            source_asset_sha256="8" * 64,
            derived_asset_id="asset-derived",
            method="rename-only",
            operator_id="privacy-operator",
            processed_at=now,
            output_sha256="8" * 64,
            visual_evidence_sha256="6" * 64,
            metadata_removal_evidence_sha256="7" * 64,
        )


@pytest.mark.parametrize(
    ("captured_under_revision_id", "captured_at_offset", "reason_code"),
    [
        (uuid4(), timedelta(), "ASSET_GATE_REVISION_MISMATCH"),
        (None, timedelta(microseconds=-1), "ASSET_CAPTURED_BEFORE_APPROVAL"),
    ],
)
def test_asset_capture_must_belong_to_current_approved_revision(
    tmp_path,
    approved_gate_factory,
    now,
    captured_under_revision_id,
    captured_at_offset,
    reason_code,
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now + captured_at_offset,
        captured_under_revision_id=captured_under_revision_id or approved.revision_id,
        bystanders_present=False,
    )
    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        context,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.DENY
    assert result.reason_code == reason_code


def test_multiple_subjects_use_strictest_decision(tmp_path, approved_gate_factory, now):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=True,
        classifications=(
            SubjectClassification(
                asset_id="asset-1",
                segment_ref="segment-1",
                subject_ref="subject-1",
                disposition=BystanderDisposition.CONSENTED_FOR_TRAINING,
                consent_version=approved.scope.training_consent_version,
                consent_evidence_sha256="9" * 64,
            ),
            SubjectClassification(
                asset_id="asset-1",
                segment_ref="segment-2",
                subject_ref="subject-2",
                disposition=BystanderDisposition.UNCONSENTED_IDENTIFIABLE,
            ),
        ),
    )
    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        context,
        MaterialUsePurpose.TRAINING,
    )
    assert result.decision is MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT


def test_anonymized_classification_requires_new_derived_asset_with_evidence(now):
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1",
            asset_sha256="8" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now,
            captured_under_revision_id=uuid4(),
            bystanders_present=True,
            classifications=(
                SubjectClassification(
                    asset_id="asset-1",
                    segment_ref="segment-1",
                    subject_ref="subject-1",
                    disposition=BystanderDisposition.ANONYMIZED,
                ),
            ),
        )
    with pytest.raises(ValidationError):
        AnonymizationRecord(
            source_asset_id="asset-1",
            source_asset_sha256="8" * 64,
            derived_asset_id="asset-1",
            method="blur",
            operator_id="privacy-operator",
            processed_at=now,
            output_sha256="5" * 64,
            visual_evidence_sha256="6" * 64,
            metadata_removal_evidence_sha256="7" * 64,
        )


def test_anonymized_asset_hash_must_match_derived_evidence(now):
    gate_revision_id = uuid4()
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1-anonymized",
            asset_sha256="0" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now,
            captured_under_revision_id=gate_revision_id,
            bystanders_present=True,
            classifications=(
                SubjectClassification(
                    asset_id="asset-1-anonymized",
                    segment_ref="segment-1",
                    subject_ref="subject-1",
                    disposition=BystanderDisposition.ANONYMIZED,
                ),
            ),
            anonymization_records=(
                AnonymizationRecord(
                    source_asset_id="asset-1",
                    source_asset_sha256="8" * 64,
                    derived_asset_id="asset-1-anonymized",
                    method="blur",
                    operator_id="privacy-operator",
                    processed_at=now,
                    output_sha256="5" * 64,
                    visual_evidence_sha256="6" * 64,
                    metadata_removal_evidence_sha256="7" * 64,
                ),
            ),
            source_asset_provenance={
                "asset_id": "asset-1",
                "sha256": "8" * 64,
                "captured_at": now,
                "gate_revision_id": gate_revision_id,
                "gate_scope_fingerprint": "9" * 64,
            },
        )


def test_anonymized_asset_accepts_matching_source_provenance(now):
    gate_revision_id = uuid4()
    context = AssetContext(
        asset_id="asset-1-anonymized",
        asset_sha256="5" * 64,
        gate_scope_fingerprint="9" * 64,
        captured_at=now,
        captured_under_revision_id=gate_revision_id,
        bystanders_present=True,
        classifications=(
            SubjectClassification(
                asset_id="asset-1-anonymized",
                segment_ref="segment-1",
                subject_ref="subject-1",
                disposition=BystanderDisposition.ANONYMIZED,
            ),
        ),
        anonymization_records=(
            AnonymizationRecord(
                source_asset_id="asset-1",
                source_asset_sha256="8" * 64,
                derived_asset_id="asset-1-anonymized",
                method="blur",
                operator_id="privacy-operator",
                processed_at=now - timedelta(seconds=1),
                output_sha256="5" * 64,
                visual_evidence_sha256="6" * 64,
                metadata_removal_evidence_sha256="7" * 64,
            ),
        ),
        source_asset_provenance={
            "asset_id": "asset-1",
            "sha256": "8" * 64,
            "captured_at": now - timedelta(seconds=2),
            "gate_revision_id": gate_revision_id,
            "gate_scope_fingerprint": "9" * 64,
        },
    )
    assert context.asset_id == "asset-1-anonymized"


def test_anonymized_asset_requires_source_provenance(now):
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1-anonymized",
            asset_sha256="5" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now,
            captured_under_revision_id=uuid4(),
            bystanders_present=True,
            classifications=(
                SubjectClassification(
                    asset_id="asset-1-anonymized",
                    segment_ref="segment-1",
                    subject_ref="subject-1",
                    disposition=BystanderDisposition.ANONYMIZED,
                ),
            ),
            anonymization_records=(
                AnonymizationRecord(
                    source_asset_id="asset-1",
                    source_asset_sha256="8" * 64,
                    derived_asset_id="asset-1-anonymized",
                    method="blur",
                    operator_id="privacy-operator",
                    processed_at=now,
                    output_sha256="5" * 64,
                    visual_evidence_sha256="6" * 64,
                    metadata_removal_evidence_sha256="7" * 64,
                ),
            ),
        )


def test_anonymization_processing_cannot_follow_derived_capture(now):
    gate_revision_id = uuid4()
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1-anonymized",
            asset_sha256="5" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now,
            captured_under_revision_id=gate_revision_id,
            bystanders_present=True,
            classifications=(
                SubjectClassification(
                    asset_id="asset-1-anonymized",
                    segment_ref="segment-1",
                    subject_ref="subject-1",
                    disposition=BystanderDisposition.ANONYMIZED,
                ),
            ),
            anonymization_records=(
                AnonymizationRecord(
                    source_asset_id="asset-1",
                    source_asset_sha256="8" * 64,
                    derived_asset_id="asset-1-anonymized",
                    method="blur",
                    operator_id="privacy-operator",
                    processed_at=now + timedelta(microseconds=1),
                    output_sha256="5" * 64,
                    visual_evidence_sha256="6" * 64,
                    metadata_removal_evidence_sha256="7" * 64,
                ),
            ),
            source_asset_provenance={
                "asset_id": "asset-1",
                "sha256": "8" * 64,
                "captured_at": now,
                "gate_revision_id": gate_revision_id,
                "gate_scope_fingerprint": "9" * 64,
            },
        )


@pytest.mark.parametrize(
    "source_provenance",
    [
        {"asset_id": "wrong-source", "sha256": "8" * 64},
        {"asset_id": "asset-1", "sha256": "0" * 64},
    ],
)
def test_anonymization_record_must_match_source_provenance(now, source_provenance):
    gate_revision_id = uuid4()
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1-anonymized",
            asset_sha256="5" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now,
            captured_under_revision_id=gate_revision_id,
            bystanders_present=True,
            classifications=(
                SubjectClassification(
                    asset_id="asset-1-anonymized",
                    segment_ref="segment-1",
                    subject_ref="subject-1",
                    disposition=BystanderDisposition.ANONYMIZED,
                ),
            ),
            anonymization_records=(
                AnonymizationRecord(
                    source_asset_id="asset-1",
                    source_asset_sha256="8" * 64,
                    derived_asset_id="asset-1-anonymized",
                    method="blur",
                    operator_id="privacy-operator",
                    processed_at=now - timedelta(seconds=1),
                    output_sha256="5" * 64,
                    visual_evidence_sha256="6" * 64,
                    metadata_removal_evidence_sha256="7" * 64,
                ),
            ),
            source_asset_provenance={
                **source_provenance,
                "captured_at": now - timedelta(seconds=2),
                "gate_revision_id": gate_revision_id,
                "gate_scope_fingerprint": "9" * 64,
            },
        )


def test_source_capture_cannot_follow_anonymization_processing(now):
    gate_revision_id = uuid4()
    with pytest.raises(ValidationError):
        AssetContext(
            asset_id="asset-1-anonymized",
            asset_sha256="5" * 64,
            gate_scope_fingerprint="9" * 64,
            captured_at=now,
            captured_under_revision_id=gate_revision_id,
            bystanders_present=True,
            classifications=(
                SubjectClassification(
                    asset_id="asset-1-anonymized",
                    segment_ref="segment-1",
                    subject_ref="subject-1",
                    disposition=BystanderDisposition.ANONYMIZED,
                ),
            ),
            anonymization_records=(
                AnonymizationRecord(
                    source_asset_id="asset-1",
                    source_asset_sha256="8" * 64,
                    derived_asset_id="asset-1-anonymized",
                    method="blur",
                    operator_id="privacy-operator",
                    processed_at=now - timedelta(seconds=1),
                    output_sha256="5" * 64,
                    visual_evidence_sha256="6" * 64,
                    metadata_removal_evidence_sha256="7" * 64,
                ),
            ),
            source_asset_provenance={
                "asset_id": "asset-1",
                "sha256": "8" * 64,
                "captured_at": now,
                "gate_revision_id": gate_revision_id,
                "gate_scope_fingerprint": "9" * 64,
            },
        )


def test_asset_scope_and_consent_version_are_bound_to_current_gate(
    tmp_path, approved_gate_factory, now
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    wrong_scope = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint="0" * 64,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )
    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        wrong_scope,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.DENY
    assert result.reason_code == "ASSET_GATE_SCOPE_MISMATCH"

    stale_consent = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=True,
        classifications=(
            SubjectClassification(
                asset_id="asset-1",
                segment_ref="segment-1",
                subject_ref="subject-1",
                disposition=BystanderDisposition.CONSENTED_FOR_SERVICE,
                consent_version="service-v0",
                consent_evidence_sha256="9" * 64,
            ),
        ),
    )
    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        stale_consent,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.DENY


def test_stale_or_blocked_revision_fails_closed(
    tmp_path, approved_gate_factory, gate_scope_factory, gate_evidence, now
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    blocked = gate_service.revise(
        approved.gate_id,
        gate_scope_factory(camera_config_id="top-view-v2"),
        "inspector-2",
        now,
        gate_evidence,
        "camera moved",
    )
    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )
    for revision_id in (approved.revision_id, blocked.revision_id):
        result = authorize_real_video_use(
            gate_service.repository,
            revision_id,
            context,
            MaterialUsePurpose.SERVICE_ANALYSIS,
        )
        assert result.decision is MaterialUseDecision.DENY


@pytest.mark.parametrize(
    "store_path",
    [
        "/dev/null/gates.json",
    ],
)
def test_repository_os_failures_return_typed_denial(store_path, now):
    from pathlib import Path

    from billiards_ai.cloud.privacy import JsonGateRepository

    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint="9" * 64,
        captured_at=now,
        captured_under_revision_id=uuid4(),
        bystanders_present=False,
    )
    result = authorize_real_video_use(
        JsonGateRepository(Path(store_path)),
        uuid4(),
        context,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.DENY
    assert result.reason_code == "GATE_RECORD_UNAVAILABLE"
