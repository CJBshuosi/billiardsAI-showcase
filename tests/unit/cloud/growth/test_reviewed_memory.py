"""Synthetic authority fixtures: local reviewed consumption, not field/user acceptance."""

import asyncio
from types import SimpleNamespace

import pytest

from billiards_ai.cloud.growth import reviewed_memory
from billiards_ai.cloud.growth.repository import InMemoryGrowthEvidenceRepository
from billiards_ai.cloud.growth.service import EvidenceIngestService
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatchV2
from billiards_ai.shared.models.metric_evidence import FieldSource, ReviewedFieldQuality
from tests.unit.cloud.growth.test_evidence_ingest import (
    NOW,
    OTHER_USER_ID,
    USER_ID,
    Authority,
    LineageAuthority,
    MetricAuthority,
    _approved,
    _batch_v2,
    _report,
    _uuid,
)


def event(batch):
    return DomainEvent(
        event_id=_uuid(800),
        event_type="evidence.session.approved.v1",
        aggregate_type="review_session",
        aggregate_id=batch.session_id,
        aggregate_version=batch.review_revision,
        occurred_at=batch.approved_at,
        correlation_id=batch.session_id,
        causation_id=_uuid(801),
        idempotency_key=f"evidence-approved:{batch.session_id}:{batch.review_revision}",
        payload={"approvedEvidenceBatch": batch.model_dump(mode="json")},
    )


def setup(batch=None, repository=None, approved=None):
    approved = approved or _approved()
    batch = batch or _batch_v2(approved)
    authority = Authority(approved)

    async def load_publication(batch_id):
        if batch_id != batch.batch_id:
            raise ValueError("unknown stored publication")
        return SimpleNamespace(batch=batch, event=event(batch))

    authority.load_publication = load_publication
    repo = repository or InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=authority,
        lineage_repository=LineageAuthority(approved),
        metric_repository=MetricAuthority(batch.metric_shots),
        growth_repository=repo,
    )
    return batch, authority, repo, service


def consume(batch, authority, service, user=USER_ID):
    return consume_event(
        event(batch),
        authenticated_user_id=user,
        approved_repository=authority,
        ingest_service=service,
    )


def consume_event(incoming, *, authenticated_user_id, approved_repository, ingest_service):
    # Synthetic async publication authority matching the real reader signature;
    # this is not PostgreSQL or field-subject integration acceptance.
    return asyncio.run(
        reviewed_memory.consume_reviewed_event(
            incoming,
            authenticated_user_id=authenticated_user_id,
            approved_repository=approved_repository,
            publication_repository=approved_repository,
            ingest_service=ingest_service,
        )
    )


def with_quality(batch, field, source, confidence):
    metric = batch.metric_shots[0]
    quality = metric.field_quality.model_copy(
        update={
            field: ReviewedFieldQuality(source=source, confidence=confidence),
        }
    )
    values = batch.model_dump(mode="python", exclude={"content_sha256"})
    values["metric_shots"] = (metric.model_copy(update={"field_quality": quality}),)
    return ApprovedEvidenceBatchV2.model_validate(
        values | {"content_sha256": canonical_sha256(values)}
    )


def test_existing_approved_v2_ingests_once_and_exposes_only_sealed_field_authority():
    batch, authority, repo, service = setup()
    first = consume(batch, authority, service)
    second = consume(batch, authority, service)
    assert first.status.value == "CREATED"
    assert second.status.value == "REPLAYED"
    assert len(repo.list_receipts()) == 1
    assert len(repo.list_outbox_events()) == 1
    fields = reviewed_memory._confirmed_fields(first.accepted_metric_input)
    assert "pot_result" in fields[batch.metric_shots[0].shot_id]
    unsealed = first.accepted_metric_input.model_copy()
    with pytest.raises(ValueError):
        reviewed_memory._confirmed_fields(unsealed)


@pytest.mark.parametrize("field", ["batch_id", "event_id", "causation_id"])
def test_forged_publication_identity_cannot_occupy_genuine_revision(field):
    batch, authority, repo, service = setup()
    incoming = event(batch)
    if field == "batch_id":
        values = batch.model_dump(mode="python", exclude={"content_sha256"})
        values["batch_id"] = _uuid(991)
        fake = ApprovedEvidenceBatchV2.model_validate(
            values | {"content_sha256": canonical_sha256(values)}
        )
        incoming = event(fake)
    else:
        incoming = incoming.model_copy(update={field: _uuid(992)})
    with pytest.raises(ValueError):
        consume_event(
            incoming,
            authenticated_user_id=USER_ID,
            approved_repository=authority,
            ingest_service=service,
        )
    assert repo.list_receipts() == ()
    assert consume(batch, authority, service).status.value == "CREATED"


@pytest.mark.parametrize(
    "source,confidence", [(FieldSource.CV, 1.0), (FieldSource.VLM, 0.8), (FieldSource.MANUAL, 0.4)]
)
def test_observed_or_low_quality_cannot_be_admitted_as_human_review(source, confidence):
    batch = with_quality(_batch_v2(_approved()), "pot_result", source, confidence)
    batch, authority, repo, service = setup(batch)
    with pytest.raises(ValueError):
        consume(batch, authority, service)
    assert repo.list_batches(USER_ID) == ()


def test_unknown_prerequisite_abstains_compound_fact_but_keeps_independent_reviewed_field():
    batch = with_quality(_batch_v2(_approved()), "geometry", FieldSource.UNKNOWN, 0.0)
    batch, authority, repo, service = setup(batch)
    result = consume(batch, authority, service)
    fields = reviewed_memory._confirmed_fields(result.accepted_metric_input)[
        batch.metric_shots[0].shot_id
    ]
    assert "pot_result" not in fields
    assert "geometry" not in fields
    assert "foul_result" in fields


@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregate_version", 99),
        ("aggregate_id", _uuid(999)),
        ("correlation_id", _uuid(999)),
        ("event_type", "growth.snapshot.created.v1"),
    ],
)
def test_event_envelope_mismatch_has_no_persistent_side_effect(field, value):
    batch, authority, repo, service = setup()
    invalid = event(batch).model_copy(update={field: value})
    with pytest.raises(ValueError):
        consume_event(
            invalid,
            authenticated_user_id=USER_ID,
            approved_repository=authority,
            ingest_service=service,
        )
    assert repo.list_batches(USER_ID) == ()


def test_wrong_authenticated_user_cannot_use_user_from_payload():
    batch, authority, repo, service = setup()
    with pytest.raises(ValueError):
        consume(batch, authority, service, OTHER_USER_ID)
    assert repo.list_batches(USER_ID) == ()


def test_same_revision_different_digest_conflicts_without_replacing_accepted_facts():
    batch, authority, repo, service = setup()
    consume(batch, authority, service)
    altered = with_quality(batch, "geometry", FieldSource.UNKNOWN, 0.0)
    with pytest.raises(ValueError):
        consume(altered, authority, service)
    assert repo.list_batches(USER_ID) == (batch,)


def test_retry_after_repository_failure_is_atomic():
    fail = True

    def inject(stage):
        if fail:
            raise RuntimeError("injected persistence failure")

    batch, authority, repo, service = setup(
        repository=InMemoryGrowthEvidenceRepository(failure_injector=inject)
    )
    with pytest.raises(RuntimeError):
        consume(batch, authority, service)
    assert repo.list_receipts() == ()
    fail = False
    assert consume(batch, authority, service).status.value == "CREATED"
    assert len(repo.list_outbox_events()) == 1


def test_correction_abstains_old_pot_statistics_and_late_old_event_cannot_restore_them():
    batch, authority, repo, service = setup()
    consume(batch, authority, service)
    before = reviewed_memory.project_current_reviewed_metrics(
        repo.list_accepted_metric_inputs(USER_ID),
        authenticated_user_id=USER_ID,
        approved_repository=authority,
        evidence_cutoff=NOW,
    )
    assert any(item.metric_code.value == "potting.success_rate" for item in before.observations)
    approved2 = _approved(_report(revision=2))
    revised = with_quality(_batch_v2(approved2), "geometry", FieldSource.UNKNOWN, 0.0)
    values = revised.model_dump(mode="python", exclude={"content_sha256"}) | {
        "batch_id": _uuid(900)
    }
    revised = ApprovedEvidenceBatchV2.model_validate(
        values | {"content_sha256": canonical_sha256(values)}
    )
    revised, authority2, _, service2 = setup(revised, repository=repo, approved=approved2)

    async def both_publications(batch_id):
        original = batch if batch_id == batch.batch_id else revised
        return SimpleNamespace(batch=original, event=event(original))

    authority2.load_publication = both_publications
    consume(revised, authority2, service2)
    with pytest.raises(ValueError):
        consume(batch, authority2, service)
    assert authority2.calls[-1] == (batch.session_id, batch.report_id, 1)
    after = reviewed_memory.project_current_reviewed_metrics(
        repo.list_accepted_metric_inputs(USER_ID),
        authenticated_user_id=USER_ID,
        approved_repository=authority2,
        evidence_cutoff=NOW,
    )
    assert not any(item.metric_code.value == "potting.success_rate" for item in after.observations)
    assert after.projection.superseded_batch_count == 1
    pot = [
        metric
        for metric in after.projection.metrics
        if metric.metric_code.value == "potting.success_rate"
    ]
    assert pot and all(metric.success_rate is None for metric in pot)
    assert len(repo.list_batches(USER_ID)) == 2  # immutable history, not accumulated results


def test_current_rejected_head_hides_previously_accepted_session_without_old_fallback():
    from billiards_ai.cloud.review.errors import ReportNotDeliverableError
    from billiards_ai.shared.models.review import ReportNotDeliverableReason

    batch, authority, repo, service = setup()
    consume(batch, authority, service)

    def revoked(*args):
        raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)

    authority.get_approved_session_data = revoked
    result = reviewed_memory.project_current_reviewed_metrics(
        repo.list_accepted_metric_inputs(USER_ID),
        authenticated_user_id=USER_ID,
        approved_repository=authority,
        evidence_cutoff=NOW,
    )
    assert result.observations == ()
    assert len(repo.list_batches(USER_ID)) == 1
    with pytest.raises(ReportNotDeliverableError):
        consume(batch, authority, service)
    again = reviewed_memory.project_current_reviewed_metrics(
        repo.list_accepted_metric_inputs(USER_ID),
        authenticated_user_id=USER_ID,
        approved_repository=authority,
        evidence_cutoff=NOW,
    )
    assert again.observations == ()
    assert len(repo.list_receipts()) == 1


def test_personal_read_cannot_mix_another_users_admissions():
    batch, authority, repo, service = setup()
    consume(batch, authority, service)
    with pytest.raises(ValueError):
        reviewed_memory.project_current_reviewed_metrics(
            repo.list_accepted_metric_inputs(USER_ID),
            authenticated_user_id=OTHER_USER_ID,
            approved_repository=authority,
            evidence_cutoff=NOW,
        )


def test_unreviewed_foul_prerequisite_cannot_create_a_legal_success_metric():
    batch = with_quality(_batch_v2(_approved()), "foul_result", FieldSource.UNKNOWN, 0.0)
    batch, authority, repo, service = setup(batch)
    with pytest.raises(ValueError):
        consume(batch, authority, service)
    assert repo.list_batches(USER_ID) == ()


def test_target_zone_requires_reviewed_geometry_before_metric_ingest():
    from billiards_ai.shared.models.metric_evidence import (
        ReviewedTargetZone,
        TargetZoneEvaluationStatus,
        TargetZoneResult,
        TargetZoneShape,
    )
    from billiards_ai.shared.models.shot import TargetZoneSource

    batch = with_quality(_batch_v2(_approved()), "geometry", FieldSource.UNKNOWN, 0.0)
    shot = batch.metric_shots[0]
    geometry = shot.table_geometry
    zone = ReviewedTargetZone(
        calibration_id=geometry.calibration_id,
        calibration_revision_id=geometry.calibration_revision_id,
        calibration_revision_number=geometry.calibration_revision_number,
        shape=TargetZoneShape.RECTANGLE,
        minimum_x_mm=50.0,
        maximum_x_mm=150.0,
        minimum_y_mm=250.0,
        maximum_y_mm=350.0,
    )
    shot = shot.model_copy(
        update={
            "target_zone_source": TargetZoneSource.SCENE_PRESET,
            "target_zone": zone,
            "cue_ball_end_mm": shot.cue_ball_start_mm,
            "target_zone_evaluation_status": TargetZoneEvaluationStatus.ELIGIBLE,
            "target_zone_result": TargetZoneResult.INSIDE_OR_TOLERATED,
        }
    )
    values = batch.model_dump(mode="python", exclude={"content_sha256"}) | {"metric_shots": (shot,)}
    batch = ApprovedEvidenceBatchV2.model_validate(
        values | {"content_sha256": canonical_sha256(values)}
    )
    batch, authority, repo, service = setup(batch)
    with pytest.raises(ValueError):
        consume(batch, authority, service)
    assert repo.list_batches(USER_ID) == ()
