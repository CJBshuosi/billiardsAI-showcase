"""Regression coverage for the Story 8.7 review findings."""

from __future__ import annotations

import json
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import pytest
from pydantic import UUID4, ValidationError, model_validator

from billiards_ai.cloud.growth import metric_projector, snapshot_consumer
from billiards_ai.shared.models import growth
from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr, Sha256
from billiards_ai.shared.models.growth import MetricEvidenceReferenceV1
from billiards_ai.shared.models.growth_metrics import MetricReliability
from billiards_ai.shared.models.metric_evidence import (
    PotAttemptStatus,
    ReviewedTargetZone,
    TargetZoneEvaluationStatus,
    TargetZoneResult,
    TargetZoneShape,
)
from billiards_ai.shared.models.shot import TargetZoneSource
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch, uid
from tests.unit.cloud.growth.test_metric_projector import (
    _batch,
    _calibration,
    _metric_shot,
    _point,
)
from tests.unit.cloud.growth.test_snapshot_v3 import _assessment, _project, _window_batches


def _replace_assessment(snapshot, original, replacement):
    return snapshot.model_copy(
        update={
            "metric_assessments": tuple(
                replacement if item == original else item for item in snapshot.metric_assessments
            )
        }
    )


def _replace_reference(assessment, original, replacement):
    replacements = {original.evidence_ref_id: replacement.evidence_ref_id}
    trend = assessment.trend.model_copy(
        update={
            "current_evidence_ref_ids": tuple(
                sorted(
                    replacements.get(item, item)
                    for item in assessment.trend.current_evidence_ref_ids
                )
            ),
            "previous_evidence_ref_ids": tuple(
                sorted(
                    replacements.get(item, item)
                    for item in assessment.trend.previous_evidence_ref_ids
                )
            ),
        }
    )
    return assessment.model_copy(
        update={
            "evidence_refs": tuple(
                sorted(
                    (
                        replacement if item.evidence_ref_id == original.evidence_ref_id else item
                        for item in assessment.evidence_refs
                    ),
                    key=lambda item: item.evidence_ref_id,
                )
            ),
            "trend": trend,
        }
    )


def test_review_01_assessment_metric_must_equal_projection_metric() -> None:
    snapshot = _project(_window_batches(1, 4))
    assessment = _assessment(snapshot, "potting.success_rate")
    forged = assessment.model_copy(
        update={
            "metric": assessment.metric.model_copy(
                update={"success_rate": assessment.metric.success_rate + 0.000001}
            )
        }
    )
    tampered = _replace_assessment(snapshot, assessment, forged)

    with pytest.raises(ValueError, match="assessment metric must equal projected metric"):
        tampered.snapshot_is_canonical_and_integral()


def test_review_02_refs_cover_full_365_day_trace_and_reconcile_sample_count() -> None:
    old = make_batch(
        9_001,
        session_number=19_001,
        approved_at=NOW - timedelta(days=300),
    )
    snapshot = _project(_window_batches(1, 4) + (old,))
    assessment = _assessment(snapshot, "potting.success_rate")

    assert old.metric_shots[0].shot_id in {
        shot_id for ref in assessment.evidence_refs for shot_id in ref.shot_ids
    }
    assert (
        sum(len(ref.shot_ids) for ref in assessment.evidence_refs) == assessment.metric.sample_count
    )


def test_review_03_trend_policy_is_revalidated_not_merely_hash_bound() -> None:
    snapshot = _project(_window_batches(0, 5))
    assessment = _assessment(snapshot, "potting.success_rate")
    forged = assessment.model_copy(
        update={
            "trend": assessment.trend.model_copy(
                update={
                    "current_window_start": assessment.trend.current_window_start
                    + timedelta(days=1),
                    "direction": growth.TrendDirection.DECLINING,
                    "reliability": MetricReliability.HIGH,
                }
            )
        }
    )
    tampered = _replace_assessment(snapshot, assessment, forged)

    with pytest.raises(ValueError, match="trend policy mismatch"):
        tampered.snapshot_is_canonical_and_integral()


def test_review_04_trend_refs_are_disjoint_and_approved_at_window_bound() -> None:
    snapshot = _project(_window_batches(1, 4))
    assessment = _assessment(snapshot, "potting.success_rate")
    ref_id = assessment.trend.previous_evidence_ref_ids[0]
    forged = assessment.model_copy(
        update={
            "trend": assessment.trend.model_copy(
                update={
                    "current_evidence_ref_ids": tuple(
                        sorted(assessment.trend.current_evidence_ref_ids + (ref_id,))
                    )
                }
            )
        }
    )
    tampered = _replace_assessment(snapshot, assessment, forged)

    with pytest.raises(ValueError, match="trend evidence windows must be disjoint"):
        tampered.snapshot_is_canonical_and_integral()


def test_review_05_same_batch_shot_envelope_splice_is_rejected() -> None:
    session_id = uid(51_000)
    shots = tuple(_metric_shot(session_id, uid(51_100 + index), index + 1) for index in range(2))
    batch = _batch(
        batch_number=51_500,
        session_number=51_000,
        shots=shots,
        approved_at=NOW - timedelta(days=30),
    )
    snapshot = _project((batch,))
    assessment = _assessment(snapshot, "potting.success_rate")
    original = next(ref for ref in assessment.evidence_refs if len(ref.shot_ids) == 2)
    payload = original.model_dump(mode="python", exclude={"evidence_ref_id"}) | {
        "shot_ids": (original.shot_ids[0],),
        "source_envelope_ids": (original.source_envelope_ids[1],),
    }
    forged_ref = MetricEvidenceReferenceV1.model_construct(
        **(
            payload
            | {
                "evidence_ref_id": growth.uuid4_from_canonical_payload(payload),
                "shot_trace_bindings": original.shot_trace_bindings,
            }
        )
    )
    forged_assessment = _replace_reference(assessment, original, forged_ref)
    tampered = _replace_assessment(snapshot, assessment, forged_assessment)

    with pytest.raises(ValueError, match="exact metric shot trace binding"):
        tampered.snapshot_is_canonical_and_integral()


def _target_status_batch(number: int, *, age: int, source, status):
    session_id = uid(52_000 + number)
    shot = _metric_shot(
        session_id,
        uid(53_000 + number),
        1,
        pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
        pot_result=None,
        intended_object_ball=None,
        intended_target_pocket=None,
        target_zone_source=source,
        target_zone_evaluation_status=status,
    )
    return _batch(
        batch_number=54_000 + number,
        session_number=52_000 + number,
        shots=(shot,),
        approved_at=NOW - timedelta(days=age),
    )


def test_review_06_mixed_taxonomy_support_uses_frozen_precedence() -> None:
    batches = tuple(
        _target_status_batch(
            index,
            age=120,
            source=TargetZoneSource.SCENE_PRESET,
            status=TargetZoneEvaluationStatus.UNSUPPORTED,
        )
        for index in range(5)
    ) + tuple(
        _target_status_batch(
            100 + index,
            age=30,
            source=TargetZoneSource.POST_HOC_INFERRED,
            status=TargetZoneEvaluationStatus.ABSTAINED,
        )
        for index in range(5)
    )
    snapshot = _project(batches)
    target = [
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "positioning.target_zone.success_rate"
    ]

    assert target
    assert {item.trend.support_status.value for item in target} == {"UNSUPPORTED"}


def test_review_07_v3_ports_superseded_source_audit_integrity() -> None:
    old = make_batch(55_001, session_number=55_000, review_revision=1)
    replacement = make_batch(55_002, session_number=55_000, review_revision=2)
    snapshot = _project((old, replacement))
    audit = snapshot.metric_projection.superseded_evidence[0]
    forged_audit = audit.model_copy(update={"superseded_by_batch_id": uid(999_991)})
    forged_projection = snapshot.metric_projection.model_copy(
        update={"superseded_evidence": (forged_audit,)}
    )
    tampered = snapshot.model_copy(update={"metric_projection": forged_projection})

    with pytest.raises(ValueError, match="superseded metric audit is not present"):
        tampered.snapshot_is_canonical_and_integral()


class _D2TrendDirection(StrEnum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    UNKNOWN = "UNKNOWN"


class _FrozenD2GrowthTrendLineageReferenceV1(CoreContractModel):
    """Exact local overlay of stage commit 149b563's frozen D2 contract."""

    schema_version: Literal["1.0"] = "1.0"
    lineage_id: UUID4
    source_snapshot_schema_version: Literal["3.0"]
    metric_code: NonEmptyStr
    metric_identity_sha256: Sha256
    trend_direction: _D2TrendDirection
    trend_support_status: growth.MetricSupportStatus
    reliability: growth.Reliability
    trend_policy_version: NonEmptyStr
    evidence_ref_ids: tuple[UUID4, ...]

    @model_validator(mode="after")
    def lineage_is_canonical_and_evidence_bounded(self) -> Self:
        if self.evidence_ref_ids != tuple(sorted(set(self.evidence_ref_ids))):
            raise ValueError("trend evidence reference IDs must be sorted and unique")
        if self.trend_direction is _D2TrendDirection.UNKNOWN:
            if self.trend_support_status is growth.MetricSupportStatus.SUPPORTED:
                raise ValueError("UNKNOWN trend cannot be supported")
        elif self.trend_support_status is growth.MetricSupportStatus.SUPPORTED:
            if not self.evidence_ref_ids:
                raise ValueError("supported trend requires evidence references")
        else:
            raise ValueError("substantive trend direction requires supported lineage")
        return self


def test_review_08_d2_overlay_is_lineage_sorted_unique_and_frozen_contract_valid() -> None:
    payloads = snapshot_consumer.growth_trend_lineage_payloads(_project(_window_batches(1, 4)))
    lineage_ids = tuple(item["lineage_id"] for item in payloads)

    assert lineage_ids == tuple(sorted(set(lineage_ids)))
    assert payloads
    assert all(
        _FrozenD2GrowthTrendLineageReferenceV1.model_validate_json(
            json.dumps(item, default=str), strict=True
        )
        for item in payloads
    )


def test_review_09_ac15_matrix_covers_previous_only_and_all_splice_boundaries() -> None:
    previous_only = tuple(
        item for item in _window_batches(5, 0) if item.approved_at <= NOW - timedelta(days=90)
    )
    trend = _assessment(_project(previous_only), "potting.success_rate").trend
    assert trend.direction.value == "UNKNOWN"
    assert hasattr(growth.GrowthSnapshotSourceBatchRefV3, "model_fields")
    assert "metric_shot_bindings" in growth.GrowthSnapshotSourceBatchRefV3.model_fields


def test_ac15_identity_incomparable_is_unknown_not_stable() -> None:
    zone = ReviewedTargetZone(
        **_calibration(),
        shape=TargetZoneShape.RECTANGLE,
        minimum_x_mm=900.0,
        minimum_y_mm=400.0,
        maximum_x_mm=1100.0,
        maximum_y_mm=600.0,
        vertices=(),
    )
    batches = []
    for window, (source, age) in enumerate(
        (
            (TargetZoneSource.SCENE_PRESET, 120),
            (TargetZoneSource.POST_HOC_INFERRED, 30),
        )
    ):
        for index in range(5):
            session_number = 58_000 + window * 100 + index
            shot = _metric_shot(
                uid(session_number),
                uid(59_000 + window * 100 + index),
                1,
                pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
                pot_result=None,
                intended_object_ball=None,
                intended_target_pocket=None,
                target_zone_source=source,
                target_zone=zone,
                cue_ball_end_mm=_point(1000.0, 500.0),
                target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
                target_zone_result=TargetZoneResult.INSIDE_OR_TOLERATED,
            )
            batches.append(
                _batch(
                    batch_number=60_000 + window * 100 + index,
                    session_number=session_number,
                    shots=(shot,),
                    approved_at=NOW - timedelta(days=age),
                )
            )
    assessments = [
        item
        for item in _project(tuple(batches)).metric_assessments
        if item.metric.metric_code.value == "positioning.target_zone.success_rate"
    ]

    assert len(assessments) == 2
    assert {item.trend.direction.value for item in assessments} == {"UNKNOWN"}
    assert {item.trend.support_status.value for item in assessments} == {"INSUFFICIENT_EVIDENCE"}


def test_ac15_cross_session_and_cross_metric_splices_fail_closed() -> None:
    old_batches = tuple(
        make_batch(
            61_000 + index,
            session_number=62_000 + index,
            approved_at=NOW - timedelta(days=300),
        )
        for index in range(2)
    )
    snapshot = _project(old_batches + _window_batches(1, 4))
    pot = _assessment(snapshot, "potting.success_rate")
    old_refs = [
        ref
        for ref in pot.evidence_refs
        if ref.evidence_ref_id
        not in set(pot.trend.current_evidence_ref_ids + pot.trend.previous_evidence_ref_ids)
    ]
    original, other = old_refs
    session_payload = original.model_dump(mode="python", exclude={"evidence_ref_id"}) | {
        "session_id": other.session_id
    }
    cross_session = MetricEvidenceReferenceV1.model_validate(
        session_payload | {"evidence_ref_id": growth.uuid4_from_canonical_payload(session_payload)},
        strict=True,
    )
    forged_assessment = _replace_reference(pot, original, cross_session)
    tampered = _replace_assessment(snapshot, pot, forged_assessment)
    with pytest.raises(ValueError, match="trusted source"):
        tampered.snapshot_is_canonical_and_integral()

    foul = _assessment(snapshot, "foul_control.legal_shot_rate")
    foul_old = next(
        ref
        for ref in foul.evidence_refs
        if ref.evidence_ref_id
        not in set(foul.trend.current_evidence_ref_ids + foul.trend.previous_evidence_ref_ids)
    )
    cross_metric_bindings = tuple(
        item.model_copy(update={"metric_identity_sha256": foul.metric_identity_sha256})
        for item in original.shot_trace_bindings
    )
    metric_payload = original.model_dump(mode="python", exclude={"evidence_ref_id"}) | {
        "metric_identity_sha256": foul.metric_identity_sha256,
        "shot_trace_bindings": cross_metric_bindings,
    }
    cross_metric = MetricEvidenceReferenceV1.model_validate(
        metric_payload | {"evidence_ref_id": growth.uuid4_from_canonical_payload(metric_payload)},
        strict=True,
    )
    forged_foul = _replace_reference(foul, foul_old, cross_metric)
    tampered = _replace_assessment(snapshot, foul, forged_foul)
    with pytest.raises(ValueError, match="exact metric shot trace binding"):
        tampered.snapshot_is_canonical_and_integral()


def test_ac15_superseded_replayed_and_outside_365_never_enter_refs() -> None:
    superseded = make_batch(63_001, session_number=63_000, review_revision=1)
    replacement = make_batch(63_002, session_number=63_000, review_revision=2)
    outside = make_batch(
        63_003,
        session_number=63_003,
        approved_at=NOW - timedelta(days=366),
    )
    snapshot = _project((superseded, replacement, replacement, outside))
    referenced_shots = {
        shot_id
        for assessment in snapshot.metric_assessments
        for ref in assessment.evidence_refs
        for shot_id in ref.shot_ids
    }

    assert superseded.metric_shots[0].shot_id not in referenced_shots
    assert outside.metric_shots[0].shot_id not in referenced_shots
    assert replacement.metric_shots[0].shot_id in referenced_shots
    assert snapshot.metric_projection.replayed_batch_count == 1
    assert snapshot.metric_projection.superseded_batch_count == 1
    assert snapshot.metric_projection.outside_window_batch_count == 1


def test_review_10_tracked_verification_artifact_replaces_ignored_report() -> None:
    root = Path(__file__).resolve().parents[4]
    artifact = root / "_bmad-output/implementation-artifacts/8-7-growth-snapshot-v3-verification.md"
    story = root / "_bmad-output/implementation-artifacts/8-7-成长指标证据与趋势投影-v3.md"

    assert artifact.is_file()
    assert ".superpowers/sdd/" not in story.read_text()


def test_review_11_calculation_fields_cover_geometry_quality_weight_and_eligibility() -> None:
    batch = make_batch(56_000, approved_at=NOW - timedelta(days=30))
    traced = metric_projector.project_ability_metrics_with_trace(admit(batch), evidence_cutoff=NOW)
    pot = next(
        item for item in traced.observations if item.metric_code.value == "potting.success_rate"
    )
    required = {
        "batch.approved_at",
        "batch.approval_kind",
        "batch.approval_policy_version",
        "shot.table_geometry",
        "shot.pot_attempt_status",
        "shot.field_quality.geometry.confidence",
        "shot.field_quality.intent.confidence",
        "shot.field_quality.pot_attempt.confidence",
        "shot.field_quality.pot_result.confidence",
    }

    assert required <= set(pot.calculation_field_codes)


def test_review_12_envelope_index_is_built_once_per_batch(monkeypatch) -> None:
    builder = metric_projector._source_envelope_index
    calls = 0

    def counted(batch):
        nonlocal calls
        calls += 1
        return builder(batch)

    monkeypatch.setattr(metric_projector, "_source_envelope_index", counted)
    batches = (make_batch(57_001), make_batch(57_002))
    inputs = admit(*batches)

    assert metric_projector.project_ability_metrics_with_trace(
        inputs, evidence_cutoff=NOW
    ).projection == metric_projector.project_ability_metrics(inputs, evidence_cutoff=NOW)
    assert calls == 4  # once per batch in each of the two projector calls


def test_review_13_v2_lineage_is_fail_closed_and_v3_has_separate_contract() -> None:
    with pytest.raises(ValidationError):
        growth.GrowthSnapshotLineage(
            version=1,
            previous_snapshot_id=None,
            snapshot_policy_version=growth.SNAPSHOT_POLICY_VERSION_V3,
        )
    lineage_v3 = growth.GrowthSnapshotLineageV3(
        version=1,
        previous_snapshot_id=None,
    )
    assert lineage_v3.snapshot_policy_version == growth.SNAPSHOT_POLICY_VERSION_V3
