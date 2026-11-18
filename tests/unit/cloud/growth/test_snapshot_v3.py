"""Story 8.7 contract, trend policy, and deterministic projection tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from billiards_ai.cloud.growth import snapshot_consumer, snapshot_projector
from billiards_ai.cloud.growth.metric_projector import (
    project_ability_metrics,
    project_ability_metrics_with_trace,
)
from billiards_ai.cloud.growth.repository import stored_model_values
from billiards_ai.shared.models import growth
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import AcceptedMetricEvidenceInput
from billiards_ai.shared.models.growth import (
    GrowthSnapshotLineage,
    GrowthSnapshotLineageV3,
    MetricEvidenceReferenceV1,
)
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


def _window_batches(
    previous_successes: int,
    current_successes: int,
    *,
    count: int = 5,
):
    batches = []
    for index in range(count):
        batches.append(
            make_batch(
                100 + index,
                session_number=1_000 + index,
                approved_at=NOW - timedelta(days=120),
                successful=index < previous_successes,
            )
        )
        batches.append(
            make_batch(
                200 + index,
                session_number=2_000 + index,
                approved_at=NOW - timedelta(days=30),
                successful=index < current_successes,
            )
        )
    return tuple(batches)


def _project(batches, *, cutoff=NOW):
    return snapshot_projector.project_growth_snapshot_v3(
        admit(*batches),
        evidence_cutoff=cutoff,
        lineage=GrowthSnapshotLineageV3(
            version=1,
            previous_snapshot_id=None,
            snapshot_policy_version="growth-snapshot-rd-v3.0.0",
        ),
    )


def _assessment(snapshot, metric_code: str):
    return next(
        item for item in snapshot.metric_assessments if item.metric.metric_code == metric_code
    )


def _error_batches(previous_error: float, current_error: float):
    zone = ReviewedTargetZone(
        **_calibration(),
        shape=TargetZoneShape.RECTANGLE,
        minimum_x_mm=1000.0,
        minimum_y_mm=400.0,
        maximum_x_mm=1200.0,
        maximum_y_mm=600.0,
        vertices=(),
    )
    batches = []
    for window, error, age in ((0, previous_error, 120), (1, current_error, 30)):
        for index in range(5):
            session_number = 30_000 + window * 100 + index
            shot = _metric_shot(
                uid(session_number),
                uid(31_000 + window * 100 + index),
                1,
                pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
                pot_result=None,
                intended_object_ball=None,
                intended_target_pocket=None,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                target_zone=zone,
                cue_ball_end_mm=_point(1200.0 + error, 500.0),
                target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
                target_zone_result=(
                    TargetZoneResult.INSIDE_OR_TOLERATED
                    if error <= 20.0
                    else TargetZoneResult.OUTSIDE
                ),
            )
            batches.append(
                _batch(
                    batch_number=32_000 + window * 100 + index,
                    session_number=session_number,
                    shots=(shot,),
                    approved_at=NOW - timedelta(days=age),
                )
            )
    return tuple(batches)


def _pot_status_batches(status: PotAttemptStatus, *, count: int = 5):
    batches = []
    for window, age in ((0, 120), (1, 30)):
        for index in range(count):
            session_number = 40_000 + window * 100 + index
            updates = {}
            if status is not PotAttemptStatus.ATTEMPTED:
                updates = {
                    "pot_attempt_status": status,
                    "pot_result": None,
                    "intended_object_ball": None,
                    "intended_target_pocket": None,
                }
            shot = _metric_shot(
                uid(session_number),
                uid(41_000 + window * 100 + index),
                1,
                **updates,
            )
            batches.append(
                _batch(
                    batch_number=42_000 + window * 100 + index,
                    session_number=session_number,
                    shots=(shot,),
                    approved_at=NOW - timedelta(days=age),
                )
            )
    return tuple(batches)


def test_same_pass_trace_preserves_v2_projection_and_exact_shot_envelope_binding() -> None:
    session_id = uid(50_000)
    shots = tuple(_metric_shot(session_id, uid(50_100 + index), index + 1) for index in range(2))
    batch = _batch(batch_number=50_200, session_number=50_000, shots=shots)
    admitted = admit(batch)

    legacy = project_ability_metrics(admitted, evidence_cutoff=NOW)
    traced = project_ability_metrics_with_trace(admitted, evidence_cutoff=NOW)

    assert traced.projection == legacy
    expected = dict(zip((item.shot_id for item in shots), batch.source_envelope_ids, strict=True))
    assert traced.observations
    assert all(
        item.source_envelope_ids == (expected[item.shot_id],) for item in traced.observations
    )


def test_v3_contract_is_strict_frozen_canonical_and_v1_v2_remain_compatible() -> None:
    snapshot = _project(_window_batches(0, 5))

    assert snapshot.schema_version == "3.0"
    assert snapshot.snapshot_policy_version == "growth-snapshot-rd-v3.0.0"
    assert snapshot.trend_policy_version == "growth-trend-rd-v1.0.0"
    assert TypeAdapter(growth.GrowthEvidenceSnapshotAny).validate_python(snapshot) == snapshot
    assert growth.GrowthEvidenceSnapshot.model_fields["schema_version"].default == "1.0"
    assert growth.GrowthEvidenceSnapshotV2.model_fields["schema_version"].default == "2.0"
    with pytest.raises(ValidationError):
        growth.GrowthEvidenceSnapshotV3.model_validate(
            snapshot.model_dump(mode="python") | {"unexpected": True}, strict=True
        )
    with pytest.raises(ValidationError):
        snapshot.version = 2
    with pytest.raises(ValidationError):
        growth.GrowthEvidenceSnapshotV3.model_validate(
            snapshot.model_dump(mode="python") | {"schema_version": "4.0"}, strict=True
        )


def test_metric_lineage_is_per_metric_raw_free_canonical_and_identity_bound() -> None:
    snapshot = _project(_window_batches(1, 4))
    identities = []
    reference_ids = []
    forbidden = ("raw", "trajectory", "locator", "prompt", "provider", "openid", "token")

    for assessment in snapshot.metric_assessments:
        identities.append(assessment.metric_identity_sha256)
        expected_identity = canonical_sha256(
            {
                "metric_code": assessment.metric.metric_code.value,
                "taxonomy": assessment.metric.taxonomy.model_dump(mode="json"),
                "calculation_policy_version": assessment.metric.calculation_policy_version,
            }
        )
        assert assessment.metric_identity_sha256 == expected_identity
        if assessment.metric.sample_count:
            assert assessment.evidence_refs
        assert all(
            ref.metric_identity_sha256 == expected_identity for ref in assessment.evidence_refs
        )
        for ref in assessment.evidence_refs:
            reference_ids.append(ref.evidence_ref_id)
            assert ref.shot_ids == tuple(sorted(set(ref.shot_ids)))
            assert ref.source_envelope_ids == tuple(sorted(set(ref.source_envelope_ids)))
            assert ref.calculation_field_codes == tuple(sorted(set(ref.calculation_field_codes)))
            payload = ref.model_dump(mode="json", exclude={"evidence_ref_id"})
            assert ref.evidence_ref_id == growth.uuid4_from_canonical_payload(payload)
            assert not any(word in str(payload).lower() for word in forbidden)
    assert len(identities) == len(set(identities))
    assert len(reference_ids) == len(set(reference_ids))


@pytest.mark.parametrize(
    ("previous", "current", "direction"),
    [(0, 5, "IMPROVING"), (5, 0, "DECLINING"), (3, 3, "STABLE")],
)
def test_rate_trend_three_directions(previous: int, current: int, direction: str) -> None:
    assessment = _assessment(_project(_window_batches(previous, current)), "potting.success_rate")
    assert assessment.trend.direction.value == direction
    assert assessment.trend.support_status.value == "SUPPORTED"
    assert assessment.trend.current_window_start == NOW - timedelta(days=90)
    assert assessment.trend.current_window_end == NOW
    assert assessment.trend.previous_window_start == NOW - timedelta(days=180)
    assert assessment.trend.previous_window_end == NOW - timedelta(days=90)


def test_rate_threshold_equality_and_adjacent_window_boundary_are_deterministic() -> None:
    batches = list(_window_batches(10, 11, count=20))
    boundary = make_batch(
        990,
        session_number=9_990,
        approved_at=NOW - timedelta(days=90),
        successful=False,
    )
    snapshot = _project(tuple(batches))
    pot = _assessment(snapshot, "potting.success_rate")

    assert pot.trend.direction.value == "IMPROVING"
    assert pot.trend.delta == pytest.approx(0.05)
    boundary_pot = _assessment(_project(tuple(batches) + (boundary,)), "potting.success_rate")
    previous_ids = {
        shot_id
        for ref in boundary_pot.evidence_refs
        if ref.evidence_ref_id in boundary_pot.trend.previous_evidence_ref_ids
        for shot_id in ref.shot_ids
    }
    current_ids = {
        shot_id
        for ref in boundary_pot.evidence_refs
        if ref.evidence_ref_id in boundary_pot.trend.current_evidence_ref_ids
        for shot_id in ref.shot_ids
    }
    assert boundary.metric_shots[0].shot_id in previous_ids
    assert boundary.metric_shots[0].shot_id not in current_ids


@pytest.mark.parametrize(
    ("previous", "current", "direction"),
    [
        (100.0, 75.0, "IMPROVING"),
        (75.0, 100.0, "DECLINING"),
        (100.0, 110.0, "STABLE"),
    ],
)
def test_error_trend_polarity_three_directions_and_threshold_equality(
    previous: float, current: float, direction: str
) -> None:
    assessment = _assessment(
        _project(_error_batches(previous, current)),
        "positioning.target_zone.error_mm",
    )
    assert assessment.trend.direction.value == direction
    assert assessment.trend.delta == pytest.approx(current - previous)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (PotAttemptStatus.NOT_ATTEMPTED, "INSUFFICIENT_EVIDENCE"),
        (PotAttemptStatus.UNSUPPORTED, "UNSUPPORTED"),
        (PotAttemptStatus.ABSTAINED, "ABSTAINED"),
        (PotAttemptStatus.ATTEMPTED, "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_unknown_support_matrix_never_fabricates_stable(
    status: PotAttemptStatus, expected: str
) -> None:
    count = 1 if status is PotAttemptStatus.ATTEMPTED else 5
    trend = _assessment(
        _project(_pot_status_batches(status, count=count)), "potting.success_rate"
    ).trend
    assert trend.direction.value == "UNKNOWN"
    assert trend.support_status.value == expected
    assert trend.current_value is trend.previous_value is trend.delta is None


def test_cross_batch_shot_splice_fails_before_hash_validation() -> None:
    snapshot = _project(_window_batches(1, 4))
    assessment = _assessment(snapshot, "potting.success_rate")
    original = assessment.evidence_refs[0]
    other = assessment.evidence_refs[-1]
    assert original.batch_id != other.batch_id
    forged_payload = original.model_dump(mode="python", exclude={"evidence_ref_id"}) | {
        "shot_ids": other.shot_ids
    }
    with pytest.raises(ValidationError, match="exact metric shot trace binding"):
        MetricEvidenceReferenceV1.model_validate(
            forged_payload
            | {"evidence_ref_id": growth.uuid4_from_canonical_payload(forged_payload)},
            strict=True,
        )


def test_missing_window_and_insufficient_evidence_never_become_stable() -> None:
    current_only = tuple(
        batch for batch in _window_batches(0, 5) if batch.approved_at > NOW - timedelta(days=90)
    )
    snapshot = _project(current_only)
    trend = _assessment(snapshot, "potting.success_rate").trend

    assert trend.direction.value == "UNKNOWN"
    assert trend.support_status.value == "INSUFFICIENT_EVIDENCE"
    assert trend.reliability.value == "UNKNOWN"
    assert trend.current_value is None
    assert trend.previous_value is None
    assert trend.delta is None


def test_canonical_replay_order_and_hash_fence_tamper_fail_closed() -> None:
    batches = _window_batches(1, 4)
    admitted = admit(*batches)
    lineage = GrowthSnapshotLineageV3(
        version=1,
        previous_snapshot_id=None,
        snapshot_policy_version="growth-snapshot-rd-v3.0.0",
    )
    forward = snapshot_projector.project_growth_snapshot_v3(
        admitted, evidence_cutoff=NOW, lineage=lineage
    )
    reverse = snapshot_projector.project_growth_snapshot_v3(
        tuple(reversed(admitted)), evidence_cutoff=NOW, lineage=lineage
    )

    assert forward.model_dump_json() == reverse.model_dump_json()
    assert forward.snapshot_id == reverse.snapshot_id
    assert forward.content_sha256 == reverse.content_sha256
    for update in (
        {"evidence_sha256": "f" * 64},
        {"content_sha256": "e" * 64},
        {"snapshot_id": uid(999_001)},
        {"read_fence": forward.read_fence.model_copy(update={"version": 2})},
        {
            "metric_assessments": tuple(reversed(forward.metric_assessments)),
        },
    ):
        with pytest.raises(ValidationError):
            growth.GrowthEvidenceSnapshotV3.model_validate(
                forward.model_dump(mode="python") | update, strict=True
            )


def test_v3_requires_live_ingest_admission_and_rejects_copy_round_trips() -> None:
    admitted = admit(*_window_batches(1, 4))
    bad_values = (
        tuple(
            AcceptedMetricEvidenceInput.model_validate(item.model_dump(mode="python"))
            for item in admitted
        ),
        tuple(deepcopy(item) for item in admitted),
        tuple(item.model_copy() for item in admitted),
        tuple(
            AcceptedMetricEvidenceInput.model_construct(**stored_model_values(item))
            for item in admitted
        ),
    )
    for values in bad_values:
        with pytest.raises(snapshot_projector.SnapshotProjectionError):
            snapshot_projector.project_growth_snapshot_v3(
                values,
                evidence_cutoff=NOW,
                lineage=GrowthSnapshotLineageV3(
                    version=1,
                    previous_snapshot_id=None,
                    snapshot_policy_version="growth-snapshot-rd-v3.0.0",
                ),
            )


def test_v2_and_v3_consumers_require_exact_major_and_exact_fence() -> None:
    v3 = _project(_window_batches(1, 4))
    admitted = admit(make_batch(1))
    v2 = snapshot_projector.project_growth_snapshot(
        admitted,
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )

    assert (
        snapshot_consumer.prepare_growth_snapshot_v3_for_consumer(
            v3.model_dump_json(), expected_fence=v3.read_fence
        )
        == v3
    )
    with pytest.raises(snapshot_consumer.SnapshotConsumerCompatibilityError):
        snapshot_consumer.prepare_growth_snapshot_for_consumer(v3, expected_fence=v3.read_fence)
    with pytest.raises(snapshot_consumer.SnapshotConsumerCompatibilityError):
        snapshot_consumer.prepare_growth_snapshot_v3_for_consumer(v2, expected_fence=v2.read_fence)


def test_d2_overlay_is_lossless_and_cannot_recompute_trend() -> None:
    snapshot = _project(_window_batches(1, 4))
    payloads = snapshot_consumer.growth_trend_lineage_payloads(snapshot)

    representable = tuple(
        item
        for item in snapshot.metric_assessments
        if item.trend.reliability is not MetricReliability.UNKNOWN
    )
    assert len(payloads) == len(representable)
    payload_by_identity = {item["metric_identity_sha256"]: item for item in payloads}
    for assessment in representable:
        payload = payload_by_identity[assessment.metric_identity_sha256]
        trend = assessment.trend
        assert payload["source_snapshot_schema_version"] == "3.0"
        assert payload["metric_identity_sha256"] == assessment.metric_identity_sha256
        assert payload["trend_direction"] == trend.direction.value
        assert payload["trend_support_status"] == trend.support_status.value
        assert payload["reliability"] == trend.reliability.value
        assert payload["evidence_ref_ids"] == tuple(
            sorted(trend.previous_evidence_ref_ids + trend.current_evidence_ref_ids)
        )
        assert set(payload) == {
            "schema_version",
            "lineage_id",
            "source_snapshot_schema_version",
            "metric_code",
            "metric_identity_sha256",
            "trend_direction",
            "trend_support_status",
            "reliability",
            "trend_policy_version",
            "evidence_ref_ids",
        }
