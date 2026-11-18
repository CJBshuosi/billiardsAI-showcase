"""Story 8.4 RED tests for deterministic versioned growth snapshots."""

from __future__ import annotations

from copy import copy
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.growth.metric_projector import project_ability_metrics
from billiards_ai.cloud.growth.snapshot_projector import (
    SnapshotProjectionError,
    project_growth_snapshot,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.growth import (
    GrowthEvidenceSnapshot,
    GrowthEvidenceSnapshotV2,
    GrowthSnapshotLineage,
    GrowthSnapshotReadFence,
    growth_snapshot_content_payload,
    growth_snapshot_evidence_payload,
    growth_snapshot_id_from_content,
)
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch, uid


def test_v1_contract_stays_compatible_while_v2_is_strict_and_frozen() -> None:
    fixture = GrowthEvidenceSnapshot.model_validate_json(
        Path("tests/fixtures/core_intelligence/v1/growth-evidence-snapshot.json").read_text()
    )
    assert fixture.schema_version == "1.0"

    snapshot = project_growth_snapshot(
        admit(make_batch(1)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    assert isinstance(snapshot, GrowthEvidenceSnapshotV2)
    assert snapshot.schema_version == "2.0"
    assert snapshot.metric_projection.metrics
    with pytest.raises(ValidationError):
        GrowthEvidenceSnapshotV2.model_validate(
            snapshot.model_dump(mode="python") | {"coach_weakness": "invented"}
        )
    with pytest.raises(ValidationError):
        snapshot.version = 9  # type: ignore[misc]


def test_same_trusted_inputs_and_policy_are_byte_identical_in_any_order() -> None:
    admitted = admit(make_batch(1), make_batch(2))
    lineage = GrowthSnapshotLineage(version=1, previous_snapshot_id=None)

    forward = project_growth_snapshot(admitted, evidence_cutoff=NOW, lineage=lineage)
    reverse = project_growth_snapshot(
        tuple(reversed(admitted)), evidence_cutoff=NOW, lineage=lineage
    )

    assert forward.model_dump_json() == reverse.model_dump_json()
    assert forward.snapshot_id == reverse.snapshot_id
    assert forward.content_sha256 == reverse.content_sha256
    assert forward.evidence_sha256 == reverse.evidence_sha256
    assert tuple(item.batch_id for item in forward.source_batches) == tuple(
        sorted((item.batch_id for item in forward.source_batches), key=str)
    )


def test_snapshot_carries_complete_source_and_projection_audit() -> None:
    old = make_batch(1, session_number=101, review_revision=1, successful=True)
    latest = make_batch(1, session_number=101, review_revision=2, successful=False)
    outside = make_batch(3, approved_at=NOW - timedelta(days=366))
    admitted = admit(old, latest, outside)

    snapshot = project_growth_snapshot(
        admitted + (admitted[1],),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )

    assert snapshot.metric_projection.superseded_batch_count == 1
    assert snapshot.metric_projection.outside_window_batch_count == 1
    assert snapshot.metric_projection.replayed_batch_count == 1
    assert len(snapshot.source_batches) == 4
    reference = snapshot.source_batches[0]
    assert reference.admission_id
    assert reference.report_id
    assert reference.report_revision >= 1
    assert reference.review_revision >= 1
    assert reference.batch_content_sha256
    assert reference.source_envelope_ids
    assert snapshot.metric_projection_binding.source_bundle_sha256 == snapshot.source_bundle_sha256


def test_untrusted_or_cross_user_inputs_fail_closed() -> None:
    first, second = admit(make_batch(1), make_batch(2, user_number=901))
    dumped = type(first).model_validate(first.model_dump(mode="python"))
    lineage = GrowthSnapshotLineage(version=1, previous_snapshot_id=None)

    with pytest.raises(SnapshotProjectionError):
        project_growth_snapshot((dumped,), evidence_cutoff=NOW, lineage=lineage)
    with pytest.raises(SnapshotProjectionError):
        project_growth_snapshot((first, second), evidence_cutoff=NOW, lineage=lineage)


def test_shallow_copy_and_model_copy_cannot_transport_admission_proof() -> None:
    (accepted,) = admit(make_batch(1))
    copies = (
        copy(accepted),
        accepted.model_copy(),
        accepted.model_copy(update={"admitted_at": accepted.admitted_at}),
    )

    for transported in copies:
        assert transported is not accepted
        assert transported.is_trusted_admission is False
        with pytest.raises(SnapshotProjectionError):
            project_growth_snapshot(
                (transported,),
                evidence_cutoff=NOW,
                lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
            )


def test_stateful_tuple_is_materialized_once_for_sources_and_metric_projection() -> None:
    first = admit(make_batch(1, user_number=900, successful=True))
    second = admit(make_batch(2, user_number=901, successful=False))

    class StatefulTuple(tuple):
        def __new__(cls):
            return super().__new__(cls, first)

        def __init__(self):
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return iter(first if self.iteration_count == 1 else second)

    inputs = StatefulTuple()
    snapshot = project_growth_snapshot(
        inputs,
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )

    assert inputs.iteration_count == 1
    assert snapshot.user_id == first[0].batch.user_id
    assert snapshot.metric_projection == project_ability_metrics(first, evidence_cutoff=NOW)


def test_projection_from_another_source_bundle_cannot_be_rehashed_into_snapshot() -> None:
    source = project_growth_snapshot(
        admit(*(make_batch(number, successful=True) for number in range(1, 6))),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    other = project_growth_snapshot(
        admit(*(make_batch(number, user_number=901, successful=False) for number in range(11, 16))),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    placeholder = source.model_copy(update={"metric_projection": other.metric_projection})
    evidence_sha256 = canonical_sha256(growth_snapshot_evidence_payload(placeholder))
    placeholder = placeholder.model_copy(update={"evidence_sha256": evidence_sha256})
    content_sha256 = canonical_sha256(growth_snapshot_content_payload(placeholder))
    snapshot_id = growth_snapshot_id_from_content(content_sha256)
    fence = GrowthSnapshotReadFence(
        user_id=source.user_id,
        snapshot_id=snapshot_id,
        version=source.version,
        content_sha256=content_sha256,
        evidence_sha256=evidence_sha256,
        taxonomy_version=source.taxonomy_version,
        calculation_policy_version=source.calculation_policy_version,
        snapshot_policy_version=source.snapshot_policy_version,
    )

    with pytest.raises(ValidationError):
        GrowthEvidenceSnapshotV2.model_validate(
            placeholder.model_dump(mode="python")
            | {
                "snapshot_id": snapshot_id,
                "content_sha256": content_sha256,
                "read_fence": fence,
            },
            strict=True,
        )


def test_hash_identity_or_read_fence_tampering_fails_strict_revalidation() -> None:
    snapshot = project_growth_snapshot(
        admit(make_batch(1)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )

    for update in (
        {"content_sha256": "0" * 64},
        {"evidence_sha256": "1" * 64},
        {"snapshot_id": uid(999_999)},
        {"read_fence": snapshot.read_fence.model_copy(update={"version": 2})},
    ):
        with pytest.raises(ValidationError):
            GrowthEvidenceSnapshotV2.model_validate(
                snapshot.model_dump(mode="python") | update,
                strict=True,
            )
