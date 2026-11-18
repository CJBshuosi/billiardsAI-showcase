from __future__ import annotations

import pytest

from billiards_ai.cloud.coach_agent import (
    CoachContextError,
    CoachContextErrorCode,
    CoachTaskType,
    RepositoryCoachContextPort,
)
from billiards_ai.cloud.growth.snapshot_repository import InMemoryGrowthSnapshotRepository
from billiards_ai.shared.models.growth import SNAPSHOT_POLICY_VERSION_V3, GrowthEvidenceSnapshotV3
from tests.integration.cloud.coach_agent.test_repository_context_port import (
    ScriptedAuxiliaryAuthority,
    _append_snapshot,
    _auxiliary_read,
)
from tests.unit.cloud.growth.snapshot_test_support import NOW, uid
from tests.unit.cloud.growth.test_snapshot_v3 import _project, _window_batches


def test_repository_context_port_reads_exact_v3_policy_and_fence() -> None:
    snapshot = _project(_window_batches(0, 5))
    repository = InMemoryGrowthSnapshotRepository()
    stored = repository.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="story-9-4-v3",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(980_001),
    ).snapshot
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"]),
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
    )

    context = port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert context.growth_snapshot == stored
    assert context.growth_snapshot.schema_version == "3.0"
    assert context.read_fence.growth_snapshot_read_fence == stored.read_fence
    assert (
        context.model_dump_json()
        == port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT).model_dump_json()
    )


def test_repository_v3_policy_never_falls_back_to_the_v2_namespace() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"]),
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
    )

    with pytest.raises(CoachContextError) as exc:
        port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert exc.value.code is CoachContextErrorCode.SNAPSHOT_VERSION_NOT_FOUND
    assert exc.value.__cause__ is None


def test_repository_port_rejects_a_value_from_the_wrong_policy_namespace() -> None:
    v2_repository = InMemoryGrowthSnapshotRepository()
    v2_snapshot = _append_snapshot(
        v2_repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )

    class WrongPolicyRepository:
        def get_version(self, user_id, version, *, snapshot_policy_version):
            del user_id, version, snapshot_policy_version
            return v2_snapshot

    port = RepositoryCoachContextPort(
        snapshot_repository=WrongPolicyRepository(),  # type: ignore[arg-type]
        auxiliary_authority=ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"]),
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
    )

    with pytest.raises(CoachContextError) as exc:
        port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE


def test_repository_port_maps_recursive_v3_construct_bypass_to_stable_error() -> None:
    snapshot = _project(_window_batches(0, 5))
    recursive_metrics: list[object] = []
    recursive_metrics.append(recursive_metrics)
    stored_values = dict(vars(snapshot))
    stored_values["metric_assessments"] = recursive_metrics
    malformed = GrowthEvidenceSnapshotV3.model_construct(**stored_values)

    class RecursiveRepository:
        def get_version(self, user_id, version, *, snapshot_policy_version):
            del user_id, version, snapshot_policy_version
            return malformed

    port = RepositoryCoachContextPort(
        snapshot_repository=RecursiveRepository(),  # type: ignore[arg-type]
        auxiliary_authority=ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"]),
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
    )

    with pytest.raises(CoachContextError) as exc:
        port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE
    assert exc.value.__cause__ is None
