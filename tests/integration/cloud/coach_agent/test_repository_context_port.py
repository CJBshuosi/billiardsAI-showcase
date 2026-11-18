"""Story 9.2 production acceptance over the real Story 8.4 snapshot authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from billiards_ai.cloud.coach_agent import (
    CapabilityManifest,
    CoachAgentContext,
    CoachContextAuxiliaryRead,
    CoachContextError,
    CoachContextErrorCode,
    CoachTaskType,
    RepositoryCoachContextPort,
)
from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot
from billiards_ai.cloud.growth.snapshot_repository import InMemoryGrowthSnapshotRepository
from billiards_ai.cloud.training.catalog import DrillCatalog
from billiards_ai.shared.models.growth import GrowthEvidenceSnapshotV2, GrowthSnapshotLineage
from tests.unit.cloud.coach_agent.test_context import _definition
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch, uid

AS_OF = datetime(2026, 8, 5, 12, 30, tzinfo=UTC)
USER_ID = uid(900)


def _manifest(*, user_id: UUID = USER_ID) -> CapabilityManifest:
    return CapabilityManifest(
        manifest_version="capability-2026-08-05-r1",
        user_id=user_id,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        available_capabilities=("TABLE_MM_COORDINATES",),
        available_equipment_codes=("STANDARD_TABLE",),
    )


def _auxiliary_read(
    *,
    authority_revision: str = "aux-r1",
    user_id: UUID = USER_ID,
    drill_catalog: DrillCatalog | None = None,
) -> CoachContextAuxiliaryRead:
    manifest = _manifest(user_id=user_id)
    return CoachContextAuxiliaryRead(
        user_id=user_id,
        as_of=AS_OF,
        authority_revision=authority_revision,
        history_revision="history-r1",
        drill_catalog=drill_catalog,
        drill_profile=None if drill_catalog is None else manifest.eligibility_profile(),
        capability_manifest=manifest,
    )


@dataclass
class ScriptedAuxiliaryAuthority:
    reads: list[CoachContextAuxiliaryRead]
    current_revisions: list[str]
    read_count: int = 0

    def read(self, user_id: UUID, task_type: CoachTaskType) -> CoachContextAuxiliaryRead:
        del user_id, task_type
        index = min(self.read_count, len(self.reads) - 1)
        self.read_count += 1
        return self.reads[index]

    def get_revision(self, user_id: UUID, task_type: CoachTaskType) -> str:
        del user_id, task_type
        index = min(self.read_count - 1, len(self.current_revisions) - 1)
        return self.current_revisions[index]


def _append_snapshot(
    repository: InMemoryGrowthSnapshotRepository,
    *,
    version: int,
    previous_snapshot_id: UUID | None,
    batch_numbers: tuple[int, ...],
) -> GrowthEvidenceSnapshotV2:
    snapshot = project_growth_snapshot(
        admit(*(make_batch(number) for number in batch_numbers)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(
            version=version,
            previous_snapshot_id=previous_snapshot_id,
        ),
    )
    return repository.compare_and_append(
        snapshot,
        expected_previous_version=None if version == 1 else version - 1,
        idempotency_key=f"snapshot-v{version}",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(910_000 + version),
    ).snapshot


def test_exact_v2_repository_version_builds_round_trippable_deterministic_context() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    snapshot = _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    authority = ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"])
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=authority,
    )

    first = port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)
    second = port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert first == second
    assert first.growth_snapshot == snapshot
    assert first.growth_snapshot.schema_version == "2.0"
    assert first.read_fence.growth_snapshot_read_fence == snapshot.read_fence
    assert first.read_fence.authority_revision == "aux-r1"
    assert CoachAgentContext.model_validate_json(first.model_dump_json(), strict=True) == first


def test_exact_historical_context_is_unchanged_after_a_new_snapshot_arrives() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first_snapshot = _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    authority = ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"])
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=authority,
    )
    before = port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)
    canonical_before = before.model_dump_json()

    _append_snapshot(
        repository,
        version=2,
        previous_snapshot_id=first_snapshot.snapshot_id,
        batch_numbers=(1, 2),
    )
    after = port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert after == before
    assert after.model_dump_json() == canonical_before
    assert after.snapshot_version == 1


def test_training_context_binds_exact_catalog_version_and_capability_eligibility() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    drill = _definition(drill_id="v2-exact-catalog")
    catalog = DrillCatalog.build(
        catalog_version="catalog-v2-exact-r7",
        created_at=datetime(2026, 8, 5, 8, tzinfo=UTC),
        drills=(drill,),
    )
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=ScriptedAuxiliaryAuthority(
            [_auxiliary_read(drill_catalog=catalog)],
            ["aux-r1"],
        ),
    )

    context = port.build_context(USER_ID, 1, CoachTaskType.TRAINING_PLAN)

    assert context.available_drills == (drill,)
    assert context.catalog_version == "catalog-v2-exact-r7"
    assert context.read_fence.catalog_version == context.catalog_version
    assert context.capability_manifest == _manifest()


def test_auxiliary_revision_change_retries_the_entire_read() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    authority = ScriptedAuxiliaryAuthority(
        [
            _auxiliary_read(authority_revision="aux-r1"),
            _auxiliary_read(authority_revision="aux-r2"),
        ],
        ["aux-r2", "aux-r2"],
    )
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=authority,
        max_attempts=2,
    )

    context = port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert authority.read_count == 2
    assert context.read_fence.authority_revision == "aux-r2"


def test_auxiliary_revision_retry_exhaustion_fails_closed() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    authority = ScriptedAuxiliaryAuthority(
        [_auxiliary_read(authority_revision="aux-r1")],
        ["aux-r2"],
    )
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=authority,
        max_attempts=2,
    )

    with pytest.raises(CoachContextError) as exc:
        port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert authority.read_count == 2
    assert exc.value.code is CoachContextErrorCode.READ_FENCE_CONFLICT
    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    ("user_id", "version", "expected_code"),
    [
        (uid(900), 2, CoachContextErrorCode.SNAPSHOT_VERSION_NOT_FOUND),
        (uid(901), 1, CoachContextErrorCode.SNAPSHOT_VERSION_NOT_FOUND),
    ],
)
def test_missing_or_cross_user_version_never_falls_back_to_current(
    user_id: UUID,
    version: int,
    expected_code: CoachContextErrorCode,
) -> None:
    repository = InMemoryGrowthSnapshotRepository()
    _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=ScriptedAuxiliaryAuthority([_auxiliary_read()], ["aux-r1"]),
    )

    with pytest.raises(CoachContextError) as exc:
        port.build_context(user_id, version, CoachTaskType.GROWTH_ASSESSMENT)

    assert exc.value.code is expected_code


def test_cross_user_auxiliary_read_fails_closed_without_retrying_as_success() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    _append_snapshot(
        repository,
        version=1,
        previous_snapshot_id=None,
        batch_numbers=(1,),
    )
    authority = ScriptedAuxiliaryAuthority(
        [_auxiliary_read(user_id=uid(901))],
        ["aux-r1"],
    )
    port = RepositoryCoachContextPort(
        snapshot_repository=repository,
        auxiliary_authority=authority,
    )

    with pytest.raises(CoachContextError) as exc:
        port.build_context(uid(900), 1, CoachTaskType.GROWTH_ASSESSMENT)

    assert exc.value.code is CoachContextErrorCode.CROSS_USER_LINEAGE
