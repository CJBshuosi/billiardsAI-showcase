"""Exact-version production orchestration for immutable Coach Context values."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import UUID4, StrictInt, TypeAdapter, ValidationError, field_validator

from billiards_ai.cloud.coach_agent.context import (
    AssessmentHistoryPointer,
    CapabilityManifest,
    CoachAgentContext,
    CoachContextError,
    CoachContextErrorCode,
    CoachContextModel,
    CoachContextSourceBundle,
    CoachSourceReadFenceV2,
    CoachSourceReadFenceV3,
    CoachTaskType,
    PlanHistoryPointer,
    PlanOutcomeHistoryPointer,
    _stored_values,
    build_coach_context,
    coach_history_content_sha256,
)
from billiards_ai.cloud.growth.snapshot_consumer import (
    SnapshotConsumerCompatibilityError,
    prepare_growth_snapshot_for_consumer,
    prepare_growth_snapshot_v3_for_consumer,
)
from billiards_ai.cloud.growth.snapshot_repository import (
    GrowthSnapshotRepository,
    SnapshotNotFound,
)
from billiards_ai.cloud.training.catalog import DrillCatalog, DrillEligibilityProfile
from billiards_ai.shared.models.events import NonEmptyStr, require_utc
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION,
    SNAPSHOT_POLICY_VERSION_V3,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
)

_UUID4 = TypeAdapter(UUID4)


class CoachContextAuxiliaryRead(CoachContextModel):
    """One frozen history/catalog/capability authority read."""

    user_id: UUID4
    as_of: datetime
    authority_revision: NonEmptyStr
    history_revision: NonEmptyStr
    prior_assessments: tuple[AssessmentHistoryPointer, ...] = ()
    prior_plans: tuple[PlanHistoryPointer, ...] = ()
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...] = ()
    drill_catalog: DrillCatalog | None = None
    drill_profile: DrillEligibilityProfile | None = None
    capability_manifest: CapabilityManifest

    _as_of_utc = field_validator("as_of")(require_utc)


class CoachContextAuxiliaryAuthority(Protocol):
    def read(
        self,
        user_id: UUID,
        task_type: CoachTaskType,
    ) -> CoachContextAuxiliaryRead: ...

    def get_revision(self, user_id: UUID, task_type: CoachTaskType) -> str: ...


class RepositoryCoachContextPort:
    """Build Context from Story 8.4 exact versions and one auxiliary authority."""

    def __init__(
        self,
        *,
        snapshot_repository: GrowthSnapshotRepository,
        auxiliary_authority: CoachContextAuxiliaryAuthority,
        max_attempts: StrictInt = 3,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive strict integer")
        self._snapshot_repository = snapshot_repository
        self._auxiliary_authority = auxiliary_authority
        self._max_attempts = max_attempts
        if snapshot_policy_version not in {
            SNAPSHOT_POLICY_VERSION,
            SNAPSHOT_POLICY_VERSION_V3,
        }:
            raise ValueError("unsupported snapshot policy version")
        self._snapshot_policy_version = snapshot_policy_version

    def build_context(
        self,
        user_id: UUID,
        snapshot_version: int,
        task_type: CoachTaskType,
    ) -> CoachAgentContext:
        try:
            exact_user_id = _UUID4.validate_python(user_id, strict=True)
        except ValidationError:
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
        if (
            isinstance(snapshot_version, bool)
            or not isinstance(snapshot_version, int)
            or snapshot_version < 1
            or not isinstance(task_type, CoachTaskType)
        ):
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None

        for _ in range(self._max_attempts):
            context, authority_revision = self._build_attempt(
                exact_user_id,
                snapshot_version,
                task_type,
            )
            try:
                current_revision = self._auxiliary_authority.get_revision(
                    exact_user_id,
                    task_type,
                )
            except Exception:
                raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
            if not isinstance(current_revision, str) or not current_revision.strip():
                raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
            if current_revision == authority_revision:
                return context
        raise CoachContextError(CoachContextErrorCode.READ_FENCE_CONFLICT) from None

    def _build_attempt(
        self,
        user_id: UUID,
        snapshot_version: int,
        task_type: CoachTaskType,
    ) -> tuple[CoachAgentContext, str]:
        try:
            stored_snapshot = self._snapshot_repository.get_version(
                user_id,
                snapshot_version,
                snapshot_policy_version=self._snapshot_policy_version,
            )
        except SnapshotNotFound:
            raise CoachContextError(CoachContextErrorCode.SNAPSHOT_VERSION_NOT_FOUND) from None
        except Exception:
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
        try:
            if stored_snapshot.snapshot_policy_version != self._snapshot_policy_version:
                raise SnapshotConsumerCompatibilityError(
                    "snapshot does not match the requested policy namespace"
                )
            if isinstance(stored_snapshot, GrowthEvidenceSnapshotV3):
                snapshot = prepare_growth_snapshot_v3_for_consumer(
                    stored_snapshot,
                    expected_fence=stored_snapshot.read_fence,
                )
            elif isinstance(stored_snapshot, GrowthEvidenceSnapshotV2):
                snapshot = prepare_growth_snapshot_for_consumer(
                    stored_snapshot,
                    expected_fence=stored_snapshot.read_fence,
                )
            else:
                raise SnapshotConsumerCompatibilityError("unsupported snapshot schema")
        except (
            AttributeError,
            RecursionError,
            SnapshotConsumerCompatibilityError,
            TypeError,
            ValidationError,
            ValueError,
        ):
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
        if snapshot.user_id != user_id or snapshot.version != snapshot_version:
            raise CoachContextError(CoachContextErrorCode.CROSS_USER_LINEAGE) from None

        try:
            raw_auxiliary = self._auxiliary_authority.read(user_id, task_type)
            auxiliary = CoachContextAuxiliaryRead.model_validate(
                _stored_values(raw_auxiliary),
                strict=True,
            )
        except (AttributeError, TypeError, ValidationError, ValueError):
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
        except Exception:
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
        if auxiliary.user_id != user_id or auxiliary.capability_manifest.user_id != user_id:
            raise CoachContextError(CoachContextErrorCode.CROSS_USER_LINEAGE) from None

        history_content_sha256 = coach_history_content_sha256(
            task_type,
            auxiliary.prior_assessments,
            auxiliary.prior_plans,
            auxiliary.plan_outcomes,
        )
        catalog_version = (
            None if auxiliary.drill_catalog is None else auxiliary.drill_catalog.catalog_version
        )
        fence_factory = (
            CoachSourceReadFenceV3.build_v3
            if isinstance(snapshot, GrowthEvidenceSnapshotV3)
            else CoachSourceReadFenceV2.build_v2
        )
        read_fence = fence_factory(
            snapshot_read_fence=snapshot.read_fence,
            authority_revision=auxiliary.authority_revision,
            history_revision=auxiliary.history_revision,
            history_content_sha256=history_content_sha256,
            catalog_version=catalog_version,
            capability_manifest_version=auxiliary.capability_manifest.manifest_version,
        )
        source = CoachContextSourceBundle(
            task_type=task_type,
            user_id=user_id,
            as_of=auxiliary.as_of,
            read_fence=read_fence,
            growth_snapshot=snapshot,
            prior_assessments=auxiliary.prior_assessments,
            prior_plans=auxiliary.prior_plans,
            plan_outcomes=auxiliary.plan_outcomes,
            drill_catalog=auxiliary.drill_catalog,
            drill_profile=auxiliary.drill_profile,
            capability_manifest=auxiliary.capability_manifest,
        )
        return build_coach_context(source), auxiliary.authority_revision


__all__ = [
    "CoachContextAuxiliaryAuthority",
    "CoachContextAuxiliaryRead",
    "RepositoryCoachContextPort",
]
