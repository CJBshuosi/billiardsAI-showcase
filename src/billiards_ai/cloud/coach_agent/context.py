"""Immutable Coach Agent context values built from an explicit frozen source bundle.

This module deliberately contains no repository adapter. Story 8.4 must provide the
versioned snapshot authority and atomic read fence before production wiring is safe.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from billiards_ai.cloud.training.catalog import (
    DrillCatalog,
    DrillDefinition,
    DrillEligibilityProfile,
    list_eligible_drills,
)
from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.growth import (
    GrowthEvidenceSnapshotAny,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
    GrowthSnapshotReadFence,
    GrowthSnapshotReadFenceV3,
)

_SENSITIVE_KEY = re.compile(
    r"(?i)(?:^|_)(?:openid|open_id|name|full_name|phone|mobile|email|jwt|"
    r"auth_token|access_token|refresh_token|prompt|provider|database|db_row|sql|"
    r"raw_observation|trajectory|screenshot|media_(?:url|uri|path|locator)|"
    r"video_(?:url|uri|path|locator))(?:$|_)"
)
_SENSITIVE_VALUE = (
    re.compile(
        r"(?i)(?:https?|s3|gs|file|postgres(?:ql)?|mysql|mariadb|"
        r"mongodb(?:\+srv)?|redis|sqlite)://|www\."
    ),
    re.compile(r"(?i)\b(?:bearer\s+[a-z0-9._-]{8,}|sk-[a-z0-9_-]{8,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?<![\w-])(?:\+\d[\d ()-]{8,}\d|\d{10,15})(?![\w-])"),
    re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|client[_-]?secret|"
        r"private[_-]?key)\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)(?:raw[_ -]?observation|screenshot)"),
)


class _SourceTraversalError(ValueError):
    pass


class CoachTaskType(StrEnum):
    """Supported context projections; workflow execution is intentionally out of scope."""

    GROWTH_ASSESSMENT = "GROWTH_ASSESSMENT"
    TRAINING_PLAN = "TRAINING_PLAN"


class CoachContextErrorCode(StrEnum):
    INVALID_SOURCE_BUNDLE = "INVALID_SOURCE_BUNDLE"
    INVALID_CONTENT_HASH = "INVALID_CONTENT_HASH"
    NON_CANONICAL_ORDER = "NON_CANONICAL_ORDER"
    READ_FENCE_MISMATCH = "READ_FENCE_MISMATCH"
    CROSS_USER_LINEAGE = "CROSS_USER_LINEAGE"
    SENSITIVE_CONTENT = "SENSITIVE_CONTENT"
    DANGLING_EVIDENCE_REFERENCE = "DANGLING_EVIDENCE_REFERENCE"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    TASK_SOURCE_MISMATCH = "TASK_SOURCE_MISMATCH"
    SOURCE_FROM_FUTURE = "SOURCE_FROM_FUTURE"
    SNAPSHOT_VERSION_NOT_FOUND = "SNAPSHOT_VERSION_NOT_FOUND"
    READ_FENCE_CONFLICT = "READ_FENCE_CONFLICT"


class CoachContextError(ValueError):
    """Stable fail-closed error that never embeds source payloads."""

    def __init__(self, code: CoachContextErrorCode):
        self.code = code
        super().__init__(code.value)


class _SemanticError(ValueError):
    def __init__(self, code: CoachContextErrorCode, message: str):
        self.code = code
        super().__init__(message)


def _stored_values(value: object, active: set[int] | None = None) -> object:
    """Recover stored values recursively so model_construct/copy cannot hide invalid data."""

    active = set() if active is None else active
    if not isinstance(value, (BaseModel, tuple, list, dict)):
        return value
    identity = id(value)
    if identity in active:
        raise _SourceTraversalError("recursive source container")
    active.add(identity)
    try:
        if isinstance(value, BaseModel):
            values = dict(vars(value))
            if value.__pydantic_extra__:
                values.update(value.__pydantic_extra__)
            return {key: _stored_values(item, active) for key, item in values.items()}
        if isinstance(value, tuple):
            return tuple(_stored_values(item, active) for item in value)
        if isinstance(value, list):
            return [_stored_values(item, active) for item in value]
        assert isinstance(value, dict)
        return {key: _stored_values(item, active) for key, item in value.items()}
    finally:
        active.remove(identity)


def _contains_non_finite_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite_number(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_non_finite_number(item) for item in value)
    return False


def _contains_sensitive_content(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and _SENSITIVE_KEY.search(key))
            or _contains_sensitive_content(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_sensitive_content(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _SENSITIVE_VALUE)
    return False


def _deterministic_uuid4(content_sha256: str) -> UUID:
    raw = bytearray(bytes.fromhex(content_sha256)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


class CoachContextModel(CoreContractModel):
    """Strict value base that also revalidates copied and nested instances."""

    model_config = ConfigDict(revalidate_instances="always")

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_values(copied), strict=True)


class CapabilityManifest(CoachContextModel):
    schema_version: Literal["1.0"] = "1.0"
    manifest_version: NonEmptyStr
    user_id: UUID4
    user_level: StrictInt = Field(ge=1)
    acquired_skill_codes: tuple[NonEmptyStr, ...] = ()
    available_capabilities: tuple[NonEmptyStr, ...] = ()
    available_equipment_codes: tuple[NonEmptyStr, ...] = ()
    active_contraindication_codes: tuple[NonEmptyStr, ...] = ()

    @field_validator(
        "acquired_skill_codes",
        "available_capabilities",
        "available_equipment_codes",
        "active_contraindication_codes",
    )
    @classmethod
    def code_sets_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("code tuples must be sorted and unique")
        return value

    def eligibility_profile(self) -> DrillEligibilityProfile:
        return DrillEligibilityProfile(
            user_level=self.user_level,
            acquired_skill_codes=self.acquired_skill_codes,
            available_capabilities=self.available_capabilities,
            available_equipment_codes=self.available_equipment_codes,
            active_contraindication_codes=self.active_contraindication_codes,
        )


class CoachSourceReadFence(CoachContextModel):
    """Caller-supplied lineage for one asserted atomic authority read."""

    schema_version: Literal["1.0"] = "1.0"
    fence_id: NonEmptyStr
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    history_revision: NonEmptyStr
    history_content_sha256: Sha256
    catalog_version: NonEmptyStr | None = None
    capability_manifest_version: NonEmptyStr


class CoachSourceReadFenceV2(CoachSourceReadFence):
    """Story 8.4 exact snapshot and auxiliary authority lineage."""

    growth_snapshot_read_fence: GrowthSnapshotReadFence
    authority_revision: NonEmptyStr

    @classmethod
    def build_v2(
        cls,
        *,
        snapshot_read_fence: GrowthSnapshotReadFence,
        authority_revision: str,
        history_revision: str,
        history_content_sha256: str,
        catalog_version: str | None,
        capability_manifest_version: str,
    ) -> CoachSourceReadFenceV2:
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "snapshot_id": snapshot_read_fence.snapshot_id,
            "snapshot_version": snapshot_read_fence.version,
            "history_revision": history_revision,
            "history_content_sha256": history_content_sha256,
            "catalog_version": catalog_version,
            "capability_manifest_version": capability_manifest_version,
            "growth_snapshot_read_fence": snapshot_read_fence,
            "authority_revision": authority_revision,
        }
        return cls(fence_id=canonical_sha256(payload), **payload)


class CoachSourceReadFenceV3(CoachSourceReadFence):
    """Story 8.7 exact V3 snapshot and auxiliary authority lineage."""

    growth_snapshot_read_fence: GrowthSnapshotReadFenceV3
    authority_revision: NonEmptyStr

    @classmethod
    def build_v3(
        cls,
        *,
        snapshot_read_fence: GrowthSnapshotReadFenceV3,
        authority_revision: str,
        history_revision: str,
        history_content_sha256: str,
        catalog_version: str | None,
        capability_manifest_version: str,
    ) -> CoachSourceReadFenceV3:
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "snapshot_id": snapshot_read_fence.snapshot_id,
            "snapshot_version": snapshot_read_fence.version,
            "history_revision": history_revision,
            "history_content_sha256": history_content_sha256,
            "catalog_version": catalog_version,
            "capability_manifest_version": capability_manifest_version,
            "growth_snapshot_read_fence": snapshot_read_fence,
            "authority_revision": authority_revision,
        }
        return cls(fence_id=canonical_sha256(payload), **payload)


class AssessmentHistoryPointer(CoachContextModel):
    schema_version: Literal["1.0"] = "1.0"
    assessment_id: UUID4
    user_id: UUID4
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    revision: StrictInt = Field(ge=1)
    created_at: datetime
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)


class PlanHistoryPointer(CoachContextModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: UUID4
    user_id: UUID4
    plan_version: StrictInt = Field(ge=1)
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    created_at: datetime
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)


class PlanOutcomeHistoryPointer(CoachContextModel):
    schema_version: Literal["1.0"] = "1.0"
    outcome_id: UUID4
    user_id: UUID4
    plan_id: UUID4
    plan_version: StrictInt = Field(ge=1)
    completed_at: datetime
    content_sha256: Sha256

    _completed_at_utc = field_validator("completed_at")(require_utc)


class CoachContextSourceBundle(CoachContextModel):
    """Explicit immutable values read elsewhere; not proof that the read was atomic."""

    schema_version: Literal["1.0"] = "1.0"
    task_type: CoachTaskType
    user_id: UUID4
    as_of: datetime
    read_fence: CoachSourceReadFenceV3 | CoachSourceReadFenceV2 | CoachSourceReadFence
    growth_snapshot: GrowthEvidenceSnapshotAny
    prior_assessments: tuple[AssessmentHistoryPointer, ...] = ()
    prior_plans: tuple[PlanHistoryPointer, ...] = ()
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...] = ()
    drill_catalog: DrillCatalog | None = None
    drill_profile: DrillEligibilityProfile | None = None
    capability_manifest: CapabilityManifest

    _as_of_utc = field_validator("as_of")(require_utc)


class CoachAgentContext(CoachContextModel):
    schema_version: Literal["1.0"] = "1.0"
    context_id: UUID4
    task_type: CoachTaskType
    user_id: UUID4
    as_of: datetime
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    growth_snapshot: GrowthEvidenceSnapshotAny
    prior_assessments: tuple[AssessmentHistoryPointer, ...] = ()
    prior_plans: tuple[PlanHistoryPointer, ...] = ()
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...] = ()
    available_drills: tuple[DrillDefinition, ...] = ()
    catalog_version: NonEmptyStr | None = None
    capability_manifest: CapabilityManifest
    read_fence: CoachSourceReadFenceV3 | CoachSourceReadFenceV2 | CoachSourceReadFence
    context_sha256: Sha256

    _as_of_utc = field_validator("as_of")(require_utc)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Self:
        try:
            raw = _stored_values(obj)
        except (RecursionError, _SourceTraversalError) as error:
            raise ValueError("context contains recursive values") from error
        return super().model_validate(raw, **kwargs)

    @model_validator(mode="after")
    def identity_hash_and_snapshot_binding_are_exact(self) -> CoachAgentContext:
        raw = _stored_values(self)
        if _contains_non_finite_number(raw):
            raise ValueError("context contains non-finite numbers")
        if _contains_sensitive_content(raw):
            raise ValueError("context contains sensitive content")
        _validate_context_semantics(self)
        expected_hash = canonical_sha256(
            self.model_dump(mode="json", exclude={"context_id", "context_sha256"})
        )
        if self.context_sha256 != expected_hash:
            raise ValueError("context hash does not match canonical content")
        if self.context_id != _deterministic_uuid4(expected_hash):
            raise ValueError("context id does not match canonical content hash")
        return self


def _semantic_failure(code: CoachContextErrorCode, message: str) -> None:
    raise _SemanticError(code, message)


def _projected_history(
    task_type: CoachTaskType,
    prior_assessments: tuple[AssessmentHistoryPointer, ...],
    prior_plans: tuple[PlanHistoryPointer, ...],
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...],
) -> tuple[
    tuple[AssessmentHistoryPointer, ...],
    tuple[PlanHistoryPointer, ...],
    tuple[PlanOutcomeHistoryPointer, ...],
]:
    if task_type is CoachTaskType.GROWTH_ASSESSMENT:
        return prior_assessments, (), ()
    return prior_assessments, prior_plans, plan_outcomes


def coach_history_content_sha256(
    task_type: CoachTaskType,
    prior_assessments: tuple[AssessmentHistoryPointer, ...],
    prior_plans: tuple[PlanHistoryPointer, ...],
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...],
) -> str:
    assessments, plans, outcomes = _projected_history(
        task_type, prior_assessments, prior_plans, plan_outcomes
    )
    return canonical_sha256(
        {
            "prior_assessments": tuple(item.model_dump(mode="json") for item in assessments),
            "prior_plans": tuple(item.model_dump(mode="json") for item in plans),
            "plan_outcomes": tuple(item.model_dump(mode="json") for item in outcomes),
        }
    )


def _validate_snapshot_semantics(
    snapshot: GrowthEvidenceSnapshotAny,
    *,
    as_of: datetime,
) -> None:
    if isinstance(snapshot, (GrowthEvidenceSnapshotV2, GrowthEvidenceSnapshotV3)):
        if snapshot.created_at > as_of:
            _semantic_failure(
                CoachContextErrorCode.SOURCE_FROM_FUTURE,
                "growth snapshot is later than context as_of",
            )
        return
    expected_hash = canonical_sha256(snapshot.model_dump(mode="json", exclude={"content_sha256"}))
    if snapshot.content_sha256 != expected_hash:
        _semantic_failure(
            CoachContextErrorCode.INVALID_CONTENT_HASH,
            "growth snapshot content hash is invalid",
        )
    if snapshot.created_at > as_of:
        _semantic_failure(
            CoachContextErrorCode.SOURCE_FROM_FUTURE,
            "growth snapshot is later than context as_of",
        )
    if snapshot.source_batch_ids != tuple(sorted(snapshot.source_batch_ids, key=str)):
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "growth snapshot source batches are not canonical",
        )
    if snapshot.metrics != tuple(sorted(snapshot.metrics, key=lambda item: item.metric_code)):
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "growth snapshot metrics are not canonical",
        )
    expected_refs = tuple(
        sorted(
            snapshot.evidence_refs,
            key=lambda item: (item.metric_code, str(item.evidence_ref_id)),
        )
    )
    if snapshot.evidence_refs != expected_refs:
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "growth snapshot evidence references are not canonical",
        )
    for item in snapshot.evidence_refs:
        if item.shot_ids != tuple(sorted(set(item.shot_ids), key=str)):
            _semantic_failure(
                CoachContextErrorCode.NON_CANONICAL_ORDER,
                "evidence reference shot ids must be canonical and unique",
            )
    metric_codes = {item.metric_code for item in snapshot.metrics}
    if any(item.metric_code not in metric_codes for item in snapshot.evidence_refs):
        _semantic_failure(
            CoachContextErrorCode.DANGLING_EVIDENCE_REFERENCE,
            "evidence reference points to a missing metric",
        )
    plan_keys = tuple((item.plan_id, item.plan_version) for item in snapshot.plan_history)
    expected_plan_history = tuple(
        sorted(
            snapshot.plan_history,
            key=lambda item: (str(item.plan_id), item.plan_version),
        )
    )
    if len(plan_keys) != len(set(plan_keys)) or snapshot.plan_history != expected_plan_history:
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "growth snapshot plan history must be canonical and unique",
        )
    if any(
        timestamp is not None and timestamp > as_of
        for item in snapshot.plan_history
        for timestamp in (item.started_at, item.completed_at)
    ):
        _semantic_failure(
            CoachContextErrorCode.SOURCE_FROM_FUTURE,
            "growth snapshot plan history is later than context as_of",
        )


def _validate_history_semantics(
    *,
    task_type: CoachTaskType,
    user_id: UUID,
    snapshot: GrowthEvidenceSnapshotAny,
    as_of: datetime,
    prior_assessments: tuple[AssessmentHistoryPointer, ...],
    prior_plans: tuple[PlanHistoryPointer, ...],
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...],
    read_fence: CoachSourceReadFenceV3 | CoachSourceReadFenceV2 | CoachSourceReadFence,
) -> None:
    histories: tuple[object, ...] = (
        *prior_assessments,
        *prior_plans,
        *plan_outcomes,
    )
    if any(item.user_id != user_id for item in histories):
        _semantic_failure(
            CoachContextErrorCode.CROSS_USER_LINEAGE,
            "context history contains cross-user lineage",
        )
    if any(
        item.snapshot_id != snapshot.snapshot_id or item.snapshot_version != snapshot.version
        for item in (*prior_assessments, *prior_plans)
    ):
        _semantic_failure(
            CoachContextErrorCode.READ_FENCE_MISMATCH,
            "context history snapshot does not match read fence",
        )
    if (
        any(item.created_at > as_of for item in prior_assessments)
        or any(item.created_at > as_of for item in prior_plans)
        or any(item.completed_at > as_of for item in plan_outcomes)
    ):
        _semantic_failure(
            CoachContextErrorCode.SOURCE_FROM_FUTURE,
            "context history is later than context as_of",
        )
    assessment_keys = tuple((item.assessment_id, item.revision) for item in prior_assessments)
    plan_keys = tuple((item.plan_id, item.plan_version) for item in prior_plans)
    outcome_keys = tuple(item.outcome_id for item in plan_outcomes)
    if (
        len(assessment_keys) != len(set(assessment_keys))
        or len(plan_keys) != len(set(plan_keys))
        or len(outcome_keys) != len(set(outcome_keys))
    ):
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "context history logical keys must be unique",
        )
    if (
        prior_assessments
        != tuple(
            sorted(
                prior_assessments,
                key=lambda item: (item.created_at, str(item.assessment_id), item.revision),
            )
        )
        or prior_plans
        != tuple(
            sorted(
                prior_plans,
                key=lambda item: (item.created_at, str(item.plan_id), item.plan_version),
            )
        )
        or plan_outcomes
        != tuple(sorted(plan_outcomes, key=lambda item: (item.completed_at, str(item.outcome_id))))
    ):
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "context history collections are not canonical",
        )
    available_assessments = set(assessment_keys)
    if any(
        (item.assessment_id, item.assessment_revision) not in available_assessments
        for item in prior_plans
    ):
        _semantic_failure(
            CoachContextErrorCode.READ_FENCE_MISMATCH,
            "plan history does not reference an exact assessment in the read fence",
        )
    available_plans = set(plan_keys)
    if any((item.plan_id, item.plan_version) not in available_plans for item in plan_outcomes):
        _semantic_failure(
            CoachContextErrorCode.READ_FENCE_MISMATCH,
            "outcome history does not reference an exact plan in the read fence",
        )
    if read_fence.history_content_sha256 != coach_history_content_sha256(
        task_type, prior_assessments, prior_plans, plan_outcomes
    ):
        _semantic_failure(
            CoachContextErrorCode.READ_FENCE_MISMATCH,
            "context history content does not match read fence",
        )


def _validate_read_fence(
    *,
    read_fence: CoachSourceReadFenceV3 | CoachSourceReadFenceV2 | CoachSourceReadFence,
    snapshot: GrowthEvidenceSnapshotAny,
    catalog_version: str | None,
    capability_manifest: CapabilityManifest,
) -> None:
    if (
        read_fence.snapshot_id != snapshot.snapshot_id
        or read_fence.snapshot_version != snapshot.version
        or read_fence.catalog_version != catalog_version
        or read_fence.capability_manifest_version != capability_manifest.manifest_version
    ):
        _semantic_failure(
            CoachContextErrorCode.READ_FENCE_MISMATCH,
            "context read fence does not match source lineage",
        )
    if isinstance(snapshot, GrowthEvidenceSnapshotV3):
        if (
            not isinstance(read_fence, CoachSourceReadFenceV3)
            or read_fence.growth_snapshot_read_fence != snapshot.read_fence
            or read_fence.fence_id
            != canonical_sha256(read_fence.model_dump(mode="json", exclude={"fence_id"}))
        ):
            _semantic_failure(
                CoachContextErrorCode.READ_FENCE_MISMATCH,
                "context v3 read fence does not match source lineage",
            )
    elif isinstance(snapshot, GrowthEvidenceSnapshotV2):
        if (
            not isinstance(read_fence, CoachSourceReadFenceV2)
            or isinstance(read_fence, CoachSourceReadFenceV3)
            or read_fence.growth_snapshot_read_fence != snapshot.read_fence
            or read_fence.fence_id
            != canonical_sha256(read_fence.model_dump(mode="json", exclude={"fence_id"}))
        ):
            _semantic_failure(
                CoachContextErrorCode.READ_FENCE_MISMATCH,
                "context v2 read fence does not match source lineage",
            )
    elif isinstance(read_fence, (CoachSourceReadFenceV2, CoachSourceReadFenceV3)):
        _semantic_failure(
            CoachContextErrorCode.READ_FENCE_MISMATCH,
            "context v1 cannot carry v2 authority lineage",
        )


def _validate_available_drills(
    *,
    task_type: CoachTaskType,
    as_of: datetime,
    catalog_version: str | None,
    available_drills: tuple[DrillDefinition, ...],
    capability_manifest: CapabilityManifest,
) -> None:
    if task_type is CoachTaskType.GROWTH_ASSESSMENT:
        if catalog_version is not None or available_drills:
            _semantic_failure(
                CoachContextErrorCode.TASK_SOURCE_MISMATCH,
                "growth assessment context contains training-only source data",
            )
        return
    if catalog_version is None:
        _semantic_failure(
            CoachContextErrorCode.TASK_SOURCE_MISMATCH,
            "training plan context requires a catalog version",
        )
    if available_drills != tuple(
        sorted(
            available_drills,
            key=lambda item: (item.drill_id, item.revision, item.content_sha256),
        )
    ) or len({item.drill_id for item in available_drills}) != len(available_drills):
        _semantic_failure(
            CoachContextErrorCode.NON_CANONICAL_ORDER,
            "available drills must be canonical unique current heads",
        )
    profile = capability_manifest.eligibility_profile()
    if any(item.eligibility_violations(profile) for item in available_drills):
        _semantic_failure(
            CoachContextErrorCode.CAPABILITY_MISMATCH,
            "available drill is not eligible for the capability manifest",
        )
    if any(item.review_lineage.reviewed_at > as_of for item in available_drills):
        _semantic_failure(
            CoachContextErrorCode.SOURCE_FROM_FUTURE,
            "available drill review lineage is later than context as_of",
        )


def _validate_context_semantics(context: CoachAgentContext) -> None:
    snapshot = context.growth_snapshot
    if (
        context.snapshot_id != snapshot.snapshot_id
        or context.snapshot_version != snapshot.version
        or context.user_id != snapshot.user_id
        or context.capability_manifest.user_id != context.user_id
    ):
        _semantic_failure(
            CoachContextErrorCode.CROSS_USER_LINEAGE,
            "context snapshot or capability contains cross-user lineage",
        )
    _validate_snapshot_semantics(snapshot, as_of=context.as_of)
    _validate_history_semantics(
        task_type=context.task_type,
        user_id=context.user_id,
        snapshot=snapshot,
        as_of=context.as_of,
        prior_assessments=context.prior_assessments,
        prior_plans=context.prior_plans,
        plan_outcomes=context.plan_outcomes,
        read_fence=context.read_fence,
    )
    _validate_read_fence(
        read_fence=context.read_fence,
        snapshot=snapshot,
        catalog_version=context.catalog_version,
        capability_manifest=context.capability_manifest,
    )
    if context.task_type is CoachTaskType.GROWTH_ASSESSMENT and (
        context.prior_plans or context.plan_outcomes
    ):
        _semantic_failure(
            CoachContextErrorCode.TASK_SOURCE_MISMATCH,
            "growth assessment context contains training-only history",
        )
    _validate_available_drills(
        task_type=context.task_type,
        as_of=context.as_of,
        catalog_version=context.catalog_version,
        available_drills=context.available_drills,
        capability_manifest=context.capability_manifest,
    )


def _validate_source_semantics(bundle: CoachContextSourceBundle) -> None:
    snapshot = bundle.growth_snapshot
    catalog_version = None if bundle.drill_catalog is None else bundle.drill_catalog.catalog_version
    if snapshot.user_id != bundle.user_id or bundle.capability_manifest.user_id != bundle.user_id:
        _semantic_failure(
            CoachContextErrorCode.CROSS_USER_LINEAGE,
            "source snapshot or capability contains cross-user lineage",
        )
    _validate_snapshot_semantics(snapshot, as_of=bundle.as_of)
    _validate_history_semantics(
        task_type=bundle.task_type,
        user_id=bundle.user_id,
        snapshot=snapshot,
        as_of=bundle.as_of,
        prior_assessments=bundle.prior_assessments,
        prior_plans=bundle.prior_plans,
        plan_outcomes=bundle.plan_outcomes,
        read_fence=bundle.read_fence,
    )
    _validate_read_fence(
        read_fence=bundle.read_fence,
        snapshot=snapshot,
        catalog_version=catalog_version,
        capability_manifest=bundle.capability_manifest,
    )


def _available_drills(bundle: CoachContextSourceBundle) -> tuple[DrillDefinition, ...]:
    catalog = bundle.drill_catalog
    profile = bundle.drill_profile
    if bundle.task_type is CoachTaskType.GROWTH_ASSESSMENT:
        if catalog is not None or profile is not None:
            _semantic_failure(
                CoachContextErrorCode.TASK_SOURCE_MISMATCH,
                "growth assessment source contains training-only catalog data",
            )
        return ()
    if catalog is None or profile is None:
        _semantic_failure(
            CoachContextErrorCode.TASK_SOURCE_MISMATCH,
            "training plan source requires catalog and eligibility profile",
        )
    if catalog.created_at > bundle.as_of or any(
        item.review_lineage.reviewed_at > bundle.as_of
        for item in (catalog.revision_history or catalog.drills)
    ):
        _semantic_failure(
            CoachContextErrorCode.SOURCE_FROM_FUTURE,
            "drill catalog or review lineage is later than context as_of",
        )
    if profile != bundle.capability_manifest.eligibility_profile():
        _semantic_failure(
            CoachContextErrorCode.CAPABILITY_MISMATCH,
            "drill profile does not match capability manifest",
        )
    return list_eligible_drills(catalog=catalog, profile=profile)


def build_coach_context(source: CoachContextSourceBundle) -> CoachAgentContext:
    """Build one deterministic context without I/O, clocks, randomness, or repositories."""

    try:
        raw_source = _stored_values(source)
        if _contains_non_finite_number(raw_source):
            raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE)
        if _contains_sensitive_content(raw_source):
            raise CoachContextError(CoachContextErrorCode.SENSITIVE_CONTENT)
        bundle = CoachContextSourceBundle.model_validate(raw_source, strict=True)
        _validate_source_semantics(bundle)
        available_drills = _available_drills(bundle)
    except CoachContextError:
        raise
    except _SemanticError as error:
        raise CoachContextError(error.code) from None
    except (RecursionError, _SourceTraversalError, ValidationError):
        raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None

    prior_assessments, prior_plans, plan_outcomes = _projected_history(
        bundle.task_type,
        bundle.prior_assessments,
        bundle.prior_plans,
        bundle.plan_outcomes,
    )
    catalog_version = None if bundle.drill_catalog is None else bundle.drill_catalog.catalog_version
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "task_type": bundle.task_type,
        "user_id": bundle.user_id,
        "as_of": bundle.as_of,
        "snapshot_id": bundle.growth_snapshot.snapshot_id,
        "snapshot_version": bundle.growth_snapshot.version,
        "growth_snapshot": bundle.growth_snapshot,
        "prior_assessments": prior_assessments,
        "prior_plans": prior_plans,
        "plan_outcomes": plan_outcomes,
        "available_drills": available_drills,
        "catalog_version": catalog_version,
        "capability_manifest": bundle.capability_manifest,
        "read_fence": bundle.read_fence,
    }
    context_sha256 = canonical_sha256(payload)
    context_id = _deterministic_uuid4(context_sha256)
    try:
        return CoachAgentContext.model_validate(
            payload | {"context_id": context_id, "context_sha256": context_sha256},
            strict=True,
        )
    except ValidationError:
        raise CoachContextError(CoachContextErrorCode.INVALID_SOURCE_BUNDLE) from None
