"""Shared immutable drill catalog contracts, independent of cloud services."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)

_SKIP_CONTENT_HASH = object()
_CODE_SET_ADAPTER = TypeAdapter(
    tuple[NonEmptyStr, ...],
    config=ConfigDict(strict=True),
)
_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)(?:https?|s3|gs|file)://|www\."),
    re.compile(
        r"(?i)(?:^|[^a-z0-9])(?:openid|open_id|token|prompt|provider|"
        r"raw[_ -]?observation)(?:$|[^a-z0-9])"
    ),
    re.compile(r"(?i)(?:media|video)[_ -]?(?:url|uri|path|locator)"),
    re.compile(
        r"(?i)(?:^|[\s=:])(?:bearer\s+[a-z0-9._-]{8,}|sk-[a-z0-9_-]{8,}|"
        r"(?:api|app|client)[_-]?secret|password|passwd|private[_-]?key)"
    ),
)


def drill_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _string_values(nested)]
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in _string_values(nested)]
    return []


class DrillContractModel(CoreContractModel):
    """Revalidate copies and nested instances so callers cannot bypass policy checks."""

    model_config = ConfigDict(revalidate_instances="always")

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(drill_model_values(copied))

    @model_validator(mode="after")
    def sensitive_content_is_forbidden(self) -> Self:
        values = self.model_dump(mode="json")
        if any(
            pattern.search(value)
            for value in _string_values(values)
            for pattern in _SENSITIVE_PATTERNS
        ):
            raise ValueError("sensitive content is forbidden in drill catalog models")
        return self


class DrillReviewStatus(StrEnum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    DISABLED = "DISABLED"
    RETIRED = "RETIRED"


class DrillViolationCode(StrEnum):
    CATALOG_VERSION_MISMATCH = "CATALOG_VERSION_MISMATCH"
    DRILL_NOT_FOUND = "DRILL_NOT_FOUND"
    DRILL_REVISION_STALE = "DRILL_REVISION_STALE"
    REQUIRED_CAPABILITIES_MISMATCH = "REQUIRED_CAPABILITIES_MISMATCH"
    DRILL_NOT_APPROVED = "DRILL_NOT_APPROVED"
    USER_LEVEL_OUT_OF_RANGE = "USER_LEVEL_OUT_OF_RANGE"
    PREREQUISITE_SKILL_MISSING = "PREREQUISITE_SKILL_MISSING"
    DEVICE_CAPABILITY_MISSING = "DEVICE_CAPABILITY_MISSING"
    REQUIRED_EQUIPMENT_MISSING = "REQUIRED_EQUIPMENT_MISSING"
    CONTRAINDICATION_PRESENT = "CONTRAINDICATION_PRESENT"


class TablePointMm(DrillContractModel):
    x_mm: StrictInt = Field(ge=0)
    y_mm: StrictInt = Field(ge=0)


class BallPlacement(DrillContractModel):
    ball_code: NonEmptyStr
    position: TablePointMm


class TableTargetZoneMm(DrillContractModel):
    zone_code: NonEmptyStr
    minimum_x_mm: StrictInt = Field(ge=0)
    maximum_x_mm: StrictInt = Field(ge=0)
    minimum_y_mm: StrictInt = Field(ge=0)
    maximum_y_mm: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> TableTargetZoneMm:
        if self.minimum_x_mm >= self.maximum_x_mm or self.minimum_y_mm >= self.maximum_y_mm:
            raise ValueError("target zone bounds must have positive area")
        return self


class TableSetup(DrillContractModel):
    table_width_mm: StrictInt = Field(gt=0)
    table_length_mm: StrictInt = Field(gt=0)
    placements: tuple[BallPlacement, ...] = Field(min_length=1)
    target_pocket_code: NonEmptyStr | None = None
    target_zone: TableTargetZoneMm | None = None

    @model_validator(mode="after")
    def setup_is_unambiguous_and_on_table(self) -> TableSetup:
        if (self.target_pocket_code is None) == (self.target_zone is None):
            raise ValueError("setup must define exactly one target pocket or target zone")
        ball_codes = tuple(item.ball_code for item in self.placements)
        if len(ball_codes) != len(set(ball_codes)):
            raise ValueError("setup ball placements must be unique")
        coordinates = tuple((item.position.x_mm, item.position.y_mm) for item in self.placements)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("setup ball placement coordinates must be unique")
        if any(
            item.position.x_mm > self.table_length_mm or item.position.y_mm > self.table_width_mm
            for item in self.placements
        ):
            raise ValueError("ball placement must be within table millimeter bounds")
        zone = self.target_zone
        if zone and (
            zone.maximum_x_mm > self.table_length_mm or zone.maximum_y_mm > self.table_width_mm
        ):
            raise ValueError("target zone must be within table millimeter bounds")
        return self


class SuccessMetricDefinition(DrillContractModel):
    metric_code: NonEmptyStr
    target_attempts: StrictInt = Field(gt=0)
    minimum_successes: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def successes_do_not_exceed_attempts(self) -> SuccessMetricDefinition:
        if self.minimum_successes > self.target_attempts:
            raise ValueError("minimum successes cannot exceed target attempts")
        return self


class ProgressionRule(DrillContractModel):
    reason_code: NonEmptyStr
    threshold: StrictInt = Field(gt=0)
    level_delta: StrictInt = Field(gt=0)


class RegressionRule(DrillContractModel):
    reason_code: NonEmptyStr
    threshold: StrictInt = Field(gt=0)
    level_delta: StrictInt = Field(lt=0)


class ReviewLineage(DrillContractModel):
    reviewer_id: UUID4
    reviewed_at: datetime
    reason_code: NonEmptyStr

    _reviewed_at_utc = field_validator("reviewed_at")(require_utc)


class DrillEligibilityProfile(DrillContractModel):
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


class DrillDefinition(DrillContractModel):
    schema_version: Literal["1.0"] = "1.0"
    drill_id: NonEmptyStr
    revision: StrictInt = Field(ge=1)
    supersedes_revision: StrictInt | None = Field(default=None, ge=1)
    supersedes_content_sha256: Sha256 | None = None
    skill_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    minimum_level: StrictInt = Field(ge=1)
    maximum_level: StrictInt = Field(ge=1)
    prerequisite_skill_codes: tuple[NonEmptyStr, ...] = ()
    setup: TableSetup
    instructions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    success_metrics: tuple[SuccessMetricDefinition, ...] = Field(min_length=1)
    progression_rule: ProgressionRule
    regression_rule: RegressionRule
    required_equipment_codes: tuple[NonEmptyStr, ...] = ()
    required_capabilities: tuple[NonEmptyStr, ...] = ()
    safety_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    contraindication_codes: tuple[NonEmptyStr, ...] = ()
    review_status: DrillReviewStatus
    review_lineage: ReviewLineage
    content_sha256: Sha256

    @field_validator(
        "skill_codes",
        "prerequisite_skill_codes",
        "required_equipment_codes",
        "required_capabilities",
        "safety_codes",
        "contraindication_codes",
    )
    @classmethod
    def code_sets_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("code tuples must be sorted and unique")
        return value

    @classmethod
    def build(cls, **values: object) -> DrillDefinition:
        payload = dict(values)
        payload.pop("content_sha256", None)
        payload.setdefault("schema_version", "1.0")
        for field_name in (
            "skill_codes",
            "prerequisite_skill_codes",
            "required_equipment_codes",
            "required_capabilities",
            "safety_codes",
            "contraindication_codes",
        ):
            if field_name in payload:
                validated_codes = _CODE_SET_ADAPTER.validate_python(payload[field_name])
                payload[field_name] = tuple(sorted(set(validated_codes)))
        validated = cls.model_validate(
            payload | {"content_sha256": "0" * 64},
            context=_SKIP_CONTENT_HASH,
        )
        normalized = validated.model_dump(mode="python", exclude={"content_sha256"})
        return cls.model_validate(normalized | {"content_sha256": canonical_sha256(normalized)})

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("DrillDefinition updates must use review_drill")
        return super().model_copy(deep=deep)

    @model_validator(mode="after")
    def revision_levels_metrics_and_hash_are_exact(self, info: ValidationInfo) -> DrillDefinition:
        expected_supersedes = None if self.revision == 1 else self.revision - 1
        if self.supersedes_revision != expected_supersedes:
            raise ValueError("supersedes revision must bind the immediately previous revision")
        if self.revision == 1 and self.supersedes_content_sha256 is not None:
            raise ValueError("initial revision cannot bind a supersedes content hash")
        if self.revision > 1 and self.supersedes_content_sha256 is None:
            raise ValueError("supersedes content hash is required after the initial revision")
        if self.minimum_level > self.maximum_level:
            raise ValueError("minimum level cannot exceed maximum level")
        if (
            self.setup.target_zone is not None
            and "CUE_BALL_TARGET_ZONE" not in self.required_capabilities
        ):
            raise ValueError("target_zone requires the CUE_BALL_TARGET_ZONE capability")
        metric_codes = tuple(item.metric_code for item in self.success_metrics)
        if len(metric_codes) != len(set(metric_codes)):
            raise ValueError("success metric codes must be unique")
        if info.context is not _SKIP_CONTENT_HASH:
            expected_hash = canonical_sha256(
                self.model_dump(mode="json", exclude={"content_sha256"})
            )
            if self.content_sha256 != expected_hash:
                raise ValueError("drill content hash does not match canonical content")
        return self

    def eligibility_violations(
        self, profile: DrillEligibilityProfile
    ) -> tuple[DrillViolationCode, ...]:
        definition = type(self).model_validate(drill_model_values(self))
        profile = DrillEligibilityProfile.model_validate(drill_model_values(profile))
        violations: list[DrillViolationCode] = []
        if definition.review_status is not DrillReviewStatus.APPROVED:
            violations.append(DrillViolationCode.DRILL_NOT_APPROVED)
        if not definition.minimum_level <= profile.user_level <= definition.maximum_level:
            violations.append(DrillViolationCode.USER_LEVEL_OUT_OF_RANGE)
        if not set(definition.prerequisite_skill_codes).issubset(profile.acquired_skill_codes):
            violations.append(DrillViolationCode.PREREQUISITE_SKILL_MISSING)
        if not set(definition.required_capabilities).issubset(profile.available_capabilities):
            violations.append(DrillViolationCode.DEVICE_CAPABILITY_MISSING)
        if not set(definition.required_equipment_codes).issubset(profile.available_equipment_codes):
            violations.append(DrillViolationCode.REQUIRED_EQUIPMENT_MISSING)
        if set(definition.contraindication_codes).intersection(
            profile.active_contraindication_codes
        ):
            violations.append(DrillViolationCode.CONTRAINDICATION_PRESENT)
        return tuple(violations)


class DrillCatalog(DrillContractModel):
    schema_version: Literal["1.0"] = "1.0"
    catalog_version: NonEmptyStr
    created_at: datetime
    revision_history: tuple[DrillDefinition, ...] | None = None
    drills: tuple[DrillDefinition, ...] = Field(min_length=1)
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)

    @classmethod
    def build(cls, **values: object) -> DrillCatalog:
        payload = dict(values)
        payload.pop("content_sha256", None)
        payload.setdefault("schema_version", "1.0")
        source_history = payload.pop("revision_history", payload.pop("drills", None))
        if source_history is None:
            raise ValueError("revision_history is required")
        normalized_history = tuple(
            DrillDefinition.model_validate(
                drill_model_values(item) if isinstance(item, BaseModel) else item
            )
            for item in source_history  # type: ignore[union-attr]
        )
        normalized_history = tuple(
            sorted(
                normalized_history,
                key=lambda item: (item.drill_id, item.revision, item.content_sha256),
            )
        )
        current_heads: dict[str, DrillDefinition] = {}
        for definition in normalized_history:
            current_heads[definition.drill_id] = definition
        payload["revision_history"] = normalized_history
        payload["drills"] = tuple(current_heads[drill_id] for drill_id in sorted(current_heads))
        validated = cls.model_validate(
            payload | {"content_sha256": "0" * 64},
            context=_SKIP_CONTENT_HASH,
        )
        normalized = validated.model_dump(mode="python", exclude={"content_sha256"})
        return cls.model_validate(normalized | {"content_sha256": canonical_sha256(normalized)})

    @model_validator(mode="after")
    def history_heads_order_and_hash_are_exact(self, info: ValidationInfo) -> DrillCatalog:
        history_value = self.revision_history
        if history_value is None:
            history_value = self.drills
            object.__setattr__(self, "revision_history", history_value)
        if not history_value:
            raise ValueError("revision history must not be empty")
        expected_history_order = tuple(
            sorted(
                history_value,
                key=lambda item: (item.drill_id, item.revision, item.content_sha256),
            )
        )
        if history_value != expected_history_order:
            raise ValueError("revision history must use stable drill id and revision order")

        histories: dict[str, list[DrillDefinition]] = {}
        for definition in history_value:
            histories.setdefault(definition.drill_id, []).append(definition)
        expected_heads: list[DrillDefinition] = []
        for drill_id in sorted(histories):
            history = histories[drill_id]
            revisions = tuple(item.revision for item in history)
            if revisions != tuple(range(1, len(history) + 1)):
                raise ValueError(
                    "revision history must be complete with unique, unbranched revisions"
                )
            for previous, current in zip(history, history[1:], strict=False):
                if previous.review_status is DrillReviewStatus.RETIRED:
                    raise ValueError("RETIRED is terminal in revision history")
                if (
                    current.supersedes_revision != previous.revision
                    or current.supersedes_content_sha256 != previous.content_sha256
                ):
                    raise ValueError(
                        "revision history supersedes binding must match previous content"
                    )
                if current.review_lineage.reviewed_at <= previous.review_lineage.reviewed_at:
                    raise ValueError("revision history reviewed_at must advance")
            expected_heads.append(history[-1])

        head_ids = tuple(item.drill_id for item in self.drills)
        if len(head_ids) != len(set(head_ids)):
            raise ValueError("each drill id must have one unique current head")
        if self.drills != tuple(expected_heads):
            raise ValueError("catalog heads must use stable drill id and revision order")
        if info.context is not _SKIP_CONTENT_HASH:
            expected_hash = canonical_sha256(
                self.model_dump(mode="json", exclude={"content_sha256"})
            )
            if self.content_sha256 != expected_hash:
                raise ValueError("catalog content hash does not match canonical content")
        return self


class DrillReferenceValidation(DrillContractModel):
    is_valid: bool
    drill: DrillDefinition | None = None
    violation_codes: tuple[DrillViolationCode, ...] = ()

    @model_validator(mode="after")
    def result_is_consistent(self) -> DrillReferenceValidation:
        if self.is_valid and (self.drill is None or self.violation_codes):
            raise ValueError(
                "reference validation result is inconsistent: "
                "valid requires a drill and no violations"
            )
        if not self.is_valid and (self.drill is not None or not self.violation_codes):
            raise ValueError("reference validation result is inconsistent")
        return self
