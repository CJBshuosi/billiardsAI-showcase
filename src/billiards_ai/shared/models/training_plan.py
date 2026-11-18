"""Versioned training-plan and publication contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator

from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.growth import Reliability


class DrillReference(CoreContractModel):
    catalog_version: NonEmptyStr
    drill_id: NonEmptyStr
    drill_revision: StrictInt = Field(ge=1)
    required_capabilities: tuple[NonEmptyStr, ...] = ()


class TrainingBlock(CoreContractModel):
    block_id: UUID4
    order: StrictInt = Field(ge=1)
    drill: DrillReference
    target_finding_id: UUID4
    target_repetitions: StrictInt = Field(gt=0)
    progress_after_successes: StrictInt = Field(gt=0)
    regress_after_failures: StrictInt = Field(gt=0)


class TrainingPlan(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: UUID4
    user_id: UUID4
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    version: StrictInt = Field(ge=1)
    created_at: datetime
    valid_until: datetime
    blocks: tuple[TrainingBlock, ...] = Field(min_length=1)
    fallback_drill: DrillReference
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)
    _valid_until_utc = field_validator("valid_until")(require_utc)

    @model_validator(mode="after")
    def blocks_are_ordered_and_plan_is_live(self) -> TrainingPlan:
        orders = tuple(item.order for item in self.blocks)
        if orders != tuple(range(1, len(self.blocks) + 1)):
            raise ValueError("training block order must be contiguous from one")
        block_ids = tuple(item.block_id for item in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("training block ids must be unique")
        if self.valid_until <= self.created_at:
            raise ValueError("plan validity must follow creation")
        return self


class GrowthRuntimeView(CoreContractModel):
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    focus_metric_codes: tuple[NonEmptyStr, ...]
    minimum_reliability: Reliability


class PublishedPlanBundle(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    bundle_id: UUID4
    plan: TrainingPlan
    target_device_id: NonEmptyStr
    growth_runtime_view: GrowthRuntimeView
    published_at: datetime
    expires_at: datetime
    plan_sha256: Sha256
    signature_key_id: NonEmptyStr
    signature: NonEmptyStr

    _published_at_utc = field_validator("published_at")(require_utc)
    _expires_at_utc = field_validator("expires_at")(require_utc)

    @model_validator(mode="after")
    def publication_binds_exact_plan(self) -> PublishedPlanBundle:
        if canonical_sha256(self.plan) != self.plan_sha256:
            raise ValueError("published plan hash does not match embedded plan")
        if self.expires_at > self.plan.valid_until or self.expires_at <= self.published_at:
            raise ValueError("bundle expiry must be live and within plan validity")
        return self
