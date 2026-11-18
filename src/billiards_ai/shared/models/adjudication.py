"""Frozen contracts for optional Story 2.8 ambiguity adjudication."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import UUID4, Field, StrictBool, model_validator

from billiards_ai.shared.constants import BALL_IDS
from billiards_ai.shared.models.observation import ObservationConfidence, ObservationSource
from billiards_ai.shared.models.privacy import Sha256, StrictFrozenModel


class AdjudicationStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    DISABLED = "DISABLED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_MISMATCH = "APPROVAL_MISMATCH"
    COST_LIMIT_EXCEEDED = "COST_LIMIT_EXCEEDED"
    SUCCEEDED = "SUCCEEDED"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    PARSE_FAILED = "PARSE_FAILED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class AdjudicationSource(StrEnum):
    NONE = "NONE"
    CLASSIC_CV = "CLASSIC_CV"
    EXTERNAL_PROVIDER = "EXTERNAL_PROVIDER"


class AdjudicationCandidate(StrictFrozenModel):
    ball_id: int = Field(ge=1, le=max(BALL_IDS))
    confidence: ObservationConfidence
    source: ObservationSource

    @model_validator(mode="after")
    def candidate_is_uncertain(self) -> AdjudicationCandidate:
        if self.confidence not in {ObservationConfidence.LOW, ObservationConfidence.MEDIUM}:
            raise ValueError("adjudication candidates must have low or medium confidence")
        return self


class AdjudicationRequest(StrictFrozenModel):
    """The complete provider boundary; it intentionally cannot carry media."""

    schema_version: Literal["1.0"] = "1.0"
    shot_id: UUID4
    idempotency_key: Sha256
    evidence_signature: Sha256
    candidates: tuple[AdjudicationCandidate, ...]
    estimated_cost_cny: Decimal = Field(ge=0, max_digits=10, decimal_places=2)

    @model_validator(mode="after")
    def candidates_are_canonical(self) -> AdjudicationRequest:
        candidate_ids = tuple(candidate.ball_id for candidate in self.candidates)
        if not candidate_ids:
            raise ValueError("at least one adjudication candidate is required")
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("adjudication candidate IDs must be sorted and unique")
        return self


class AdjudicatedStruckBall(StrictFrozenModel):
    ball_id: int = Field(ge=1, le=max(BALL_IDS))


class AdjudicationRawObservation(StrictFrozenModel):
    """Provider-specific and free-text material, excluded from business fields."""

    provider_output: str | None = None
    failure_detail: str | None = None


class AdjudicationResult(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    shot_id: UUID4
    idempotency_key: Sha256
    evidence_signature: Sha256
    result: AdjudicatedStruckBall | None
    source: AdjudicationSource
    confidence: ObservationConfidence
    status: AdjudicationStatus
    retryable: StrictBool
    estimated_cost_cny: Decimal = Field(ge=0, max_digits=10, decimal_places=2)
    raw_observation: AdjudicationRawObservation

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> AdjudicationResult:
        if self.status is AdjudicationStatus.SUCCEEDED:
            if (
                self.result is None
                or self.source is not AdjudicationSource.EXTERNAL_PROVIDER
                or self.confidence is ObservationConfidence.UNKNOWN
                or self.retryable
            ):
                raise ValueError("successful adjudication requires a typed provider result")
            return self

        if self.result is not None:
            raise ValueError("only successful adjudication may contain a result")
        if self.retryable and self.status is not AdjudicationStatus.PROVIDER_FAILED:
            raise ValueError("retryable is reserved for provider failure")
        if self.status is AdjudicationStatus.PROVIDER_FAILED:
            if self.source is not AdjudicationSource.EXTERNAL_PROVIDER:
                raise ValueError("provider failure requires external provider source")
        elif self.retryable:
            raise ValueError("retryable is reserved for provider failure")
        return self
