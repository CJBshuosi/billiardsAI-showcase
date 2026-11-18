"""Fail-closed application boundary for PlanOutcome history ingestion."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from billiards_ai.cloud.growth.outcome_repository import (
    OutcomeIngestError,
    OutcomeIngestErrorCode,
    PlanOutcomeIngestResult,
    PlanOutcomeRepository,
    revalidate_plan_outcome,
    validate_outcome_event_lineage,
)
from billiards_ai.shared.models.events import DomainEvent
from billiards_ai.shared.models.growth import PlanOutcome


class AsyncPlanOutcomeRepository(Protocol):
    async def append(
        self,
        outcome: PlanOutcome,
        *,
        source_event: DomainEvent,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> PlanOutcomeIngestResult: ...


class PlanOutcomeIngestService:
    def __init__(
        self,
        *,
        repository: PlanOutcomeRepository,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ):
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._uuid_factory = uuid_factory or uuid4

    def ingest(
        self,
        outcome: PlanOutcome,
        *,
        source_event: DomainEvent,
        authenticated_user_id: UUID,
    ) -> PlanOutcomeIngestResult:
        validated_outcome = revalidate_plan_outcome(outcome)
        if validated_outcome.user_id != authenticated_user_id:
            raise OutcomeIngestError(OutcomeIngestErrorCode.RESOURCE_NOT_OWNED)
        validated_outcome, validated_event = validate_outcome_event_lineage(
            validated_outcome, source_event
        )
        return self._repository.append(
            validated_outcome,
            source_event=validated_event,
            clock=self._clock,
            uuid_factory=self._uuid_factory,
        )


class AsyncPlanOutcomeIngestService:
    """Async application boundary for durable PlanOutcome repositories."""

    def __init__(
        self,
        *,
        repository: AsyncPlanOutcomeRepository,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._uuid_factory = uuid_factory or uuid4

    async def ingest(
        self,
        outcome: PlanOutcome,
        *,
        source_event: DomainEvent,
        authenticated_user_id: UUID,
    ) -> PlanOutcomeIngestResult:
        validated_outcome = revalidate_plan_outcome(outcome)
        if validated_outcome.user_id != authenticated_user_id:
            raise OutcomeIngestError(OutcomeIngestErrorCode.RESOURCE_NOT_OWNED)
        validated_outcome, validated_event = validate_outcome_event_lineage(
            validated_outcome, source_event
        )
        return await self._repository.append(
            validated_outcome,
            source_event=validated_event,
            clock=self._clock,
            uuid_factory=self._uuid_factory,
        )
