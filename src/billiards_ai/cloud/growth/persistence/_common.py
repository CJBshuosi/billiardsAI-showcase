from __future__ import annotations

import json

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from billiards_ai.cloud.review.persistence.privacy import assert_safe_persistence_payload
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256


def strict_json_model[ModelT: BaseModel](model: type[ModelT], value: object) -> ModelT:
    return model.model_validate_json(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        strict=True,
    )


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def insert_outbox(
    session: AsyncSession,
    event: DomainEvent,
    *,
    owner_user_id,
    aggregate_namespace: str,
) -> None:
    assert_safe_persistence_payload(event.payload)
    payload = event.payload
    event_value = event.model_dump(mode="json")
    await session.execute(
        text(
            "INSERT INTO core_intelligence_outbox "
            "(event_id,owner_user_id,aggregate_namespace,schema_version,event_type,"
            "aggregate_type,aggregate_id,"
            "aggregate_version,occurred_at,causation_id,correlation_id,idempotency_key,"
            "payload,payload_sha256,event_sha256,available_at,attempt_count) VALUES "
            "(:event_id,:owner_user_id,:aggregate_namespace,:schema_version,:event_type,"
            ":aggregate_type,:aggregate_id,"
            ":aggregate_version,:occurred_at,:causation_id,:correlation_id,:idempotency_key,"
            "CAST(:payload AS jsonb),:payload_sha256,:event_sha256,:occurred_at,0)"
        ),
        {
            "event_id": event.event_id,
            "owner_user_id": owner_user_id,
            "aggregate_namespace": aggregate_namespace,
            "schema_version": event.schema_version,
            "event_type": event.event_type,
            "aggregate_type": event.aggregate_type,
            "aggregate_id": event.aggregate_id,
            "aggregate_version": event.aggregate_version,
            "occurred_at": event.occurred_at,
            "causation_id": event.causation_id,
            "correlation_id": event.correlation_id,
            "idempotency_key": event.idempotency_key,
            "payload": json_text(payload),
            "payload_sha256": canonical_sha256(payload),
            "event_sha256": canonical_sha256(event_value),
        },
    )


def event_from_row(row) -> DomainEvent:
    assert_safe_persistence_payload(row["payload"])
    if canonical_sha256(row["payload"]) != row["payload_sha256"]:
        raise ValueError("outbox payload digest mismatch")
    event = DomainEvent(
        schema_version=row["schema_version"],
        event_id=row["event_id"],
        event_type=row["event_type"],
        aggregate_type=row["aggregate_type"],
        aggregate_id=row["aggregate_id"],
        aggregate_version=row["aggregate_version"],
        occurred_at=row["occurred_at"],
        causation_id=row["causation_id"],
        correlation_id=row["correlation_id"],
        idempotency_key=row["idempotency_key"],
        payload=row["payload"],
    )
    if canonical_sha256(event.model_dump(mode="json")) != row["event_sha256"]:
        raise ValueError("outbox event digest mismatch")
    return event
