"""Versioned primitives shared by the four core-intelligence mainlines."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StringConstraints,
    field_validator,
)
from pydantic_core import to_jsonable_python

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
EventTypeV1 = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]*\.v1$")]


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("datetime must use UTC")
    return value.astimezone(UTC)


def canonical_sha256(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            to_jsonable_python(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class CoreContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DomainEvent(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID4
    event_type: EventTypeV1
    aggregate_type: NonEmptyStr
    aggregate_id: UUID4
    aggregate_version: StrictInt = Field(ge=1)
    occurred_at: datetime
    causation_id: UUID4 | None = None
    correlation_id: UUID4
    idempotency_key: NonEmptyStr
    payload: dict[str, JsonValue]

    _occurred_at_utc = field_validator("occurred_at")(require_utc)
