"""Pure, strict identities for server-issued managed sessions.

These values bind configuration and server responses. They grant no media,
storage, terminal-close, or physical-device authority.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

# Keep this wire-contract value independent of the rule engine (which loads lab).
_RULE_PROFILE_ID = "cbsa-chinese-pool-2017-observable-v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
_UUID_V4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z"
)


def validate_managed_scope_id(value: str, *, name: str) -> str:
    """Validate without coercing or normalizing an issued/configured identity."""
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None or value in (".", ".."):
        raise ValueError(f"invalid managed {name}")
    return value


def validate_managed_uuid(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _UUID_V4.fullmatch(value) is None:
        raise ValueError(f"invalid managed {name}")
    return value


def validate_managed_artifact_id(value: str) -> str:
    return validate_managed_scope_id(value, name="artifact ID")


def validate_managed_rule_profile_id(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or value != _RULE_PROFILE_ID):
        raise ValueError("invalid managed rule profile ID")
    return value


def validate_managed_utc_timestamp(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"invalid managed {name}")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"invalid managed {name}") from None
    return value


@dataclass(frozen=True, slots=True)
class ManagedSessionScope:
    environment_id: str
    device_id: str
    table_id: str

    def __post_init__(self) -> None:
        for name in ("environment_id", "device_id", "table_id"):
            validate_managed_scope_id(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class ManagedSessionBinding:
    schema_version: str
    session_id: str
    birth_id: str
    birth_contract: str
    table_id: str
    device_id: str
    status: Literal["open", "ended"]

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, field.name), str) for field in fields(self)):
            raise ValueError("invalid managed session field type")
        validate_managed_uuid(self.birth_id, name="birth ID")
        validate_managed_scope_id(self.table_id, name="table ID")
        validate_managed_scope_id(self.device_id, name="device ID")
        if (
            self.schema_version != "1.0"
            or self.birth_contract != "managed_session_birth_v1"
            or self.session_id != f"managed-{self.birth_id}"
            or self.status not in ("open", "ended")
        ):
            raise ValueError("invalid managed session binding")

    @classmethod
    def from_payload(
        cls, payload: object, *, expected_table_id: str, expected_device_id: str
    ) -> ManagedSessionBinding:
        validate_managed_scope_id(expected_table_id, name="expected table ID")
        validate_managed_scope_id(expected_device_id, name="expected device ID")
        if not isinstance(payload, Mapping) or set(payload) != {
            field.name for field in fields(cls)
        }:
            raise ValueError("invalid managed session payload fields")
        binding = cls(**payload)
        if binding.table_id != expected_table_id or binding.device_id != expected_device_id:
            raise ValueError("mismatched managed session scope")
        return binding


@dataclass(frozen=True, slots=True)
class ManagedOpenRequest:
    scope: ManagedSessionScope
    artifact_id: str
    rule_profile_id: str | None
    open_request_id: str
    phase: Literal["pending", "bound", "materialized"]
    binding: ManagedSessionBinding | None
    created_at_utc: str
    updated_at_utc: str
    lifecycle: Literal["active", "retired"] = "active"
    retired_at_utc: str | None = None
    sealed_close: ManagedSessionCloseState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ManagedSessionScope):
            raise ValueError("invalid managed session scope")
        validate_managed_artifact_id(self.artifact_id)
        validate_managed_rule_profile_id(self.rule_profile_id)
        validate_managed_uuid(self.open_request_id, name="open request ID")
        validate_managed_utc_timestamp(self.created_at_utc, name="created timestamp")
        validate_managed_utc_timestamp(self.updated_at_utc, name="updated timestamp")
        if not isinstance(self.phase, str) or self.phase not in (
            "pending",
            "bound",
            "materialized",
        ):
            raise ValueError("invalid managed request phase")
        if self.phase == "pending":
            if self.binding is not None:
                raise ValueError("pending managed request cannot have a binding")
        elif (
            not isinstance(self.binding, ManagedSessionBinding)
            or self.binding.table_id != self.scope.table_id
            or self.binding.device_id != self.scope.device_id
        ):
            raise ValueError("managed request requires a matching binding")
        created = datetime.fromisoformat(self.created_at_utc)
        updated = datetime.fromisoformat(self.updated_at_utc)
        if updated < created:
            raise ValueError("invalid managed request timestamp order")
        if not isinstance(self.lifecycle, str) or self.lifecycle not in ("active", "retired"):
            raise ValueError("invalid managed request lifecycle")
        if self.lifecycle == "active":
            if self.retired_at_utc is not None or self.sealed_close is not None:
                raise ValueError("active managed request cannot have retirement metadata")
        else:
            if (
                self.phase != "materialized"
                or self.binding is None
                or self.binding.status != "ended"
                or not isinstance(self.sealed_close, ManagedSessionCloseState)
            ):
                raise ValueError("retired managed request requires ended binding and sealed close")
            close = ManagedSessionCloseState.from_payload(
                {
                    field.name: getattr(self.sealed_close, field.name)
                    for field in fields(ManagedSessionCloseState)
                },
                binding=self.binding,
            )
            if close.coverage != "sealed":
                raise ValueError("retired managed request requires sealed close")
            validate_managed_utc_timestamp(self.retired_at_utc, name="retirement timestamp")
            if (
                self.updated_at_utc != self.retired_at_utc
                or datetime.fromisoformat(self.retired_at_utc) < created
            ):
                raise ValueError("invalid managed retirement timestamp order")


@runtime_checkable
class ManagedSessionGateway(Protocol):
    """Optional capability; legacy control implementations need not provide it."""

    def open_managed_session(
        self, *, open_request_id: str, expected_table_id: str
    ) -> ManagedSessionBinding: ...


@dataclass(frozen=True, slots=True)
class ManagedSessionCloseState:
    schema_version: str
    session_id: str
    birth_id: str
    birth_contract: str
    table_id: str
    device_id: str
    coverage: Literal["closing", "sealed"]
    closed_target: int
    contiguous_ack_sequence: int

    def __post_init__(self) -> None:
        # The six immutable identity fields share the binding's strict contract.
        ManagedSessionBinding(
            self.schema_version,
            self.session_id,
            self.birth_id,
            self.birth_contract,
            self.table_id,
            self.device_id,
            "ended",
        )
        if not isinstance(self.coverage, str) or self.coverage not in ("closing", "sealed"):
            raise ValueError("invalid managed close coverage")
        if any(
            type(value) is not int or not 0 <= value <= 9007199254740991
            for value in (self.closed_target, self.contiguous_ack_sequence)
        ):
            raise ValueError("invalid managed close prefix")
        if self.contiguous_ack_sequence > self.closed_target or (
            self.coverage == "sealed" and self.contiguous_ack_sequence != self.closed_target
        ):
            raise ValueError("invalid managed close acknowledgement")

    @classmethod
    def from_payload(
        cls, payload: object, *, binding: ManagedSessionBinding
    ) -> ManagedSessionCloseState:
        if not isinstance(binding, ManagedSessionBinding):
            raise ValueError("invalid managed session binding")
        if not isinstance(payload, Mapping) or set(payload) != {
            field.name for field in fields(cls)
        }:
            raise ValueError("invalid managed close payload fields")
        state = cls(**payload)
        if any(
            getattr(state, name) != getattr(binding, name)
            for name in (
                "schema_version",
                "session_id",
                "birth_id",
                "birth_contract",
                "table_id",
                "device_id",
            )
        ):
            raise ValueError("mismatched managed close scope")
        return state


@runtime_checkable
class ManagedSessionCloseGateway(Protocol):
    """Optional close capability, separate from allocation and legacy controls."""

    def close_managed_session(
        self, *, binding: ManagedSessionBinding
    ) -> ManagedSessionCloseState: ...
