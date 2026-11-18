"""Pure local owner-erasure identities; these values do not prove cloud erasure."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.managed_sessions import (
    validate_managed_scope_id,
    validate_managed_uuid,
)
from billiards_ai.edge.route_assistance.postgame import (
    MAX_SAVED_CLIP_BYTES,
    session_scoped_clip_id,
    validate_save_request_id,
)


def _identity(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(unicodedata.category(c) in {"Cc", "Cs"} for c in value)
        or len(value.encode("utf-8")) > 256
    ):
        raise ValueError("invalid owner delete identity")


def _digest(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid owner delete digest")


@dataclass(frozen=True, slots=True)
class OwnerDeleteDeviceTask:
    contract: str
    deletion_job_id: str
    environment_id: str
    device_id: str
    table_id: str
    session_id: str
    shot_id: str
    source_job_id: str
    artifact_id: str
    clip_id: str
    owner_user_id: str
    actor_role: WristbandRole
    content_sha256: str
    byte_size: int
    ownership_receipt_sha256: str
    save_request_id: str
    save_receipt_sha256: str
    postgame_manifest_sha256: str
    cloud_erase_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name not in {"byte_size", "actor_role"}:
                _identity(getattr(self, field.name))
        if self.contract != "owner_delete_device_v1":
            raise ValueError("invalid owner delete contract")
        for name in ("environment_id", "device_id", "table_id"):
            validate_managed_scope_id(getattr(self, name), name=name)
        if not self.session_id.startswith("managed-"):
            raise ValueError("owner delete requires canonical managed session")
        validate_managed_uuid(self.session_id[len("managed-") :], name="birth ID")
        validate_save_request_id(self.save_request_id)
        for name in (
            "artifact_id",
            "content_sha256",
            "ownership_receipt_sha256",
            "save_receipt_sha256",
            "postgame_manifest_sha256",
            "cloud_erase_receipt_sha256",
        ):
            _digest(getattr(self, name))
        if not isinstance(self.actor_role, WristbandRole) or self.actor_role not in (
            WristbandRole.RED,
            WristbandRole.BLUE,
        ):
            raise ValueError("invalid owner delete role")
        if type(self.byte_size) is not int or not 0 < self.byte_size <= MAX_SAVED_CLIP_BYTES:
            raise ValueError("invalid owner delete byte size")
        if (
            self.clip_id
            != session_scoped_clip_id(
                self.session_id, self.shot_id, self.source_job_id, self.artifact_id
            )
            or self.deletion_job_id
            != "owner-delete-" + sha256(self.clip_id.encode()).hexdigest()[:32]
        ):
            raise ValueError("invalid owner delete clip identity")

    @classmethod
    def from_payload(cls, payload: object) -> OwnerDeleteDeviceTask:
        if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("invalid owner delete task fields")
        values = dict(payload)
        if not isinstance(values["actor_role"], str):
            raise ValueError("invalid owner delete role")
        values["actor_role"] = WristbandRole(values["actor_role"])
        return cls(**values)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def task_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OwnerDeleteLocalState:
    task: OwnerDeleteDeviceTask
    task_sha256: str
    local_path: Path
    status: str
    attempts: int
    claim_token: str | None
    lease_until_utc: str | None
    progress_json: str
    receipt_json: str | None
    receipt_sha256: str | None

    @property
    def session_id(self) -> str:
        return self.task.session_id

    @property
    def shot_id(self) -> str:
        return self.task.shot_id

    @property
    def postgame_manifest_sha256(self) -> str:
        return self.task.postgame_manifest_sha256

    @property
    def job_id(self) -> str:
        return self.task.source_job_id

    @property
    def artifact_id(self) -> str:
        return self.task.artifact_id

    @property
    def clip_id(self) -> str:
        return self.task.clip_id

    @property
    def content_sha256(self) -> str:
        return self.task.content_sha256

    @property
    def byte_size(self) -> int:
        return self.task.byte_size

    @property
    def content_type(self) -> str:
        return "video/mp4"


@dataclass(frozen=True, slots=True)
class LocalOwnerDeleteAttempt:
    outcome: str
    error_code: str | None = None
    reclaimed_bytes: int = 0
    session_id: str | None = None
    shot_id: str | None = None
