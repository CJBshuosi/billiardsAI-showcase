"""Protected identity resolution for REAL_VIDEO_GATE approval."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pwd
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from billiards_ai.shared.models import ApproverIdentity

from .errors import GateApprovalError, GateRepositoryError

APPROVER_CREDENTIAL_ID = "phase0-local-approval"
_APPROVER_CREDENTIAL_RELATIVE_PATH = Path(".config/billiards-ai/approver.json")
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_KeyId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


LEGACY_APPROVAL_KEY_ID = "legacy-v1"


class _CredentialPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    approver_id: Literal["Boyu"]
    credential_id: Literal["phase0-local-approval"]
    approval_hmac_key: _Sha256


class _CredentialPayloadV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    approver_id: Literal["Boyu"]
    credential_id: Literal["phase0-local-approval"]
    active_key_id: _KeyId
    keys: dict[_KeyId, _Sha256] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def raw_key_ids_are_unambiguous(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        active_key_id = value.get("active_key_id")
        if isinstance(active_key_id, str) and not active_key_id.strip():
            raise ValueError("active approval key id must not be blank")
        keys = value.get("keys")
        if not isinstance(keys, dict):
            return value
        normalized_to_raw: dict[str, str] = {}
        for raw_key_id in keys:
            if not isinstance(raw_key_id, str):
                continue
            normalized = raw_key_id.strip()
            if not normalized:
                raise ValueError("approval key ids must not be blank")
            previous = normalized_to_raw.get(normalized)
            if previous is not None and previous != raw_key_id:
                raise ValueError("approval key ids collide after whitespace normalization")
            normalized_to_raw[normalized] = raw_key_id
        return value

    @model_validator(mode="after")
    def active_key_exists(self) -> _CredentialPayloadV2:
        if self.active_key_id not in self.keys:
            raise ValueError("active approval key must exist in the keyring")
        return self


@dataclass(frozen=True)
class _VerifiedApproverIdentity:
    approver_id: str
    credential_id: str
    active_key_id: str
    approval_hmac_keys: dict[str, bytes]

    def approval_mac(
        self,
        *,
        key_id: str,
        payload: dict[str, Any],
    ) -> str:
        try:
            key = self.approval_hmac_keys[key_id]
        except KeyError as exc:
            raise GateApprovalError("approval key is not present in the trusted keyring") from exc
        return _calculate_approval_mac(
            key=key,
            payload=payload,
        )

    def legacy_approval_mac(
        self,
        *,
        gate_id: UUID,
        revision_id: UUID,
        gate_fingerprint: str,
        approved_at: datetime,
        approval_evidence_sha256: str,
    ) -> str:
        try:
            key = self.approval_hmac_keys[LEGACY_APPROVAL_KEY_ID]
        except KeyError as exc:
            raise GateApprovalError("legacy approval key is not present") from exc
        approved_at_utc = approved_at.astimezone(UTC).isoformat()
        payload = "|".join(
            (
                str(gate_id),
                str(revision_id),
                gate_fingerprint,
                approved_at_utc,
                approval_evidence_sha256,
            )
        ).encode("utf-8")
        return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _calculate_approval_mac(
    *,
    key: bytes,
    payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _account_home() -> Path:
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError) as exc:
        raise GateRepositoryError("operating-system account home could not be resolved") from exc


def _require_private_directory(path: Path, *, label: str) -> None:
    try:
        info = path.stat()
    except OSError as exc:
        raise GateRepositoryError(f"{label} could not be inspected") from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise GateRepositoryError(f"{label} must be a real directory")
    if info.st_uid != os.geteuid():
        raise GateRepositoryError(f"{label} must be owned by the current account")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise GateRepositoryError(f"{label} permissions must not grant group or other access")
    for ancestor in path.parents:
        try:
            ancestor_info = ancestor.lstat()
        except OSError as exc:
            raise GateRepositoryError(f"{label} ancestor could not be inspected") from exc
        if stat.S_ISLNK(ancestor_info.st_mode) or not stat.S_ISDIR(ancestor_info.st_mode):
            raise GateRepositoryError(f"{label} ancestors must be real directories")
        if ancestor_info.st_uid not in {os.geteuid(), 0}:
            raise GateRepositoryError(
                f"{label} ancestors must be owned by the current account or root"
            )
        ancestor_mode = stat.S_IMODE(ancestor_info.st_mode)
        is_root_sticky_directory = ancestor_info.st_uid == 0 and bool(ancestor_mode & stat.S_ISVTX)
        if ancestor_mode & 0o022 and not is_root_sticky_directory:
            raise GateRepositoryError(
                f"{label} ancestors must not be writable by group or other accounts"
            )


def _read_private_bytes(path: Path, *, label: str) -> bytes:
    """Read trusted bytes from the same descriptor that is permission-checked."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise GateRepositoryError(f"{label} could not be opened safely") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise GateRepositoryError(f"{label} must be a regular file")
        if info.st_uid != os.geteuid():
            raise GateRepositoryError(f"{label} must be owned by the current account")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise GateRepositoryError(f"{label} permissions must be exactly 0600")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _read_private_json(path: Path, *, label: str) -> Any:
    return json.loads(_read_private_bytes(path, label=label).decode("utf-8"))


def _is_inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path.parent, *path.parents))


class FileApproverIdentityProvider:
    """Resolve the approver from a protected file outside the repository."""

    @property
    def path(self) -> Path:
        return _account_home() / _APPROVER_CREDENTIAL_RELATIVE_PATH

    def current_identity(self) -> _VerifiedApproverIdentity:
        try:
            if _is_inside_git_worktree(self.path):
                raise GateRepositoryError("approval credential must stay outside a Git worktree")
            _require_private_directory(self.path.parent, label="approval credential directory")
            payload = _read_private_json(self.path, label="approval credential")
            if not isinstance(payload, dict):
                raise GateRepositoryError("approval credential must be a JSON object")
            if payload.get("schema_version") == "1.0":
                credential_v1 = _CredentialPayloadV1.model_validate(payload)
                approver_id = credential_v1.approver_id
                credential_id = credential_v1.credential_id
                active_key_id = LEGACY_APPROVAL_KEY_ID
                keys = {LEGACY_APPROVAL_KEY_ID: credential_v1.approval_hmac_key}
            else:
                credential_v2 = _CredentialPayloadV2.model_validate(payload)
                approver_id = credential_v2.approver_id
                credential_id = credential_v2.credential_id
                active_key_id = credential_v2.active_key_id
                keys = credential_v2.keys
            identity = ApproverIdentity.model_validate(
                {
                    "schema_version": "1.0",
                    "approver_id": approver_id,
                    "credential_id": credential_id,
                }
            )
            if identity.credential_id != APPROVER_CREDENTIAL_ID:
                raise GateApprovalError("approval credential is not on the trusted allowlist")
            return _VerifiedApproverIdentity(
                approver_id=identity.approver_id,
                credential_id=identity.credential_id,
                active_key_id=active_key_id,
                approval_hmac_keys={key_id: bytes.fromhex(key) for key_id, key in keys.items()},
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            GateRepositoryError,
        ) as exc:
            raise GateApprovalError("approval identity could not be verified") from exc
