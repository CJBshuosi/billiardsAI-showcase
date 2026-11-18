"""Store-scoped, recoverable chain-head anchoring for the Phase 0 repository."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from billiards_ai.shared.models import GateRevision

from .errors import GateRepositoryError
from .identity import _account_home, _read_private_json, _require_private_directory

_INTEGRITY_DIRECTORY = Path(".config/billiards-ai")
_CHAIN_KEY_NAME = "gate-head.key"
_CHAIN_REGISTRY_NAME = "gate-heads.json"
_CHAIN_LOCK_NAME = ".gate-heads.lock"
_CHAIN_KEY_ID = "phase0-gate-head-v1"
_REGISTRY_SCHEMA_VERSION = "2.0"


class _ChainKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    key_id: Literal["phase0-gate-head-v1"] = _CHAIN_KEY_ID
    hmac_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ChainHead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision_id: UUID
    revision_number: int = Field(ge=1)
    chain_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    key_id: Literal["phase0-gate-head-v1"] = _CHAIN_KEY_ID
    head_mac: str = Field(pattern=r"^[0-9a-f]{64}$")


class _PendingTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: UUID
    old_store_present: bool
    old_store_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    old_heads: dict[str, dict]
    new_store_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    new_heads: dict[str, dict]


class _StoreNamespace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: Literal["phase0-gate-head-v1"] = _CHAIN_KEY_ID
    store_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    heads: dict[str, dict]
    pending: _PendingTransaction | None = None
    namespace_mac: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def store_payload_digest(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def store_identity(path: Path) -> str:
    try:
        canonical_path = str(path.resolve(strict=False))
    except (OSError, RuntimeError) as exc:
        raise GateRepositoryError("gate store identity could not be resolved") from exc
    return hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()


def revision_chain_digest(history: Sequence[GateRevision]) -> str:
    digest = hashlib.sha256(b"billiards-ai-gate-chain-v1").hexdigest()
    for revision in history:
        digest = hashlib.sha256(
            bytes.fromhex(digest) + _canonical_json(revision.model_dump(mode="json"))
        ).hexdigest()
    return digest


def raw_revision_chain_digest(history: Sequence[dict]) -> str:
    """Digest the exact persisted revision fields without model normalization."""
    digest = hashlib.sha256(b"billiards-ai-gate-chain-v1").hexdigest()
    for revision in history:
        digest = hashlib.sha256(bytes.fromhex(digest) + _canonical_json(revision)).hexdigest()
    return digest


class GateHeadRegistry:
    """HMAC-authenticated namespaces and pending transactions for gate stores."""

    @property
    def directory(self) -> Path:
        return _account_home() / _INTEGRITY_DIRECTORY

    @property
    def key_path(self) -> Path:
        return self.directory / _CHAIN_KEY_NAME

    @property
    def registry_path(self) -> Path:
        return self.directory / _CHAIN_REGISTRY_NAME

    @property
    def lock_path(self) -> Path:
        return self.directory / _CHAIN_LOCK_NAME

    def _ensure_directory(self) -> None:
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            _require_private_directory(self.directory, label="gate head directory")
        except OSError as exc:
            raise GateRepositoryError("gate head directory could not be created") from exc

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self._ensure_directory()
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise GateRepositoryError("gate head lock could not be opened") from exc
        locked = False
        try:
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise GateRepositoryError("gate head lock must be a regular file")
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise GateRepositoryError("gate head lock ownership or permissions are unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        except GateRepositoryError:
            raise
        except OSError as exc:
            raise GateRepositoryError("gate head lock operation failed") from exc
        finally:
            try:
                if locked:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                raise GateRepositoryError("gate head lock could not be released") from exc
            finally:
                try:
                    os.close(fd)
                except OSError as exc:
                    raise GateRepositoryError("gate head lock could not be closed") from exc

    def _load_key_unlocked(self) -> _ChainKey:
        if not self.key_path.exists():
            raise GateRepositoryError("gate head key is missing")
        try:
            return _ChainKey.model_validate(
                _read_private_json(self.key_path, label="gate head key")
            )
        except (ValidationError, ValueError) as exc:
            raise GateRepositoryError("gate head key is invalid") from exc

    def _load_or_create_key_unlocked(self) -> _ChainKey:
        if self.key_path.exists():
            return self._load_key_unlocked()
        key = _ChainKey(hmac_key=secrets.token_hex(32))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.key_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    key.model_dump(mode="json"),
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise GateRepositoryError("gate head key could not be created") from exc
        return key

    def _read_registry_unlocked(self) -> dict:
        if not self.registry_path.exists():
            return {"schema_version": _REGISTRY_SCHEMA_VERSION, "stores": {}}
        try:
            payload = _read_private_json(self.registry_path, label="gate head registry")
        except (OSError, UnicodeError, json.JSONDecodeError, GateRepositoryError) as exc:
            raise GateRepositoryError("gate head registry could not be trusted") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _REGISTRY_SCHEMA_VERSION
            or not isinstance(payload.get("stores"), dict)
        ):
            raise GateRepositoryError("gate head registry has an unsupported structure")
        return payload

    def _read_registry_for_migration_unlocked(self) -> dict:
        if not self.registry_path.exists():
            return {"schema_version": _REGISTRY_SCHEMA_VERSION, "stores": {}}
        try:
            payload = _read_private_json(self.registry_path, label="gate head registry")
        except (OSError, UnicodeError, json.JSONDecodeError, GateRepositoryError) as exc:
            raise GateRepositoryError("gate head registry could not be trusted") from exc
        is_schema1 = (
            isinstance(payload, dict)
            and payload.get("schema_version") == "1.0"
            and isinstance(payload.get("gates"), dict)
            and set(payload) == {"schema_version", "gates"}
        )
        is_schema2 = (
            isinstance(payload, dict)
            and payload.get("schema_version") == _REGISTRY_SCHEMA_VERSION
            and isinstance(payload.get("stores"), dict)
        )
        if not is_schema1 and not is_schema2:
            raise GateRepositoryError("gate head registry has an unsupported structure")
        return payload

    def _write_registry_unlocked(self, payload: dict) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{_CHAIN_REGISTRY_NAME}.",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.registry_path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise GateRepositoryError("gate head registry could not be written atomically") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    raise GateRepositoryError(
                        "gate head registry temporary file could not be removed"
                    ) from exc

    @staticmethod
    def _head_payload(store_id: str, gate_id: UUID, head: _ChainHead | dict) -> dict:
        values = head.model_dump(mode="json") if isinstance(head, _ChainHead) else dict(head)
        values.pop("head_mac", None)
        return {"store_identity": store_id, "gate_id": str(gate_id), **values}

    @classmethod
    def _head_mac(
        cls, key: _ChainKey, store_id: str, gate_id: UUID, head: _ChainHead | dict
    ) -> str:
        return hmac.new(
            bytes.fromhex(key.hmac_key),
            _canonical_json(cls._head_payload(store_id, gate_id, head)),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _schema1_head_payload(gate_id: UUID, head: _ChainHead | dict) -> dict:
        values = head.model_dump(mode="json") if isinstance(head, _ChainHead) else dict(head)
        values.pop("head_mac", None)
        return {"gate_id": str(gate_id), **values}

    @classmethod
    def _schema1_head_mac(cls, key: _ChainKey, gate_id: UUID, head: _ChainHead | dict) -> str:
        return hmac.new(
            bytes.fromhex(key.hmac_key),
            _canonical_json(cls._schema1_head_payload(gate_id, head)),
            hashlib.sha256,
        ).hexdigest()

    @classmethod
    def _heads(
        cls,
        key: _ChainKey,
        store_id: str,
        histories: dict[str, tuple[GateRevision, ...]],
    ) -> dict[str, dict]:
        heads: dict[str, dict] = {}
        for gate_key, history in histories.items():
            current = history[-1]
            unsigned = {
                "revision_id": str(current.revision_id),
                "revision_number": current.revision_number,
                "chain_digest": revision_chain_digest(history),
                "key_id": key.key_id,
            }
            heads[gate_key] = {
                **unsigned,
                "head_mac": cls._head_mac(key, store_id, UUID(gate_key), unsigned),
            }
        return heads

    @staticmethod
    def _namespace_payload(store_id: str, namespace: _StoreNamespace | dict) -> dict:
        values = (
            namespace.model_dump(mode="json")
            if isinstance(namespace, _StoreNamespace)
            else dict(namespace)
        )
        values.pop("namespace_mac", None)
        return {"store_identity": store_id, **values}

    @classmethod
    def _namespace_mac(
        cls, key: _ChainKey, store_id: str, namespace: _StoreNamespace | dict
    ) -> str:
        return hmac.new(
            bytes.fromhex(key.hmac_key),
            _canonical_json(cls._namespace_payload(store_id, namespace)),
            hashlib.sha256,
        ).hexdigest()

    @classmethod
    def _signed_namespace(cls, key: _ChainKey, store_id: str, values: dict) -> dict:
        unsigned = dict(values)
        unsigned.pop("namespace_mac", None)
        return {**unsigned, "namespace_mac": cls._namespace_mac(key, store_id, unsigned)}

    @classmethod
    def _validated_namespace(
        cls, key: _ChainKey, store_id: str, raw_namespace: object
    ) -> _StoreNamespace:
        try:
            namespace = _StoreNamespace.model_validate(raw_namespace)
        except ValidationError as exc:
            raise GateRepositoryError("gate store namespace is invalid") from exc
        expected_mac = cls._namespace_mac(key, store_id, namespace)
        if not hmac.compare_digest(namespace.namespace_mac, expected_mac):
            raise GateRepositoryError("gate store namespace integrity check failed")
        for gate_key, raw_head in namespace.heads.items():
            try:
                gate_id = UUID(gate_key)
                head = _ChainHead.model_validate(raw_head)
            except (ValueError, ValidationError) as exc:
                raise GateRepositoryError("gate head registry contains an invalid entry") from exc
            expected_head_mac = cls._head_mac(key, store_id, gate_id, head)
            if not hmac.compare_digest(head.head_mac, expected_head_mac):
                raise GateRepositoryError("gate head registry integrity check failed")
        return namespace

    def verify_or_recover(
        self,
        *,
        store_path: Path,
        store_present: bool,
        payload: dict,
        histories: dict[str, tuple[GateRevision, ...]],
    ) -> None:
        store_id = store_identity(store_path)
        current_digest = store_payload_digest(payload)
        with self._lock():
            registry = self._read_registry_unlocked()
            raw_namespace = registry["stores"].get(store_id)
            if raw_namespace is None:
                if histories:
                    raise GateRepositoryError(
                        "non-empty gate store requires explicit integrity migration"
                    )
                return
            key = self._load_key_unlocked()
            namespace = self._validated_namespace(key, store_id, raw_namespace)
            if namespace.pending is not None:
                pending = namespace.pending
                matches_old = (
                    store_present == pending.old_store_present
                    and current_digest == pending.old_store_digest
                )
                matches_new = store_present and current_digest == pending.new_store_digest
                if matches_old:
                    if not pending.old_store_present:
                        del registry["stores"][store_id]
                        self._write_registry_unlocked(registry)
                        return
                    values = {
                        "key_id": key.key_id,
                        "store_digest": pending.old_store_digest,
                        "heads": pending.old_heads,
                        "pending": None,
                    }
                elif matches_new:
                    values = {
                        "key_id": key.key_id,
                        "store_digest": pending.new_store_digest,
                        "heads": pending.new_heads,
                        "pending": None,
                    }
                else:
                    raise GateRepositoryError(
                        "gate store does not match either side of the pending transaction"
                    )
                registry["stores"][store_id] = self._signed_namespace(key, store_id, values)
                self._write_registry_unlocked(registry)
                namespace = self._validated_namespace(key, store_id, registry["stores"][store_id])
            if not store_present:
                raise GateRepositoryError("anchored gate store file is missing")
            expected_heads = self._heads(key, store_id, histories)
            if namespace.store_digest != current_digest or namespace.heads != expected_heads:
                raise GateRepositoryError("gate store does not match its trusted namespace")

    def prepare_update(
        self,
        *,
        store_path: Path,
        old_store_present: bool,
        old_payload: dict,
        old_histories: dict[str, tuple[GateRevision, ...]],
        new_payload: dict,
        new_histories: dict[str, tuple[GateRevision, ...]],
    ) -> UUID:
        store_id = store_identity(store_path)
        transaction_id = uuid4()
        try:
            with self._lock():
                registry_present = self.registry_path.exists()
                registry = self._read_registry_unlocked()
                if not registry_present and self.key_path.exists():
                    raise GateRepositoryError(
                        "gate head registry is missing while gate head key exists"
                    )
                key = (
                    self._load_key_unlocked()
                    if registry_present
                    else self._load_or_create_key_unlocked()
                )
                old_digest = store_payload_digest(old_payload)
                old_heads = self._heads(key, store_id, old_histories)
                raw_namespace = registry["stores"].get(store_id)
                if raw_namespace is not None:
                    namespace = self._validated_namespace(key, store_id, raw_namespace)
                    if namespace.pending is not None:
                        raise GateRepositoryError("gate store already has a pending transaction")
                    if namespace.store_digest != old_digest or namespace.heads != old_heads:
                        raise GateRepositoryError("gate store changed before transaction prepare")
                elif old_histories:
                    raise GateRepositoryError(
                        "non-empty gate store requires explicit integrity migration"
                    )
                pending = _PendingTransaction(
                    transaction_id=transaction_id,
                    old_store_present=old_store_present,
                    old_store_digest=old_digest,
                    old_heads=old_heads,
                    new_store_digest=store_payload_digest(new_payload),
                    new_heads=self._heads(key, store_id, new_histories),
                )
                values = {
                    "key_id": key.key_id,
                    "store_digest": old_digest,
                    "heads": old_heads,
                    "pending": pending.model_dump(mode="json"),
                }
                registry["stores"][store_id] = self._signed_namespace(key, store_id, values)
                self._write_registry_unlocked(registry)
        except OSError as exc:
            raise GateRepositoryError("gate transaction could not be prepared") from exc
        return transaction_id

    def finalize_update(self, *, store_path: Path, transaction_id: UUID) -> None:
        store_id = store_identity(store_path)
        try:
            with self._lock():
                registry = self._read_registry_unlocked()
                key = self._load_key_unlocked()
                namespace = self._validated_namespace(
                    key, store_id, registry["stores"].get(store_id)
                )
                pending = namespace.pending
                if pending is None or pending.transaction_id != transaction_id:
                    raise GateRepositoryError("gate transaction cannot be finalized")
                values = {
                    "key_id": key.key_id,
                    "store_digest": pending.new_store_digest,
                    "heads": pending.new_heads,
                    "pending": None,
                }
                registry["stores"][store_id] = self._signed_namespace(key, store_id, values)
                self._write_registry_unlocked(registry)
        except OSError as exc:
            raise GateRepositoryError("gate transaction could not be finalized") from exc

    @classmethod
    def _validate_schema1_heads(
        cls,
        key: _ChainKey,
        raw_heads: dict,
        histories: dict[str, tuple[GateRevision, ...]],
        raw_histories: dict[str, list[dict]],
    ) -> None:
        if set(raw_heads) != set(histories):
            raise GateRepositoryError("schema 1 registry and legacy store contain different gates")
        for gate_key, history in histories.items():
            raw_history = raw_histories.get(gate_key)
            if not isinstance(raw_history, list):
                raise GateRepositoryError("schema 1 store history is invalid")
            for raw_revision in raw_history:
                if not isinstance(raw_revision, dict):
                    raise GateRepositoryError("schema 1 store revision is invalid")
                approval = raw_revision.get("approval")
                if isinstance(approval, dict) and "mac_version" in approval:
                    raise GateRepositoryError("schema 1 approvals must not declare a MAC version")
            try:
                gate_id = UUID(gate_key)
                head = _ChainHead.model_validate(raw_heads[gate_key])
            except (KeyError, ValueError, ValidationError) as exc:
                raise GateRepositoryError("schema 1 registry contains an invalid head") from exc
            expected_mac = cls._schema1_head_mac(key, gate_id, head)
            current = history[-1]
            if not hmac.compare_digest(head.head_mac, expected_mac):
                raise GateRepositoryError("schema 1 registry integrity check failed")
            if (
                head.revision_id != current.revision_id
                or head.revision_number != current.revision_number
                or head.chain_digest != raw_revision_chain_digest(raw_history)
            ):
                raise GateRepositoryError(
                    "legacy gate history does not match the schema 1 trusted head"
                )

    def migrate_store_namespace(
        self,
        *,
        store_path: Path,
        payload: dict,
        histories: dict[str, tuple[GateRevision, ...]],
        dry_run: bool,
    ) -> None:
        store_id = store_identity(store_path)
        with self._lock():
            registry_present = self.registry_path.exists()
            registry = self._read_registry_for_migration_unlocked()
            if registry["schema_version"] == "1.0":
                key = self._load_key_unlocked()
                self._validate_schema1_heads(
                    key,
                    registry["gates"],
                    histories,
                    payload["gates"],
                )
                target_registry = {"schema_version": _REGISTRY_SCHEMA_VERSION, "stores": {}}
            else:
                target_registry = registry
                if not registry_present and self.key_path.exists():
                    raise GateRepositoryError(
                        "gate head registry is missing while gate head key exists"
                    )
                if not registry_present and dry_run:
                    return
                key = (
                    self._load_key_unlocked()
                    if registry_present
                    else self._load_or_create_key_unlocked()
                )
            values = {
                "key_id": key.key_id,
                "store_digest": store_payload_digest(payload),
                "heads": self._heads(key, store_id, histories),
                "pending": None,
            }
            desired_namespace = self._signed_namespace(key, store_id, values)
            existing_namespace = target_registry["stores"].get(store_id)
            if existing_namespace is not None:
                validated = self._validated_namespace(key, store_id, existing_namespace)
                if validated.model_dump(mode="json") != desired_namespace:
                    raise GateRepositoryError(
                        "gate store already has a different integrity namespace"
                    )
                return
            if dry_run:
                return
            target_registry["stores"][store_id] = desired_namespace
            self._write_registry_unlocked(target_registry)
