"""Process-safe Phase 0 JSON repository for media retention and deletion."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.shared.models import (
    DeletionRequest,
    DeletionRequestStatus,
    DeletionScope,
    MediaDeletionStatus,
    RetainedMediaAsset,
)

STORE_SCHEMA_VERSION = "2.0"
LEGACY_STORE_SCHEMA_VERSION = "1.0"
T = TypeVar("T")


def _idempotency_token(user_id: str, idempotency_key: str) -> str:
    canonical = json.dumps(
        [user_id, idempotency_key], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class RetentionError(RuntimeError):
    """Base retention-domain error."""


class RetentionRepositoryError(RetentionError):
    """The retention store cannot be trusted or persisted."""


class RetentionConflictError(RetentionError):
    """A uniqueness, idempotency, or optimistic-write constraint failed."""


class RetentionOwnershipError(RetentionError):
    """The authenticated user does not own the complete requested scope."""


class RetentionNotFoundError(RetentionError):
    """A requested private record is unavailable."""


class MediaUnavailableError(RetentionError):
    """A media locator must not be returned for this asset."""


class JsonRetentionRepository:
    """Versioned JSON inventory with atomic writes and an idempotency index."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock_path = self.path.with_name(f".{self.path.name}-lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "revision": 0,
            "assets": {},
            "requests": {},
            "idempotency": {},
            "aggregates": {},
        }

    @contextmanager
    def _lock(self) -> Iterator[None]:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            pass
        try:
            parent = self.path.parent.lstat()
            if (
                stat.S_ISLNK(parent.st_mode)
                or not stat.S_ISDIR(parent.st_mode)
                or parent.st_uid != os.geteuid()
                or stat.S_IMODE(parent.st_mode) != 0o700
            ):
                raise RetentionRepositoryError("retention directory is unsafe")
            if self._lock_path.is_symlink() or self.path.is_symlink():
                raise RetentionRepositoryError("retention paths must not be symlinks")
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self._lock_path, flags, 0o600)
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise RetentionRepositoryError("retention lock is unsafe")
        except OSError as exc:
            raise RetentionRepositoryError("retention lock could not be opened") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        try:
            fd = os.open(self.path, flags)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise RetentionRepositoryError("retention store is unsafe")
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise RetentionRepositoryError("retention store must be mode 0600")
            with os.fdopen(fd, encoding="utf-8") as handle:
                fd = -1
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RetentionRepositoryError("retention store could not be trusted") from exc
        finally:
            if fd >= 0:
                os.close(fd)
        self._validate(payload)
        return payload

    @staticmethod
    def _escape_pointer_token(token: str) -> str:
        return token.replace("~", "~0").replace("/", "~1")

    @classmethod
    def _discover_media_references(
        cls, value: Any, assets: dict[str, RetainedMediaAsset], *, path: str = ""
    ) -> dict[str, set[str]]:
        tokens: dict[str, set[str]] = {}
        for asset_id, asset in assets.items():
            for token in (asset_id, asset.content_sha256, asset.storage_locator):
                if token:
                    tokens.setdefault(token, set()).add(asset_id)
        discovered: dict[str, set[str]] = {}

        def record(location: str, candidate: str) -> None:
            matched = {
                asset_id
                for token, asset_ids in tokens.items()
                if token in candidate
                for asset_id in asset_ids
            }
            if matched:
                discovered.setdefault(location, set()).update(matched)

        def walk(item: Any, location: str) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    child = f"{location}/{cls._escape_pointer_token(str(key))}"
                    record(child, str(key))
                    walk(nested, child)
            elif isinstance(item, list):
                for index, nested in enumerate(item):
                    walk(nested, f"{location}/{index}")
            elif isinstance(item, str):
                record(location, item)

        walk(value, path)
        return discovered

    @classmethod
    def _validate_aggregate(
        cls,
        aggregate: object,
        assets: dict[str, RetainedMediaAsset],
        error_type: type[RetentionError],
    ) -> None:
        if not isinstance(aggregate, dict) or not isinstance(aggregate.get("owner_user_id"), str):
            raise error_type("invalid aggregate record")
        if not isinstance(aggregate.get("data"), dict):
            raise error_type("invalid aggregate data")
        media_ids = aggregate.get("media_asset_ids")
        paths = aggregate.get("media_ref_paths")
        if not isinstance(media_ids, list) or not all(isinstance(item, str) for item in media_ids):
            raise error_type("invalid aggregate media references")
        if len(media_ids) != len(set(media_ids)):
            raise error_type("aggregate media references must be unique")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise error_type("invalid aggregate media reference paths")
        if len(paths) != len(set(paths)):
            raise error_type("aggregate media reference paths must be unique")
        indexed = set(media_ids)
        for asset_id in indexed:
            asset = assets.get(asset_id)
            if asset is None:
                raise error_type("aggregate references an unknown media asset")
            if aggregate["owner_user_id"] not in asset.owner_scope.subject_user_ids:
                raise error_type("aggregate media owner does not match")
        discovered = cls._discover_media_references(aggregate["data"], assets)
        discovered_ids = set().union(*discovered.values()) if discovered else set()
        if not discovered_ids.issubset(indexed):
            raise error_type("aggregate contains an unindexed media reference")
        if set(discovered) != set(paths):
            raise error_type("aggregate media paths do not match detected references")

    @classmethod
    def _validate(cls, payload: object) -> None:
        if not isinstance(payload, dict) or payload.get("schema_version") not in {
            LEGACY_STORE_SCHEMA_VERSION,
            STORE_SCHEMA_VERSION,
        }:
            raise RetentionRepositoryError("unsupported retention store schema")
        if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
            raise RetentionRepositoryError("invalid retention store revision")
        for key in ("assets", "requests", "idempotency", "aggregates"):
            if not isinstance(payload.get(key), dict):
                raise RetentionRepositoryError(f"invalid retention {key} structure")
        try:
            assets = {
                key: RetainedMediaAsset.model_validate(value)
                for key, value in payload["assets"].items()
            }
            requests = {
                key: DeletionRequest.model_validate(value)
                for key, value in payload["requests"].items()
            }
        except ValidationError as exc:
            raise RetentionRepositoryError("retention store contains invalid records") from exc
        if any(str(value.asset_id) != key for key, value in assets.items()):
            raise RetentionRepositoryError("asset key does not match asset ID")
        for asset in assets.values():
            for source_id in asset.source_asset_ids:
                source = assets.get(str(source_id))
                if source is None:
                    raise RetentionRepositoryError("derived asset has an unknown source")
                if source.owner_scope != asset.owner_scope or source.session_id != asset.session_id:
                    raise RetentionRepositoryError("derived asset crosses owner or session")
        for asset in assets.values():
            pending = list(asset.source_asset_ids)
            visited: set[UUID] = set()
            while pending:
                source_id = pending.pop()
                if source_id == asset.asset_id:
                    raise RetentionRepositoryError("media derivation graph contains a cycle")
                if source_id in visited:
                    continue
                visited.add(source_id)
                pending.extend(assets[str(source_id)].source_asset_ids)
        active_locators = [
            asset.storage_locator
            for asset in assets.values()
            if asset.deletion_status is not MediaDeletionStatus.DELETED
        ]
        if len(active_locators) != len(set(active_locators)):
            raise RetentionRepositoryError("non-deleted storage locators must be globally unique")
        if any(str(value.request_id) != key for key, value in requests.items()):
            raise RetentionRepositoryError("request key does not match request ID")
        for token, record in payload["idempotency"].items():
            if not isinstance(token, str) or not isinstance(record, dict):
                raise RetentionRepositoryError("invalid idempotency record")
            if record.get("request_id") not in payload["requests"]:
                raise RetentionRepositoryError("idempotency record points to no request")
            request = requests[record["request_id"]]
            if token != _idempotency_token(request.user_id, request.idempotency_key):
                raise RetentionRepositoryError("idempotency identity does not match request")
            if record.get("scope_key") != request.scope.canonical_key:
                raise RetentionRepositoryError("idempotency scope does not match request")
        indexed_request_ids = {record["request_id"] for record in payload["idempotency"].values()}
        if indexed_request_ids != set(payload["requests"]):
            raise RetentionRepositoryError("every request must have exactly one idempotency index")
        for aggregate in payload["aggregates"].values():
            cls._validate_aggregate(aggregate, assets, RetentionRepositoryError)

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self._validate(payload)
        temp_path: Path | None = None
        replaced = False
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                json.dump(
                    payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            replaced = True
            directory_fd = -1
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                os.fsync(directory_fd)
            except OSError as exc:
                warnings.warn(
                    f"retention store was replaced but directory fsync failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            finally:
                if directory_fd >= 0:
                    os.close(directory_fd)
        except OSError as exc:
            if replaced:
                warnings.warn(
                    f"retention store was replaced but durability confirmation failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise RetentionRepositoryError("retention store write failed") from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _mutate(self, operation: Callable[[dict[str, Any]], T]) -> T:
        with self._lock():
            payload = self._read_unlocked()
            payload["schema_version"] = STORE_SCHEMA_VERSION
            result = operation(payload)
            # Phase 0 serializes writers with flock. ``revision`` is a monotonic
            # audit counter, not a cross-host or optimistic-CAS token.
            payload["revision"] += 1
            self._write_unlocked(payload)
            return result

    def _snapshot(self) -> dict[str, Any]:
        with self._lock():
            return self._read_unlocked()

    @staticmethod
    def _request_blocks_registration(
        payload: dict[str, Any], request: DeletionRequest, asset: RetainedMediaAsset
    ) -> bool:
        if request.user_id not in asset.owner_scope.subject_user_ids or request.status not in {
            DeletionRequestStatus.RECEIVED,
            DeletionRequestStatus.IN_PROGRESS,
            DeletionRequestStatus.COMPLETED,
        }:
            return False
        if request.scope.all_user_media:
            return (
                request.status is not DeletionRequestStatus.COMPLETED
                or asset.captured_at <= request.created_at
            )
        if asset.session_id in request.scope.session_ids:
            return True
        if asset.asset_id in request.frozen_asset_ids:
            return True
        if set(asset.source_asset_ids).intersection(request.frozen_asset_ids):
            return True
        frozen_sessions = {
            RetainedMediaAsset.model_validate(payload["assets"][str(asset_id)]).session_id
            for asset_id in request.frozen_asset_ids
            if str(asset_id) in payload["assets"]
        }
        return bool(
            asset.session_id in frozen_sessions
            and (request.scope.session_ids or request.status is DeletionRequestStatus.COMPLETED)
        )

    @classmethod
    def _insert_asset(
        cls,
        payload: dict[str, Any],
        asset: RetainedMediaAsset,
        *,
        enforce_deletion_boundaries: bool,
    ) -> RetainedMediaAsset:
        key = str(asset.asset_id)
        if key in payload["assets"]:
            existing = RetainedMediaAsset.model_validate(payload["assets"][key])
            if existing == asset:
                return existing
            raise RetentionConflictError("asset ID already exists")
        known = payload["assets"]
        for raw in known.values():
            existing = RetainedMediaAsset.model_validate(raw)
            if (
                existing.deletion_status is not MediaDeletionStatus.DELETED
                and existing.storage_locator == asset.storage_locator
            ):
                raise RetentionConflictError("storage locator is already registered")
            if (
                enforce_deletion_boundaries
                and set(existing.owner_scope.subject_user_ids).intersection(
                    asset.owner_scope.subject_user_ids
                )
                and existing.content_sha256 == asset.content_sha256
                and existing.deletion_status is MediaDeletionStatus.DELETED
            ):
                raise MediaUnavailableError("deleted content cannot be registered again")
        for source_id in asset.source_asset_ids:
            source_raw = known.get(str(source_id))
            if source_raw is None:
                raise RetentionConflictError("source asset is not registered")
            source = RetainedMediaAsset.model_validate(source_raw)
            if source.owner_scope != asset.owner_scope or source.session_id != asset.session_id:
                raise RetentionOwnershipError("derived asset must share owner and session")
        if enforce_deletion_boundaries:
            for raw in payload["requests"].values():
                request = DeletionRequest.model_validate(raw)
                if cls._request_blocks_registration(payload, request, asset):
                    raise MediaUnavailableError("deletion scope blocks media registration")
        payload["assets"][key] = asset.model_dump(mode="json")
        return asset

    def _register_authorized_asset(self, asset: RetainedMediaAsset) -> RetainedMediaAsset:
        """Persist a Gate-authorized asset under the deletion-scope flock boundary."""

        return self._mutate(
            lambda payload: self._insert_asset(payload, asset, enforce_deletion_boundaries=True)
        )

    def register_pending_binding(self, asset: RetainedMediaAsset) -> RetainedMediaAsset:
        """Register an inaccessible saga record before the session CAS linearization point."""

        if asset.deletion_status is not MediaDeletionStatus.PENDING_BINDING:
            raise RetentionConflictError("pending registration requires PENDING_BINDING")
        return self._mutate(
            lambda payload: self._insert_asset(payload, asset, enforce_deletion_boundaries=True)
        )

    def activate_pending_binding(
        self, asset_id: UUID, *, expected_session_id: UUID
    ) -> RetainedMediaAsset:
        """Idempotently activate only the deterministic session-owned pending record."""

        def operation(payload: dict[str, Any]) -> RetainedMediaAsset:
            raw = payload["assets"].get(str(asset_id))
            if raw is None:
                raise RetentionNotFoundError("asset was not found")
            current = RetainedMediaAsset.model_validate(raw)
            if current.session_id != expected_session_id:
                raise RetentionOwnershipError("pending asset session mismatch")
            if current.deletion_status is MediaDeletionStatus.ACTIVE:
                return current
            if current.deletion_status is not MediaDeletionStatus.PENDING_BINDING:
                raise RetentionConflictError("pending asset cannot be activated")
            updated = RetainedMediaAsset.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "deletion_status": MediaDeletionStatus.ACTIVE,
                    "updated_at": datetime.now(UTC),
                }
            )
            payload["assets"][str(asset_id)] = updated.model_dump(mode="json")
            return updated

        return self._mutate(operation)

    def abandon_pending_binding(self, asset_id: UUID, *, expected_session_id: UUID) -> None:
        """Compensate only a never-activated record with no accepted session binding."""

        def operation(payload: dict[str, Any]) -> None:
            raw = payload["assets"].get(str(asset_id))
            if raw is None:
                return
            asset = RetainedMediaAsset.model_validate(raw)
            if (
                asset.session_id != expected_session_id
                or asset.deletion_status is not MediaDeletionStatus.PENDING_BINDING
            ):
                raise RetentionConflictError("retention record cannot be abandoned")
            del payload["assets"][str(asset_id)]

        self._mutate(operation)

    def quarantine_unbound_asset(self, asset_id: UUID, *, expected_session_id: UUID) -> None:
        """Compensate a saga asset that never reached the session MEDIA_BOUND gate."""

        def operation(payload: dict[str, Any]) -> None:
            raw = payload["assets"].get(str(asset_id))
            if raw is None:
                return
            asset = RetainedMediaAsset.model_validate(raw)
            if asset.session_id != expected_session_id or asset.deletion_status not in {
                MediaDeletionStatus.PENDING_BINDING,
                MediaDeletionStatus.ACTIVE,
            }:
                raise RetentionConflictError("retention record cannot be quarantined")
            del payload["assets"][str(asset_id)]

        self._mutate(operation)

    def _insert_asset_for_testing(self, asset: RetainedMediaAsset) -> RetainedMediaAsset:
        """Seed synthetic inventories without representing a production ingest API."""

        return self._mutate(
            lambda payload: self._insert_asset(payload, asset, enforce_deletion_boundaries=False)
        )

    def save_asset(
        self,
        asset: RetainedMediaAsset,
        *,
        expected_current: RetainedMediaAsset | None = None,
    ) -> RetainedMediaAsset:
        def operation(payload: dict[str, Any]) -> RetainedMediaAsset:
            raw = payload["assets"].get(str(asset.asset_id))
            if raw is None:
                raise RetentionNotFoundError("asset was not found")
            previous = RetainedMediaAsset.model_validate(raw)
            if expected_current is not None and previous != expected_current:
                raise RetentionConflictError("asset changed before update")
            immutable_fields = (
                "asset_id",
                "session_id",
                "owner_scope",
                "kind",
                "content_sha256",
                "captured_at",
                "source_asset_ids",
                "created_at",
            )
            if any(getattr(previous, field) != getattr(asset, field) for field in immutable_fields):
                raise RetentionConflictError("asset identity and provenance are immutable")
            if previous.retention_expires_at is not None and (
                asset.retention_expires_at != previous.retention_expires_at
            ):
                raise RetentionConflictError("active retention deadline is immutable")
            for milestone in ("reviewed_at", "delivered_at"):
                previous_value = getattr(previous, milestone)
                if previous_value is not None and getattr(asset, milestone) != previous_value:
                    raise RetentionConflictError("recorded retention milestone is immutable")
            allowed_status = {
                MediaDeletionStatus.PENDING_BINDING: {
                    MediaDeletionStatus.PENDING_BINDING,
                    MediaDeletionStatus.ACTIVE,
                    MediaDeletionStatus.PENDING_DELETION,
                },
                MediaDeletionStatus.ACTIVE: {
                    MediaDeletionStatus.ACTIVE,
                    MediaDeletionStatus.PENDING_DELETION,
                    MediaDeletionStatus.DELETION_FAILED,
                },
                MediaDeletionStatus.PENDING_DELETION: {
                    MediaDeletionStatus.DELETED,
                    MediaDeletionStatus.DELETION_FAILED,
                },
                MediaDeletionStatus.DELETION_FAILED: {
                    MediaDeletionStatus.PENDING_DELETION,
                    MediaDeletionStatus.DELETION_FAILED,
                },
                MediaDeletionStatus.DELETED: {MediaDeletionStatus.DELETED},
            }
            if asset.deletion_status not in allowed_status[previous.deletion_status]:
                raise RetentionConflictError("invalid asset deletion-state transition")
            if previous.storage_locator != asset.storage_locator and (
                asset.deletion_status is not MediaDeletionStatus.DELETED
                or asset.storage_locator is not None
            ):
                raise RetentionConflictError("asset locator can only be cleared on deletion")
            payload["assets"][str(asset.asset_id)] = asset.model_dump(mode="json")
            return asset

        return self._mutate(operation)

    def get_asset(self, asset_id: UUID) -> RetainedMediaAsset:
        raw = self._snapshot()["assets"].get(str(asset_id))
        if raw is None:
            raise RetentionNotFoundError("asset was not found")
        return RetainedMediaAsset.model_validate(raw)

    def list_assets(self) -> tuple[RetainedMediaAsset, ...]:
        return tuple(
            RetainedMediaAsset.model_validate(item) for item in self._snapshot()["assets"].values()
        )

    def _get_active_asset_with_locator(self, asset_id: UUID) -> tuple[RetainedMediaAsset, str]:
        asset = self.get_asset(asset_id)
        if asset.deletion_status is not MediaDeletionStatus.ACTIVE or asset.storage_locator is None:
            raise MediaUnavailableError("media is not available")
        return asset, asset.storage_locator

    @staticmethod
    def _resolve_scope_in_payload(
        payload: dict[str, Any], user_id: str, scope: DeletionScope
    ) -> tuple[UUID, ...]:
        assets = tuple(
            asset
            for raw in payload["assets"].values()
            if (asset := RetainedMediaAsset.model_validate(raw)).deletion_status
            is not MediaDeletionStatus.PENDING_BINDING
        )
        by_id = {asset.asset_id: asset for asset in assets}
        selected: set[UUID] = set()
        if scope.all_user_media:
            selected.update(
                asset.asset_id for asset in assets if user_id in asset.owner_scope.subject_user_ids
            )
        else:
            for session_id in scope.session_ids:
                session_assets = [asset for asset in assets if asset.session_id == session_id]
                if not session_assets:
                    raise RetentionNotFoundError("deletion scope was not found")
                if any(
                    user_id not in asset.owner_scope.subject_user_ids for asset in session_assets
                ):
                    raise RetentionOwnershipError("deletion scope is not wholly owned")
                selected.update(asset.asset_id for asset in session_assets)
            for asset_id in scope.asset_ids:
                asset = by_id.get(asset_id)
                if asset is None:
                    raise RetentionNotFoundError("deletion scope was not found")
                if user_id not in asset.owner_scope.subject_user_ids:
                    raise RetentionOwnershipError("deletion scope is not wholly owned")
                selected.add(asset_id)
        changed = True
        while changed:
            changed = False
            for asset in assets:
                if asset.asset_id not in selected and selected.intersection(asset.source_asset_ids):
                    if user_id not in asset.owner_scope.subject_user_ids:
                        raise RetentionOwnershipError("derived deletion scope is not wholly owned")
                    selected.add(asset.asset_id)
                    changed = True
        return tuple(sorted(selected, key=str))

    def create_request(self, request: DeletionRequest) -> DeletionRequest:
        token = _idempotency_token(request.user_id, request.idempotency_key)

        def operation(payload: dict[str, Any]) -> DeletionRequest:
            existing_index = payload["idempotency"].get(token)
            if existing_index is not None:
                if existing_index["scope_key"] != request.scope.canonical_key:
                    raise RetentionConflictError("idempotency key was used for another scope")
                return DeletionRequest.model_validate(
                    payload["requests"][existing_index["request_id"]]
                )
            key = str(request.request_id)
            if key in payload["requests"]:
                raise RetentionConflictError("request ID already exists")
            payload["requests"][key] = request.model_dump(mode="json")
            payload["idempotency"][token] = {
                "request_id": key,
                "scope_key": request.scope.canonical_key,
            }
            return request

        return self._mutate(operation)

    def save_request(self, request: DeletionRequest) -> DeletionRequest:
        def operation(payload: dict[str, Any]) -> DeletionRequest:
            raw = payload["requests"].get(str(request.request_id))
            if raw is None:
                raise RetentionNotFoundError("deletion request was not found")
            previous = DeletionRequest.model_validate(raw)
            immutable_fields = (
                "request_id",
                "user_id",
                "scope",
                "idempotency_key",
                "created_at",
            )
            if any(
                getattr(previous, field) != getattr(request, field) for field in immutable_fields
            ):
                raise RetentionConflictError("deletion request identity and scope are immutable")
            if not set(previous.frozen_asset_ids).issubset(request.frozen_asset_ids):
                raise RetentionConflictError("frozen deletion scope is append-only")
            if request.attempts[: len(previous.attempts)] != previous.attempts:
                raise RetentionConflictError("deletion attempts are append-only")
            if request.events[: len(previous.events)] != previous.events:
                raise RetentionConflictError("deletion events are append-only")
            allowed_status = {
                DeletionRequestStatus.RECEIVED: {DeletionRequestStatus.IN_PROGRESS},
                DeletionRequestStatus.IN_PROGRESS: {
                    DeletionRequestStatus.COMPLETED,
                    DeletionRequestStatus.FAILED,
                },
                DeletionRequestStatus.FAILED: {DeletionRequestStatus.IN_PROGRESS},
                DeletionRequestStatus.COMPLETED: set(),
            }
            if request.status not in allowed_status[previous.status]:
                raise RetentionConflictError("invalid deletion-request state transition")
            payload["requests"][str(request.request_id)] = request.model_dump(mode="json")
            return request

        return self._mutate(operation)

    def finalize_completed_request(self, request: DeletionRequest) -> DeletionRequest:
        """Linearize final scope validation and the terminal COMPLETED write."""

        if request.status is not DeletionRequestStatus.COMPLETED:
            raise RetentionConflictError("only a completed request can be finalized")

        def operation(payload: dict[str, Any]) -> DeletionRequest:
            raw = payload["requests"].get(str(request.request_id))
            if raw is None:
                raise RetentionNotFoundError("deletion request was not found")
            previous = DeletionRequest.model_validate(raw)
            if previous.status is not DeletionRequestStatus.IN_PROGRESS:
                raise RetentionConflictError("request is not in progress")
            if previous.frozen_asset_ids != request.frozen_asset_ids:
                raise RetentionConflictError("completed scope differs from claimed scope")
            current_scope = self._resolve_scope_in_payload(
                payload, previous.user_id, previous.scope
            )
            if not set(current_scope).issubset(previous.frozen_asset_ids):
                raise RetentionConflictError("deletion scope expanded before completion")
            for asset_id in previous.frozen_asset_ids:
                asset = RetainedMediaAsset.model_validate(payload["assets"][str(asset_id)])
                if (
                    asset.deletion_status is not MediaDeletionStatus.DELETED
                    or asset.storage_locator is not None
                ):
                    raise RetentionConflictError("deletion scope is not fully deleted")
            if request.attempts[: len(previous.attempts)] != previous.attempts:
                raise RetentionConflictError("deletion attempts are append-only")
            if request.events[: len(previous.events)] != previous.events:
                raise RetentionConflictError("deletion events are append-only")
            payload["requests"][str(request.request_id)] = request.model_dump(mode="json")
            return request

        return self._mutate(operation)

    def expand_in_progress_scope(
        self,
        request_id: UUID,
        *,
        expected_frozen_asset_ids: tuple[UUID, ...],
        expanded_frozen_asset_ids: tuple[UUID, ...],
        updated_at: datetime,
    ) -> DeletionRequest:
        """Grow an IN_PROGRESS scope without opening a same-state execution transition."""

        def operation(payload: dict[str, Any]) -> DeletionRequest:
            raw = payload["requests"].get(str(request_id))
            if raw is None:
                raise RetentionNotFoundError("deletion request was not found")
            current = DeletionRequest.model_validate(raw)
            if current.status is not DeletionRequestStatus.IN_PROGRESS:
                raise RetentionConflictError("only an in-progress request scope can expand")
            if current.frozen_asset_ids != expected_frozen_asset_ids:
                raise RetentionConflictError("deletion scope changed before expansion")
            if not set(current.frozen_asset_ids).issubset(expanded_frozen_asset_ids):
                raise RetentionConflictError("frozen deletion scope cannot shrink")
            expanded = DeletionRequest.model_validate(
                current.model_copy(
                    update={
                        "frozen_asset_ids": tuple(sorted(set(expanded_frozen_asset_ids), key=str)),
                        "updated_at": updated_at,
                    }
                ).model_dump()
            )
            payload["requests"][str(request_id)] = expanded.model_dump(mode="json")
            return expanded

        return self._mutate(operation)

    def get_request(self, request_id: UUID) -> DeletionRequest:
        raw = self._snapshot()["requests"].get(str(request_id))
        if raw is None:
            raise RetentionNotFoundError("deletion request was not found")
        return DeletionRequest.model_validate(raw)

    def list_requests(self) -> tuple[DeletionRequest, ...]:
        return tuple(
            DeletionRequest.model_validate(item) for item in self._snapshot()["requests"].values()
        )

    def register_aggregate(
        self,
        aggregate_id: str,
        owner_user_id: str,
        data: dict[str, Any],
        media_asset_ids: tuple[UUID, ...],
        *,
        media_ref_paths: tuple[str, ...],
    ) -> None:
        def operation(payload: dict[str, Any]) -> None:
            aggregate = {
                "owner_user_id": owner_user_id,
                "data": data,
                "media_asset_ids": [str(item) for item in media_asset_ids],
                "media_ref_paths": list(media_ref_paths),
            }
            assets = {
                asset_id: RetainedMediaAsset.model_validate(raw)
                for asset_id, raw in payload["assets"].items()
            }
            self._validate_aggregate(aggregate, assets, RetentionConflictError)
            payload["aggregates"][aggregate_id] = aggregate

        self._mutate(operation)

    @staticmethod
    def _pointer_tokens(path: str) -> tuple[str, ...]:
        if not path.startswith("/") or path == "/":
            raise RetentionConflictError("media reference path must be a JSON pointer")
        return tuple(token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/"))

    @classmethod
    def _resolve_json_pointer(cls, data: Any, path: str) -> Any:
        current = data
        for token in cls._pointer_tokens(path):
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                current = current[int(token)]
            else:
                raise RetentionConflictError("declared media reference path does not exist")
        return current

    @classmethod
    def _remove_json_pointer(cls, data: Any, path: str) -> None:
        tokens = cls._pointer_tokens(path)
        current = data
        for token in tokens[:-1]:
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                current = current[int(token)]
            else:
                raise RetentionRepositoryError("declared media reference path became invalid")
        leaf = tokens[-1]
        if isinstance(current, dict) and leaf in current:
            current.pop(leaf)
        elif isinstance(current, list) and leaf.isdigit() and int(leaf) < len(current):
            current.pop(int(leaf))
        else:
            raise RetentionRepositoryError("declared media reference path became invalid")

    @classmethod
    def _assert_no_forbidden_reference(
        cls, value: Any, forbidden: set[str], *, location: str = "$"
    ) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if any(token in str(key) for token in forbidden):
                    raise RetentionRepositoryError(
                        f"undeclared media reference remains at {location} key"
                    )
                cls._assert_no_forbidden_reference(item, forbidden, location=f"{location}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                cls._assert_no_forbidden_reference(item, forbidden, location=f"{location}/{index}")
        elif isinstance(value, str) and any(token in value for token in forbidden):
            raise RetentionRepositoryError(f"undeclared media reference remains at {location}")

    def scrub_media_references(self, asset_ids: tuple[UUID, ...]) -> None:
        deleted = {str(item) for item in asset_ids}

        def operation(payload: dict[str, Any]) -> None:
            forbidden = set(deleted)
            for asset_id in deleted:
                asset = payload["assets"].get(asset_id)
                if asset is not None:
                    forbidden.add(asset["content_sha256"])
                    if asset.get("storage_locator"):
                        forbidden.add(asset["storage_locator"])

            for aggregate in payload["aggregates"].values():
                referenced = set(aggregate.get("media_asset_ids", []))
                if not referenced.intersection(deleted):
                    continue
                for path in sorted(
                    aggregate["media_ref_paths"],
                    key=lambda item: (item.count("/"), item),
                    reverse=True,
                ):
                    self._remove_json_pointer(aggregate["data"], path)
                aggregate["media_ref_paths"] = []
                aggregate["media_asset_ids"] = [
                    item for item in aggregate.get("media_asset_ids", []) if item not in deleted
                ]
                self._assert_no_forbidden_reference(aggregate["data"], forbidden)

        self._mutate(operation)

    def get_aggregate(self, aggregate_id: str) -> dict[str, Any]:
        aggregate = self._snapshot()["aggregates"].get(aggregate_id)
        if aggregate is None:
            raise RetentionNotFoundError("aggregate was not found")
        return aggregate
