"""Append-only JSON repository for REAL_VIDEO_GATE revisions."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.shared.models import GateRevision, RealVideoGateStatus

from .errors import GateApprovalError, GateConflictError, GateRepositoryError
from .identity import (
    LEGACY_APPROVAL_KEY_ID,
    FileApproverIdentityProvider,
    _read_private_bytes,
    _read_private_json,
    _require_private_directory,
)
from .integrity import GateHeadRegistry, raw_revision_chain_digest, revision_chain_digest

STORE_SCHEMA_VERSION = "1.0"
_REPOSITORY_WRITER_TOKEN = object()


def _approval_security_payload(
    revision: GateRevision,
    *,
    previous_chain_digest: str,
    mac_version: str | None = None,
) -> dict[str, object]:
    if revision.approval is None:
        raise GateRepositoryError("approved revision is missing approval evidence")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "gate_id": str(revision.gate_id),
        "revision_id": str(revision.revision_id),
        "revision_number": revision.revision_number,
        "supersedes_revision_id": str(revision.supersedes_revision_id),
        "status": revision.status.value,
        "scope_fingerprint": revision.scope.fingerprint,
        "evidence": [item.model_dump(mode="json") for item in revision.evidence],
        "inspector_id": revision.inspector_id,
        "inspected_at": revision.inspected_at.isoformat(),
        "approver_id": revision.approval.approver_id,
        "credential_id": revision.approval.credential_id,
        "key_id": revision.approval.key_id,
        "approved_at": revision.approval.approved_at.isoformat(),
        "approval_evidence_sha256": revision.approval.approval_evidence_sha256,
        "previous_chain_digest": previous_chain_digest,
    }
    if mac_version is not None:
        payload["mac_version"] = mac_version
    return payload


class JsonGateRepository:
    """Append-only, process-safe Phase 0 repository for gate revisions."""

    def __init__(self, path: Path | str):
        try:
            self.path = Path(os.path.abspath(os.fspath(path)))
        except (OSError, RuntimeError, TypeError) as exc:
            raise GateRepositoryError("gate store path could not be resolved") from exc
        self._lock_path = self.path.with_name(f".{self.path.name}-lock")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _require_private_directory(self.path.parent, label="gate store directory")
        except GateRepositoryError:
            raise
        except (OSError, UnicodeError) as exc:
            raise GateRepositoryError("gate store directory could not be trusted") from exc
        try:
            if self._lock_path.is_symlink():
                raise GateRepositoryError("gate lock must not be a symlink")
        except OSError as exc:
            raise GateRepositoryError("gate lock path could not be inspected") from exc
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self._lock_path, flags, 0o600)
        except OSError as exc:
            raise GateRepositoryError("gate lock could not be opened") from exc
        locked = False
        try:
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise GateRepositoryError("gate lock ownership or permissions are unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        except GateRepositoryError:
            raise
        except OSError as exc:
            raise GateRepositoryError("gate lock operation failed") from exc
        finally:
            try:
                if locked:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                raise GateRepositoryError("gate lock could not be released") from exc
            finally:
                try:
                    os.close(fd)
                except OSError as exc:
                    raise GateRepositoryError("gate lock could not be closed") from exc

    def _empty_payload(self) -> dict:
        return {"schema_version": STORE_SCHEMA_VERSION, "gates": {}}

    def _read_unlocked(self) -> dict:
        store_present = self.path.exists()
        payload = self._read_payload_unlocked()
        histories = self._validated_store(payload)
        GateHeadRegistry().verify_or_recover(
            store_path=self.path,
            store_present=store_present,
            payload=payload,
            histories=histories,
        )
        return payload

    def _read_payload_unlocked(self) -> dict:
        if not self.path.exists():
            return self._empty_payload()
        try:
            payload = _read_private_json(self.path, label="gate store")
        except (OSError, UnicodeError, json.JSONDecodeError, GateRepositoryError) as exc:
            raise GateRepositoryError("gate store could not be trusted") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != STORE_SCHEMA_VERSION:
            raise GateRepositoryError("unsupported gate store schema")
        if not isinstance(payload.get("gates"), dict):
            raise GateRepositoryError("invalid gate store structure")
        return payload

    def _write_unlocked(self, payload: dict) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
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
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise GateRepositoryError("gate store could not be written atomically") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    raise GateRepositoryError(
                        "gate store temporary file could not be removed"
                    ) from exc

    @staticmethod
    def _validate_approval_against_previous(revision: GateRevision, previous: GateRevision) -> None:
        if revision.scope != previous.scope:
            raise GateRepositoryError("approval must preserve the inspected gate scope")
        if revision.evidence != previous.evidence:
            raise GateRepositoryError("approval must preserve the inspected evidence")
        if revision.inspector_id != previous.inspector_id:
            raise GateRepositoryError("approval must preserve the inspector identity")
        if revision.inspected_at != previous.inspected_at:
            raise GateRepositoryError("approval must preserve the inspection time")

    @staticmethod
    def _verify_approval_mac(
        revision: GateRevision,
        *,
        mac_algorithm: str,
        normalized_previous_chain_digest: str,
        raw_previous_chain_digest: str,
    ) -> None:
        if revision.approval is None:
            raise GateRepositoryError("approved revision is missing approval evidence")
        try:
            identity = FileApproverIdentityProvider().current_identity()
        except GateApprovalError as exc:
            raise GateRepositoryError("approval credential could not be verified") from exc
        if (
            revision.approval.approver_id != identity.approver_id
            or revision.approval.credential_id != identity.credential_id
        ):
            raise GateRepositoryError("gate approval identity is not trusted")
        try:
            if mac_algorithm == "1.0":
                if revision.approval.key_id != LEGACY_APPROVAL_KEY_ID:
                    raise GateApprovalError("legacy approval has an invalid key id")
                expected_mac = identity.legacy_approval_mac(
                    gate_id=revision.gate_id,
                    revision_id=revision.revision_id,
                    gate_fingerprint=revision.approval.gate_fingerprint,
                    approved_at=revision.approval.approved_at,
                    approval_evidence_sha256=(revision.approval.approval_evidence_sha256),
                )
            elif mac_algorithm in {"2.0", "3.0", "schema1-raw"}:
                previous_chain_digest = (
                    normalized_previous_chain_digest
                    if mac_algorithm == "2.0"
                    else raw_previous_chain_digest
                )
                expected_mac = identity.approval_mac(
                    key_id=revision.approval.key_id,
                    payload=_approval_security_payload(
                        revision,
                        previous_chain_digest=previous_chain_digest,
                        mac_version="3.0" if mac_algorithm == "3.0" else None,
                    ),
                )
            else:
                raise GateRepositoryError("unsupported approval MAC algorithm")
        except GateApprovalError as exc:
            raise GateRepositoryError("approval key could not be verified") from exc
        if not hmac.compare_digest(revision.approval.approval_mac, expected_mac):
            raise GateRepositoryError("gate approval integrity check failed")

    @staticmethod
    def _validated_history(gate_key: str, raw_history: object) -> tuple[GateRevision, ...]:
        if not isinstance(raw_history, list):
            raise GateRepositoryError("gate history is not a list")
        if not raw_history:
            raise GateRepositoryError("gate history must not be empty")
        try:
            history = tuple(GateRevision.model_validate(item) for item in raw_history)
        except ValidationError as exc:
            raise GateRepositoryError("gate history contains an invalid revision") from exc
        for index, revision in enumerate(history, start=1):
            if str(revision.gate_id) != gate_key:
                raise GateRepositoryError("gate history does not match its store key")
            if revision.revision_number != index:
                raise GateRepositoryError("gate revision sequence is not contiguous")
            if index > 1 and revision.supersedes_revision_id != history[index - 2].revision_id:
                raise GateRepositoryError("gate revision chain is broken")
            if index == 1 and revision.status is not RealVideoGateStatus.BLOCKED:
                raise GateRepositoryError("the first gate revision must be BLOCKED")
            if index > 1 and revision.created_at < history[index - 2].created_at:
                raise GateRepositoryError("gate revision timeline moves backwards")
            if revision.status is RealVideoGateStatus.APPROVED:
                previous = history[index - 2] if index > 1 else None
                if previous is None or previous.status is not RealVideoGateStatus.BLOCKED:
                    raise GateRepositoryError("approval must transition from BLOCKED")
                JsonGateRepository._validate_approval_against_previous(revision, previous)
                raw_approval = raw_history[index - 1].get("approval")
                if not isinstance(raw_approval, dict):
                    raise GateRepositoryError("approved revision is missing raw approval evidence")
                if "mac_version" in raw_approval:
                    mac_algorithm = raw_approval["mac_version"]
                elif "key_id" in raw_approval:
                    mac_algorithm = "schema1-raw"
                else:
                    mac_algorithm = "1.0"
                JsonGateRepository._verify_approval_mac(
                    revision,
                    mac_algorithm=mac_algorithm,
                    normalized_previous_chain_digest=revision_chain_digest(history[: index - 1]),
                    raw_previous_chain_digest=raw_revision_chain_digest(raw_history[: index - 1]),
                )
        return history

    @classmethod
    def _validated_store(cls, payload: dict) -> dict[str, tuple[GateRevision, ...]]:
        histories: dict[str, tuple[GateRevision, ...]] = {}
        seen_revision_ids: set[UUID] = set()
        for gate_key, raw_history in payload["gates"].items():
            try:
                UUID(gate_key)
            except (TypeError, ValueError) as exc:
                raise GateRepositoryError("gate store key is not a UUID") from exc
            history = cls._validated_history(gate_key, raw_history)
            for revision in history:
                if revision.revision_id in seen_revision_ids:
                    raise GateRepositoryError("gate revision IDs must be globally unique")
                seen_revision_ids.add(revision.revision_id)
            histories[gate_key] = history
        return histories

    def _append_revision(
        self,
        revision: GateRevision,
        *,
        expected_current_revision_id: UUID | None,
        _writer_token: object | None = None,
    ) -> GateRevision:
        with self._lock():
            old_store_present = self.path.exists()
            payload = self._read_unlocked()
            old_payload = copy.deepcopy(payload)
            old_histories = self._validated_store(old_payload)
            gate_key = str(revision.gate_id)
            raw_history = payload["gates"].get(gate_key)
            history = () if raw_history is None else self._validated_history(gate_key, raw_history)
            current_id = history[-1].revision_id if history else None
            if current_id != expected_current_revision_id:
                raise GateConflictError("gate revision changed before append")
            if revision.revision_number != len(history) + 1:
                raise GateConflictError("gate revision number is stale")
            if history and revision.supersedes_revision_id != current_id:
                raise GateConflictError("revision does not extend current gate state")
            if not history and revision.status is not RealVideoGateStatus.BLOCKED:
                raise GateRepositoryError("new gates must start BLOCKED")
            if revision.status is RealVideoGateStatus.APPROVED:
                if not history or history[-1].status is not RealVideoGateStatus.BLOCKED:
                    raise GateRepositoryError("approval must transition from BLOCKED")
                self._validate_approval_against_previous(revision, history[-1])
                if _writer_token is not _REPOSITORY_WRITER_TOKEN:
                    raise GateRepositoryError(
                        "approved revisions require the trusted service writer"
                    )
                self._verify_approval_mac(
                    revision,
                    mac_algorithm=revision.approval.mac_version,
                    normalized_previous_chain_digest=revision_chain_digest(history),
                    raw_previous_chain_digest=raw_revision_chain_digest(raw_history),
                )
            payload["gates"].setdefault(gate_key, []).append(revision.model_dump(mode="json"))
            histories = self._validated_store(payload)
            registry = GateHeadRegistry()
            transaction_id = registry.prepare_update(
                store_path=self.path,
                old_store_present=old_store_present,
                old_payload=old_payload,
                old_histories=old_histories,
                new_payload=payload,
                new_histories=histories,
            )
            try:
                self._write_unlocked(payload)
            except GateRepositoryError:
                raise
            except OSError as exc:
                raise GateRepositoryError("gate store transaction write failed") from exc
            registry.finalize_update(
                store_path=self.path,
                transaction_id=transaction_id,
            )
        return revision

    def migrate_legacy_store(self, *, expected_store_sha256: str, dry_run: bool) -> None:
        if (
            not isinstance(expected_store_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_store_sha256) is None
        ):
            raise GateRepositoryError(
                "expected store SHA-256 must be 64 lowercase hexadecimal characters"
            )
        with self._lock():
            if not self.path.exists():
                raise GateRepositoryError("legacy gate store does not exist")
            try:
                store_bytes = _read_private_bytes(self.path, label="legacy gate store")
                actual_sha256 = hashlib.sha256(store_bytes).hexdigest()
                if not hmac.compare_digest(actual_sha256, expected_store_sha256):
                    raise GateRepositoryError("legacy gate store digest does not match")
                payload = json.loads(store_bytes.decode("utf-8"))
            except GateRepositoryError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise GateRepositoryError("legacy gate store could not be read") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != STORE_SCHEMA_VERSION
                or not isinstance(payload.get("gates"), dict)
            ):
                raise GateRepositoryError("legacy gate store structure is invalid")
            histories = self._validated_store(payload)
            if not histories:
                raise GateRepositoryError("legacy migration requires a non-empty store")
            GateHeadRegistry().migrate_store_namespace(
                store_path=self.path,
                payload=payload,
                histories=histories,
                dry_run=dry_run,
            )

    def get_history(self, gate_id: UUID) -> tuple[GateRevision, ...]:
        with self._lock():
            payload = self._read_unlocked()
            raw_history = payload["gates"].get(str(gate_id))
            if raw_history is None:
                return ()
            return self._validated_history(str(gate_id), raw_history)

    def _get_history_with_chain_digest(self, gate_id: UUID) -> tuple[tuple[GateRevision, ...], str]:
        with self._lock():
            payload = self._read_unlocked()
            raw_history = payload["gates"].get(str(gate_id))
            if raw_history is None:
                return (), raw_revision_chain_digest(())
            return (
                self._validated_history(str(gate_id), raw_history),
                raw_revision_chain_digest(raw_history),
            )

    def get_current(self, gate_id: UUID) -> GateRevision:
        history = self.get_history(gate_id)
        if not history:
            raise GateRepositoryError("gate was not found")
        return history[-1]

    def resolve_revision(self, revision_id: UUID) -> tuple[GateRevision, GateRevision]:
        """Return ``(requested, current)`` in one locked snapshot."""

        with self._lock():
            payload = self._read_unlocked()
            for gate_key, raw_history in payload["gates"].items():
                history = self._validated_history(gate_key, raw_history)
                for revision in history:
                    if revision.revision_id == revision_id:
                        return revision, history[-1]
        raise GateRepositoryError("gate revision was not found")
