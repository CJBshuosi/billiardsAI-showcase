"""Process-safe single-file authority for Story 6.1 table admission."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from billiards_ai.shared.models import (
    CaptureAuditEvent,
    CaptureIdempotencyRecord,
    CloudSession,
    CloudSessionStatus,
    ConsentDecision,
    ConsentDecisionRecord,
    ConsentPolicyView,
    ConsentSubmissionRequest,
    ConsentWithdrawalRequest,
    CorrectIntervalRequest,
    JoinTableRequest,
    MaterialUsePurpose,
    SessionAdmissionErrorCode,
    SessionAdmissionIdempotencyRecord,
    SessionCaptureErrorCode,
    SessionIntervalRecord,
    SessionMediaBindingRecord,
    SessionParticipant,
    TableBinding,
    TableBindingErrorCode,
    TableBindingStatus,
    TableCapacityStatus,
    TableQrRegistration,
    TableQrStatus,
    TableRegistration,
    TableRegistrationStatus,
    UploadAttemptRecord,
    UploadAttemptStatus,
)

from .errors import (
    SessionAdmissionRepositoryDomainError,
    SessionCaptureRepositoryDomainError,
    TableBindingCommitStateUnknownError,
    TableBindingDomainError,
    TableBindingRepositoryError,
)

STORE_SCHEMA_VERSION = "3.0"
PREVIOUS_STORE_SCHEMA_VERSION = "2.0"
LEGACY_STORE_SCHEMA_VERSION = "1.0"
MAX_STORE_BYTES = 4_194_304
MAX_STORE_DEPTH = 32
MAX_RECORDS_PER_COLLECTION = 1_000


@dataclass(frozen=True, slots=True)
class PendingBindingCandidate:
    binding_id: UUID
    created_at: datetime


@dataclass(frozen=True, slots=True)
class BindingSnapshot:
    binding: TableBinding
    table: TableRegistration
    capacity_status: TableCapacityStatus


@dataclass(frozen=True, slots=True)
class QrProvisioningResult:
    table_registration: TableRegistration
    qr_registration: TableQrRegistration
    qr_token: str


@dataclass(frozen=True, slots=True)
class AdmissionCandidate:
    service_consent_id: UUID
    training_consent_id: UUID
    session_id: UUID | None
    participant_id: UUID | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class WithdrawalCandidate:
    consent_id: UUID
    decided_at: datetime


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    binding: TableBinding
    table: TableRegistration
    service_consent: ConsentDecisionRecord
    training_consent: ConsentDecisionRecord
    session: CloudSession | None
    participant: SessionParticipant | None


class AdmissionPolicyEvidence(Protocol):
    """Narrow immutable policy evidence accepted by the atomic admission boundary."""

    view: ConsentPolicyView

    def verify_submission(self, request: ConsentSubmissionRequest) -> None: ...


class TableBindingRepository(Protocol):
    def resolve_or_create_pending(
        self,
        *,
        qr_token: str,
        user_id: UUID,
        candidate_factory: Callable[[], PendingBindingCandidate],
    ) -> BindingSnapshot: ...

    def find_owned(self, *, binding_id: UUID, user_id: UUID) -> BindingSnapshot | None: ...

    def recover_admission(
        self,
        *,
        request: ConsentSubmissionRequest,
        user_id: UUID,
        key_digest: str,
        payload_digest: str,
    ) -> AdmissionSnapshot | None: ...

    def read_admission(self, *, binding_id: UUID, user_id: UUID) -> AdmissionSnapshot | None: ...

    def admit(
        self,
        *,
        request: ConsentSubmissionRequest,
        user_id: UUID,
        policy: AdmissionPolicyEvidence,
        key_digest: str,
        payload_digest: str,
        candidate_factory: Callable[[bool, bool], AdmissionCandidate],
    ) -> AdmissionSnapshot: ...

    def withdraw(
        self,
        *,
        request: ConsentWithdrawalRequest,
        user_id: UUID,
        key_digest: str,
        payload_digest: str,
        candidate_factory: Callable[[], WithdrawalCandidate],
    ) -> AdmissionSnapshot: ...


def _structure_depth(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_STORE_DEPTH:
        return depth
    if isinstance(value, dict):
        return max(
            (_structure_depth(item, depth=depth + 1) for item in value.values()),
            default=depth,
        )
    if isinstance(value, list):
        return max((_structure_depth(item, depth=depth + 1) for item in value), default=depth)
    return depth


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _revalidate(model: type[Any], value: Any) -> Any:
    payload = value.model_dump(mode="python") if hasattr(value, "model_dump") else value
    return model.model_validate(payload)


class JsonTableBindingRepository:
    """Strict JSON snapshot with process-locked read-modify-write operations."""

    def __init__(self, path: Path | str):
        try:
            self.path = Path(os.path.abspath(os.fspath(path)))
        except (OSError, RuntimeError, TypeError) as exc:
            raise TableBindingRepositoryError("table-binding path is unavailable") from exc
        self._lock_path = self.path.with_name(f".{self.path.name}-lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "revision": 0,
            "table_registrations": {},
            "qr_registrations": {},
            "bindings": {},
            "consent_decisions": {},
            "sessions": {},
            "session_participants": {},
            "idempotency_records": {},
            "capture_idempotency_records": {},
            "capture_audit_events": {},
            "upload_attempts": {},
            "media_bindings": {},
        }

    @staticmethod
    def _upgrade_payload(payload: dict[str, Any]) -> None:
        if payload.get("schema_version") == LEGACY_STORE_SCHEMA_VERSION:
            payload["schema_version"] = PREVIOUS_STORE_SCHEMA_VERSION
            payload["consent_decisions"] = {}
            payload["sessions"] = {}
            payload["session_participants"] = {}
            payload["idempotency_records"] = {}
        if payload.get("schema_version") == PREVIOUS_STORE_SCHEMA_VERSION:
            payload["schema_version"] = STORE_SCHEMA_VERSION
            payload["capture_idempotency_records"] = {}
            payload["capture_audit_events"] = {}
            payload["upload_attempts"] = {}
            payload["media_bindings"] = {}

    @staticmethod
    def _require_safe_ancestors(path: Path) -> None:
        current = Path(path.anchor)
        try:
            components = path.parts[1:]
            for component in components:
                current /= component
                info = current.lstat()
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise TableBindingRepositoryError("table-binding directory ancestry is unsafe")
                if current == path:
                    if info.st_uid != os.geteuid() or mode != 0o700 or info.st_nlink < 1:
                        raise TableBindingRepositoryError("table-binding directory is unsafe")
                elif info.st_uid not in {0, os.geteuid()} or mode & 0o022:
                    raise TableBindingRepositoryError("table-binding directory ancestry is unsafe")
        except OSError as exc:
            raise TableBindingRepositoryError("table-binding directory is unavailable") from exc

    @staticmethod
    def _lock_metadata_is_safe(info: os.stat_result) -> bool:
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_nlink == 1
        )

    def _open_lock(self) -> int:
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            path_info = self._lock_path.lstat()
        except FileNotFoundError:
            try:
                fd = os.open(self._lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return self._open_lock()
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if not self._lock_metadata_is_safe(info):
                os.close(fd)
                raise TableBindingRepositoryError("table-binding lock is unsafe") from None
            return fd
        if not self._lock_metadata_is_safe(path_info):
            raise TableBindingRepositoryError("table-binding lock is unsafe")
        fd = os.open(self._lock_path, flags)
        info = os.fstat(fd)
        if (
            not self._lock_metadata_is_safe(info)
            or info.st_dev != path_info.st_dev
            or info.st_ino != path_info.st_ino
        ):
            os.close(fd)
            raise TableBindingRepositoryError("table-binding lock is unsafe")
        return fd

    @contextmanager
    def _lock(self) -> Iterator[None]:
        fd = -1
        locked = False
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._require_safe_ancestors(self.path.parent)
            if self.path.is_symlink():
                raise TableBindingRepositoryError("table-binding paths are unsafe")
            fd = self._open_lock()
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        except TableBindingRepositoryError:
            raise
        except OSError as exc:
            raise TableBindingRepositoryError("table-binding lock is unavailable") from exc
        finally:
            if fd >= 0:
                try:
                    if locked:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                finally:
                    os.close(fd)

    def _read_unlocked(
        self, *, validate_authority: bool = True
    ) -> tuple[dict[str, Any], bytes | None]:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return self._empty(), None
        except OSError as exc:
            raise TableBindingRepositoryError("table-binding store is unavailable") from exc
        if stat.S_ISLNK(info.st_mode):
            raise TableBindingRepositoryError("table-binding store is unsafe")
        fd = -1
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.path, flags)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise TableBindingRepositoryError("table-binding store is unsafe")
            if info.st_size > MAX_STORE_BYTES:
                raise TableBindingRepositoryError("table-binding store exceeds its size limit")
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                raw = handle.read(MAX_STORE_BYTES + 1)
            if len(raw) > MAX_STORE_BYTES:
                raise TableBindingRepositoryError("table-binding store exceeds its size limit")
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
            if _structure_depth(payload) > MAX_STORE_DEPTH:
                raise TableBindingRepositoryError("table-binding store exceeds its depth limit")
            self._validated_store(payload, validate_authority=validate_authority)
            return payload, raw
        except TableBindingRepositoryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise TableBindingRepositoryError("table-binding store cannot be trusted") from exc
        finally:
            if fd >= 0:
                os.close(fd)

    @classmethod
    def _validated_store(
        cls, payload: object, *, validate_authority: bool = True
    ) -> tuple[
        dict[str, TableRegistration],
        dict[str, TableQrRegistration],
        dict[str, TableBinding],
    ]:
        legacy_expected = {
            "schema_version",
            "revision",
            "table_registrations",
            "qr_registrations",
            "bindings",
        }
        current_expected = legacy_expected | {
            "consent_decisions",
            "sessions",
            "session_participants",
            "idempotency_records",
        }
        v3_expected = current_expected | {
            "capture_idempotency_records",
            "capture_audit_events",
            "upload_attempts",
            "media_bindings",
        }
        if not isinstance(payload, dict):
            raise TableBindingRepositoryError("invalid table-binding store structure")
        schema_version = payload.get("schema_version")
        expected = (
            legacy_expected
            if schema_version == LEGACY_STORE_SCHEMA_VERSION
            else current_expected
            if schema_version == PREVIOUS_STORE_SCHEMA_VERSION
            else v3_expected
        )
        if set(payload) != expected:
            raise TableBindingRepositoryError("invalid table-binding store structure")
        if schema_version not in {
            LEGACY_STORE_SCHEMA_VERSION,
            PREVIOUS_STORE_SCHEMA_VERSION,
            STORE_SCHEMA_VERSION,
        }:
            raise TableBindingRepositoryError("unsupported table-binding store schema")
        revision = payload.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise TableBindingRepositoryError("invalid table-binding store revision")
        collections = [
            payload.get("table_registrations"),
            payload.get("qr_registrations"),
            payload.get("bindings"),
        ]
        if schema_version in {PREVIOUS_STORE_SCHEMA_VERSION, STORE_SCHEMA_VERSION}:
            collections.extend(
                [
                    payload.get("consent_decisions"),
                    payload.get("sessions"),
                    payload.get("session_participants"),
                    payload.get("idempotency_records"),
                ]
            )
        if schema_version == STORE_SCHEMA_VERSION:
            collections.extend(
                [payload.get("capture_idempotency_records"), payload.get("capture_audit_events")]
                + [payload.get("upload_attempts"), payload.get("media_bindings")]
            )
        if any(not isinstance(collection, dict) for collection in collections):
            raise TableBindingRepositoryError("invalid table-binding record collections")
        if any(len(collection) > MAX_RECORDS_PER_COLLECTION for collection in collections):
            raise TableBindingRepositoryError("table-binding record limit exceeded")
        try:
            tables = {
                key: _revalidate(TableRegistration, value)
                for key, value in payload["table_registrations"].items()
            }
            qrs = {
                key: _revalidate(TableQrRegistration, value)
                for key, value in payload["qr_registrations"].items()
            }
            bindings = {
                key: _revalidate(TableBinding, value) for key, value in payload["bindings"].items()
            }
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
            raise TableBindingRepositoryError(
                "table-binding store contains invalid records"
            ) from exc
        if any(str(item.table_registration_id) != key for key, item in tables.items()):
            raise TableBindingRepositoryError("table registration key mismatch")
        if any(str(item.qr_registration_id) != key for key, item in qrs.items()):
            raise TableBindingRepositoryError("QR registration key mismatch")
        if any(str(item.binding_id) != key for key, item in bindings.items()):
            raise TableBindingRepositoryError("binding key mismatch")
        if len({item.device_id for item in tables.values()}) != len(tables):
            raise TableBindingRepositoryError("device registration is not unique")
        table_keys = {(item.venue_id, item.table_id) for item in tables.values()}
        if len(table_keys) != len(tables):
            raise TableBindingRepositoryError("venue table registration is not unique")
        if len({item.token_digest for item in qrs.values()}) != len(qrs):
            raise TableBindingRepositoryError("QR digest is not unique")
        qr_versions = {(item.table_registration_id, item.qr_version) for item in qrs.values()}
        if len(qr_versions) != len(qrs):
            raise TableBindingRepositoryError("QR version is not unique for its table")
        table_registration_ids = {item.table_registration_id for item in tables.values()}
        if any(
            item.table_registration_id not in table_registration_ids
            or item.qr_registration_id not in {x.qr_registration_id for x in qrs.values()}
            for item in bindings.values()
        ):
            raise TableBindingRepositoryError("binding references unknown authority")
        for binding in bindings.values():
            qr = qrs[str(binding.qr_registration_id)]
            if (
                qr.table_registration_id != binding.table_registration_id
                or qr.qr_version != binding.qr_version
            ):
                raise TableBindingRepositoryError("binding QR relationship is inconsistent")
        live = {
            TableBindingStatus.PENDING_CONSENT,
            TableBindingStatus.ACTIVE,
        }
        live_users = [item.user_id for item in bindings.values() if item.status in live]
        if len(live_users) != len(set(live_users)):
            raise TableBindingRepositoryError("user has multiple live bindings")
        if validate_authority:
            cls._validate_authority_invariants(tables, qrs, bindings)
        if schema_version in {PREVIOUS_STORE_SCHEMA_VERSION, STORE_SCHEMA_VERSION}:
            cls._validate_admission_authority(payload, tables, qrs, bindings)
        if schema_version == STORE_SCHEMA_VERSION:
            try:
                idempotency = {
                    key: _revalidate(CaptureIdempotencyRecord, value)
                    for key, value in payload["capture_idempotency_records"].items()
                }
                audits = {
                    key: _revalidate(CaptureAuditEvent, value)
                    for key, value in payload["capture_audit_events"].items()
                }
                attempts = {
                    key: _revalidate(UploadAttemptRecord, value)
                    for key, value in payload["upload_attempts"].items()
                }
                media = {
                    key: _revalidate(SessionMediaBindingRecord, value)
                    for key, value in payload["media_bindings"].items()
                }
                sessions = {
                    key: _revalidate(CloudSession, value)
                    for key, value in payload["sessions"].items()
                }
                participants = tuple(
                    _revalidate(SessionParticipant, value)
                    for value in payload["session_participants"].values()
                )
            except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
                raise TableBindingRepositoryError(
                    "capture authority contains invalid records"
                ) from exc
            if any(item.record_id != key for key, item in idempotency.items()):
                raise TableBindingRepositoryError("capture idempotency key mismatch")
            if any(str(item.sequence) != key for key, item in audits.items()):
                raise TableBindingRepositoryError("capture audit sequence mismatch")
            if sorted(item.sequence for item in audits.values()) != list(range(1, len(audits) + 1)):
                raise TableBindingRepositoryError("capture audit sequence is not contiguous")
            if len({item.event_id for item in audits.values()}) != len(audits):
                raise TableBindingRepositoryError("capture audit event ID is not unique")
            if any(str(item.attempt_id) != key for key, item in attempts.items()):
                raise TableBindingRepositoryError("upload attempt key mismatch")
            if any(str(item.session_id) != key for key, item in media.items()):
                raise TableBindingRepositoryError("media binding key mismatch")
            session_ids = {item.session_id for item in sessions.values()}
            if any(
                item.session_id not in session_ids
                or item.result_session_revision > sessions[str(item.session_id)].revision
                or (
                    item.result_interval_revision is not None
                    and item.result_interval_revision
                    > len(sessions[str(item.session_id)].interval_history)
                )
                for item in idempotency.values()
            ):
                raise TableBindingRepositoryError("capture idempotency references unknown session")
            for item in audits.values():
                session = sessions.get(str(item.session_id))
                attempt = attempts.get(str(item.attempt_id)) if item.attempt_id else None
                if (
                    session is None
                    or item.table_registration_id != session.table_registration_id
                    or (item.attempt_id is not None and attempt is None)
                    or (attempt is not None and attempt.session_id != item.session_id)
                    or (
                        item.asset_id is not None
                        and attempt is not None
                        and attempt.asset_id != item.asset_id
                    )
                ):
                    raise TableBindingRepositoryError("capture audit references unknown authority")
            if any(
                item.session_id not in session_ids
                or item.interval_revision > len(sessions[str(item.session_id)].interval_history)
                for item in attempts.values()
            ):
                raise TableBindingRepositoryError("upload attempt references unknown session")
            subjects_by_session = {
                session_id: tuple(
                    sorted(
                        item.user_id for item in participants if str(item.session_id) == session_id
                    )
                )
                for session_id in sessions
            }
            if any(
                item.object_staged
                and item.subject_user_ids != subjects_by_session[str(item.session_id)]
                for item in attempts.values()
            ):
                raise TableBindingRepositoryError("upload attempt subjects are inconsistent")
            for binding in media.values():
                attempt = attempts.get(str(binding.attempt_id))
                session = sessions.get(str(binding.session_id))
                if (
                    attempt is None
                    or session is None
                    or attempt.session_id != binding.session_id
                    or attempt.asset_id != binding.asset_id
                    or attempt.content_sha256 != binding.content_sha256
                    or attempt.storage_locator != binding.storage_locator
                    or attempt.subject_user_ids != binding.subject_user_ids
                    or attempt.interval_revision != binding.interval_revision
                    or binding.subject_user_ids != subjects_by_session[str(binding.session_id)]
                    or session.media_asset_id != binding.asset_id
                    or session.media_content_sha256 != binding.content_sha256
                ):
                    raise TableBindingRepositoryError("media binding authority is inconsistent")
            hashes = [item.content_sha256 for item in media.values()]
            if len(hashes) != len(set(hashes)):
                raise TableBindingRepositoryError("media content binding is not unique")
        return tables, qrs, bindings

    @classmethod
    def _validate_admission_authority(
        cls,
        payload: dict[str, Any],
        tables: dict[str, TableRegistration],
        qrs: dict[str, TableQrRegistration],
        bindings: dict[str, TableBinding],
    ) -> None:
        try:
            consents = {
                key: _revalidate(ConsentDecisionRecord, value)
                for key, value in payload["consent_decisions"].items()
            }
            sessions = {
                key: _revalidate(CloudSession, value) for key, value in payload["sessions"].items()
            }
            participants = {
                key: _revalidate(SessionParticipant, value)
                for key, value in payload["session_participants"].items()
            }
            idempotency = {
                key: _revalidate(SessionAdmissionIdempotencyRecord, value)
                for key, value in payload["idempotency_records"].items()
            }
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
            raise TableBindingRepositoryError("session authority contains invalid records") from exc
        if any(str(item.consent_id) != key for key, item in consents.items()):
            raise TableBindingRepositoryError("consent decision key mismatch")
        if any(str(item.session_id) != key for key, item in sessions.items()):
            raise TableBindingRepositoryError("session key mismatch")
        if any(str(item.participant_id) != key for key, item in participants.items()):
            raise TableBindingRepositoryError("participant key mismatch")
        if any(item.record_id != key for key, item in idempotency.items()):
            raise TableBindingRepositoryError("idempotency key mismatch")
        all_ids = (
            {item.table_registration_id for item in tables.values()}
            | {item.qr_registration_id for item in qrs.values()}
            | {item.binding_id for item in bindings.values()}
            | {item.consent_id for item in consents.values()}
            | {item.session_id for item in sessions.values()}
            | {item.participant_id for item in participants.values()}
        )
        expected_id_count = (
            len(tables)
            + len(qrs)
            + len(bindings)
            + len(consents)
            + len(sessions)
            + len(participants)
        )
        if len(all_ids) != expected_id_count:
            raise TableBindingRepositoryError("session authority UUIDs are not globally unique")
        table_ids = {item.table_registration_id for item in tables.values()}
        if any(item.table_registration_id not in table_ids for item in sessions.values()):
            raise TableBindingRepositoryError("session references unknown table")
        open_by_table: dict[UUID, list[CloudSession]] = {}
        for session in sessions.values():
            if session.status is CloudSessionStatus.OPEN:
                open_by_table.setdefault(session.table_registration_id, []).append(session)
        if any(len(items) > 1 for items in open_by_table.values()):
            raise TableBindingRepositoryError("table has multiple open sessions")
        for participant in participants.values():
            binding = bindings.get(str(participant.binding_id))
            session = sessions.get(str(participant.session_id))
            service = consents.get(str(participant.service_consent_id))
            training = consents.get(str(participant.training_consent_id))
            if (
                binding is None
                or session is None
                or service is None
                or training is None
                or binding.user_id != participant.user_id
                or binding.table_registration_id != session.table_registration_id
                or service.user_id != participant.user_id
                or service.binding_id != participant.binding_id
                or service.session_id != participant.session_id
                or service.participant_id != participant.participant_id
                or service.purpose.value != "SERVICE_ANALYSIS"
                or training.user_id != participant.user_id
                or training.binding_id != participant.binding_id
                or training.session_id != participant.session_id
                or training.participant_id != participant.participant_id
                or training.purpose.value != "TRAINING"
            ):
                raise TableBindingRepositoryError("participant consent authority is inconsistent")
        participant_bindings = {item.binding_id for item in participants.values()}
        if any(
            binding.status is TableBindingStatus.PENDING_CONSENT
            and binding.binding_id in participant_bindings
            for binding in bindings.values()
        ):
            raise TableBindingRepositoryError("pending binding has a participant")
        # A pristine pre-consent v2 root remains compatible with 6.1 fixtures. Once
        # session authority exists, every ACTIVE binding must be represented.
        if consents or sessions or participants or idempotency:
            if any(
                binding.status is TableBindingStatus.ACTIVE
                and binding.binding_id not in participant_bindings
                for binding in bindings.values()
            ):
                raise TableBindingRepositoryError("active binding lacks a participant")
        for session in sessions.values():
            members = [p for p in participants.values() if p.session_id == session.session_id]
            if not 1 <= len(members) <= 2 or len({p.user_id for p in members}) != len(members):
                raise TableBindingRepositoryError("session participant capacity is inconsistent")
        for consent in consents.values():
            binding = bindings.get(str(consent.binding_id))
            if binding is None or binding.user_id != consent.user_id:
                raise TableBindingRepositoryError("consent references unknown binding owner")
            if consent.participant_id is None:
                if consent.decision is not ConsentDecision.DECLINED:
                    raise TableBindingRepositoryError("unlinked consent is not a decline audit")
            else:
                participant = participants.get(str(consent.participant_id))
                session = sessions.get(str(consent.session_id))
                if (
                    participant is None
                    or session is None
                    or participant.user_id != consent.user_id
                    or participant.binding_id != consent.binding_id
                    or participant.session_id != consent.session_id
                ):
                    raise TableBindingRepositoryError(
                        "consent participant authority is inconsistent"
                    )
            if consent.supersedes is not None:
                previous = consents.get(str(consent.supersedes))
                if (
                    previous is None
                    or consent.decision is not ConsentDecision.WITHDRAWN
                    or previous.decision is not ConsentDecision.GRANTED
                    or previous.user_id != consent.user_id
                    or previous.binding_id != consent.binding_id
                    or previous.participant_id != consent.participant_id
                    or previous.purpose != consent.purpose
                    or previous.decided_at > consent.decided_at
                ):
                    raise TableBindingRepositoryError("consent withdrawal chain is inconsistent")
        superseded_ids = [
            item.supersedes for item in consents.values() if item.supersedes is not None
        ]
        if len(superseded_ids) != len(set(superseded_ids)):
            raise TableBindingRepositoryError("consent withdrawal chain forks")
        for participant in participants.values():
            purpose_heads = {
                MaterialUsePurpose.SERVICE_ANALYSIS: participant.service_consent_id,
                MaterialUsePurpose.TRAINING: participant.training_consent_id,
            }
            for purpose, head_id in purpose_heads.items():
                if any(item.supersedes == head_id for item in consents.values()):
                    raise TableBindingRepositoryError("participant does not reference consent head")
                purpose_records = {
                    item.consent_id: item
                    for item in consents.values()
                    if item.participant_id == participant.participant_id and item.purpose is purpose
                }
                roots = [item for item in purpose_records.values() if item.supersedes is None]
                if len(roots) != 1 or head_id not in purpose_records:
                    raise TableBindingRepositoryError("participant consent chain is disconnected")
                if (
                    purpose is MaterialUsePurpose.SERVICE_ANALYSIS
                    and roots[0].decision is not ConsentDecision.GRANTED
                ):
                    raise TableBindingRepositoryError(
                        "service consent chain must start with a grant"
                    )
                visited: set[UUID] = set()
                current = purpose_records[head_id]
                while True:
                    if current.consent_id in visited:
                        raise TableBindingRepositoryError("participant consent chain cycles")
                    visited.add(current.consent_id)
                    if current.supersedes is None:
                        break
                    previous = purpose_records.get(current.supersedes)
                    if previous is None:
                        raise TableBindingRepositoryError(
                            "participant consent chain is disconnected"
                        )
                    current = previous
                if visited != set(purpose_records):
                    raise TableBindingRepositoryError("participant consent chain is disconnected")
        for record in idempotency.values():
            binding = bindings.get(str(record.binding_id))
            if binding is None or binding.user_id != record.user_id:
                raise TableBindingRepositoryError("idempotency references unknown binding")
            expected_record_id = (
                cls._idempotency_record_key(
                    "ADMIT", record.user_id, record.binding_id, record.key_digest
                )
                if record.operation == "ADMIT"
                else cls._idempotency_record_key(
                    "WITHDRAW",
                    record.user_id,
                    record.participant_id,
                    record.purpose,
                    record.key_digest,
                )
            )
            if record.record_id != expected_record_id:
                raise TableBindingRepositoryError("idempotency scope digest mismatches")
            result_consents = [consents.get(str(item)) for item in record.result_consent_ids]
            if any(
                item is None
                or item.user_id != record.user_id
                or item.binding_id != record.binding_id
                for item in result_consents
            ):
                raise TableBindingRepositoryError("idempotency references unknown consent")
            result_session = (
                None
                if record.result_session_id is None
                else sessions.get(str(record.result_session_id))
            )
            result_participant = (
                None
                if record.result_participant_id is None
                else participants.get(str(record.result_participant_id))
            )
            if (record.result_session_id is None) != (record.result_participant_id is None):
                raise TableBindingRepositoryError("idempotency result links are incomplete")
            if record.result_session_id is not None and (
                result_session is None or result_participant is None
            ):
                raise TableBindingRepositoryError("idempotency result links are unavailable")
            if result_participant is not None and (
                result_participant.user_id != record.user_id
                or result_participant.binding_id != record.binding_id
                or result_participant.session_id != record.result_session_id
                or result_session.table_registration_id != binding.table_registration_id
            ):
                raise TableBindingRepositoryError("idempotency participant authority mismatches")
            if record.operation == "ADMIT":
                decisions_by_purpose = {
                    item.purpose: item for item in result_consents if item is not None
                }
                if set(decisions_by_purpose) != {
                    MaterialUsePurpose.SERVICE_ANALYSIS,
                    MaterialUsePurpose.TRAINING,
                }:
                    raise TableBindingRepositoryError("admission idempotency purposes mismatch")
                service_result = decisions_by_purpose[MaterialUsePurpose.SERVICE_ANALYSIS]
                training_result = decisions_by_purpose[MaterialUsePurpose.TRAINING]
                if result_participant is None:
                    if any(
                        item.decision is not ConsentDecision.DECLINED
                        or item.session_id is not None
                        or item.participant_id is not None
                        for item in result_consents
                        if item is not None
                    ):
                        raise TableBindingRepositoryError(
                            "decline idempotency result is inconsistent"
                        )
                else:
                    if (
                        service_result.decision is not ConsentDecision.GRANTED
                        or training_result.decision
                        not in {ConsentDecision.GRANTED, ConsentDecision.DECLINED}
                        or service_result.supersedes is not None
                        or training_result.supersedes is not None
                    ):
                        raise TableBindingRepositoryError(
                            "admission idempotency decisions mismatch"
                        )
                    if any(
                        item.session_id != record.result_session_id
                        or item.participant_id != record.result_participant_id
                        for item in result_consents
                        if item is not None
                    ):
                        raise TableBindingRepositoryError(
                            "admission idempotency consent links mismatch"
                        )
            else:
                result = result_consents[0]
                if (
                    result is None
                    or result_participant is None
                    or record.participant_id != record.result_participant_id
                    or result.decision is not ConsentDecision.WITHDRAWN
                    or result.purpose is not record.purpose
                    or result.session_id != record.result_session_id
                    or result.participant_id != record.result_participant_id
                ):
                    raise TableBindingRepositoryError(
                        "withdrawal idempotency result is inconsistent"
                    )

    @staticmethod
    def _validate_authority_invariants(
        tables: dict[str, TableRegistration],
        qrs: dict[str, TableQrRegistration],
        bindings: dict[str, TableBinding],
    ) -> None:
        for table in tables.values():
            current = [
                qr
                for qr in qrs.values()
                if qr.table_registration_id == table.table_registration_id
                and qr.status is TableQrStatus.CURRENT
            ]
            if len(current) != 1 or current[0].qr_version != table.current_qr_version:
                raise TableBindingRepositoryError("current QR authority is inconsistent")
            active_users = {
                binding.user_id
                for binding in bindings.values()
                if binding.table_registration_id == table.table_registration_id
                and binding.status is TableBindingStatus.ACTIVE
            }
            if len(active_users) > 2:
                raise TableBindingRepositoryError("table capacity authority is inconsistent")

    @staticmethod
    def _serialized(payload: dict[str, Any]) -> bytes:
        try:
            raw = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError) as exc:
            raise TableBindingRepositoryError(
                "table-binding snapshot cannot be serialized"
            ) from exc
        if len(raw) > MAX_STORE_BYTES:
            raise TableBindingRepositoryError("table-binding snapshot exceeds its size limit")
        return raw

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        directory_fd = os.open(self.path.parent, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _restore_after_failed_commit(self, previous: bytes | None) -> None:
        rollback_path: Path | None = None
        try:
            if previous is None:
                self.path.unlink(missing_ok=True)
                self._fsync_directory()
                return
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.rollback.",
                delete=False,
            ) as handle:
                rollback_path = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                handle.write(previous)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(rollback_path, self.path)
            rollback_path = None
            self._fsync_directory()
        except OSError:
            raise TableBindingCommitStateUnknownError(
                "table-binding commit state is unknown"
            ) from None
        finally:
            if rollback_path is not None:
                try:
                    rollback_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_unlocked(self, payload: dict[str, Any], *, previous: bytes | None) -> None:
        self._upgrade_payload(payload)
        self._validated_store(payload)
        raw = self._serialized(payload)
        temporary_path: Path | None = None
        replaced = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            replaced = True
            self._fsync_directory()
        except OSError:
            if replaced:
                try:
                    self._restore_after_failed_commit(previous)
                except TableBindingCommitStateUnknownError:
                    raise
                raise TableBindingRepositoryError(
                    "table-binding store commit failed; previous snapshot restored"
                ) from None
            raise TableBindingRepositoryError("table-binding store commit failed") from None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _new_token(self) -> tuple[str, str]:
        try:
            raw = secrets.token_bytes(32)
        except Exception:
            raise TableBindingRepositoryError("QR token generation failed") from None
        if not isinstance(raw, bytes) or len(raw) != 32 or len(set(raw)) < 16:
            raise TableBindingRepositoryError("QR token generation failed")
        token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        try:
            decoded = base64.urlsafe_b64decode(token + "=")
        except (ValueError, UnicodeError) as exc:
            raise TableBindingRepositoryError("QR token generation failed") from exc
        if len(token) != 43 or decoded != raw:
            raise TableBindingRepositoryError("QR token generation failed")
        return token, hashlib.sha256(token.encode("ascii")).hexdigest()

    @staticmethod
    def _new_uuid() -> UUID:
        try:
            value = uuid4()
        except Exception:
            raise TableBindingRepositoryError("UUID generation failed") from None
        if not isinstance(value, UUID) or value.version != 4:
            raise TableBindingRepositoryError("UUID generation failed")
        return value

    @staticmethod
    def _uuid_is_available(value: UUID, payload: dict[str, Any]) -> bool:
        key = str(value)
        return all(
            key not in payload.get(collection, {})
            for collection in (
                "table_registrations",
                "qr_registrations",
                "bindings",
                "consent_decisions",
                "sessions",
                "session_participants",
            )
        )

    @staticmethod
    def _capacity(active_count: int) -> TableCapacityStatus:
        if active_count <= 0:
            return TableCapacityStatus.AVAILABLE
        if active_count == 1:
            return TableCapacityStatus.ONE_ACTIVE
        return TableCapacityStatus.FULL

    def provision_table(self, table: TableRegistration) -> QrProvisioningResult:
        try:
            table = _revalidate(TableRegistration, table)
        except Exception:
            raise TableBindingRepositoryError("table registration is invalid") from None
        with self._lock():
            payload, previous = self._read_unlocked()
            tables, qrs, _ = self._validated_store(payload)
            if table.current_qr_version != 1:
                raise TableBindingRepositoryError("new table must start at QR version one")
            if str(table.table_registration_id) in tables:
                raise TableBindingRepositoryError("table registration already exists")
            token, digest = self._new_token()
            if any(item.token_digest == digest for item in qrs.values()):
                raise TableBindingRepositoryError("QR digest already exists")
            qr_id = self._new_uuid()
            if not self._uuid_is_available(qr_id, payload):
                raise TableBindingRepositoryError("QR registration ID already exists")
            try:
                qr = TableQrRegistration(
                    qr_registration_id=qr_id,
                    token_digest=digest,
                    table_registration_id=table.table_registration_id,
                    qr_version=1,
                    status=TableQrStatus.CURRENT,
                )
                payload["table_registrations"][str(table.table_registration_id)] = table.model_dump(
                    mode="json"
                )
                payload["qr_registrations"][str(qr.qr_registration_id)] = qr.model_dump(mode="json")
                payload["revision"] += 1
                self._write_unlocked(payload, previous=previous)
            except (TypeError, ValidationError, ValueError) as exc:
                raise TableBindingRepositoryError("table provisioning failed") from exc
            return QrProvisioningResult(table, qr, token)

    def rotate_qr(self, table_registration_id: UUID) -> QrProvisioningResult:
        with self._lock():
            payload, previous = self._read_unlocked()
            tables, qrs, _ = self._validated_store(payload)
            table = tables.get(str(table_registration_id))
            if table is None:
                raise TableBindingRepositoryError("table registration is unavailable")
            current = [
                item
                for item in qrs.values()
                if item.table_registration_id == table.table_registration_id
                and item.qr_version == table.current_qr_version
                and item.status is TableQrStatus.CURRENT
            ]
            if len(current) != 1:
                raise TableBindingRepositoryError("current QR authority is inconsistent")
            token, digest = self._new_token()
            if any(item.token_digest == digest for item in qrs.values()):
                raise TableBindingRepositoryError("QR digest already exists")
            qr_id = self._new_uuid()
            if not self._uuid_is_available(qr_id, payload):
                raise TableBindingRepositoryError("QR registration ID already exists")
            try:
                updated_table = TableRegistration.model_validate(
                    {**table.model_dump(), "current_qr_version": table.current_qr_version + 1}
                )
                rotated = TableQrRegistration.model_validate(
                    {**current[0].model_dump(), "status": TableQrStatus.ROTATED}
                )
                qr = TableQrRegistration(
                    qr_registration_id=qr_id,
                    token_digest=digest,
                    table_registration_id=table.table_registration_id,
                    qr_version=updated_table.current_qr_version,
                    status=TableQrStatus.CURRENT,
                )
                payload["table_registrations"][str(table.table_registration_id)] = (
                    updated_table.model_dump(mode="json")
                )
                payload["qr_registrations"][str(rotated.qr_registration_id)] = rotated.model_dump(
                    mode="json"
                )
                payload["qr_registrations"][str(qr.qr_registration_id)] = qr.model_dump(mode="json")
                payload["revision"] += 1
                self._write_unlocked(payload, previous=previous)
            except (TypeError, ValidationError, ValueError) as exc:
                raise TableBindingRepositoryError("QR rotation failed") from exc
            return QrProvisioningResult(updated_table, qr, token)

    def resolve_or_create_pending(
        self,
        *,
        qr_token: str,
        user_id: UUID,
        candidate_factory: Callable[[], PendingBindingCandidate],
    ) -> BindingSnapshot:
        try:
            JoinTableRequest(schema_version="1.0", qr_token=qr_token)
        except ValidationError:
            raise TableBindingDomainError(TableBindingErrorCode.QR_INVALID) from None
        digest = hashlib.sha256(qr_token.encode("ascii")).hexdigest()
        with self._lock():
            payload, previous = self._read_unlocked(validate_authority=False)
            tables, qrs, bindings = self._validated_store(payload, validate_authority=False)
            qr = next((item for item in qrs.values() if item.token_digest == digest), None)
            if qr is None:
                self._validate_authority_invariants(tables, qrs, bindings)
                raise TableBindingDomainError(TableBindingErrorCode.QR_INVALID)
            if qr.status is not TableQrStatus.CURRENT:
                raise TableBindingDomainError(TableBindingErrorCode.QR_BINDING_INACTIVE)
            table = tables.get(str(qr.table_registration_id))
            if table is None or qr.qr_version != table.current_qr_version:
                raise TableBindingDomainError(TableBindingErrorCode.TABLE_BINDING_MISMATCH)
            self._validate_authority_invariants(tables, qrs, bindings)
            if table.status is not TableRegistrationStatus.ACTIVE:
                raise TableBindingDomainError(TableBindingErrorCode.TABLE_UNAVAILABLE)
            existing = next(
                (
                    item
                    for item in bindings.values()
                    if item.user_id == user_id
                    and item.status
                    in {TableBindingStatus.PENDING_CONSENT, TableBindingStatus.ACTIVE}
                ),
                None,
            )
            active_count = len(
                {
                    item.user_id
                    for item in bindings.values()
                    if item.table_registration_id == table.table_registration_id
                    and item.status is TableBindingStatus.ACTIVE
                }
            )
            if existing is not None:
                if existing.table_registration_id != table.table_registration_id:
                    raise TableBindingDomainError(TableBindingErrorCode.USER_ALREADY_BOUND)
                return BindingSnapshot(existing, table, self._capacity(active_count))
            if active_count >= 2:
                raise TableBindingDomainError(TableBindingErrorCode.TABLE_CAPACITY_REACHED)
            try:
                candidate_binding = candidate_factory()
                binding = TableBinding(
                    binding_id=candidate_binding.binding_id,
                    user_id=user_id,
                    table_registration_id=table.table_registration_id,
                    qr_registration_id=qr.qr_registration_id,
                    qr_version=qr.qr_version,
                    status=TableBindingStatus.PENDING_CONSENT,
                    created_at=candidate_binding.created_at,
                    activated_at=None,
                )
            except Exception:
                raise TableBindingRepositoryError("pending binding candidate is invalid") from None
            if not self._uuid_is_available(binding.binding_id, payload):
                raise TableBindingRepositoryError("binding ID already exists")
            payload["bindings"][str(binding.binding_id)] = binding.model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
            return BindingSnapshot(binding, table, self._capacity(active_count))

    def find_owned(self, *, binding_id: UUID, user_id: UUID) -> BindingSnapshot | None:
        with self._lock():
            payload, _ = self._read_unlocked()
            tables, _, bindings = self._validated_store(payload)
            binding = bindings.get(str(binding_id))
            if binding is None or binding.user_id != user_id:
                return None
            table = tables.get(str(binding.table_registration_id))
            if table is None:
                raise TableBindingRepositoryError("binding table is unavailable")
            active_count = len(
                {
                    item.user_id
                    for item in bindings.values()
                    if item.table_registration_id == table.table_registration_id
                    and item.status is TableBindingStatus.ACTIVE
                }
            )
            return BindingSnapshot(binding, table, self._capacity(active_count))

    @staticmethod
    def _idempotency_record_key(*parts: object) -> str:
        raw = ":".join(str(part) for part in parts).encode("ascii")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _snapshot_from_record(
        cls,
        payload: dict[str, Any],
        record: SessionAdmissionIdempotencyRecord,
    ) -> AdmissionSnapshot:
        tables, _, bindings = cls._validated_store(payload)
        binding = bindings.get(str(record.binding_id))
        if binding is None:
            raise TableBindingRepositoryError("idempotency binding is unavailable")
        if (
            record.operation == "ADMIT"
            and record.result_session_id is None
            and record.result_participant_id is None
        ):
            binding = TableBinding.model_validate(
                {
                    **binding.model_dump(mode="python"),
                    "status": TableBindingStatus.PENDING_CONSENT,
                    "activated_at": None,
                }
            )
        table = tables.get(str(binding.table_registration_id))
        if table is None:
            raise TableBindingRepositoryError("idempotency table is unavailable")
        consents = {
            key: ConsentDecisionRecord.model_validate(value)
            for key, value in payload["consent_decisions"].items()
        }
        sessions = {
            key: CloudSession.model_validate(value) for key, value in payload["sessions"].items()
        }
        participants = {
            key: SessionParticipant.model_validate(value)
            for key, value in payload["session_participants"].items()
        }
        if record.operation == "ADMIT":
            if len(record.result_consent_ids) != 2:
                raise TableBindingRepositoryError("admission idempotency result is invalid")
            decisions = [consents.get(str(item)) for item in record.result_consent_ids]
            if any(item is None for item in decisions):
                raise TableBindingRepositoryError("admission consent result is unavailable")
            service = next(
                (item for item in decisions if item.purpose is MaterialUsePurpose.SERVICE_ANALYSIS),
                None,
            )
            training = next(
                (item for item in decisions if item.purpose is MaterialUsePurpose.TRAINING), None
            )
        else:
            participant = participants.get(str(record.result_participant_id))
            if participant is None:
                raise TableBindingRepositoryError("withdrawal participant is unavailable")
            service = consents.get(str(participant.service_consent_id))
            training = consents.get(str(participant.training_consent_id))
        if service is None or training is None:
            raise TableBindingRepositoryError("idempotency consent heads are unavailable")
        session = (
            None
            if record.result_session_id is None
            else sessions.get(str(record.result_session_id))
        )
        participant = (
            None
            if record.result_participant_id is None
            else participants.get(str(record.result_participant_id))
        )
        if (record.result_session_id is None) != (record.result_participant_id is None):
            raise TableBindingRepositoryError("idempotency admission links are inconsistent")
        if record.result_session_id is not None and (session is None or participant is None):
            raise TableBindingRepositoryError("idempotency admission result is unavailable")
        return AdmissionSnapshot(binding, table, service, training, session, participant)

    def recover_admission(
        self,
        *,
        request: ConsentSubmissionRequest,
        user_id: UUID,
        key_digest: str,
        payload_digest: str,
    ) -> AdmissionSnapshot | None:
        record_key = self._idempotency_record_key("ADMIT", user_id, request.binding_id, key_digest)
        with self._lock():
            payload, _ = self._read_unlocked()
            if payload["schema_version"] == LEGACY_STORE_SCHEMA_VERSION:
                return None
            raw = payload["idempotency_records"].get(record_key)
            if raw is None:
                return None
            record = SessionAdmissionIdempotencyRecord.model_validate(raw)
            if record.payload_digest != payload_digest:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.IDEMPOTENCY_CONFLICT
                )
            return self._snapshot_from_record(payload, record)

    def read_admission(self, *, binding_id: UUID, user_id: UUID) -> AdmissionSnapshot | None:
        with self._lock():
            payload, _ = self._read_unlocked()
            if payload["schema_version"] == LEGACY_STORE_SCHEMA_VERSION:
                return None
            tables, _, bindings = self._validated_store(payload)
            binding = bindings.get(str(binding_id))
            if binding is None or binding.user_id != user_id:
                return None
            matching = [
                SessionParticipant.model_validate(value)
                for value in payload["session_participants"].values()
                if value.get("binding_id") == str(binding_id)
            ]
            if len(matching) > 1:
                raise TableBindingRepositoryError("binding has multiple participants")
            if matching:
                participant = matching[0]
                table = tables.get(str(binding.table_registration_id))
                session_raw = payload["sessions"].get(str(participant.session_id))
                service_raw = payload["consent_decisions"].get(str(participant.service_consent_id))
                training_raw = payload["consent_decisions"].get(
                    str(participant.training_consent_id)
                )
                if (
                    table is None
                    or session_raw is None
                    or service_raw is None
                    or training_raw is None
                ):
                    raise TableBindingRepositoryError("admission authority is unavailable")
                return AdmissionSnapshot(
                    binding,
                    table,
                    ConsentDecisionRecord.model_validate(service_raw),
                    ConsentDecisionRecord.model_validate(training_raw),
                    CloudSession.model_validate(session_raw),
                    participant,
                )
            records = [
                SessionAdmissionIdempotencyRecord.model_validate(value)
                for value in payload["idempotency_records"].values()
                if value.get("operation") == "ADMIT"
                and value.get("binding_id") == str(binding_id)
                and value.get("user_id") == str(user_id)
            ]
            if not records:
                return None
            latest = max(records, key=lambda item: (item.created_at, item.record_id))
            return self._snapshot_from_record(payload, latest)

    @staticmethod
    def _require_current_binding_authority(
        binding: TableBinding,
        table: TableRegistration,
        qrs: dict[str, TableQrRegistration],
    ) -> None:
        qr = qrs.get(str(binding.qr_registration_id))
        if (
            qr is None
            or qr.table_registration_id != binding.table_registration_id
            or qr.qr_version != binding.qr_version
            or qr.status is not TableQrStatus.CURRENT
            or binding.qr_version != table.current_qr_version
        ):
            raise SessionAdmissionRepositoryDomainError(
                SessionAdmissionErrorCode.TABLE_BINDING_MISMATCH
            )
        if table.status is not TableRegistrationStatus.ACTIVE:
            raise SessionAdmissionRepositoryDomainError(SessionAdmissionErrorCode.TABLE_UNAVAILABLE)

    def admit(
        self,
        *,
        request: ConsentSubmissionRequest,
        user_id: UUID,
        policy: AdmissionPolicyEvidence,
        key_digest: str,
        payload_digest: str,
        candidate_factory: Callable[[bool, bool], AdmissionCandidate],
    ) -> AdmissionSnapshot:
        record_key = self._idempotency_record_key("ADMIT", user_id, request.binding_id, key_digest)
        with self._lock():
            payload, previous = self._read_unlocked()
            if payload["schema_version"] == STORE_SCHEMA_VERSION:
                existing_raw = payload["idempotency_records"].get(record_key)
                if existing_raw is not None:
                    existing = SessionAdmissionIdempotencyRecord.model_validate(existing_raw)
                    if existing.payload_digest != payload_digest:
                        raise SessionAdmissionRepositoryDomainError(
                            SessionAdmissionErrorCode.IDEMPOTENCY_CONFLICT
                        )
                    return self._snapshot_from_record(payload, existing)
            tables, qrs, bindings = self._validated_store(payload)
            binding = bindings.get(str(request.binding_id))
            if binding is None or binding.user_id != user_id:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.RESOURCE_NOT_FOUND
                )
            table = tables.get(str(binding.table_registration_id))
            if table is None:
                raise TableBindingRepositoryError("binding table is unavailable")
            self._require_current_binding_authority(binding, table, qrs)
            if binding.status is not TableBindingStatus.PENDING_CONSENT:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.TABLE_BINDING_MISMATCH
                )
            try:
                view = _revalidate(ConsentPolicyView, policy.view)
            except Exception:
                raise TableBindingRepositoryError("admission policy evidence is invalid") from None
            if view.binding_id != binding.binding_id:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.CONSENT_SCOPE_MISMATCH
                )
            policy.verify_submission(request)
            active_count = len(
                {
                    item.user_id
                    for item in bindings.values()
                    if item.table_registration_id == binding.table_registration_id
                    and item.status is TableBindingStatus.ACTIVE
                }
            )
            service_granted = request.service_decision is ConsentDecision.GRANTED
            if service_granted and active_count >= 2:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.TABLE_CAPACITY_REACHED
                )
            sessions = (
                {}
                if payload["schema_version"] == LEGACY_STORE_SCHEMA_VERSION
                else {
                    key: CloudSession.model_validate(value)
                    for key, value in payload["sessions"].items()
                }
            )
            table_sessions = [
                item
                for item in sessions.values()
                if item.table_registration_id == table.table_registration_id
                and item.status is CloudSessionStatus.OPEN
            ]
            if table_sessions and table_sessions[0].roster_frozen_at is not None:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.SESSION_ROSTER_FROZEN
                )
            if len(table_sessions) > 1 or (active_count == 1 and not table_sessions):
                raise TableBindingRepositoryError("open session authority is inconsistent")
            need_session = service_granted and not table_sessions
            try:
                candidate = candidate_factory(service_granted, need_session)
            except Exception:
                raise TableBindingRepositoryError(
                    "session admission candidate generation failed"
                ) from None
            try:
                if candidate.created_at < binding.created_at:
                    raise ValueError("admission candidate predates binding")
                candidate_ids = {
                    candidate.service_consent_id,
                    candidate.training_consent_id,
                }
                if service_granted:
                    if candidate.participant_id is None:
                        raise ValueError("admission candidate lacks participant")
                    candidate_ids.add(candidate.participant_id)
                    if need_session:
                        if candidate.session_id is None:
                            raise ValueError("admission candidate lacks session")
                        candidate_ids.add(candidate.session_id)
                expected_candidates = (
                    4 if service_granted and need_session else 3 if service_granted else 2
                )
                if len(candidate_ids) != expected_candidates:
                    raise ValueError("admission candidate IDs are not unique")
                if any(not self._uuid_is_available(item, payload) for item in candidate_ids):
                    raise ValueError("admission candidate ID already exists")
                session = table_sessions[0] if service_granted and table_sessions else None
                if need_session:
                    session = CloudSession(
                        session_id=candidate.session_id,
                        table_registration_id=table.table_registration_id,
                        created_at=candidate.created_at,
                    )
                participant_id = candidate.participant_id if service_granted else None
                session_id = session.session_id if session is not None else None
                service = ConsentDecisionRecord(
                    consent_id=candidate.service_consent_id,
                    user_id=user_id,
                    binding_id=binding.binding_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
                    decision=request.service_decision,
                    notice_version=view.service_notice.version,
                    notice_sha256=view.service_notice.content_sha256,
                    gate_revision_id=view.gate_revision_id,
                    gate_scope_fingerprint=view.scope_fingerprint,
                    decided_at=candidate.created_at,
                )
                training = ConsentDecisionRecord(
                    consent_id=candidate.training_consent_id,
                    user_id=user_id,
                    binding_id=binding.binding_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    purpose=MaterialUsePurpose.TRAINING,
                    decision=request.training_decision,
                    notice_version=view.training_notice.version,
                    notice_sha256=view.training_notice.content_sha256,
                    gate_revision_id=view.gate_revision_id,
                    gate_scope_fingerprint=view.scope_fingerprint,
                    decided_at=candidate.created_at,
                )
                participant = None
                updated_binding = binding
                if service_granted:
                    participant = SessionParticipant(
                        participant_id=participant_id,
                        session_id=session_id,
                        user_id=user_id,
                        binding_id=binding.binding_id,
                        service_consent_id=service.consent_id,
                        training_consent_id=training.consent_id,
                        joined_at=candidate.created_at,
                    )
                    updated_binding = TableBinding.model_validate(
                        {
                            **binding.model_dump(mode="python"),
                            "status": TableBindingStatus.ACTIVE,
                            "activated_at": candidate.created_at,
                        }
                    )
                self._upgrade_payload(payload)
                payload["consent_decisions"][str(service.consent_id)] = service.model_dump(
                    mode="json"
                )
                payload["consent_decisions"][str(training.consent_id)] = training.model_dump(
                    mode="json"
                )
                payload["bindings"][str(binding.binding_id)] = updated_binding.model_dump(
                    mode="json"
                )
                if need_session and session is not None:
                    payload["sessions"][str(session.session_id)] = session.model_dump(mode="json")
                if participant is not None:
                    payload["session_participants"][str(participant.participant_id)] = (
                        participant.model_dump(mode="json")
                    )
                idem = SessionAdmissionIdempotencyRecord(
                    record_id=record_key,
                    operation="ADMIT",
                    user_id=user_id,
                    binding_id=binding.binding_id,
                    key_digest=key_digest,
                    payload_digest=payload_digest,
                    result_consent_ids=(service.consent_id, training.consent_id),
                    result_session_id=session_id,
                    result_participant_id=participant_id,
                    created_at=candidate.created_at,
                )
                payload["idempotency_records"][record_key] = idem.model_dump(mode="json")
                payload["revision"] += 1
                self._write_unlocked(payload, previous=previous)
            except SessionAdmissionRepositoryDomainError:
                raise
            except (AttributeError, TypeError, ValidationError, ValueError) as exc:
                raise TableBindingRepositoryError("session admission candidate is invalid") from exc
            return AdmissionSnapshot(
                updated_binding, table, service, training, session, participant
            )

    def withdraw(
        self,
        *,
        request: ConsentWithdrawalRequest,
        user_id: UUID,
        key_digest: str,
        payload_digest: str,
        candidate_factory: Callable[[], WithdrawalCandidate],
    ) -> AdmissionSnapshot:
        record_key = self._idempotency_record_key(
            "WITHDRAW", user_id, request.participant_id, request.purpose, key_digest
        )
        with self._lock():
            payload, previous = self._read_unlocked()
            if payload["schema_version"] == LEGACY_STORE_SCHEMA_VERSION:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.RESOURCE_NOT_FOUND
                )
            existing_raw = payload["idempotency_records"].get(record_key)
            if existing_raw is not None:
                existing = SessionAdmissionIdempotencyRecord.model_validate(existing_raw)
                if existing.payload_digest != payload_digest:
                    raise SessionAdmissionRepositoryDomainError(
                        SessionAdmissionErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return self._snapshot_from_record(payload, existing)
            tables, _, bindings = self._validated_store(payload)
            participants = {
                key: SessionParticipant.model_validate(value)
                for key, value in payload["session_participants"].items()
            }
            consents = {
                key: ConsentDecisionRecord.model_validate(value)
                for key, value in payload["consent_decisions"].items()
            }
            participant = participants.get(str(request.participant_id))
            if participant is None or participant.user_id != user_id:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.RESOURCE_NOT_FOUND
                )
            binding = bindings.get(str(participant.binding_id))
            if binding is None:
                raise TableBindingRepositoryError("participant binding is unavailable")
            table = tables.get(str(binding.table_registration_id))
            if table is None:
                raise TableBindingRepositoryError("participant table is unavailable")
            head_id = (
                participant.service_consent_id
                if request.purpose is MaterialUsePurpose.SERVICE_ANALYSIS
                else participant.training_consent_id
            )
            head = consents.get(str(head_id))
            if head is None or head.user_id != user_id or head.purpose is not request.purpose:
                raise TableBindingRepositoryError("current consent head is unavailable")
            if head.decision is ConsentDecision.WITHDRAWN:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.CONSENT_ALREADY_WITHDRAWN
                )
            if head.decision is not ConsentDecision.GRANTED:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.CONSENT_REQUEST_INVALID
                )
            try:
                candidate = candidate_factory()
                if not self._uuid_is_available(candidate.consent_id, payload):
                    raise ValueError("withdrawal candidate ID already exists")
                withdrawal = ConsentDecisionRecord(
                    consent_id=candidate.consent_id,
                    user_id=user_id,
                    binding_id=binding.binding_id,
                    session_id=participant.session_id,
                    participant_id=participant.participant_id,
                    purpose=request.purpose,
                    decision=ConsentDecision.WITHDRAWN,
                    notice_version=head.notice_version,
                    notice_sha256=head.notice_sha256,
                    gate_revision_id=head.gate_revision_id,
                    gate_scope_fingerprint=head.gate_scope_fingerprint,
                    decided_at=candidate.decided_at,
                    withdrawn_at=candidate.decided_at,
                    supersedes=head.consent_id,
                )
                update_field = (
                    "service_consent_id"
                    if request.purpose is MaterialUsePurpose.SERVICE_ANALYSIS
                    else "training_consent_id"
                )
                participant = SessionParticipant.model_validate(
                    {
                        **participant.model_dump(mode="python"),
                        update_field: withdrawal.consent_id,
                    }
                )
                payload["consent_decisions"][str(withdrawal.consent_id)] = withdrawal.model_dump(
                    mode="json"
                )
                payload["session_participants"][str(participant.participant_id)] = (
                    participant.model_dump(mode="json")
                )
                idem = SessionAdmissionIdempotencyRecord(
                    record_id=record_key,
                    operation="WITHDRAW",
                    user_id=user_id,
                    binding_id=binding.binding_id,
                    participant_id=participant.participant_id,
                    purpose=request.purpose,
                    key_digest=key_digest,
                    payload_digest=payload_digest,
                    result_consent_ids=(withdrawal.consent_id,),
                    result_session_id=participant.session_id,
                    result_participant_id=participant.participant_id,
                    created_at=candidate.decided_at,
                )
                payload["idempotency_records"][record_key] = idem.model_dump(mode="json")
                payload["revision"] += 1
                self._write_unlocked(payload, previous=previous)
            except (AttributeError, TypeError, ValidationError, ValueError) as exc:
                raise TableBindingRepositoryError(
                    "consent withdrawal candidate is invalid"
                ) from exc
            session = CloudSession.model_validate(payload["sessions"][str(participant.session_id)])
            service = (
                consents[str(participant.service_consent_id)]
                if request.purpose is MaterialUsePurpose.TRAINING
                else withdrawal
            )
            training = (
                consents[str(participant.training_consent_id)]
                if request.purpose is MaterialUsePurpose.SERVICE_ANALYSIS
                else withdrawal
            )
            return AdmissionSnapshot(binding, table, service, training, session, participant)

    @staticmethod
    def _capture_record_key(operation: str, staff_id: UUID, session_id: UUID, key: str) -> str:
        raw = json.dumps(
            [operation, str(staff_id), str(session_id), hashlib.sha256(key.encode()).hexdigest()],
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _capture_digest(request: Any) -> str:
        body = request.model_dump(mode="json", exclude={"idempotency_key"})
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _capture_has_upload_authority(payload: dict[str, Any], session_id: UUID) -> bool:
        return str(session_id) in payload["media_bindings"] or any(
            item["session_id"] == str(session_id)
            and (item["status"] in {"STAGED", "COMPLETED"} or item.get("cleanup_required") is True)
            for item in payload["upload_attempts"].values()
        )

    def _capture_mutation(
        self,
        *,
        operation: str,
        request: Any,
        principal: Any,
        now: datetime,
        policy_provider: Any,
    ):
        record_key = self._capture_record_key(
            operation, principal.staff_id, request.session_id, request.idempotency_key
        )
        digest = self._capture_digest(request)
        with self._lock():
            payload, previous = self._read_unlocked()
            self._upgrade_payload(payload)
            existing = payload["capture_idempotency_records"].get(record_key)
            if existing is not None:
                if existing["payload_digest"] != digest:
                    raise SessionAdmissionRepositoryDomainError(
                        SessionAdmissionErrorCode.IDEMPOTENCY_CONFLICT
                    )
                replay_session = CloudSession.model_validate(
                    payload["sessions"][existing["session_id"]]
                )
                replay_table = TableRegistration.model_validate(
                    payload["table_registrations"][str(replay_session.table_registration_id)]
                )
                if replay_table.venue_id not in principal.venue_ids:
                    raise TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
                if (
                    replay_session.revision != existing["result_session_revision"]
                    or replay_session.status.value != existing["result_status"]
                ):
                    return replay_session.model_copy(
                        update={
                            "revision": existing["result_session_revision"],
                            "status": existing["result_status"],
                            "interval_history": replay_session.interval_history[
                                : existing["result_interval_revision"]
                            ],
                        }
                    )
                return replay_session
            tables, _, bindings = self._validated_store(payload)
            raw = payload["sessions"].get(str(request.session_id))
            if raw is None:
                raise TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
            session = CloudSession.model_validate(raw)
            table = tables[str(session.table_registration_id)]
            if table.venue_id not in principal.venue_ids:
                raise TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
            participants = [
                SessionParticipant.model_validate(item)
                for item in payload["session_participants"].values()
                if item["session_id"] == str(session.session_id)
            ]
            if (
                not 1 <= len(participants) <= 2
                or table.status is not TableRegistrationStatus.ACTIVE
            ):
                raise TableBindingRepositoryError("capture session authority is unavailable")
            consents = {
                key: ConsentDecisionRecord.model_validate(item)
                for key, item in payload["consent_decisions"].items()
            }
            if any(
                consents[str(item.service_consent_id)].decision is not ConsentDecision.GRANTED
                for item in participants
            ):
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.CONSENT_ALREADY_WITHDRAWN
                )
            try:
                policy_provider.verify_capture(table)
            except Exception:
                raise SessionAdmissionRepositoryDomainError(
                    SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
                ) from None
            history = session.interval_history
            if operation == "START":
                if session.status is not CloudSessionStatus.OPEN or history:
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.SESSION_ALREADY_STARTED
                        if history
                        else SessionCaptureErrorCode.SESSION_NOT_ACTIVE
                    )
                if now < session.created_at or any(
                    now < consents[str(item.service_consent_id)].decided_at for item in participants
                ):
                    raise TableBindingRepositoryError("capture start predates authority")
                interval = SessionIntervalRecord(
                    revision=1,
                    start_at=now,
                    provenance="SERVER_MARKED",
                    operator_id=principal.staff_id,
                    recorded_at=now,
                )
                session = CloudSession.model_validate(
                    {
                        **session.model_dump(mode="python"),
                        "revision": session.revision + 1,
                        "roster_frozen_at": now,
                        "interval_history": (interval,),
                    }
                )
            elif operation == "END":
                if (
                    session.status is not CloudSessionStatus.OPEN
                    or not history
                    or history[-1].end_at
                ):
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.SESSION_NOT_STARTED
                        if not history
                        else SessionCaptureErrorCode.SESSION_ALREADY_ENDED
                    )
                if now <= history[-1].start_at:
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.INVALID_TIME_RANGE
                    )
                ended = SessionIntervalRecord.model_validate(
                    {**history[-1].model_dump(mode="python"), "end_at": now}
                )
                session = CloudSession.model_validate(
                    {
                        **session.model_dump(mode="python"),
                        "status": CloudSessionStatus.ENDED_AWAITING_UPLOAD,
                        "revision": session.revision + 1,
                        "interval_history": (*history[:-1], ended),
                    }
                )
                for participant in participants:
                    binding = bindings[str(participant.binding_id)]
                    payload["bindings"][str(binding.binding_id)] = TableBinding.model_validate(
                        {
                            **binding.model_dump(mode="python"),
                            "status": TableBindingStatus.COMPLETED,
                            "released_at": now,
                        }
                    ).model_dump(mode="json")
            elif operation == "CORRECT":
                if (
                    session.status is not CloudSessionStatus.ENDED_AWAITING_UPLOAD
                    or request.expected_interval_revision != len(history)
                ):
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.REVISION_CONFLICT
                    )
                if str(session.session_id) in payload["media_bindings"]:
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.UPLOAD_NOT_READY
                    )
                staged = [
                    item
                    for item in payload["upload_attempts"].values()
                    if item["session_id"] == str(session.session_id) and item["status"] == "STAGED"
                ]
                if staged:
                    for item in staged:
                        item["status"] = "FAILED"
                        item["cleanup_required"] = item.get("object_staged") is True
                    payload["revision"] += 1
                    self._write_unlocked(payload, previous=previous)
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.UPLOAD_NOT_READY
                    )
                if self._capture_has_upload_authority(payload, session.session_id):
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.UPLOAD_NOT_READY
                    )
                if request.start_at < session.created_at or any(
                    request.start_at < consents[str(item.service_consent_id)].decided_at
                    for item in participants
                ):
                    raise TableBindingRepositoryError("corrected interval predates authority")
                for other_raw in payload["sessions"].values():
                    other = CloudSession.model_validate(other_raw)
                    if (
                        other.session_id == session.session_id
                        or other.table_registration_id != session.table_registration_id
                        or other.status is CloudSessionStatus.CANCELLED
                        or not other.interval_history
                        or other.interval_history[-1].end_at is None
                    ):
                        continue
                    frozen = other.interval_history[-1]
                    if request.start_at < frozen.end_at and frozen.start_at < request.end_at:
                        raise SessionCaptureRepositoryDomainError(
                            SessionCaptureErrorCode.INTERVAL_OVERLAP
                        )
                interval = SessionIntervalRecord(
                    revision=len(history) + 1,
                    start_at=request.start_at,
                    end_at=request.end_at,
                    provenance="STAFF_ATTESTED",
                    operator_id=principal.staff_id,
                    recorded_at=now,
                    reason=request.reason,
                )
                session = CloudSession.model_validate(
                    {
                        **session.model_dump(mode="python"),
                        "revision": session.revision + 1,
                        "interval_history": (*history, interval),
                    }
                )
            elif operation == "CANCEL":
                if session.status not in {
                    CloudSessionStatus.OPEN,
                    CloudSessionStatus.ENDED_AWAITING_UPLOAD,
                }:
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.SESSION_NOT_ACTIVE
                    )
                media_binding = payload["media_bindings"].pop(str(session.session_id), None)
                if media_binding is not None:
                    attempt = payload["upload_attempts"][media_binding["attempt_id"]]
                    if attempt["status"] == "COMPLETED":
                        raise SessionCaptureRepositoryDomainError(
                            SessionCaptureErrorCode.SESSION_MEDIA_ALREADY_BOUND
                        )
                    attempt["status"] = "FAILED"
                    attempt["cleanup_required"] = True
                session = CloudSession.model_validate(
                    {
                        **session.model_dump(mode="python"),
                        "status": CloudSessionStatus.CANCELLED,
                        "revision": session.revision + 1,
                        "media_asset_id": None,
                        "media_content_sha256": None,
                    }
                )
                for participant in participants:
                    binding = bindings[str(participant.binding_id)]
                    if binding.status is TableBindingStatus.ACTIVE:
                        payload["bindings"][str(binding.binding_id)] = TableBinding.model_validate(
                            {
                                **binding.model_dump(mode="python"),
                                "status": TableBindingStatus.CANCELLED,
                                "released_at": now,
                            }
                        ).model_dump(mode="json")
            else:
                raise TableBindingRepositoryError("unknown capture operation")
            payload["sessions"][str(session.session_id)] = session.model_dump(mode="json")
            key_digest = hashlib.sha256(request.idempotency_key.encode()).hexdigest()
            payload["capture_idempotency_records"][record_key] = CaptureIdempotencyRecord(
                record_id=record_key,
                operation=operation,
                staff_id=principal.staff_id,
                session_id=session.session_id,
                key_digest=key_digest,
                payload_digest=digest,
                result_session_revision=session.revision,
                result_interval_revision=len(session.interval_history) or None,
                result_status=session.status.value,
                created_at=now,
            ).model_dump(mode="json")
            sequence = len(payload["capture_audit_events"]) + 1
            payload["capture_audit_events"][str(sequence)] = CaptureAuditEvent(
                event_id=uuid4(),
                sequence=sequence,
                operator_id=principal.staff_id,
                session_id=session.session_id,
                table_registration_id=session.table_registration_id,
                event_type=(
                    "INTERVAL_CORRECTED"
                    if operation == "CORRECT"
                    else "CANCELLED"
                    if operation == "CANCEL"
                    else f"{operation}_MARKED"
                ),
                occurred_at=now,
                result="SUCCESS",
                idempotency_digest=key_digest,
                reason=getattr(request, "reason", None),
            ).model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
            return session

    def start_capture(self, *, request, principal, now, policy_provider):
        return self._capture_mutation(
            operation="START",
            request=request,
            principal=principal,
            now=now,
            policy_provider=policy_provider,
        )

    def end_capture(self, *, request, principal, now, policy_provider):
        return self._capture_mutation(
            operation="END",
            request=request,
            principal=principal,
            now=now,
            policy_provider=policy_provider,
        )

    def correct_interval(self, *, request: CorrectIntervalRequest, principal, now, policy_provider):
        return self._capture_mutation(
            operation="CORRECT",
            request=request,
            principal=principal,
            now=now,
            policy_provider=policy_provider,
        )

    def cancel_capture(self, *, request, principal, now, policy_provider):
        return self._capture_mutation(
            operation="CANCEL",
            request=request,
            principal=principal,
            now=now,
            policy_provider=policy_provider,
        )

    def list_staff_sessions(self, *, principal: Any, limit: int = 100):
        if isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("staff session list limit is invalid")
        with self._lock():
            payload, _ = self._read_unlocked()
            tables, _, _ = self._validated_store(payload)
            participants = [
                SessionParticipant.model_validate(item)
                for item in payload.get("session_participants", {}).values()
            ]
            result = []
            for raw in payload.get("sessions", {}).values():
                session = CloudSession.model_validate(raw)
                table = tables[str(session.table_registration_id)]
                if table.venue_id not in principal.venue_ids or session.status not in {
                    CloudSessionStatus.OPEN,
                    CloudSessionStatus.ENDED_AWAITING_UPLOAD,
                }:
                    continue
                count = sum(item.session_id == session.session_id for item in participants)
                result.append((session, table, count))
            return tuple(
                sorted(result, key=lambda item: (item[0].created_at, str(item[0].session_id)))
            )[:limit]

    def read_staff_session(self, *, session_id: UUID, principal: Any):
        return self._read_staff_capture_direct(session_id=session_id, principal=principal)

    def _read_staff_capture_direct(self, *, session_id: UUID, principal: Any):
        with self._lock():
            payload, _ = self._read_unlocked()
            tables, _, _ = self._validated_store(payload)
            raw = payload.get("sessions", {}).get(str(session_id))
            if raw is None:
                return None
            session = CloudSession.model_validate(raw)
            table = tables[str(session.table_registration_id)]
            if table.venue_id not in principal.venue_ids or session.status not in {
                CloudSessionStatus.OPEN,
                CloudSessionStatus.ENDED_AWAITING_UPLOAD,
            }:
                return None
            count = sum(
                item.get("session_id") == str(session_id)
                for item in payload.get("session_participants", {}).values()
            )
            return session, table, count

    def prepare_upload_attempt(
        self, *, request: Any, principal: Any, policy_provider: Any, now: datetime
    ):
        digest = hashlib.sha256(request.idempotency_key.encode()).hexdigest()
        with self._lock():
            payload, previous = self._read_unlocked()
            self._upgrade_payload(payload)
            tables, _, _ = self._validated_store(payload)
            raw = payload["sessions"].get(str(request.session_id))
            if raw is None:
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.RESOURCE_NOT_FOUND
                )
            session = CloudSession.model_validate(raw)
            table = tables[str(session.table_registration_id)]
            if table.venue_id not in principal.venue_ids:
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.RESOURCE_NOT_FOUND
                )
            try:
                policy_provider.verify_capture(table)
            except Exception:
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
                ) from None
            participants = [
                SessionParticipant.model_validate(item)
                for item in payload["session_participants"].values()
                if item["session_id"] == str(session.session_id)
            ]
            consents = {
                key: ConsentDecisionRecord.model_validate(item)
                for key, item in payload["consent_decisions"].items()
            }
            for item in payload["upload_attempts"].values():
                if (
                    item["session_id"] == str(session.session_id)
                    and item["idempotency_digest"] == digest
                    and item["staff_id"] == str(principal.staff_id)
                ):
                    return (
                        UploadAttemptRecord.model_validate(item),
                        session,
                        table,
                        tuple(participant.user_id for participant in participants),
                    )
            binding = payload["media_bindings"].get(str(session.session_id))
            if binding is not None:
                return (
                    None,
                    session,
                    table,
                    tuple(participant.user_id for participant in participants),
                )
            if session.status is not CloudSessionStatus.ENDED_AWAITING_UPLOAD:
                raise SessionCaptureRepositoryDomainError(SessionCaptureErrorCode.UPLOAD_NOT_READY)
            if any(
                consents[str(item.service_consent_id)].decision is not ConsentDecision.GRANTED
                for item in participants
            ):
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.SERVICE_CONSENT_NOT_CURRENT
                )
            prior = [
                UploadAttemptRecord.model_validate(item)
                for item in payload["upload_attempts"].values()
                if item["session_id"] == str(session.session_id) and item["status"] == "FAILED"
            ]
            attempt = UploadAttemptRecord(
                attempt_id=uuid4(),
                session_id=session.session_id,
                staff_id=principal.staff_id,
                idempotency_digest=digest,
                status="STAGED",
                expected_session_revision=session.revision,
                interval_revision=len(session.interval_history),
                created_at=now,
                supersedes=prior[-1].attempt_id if prior else None,
            )
            payload["upload_attempts"][str(attempt.attempt_id)] = attempt.model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
            return (
                attempt,
                session,
                table,
                tuple(participant.user_id for participant in participants),
            )

    def complete_upload_attempt(
        self,
        *,
        attempt_id: UUID,
        content_sha256: str,
        asset_id: UUID,
        size_bytes: int,
        storage_locator: str,
        gate_revision_id: UUID,
        subject_user_ids: tuple[str, ...],
    ) -> UploadAttemptRecord:
        with self._lock():
            payload, previous = self._read_unlocked()
            raw = payload["upload_attempts"].get(str(attempt_id))
            if raw is None:
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.RESOURCE_NOT_FOUND
                )
            attempt = UploadAttemptRecord.model_validate(raw)
            if attempt.object_staged:
                if attempt.content_sha256 == content_sha256:
                    return attempt
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.IDEMPOTENCY_CONFLICT
                )
            attempt = UploadAttemptRecord.model_validate(
                {
                    **attempt.model_dump(mode="python"),
                    "status": "STAGED",
                    "object_staged": True,
                    "content_sha256": content_sha256,
                    "asset_id": asset_id,
                    "size_bytes": size_bytes,
                    "storage_locator": storage_locator,
                    "gate_revision_id": gate_revision_id,
                    "subject_user_ids": tuple(sorted(str(item) for item in subject_user_ids)),
                }
            )
            payload["upload_attempts"][str(attempt_id)] = attempt.model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
            return attempt

    def get_upload_attempt(self, *, attempt_id: UUID) -> UploadAttemptRecord | None:
        with self._lock():
            payload, _ = self._read_unlocked()
            raw = payload.get("upload_attempts", {}).get(str(attempt_id))
            return None if raw is None else UploadAttemptRecord.model_validate(raw)

    def pending_upload_attempts(self, *, limit: int = 100):
        with self._lock():
            payload, _ = self._read_unlocked()
            return tuple(
                UploadAttemptRecord.model_validate(item)
                for item in payload.get("upload_attempts", {}).values()
                if item["status"] == "STAGED"
            )[:limit]

    def prepare_upload_authority(self, *, session_id: UUID, principal: Any, policy_provider: Any):
        with self._lock():
            payload, _ = self._read_unlocked()
            tables, _, _ = self._validated_store(payload)
            raw = payload.get("sessions", {}).get(str(session_id))
            if raw is None:
                raise TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
            session = CloudSession.model_validate(raw)
            table = tables[str(session.table_registration_id)]
            if table.venue_id not in principal.venue_ids:
                raise TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
            try:
                policy_provider.verify_capture(table)
            except Exception:
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
                ) from None
            participants = [
                SessionParticipant.model_validate(item)
                for item in payload["session_participants"].values()
                if item["session_id"] == str(session_id)
            ]
            consents = {
                key: ConsentDecisionRecord.model_validate(item)
                for key, item in payload["consent_decisions"].items()
            }
            if session.status is not CloudSessionStatus.ENDED_AWAITING_UPLOAD:
                raise SessionCaptureRepositoryDomainError(SessionCaptureErrorCode.UPLOAD_NOT_READY)
            if any(
                consents[str(item.service_consent_id)].decision is not ConsentDecision.GRANTED
                for item in participants
            ):
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.SERVICE_CONSENT_NOT_CURRENT
                )
            return session, table, tuple(item.user_id for item in participants)

    def bind_staged_upload(
        self,
        *,
        session_id: UUID,
        attempt_id: UUID,
        asset_id: UUID,
        content_sha256: str,
        size_bytes: int,
        storage_locator: str,
        principal: Any,
        policy_provider: Any,
        now: datetime,
    ):
        with self._lock():
            payload, previous = self._read_unlocked()
            self._upgrade_payload(payload)
            tables, _, _ = self._validated_store(payload)
            session = CloudSession.model_validate(payload["sessions"][str(session_id)])
            table = tables[str(session.table_registration_id)]
            if table.venue_id not in principal.venue_ids:
                raise TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
            try:
                policy_provider.verify_capture(table)
            except Exception:
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
                ) from None
            participants = [
                SessionParticipant.model_validate(item)
                for item in payload["session_participants"].values()
                if item["session_id"] == str(session_id)
            ]
            consents = {
                key: ConsentDecisionRecord.model_validate(item)
                for key, item in payload["consent_decisions"].items()
            }
            if session.status is not CloudSessionStatus.ENDED_AWAITING_UPLOAD:
                raise SessionCaptureRepositoryDomainError(SessionCaptureErrorCode.UPLOAD_NOT_READY)
            if any(
                consents[str(item.service_consent_id)].decision is not ConsentDecision.GRANTED
                for item in participants
            ):
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.SERVICE_CONSENT_NOT_CURRENT
                )
            attempt = UploadAttemptRecord.model_validate(
                payload["upload_attempts"].get(str(attempt_id))
            )
            if (
                attempt.status is not UploadAttemptStatus.STAGED
                or attempt.session_id != session_id
                or attempt.content_sha256 != content_sha256
                or attempt.asset_id != asset_id
                or attempt.storage_locator != storage_locator
                or attempt.expected_session_revision != session.revision
                or attempt.interval_revision != len(session.interval_history)
            ):
                raise SessionCaptureRepositoryDomainError(SessionCaptureErrorCode.REVISION_CONFLICT)
            subjects = tuple(item.user_id for item in participants)
            existing = payload["media_bindings"].get(str(session_id))
            if existing is not None:
                if existing["content_sha256"] == content_sha256:
                    return existing
                raise SessionCaptureRepositoryDomainError(
                    SessionCaptureErrorCode.SESSION_MEDIA_ALREADY_BOUND
                )
            for binding in payload["media_bindings"].values():
                if binding["content_sha256"] == content_sha256:
                    raise SessionCaptureRepositoryDomainError(
                        SessionCaptureErrorCode.DUPLICATE_UPLOAD
                    )
            binding_model = SessionMediaBindingRecord(
                session_id=session_id,
                attempt_id=attempt_id,
                asset_id=asset_id,
                content_sha256=content_sha256,
                storage_locator=storage_locator,
                subject_user_ids=tuple(sorted(str(item) for item in subjects)),
                interval_revision=len(session.interval_history),
                accepted_at=now,
            )
            binding = binding_model.model_dump(mode="json")
            payload["media_bindings"][str(session_id)] = binding
            sequence = len(payload["capture_audit_events"]) + 1
            payload["capture_audit_events"][str(sequence)] = CaptureAuditEvent(
                event_id=uuid4(),
                sequence=sequence,
                operator_id=principal.staff_id,
                session_id=session_id,
                table_registration_id=session.table_registration_id,
                event_type="UPLOAD_STAGED",
                occurred_at=now,
                result="SUCCESS",
                idempotency_digest=attempt.idempotency_digest,
                attempt_id=attempt_id,
                asset_id=asset_id,
            ).model_dump(mode="json")
            payload["sessions"][str(session_id)] = CloudSession.model_validate(
                {
                    **session.model_dump(mode="python"),
                    "revision": session.revision + 1,
                    "media_asset_id": asset_id,
                    "media_content_sha256": content_sha256,
                }
            ).model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
            return binding

    def mark_media_bound(self, *, session_id: UUID, asset_id: UUID, now: datetime):
        with self._lock():
            payload, previous = self._read_unlocked()
            session = CloudSession.model_validate(payload["sessions"][str(session_id)])
            if session.status is CloudSessionStatus.MEDIA_BOUND:
                return session
            if session.media_asset_id != asset_id:
                raise TableBindingRepositoryError("session binding changed")
            session = CloudSession.model_validate(
                {
                    **session.model_dump(mode="python"),
                    "status": CloudSessionStatus.MEDIA_BOUND,
                    "revision": session.revision + 1,
                }
            )
            payload["sessions"][str(session_id)] = session.model_dump(mode="json")
            attempt_id = payload["media_bindings"][str(session_id)]["attempt_id"]
            payload["upload_attempts"][attempt_id]["status"] = "COMPLETED"
            sequence = len(payload["capture_audit_events"]) + 1
            payload["capture_audit_events"][str(sequence)] = CaptureAuditEvent(
                event_id=uuid4(),
                sequence=sequence,
                operator_id="SYSTEM_RECONCILER",
                session_id=session_id,
                table_registration_id=session.table_registration_id,
                event_type="UPLOAD_COMPLETED",
                occurred_at=now,
                result="SUCCESS",
                attempt_id=UUID(attempt_id),
                asset_id=asset_id,
            ).model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
            return session

    def pending_media_bindings(self, *, limit: int = 100):
        with self._lock():
            payload, _ = self._read_unlocked()
            return tuple(
                binding
                for binding in payload.get("media_bindings", {}).values()
                if CloudSession.model_validate(payload["sessions"][binding["session_id"]]).status
                is not CloudSessionStatus.MEDIA_BOUND
            )[:limit]

    def pending_upload_cleanups(self, *, limit: int = 100):
        with self._lock():
            payload, _ = self._read_unlocked()
            return tuple(
                item
                for item in payload.get("upload_attempts", {}).values()
                if item.get("cleanup_required") is True
            )[:limit]

    def mark_upload_cleanup_complete(self, *, attempt_id: UUID) -> None:
        with self._lock():
            payload, previous = self._read_unlocked()
            attempt = payload["upload_attempts"].get(str(attempt_id))
            if attempt is None or attempt.get("cleanup_required") is not True:
                return
            attempt["cleanup_required"] = False
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)

    def get_media_binding(self, *, session_id: UUID):
        with self._lock():
            payload, _ = self._read_unlocked()
            return payload.get("media_bindings", {}).get(str(session_id))

    def is_media_bound(self, *, session_id: UUID, asset_id: UUID) -> bool:
        with self._lock():
            payload, _ = self._read_unlocked()
            raw = payload.get("sessions", {}).get(str(session_id))
            if raw is None:
                return False
            session = CloudSession.model_validate(raw)
            return (
                session.status is CloudSessionStatus.MEDIA_BOUND
                and session.media_asset_id == asset_id
            )

    def record_upload_failed(
        self,
        *,
        session_id: UUID,
        attempt_id: UUID,
        content_sha256: str,
        principal: Any,
        now: datetime,
    ) -> None:
        with self._lock():
            payload, previous = self._read_unlocked()
            self._upgrade_payload(payload)
            key = str(attempt_id)
            existing = payload["upload_attempts"].get(key)
            if existing is not None:
                if existing["status"] == "FAILED":
                    return
                if existing["status"] == "COMPLETED":
                    raise TableBindingRepositoryError("upload attempt is already authoritative")
                existing["status"] = "FAILED"
                existing["cleanup_required"] = existing.get("storage_locator") is not None
            else:
                raise TableBindingRepositoryError("upload attempt was not prepared")
            session = CloudSession.model_validate(payload["sessions"][str(session_id)])
            binding = payload["media_bindings"].get(str(session_id))
            if binding is not None and binding["attempt_id"] == key:
                del payload["media_bindings"][str(session_id)]
                session = CloudSession.model_validate(
                    {
                        **session.model_dump(mode="python"),
                        "revision": session.revision + 1,
                        "media_asset_id": None,
                        "media_content_sha256": None,
                    }
                )
                payload["sessions"][str(session_id)] = session.model_dump(mode="json")
            sequence = len(payload["capture_audit_events"]) + 1
            payload["capture_audit_events"][str(sequence)] = CaptureAuditEvent(
                event_id=uuid4(),
                sequence=sequence,
                operator_id=principal.staff_id,
                session_id=session_id,
                table_registration_id=session.table_registration_id,
                event_type="UPLOAD_FAILED",
                occurred_at=now,
                result="FAILED",
                error_code=SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE,
                attempt_id=attempt_id,
            ).model_dump(mode="json")
            payload["revision"] += 1
            self._write_unlocked(payload, previous=previous)
