"""Shared synthetic fixtures for privacy and gate tests."""

import json
import pwd
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.cloud.privacy import (
    JsonGateRepository,
    RealVideoGateService,
)
from billiards_ai.cloud.privacy import service as gate_service_module
from billiards_ai.cloud.privacy.media_store import ControlledFilesystemMediaStore
from billiards_ai.cloud.privacy.retention_repository import JsonRetentionRepository
from billiards_ai.shared.models import (
    EvidenceReference,
    GateChecklist,
    GateScope,
    MaterialUsePurpose,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated_os_account_home(tmp_path: Path, monkeypatch):
    """Keep account-home trust artifacts inside each test's temporary directory."""
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(tmp_path)),
    )


@pytest.fixture
def complete_checklist_factory():
    def factory(**overrides: bool) -> GateChecklist:
        values = {
            "table_view_only": True,
            "audio_disabled": True,
            "notice_posted": True,
            "operator_contact_available": True,
            "separate_service_consent": True,
            "separate_training_consent": True,
            "purpose_authorization_defined": True,
            "access_path_defined": True,
            "retention_path_defined": True,
            "deletion_path_defined": True,
            "marked_session_clips_only": True,
            "full_stream_stays_local": True,
        }
        values.update(overrides)
        return GateChecklist(**values)

    return factory


@pytest.fixture
def gate_scope_factory(complete_checklist_factory):
    def factory(**overrides) -> GateScope:
        values = {
            "venue_id": "dilan",
            "table_id": "table-3",
            "device_id": "camera-3",
            "camera_config_id": "top-view-v1",
            "checklist": complete_checklist_factory(),
            "operator_name": "Dilan",
            "operator_contact": "privacy@example.test",
            "service_consent_version": "service-v1",
            "service_consent_sha256": "1" * 64,
            "training_consent_version": "training-v1",
            "training_consent_sha256": "2" * 64,
            "access_policy_ref": "policy://access/v1",
            "retention_policy_ref": "policy://retention/v1",
            "deletion_policy_ref": "policy://deletion/v1",
            "approved_purposes": tuple(MaterialUsePurpose),
        }
        values.update(overrides)
        return GateScope(**values)

    return factory


@pytest.fixture
def gate_evidence(now: datetime) -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(
            evidence_ref="controlled://gate/table-3/view.jpg",
            sha256="3" * 64,
            captured_at=now,
        ),
    )


@pytest.fixture
def credential_file_factory(monkeypatch):
    def factory(
        tmp_path: Path,
        *,
        approver_id: str = "Boyu",
        credential_id: str = "phase0-local-approval",
        approval_hmac_key: str = "a" * 64,
        active_key_id: str | None = None,
        keys: dict[str, str] | None = None,
    ) -> Path:
        monkeypatch.setattr(
            pwd,
            "getpwuid",
            lambda _uid: SimpleNamespace(pw_dir=str(tmp_path)),
        )
        credential_directory = tmp_path / ".config" / "billiards-ai"
        credential_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        credential_directory.chmod(0o700)
        path = credential_directory / "approver.json"
        selected_key_id = active_key_id or "key-a"
        payload = {
            "schema_version": "2.0",
            "approver_id": approver_id,
            "credential_id": credential_id,
            "active_key_id": selected_key_id,
            "keys": keys or {selected_key_id: approval_hmac_key},
        }
        path.write_text(json.dumps(payload))
        path.chmod(0o600)
        return path

    return factory


@pytest.fixture
def gate_service_factory(credential_file_factory, monkeypatch, now):
    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: now)

    def factory(tmp_path: Path, *, approver_id: str = "Boyu") -> RealVideoGateService:
        repository = JsonGateRepository(tmp_path / "gates.json")
        credential_file_factory(tmp_path, approver_id=approver_id)
        return RealVideoGateService(repository)

    return factory


@pytest.fixture
def approved_gate_factory(gate_service_factory, gate_scope_factory, gate_evidence, now):
    def factory(tmp_path: Path):
        gate_service = gate_service_factory(tmp_path)
        blocked = gate_service.create_gate(
            scope=gate_scope_factory(),
            inspector_id="inspector-1",
            inspected_at=now,
            evidence=gate_evidence,
        )
        approved = gate_service.approve(
            blocked.gate_id,
            approval_evidence_sha256="4" * 64,
        )
        return gate_service, blocked, approved

    return factory


@pytest.fixture
def repo(tmp_path: Path) -> JsonRetentionRepository:
    return JsonRetentionRepository(tmp_path / "retention" / "store.json")


@pytest.fixture
def controlled_media_store_factory():
    def factory(tmp_path: Path, store_type=ControlledFilesystemMediaStore):
        root = tmp_path / "controlled" / "media"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        sentinel = root / ControlledFilesystemMediaStore.SENTINEL_NAME
        sentinel.write_text(ControlledFilesystemMediaStore.SENTINEL_CONTENT, encoding="utf-8")
        sentinel.chmod(0o600)
        return store_type(root)

    return factory
